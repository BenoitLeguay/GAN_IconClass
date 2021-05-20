import torch
from torch import nn
from torch.distributions import Categorical
import utils as ut
import variable as var
from tqdm import tqdm
import os
from torch.utils.tensorboard import SummaryWriter
from frechet_inception_distance import InceptionV3, calculate_fid

"""
TODO:
- 1. Normalize the inputs ([-1; 1] & TanH): DONE
- 2: A modified loss function (flip label when training generator): DONE
- 3: Use a spherical Z (sample Z from gaussian): DONE
- 4: BatchNorm: DONE
- 5: Avoid Sparse Gradients: ReLU, MaxPool (using LeakyReLu): DONE
    For Down sampling, use: Average Pooling, Conv2d + stride: X
    For Up sampling, use: PixelShuffle, ConvTranspose2d + stride: X
- 6: Use Soft and Noisy Labels
    Label Smoothing: X
    Occasionally flip label for D: X
- 7: DCGAN / Hybrid Models: DONE
- 8: Use stability tricks from RL
    Experience Replay: X
    Occasionally swap to old G and D for k iterations
- 9: Optimizer
    SGD for D: X
    Adam for G: DONE
- 10: Track failures early
    D loss goes to 0: failure mode
    check norms of gradients: if they are over 100 things are screwing up
    when things are working, D loss has low variance and goes down over time vs having huge variance and spiking
    if loss of generator steadily decreases, then it's fooling D with garbage (says martin)
- 12: If you have labels, use them: DONE
- 13: Add noise to inputs, decay over time: X
- 16: Discrete variables in Conditional GANs: X
    Use an Embedding layer
    Add as additional channels to images
    Keep embedding dimensionality low and up sample to match image channel size
- 17: Use Dropouts in G in both train and test phase: DONE

"""


class AuxGAN:
    def __init__(self, params, normal_weight_init=True):
        self.adv_bce_loss = nn.BCELoss()
        self.aux_nll_loss = nn.NLLLoss()

        self.discriminator = AuxDiscriminator(params["n_classes"], params["disc"]["n_feature"],
                                              params["disc"]["n_channel"],
                                              n_conv_block=params["n_conv_block"]).to(var.device)

        self.generator = self.init_generator(params["generator_type"], params)

        self.disc_optim = torch.optim.Adam(self.discriminator.parameters(), lr=params['disc']['lr'],
                                                 betas=tuple(params['disc']['betas'].values()))
        """
        self.disc_optim = torch.optim.SGD(self.discriminator.parameters(), lr=params['disc']['lr'])
        """
        self.gen_optim = torch.optim.Adam(self.generator.parameters(), lr=params['gen']['lr'],
                                          betas=tuple(params['gen']['betas'].values()))

        self.z_dim = params['z_dim']
        self.n_classes = params["n_classes"]
        self.label_smoothing = params["label_smoothing"]

        self.use_inception = params["use_inception"]
        if self.use_inception:
            self.inception = InceptionV3()

        self.writer = None
        self.h_params_added = False
        self.step = 0
        self.epoch = 0

        self.adv_fake_acc = list()
        self.adv_real_acc = list()
        self.aux_fake_acc = list()
        self.aux_real_acc = list()
        self.fake_aux_entropy = list()
        self.real_aux_entropy = list()

        if normal_weight_init:
            self.discriminator.apply(ut.weights_init)
            self.generator.apply(ut.weights_init)

        params['GAN_type'] = self.__class__.__name__
        params['gen_optim_type'] = self.gen_optim.__class__.__name__
        params['disc_optim_type'] = self.disc_optim.__class__.__name__

        self.params = ut.flatten_dict(params)

    @staticmethod
    def init_generator(generator_type, params):
        if generator_type == 'convtranspose':
            generator = CondGeneratorCTrans(params["n_classes"], params["z_dim"], params["gen"]["n_feature"],
                                            params["gen"]["n_channel"], params["gen"]["embedding"],
                                            n_conv_block=params["n_conv_block"])
        elif generator_type == 'upsample':
            generator = CondGeneratorUpSample(params["n_classes"], params["z_dim"], params["gen"]["n_feature"],
                                              params["gen"]["n_channel"], params["output_size"],
                                              n_conv_block=params["n_conv_block"])
        elif generator_type == 'color_picker':
            generator = CondGeneratorColorPicker(params["n_classes"], params["z_dim"])
        else:
            raise NotImplementedError(f'{generator_type} generator type is not available')

        return generator.to(var.device)

    def init_tensorboard(self, main_dir='runs', subdir='train', port=8008):
        main_dir = os.path.join(var.PROJECT_DIR, main_dir)
        os.system(f'tensorboard --logdir={main_dir} --port={port} &')

        self.writer = SummaryWriter(f'{os.path.join(main_dir, subdir)}')
        if not self.h_params_added:
            # self.writer.add_hparams(self.params, {})  https://github.com/pytorch/pytorch/issues/32651
            self.h_params_added = True

    def ground_truth_label(self, shape_like_tensor, smoothing_value=.9):
        ground_truth = torch.ones_like(shape_like_tensor)
        if self.label_smoothing:
            ground_truth *= smoothing_value
        return ground_truth

    def get_random_noise(self, n):
        return torch.randn(n, self.z_dim, 1, 1, device=var.device)

    def get_random_classes(self, n):
        return torch.randint(self.n_classes, (n, ), device=var.device)

    def generate_fake(self, n, train=True, being_class=None):
        noise = self.get_random_noise(n)
        if being_class is not None:
            classes = torch.ones((n, ), device=var.device).long() * being_class
        else:
            classes = self.get_random_classes(n)

        if train:
            return self.generator.forward(noise, classes), classes
        else:
            self.generator.eval()
            with torch.no_grad():
                fake = self.generator.forward(noise, classes)
            self.generator.train()
            return fake, classes

    def get_discriminator_loss(self, batch_size, real, real_classes):
        fake, fake_classes = self.generate_fake(batch_size)

        fake_adv, fake_aux = self.discriminator(fake.detach())
        fake_loss = (self.adv_bce_loss(fake_adv, torch.zeros_like(fake_adv)) +
                     self.aux_nll_loss(fake_aux, fake_classes)) / 2

        real_adv, real_aux = self.discriminator(real)
        real_loss = (self.adv_bce_loss(real_adv, self.ground_truth_label(real_adv)) +
                     self.aux_nll_loss(real_aux, real_classes)) / 2

        disc_loss = (fake_loss + real_loss) / 2
        disc_loss.backward()

        self.compute_discriminator_metrics(real_adv, real_aux, fake_adv, fake_aux, real_classes, fake_classes)

        return disc_loss

    def get_generator_loss(self, batch_size):
        fake, fake_classes = self.generate_fake(batch_size)
        fake_adv, fake_aux = self.discriminator(fake)

        loss = (self.adv_bce_loss(fake_adv, torch.ones_like(fake_adv)) +
                self.aux_nll_loss(fake_aux, fake_classes)) / 2

        loss.backward()

        return loss

    def train(self, n_epoch, dataloader, gan_id=False, aux_fake_acc_threshold=.7):

        for _ in tqdm(range(n_epoch)):
            gen_epoch_loss = 0.0
            disc_epoch_loss = 0.0

            for i_batch, (real, real_classes) in enumerate(dataloader):

                cur_batch_size = len(real)
                real = real.to(var.device)
                real_classes = real_classes.to(var.device)

                # UPDATE DISCRIMINATOR #
                self.disc_optim.zero_grad()
                disc_loss = self.get_discriminator_loss(cur_batch_size, real, real_classes)
                self.disc_optim.step()

                # UPDATE GENERATOR #
                self.gen_optim.zero_grad()
                gen_loss = self.get_generator_loss(cur_batch_size)
                self.gen_optim.step()

                disc_epoch_loss += disc_loss.item()
                gen_epoch_loss += gen_loss.item()

                if i_batch == 0:
                    fake, classes = self.generate_fake(cur_batch_size, train=False)

                    if self.use_inception:
                        fid = calculate_fid(real.detach(), fake, self.inception, resize=(75, 75))
                        self.writer.add_scalar('Frechet Inception Distance', fid, self.epoch)

                    self.writer.add_image(f'{self.epoch}/Fake', ut.images_grid(fake))
                    self.writer.add_image(f'{self.epoch}/Real', ut.images_grid(real))

                self.step += 1

            self.writer.add_scalar('Loss/Discriminator', disc_epoch_loss / len(dataloader), self.epoch)
            self.writer.add_scalar('Loss/Generator', gen_epoch_loss / len(dataloader), self.epoch)
            aux_fake_acc = self.add_discriminator_metrics()

            self.epoch += 1
            if aux_fake_acc >= aux_fake_acc_threshold:
                break

        if gan_id:
            self.save_model(gan_id)

    def compute_discriminator_metrics(self, real_adv, real_aux, fake_adv, fake_aux, real_classes, fake_classes):
        fake_adv_label = fake_adv < .5
        real_adv_label = real_adv > .5

        fake_aux_label = torch.argmax(fake_aux, dim=1) == fake_classes
        real_aux_label = torch.argmax(real_aux, dim=1) == real_classes

        fake_entropy = (Categorical(probs=fake_aux).entropy()/fake_aux.size(1)).mean().item()
        real_entropy = (Categorical(probs=real_aux).entropy()/real_aux.size(1)).mean().item()

        self.adv_fake_acc.extend(fake_adv_label.squeeze().tolist())
        self.adv_real_acc.extend(real_adv_label.squeeze().tolist())
        self.aux_fake_acc.extend(fake_aux_label.tolist())
        self.aux_real_acc.extend(real_aux_label.tolist())
        self.fake_aux_entropy.append(fake_entropy)
        self.real_aux_entropy.append(real_entropy)

    def add_discriminator_metrics(self):
        aux_fake_acc = sum(self.aux_fake_acc) / len(self.aux_fake_acc)
        self.writer.add_scalar('Discriminator/ADV/Fake', sum(self.adv_fake_acc) / len(self.adv_fake_acc), self.epoch)
        self.writer.add_scalar('Discriminator/ADV/Real', sum(self.adv_real_acc) / len(self.adv_real_acc), self.epoch)
        self.writer.add_scalar('Discriminator/AUX/Fake', sum(self.aux_fake_acc) / len(self.aux_fake_acc), self.epoch)
        self.writer.add_scalar('Discriminator/AUX/Real', sum(self.aux_real_acc) / len(self.aux_real_acc), self.epoch)
        self.writer.add_scalar('Discriminator/AUX/Real Entropy',
                               sum(self.fake_aux_entropy) / len(self.fake_aux_entropy), self.epoch)
        self.writer.add_scalar('Discriminator/AUX/Fake Entropy',
                               sum(self.real_aux_entropy) / len(self.real_aux_entropy), self.epoch)

        self.adv_fake_acc.clear()
        self.adv_real_acc.clear()
        self.aux_fake_acc.clear()
        self.aux_real_acc.clear()
        self.fake_aux_entropy.clear()
        self.real_aux_entropy.clear()

        return aux_fake_acc

    def save_model(self, gan_id):
        model_path = os.path.join(var.PROJECT_DIR, f"data/models/{gan_id}.pth")
        torch.save({
            'step': self.step,
            'epoch': self.epoch,
            'z_dim': self.z_dim,
            'h_params_added': self.h_params_added,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'generator_optim_state_dict': self.gen_optim.state_dict(),
            'discriminator_optim_state_dict': self.disc_optim.state_dict(),
        },
            model_path)

    def load_model(self, path, train=True):
        checkpoint = torch.load(path)

        self.step = checkpoint['step']
        self.epoch = checkpoint['epoch']
        self.z_dim = checkpoint['z_dim']
        self.h_params_added = checkpoint['h_params_added']

        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.gen_optim.load_state_dict(checkpoint['generator_optim_state_dict'])
        self.disc_optim.load_state_dict(checkpoint['discriminator_optim_state_dict'])

        if train:
            self.generator.train()
            self.discriminator.train()
        else:
            self.generator.eval()
            self.discriminator.eval()


class AuxDiscriminator(nn.Module):
    def __init__(self, n_classes, n_features, n_channel, n_conv_block=3):
        super(AuxDiscriminator, self).__init__()
        self.n_features = n_features

        modules = list()
        for layer in range(n_conv_block):
            modules.append(ut.discriminator_layer(n_features * (2 ** layer)))

        self.main = nn.Sequential(
            nn.Conv2d(n_channel, n_features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Sequential(*modules),  # convolution blocks
            nn.Conv2d(n_features * (2 ** n_conv_block), n_features, 4, 1, 0, bias=False)
        )
        self.fc_adv = nn.Linear(self.n_features, 1)
        self.fc_aux = nn.Linear(self.n_features, n_classes)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        conv = self.main(x)
        flat = conv.view(-1, self.n_features)

        aux = self.softmax(self.fc_aux(flat))
        adv = self.sigmoid(self.fc_adv(flat))

        return adv, aux


class CondGeneratorCTrans(nn.Module):
    def __init__(self, n_classes, z_dim, n_features, n_channel, embedding=True, n_conv_block=3):
        super(CondGeneratorCTrans, self).__init__()

        self.n_classes = n_classes
        self.embedding = embedding

        modules = list()
        for layer in reversed(range(n_conv_block)):
            modules.append(ut.generator_layer(n_features * (2 ** layer)))

        input_size = z_dim + n_classes

        # STUDY EMBEDDING MAXIMIZATION OUTPUT (noise vector to 0) IN THE TRAINING PROCESS
        if self.embedding:
            self.label_emb = nn.Sequential(
                nn.Embedding(n_classes, z_dim),
            )
            input_size = z_dim

        self.main = nn.Sequential(
            nn.ConvTranspose2d(input_size, n_features * (2 ** n_conv_block), 4, 1, 0, bias=False),
            nn.BatchNorm2d(n_features * (2 ** n_conv_block)),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Sequential(*modules),  # convolution blocks
            nn.ConvTranspose2d(n_features, n_channel, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, classes):
        if self.embedding:
            em = self.label_emb(classes)
            em = em.view(em.shape + (1, 1))  # check here

            x = torch.mul(em, noise)
        else:
            x = self.cat_noise_classes(noise, classes)

        return self.main(x)

    def cat_noise_classes(self, noise, classes):
        one_hot = torch.eye(self.n_classes, device=var.device)
        one_hot = one_hot[classes]
        one_hot = one_hot.view(one_hot.shape + (1, 1))

        return torch.cat((noise, one_hot), axis=1)


class CondGeneratorColorPicker(nn.Module):
    """A generator for mapping a latent space to a sample space.
    Input shape: (?, latent_dim)
    Output shape: (?, 3, 96, 96)
    """

    def __init__(self, n_classes, z_dim):
        """Initialize generator.
        Args:
            latent_dim (int): latent dimension ("noise vector")
        """
        super().__init__()
        self.z_dim = z_dim
        self.n_classes = n_classes
        self._init_modules()

    @staticmethod
    def build_colourspace(input_dim: int, output_dim: int):
        """Build a small module for selecting colours."""
        colourspace = nn.Sequential(
            nn.Linear(
                input_dim,
                128,
                bias=True),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),

            nn.Linear(
                128,
                64,
                bias=True),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),

            nn.Linear(
                64,
                output_dim,
                bias=True),
            nn.Tanh(),
        )
        return colourspace

    def _init_modules(self):
        """Initialize the modules."""
        self.label_emb = nn.Sequential(
            nn.Embedding(self.n_classes, self.z_dim),
        )
        projection_widths = [self.z_dim] * 7
        self.projection_dim = sum(projection_widths) + self.z_dim
        self.projection = nn.ModuleList()
        for index, i in enumerate(projection_widths):
            self.projection.append(
                nn.Sequential(
                    nn.Linear(
                        self.z_dim + sum(projection_widths[:index]),
                        i,
                        bias=True,
                    ),
                    nn.BatchNorm1d(i),
                    nn.LeakyReLU(),
                )
            )
        self.projection_upscaler = nn.Upsample(scale_factor=2)

        self.colourspace_r = self.build_colourspace(self.projection_dim, 16)
        self.colourspace_g = self.build_colourspace(self.projection_dim, 16)
        self.colourspace_b = self.build_colourspace(self.projection_dim, 16)
        self.colourspace_upscaler = nn.Upsample(scale_factor=64)

        self.seed = nn.Sequential(
            nn.Linear(
                self.projection_dim,
                512 * 2 * 2,
                bias=True),
            nn.BatchNorm1d(512 * 2 * 2),
            nn.LeakyReLU(),
        )

        self.upscaling = nn.ModuleList()
        self.conv = nn.ModuleList()

        self.upscaling.append(nn.Upsample(scale_factor=2))
        self.conv.append(nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(
                in_channels=(512) // 4,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=True
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
        ))

        self.upscaling.append(nn.Upsample(scale_factor=2))
        self.conv.append(nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(
                in_channels=(256 + self.projection_dim) // 4,
                out_channels=128,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=True
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
        ))

        self.upscaling.append(nn.Upsample(scale_factor=2))
        self.conv.append(nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(
                in_channels=(128 + self.projection_dim) // 4,
                out_channels=64,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=True
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
        ))

        self.upscaling.append(nn.Upsample(scale_factor=2))
        self.conv.append(nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(
                in_channels=(64 + self.projection_dim) // 4,
                out_channels=64,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=True
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
        )),

        self.upscaling.append(nn.Upsample(scale_factor=2))
        self.conv.append(nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(
                in_channels=(64 + self.projection_dim) // 4,
                out_channels=32,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=True
            ),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
        ))

        self.upscaling.append(nn.Upsample(scale_factor=1))
        self.conv.append(nn.Sequential(
            nn.ZeroPad2d((2, 2, 2, 2)),
            nn.Conv2d(
                in_channels=32,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=0,
                bias=True
            ),
            nn.Softmax(dim=1),
        ))

    def forward(self, noise, classes):
        """Forward pass; map latent vectors to samples."""
        em = self.label_emb(classes)
        last = torch.mul(em.squeeze(1), noise.squeeze().squeeze())

        for module in self.projection:
            projection = module(last)
            last = torch.cat((last, projection), -1)
        projection = last
        intermediate = self.seed(projection)
        intermediate = intermediate.view((-1, 512, 2, 2))

        projection_2d = projection.view((-1, self.projection_dim, 1, 1))
        projection_2d = self.projection_upscaler(projection_2d)

        for i, (conv, upscaling) in enumerate(zip(self.conv, self.upscaling)):
            if i + 1 != len(self.upscaling):
                if i > 0:
                    intermediate = torch.cat((intermediate, projection_2d), 1)
                intermediate = torch.nn.functional.pixel_shuffle(intermediate, 2)
            intermediate = conv(intermediate)
            projection_2d = upscaling(projection_2d)

        r_space = self.colourspace_r(projection)
        r_space = r_space.view((-1, 16, 1, 1))
        r_space = self.colourspace_upscaler(r_space)
        r_space = intermediate * r_space
        r_space = torch.sum(r_space, dim=1, keepdim=True)

        g_space = self.colourspace_g(projection)
        g_space = g_space.view((-1, 16, 1, 1))
        g_space = self.colourspace_upscaler(g_space)
        g_space = intermediate * g_space
        g_space = torch.sum(g_space, dim=1, keepdim=True)

        b_space = self.colourspace_b(projection)
        b_space = b_space.view((-1, 16, 1, 1))
        b_space = self.colourspace_upscaler(b_space)
        b_space = intermediate * b_space
        b_space = torch.sum(b_space, dim=1, keepdim=True)

        output = torch.cat((r_space, g_space, b_space), dim=1)

        return output


class CondGeneratorUpSample(nn.Module):
    def __init__(self, n_classes, z_dim, n_features, n_channel, output_size, n_conv_block=3):
        super(CondGeneratorUpSample, self).__init__()

        self.init_size = output_size // (2 ** n_conv_block)  # Initial size before up sampling
        self.n_feature = n_features
        self.n_conv_block = n_conv_block

        self.l1 = nn.Linear(z_dim, n_features * (2 ** n_conv_block) * self.init_size ** 2)
        self.embedding = nn.Embedding(n_classes, z_dim)

        modules = list()
        for layer in reversed(range(n_conv_block)):
            modules.append(ut.generator_layer_up_sample(n_features * (2 ** layer)))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(n_features * (2 ** n_conv_block)),
            nn.Sequential(*modules),
            nn.Conv2d(n_features, n_channel, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noises, classes):
        em = self.embedding(classes)
        em = em.view(em.shape + (1, 1))  # check here
        x = torch.mul(em, noises)
        out = self.l1(x.squeeze())
        out = out.view(-1, self.n_feature * (2 ** self.n_conv_block),
                       self.init_size, self.init_size)
        img = self.conv_blocks(out)

        return img


def _train(gan_params, data_loader, gan_id, n_epoch):
    gan = AuxGAN(gan_params)
    checkpoint_path = os.path.join(var.PROJECT_DIR, f'data/models/{gan_id}.pth')
    if os.path.exists(checkpoint_path):
        print('RESUMING TRAINING...')
        gan.load_model(checkpoint_path)
    else:
        print('NEW TRAINING...')
    print(f'id: {gan_id}')
    gan.init_tensorboard(main_dir='runs', subdir=gan_id, port=8008)
    gan.train(n_epoch=n_epoch, dataloader=data_loader, gan_id=gan_id)
    print(f"{gan_id} TRAINED FOR {n_epoch}")
