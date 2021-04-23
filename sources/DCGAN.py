import torch
from torch import nn
import utils as ut
import variable as var
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
from frechet_inception_distance import InceptionV3, calculate_fid


class DCGAN:
    def __init__(self, params, normal_weight_init=True):
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.discriminator = Discriminator(params["disc"]["n_feature"], params["disc"]["n_channel"],
                                           n_conv_block=params["n_conv_block"]).to(var.device)

        self.generator = self.init_generator(params["generator_type"], params)

        self.disc_optim = torch.optim.Adam(self.discriminator.parameters(), lr=params['disc']['lr'],
                                           betas=tuple(params['disc']['betas'].values()))
        self.gen_optim = torch.optim.Adam(self.generator.parameters(), lr=params['gen']['lr'],
                                          betas=tuple(params['gen']['betas'].values()))
        self.z_dim = params['z_dim']
        self.use_inception = params["use_inception"]
        self.label_smoothing = params["label_smoothing"]

        if self.use_inception:
            self.inception = InceptionV3()

        self.writer = None
        self.h_params_added = False
        self.step = 0
        self.epoch = 0

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
            generator = GeneratorConvTranspose(params["z_dim"], params["gen"]["n_feature"],
                                               params["gen"]["n_channel"],
                                               n_conv_block=params["n_conv_block"])
        elif generator_type == 'upsample':
            generator = GeneratorUpSample(params["z_dim"], params["gen"]["n_feature"], params["gen"]["n_channel"],
                                          params["output_size"], n_conv_block=params["n_conv_block"])
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

    def get_random_noise(self, n):
        return torch.randn(n, self.z_dim, 1, 1, device=var.device)

    def ground_truth_label(self, shape_like_tensor, smoothing_value=.9):
        ground_truth = torch.ones_like(shape_like_tensor)
        if self.label_smoothing:
            ground_truth *= smoothing_value
        return ground_truth

    def generate_fake(self, n, train=True):
        noise = self.get_random_noise(n)
        if train:
            return self.generator.forward(noise)
        else:
            self.generator.eval()
            with torch.no_grad():
                fake = self.generator.forward(noise)
            self.generator.train()
            return fake

    def get_discriminator_loss(self, batch_size, real):
        noise = self.get_random_noise(batch_size)
        fake = self.generator.forward(noise)

        fake_pred = self.discriminator(fake.detach())
        fake_loss = self.bce_loss(fake_pred, torch.zeros_like(fake_pred))

        real_pred = self.discriminator(real)
        real_loss = self.bce_loss(real_pred, self.ground_truth_label(real_pred))

        disc_loss = (fake_loss + real_loss) / 2

        self.compute_discriminator_accuracy(real_pred, fake_pred)

        return disc_loss

    def get_generator_loss(self, batch_size):
        noise = self.get_random_noise(batch_size)
        fake = self.generator(noise)

        fake_pred = self.discriminator(fake)
        loss = self.bce_loss(fake_pred, self.ground_truth_label(fake_pred))

        return loss

    def train(self, n_epoch, dataloader, gan_id=None):

        for _ in tqdm(range(n_epoch)):
            gen_epoch_loss = 0.0
            disc_epoch_loss = 0.0

            for i_batch, (real, _) in enumerate(dataloader):
                cur_batch_size = len(real)
                real = real.to(var.device)

                # UPDATE DISCRIMINATOR #
                self.disc_optim.zero_grad()
                disc_loss = self.get_discriminator_loss(cur_batch_size, real)
                disc_loss.backward(retain_graph=True)
                self.disc_optim.step()

                # UPDATE GENERATOR #
                self.gen_optim.zero_grad()
                gen_loss = self.get_generator_loss(cur_batch_size)
                gen_loss.backward()
                self.gen_optim.step()

                disc_epoch_loss += disc_loss.item()
                gen_epoch_loss += gen_loss.item()

                if i_batch == 0:
                    fake = self.generate_fake(cur_batch_size, train=False)

                    if self.use_inception:
                        fid = calculate_fid(real.detach(), fake, self.inception, resize=(75, 75))
                        self.writer.add_scalar('Frechet Inception Distance', fid, self.epoch)

                    self.writer.add_image(f'{self.epoch}/Fake', ut.images_grid(fake))
                    self.writer.add_image(f'{self.epoch}/Real', ut.images_grid(real))

                self.step += 1

            self.writer.add_scalar('Loss/Discriminator', disc_epoch_loss / len(dataloader), self.epoch)
            self.writer.add_scalar('Loss/Generator', gen_epoch_loss / len(dataloader), self.epoch)

            self.epoch += 1

        if gan_id:
            self.save_model(gan_id)

    def compute_discriminator_accuracy(self, real_adv, fake_adv):
        fake_adv_label = fake_adv < .5
        real_adv_label = real_adv > .5

        self.writer.add_scalar('D-Accuracy/Fake', float(fake_adv_label.sum()/len(fake_adv_label)), self.step)
        self.writer.add_scalar('D-Accuracy/Real', float(real_adv_label.sum()/len(real_adv_label)), self.step)

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
            'discriminator_optim_state_dict': self.disc_optim.state_dict()
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


class Discriminator(nn.Module):
    def __init__(self, n_features, n_channel, n_conv_block=3):
        super(Discriminator, self).__init__()
        modules = list()
        for layer in range(n_conv_block):
            modules.append(ut.discriminator_layer(n_features * (2 ** layer)))

        self.main = nn.Sequential(
            nn.Conv2d(n_channel, n_features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Sequential(*modules),  # convolution blocks
            nn.Conv2d(n_features * (2 ** n_conv_block), 1, 4, 1, 0, bias=False)
        )

    def forward(self, image):
        return self.main(image)


class GeneratorConvTranspose(nn.Module):
    def __init__(self, z_dim, n_features, n_channel, n_conv_block=3):
        super(GeneratorConvTranspose, self).__init__()

        modules = list()
        for layer in reversed(range(n_conv_block)):
            modules.append(ut.generator_layer(n_features * (2 ** layer)))
        # set BatchNorm2d as first
        self.main = nn.Sequential(
            nn.ConvTranspose2d(z_dim, n_features * (2 ** n_conv_block), 4, 1, 0, bias=False),
            nn.BatchNorm2d(n_features * (2 ** n_conv_block)),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Sequential(*modules),  # convolution blocks
            nn.ConvTranspose2d(n_features, n_channel, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise):
        return self.main(noise)


class GeneratorUpSample(nn.Module):
    def __init__(self, z_dim, n_features, n_channel, output_size, n_conv_block=3):
        super(GeneratorUpSample, self).__init__()

        self.init_size = output_size // (2 ** n_conv_block)  # Initial size before up sampling
        self.n_feature = n_features
        self.n_conv_block = n_conv_block

        self.l1 = nn.Sequential(
            nn.Linear(z_dim, n_features * (2 ** n_conv_block) * self.init_size ** 2)
        )
        modules = list()
        for layer in reversed(range(n_conv_block)):
            modules.append(ut.generator_layer_up_sample(n_features * (2 ** layer)))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(n_features * (2 ** n_conv_block)),
            nn.Sequential(*modules),
            nn.Conv2d(n_features, n_channel, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise):
        noise = noise.squeeze()
        out = self.l1(noise)
        out = out.view(-1, self.n_feature * (2 ** self.n_conv_block),
                       self.init_size, self.init_size)
        img = self.conv_blocks(out)

        return img


def train(gan_params, data_loader, gan_id, n_epoch):
    gan = DCGAN(gan_params)
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

