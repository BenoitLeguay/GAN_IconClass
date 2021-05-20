import torch
import torch.nn as nn
import variable as var
import utils as ut
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from frechet_inception_distance import InceptionV3, calculate_fid


class WGAN:
    """
    DONE:
        Normalize Image between -1 and 1
        TanH as last layer for G output
        Sample Z from Gaussian distribution
        Batch Normalization for both G and D
        Avoiding Sparse Gradient using LeakyRelu

    TO DO
        Data Augmentation
        add Gaussian Noise to Critic Layers
        Experience Replay (past fake AND swap to older G and D version during training)
    """
    def __init__(self, params, normal_weight_init=True):
        self.critic = Critic(params["critic"]["n_feature"], params["critic"]["n_channel"],
                             params["n_conv_block"]).to(var.device)

        self.generator = self.init_generator(params["generator_type"], params)

        self.critic_optim = torch.optim.Adam(self.critic.parameters(),
                                             lr=params['critic']['lr'],
                                             betas=tuple(params['critic']['betas'].values()))
        self.gen_optim = torch.optim.Adam(self.generator.parameters(),
                                          lr=params['gen']['lr'],
                                          betas=tuple(params['gen']['betas'].values()))

        self.use_inception = params["use_inception"]
        self.z_dim = params['z_dim']
        self.gradient_penalty_factor = params['gradient_penalty_factor']
        self.stability_noise_std = params['stability_noise_std']
        self.noise_distrib = params["noise_distrib"]

        if self.use_inception:
            self.inception = InceptionV3()

        self.writer = None
        self.h_params_added = False
        self.step = 0
        self.epoch = 0

        if normal_weight_init:
            self.critic.apply(ut.weights_init)
            self.generator.apply(ut.weights_init)

        params['GAN_type'] = self.__class__.__name__
        params['gen_optim_type'] = self.gen_optim.__class__.__name__
        params['disc_optim_type'] = self.critic_optim.__class__.__name__

        self.params = ut.flatten_dict(params)

    @staticmethod
    def init_generator(generator_type, params):

        if generator_type == 'convtranspose':
            generator = GeneratorCTranspose(params["z_dim"], params["gen"]["n_feature"],
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
        if self.noise_distrib == 'gaussian':
            return torch.randn(n, self.z_dim, 1, 1, device=var.device)
        elif self.noise_distrib == 'uniform':
            return -2 * torch.rand(n, self.z_dim, 1, 1, device=var.device) + 1

    def apply_stability_noise(self, batch):
        return batch + torch.randn_like(batch) * self.stability_noise_std

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

    def get_gradient_penalty(self, real, fake):
        epsilon = torch.rand(len(fake), 1, 1, 1, device=var.device, requires_grad=True)
        mixes = fake * epsilon + real * (1 - epsilon)
        scores = self.critic(mixes)

        gradient = torch.autograd.grad(inputs=mixes,
                                       outputs=scores,
                                       grad_outputs=torch.ones_like(scores),
                                       create_graph=True,
                                       retain_graph=True)
        gradient = gradient[0].view(len(gradient[0]), -1)
        return torch.mean((gradient.norm(2, dim=1) - 1)**2)

    def get_critic_loss(self, critic_fake_pred, critic_real_pred, gradient_penalty):
        critic_loss = torch.mean(critic_fake_pred) - torch.mean(critic_real_pred)
        return critic_loss + self.gradient_penalty_factor * gradient_penalty

    @staticmethod
    def get_generator_loss(critic_fake_pred):
        return -torch.mean(critic_fake_pred)

    def update_critic(self, batch_size, real):
        self.critic_optim.zero_grad()
        fake = self.generate_fake(batch_size)

        fake = self.apply_stability_noise(fake)
        real = self.apply_stability_noise(real)

        critic_fake_pred = self.critic(fake.detach())
        critic_real_pred = self.critic(real)

        gradient_penalty = self.get_gradient_penalty(real, fake.detach())
        critic_loss = self.get_critic_loss(critic_fake_pred, critic_real_pred, gradient_penalty)
        critic_loss.backward(retain_graph=True)
        self.critic_optim.step()

        return critic_loss.item()

    def update_generator(self, batch_size):
        self.gen_optim.zero_grad()
        fake = self.generate_fake(batch_size)

        fake = self.apply_stability_noise(fake)

        critic_fake_pred = self.critic(fake)
        generator_loss = self.get_generator_loss(critic_fake_pred)
        generator_loss.backward()
        self.gen_optim.step()

        return generator_loss.item()

    def train(self, n_epoch, dataloader, n_critic_update=1, n_generator_update=1, gan_id=None):

        assert n_critic_update > 0 and n_generator_update > 0, "n_update must be greater than 0"

        for _ in tqdm(range(n_epoch)):
            gen_epoch_loss = 0.0
            critic_epoch_loss = 0.0

            for i_batch, (real, _) in enumerate(dataloader):
                critic_loss = 0.0
                generator_loss = 0.0

                cur_batch_size = len(real)
                real = real.to(var.device)

                for _ in range(n_critic_update):
                    critic_loss += self.update_critic(cur_batch_size, real)
                for _ in range(n_generator_update):
                    generator_loss += self.update_generator(cur_batch_size)

                critic_epoch_loss += critic_loss / n_critic_update
                gen_epoch_loss += generator_loss / n_generator_update

                if i_batch == 0:
                    fake = self.generate_fake(cur_batch_size, train=False)

                    if self.use_inception:
                        fid = calculate_fid(real.detach(), fake, self.inception, resize=(75, 75))
                        self.writer.add_scalar('Frechet Inception Distance', fid, self.epoch)

                    self.writer.add_image(f'{self.epoch}/Fake', ut.images_grid(fake))
                    self.writer.add_image(f'{self.epoch}/Real', ut.images_grid(real))

                self.step += 1

            self.writer.add_scalar('Loss/Critic', critic_epoch_loss / len(dataloader), self.epoch)
            self.writer.add_scalar('Loss/Generator', gen_epoch_loss / len(dataloader), self.epoch)

            self.epoch += 1

        if gan_id:
            self.save_model(gan_id)

    def save_model(self, gan_id):
        model_path = os.path.join(var.PROJECT_DIR, f"data/models/{gan_id}.pth")
        torch.save({
            'step': self.step,
            'z_dim': self.z_dim,
            'epoch': self.epoch,
            'h_params_added': self.h_params_added,
            'gradient_penalty_factor': self.gradient_penalty_factor,
            'stability_noise_std': self.stability_noise_std,
            'generator_state_dict': self.generator.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'generator_optim_state_dict': self.gen_optim.state_dict(),
            'critic_optim_state_dict': self.critic_optim.state_dict()
        },
            model_path)

    def load_model(self, path, train=True):
        checkpoint = torch.load(path)

        self.step = checkpoint['step']
        self.epoch = checkpoint['epoch']
        self.z_dim = checkpoint['z_dim']
        self.h_params_added = checkpoint['h_params_added']
        self.gradient_penalty_factor = checkpoint['gradient_penalty_factor']
        self.stability_noise_std = checkpoint['stability_noise_std']

        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.gen_optim.load_state_dict(checkpoint['generator_optim_state_dict'])
        self.critic_optim.load_state_dict(checkpoint['critic_optim_state_dict'])

        if train:
            self.generator.train()
            self.critic.train()
        else:
            self.generator.eval()
            self.critic.eval()


class Critic(nn.Module):
    def __init__(self, n_features, n_channel, n_conv_block=3):
        super(Critic, self).__init__()

        modules = list()
        for layer in range(n_conv_block):
            modules.append(ut.critic_layer(n_features * (2 ** layer)))

        self.main = nn.Sequential(
            nn.Conv2d(n_channel, n_features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Sequential(*modules),  # convolution blocks
            nn.Conv2d(n_features * (2 ** n_conv_block), 1, 4, 1, 0, bias=False),
        )

    def forward(self, x):
        return self.main(x)  # testing with 2 nn archi (conv produce 1d output or Linear produce it)


class GeneratorCTranspose(nn.Module):
    def __init__(self, n_latent, n_features, n_channel, n_conv_block=3):
        super(GeneratorCTranspose, self).__init__()

        modules = list()
        for layer in reversed(range(n_conv_block)):
            modules.append(ut.generator_layer(n_features * (2 ** layer)))

        self.main = nn.Sequential(
            nn.ConvTranspose2d(n_latent, n_features * (2 ** n_conv_block), 4, 1, 0, bias=False),
            nn.BatchNorm2d(n_features * (2 ** n_conv_block)),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Sequential(*modules),  # convolution blocks
            nn.ConvTranspose2d(n_features, n_channel, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)


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


class CondGeneratorColorPicker(nn.Module):
    """A generator for mapping a latent space to a sample space.
    Input shape: (?, latent_dim)
    Output shape: (?, 3, 96, 96)
    """

    def __init__(self, z_dim):
        """Initialize generator.
        Args:
            latent_dim (int): latent dimension ("noise vector")
        """
        super().__init__()
        self.z_dim = z_dim
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

    def forward(self, noise):
        """Forward pass; map latent vectors to samples."""
        last = noise.squeeze()

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


def train(gan_params, data_loader, gan_id, n_epoch):
    gan = WGAN(gan_params)
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
