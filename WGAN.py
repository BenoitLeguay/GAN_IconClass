import torch
import torch.nn as nn
import variable as var
import utils as ut
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


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
        ColourPicker (NO)
        add Gaussian Noise to Critic Layers
        Experience Replay (past fake AND swap to older G and D version during training)
    """
    def __init__(self, params, normal_weight_init=True):
        self.critic = Critic(params["critic"]["n_feature"], params["critic"]["n_channel"],
                             params["n_conv_block"]).to(var.device)
        self.generator = Generator(params["z_dim"], params["gen"]["n_feature"],
                                   params["gen"]["n_channel"], params["n_conv_block"]).to(var.device)

        self.critic_optim = torch.optim.Adam(self.critic.parameters(),
                                             lr=params['critic']['lr'],
                                             betas=params['critic']['betas'])
        self.gen_optim = torch.optim.Adam(self.generator.parameters(),
                                          lr=params['gen']['lr'],
                                          betas=params['gen']['betas'])

        self.z_dim = params['z_dim']
        self.gradient_penalty_factor = params['gradient_penalty_factor']
        self.stability_noise_std = params['stability_noise_std']

        self.writer = None
        self.step = 0

        if normal_weight_init:
            self.critic.apply(ut.weights_init)
            self.generator.apply(ut.weights_init)

        params['GAN_type'] = self.__class__.__name__
        params['gen_optim_type'] = self.gen_optim.__class__.__name__
        params['disc_optim_type'] = self.critic_optim.__class__.__name__

        self.params = ut.flatten_dict(params)

    def init_tensorboard(self, main_dir='runs', subdir='train', port=8008):
        os.system(f'tensorboard --logdir={main_dir} --port={port} &')
        self.writer = SummaryWriter(f'{main_dir}/{subdir}')

    def get_random_noise(self, n):
        return torch.randn(n, self.z_dim, 1, 1, device=var.device)

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

        self.compute_critic_accuracy(critic_fake_pred, critic_real_pred)

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
            for i_batch, (real, _) in enumerate(dataloader):
                critic_loss = 0.0
                generator_loss = 0.0

                cur_batch_size = len(real)
                real = real.to(var.device)

                for _ in range(n_critic_update):
                    critic_loss += self.update_critic(cur_batch_size, real)
                for _ in range(n_generator_update):
                    generator_loss += self.update_generator(cur_batch_size)

                self.writer.add_scalar('Loss/Train/Critic', critic_loss/n_critic_update, self.step)
                self.writer.add_scalar('Loss/Train/Generator', generator_loss/n_generator_update, self.step)

                if i_batch == 0:
                    fake = self.generate_fake(10, train=False)

                    image_id = int(self.step/len(dataloader))
                    self.writer.add_image(f'{image_id}/Fake', ut.return_tensor_images(fake))
                    self.writer.add_image(f'{image_id}/Real', ut.return_tensor_images(real))

                self.step += 1
        if gan_id:
            self.writer.add_hparams(self.params)
            self.save_model(gan_id)

    def compute_critic_accuracy(self, critic_fake_pred, critic_real_pred):
        fake_label = critic_fake_pred < .5
        real_label = critic_real_pred > .5

        self.writer.add_scalar('Accuracy/Critic/Fake', float(fake_label.sum()/len(fake_label)), self.step)
        self.writer.add_scalar('Accuracy/Critic/Real', float(real_label.sum()/len(real_label)), self.step)

    def save_model(self, gan_id):
        torch.save({
            'step': self.step,
            'z_dim': self.z_dim,
            'gradient_penalty_factor': self.gradient_penalty_factor,
            'stability_noise_std': self.stability_noise_std,
            'generator_state_dict': self.generator.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'generator_optim_state_dict': self.gen_optim.state_dict(),
            'critic_optim_state_dict': self.critic_optim.state_dict()
        },
            f"data/models/{gan_id}.pth")

    def load_model(self, path, train=True):
        checkpoint = torch.load(path)

        self.step = checkpoint['step']
        self.z_dim = checkpoint['z_dim']
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
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)


class Generator(nn.Module):
    def __init__(self, n_latent, n_features, n_channel, n_conv_block=3):
        super(Generator, self).__init__()

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


class Generatorx128(nn.Module):
    def __init__(self, n_latent, n_features, n_channel):
        super(Generatorx128, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(n_latent, n_features * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(n_features * 16),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (n_features*8) x 4 x 4
            nn.ConvTranspose2d(n_features * 16, n_features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_features * 8),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (n_features*4) x 8 x 8
            nn.ConvTranspose2d(n_features * 8, n_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_features * 4),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (n_features*2) x 16 x 16
            nn.ConvTranspose2d(n_features * 4, n_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_features * 2),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (n_features*2) x 32 x 32
            nn.ConvTranspose2d(n_features * 2, n_features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_features),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (n_features) x 64 x 64
            nn.ConvTranspose2d(n_features, n_channel, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (n_channel) x 128 x 128
        )

    def forward(self, x):
        return self.main(x)


class Criticx128(nn.Module):
    def __init__(self, n_features, n_channel):
        super(Criticx128, self).__init__()
        self.main = nn.Sequential(
            # input is (n_channel) x 64 x 64
            nn.Conv2d(n_channel, n_features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (n_features) x 32 x 32
            nn.Conv2d(n_features, n_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (n_features*2) x 16 x 16
            nn.Conv2d(n_features * 2, n_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (n_features*4) x 8 x 8
            nn.Conv2d(n_features * 4, n_features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (n_features*4) x 8 x 8
            nn.Conv2d(n_features * 8, n_features * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_features * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (n_features*8) x 4 x 4
            nn.Conv2d(n_features * 16, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)

