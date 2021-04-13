import torch
from torch import nn
import utils as ut
import variable as var
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os


class DCGAN:
    def __init__(self, params, normal_weight_init=True):
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.discriminator = Discriminator(params["disc"]["n_feature"], params["disc"]["n_channel"],
                                           n_conv_block=params["n_conv_block"]).to(var.device)
        self.generator = Generator(params["z_dim"], params["gen"]["n_feature"], params["gen"]["n_channel"],
                                   n_conv_block=params["n_conv_block"]).to(var.device)
        self.disc_optim = torch.optim.Adam(self.discriminator.parameters(), lr=params['disc']['lr'],
                                           betas=params['disc']['betas'])
        self.gen_optim = torch.optim.Adam(self.generator.parameters(), lr=params['gen']['lr'],
                                          betas=params['disc']['betas'])
        self.z_dim = params['z_dim']

        self.writer = None
        self.step = 0

        if normal_weight_init:
            self.discriminator.apply(ut.weights_init)
            self.generator.apply(ut.weights_init)

    def init_tensorboard(self, main_dir='runs', subdir='train', port=8008):
        os.system(f'tensorboard --logdir={main_dir} --port={port} &')
        self.writer = SummaryWriter(f'{main_dir}/{subdir}')

    def get_random_noise(self, n):
        return torch.randn(n, self.z_dim, 1, 1, device=var.device)

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
        real_loss = self.bce_loss(real_pred, torch.ones_like(real_pred))

        disc_loss = (fake_loss + real_loss) / 2

        return disc_loss

    def get_generator_loss(self, batch_size):
        noise = self.get_random_noise(batch_size)
        fake = self.generator(noise)

        fake_pred = self.discriminator(fake)
        loss = self.bce_loss(fake_pred, torch.ones_like(fake_pred))

        return loss

    def train(self, n_epoch, dataloader, gan_id=None):

        for _ in tqdm(range(n_epoch)):
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

                self.writer.add_scalar('Loss/Train/Discriminator', disc_loss, self.step)
                self.writer.add_scalar('Loss/Train/Generator', gen_loss, self.step)

                if i_batch == 0:
                    fake = self.generate_fake(10, train=False)

                    image_id = int(self.step / len(dataloader))
                    self.writer.add_image(f'{image_id}/Fake', ut.return_tensor_images(fake))
                    self.writer.add_image(f'{image_id}/Real', ut.return_tensor_images(real))

                self.step += 1
            if gan_id:
                self.save_model(gan_id)

    def save_model(self, gan_id):
        torch.save({
            'step': self.step,
            'z_dim': self.z_dim,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'generator_optim_state_dict': self.gen_optim.state_dict(),
            'discriminator_optim_state_dict': self.disc_optim.state_dict()
        },
            f"data/models/{gan_id}.pth")

    def load_model(self, path, train=True):
        checkpoint = torch.load(path)

        self.step = checkpoint['step']
        self.z_dim = checkpoint['z_dim']

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
            modules.append(ut.critic_layer(n_features * (2 ** layer)))

        self.main = nn.Sequential(
            nn.Conv2d(n_channel, n_features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Sequential(*modules),  # convolution blocks
            nn.Conv2d(n_features * (2 ** n_conv_block), 1, 4, 1, 0, bias=False)
        )

    def forward(self, image):
        return self.main(image)


class Generator(nn.Module):
    def __init__(self, z_dim, n_features, n_channel, n_conv_block=3):
        super(Generator, self).__init__()

        modules = list()
        for layer in reversed(range(n_conv_block)):
            modules.append(ut.generator_layer(n_features * (2 ** layer)))

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
