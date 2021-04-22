import torch
from torch import nn
import utils as ut
import variable as var
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio
import glob
import os

"""
Most trivial version of GAN
"""


class GAN:
    def __init__(self, params):
        self.loss = nn.BCEWithLogitsLoss()
        self.discriminator = Discriminator(params["gen"]["x_dim"], params["gen"]["hidden_dim"]).to(var.device)
        self.generator = Generator(params["z_dim"], params["gen"]["hidden_dim"], params["gen"]["x_dim"]).to(var.device)
        self.disc_optim = torch.optim.Adam(self.discriminator.parameters(), lr=params['disc']['lr'])
        self.gen_optim = torch.optim.Adam(self.generator.parameters(), lr=params['gen']['lr'])
        self.z_dim = params['z_dim']
        self.x_dim = params["gen"]["x_dim"]

    def get_random_noise(self, n):
        return torch.randn(n, self.z_dim, device=var.device)

    def generate_fake(self, n):
        noise = self.get_random_noise(n)
        return self.generator.forward(noise)

    def get_discriminator_loss(self, batch_size, real):
        noise = self.get_random_noise(batch_size)
        fake_batch = self.generator.forward(noise)
        disc_fake_pred = self.discriminator(fake_batch.detach())
        disc_fake_loss = self.loss(disc_fake_pred, torch.zeros_like(disc_fake_pred))
        disc_real_pred = self.discriminator(real)
        disc_real_loss = self.loss(disc_real_pred, torch.ones_like(disc_real_pred))
        disc_loss = (disc_fake_loss + disc_real_loss) / 2

        return disc_loss

    def get_generator_loss(self, batch_size):
        noise = self.get_random_noise(batch_size)
        fake = self.generator(noise)
        disc_fake_pred = self.discriminator(fake)
        loss = self.loss(disc_fake_pred, torch.ones_like(disc_fake_pred))

        return loss

    def train(self, n_epoch, dataloader, x, viz_each=200, viz_to_gif=False):
        disc_loss_history = list()
        gen_loss_history = list()
        mean_disc_loss_history = list()
        mean_gen_loss_history = list()

        step = 0

        for epoch in range(n_epoch):
            for real, _ in dataloader:
                cur_batch_size = len(real)
                real = real.view(cur_batch_size, -1).to(var.device)

                self.disc_optim.zero_grad()
                disc_loss = self.get_discriminator_loss(cur_batch_size, real)
                disc_loss.backward(retain_graph=True)
                self.disc_optim.step()

                self.gen_optim.zero_grad()
                gen_loss = self.get_generator_loss(cur_batch_size)
                gen_loss.backward()
                self.gen_optim.step()

                disc_loss_history.append(disc_loss.item())
                gen_loss_history.append(gen_loss.item())

                if step % viz_each == 0.0 and step > 0:
                    print(f"epoch: {epoch}, step: {step}")
                    mean_disc_loss_history.append(sum(disc_loss_history)/len(disc_loss_history))
                    mean_gen_loss_history.append(sum(gen_loss_history)/len(gen_loss_history))

                    fake = self.generate_fake(1000).detach().cpu().numpy()
                    if self.x_dim == 2:
                        self.scatter_2d(x, fake, viz_to_gif=viz_to_gif, step=step, epoch=epoch)
                    elif self.x_dim == 3:
                        self.scatter_3d(x, fake, viz_to_gif=viz_to_gif, step=step, epoch=epoch)

                    plt.plot(range(len(mean_disc_loss_history)), mean_disc_loss_history, label="Discriminator loss")
                    plt.plot(range(len(mean_gen_loss_history)), mean_gen_loss_history, label="Generator loss")

                    plt.legend()
                    plt.show()

                step += 1

        if viz_to_gif:
            self.save_gif()

    @staticmethod
    def scatter_3d(real, fake, viz_to_gif=False, step=None, epoch=None):
        fig = plt.figure()
        ax = Axes3D(fig)

        ax.scatter(real[:, 0], real[:, 1], real[:, 2], c='b', marker='x', label='real')
        ax.scatter(fake[:, 0], fake[:, 1], fake[:, 2], c='r', marker='o', label='fake')
        plt.legend()

        if viz_to_gif:
            plt.title(f"epoch: {epoch}")
            plt.savefig(f'_tmp/{step}.png')
        plt.show()

    @staticmethod
    def scatter_2d(real, fake, viz_to_gif=False, step=None, epoch=None):

        plt.scatter(real[:, 0], real[:, 1], c='b', marker='x', label='real')
        plt.scatter(fake[:, 0], fake[:, 1], c='r', marker='o', label='fake')
        plt.legend()

        if viz_to_gif:
            plt.title(f"epoch: {epoch}")
            plt.savefig(f'_tmp/{step}.png')
        plt.show()

    @staticmethod
    def save_gif():
        images = sorted([log for log in glob.glob("_tmp/*") if not os.path.isdir(log)],
                        key=lambda x: int(os.path.splitext(x)[0][5:]))
        with imageio.get_writer('mygif.gif', mode='I') as writer:
            for filename in images:
                image = imageio.imread(filename)
                writer.append_data(image)
        for filename in images:
            os.remove(filename)


class Generator(nn.Module):
    def __init__(self, z_dim, hidden_dim, x_dim):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim * 2, x_dim)
        )

    def forward(self, noise):
        return self.gen(noise)


class Discriminator(nn.Module):
    def __init__(self, x_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Linear(x_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, image):
        return self.disc(image)
