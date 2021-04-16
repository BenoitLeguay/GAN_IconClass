import torch
import torch.nn as nn
import variable as var
import utils as ut
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class WACGAN:
    """
    DONE:
        Normalize Image between -1 and 1
        TanH as last layer for G output
        Sample Z from Gaussian distribution
        Batch Normalization for both G and D
        Avoiding Sparse Gradient using LeakyRelu

    TO DO
        Data Augmentation
        ColourPicker
        add Gaussian Noise to Critic Layers
        Experience Replay (past fake AND swap to older G and D version during training)
    """
    def __init__(self, params, normal_weight_init=True):
        self.critic = Critic(params["n_classes"], params["critic"]["n_feature"],
                             params["critic"]["n_channel"], params["n_conv_block"]).to(var.device)
        self.generator = Generator(params["n_classes"], params["z_dim"],
                                   params["gen"]["n_feature"],  params["gen"]["n_channel"],
                                   params["n_conv_block"]).to(var.device)

        self.critic_optim = torch.optim.Adam(self.critic.parameters(),
                                             lr=params['critic']['lr'],
                                             betas=params['critic']['betas'])
        self.gen_optim = torch.optim.Adam(self.generator.parameters(),
                                          lr=params['gen']['lr'],
                                          betas=params['gen']['betas'])
        self.aux_loss = nn.NLLLoss()

        self.z_dim = params['z_dim']
        self.gradient_penalty_factor = params['gradient_penalty_factor']
        self.stability_noise_std = params['stability_noise_std']
        self.n_classes = params["n_classes"]

        self.writer = None
        self.step = 0

        if normal_weight_init:
            self.critic.apply(ut.weights_init)
            self.generator.apply(ut.weights_init)

    def init_tensorboard(self, main_dir='runs', subdir='train', port=8008):
        os.system(f'tensorboard --logdir={main_dir} --port={port} &')
        self.writer = SummaryWriter(f'{main_dir}/{subdir}')

    def get_random_noise(self, n):
        return torch.randn(n, self.z_dim, 1, 1, device=var.device)

    def get_random_classes(self, n):
        return torch.randint(self.n_classes, (n, ), device=var.device)

    def apply_stability_noise(self, batch):
        return batch + torch.randn_like(batch) * self.stability_noise_std

    def generate_fake(self, n, train=True, being_class=None):
        noise = self.get_random_noise(n)
        if being_class:
            classes = torch.ones((n,)).long() * being_class
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

    def get_gradient_penalty(self, real, fake):
        epsilon = torch.rand(len(fake), 1, 1, 1, device=var.device, requires_grad=True)
        mixes = fake * epsilon + real * (1 - epsilon)
        scores, _ = self.critic(mixes)

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

    def update_critic(self, batch_size, real, real_classes):
        self.critic_optim.zero_grad()
        fake, fake_classes = self.generate_fake(batch_size)

        fake = self.apply_stability_noise(fake)
        real = self.apply_stability_noise(real)

        fake_adv, fake_aux = self.critic(fake.detach())
        real_adv, real_aux = self.critic(real)

        self.compute_critic_accuracy(fake_adv, real_adv)

        gradient_penalty = self.get_gradient_penalty(real, fake.detach())

        critic_loss = self.get_critic_loss(fake_adv, real_adv, gradient_penalty)

        critic_loss += self.aux_loss(fake_aux, fake_classes)
        critic_loss += self.aux_loss(real_aux, real_classes)

        critic_loss.backward(retain_graph=True)
        self.critic_optim.step()

        return critic_loss.item()

    def update_generator(self, batch_size):
        self.gen_optim.zero_grad()
        fake, fake_classes = self.generate_fake(batch_size)

        fake = self.apply_stability_noise(fake)

        fake_adv, fake_aux = self.critic(fake)

        generator_loss = self.get_generator_loss(fake_adv)
        generator_loss += self.aux_loss(fake_aux, fake_classes)

        generator_loss.backward()
        self.gen_optim.step()

        return generator_loss.item()

    def train(self, n_epoch, dataloader, n_critic_update=1, n_generator_update=1, gan_id=None):

        assert n_critic_update > 0 and n_generator_update > 0, "n_update must be greater than 0"

        for _ in tqdm(range(n_epoch)):
            for i_batch, (real, real_classes) in enumerate(dataloader):

                critic_loss = 0.0
                generator_loss = 0.0

                cur_batch_size = len(real)
                real = real.to(var.device)
                real_classes = real_classes.to(var.device)

                for _ in range(n_critic_update):
                    critic_loss += self.update_critic(cur_batch_size, real, real_classes)
                for _ in range(n_generator_update):
                    generator_loss += self.update_generator(cur_batch_size)

                self.writer.add_scalar('Loss/Train/Critic', critic_loss/n_critic_update, self.step)
                self.writer.add_scalar('Loss/Train/Generator', generator_loss/n_generator_update, self.step)

                if i_batch == 0:
                    fake, fake_classes = self.generate_fake(10, train=False)

                    fake_grid = ut.return_tensor_images(fake)
                    real_grid = ut.return_tensor_images(real)

                    image_id = int(self.step/len(dataloader))
                    self.writer.add_image(f'{image_id}/Fake', fake_grid)
                    self.writer.add_image(f'{image_id}/Real', real_grid)

                self.step += 1
        if gan_id:
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
    def __init__(self, n_classes, n_features, n_channel, n_conv_block=3):
        super(Critic, self).__init__()
        self.n_features = n_features

        modules = list()
        for layer in range(n_conv_block):
            modules.append(ut.critic_layer(n_features * (2 ** layer)))

        self.main = nn.Sequential(
            nn.Conv2d(n_channel, n_features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Sequential(*modules),  # convolution blocks
            nn.Conv2d(n_features * (2 ** n_conv_block), n_features, 4, 1, 0, bias=False),
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


class Generator(nn.Module):
    def __init__(self, n_classes, n_latent, n_features, n_channel, n_conv_block=3):
        super(Generator, self).__init__()
        self.n_classes = n_classes

        modules = list()
        for layer in reversed(range(n_conv_block)):
            modules.append(ut.generator_layer(n_features * (2 ** layer)))

        self.main = nn.Sequential(
            nn.ConvTranspose2d(n_latent + n_classes, n_features * (2 ** n_conv_block), 4, 1, 0, bias=False),
            nn.BatchNorm2d(n_features * (2 ** n_conv_block)),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Sequential(*modules),  # convolution blocks
            nn.ConvTranspose2d(n_features, n_channel, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, classes):
        x = self.cat_noise_classes(noise, classes)
        return self.main(x)

    def cat_noise_classes(self, noise, classes):
        one_hot = torch.eye(self.n_classes, device=var.device)
        one_hot = one_hot[classes]
        one_hot = one_hot.view(one_hot.shape + (1, 1))

        return torch.cat((noise, one_hot), axis=1)


