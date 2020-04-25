## CycleGAN remake

# Todo-list:
# 1) Get dataset of normal items.
# 2) Make Generators A and B
# 3) Make Discriminators A and B
# 4) Compute loss with cycle consistency term + normal GAN loss, see paper!

# Program:
# Load in dataset
# Make G_A, G_B
# Make D_A, G_B
# Training Loop:
# Train GANS together in training loop
# Calculate loss

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import PIL

class Generator(nn.Module):
    """ResNet, but hardcode the layers (in contrast to the real CycleGAN).
    We don't use dropout, since the authors of CycleGAN did not use this either.
    """

    def __init__(self, opt):
        super(Generator, self).__init__()
        self.n_blocks = 6       # Hardcoded for now
        ngf = 64           # Harcoded, num filters in last conv layer

        # The ReflectionPad2d layer pads the input tensor. The value is 3 because we use a 7x7 conv layer
        self.model = [nn.ReflectionPad2d(3),
                      nn.Conv2d(opt.input_nc, ngf, kernel_size=7, padding=0, bias=False),
                      nn.BatchNorm2d(ngf),
                      nn.ReLU(True)]

        # Add downsampling layers (see CycleGAN implementation by the authors)
        n_downsampling_layers = 2
        for i in range(n_downsampling_layers):
            mult = 2 ** i
            self.model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=False),
                           nn.BatchNorm2d(ngf * mult * 2),
                           nn.ReLU(True)]

        # Add ResNet blocks
        mult = 2 ** n_downsampling_layers
        for i in range(self.n_blocks):
            self.model += [ResnetBlock(ngf, mult)] # todo: understand these blocks! https://arxiv.org/pdf/1512.03385.pdf

        # Add upsampling layers
        for i in range(n_downsampling_layers):
            mult = 2 ** (n_downsampling_layers - i)
            self.model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                              kernel_size=3, stride=2,
                                              padding=1, output_padding=1,
                                              bias=False),
                           nn.BatchNorm2d(int(ngf * mult / 2)),
                           nn.ReLU(True)]

        self.model += [nn.ReflectionPad2d(3),   # The value is 3 because we use a 7x7 conv layer
                       nn.Conv2d(ngf, opt.output_nc, kernel_size=7, padding=0), # todo: understand why this conv2d is here and not convtranspose2d
                       nn.Tanh()]

        self.model = nn.Sequential(*self.model)

    def forward(self, input):
        self.model(input)


class ResnetBlock(nn.Module):
    def __init__(self, ngf, mult):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(ngf, mult)

    def build_conv_block(self, ngf, mult):
        conv_block = [nn.ReflectionPad2d(1),  # The value is 1 because we use 3x3 convolutions
                      nn.Conv2d(ngf * mult, ngf * mult, kernel_size=3, padding=0, bias=False),
                      nn.BatchNorm2d(ngf * mult),
                      nn.ReLU(True),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(ngf * mult, ngf * mult, kernel_size=3, padding=0, bias=False),
                      nn.BatchNorm2d(ngf * mult)] 
        return nn.Sequential(*conv_block)

    def forward(self, input):
        """ResNet forward function: todo read paper and understand this! https://arxiv.org/pdf/1512.03385.pdf"""
        out = x + self.conv_block(x)
        return out


class Discriminator(nn.Module):
    """Implementing the PatchGAN used by the CycleGAN authors.""" 
    def __init__(self, opt):
        pass

    def forward(self, input):
        pass

    def backward(self):
        pass


class Options:
    """A class that keeps track of all user-defined options"""
    def __init__(self):
        self.input_nc = 3       # num channels, usually 3 (RGB)
        self.output_nc = 3      # num channels, usually 3 (RGB)


class CycleGAN:
    def __init__(self):
        self.opt = Options()        # Hardcoded options
        self.G_A = Generator(self.opt)      # todo: args
        self.G_B = Generator(self.opt)      # todo: args
        self.D_A = Discriminator(self.opt)  # todo: args
        self.D_B = Discriminator(self.opt)  # todo: args
        self.criterion = None       # MSE? BCE? + cycle consistency loss, todo: can we make class of loss?

    def train(self):
        pass

    def test(self):
        pass

    def forward(self, input):
        # call forward methods in both generators and discriminators
        pass

    def backward(self):
        pass

if __name__ == "__main__":
    print("This program is still in the making.")
    print("Instantiating CycleGAN clone... ")
    cycle_gan = CycleGAN()
    print(cycle_gan.opt)
    print(cycle_gan.G_A)
    print(cycle_gan.G_B)
    