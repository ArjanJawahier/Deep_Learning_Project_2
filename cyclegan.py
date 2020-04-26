## CycleGAN remake, we follow the structure of the PyTorch implementation
## given by the authors of the CycleGAN paper.
## PyTorch implementation: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/
## CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf

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
import itertools

class Generator(nn.Module):
    """ResNet, but hardcode the layers (in contrast to the real CycleGAN).
    We don't use dropout, since the authors of CycleGAN did not use this either.
    """

    def __init__(self, opt):
        super(Generator, self).__init__()
        self.n_blocks = 6
        ngf = 64           # Hardcoded, num filters in last conv layer

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
            self.model += [ResnetBlock(ngf, mult)] # see https://arxiv.org/pdf/1512.03385.pdf

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

    def forward(self, x):
        self.model(x)


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

    def forward(self, x):
        """ResNet forward function: https://arxiv.org/pdf/1512.03385.pdf"""
        out = x + self.conv_block(x)
        return out


class Discriminator(nn.Module):
    """Implementing the PatchGAN used by the CycleGAN authors.""" 
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        ndf = 64
        num_layers = 3
        kw = 4
        padw = 1
        self.model = [nn.Conv2d(opt.input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                      nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for i in range(1, num_layers):
        	nf_mult_prev = nf_mult
        	nf_mult = min(2 ** i, 8)
        	self.model += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 
        							 kernel_size=kw, stride=2, padding=padw, bias=False),
        				   nn.BatchNorm2d(ndf * nf_mult),
        				   nn.LeakyReLU(0.2, True)]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** num_layers, 8)
        self.model += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
        						 kernel_size=kw, stride=1, padding=padw, bias=False),
        			   nn.BatchNorm2d(ndf * nf_mult),
        			   nn.LeakyReLU(0.2, True),
        			   nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)] # 1 channel output

        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class LSGANLoss(nn.Module):
	def __init__(self):
		super(LSGANLoss, self).__init__()
		self.loss = nn.MSELoss() # see LSGAN paper as to why
		target_real_label = 1.0
		target_fake_label = 0.0
		self.register_buffer("real_label", torch.tensor(target_real_label))
		self.register_buffer("fake_label", torch.tensor(target_fake_label))

	def __call__(self, prediction, target_is_real):
		if target_is_real:
			target_tensor = self.real_label.expand_as(prediction)
		else:
			target_tensor = self.fake_label.expand_as(prediction)

		return self.loss(prediction, target_tensor)


class Options:
    """A class that keeps track of all user-defined options"""
    def __init__(self):
        self.input_nc = 3       # num channels, usually 3 (RGB)
        self.output_nc = 3      # num channels, usually 3 (RGB)
        self.num_epochs = 5
        self.lr = 0.0002		# Learning rate
        self.beta1 = 0.5 		# beta1 parameter for the Adam optimizers

class CycleGAN:
    def __init__(self, is_train):
        self.opt = Options()        # Hardcoded options
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.G_A = Generator(self.opt)
        self.G_B = Generator(self.opt)

        if is_train:
        	# Only need discriminators at training time
	        self.D_A = Discriminator(self.opt)
	        self.D_B = Discriminator(self.opt)
	        # Only need losses as training time
        	self.criterionLSGAN = LSGANLoss().to(self.device)
        	self.criterionCycle = nn.L1Loss()	# Cycle-consistency loss, see paper!
        	# TODO: might want to test out identity loss as well.

        	self.optimizer_G = optim.Adam(itertools.chain(self.G_A.parameters(), self.G_B.parameters()),
        								  lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        	self.optimizer_D = optim.Adam(itertools.chain(self.D_A.parameters(), self.D_B.parameters()),
        								  lr=self.opt.lr, betas=(self.opt.beta1, 0.999))

    def train(self):
        pass

    def test(self):
        pass

    def forward(self, x):
        # call forward methods in both generators and discriminators
        pass

    def backward(self):
        pass

if __name__ == "__main__":
    print("This program is still in the making.")
    print("Instantiating CycleGAN clone... ")
    cycle_gan = CycleGAN(is_train=True)
    print(cycle_gan.opt)
    print(cycle_gan.D_A)
    print(cycle_gan.D_B)