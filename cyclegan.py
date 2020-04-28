## CycleGAN remake, we follow the structure of the PyTorch implementation
## given by the authors of the CycleGAN paper.
## PyTorch implementation: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/
## CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf

# Todo-list:
# 1) Figure out how to combine data from both dataloaders. Have a look at the CustomDataLoader of the authors
# 2) Compute loss with cycle consistency term + normal GAN loss, see paper!
# 3) Figure out how to test.

# Program:
# Load in dataset
# Instantiate CycleGAN object, which has G_A, G_B, D_A, and D_B networks
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
import os
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
        self.layers = [nn.ReflectionPad2d(3),
                       nn.Conv2d(opt.input_nc, ngf, kernel_size=7, padding=0, bias=False),
                       nn.BatchNorm2d(ngf),
                       nn.ReLU(True)]

        # Add downsampling layers (see CycleGAN implementation by the authors)
        n_downsampling_layers = 2
        for i in range(n_downsampling_layers):
            mult = 2 ** i
            self.layers += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=False),
                            nn.BatchNorm2d(ngf * mult * 2),
                            nn.ReLU(True)]

        # Add ResNet blocks
        mult = 2 ** n_downsampling_layers
        for i in range(self.n_blocks):
            self.layers += [ResnetBlock(ngf, mult)] # see https://arxiv.org/pdf/1512.03385.pdf

        # Add upsampling layers
        for i in range(n_downsampling_layers):
            mult = 2 ** (n_downsampling_layers - i)
            self.layers += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                              kernel_size=3, stride=2,
                                              padding=1, output_padding=1,
                                              bias=False),
                           nn.BatchNorm2d(int(ngf * mult / 2)),
                           nn.ReLU(True)]

        self.layers += [nn.ReflectionPad2d(3),   # The value is 3 because we use a 7x7 conv layer
                       nn.Conv2d(ngf, opt.output_nc, kernel_size=7, padding=0), # todo: understand why this conv2d is here and not convtranspose2d
                       nn.Tanh()]

        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)


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

    def set_requires_grad(self, flag):
        for parameter in self.parameters():
            parameter.requires_grad = flag


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
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = "cpu"
        print("Keep in mind! Options.device is now cpu by default, as cuda results in OOM error.")

        self.input_nc = 3       # num channels, usually 3 (RGB)
        self.output_nc = 3      # num channels, usually 3 (RGB)
        self.num_epochs = 5
        self.lr = 0.0002        # Learning rate
        self.beta1 = 0.5        # beta1 parameter for the Adam optimizers

        # lambda parameter (how much more important 
        #is the cycle-consistency loss compared to the normal GAN loss)
        self.lambda_ = 10       


        self.workers = 2        # Number of workers for dataloader
        self.batch_size = 1    # Batch size during training
        self.image_size = 128   # Spatial size of training images.


class CycleGAN:
    def __init__(self, opt, is_train):
        self.opt = opt
        self.device = opt.device
        self.G_A = Generator(self.opt).to(self.device)
        self.G_B = Generator(self.opt).to(self.device)

        self.real_A = None # Input Data from domain A
        self.real_B = None # Input Data from domain B
        self.fake_A = None # Generated Data domain B -> A
        self.fake_B = None # Generated Data domain A -> B
        self.reconstructed_A = None # Generated Data from A -> B -> A
        self.reconstructed_B = None # Generated Data from B -> A -> B

        if is_train:
            # Only need discriminators at training time
            self.D_A = Discriminator(self.opt).to(self.device)
            self.D_B = Discriminator(self.opt).to(self.device)
            # Only need losses during training time
            self.criterionLSGAN = LSGANLoss().to(self.device) # TODO: see why .to(self.device) does not work)
            self.criterionCycle = nn.L1Loss()   # Cycle-consistency loss, see paper!
            # TODO?: might want to test out identity loss as well.

            self.optimizer_G = optim.Adam(itertools.chain(self.G_A.parameters(), self.G_B.parameters()),
                                          lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            self.optimizer_D = optim.Adam(itertools.chain(self.D_A.parameters(), self.D_B.parameters()),
                                          lr=self.opt.lr, betas=(self.opt.beta1, 0.999))


    def train(self):
        """Following the implementation of the authors of CycleGAN,
        the discriminators don't need gradients while training G_A and G_B.
        """
        self.forward()

        # Train the Generators

        self.D_A.set_requires_grad(False)
        self.D_B.set_requires_grad(False)
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero        
        lambda_ = self.opt.lambda_
        LSGANloss_A = self.criterionLSGAN(self.D_A(self.fake_B), True)           # LSGAN loss for GAN A
        LSGANloss_B = self.criterionLSGAN(self.D_B(self.fake_A), True)           # LSGAN loss for GAN B
        cycleloss_A = self.criterionCycle(self.reconstructed_A, self.real_A) * lambda_  # Forward cycle loss || G_B(G_A(A)) - A||
        cycleloss_B = self.criterionCycle(self.reconstructed_B, self.real_B) * lambda_  # Backward cycle loss || G_A(G_B(B)) - B||
        loss_G = LSGANloss_A + LSGANloss_B + cycleloss_A + cycleloss_B     # combined loss and calculate gradients
        loss_G.backward()
        self.optimizer_G.step()       # update G_A and G_B's weights

        # Train the Discriminators
        self.D_A.set_requires_grad(True)
        self.D_B.set_requires_grad(True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero

        # D_A Loss and gradient calculations
        D_A_pred_real = self.D_A(self.real_B)
        loss_D_A_real = self.criterionLSGAN(D_A_pred_real, True)
        D_A_pred_fake = self.D_A(self.fake_B.detach())
        loss_D_A_fake = self.criterionLSGAN(D_A_pred_fake, False)
        loss_D_A = (loss_D_A_real + loss_D_A_fake) * 0.5
        loss_D_A.backward()
        # D_B Loss and gradient calculations
        D_B_pred_real = self.D_B(self.real_A)
        loss_D_B_real = self.criterionLSGAN(D_B_pred_real, True)
        D_B_pred_fake = self.D_B(self.fake_A.detach())
        loss_D_B_fake = self.criterionLSGAN(D_B_pred_fake, False)
        loss_D_B = (loss_D_B_real + loss_D_B_fake) * 0.5
        loss_D_B.backward() # TODO: find out why you can use backward twice here. Isn't the gradient wrong?

        self.optimizer_D.step()  # update D_A and D_B's weights
        

    def test(self):
        # We do not need to calculate gradients during test time, as we are not updating the weights
        with torch.no_grad():
            self.forward()


    def forward(self):
        # call forward methods in both generators and discriminators
        self.fake_B = self.G_A(self.real_A)            # G_A(A)
        self.reconstructed_A = self.G_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.G_B(self.real_B)            # G_B(B)
        self.reconstructed_B = self.G_A(self.fake_A)   # G_A(G_B(B))


    def backward(self):
        pass

if __name__ == "__main__":
    print("This program is still in the making.")
    print("Instantiating CycleGAN clone... ")
    opt = Options()        # Hardcoded options
    cycle_gan = CycleGAN(opt, is_train=True)

    # Image transforms
    transform = transforms.Compose([transforms.Resize(opt.image_size),
                                    transforms.CenterCrop(opt.image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ])

    dataroot_A = f"{os.getcwd()}/data_per_painter/Pablo_Picasso"
    dataroot_B = f"{os.getcwd()}/data_per_painter/Vincent_van_Gogh"

    if os.path.exists(dataroot_A) and os.path.exists(dataroot_B):
        # Create the datasets
        dataset_A = dset.ImageFolder(root=dataroot_A, transform=transform)
        dataset_B = dset.ImageFolder(root=dataroot_B, transform=transform)

        # Create the dataloaders
        dataloader_A = torch.utils.data.DataLoader(dataset_A, batch_size=opt.batch_size,
                                                   shuffle=True, num_workers=opt.workers)
        dataloader_B = torch.utils.data.DataLoader(dataset_B, batch_size=opt.batch_size,
                                                   shuffle=True, num_workers=opt.workers)
    else:
        error_cause = dataroot_A if not os.path.exists(dataroot_A) else dataroot_B
        print("ERROR: " + error_cause + " does not exist or you do not have permission to open this file.")
        exit()

    fixed_images = dataset_A.__getitem__(0)[0].to(opt.device)
    fixed_images = torch.reshape(fixed_images, (1, 3, 128, 128))
    img_list = []   # We'll use this to visualize the progress of the GAN
    for epoch in range(opt.num_epochs):
        print("Epoch {} of {}".format(epoch,opt.num_epochs))
        # Get data from both dataloaders and give it to the cycleGAN
        for i, data in enumerate(zip(dataloader_A, dataloader_B)):
            data_A, data_B = data
            cycle_gan.real_A = data_A[0].to(opt.device)
            cycle_gan.real_B = data_B[0].to(opt.device)
            cycle_gan.train()
            
        # Visualize the progress of the CycleGAN by saving G_A's output on images from dataset_A
        with torch.no_grad():
            fake = cycle_gan.G_A(fixed_images).detach().cpu()
        grid_of_fakes = vutils.make_grid(fake, padding=2, normalize=True)
        img_list.append(grid_of_fakes)


    # Save the last grid image made to a png file
    fake_im = transforms.ToPILImage()(grid_of_fakes).convert("RGB")
    fake_im.save("test_results/latest_test_result.png", "PNG")

    # Make animation of grid images
    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=500, repeat_delay=500, blit=True)
    plt.show()
    print("Program finished without errors!")