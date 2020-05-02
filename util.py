# Utilities file

import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
from torchvision.utils import make_grid
import PIL
import torchvision.transforms as transforms

def deal_with_argv(args):
    if len(args) == 1 or args[1] == "train":
        is_train = True
    elif args[1] == "test":
        is_train = False
    else:
        print("The first command line argument should be either 'train' or 'test'")
        exit()
    return is_train

def make_generators_dir():
    # If the generators directory does not exist yet, make it
    generators_dir = f"{os.getcwd()}/generators"
    if not os.path.exists(generators_dir):
        os.mkdir(generators_dir)
        print(f"Made {generators_dir}.")

def make_animation(img_list):
    # Make animation of grid images
    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    plt.show()

def create_result_image(model_path, img_path, img_transform, return_tensor=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if os.path.exists(model_path):
        model = torch.load(model_path).to(device)
    else:
        print(f"File {model_path} not found or unable to open")
        exit()

    try:
        original_img = PIL.Image.open(img_path)
    except:
        print(f"File {original_img} not found or unable to open")
        exit(0)

    original_img = img_transform(original_img).to(device)
    original_img = torch.reshape(original_img, (1, 3, 192, 192))

    new_img = model(original_img).detach().cpu()
    if return_tensor:
        return new_img
        
    results = make_grid(new_img, padding=2, normalize=True)
    new_img = transforms.ToPILImage()(results).convert("RGB")
    return new_img