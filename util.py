# Utilities file

import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
