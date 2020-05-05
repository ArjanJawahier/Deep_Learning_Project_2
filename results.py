# Results-image maker

import os
from os import listdir
from os.path import isfile, join
import PIL
import torch
from torchvision.utils import make_grid
import torchvision.transforms as transforms
import util
from cyclegan import Generator
from cyclegan import ResnetBlock

def generate_result_image(subfolder):
    # The image we want to put in the results
    im_filepath = "Pablo_Picasso_38.jpg"

    generators_dir = os.getcwd() + "/generators/" + subfolder
    generators_list = os.listdir(generators_dir)

    image_size = 192
    im_transform = transforms.Compose([transforms.Resize(image_size),
                                       transforms.CenterCrop(image_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                      ])

    generators_list = [f for f in listdir(generators_dir) if isfile(join(generators_dir, f))]
    print(generators_list)
    # generators_list = ["G_A_5e_0lambda.pt",
    #                   "G_A_10e_0lambda.pt",
    #                   "G_A_15e_0lambda.pt",
    #                   "G_A_20e_0lambda.pt",
    #                   "G_A_25e_0lambda.pt",
    #                   "G_A_5e_1lambda.pt",
    #                   "G_A_10e_1lambda.pt",
    #                   "G_A_15e_1lambda.pt",
    #                   "G_A_20e_1lambda.pt",
    #                   "G_A_25e_1lambda.pt",
    #                   "G_A_5e_10lambda.pt",
    #                   "G_A_10e_10lambda.pt",
    #                   "G_A_15e_10lambda.pt",
    #                   "G_A_20e_10lambda.pt",
    #                   "G_A_25e_10lambda.pt",
    #                   "G_A_5e_100lambda.pt",
    #                   "G_A_10e_100lambda.pt",
    #                   "G_A_15e_100lambda.pt",
    #                   "G_A_20e_100lambda.pt",
    #                   "G_A_25e_100lambda.pt"]

    img_list = []
    for i, g in enumerate(generators_list):
        g_filepath = generators_dir + "/" + g
        im = util.create_result_image(g_filepath, im_filepath, im_transform, return_tensor=True)
        im = im.squeeze()
        img_list.append(im)

    results_grid = make_grid(img_list, nrow=5, padding=2, normalize=True)
    results = transforms.ToPILImage()(results_grid).convert("RGB")
    results.save(f"test_results/{subfolder}.jpg", "JPEG") 
    results.show()

if __name__ == "__main__":
    amount_of_runs_before = len(next(os.walk('generators')))
    generate_result_image(str(amount_of_runs_before))