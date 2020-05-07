# Results-image maker

import os
from os import listdir
from os.path import isfile, join
from pathlib import Path
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

    generators_list = sorted(Path(generators_dir).iterdir(), key=os.path.getmtime)

    img_list = []
    for g_filepath in generators_list:
        im = util.create_result_image(g_filepath, im_filepath, im_transform, return_tensor=True)
        im = im.squeeze()
        img_list.append(im)

    results_grid = make_grid(img_list, nrow=5, padding=2, normalize=True)
    results = transforms.ToPILImage()(results_grid).convert("RGB")
    results.save(f"test_results/{subfolder}.jpg", "JPEG") 
    results.show()

if __name__ == "__main__":
    amount_of_runs_before = len(next(os.walk('generators')))
    generate_result_image("2")