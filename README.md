# Deep Learning Project 2 - Style transfer using GANs

## Framework
We are using PyTorch as our Deep Learning framework.
This is thus a dependency.
To install PyTorch, you can use the package manager pip:

```bash
pip install torch===1.4.0 torchvision===0.5.0 -f https://download.pytorch.org/whl/torch_stable.html
```

## Other dependencies currently used
We are also using matplotlib and PIL for visualization purposes, which can be installed using the commands:
```bash
pip install matplotlib
pip install Pillow
```

## Dataset
We are currently using the dataset ["Best artworks of all time"](https://www.kaggle.com/ikarus777/best-artworks-of-all-time)
The dataset has to be downloaded manually. We are not uploading it to GitHub.

## Pre-knowledge
The following tutorials and documents can be read to understand what to do.
1) [PyTorch GAN Tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html).
2) [A lot of documentation and examples of GANs (highly recommended)](https://github.com/nashory/gans-awesome-applications)
3) [Style Transfer paper](https://arxiv.org/pdf/1703.07511.pdf)
4) [CycleGAN paper](https://arxiv.org/pdf/1703.10593.pdf) - We are implementing this.
5) [ResNet paper](https://arxiv.org/pdf/1512.03385.pdf) - This is what our Generators will consist of.
6) [PatchGAN paper](https://arxiv.org/pdf/1611.07004.pdf) - This is what our Disciminators will consist of.
7) [LSGAN paper](https://arxiv.org/pdf/1611.04076.pdf) - Least Squares Loss function for the discriminator.

## How to run
To run the program we'll first need to download the dataset. 
1. Go to https://www.kaggle.com/ikarus777/best-artworks-of-all-time and download the dataset, it is possible that you have to log in to kaggle first.
2. Rename the main folder of the downloaded dataset to "data", make sure that the folders inside this folder are "/images", "/resized", and the file "artists.csv" there should not be any folders between the main folder, "data", and the mentioned folders/files.
3. Copy the folder "data" to the folder where this algorithm is located.
4. Now we will move the data into the correctly usable folder for this run the dataset_dirmanager.py file: `python dataset_dirmanager.py`.
5. We're ready to run the algortihm: `python cyclegan.py`, If you want to change the default settings of this algortihm you will have to open an code editor and edit it in the class Options. The default settings are:

| Variable          | Value  | comment                                 |
|-------------------|--------|-----------------------------------------|
| input_nc          | 3      | number channels, usually 3 (RGB)        |
| output_nc         | 3      | number channels, usually 3 (RGB)        |
| num_epochs        | 25     | number of epochs                        |
| lr                | 0.0002 | learning rate                           |
| beta1             | 0.5    | beta1 parameter for the Adam optimizers |
| lambdas           | [10]   | list of Lambdas tested                  |
| lambda            | 10     | current lambda                          |
| workers           | 2      | Number of workers for dataloader        |
| batch_size        | 1      | Batch size during training              |
| image_size        | 192    | Spatial size of training images.        |
| use_identity_loss | true   |                                         |

6. Wait until the algorithm is finished.
7. Run the algorithm results.py to create an image of the results: `python results.py`.
8. The results can be found in the folder `generators`.

<!-- ## Discriminator
The discriminator coded in discriminator.py is trying to classify input images as either real or fake. When we want to transfer a style (e.g. Van Gogh) to another image (e.g. your house), the discriminator will compare the generated image of your house with the Van Gogh style to real Van Gogh images. The goal is for the discriminator to essentially randomly guess whether an image is real or fake (p = 0.5).

We will be using PatchGANs for our discriminator models. These PatchGANs can determine whether each patch of 70x70 pixels is fake or real in a convolutional fashion. This causes the PatchGAN to be applicable to images of any size, while also having fewer parameters than e.g. a 128x128 pixel GAN disciminator. 

The discriminator loss function will be MSELoss. We do this as this is recommended by the LSGAN paper. Essentially, using the least squares method instead of other loss functions will force the GANs to generate images that lie close to the decision boundary.

## Generator
The generator coded in generator.py is trying to create fake images that look like the given style. We can give the generator an input image and an input style, such that the output image has positive semantic correlation with the given input image and the given input style. The goal is for the generator to generate images such that these cannot be discriminated from real images by the discriminator.

The generator will consist of a few downsampling layers and upsampling layers, with a number of ResNet blocks in between. ResNet blocks allow deep architectures to become even deeper, as ResNet blocks combat the vanishing/exploding gradient problem.  -->