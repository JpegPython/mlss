import argparse

from pytorch_lightning import Trainer
from mlss.data_module import CityScapesDataModule
from mlss.unet_module import UNETModule




import numpy as np
import torchvision


import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam



# Function to test the model with a batch of images
#def testBatch():
    # get batch of images from the test DataLoader
    #images, labels = next(iter(test_loader))

    # show all images as one image grid
    #imageshow(torchvision.utils.make_grid(images))



def main(args: argparse.Namespace) -> None:

    

    data=CityScapesDataModule()

    model=UNETModule(in_channels=2,num_classes=3, lr=0.001)

    trainer = Trainer(gpus=1)

    trainer.fit(model, data)
    trainer.test(model, data)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Parse input arguments.
    args = parser.parse_args()
    # Call the main mehtod.
    main(args)
