import argparse

from pytorch_lightning import Trainer
import torch
from mlss.data_module import GTSRBDataModule
from mlss.unet_module import UNETModule


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

    batch_size=8
    
    
    data=GTSRBDataModule(batch_size)

    model=UNETModule(in_channels=3,num_classes=32, lr=0.001, early_stopping_patience=100, lr_scheduler_patience=10)
    

    trainer = Trainer(gpus=1)

    trainer.fit(model, datamodule=data)
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Parse input arguments.
    args = parser.parse_args()
    # Call the main mehtod.
    main(args)

