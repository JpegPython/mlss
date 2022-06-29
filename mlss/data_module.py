

from numpy import dtype
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchvision import datasets
from torchvision.transforms import ToTensor, transforms,InterpolationMode



class ImageDataModule(pl.LightningDataModule):
    def __init__(self,batch_size):
        self.batch_size=batch_size
        self.prepare_data_per_node = True
        
        self._log_hyperparams = False

    def prepare_data(self):
        dataset=datasets.Cityscapes(root=r'.\mlss\data\cityscapes' , split="train" , mode="fine",target_type='semantic')
        datasets.Cityscapes(root=r'.\mlss\data\cityscapes',split='val',mode='fine',target_type='semantic')
        datasets.Cityscapes(root=r'.\mlss\data\cityscapes',split='test',mode='fine',target_type='semantic')
        

            


    

    def setup(self,stage=None):

        transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(572),
        transforms.CenterCrop(572)
        ])

        target_transformations= transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(560,interpolation=InterpolationMode.NEAREST),
            transforms.CenterCrop(560)
            
        ])

        self.Cityscapes_train=datasets.Cityscapes(
            root=r'.\mlss\data\cityscapes',
            split='train',
            mode='fine',
            target_type='semantic', 
            transform=transformations,
            target_transform=target_transformations
            
            
        )
       
        dataset=self.Cityscapes_val=datasets.Cityscapes(
            root=r'.\mlss\data\cityscapes',
            split='val',
            mode='fine',
            target_type='semantic', 
            transform=transformations,
            target_transform=target_transformations)
        

        
        self.Cityscapes_test=datasets.Cityscapes(
            root=r'.\mlss\data\cityscapes',
            split='test',
            mode='fine',
            target_type='semantic', 
            transform=transformations,
            target_transform=target_transformations       
        )

        




    def train_dataloader(self):

        Cityscapes_train = DataLoader(
        self.Cityscapes_train,
        batch_size=self.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True)

        return Cityscapes_train



    
    def val_dataloader(self):

        Cityscapes_val = DataLoader(
        self.Cityscapes_val,
        batch_size=self.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
        drop_last=True)

        return Cityscapes_val


    def test_dataloader(self):

        Cityscapes_test = DataLoader(
        self.Cityscapes_test,
        batch_size=self.batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=True)

        return Cityscapes_test







