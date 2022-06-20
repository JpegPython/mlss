
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchvision import datasets
from torchvision.transforms import ToTensor, transforms



class GTSRBDataModule(pl.LightningDataModule):
    def __init__(self,batch_size, device='cuda'):
        self.batch_size=batch_size
        self.prepare_data_per_node = True
        
        self._log_hyperparams = False

    def prepare_data(self):
        datasets.GTSRB(root=r'.\mlss\GTSRB',split='train',download=True)
        datasets.GTSRB(root=r'.\mlss\GTSRB',split='test',download=True)

    def setup(self,stage=None):

        transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(572),
        transforms.CenterCrop(572)
        ])

        GTSRB_train=datasets.GTSRB(
            root =r'.\mlss\GTSRB',
            split='train', 
            transform=transformations
        )
        
        self.GTSRB_test=datasets.GTSRB(
            root =r'.\mlss\GTSRB',
            split='test',
            transform=transformations
        )

        train_data_size=int(len(GTSRB_train) * 0.8)
        valid_data_size=len(GTSRB_train)-train_data_size

        self.GTSRB_train,self.GTSRB_val=random_split(GTSRB_train,[train_data_size,valid_data_size])
        self.GTSRB_test=datasets.GTSRB(root='.\mlss\GTSRB',split='test')




    def train_dataloader(self):

        GTSRB_train = DataLoader(
        self.GTSRB_train,
        batch_size=self.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=False,
        drop_last=True)

        return GTSRB_train


    
    def val_dataloader(self):

        GTSRB_val = DataLoader(
        self.GTSRB_val,
        batch_size=self.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
        drop_last=True)

        return GTSRB_val


    def test_dataloader(self):

        GTSRB_test = DataLoader(
        self.GTSRB_test,
        batch_size=self.batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=True)

        return GTSRB_test



