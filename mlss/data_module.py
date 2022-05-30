import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, transforms



transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(572),
    transforms.CenterCrop(572)
    ])


class CelebADataModule(pl.LightingDataModule):
    def __init__(self,batch_size=32):
        self.batch_size=batch_size

    def prepare_data(self):


        self.training_data = datasets.CelebA(
        root =r'.\mlss\CelebA',
        split='train',  
        transform=transformations,
        download=True
        )

        self.validation_data = datasets.CelebA(
            root =r'.\mlss\CelebA',
            split='valid',  
            transform=transformations,
            download=True
        )

        self.test_data = datasets.CelebA(
            root=r'.\mlss\CelebA',
            split='test',
            transform=transformations,
            download=True
        )


    def train_dataloader(self):

        train_loader = DataLoader(
        self.training_data,
        batch_size=self.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True)

        return train_loader


    
    def val_dataloader(self):

        val_loader = DataLoader(
        self.validation_data,
        batch_size=self.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True)

        return val_loader


    def test_dataloader(self):

        test_loader = DataLoader(
        self.test_data,
        batch_size=self.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=True)

        return test_loader



