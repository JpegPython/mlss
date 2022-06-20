from .typing import ImageBatch, Logits, SemanticBatch
from typing import Tuple
import torch
import torch.nn as nn
import torchvision.transforms
import torch.nn.functional as F

class Block(nn.Module):
    # double convolution and ReLU
    def __init__(self, in_chl, out_chl):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chl, out_chl, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_chl, out_chl, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
    def forward(self, x):

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
       
        return out


class Encoder(nn.Module):
    def __init__(self, enc_chs: Tuple[int, ...] = (3, 64, 128, 256, 512, 1024)):
        super().__init__()
        
        block_list=[
            Block(enc_chs[i], enc_chs[i + 1]) \
            for i in range(len(enc_chs) - 1)
        ]

        self.enc_blocks = nn.ModuleList(block_list)
        
        self.pool = nn.MaxPool2d(2,2)

    
    
    def forward(self, x):

        features = []

        for block in self.enc_blocks:
            x = block(x)
            features.append(x)
            x = self.pool(x)

        return features


class Decoder(nn.Module):
    def __init__(self, dec_chs: Tuple[int, ...] = (1024, 512, 256, 128, 64)):

        super().__init__()
        self.dec_chs = dec_chs


        up_list = [
            nn.ConvTranspose2d(dec_chs[i], dec_chs[i + 1], 2, 2) \
            for i in range(len(dec_chs) - 1)
        ]
        self.upconvs = nn.ModuleList(up_list)


        block_list =[
            Block(dec_chs[i], dec_chs[i + 1]) \
            for i in range(len(dec_chs) - 1)
        ]
        self.blocks_decoder = nn.ModuleList(block_list)

    def forward(self, x, encoder_features):
        for i in range(len(self.dec_chs) - 1):
            x = self.upconvs[i](x)
            enc_features = self.crop(encoder_features[i], x)
            x = torch.cat([x, enc_features], dim=1)
            x = self.blocks_decoder[i](x)
        return x

    def crop(self, features, x):
        _, _, H, W = x.shape
        features = torchvision.transforms.CenterCrop([H, W])(features)
        return features





class UNET(nn.Module):
    def __init__(self,num_classes, enc_chs=(3, 64, 128, 256, 512, 1024), dec_chs=(1024, 512, 256, 128, 64)):

        super().__init__()

        self.encoder = Encoder(enc_chs)
        self.decoder = Decoder(dec_chs)

        self.finalConv = nn.Conv2d(dec_chs[-1], num_classes, kernel_size=1)
        

    def forward(self, x: ImageBatch) -> Logits:
        enc_features = self.encoder(x)
        out = self.decoder(enc_features[::-1][0], enc_features[::-1][1:])
        out = self.finalConv(out)
        return out



