from .typing import ImageBatch, Logits
from typing import Tuple
import torch
import torch.nn as nn
import torchvision.transforms


class Block(nn.Module):
    # double convolution and ReLU
    def __init__(self, in_chl, out_chl):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chl, out_chl, 3)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_chl, out_chl, 3)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x


class Encoder(nn.Module):
    def __init__(self, channels: Tuple[int, ...] = (3, 64, 128, 256, 512, 1024)):
        super().__init__()

        self.blocks_encoder = nn.ModuleList([Block(channels[i], channels[i + 1]) for i in range(len(channels) - 1)])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        features = []
        for block in self.blocks_encoder:
            x = block(x)
            features.append(x)
            x = self.pool(x)
        return features


class Decoder(nn.Module):
    def __init__(self, channels: Tuple[int, ...] = (1024, 512, 256, 128, 64)):
        super().__init__()
        self.channels = channels

        self.decoder_convs = nn.ModuleList([nn.ConvTranspose2d(channels[i], channels[i + 1], 2, 2) for i in range(len(channels) - 1)])
        self.blocks_decoder = nn.ModuleList([Block(channels[i], channels[i + 1]) for i in range(len(channels) - 1)])

    def crop(self, enc_features, x):
        _, _, H, W = x.shape
        enc_features = torchvision.transforms.CenterCrop([H, W])(enc_features)
        return enc_features

    def forward(self, x, encoder_features):
        for i in range(len(self.channels) - 1):
            x = self.decoder_convs[i](x)
            enc_features = self.crop(encoder_features[i], x)
            x = torch.cat([x, enc_features], dim=1)
            x = self.blocks_decoder[i](x)
        return x




class UNET(nn.Module):
    def __init__(self, enc_channels=(3, 64, 128, 256, 512, 1024), dec_channels=(1024, 512, 256, 128, 64),
                 retain_dim=False):
        super().__init__()
        self.encoder = Encoder(enc_channels)
        self.decoder = Decoder(dec_channels)

        self.finalConv = nn.Conv2d(dec_channels[-1], 2,1)
        self.retain_dim = retain_dim

    def forward(self, x: ImageBatch) -> Logits:
        enc_features = self.encoder(x)
        out = self.decoder(enc_features[::-1][0], enc_features[::-1][1:])
        out = self.finalConv(out)
        return out



