# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import warnings

warnings.filterwarnings("ignore")


class AdaIN(nn.Module):
    '''
    adaptive instance normalization
    '''
    def __init__(self, n_in, n_channel):
        super(AdaIN, self).__init__()
        self.linear = nn.Linear(n_in, n_channel * 2)
        self.norm = nn.InstanceNorm2d(n_channel)
        
    def forward(self, image, style):
        style = self.linear(style)
        factor, bias = style.chunk(2, 1)
        
        result = self.norm(image)
        result = result * factor.unsqueeze(-1).unsqueeze(-1) + bias.unsqueeze(-1).unsqueeze(-1)
        return result

class ConvBlock(nn.Module):
    def __init__(self, n_in, n_middle, n_out):
        super(ConvBlock, self).__init__()
        self.n_in = n_in
        self.n_out = n_out

        self.conv1 = nn.Conv2d(n_in, n_middle, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(n_middle)
        self.act1 = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(n_middle, n_out, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(n_out)
        
        if n_in != n_out:
            self.identity = nn.Conv2d(n_in, n_out, kernel_size=1, stride=1, padding=0)
        else:
            self.identity = nn.Identity()

    def forward(self, x):
        identity = self.identity(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)

        return x + identity


class Encoder(nn.Module):
    def __init__(self, n_in, nf):
        super(Encoder, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(n_in, nf, kernel_size=7, stride=1, padding=3),
            nn.SiLU(inplace=True),
            ConvBlock(nf, nf, nf),    # [128, 128]
            nn.Conv2d(nf, nf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(nf * 2),
            nn.SiLU(inplace=True),    # [64, 64]
            ConvBlock(nf * 2, nf * 2, nf * 2),  # [64, 64]
            nn.Conv2d(nf * 2, nf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(nf * 4),
            nn.SiLU(inplace=True),  # [32, 32]
            ConvBlock(nf * 4, nf * 4, nf * 4),  # [32, 32]
            nn.Conv2d(nf * 4, nf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(nf * 8),
            nn.SiLU(inplace=True),  # [16, 16]
            ConvBlock(nf * 8, nf * 8, nf * 8),  # [16, 16]
            ConvBlock(nf * 8, nf * 8, nf * 8),  # [16, 16]
            ConvBlock(nf * 8, nf * 8, nf * 8),  # [16, 16]
            # nn.Conv2d(nf * 8, nf * 8, kernel_size=4, stride=2, paddding=1),
            # nn.LeakyReLU(0.2, inplace=True),  # [8, 8]
        )

    def forward(self, x):
        return self.main(x)


class Decoder(nn.Module):
    def __init__(self, num_embedding, n_out, nf, emb_dim=1024):
        super(Decoder, self).__init__()
        self.nf = nf

        self.embedding = nn.Embedding(num_embedding, emb_dim)
        self.emb_to_linear = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(inplace=True),
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(inplace=True),
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(inplace=True),
            nn.Linear(emb_dim, emb_dim),
        )

        self.blocks = nn.ModuleList([
            AdaIN(emb_dim, nf * 8),
            ConvBlock(nf * 8, nf * 8, nf * 8),  # [16, 16]
            AdaIN(emb_dim, nf * 8),
            ConvBlock(nf * 8, nf * 8, nf * 8),  # [16, 16]
            AdaIN(emb_dim, nf * 8),
            ConvBlock(nf * 8, nf * 4, nf * 4),     # [16, 16]
            nn.Upsample(scale_factor=2),           # [32, 32]
            AdaIN(emb_dim, nf * 4),
            ConvBlock(nf * 4, nf * 2, nf * 2),     # [32, 32]
            nn.Upsample(scale_factor=2),           # [64, 64]
            AdaIN(emb_dim, nf * 2),
            ConvBlock(nf * 2, nf, nf),             # [64, 64]
            nn.Upsample(scale_factor=2),           # [128, 128]
            nn.Conv2d(nf, n_out, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        ])
        for param in self.embedding.parameters():
            nn.init.trunc_normal_(param, 0, 0.02)

    def forward(self, x, id):
        embedding = self.embedding(id)
        embedding = self.emb_to_linear(embedding)
        
        for block in self.blocks:
            if isinstance(block, AdaIN):
                x = block(x, embedding)
            else:
                x = block(x)
        
        return x
        

class Discriminator(nn.Module):
    def __init__(self, n_in, nf):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(n_in, nf, kernel_size=7, stride=1, padding=3),
            nn.SiLU(inplace=True),
            ConvBlock(nf, nf, nf),    # [128, 128]
            nn.Conv2d(nf, nf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(nf * 2),
            nn.SiLU(inplace=True),    # [64, 64]
            ConvBlock(nf * 2, nf * 2, nf * 2),  # [64, 64]
            nn.Conv2d(nf * 2, nf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(nf * 4),
            nn.SiLU(inplace=True),  # [32, 32]
            ConvBlock(nf * 4, nf * 4, nf * 4),  # [32, 32]
            nn.Conv2d(nf * 4, nf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(nf * 8),
            nn.SiLU(inplace=True),  # [16, 16]
            ConvBlock(nf * 8, nf * 8, nf * 8),  # [16, 16]
            nn.Conv2d(nf * 8, nf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(nf * 8),
            nn.SiLU(inplace=True),  # [8, 8]
            ConvBlock(nf * 8, nf * 8, nf * 8),  # [8, 8]
            nn.Conv2d(nf * 8, nf * 8, kernel_size=4, stride=2, padding=1), # [4, 4]
            nn.BatchNorm2d(nf * 8),
            nn.SiLU(inplace=True),
            nn.Flatten(),
            nn.Linear(nf * 8 * 4 * 4, 1),
        )

    def forward(self, x):
        return self.main(x)


if __name__ == '__main__':
    nf = 32
    encoder = Encoder(1, nf)
    decoder = Decoder(1, nf)

    x = torch.randn(1, 1, 128, 128)
    id = torch.LongTensor([0])
    print(id.size())
    encoded = encoder(x)
    print(encoded.size())
    decoded = decoder(encoded, id)
    print(decoded.size())