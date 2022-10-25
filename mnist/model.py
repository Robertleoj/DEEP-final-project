import torch
import math
from torch import nn


class SinusoidalPositionEmbeddings(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * - embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings



class ConvBlock(nn.Module):
    def __init__(self, start_channels, channels, block_size, time_embed_dim=6, dropout=0.1):
        super().__init__()

        self.time_emb = (
            nn.Sequential(nn.Linear(time_embed_dim, channels))
        )

        self.first_conv = nn.Sequential(
            nn.Conv2d(start_channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.convs = nn.ModuleList()

        for _ in range(block_size - 1):
            f = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

            self.convs.append(f)


    def forward(self, x, time_emb):

        t_out = self.time_emb(time_emb)
        x = self.first_conv(x) + t_out.reshape((t_out.size(0), t_out.size(1), 1, 1))

        for l in self.convs:
            x = l(x)

        return x

    


class Model(nn.Module):
    def __init__(
        self, 
        u_depth=3, 
        h_channels=32, 
        block_size=3, 
        dim=28, 
        time_embed_dim=6,
        device='cuda'
    ):
        super().__init__()

        self.device = device
        self.pos_embedder = SinusoidalPositionEmbeddings(time_embed_dim)

        self.init_conv = nn.Conv2d(3, h_channels, 3, padding=1)

        self.downconvs = nn.ModuleList()
        self.upconvs = []
        self.conv_ts = []


        for i in range(u_depth):
            ch = h_channels * (2 ** i)

            dc = ConvBlock(ch//2 if i != 0 else ch, ch, block_size)

            self.downconvs.append(dc)

            uc = ConvBlock(ch * 2, ch, block_size)

            self.upconvs.append(uc)


            u = nn.Sequential(
                nn.ConvTranspose2d(ch * 2, ch, 2, stride=2),
                nn.ReLU()
            )

            self.conv_ts.append(u)

        self.upconvs = nn.ModuleList(self.upconvs[::-1])
        self.conv_ts = nn.ModuleList(self.conv_ts[::-1])


        bottom_ch = h_channels * (2 ** u_depth)

        self.bottom = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(bottom_ch // 2 if i == 0 else bottom_ch, bottom_ch, 3, padding=1),
                nn.ReLU(),
            ) for i in range(block_size)]
        )

        self.upsamples = nn.ModuleList()

        self.outconv = nn.Conv2d(h_channels, 3, 3, padding=1)


    def forward(self, x, t=None):

        if t is None:
            print("no t")
            t = torch.ones(x.shape[0]).to(self.device)
        
        time_emb = self.pos_embedder(t)

        time_embedding = self.pos_embedder(t)


        x = self.init_conv(x)

        u_outs = []

        for l in self.downconvs:
            x = l(x, time_emb)
            u_outs.append(x)
            x = nn.functional.max_pool2d(x, 2, 2)

        u_outs.reverse()

        x = self.bottom(x)

        for up, fw, res in zip(self.conv_ts, self.upconvs, u_outs):
            x = up(x)
            # break
            x = torch.cat((res, x), dim=1)
            x = fw(x, time_emb)

        return self.outconv(x)