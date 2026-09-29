import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()
        self.pool = nn.MaxPool3d(2)
        self.dropout = nn.Dropout3d(dropout)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x):
        return self.conv(self.dropout(self.pool(x)))


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class NNUNet3D(nn.Module):
    """3D U-Net predicting a residual added to the input LR volume (input-skip super-resolution)."""

    def __init__(self, in_channels=1, out_channels=1, base_features=32, drop=0.0):
        super().__init__()
        f = base_features
        self.enc1 = ConvBlock(in_channels, f)
        self.enc2 = DownBlock(f, f * 2, drop)
        self.enc3 = DownBlock(f * 2, f * 4, drop)
        self.enc4 = DownBlock(f * 4, f * 8, drop)
        self.bottom = DownBlock(f * 8, f * 16, 0)

        self.up4 = UpBlock(f * 16 + f * 8, f * 8)
        self.up3 = UpBlock(f * 8 + f * 4, f * 4)
        self.up2 = UpBlock(f * 4 + f * 2, f * 2)
        self.up1 = UpBlock(f * 2 + f, f)

        self.final_conv = nn.Conv3d(f, out_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b = self.bottom(e4)
        d4 = self.up4(b, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        out = self.final_conv(d1)
        return self.relu(x + out)
