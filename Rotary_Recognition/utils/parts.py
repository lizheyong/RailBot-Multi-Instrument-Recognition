import torch.nn as nn


class Conv(nn.Module):
    """(Conv => BN => ReLU =>MaxPool)"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.conv(x)

class FC(nn.Module):
    """Flatten => FC => out(h)"""

    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),#12x12x32=4608 53x55x32=93280
            nn.Linear(4608, 1000),
            nn.Linear(1000, 500),
            nn.Linear(500, 8)
        )
    def forward(self, x):
        return self.fc(x)


class FCDilas(nn.Module):
    """Flatten => FC => out(h)"""

    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),#12x12x32=4608 53x55x32=93280
            nn.Linear(4608, 1000),
            nn.Linear(1000, 500),
            nn.Linear(500, 2)
        )
    def forward(self, x):
        return self.fc(x)