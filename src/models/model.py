import torch.nn as nn


class Residual_Block(nn.Module):
    def __init__(self):
        super(Residual_Block, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )

    def forward(self, x):
        x = x + self.block(x)
        return x


class Generator(nn.Module):
    def __init__(self, n_residual=2):
        super(Generator, self).__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )

        self.residual_blocks = nn.Sequential(*[Residual_Block() for n in range(n_residual)])

        self.after = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
        )

        self.last = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU(),
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU(),
            nn.Conv2d(64, 3, kernel_size=9, padding=4)
        )

    def forward(self, x):
        x = self.initial(x)
        res = x

        x = self.residual_blocks(x)
        x = res + self.after(x)
        return self.last(x)


class Discriminator(nn.Module):
    def __init__(self, features=16):
        super(Discriminator, self).__init__()

        self.network = nn.Sequential(
            nn.Conv2d(3, features, kernel_size=3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features, features, kernel_size=3),
            nn.BatchNorm2d(features),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features, features * 2, kernel_size=3),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features * 2, features * 2, kernel_size=3),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features * 2, features * 4, kernel_size=3),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.network(x)
