import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import Config


class Discriminator(nn.Module):
    def __init__(self, config:Config):
        super().__init__()
        self.input = nn.Conv3d(in_channels=2, out_channels=config.hidden_ch, kernel_size=1)
        self.sequential = nn.Sequential()
        for i in range(config.n_block):
            self.sequential.add_module(
                f"Block{i}", Block(config)
            )
        self.sequential.add_module(
            "ConvOut", nn.Conv3d(in_channels=config.hidden_ch, out_channels=2, kernel_size=1)
        )
        self.sequential.add_module(
            "ReLU-Out", nn.ReLU()
        )
        self.linear = nn.Linear(in_features=128, out_features=3)

    def forward(self, x):
        """
        input:
            x: torch.Float([B, 128])
        """
        x = F.relu(self.input(torch.reshape(x, (-1, 2, 4, 4, 4))))
        x1 = self.sequential(x)
        x2 = torch.reshape(x1, (-1, 128)) # [B, 128]
        x3 = F.relu(self.linear(x2))  # [B, 3]
        return torch.softmax(x3, dim=1)

    def value(self, x):
        x = F.relu(self.input(torch.reshape(x, (-1, 2, 4, 4, 4))))
        x1 = self.sequential(x)
        x2 = torch.reshape(x1, (-1, 128)) # [B, 128]
        x3 = self.linear(x2)  # [B, 3]
        return F.relu(x3)


class Block(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels=config.hidden_ch, out_channels=config.hidden_ch, kernel_size=(1, 1, 3), padding=(0, 0, 1))
        self.conv2 = nn.Conv3d(in_channels=config.hidden_ch, out_channels=config.hidden_ch, kernel_size=(1, 3, 1), padding=(0, 1, 0))
        self.conv3 = nn.Conv3d(in_channels=config.hidden_ch, out_channels=config.hidden_ch, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.relu = nn.ReLU()

    def forward(self, input_x):
        x = self.conv1(input_x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.relu(x + input_x)