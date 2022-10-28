import torch
from torch import nn

# Residual block
class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNet, self).__init__()
        self.line1 = nn.Linear(in_channels, in_channels*2)
        self.bn1 = nn.BatchNorm1d(in_channels*2)
        self.relu = nn.ReLU(inplace=True)
        self.line2 = nn.Linear(in_channels*2, out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.line3 = nn.Linear(in_channels, out_channels)
        self.bn3 = nn.BatchNorm1d(out_channels)


    def forward(self, x):
        residual = x
        out = self.line1(x)
        out = torch.unsqueeze(out, 2)
        out = self.bn1(out)
        out = torch.squeeze(out, 2)
        out = self.relu(out)
        out = self.line2(out)
        out = torch.unsqueeze(out, 2)
        out = self.bn2(out)
        out = torch.squeeze(out, 2)
        skip = self.line3(residual)
        skip = torch.unsqueeze(skip, 2)
        skip = self.bn3(skip)
        skip = torch.squeeze(skip, 2)
        out += skip
        out = self.relu(out)
        return out