import torch.nn as nn
import torch.nn.functional as F


class ResNetBlock(nn.Module):

    def __init__(self, ifm, ofm, stride=1, groups=[1,1]):
        super(ResNetBlock, self).__init__()

        self.conv1 = nn.Conv2d(ifm, ofm, kernel_size=3, stride=stride, padding=1, groups = groups[0], bias=False)
        self.bn1 = nn.BatchNorm2d(ofm)
        self.conv2 = nn.Conv2d(ofm, ofm, kernel_size=3, stride=1, padding=1, groups = groups[0], bias=False)
        self.bn2 = nn.BatchNorm2d(ofm)

        self.shortcut = nn.Sequential()
        if stride != 1 or ifm != ofm:
            self.shortcut = nn.Sequential(
                nn.Conv2d(ifm, ofm, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(ofm)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.ifm = blocks[0][0]
        self.conv1 = nn.Conv2d(3, self.ifm, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.ifm)
        blocks_array = []
        previous_fmaps = blocks[0][0]
        for (fmaps, stride, groups) in blocks:
            blocks_array.append(ResNetBlock(previous_fmaps, fmaps, stride, groups))
            previous_fmaps = fmaps
        self.blocks = nn.ModuleList(blocks_array)
        self.linear = nn.Linear(blocks[-1][0], num_classes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        for block in self.blocks:
            out = block(out)

        out = F.avg_pool2d(out, (out.shape[2], out.shape[3]))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

