# -*- coding:utf-8 -*-
import torch


def conv1x1(in_channels: int, out_channels: int, stride: int=1, bias: bool=False):
    return torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=bias)


def conv3x3(in_channels: int, out_channels: int, stride: int=1, bias: bool=False):
    return torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=bias)


class BasicBlock(torch.nn.Module):
    """
        bottleneck for resnet18 and resnet34
    """
    expansion = 1
    def __init__(self, in_channels: int, out_channels: int, stride: int, is_bn: bool):
        super(BasicBlock, self).__init__()
        self.conv = torch.nn.Sequential(
                        conv3x3(in_channels, out_channels, stride=stride),
                        torch.nn.BatchNorm2d(out_channels) if is_bn else torch.nn.Sequential(),
                        torch.nn.ReLU(inplace=True),
                        conv3x3(out_channels, out_channels),
                        torch.nn.BatchNorm2d(out_channels) if is_bn else torch.nn.Sequential(),)
        self.shortcut = torch.nn.Sequential()
        if (stride != 1) or (in_channels != out_channels):
            self.shortcut = torch.nn.Sequential(
                                conv1x1(in_channels, out_channels, stride=stride), 
                                torch.nn.BatchNorm2d(out_channels),)

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class BottleNeck(torch.nn.Module):
    """
        bottleneck for resnet50 and resnet101
    """
    expansion = 4
    def __init__(self, in_channels: int, out_channels: int, stride: int, is_bn: bool):
        super(BottleNeck, self).__init__()
        self.conv = torch.nn.Sequential(
                        conv1x1(in_channels, out_channels),
                        torch.nn.BatchNorm2d(out_channels) if is_bn else torch.nn.Sequential(),
                        torch.nn.ReLU(inplace=True),
                        conv3x3(out_channels, out_channels, stride=stride),
                        torch.nn.BatchNorm2d(out_channels) if is_bn else torch.nn.Sequential(),
                        torch.nn.ReLU(inplace=True),
                        conv1x1(out_channels, (out_channels*BottleNeck.expansion)),
                        torch.nn.BatchNorm2d(out_channels) if is_bn else torch.nn.Sequential(),)
        self.shortcut = torch.nn.Sequential()
        if (stride != 1) or (in_channels != (out_channels*BottleNeck.expansion)):
            self.shortcut = torch.nn.Sequential(
                                conv1x1(in_channels, (out_channels*BottleNeck.expansion), stride=stride), 
                                torch.nn.BatchNorm2d(out_channels),)

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class ResNet(torch.nn.Module):
    def __init__(self, block, blocks, class_num=1000, is_bn=True):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
                        torch.nn.BatchNorm2d(64) if is_bn else torch.nn.Sequential(),
                        torch.nn.ReLU(inplace=True),
                        torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1),)
        self.conv2 = self._make_layer(block, 64, blocks[0], 1, True)
        self.conv3 = self._make_layer(block, 128, blocks[1], 2, True)
        self.conv4 = self._make_layer(block, 256, blocks[2], 2, True)
        self.conv5 = self._make_layer(block, 512, blocks[3], 2, True)
        self.fc = torch.nn.Sequential(
                        torch.nn.AdaptiveAvgPool2d((1, 1)),
                        torch.nn.Linear((512*block.expansion), class_num, bias=True),
                        torch.nn.LogSoftmax(dim=1),)

    def _make_layer(self, block, out_channels: int, block_num: int, stride: int, is_bn: bool):
        layers = []
        strides = [stride,] + [1] * (block_num - 1)
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s, is_bn))
            self.in_channels = out_channels * block.expansion
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = out.view(out.size(0), -1)
        return self.fc(out)

def resnet18():
    return ResNet(BasicBlock, [2, 2, 2, 2]) 

def resnet34():
    return ResNet(BasicBlock, [3, 4, 6, 3]) 

def resnet50():
    return ResNet(BottleNeck, [3, 4, 6, 3]) 

def resnet101():
    return ResNet(BottleNeck, [3, 4, 23, 3]) 

def resnet152():
    return ResNet(BottleNeck, [3, 8, 36, 3]) 


if __name__ == '__main__':
    model = resnet50()
    print(model)
