import torch
import torch.nn as nn

import torchvision.models as models

class AlexNet(nn.Module):

    def __init__(self, num_classes: int = 1000) -> None:

        super(AlexNet, self).__init__()

        alexnet_pretrained = models.alexnet(pretrained=True)

        self.conv1 = nn.Sequential()
        for i in range(3):
            self.conv1.add_module("conv1_"+str(i), alexnet_pretrained.features[i])

        self.norm1 = nn.LocalResponseNorm(5)

        self.conv2 = nn.Sequential()
        for i in range(3,6):
            self.conv2.add_module("conv2_"+str(i), alexnet_pretrained.features[i])

        self.norm2 = nn.LocalResponseNorm(5)

        self.conv345 = nn.Sequential()
        for i in range(6, 13):
            self.conv345.add_module("conv345_" + str(i), alexnet_pretrained.features[i])

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.fc6 = nn.Sequential()
        for i in range(3):
            self.fc6.add_module("fc6_" + str(i), alexnet_pretrained.classifier[i])

        self.fc7 = nn.Sequential()
        for i in range(3,6):
            self.fc7.add_module("fc7_" + str(i), alexnet_pretrained.classifier[i])

        self.bottleneck = nn.Sequential(nn.Linear(4096, 256),
                                        nn.ReLU(),
                                        nn.Dropout(0.5))
        self.classfier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.conv345(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        f6 = self.fc6(x)
        f7 = self.fc7(f6)
        bottleneck = self.bottleneck(f7)
        y = self.classfier(bottleneck)
        if self.training:
            return f6, f7, bottleneck, y
        else:
            return y


def test(model):
    batch_data = torch.randn((32,3,224,224))
    output = model(batch_data)
    return output


if __name__ == '__main__':

    model = AlexNet(num_classes=31)

    output = test(model)