import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights


class ResNet50Backbone(torch.nn.Module):
    def __init__(self, weights=ResNet50_Weights.DEFAULT):
        super().__init__()
        # Load pretrained torchvision ResNet50
        base_model = resnet50(weights=weights)

        # copy base layers (except for fc head)
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2  # C3 -> 28x28x512
        self.layer3 = base_model.layer3  # C4 -> 14x14x1024
        self.layer4 = base_model.layer4  # C5 -> 7x7x2048

        self.aux1 = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )  # output: 4x4x1024 (approximate due to stride 2)

        self.aux2 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )  # output: 2x2x512

        self.aux3 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )  # output: 1x1x256

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        c1 = self.maxpool(x)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)    # C3 -> 28 * 28
        c4 = self.layer3(c3)    # C4 -> 14 * 14
        c5 = self.layer4(c4)    # C5 -> 7 * 7

        c6 = self.aux1(c5)      # C6 -> 4 * 4
        c7 = self.aux2(c6)      # C7 -> 2 * 2
        c8 = self.aux3(c7)      # C8 -> 1 * 1

        # return {"c3": c3, "c4": c4, "c5": c5, "c6": c6, "c7": c7, "c8": c8}
        return c3, c4, c5, c6, c7, c8


if __name__ == "__main__":
    model = ResNet50Backbone()
    features = model(torch.randn(1, 3, 224, 224))

    print([k.size() for k in features])