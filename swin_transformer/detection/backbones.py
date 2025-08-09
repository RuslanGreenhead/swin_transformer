import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights


class ResNet50Backbone(torch.nn.Module):
    def __init__(self, img_size=224, weights="IMAGENET_V2"):
        super().__init__()

        # load pure ResNet50
        base_model = resnet50(weights=None)
        # load pretrained weights
        if weights == "IMAGENET_V2":
            state_dict = torch.load("../saved_weights/resnet50_statedict.pth", weights_only=True)
            base_model.load_state_dict(state_dict)

        # copy base layers (except for fc head)
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2  
        self.layer3 = base_model.layer3 
        self.layer4 = base_model.layer4

        self.aux1 = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.aux2 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.aux3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=(1 if img_size == 224 else 0)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self._init_auxilary_layers()


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        c1 = self.maxpool(x)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)    # (224x224) -> 512x28x28      (300x300) -> 512x38x38
        c4 = self.layer3(c3)    # (224x224) -> 1024x14x14     (300x300) -> 1024x19x19
        c5 = self.layer4(c4)    # (224x224) -> 2048x7x7       (300x300) -> 2048x10x10
    
        c6 = self.aux1(c5)      # (224x224) -> 512x4x4        (300x300) -> 512x5x5
        c7 = self.aux2(c6)      # (224x224) -> 256x2x2        (300x300) -> 256x3x3
        c8 = self.aux3(c7)      # (224x224) -> 256x1x1        (300x300) -> 256x1x1

        return c3, c4, c5, c6, c7, c8
    

    def _init_auxilary_layers(self):
        for name, module in self.named_modules():
            if "aux" in name:
                for m in module.modules():
                    if isinstance(m, (nn.Conv2d, nn.Linear)):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)


if __name__ == "__main__":
    model = ResNet50Backbone()
    features = model(torch.randn(1, 3, 224, 224))

    print([k.size() for k in features])