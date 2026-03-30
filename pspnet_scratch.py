import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins=(1, 2, 3, 6)):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(nn.functional.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)

class PSPNetScratch(nn.Module):
    def __init__(self, num_classes=21, use_aux=False):
        super(PSPNetScratch, self).__init__()
        
        # Load a pretrained ResNet50 for backbone
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        # Note: True PSPNet converts the ResNet backbone layers 3 and 4 to use dilated convolutions
        # to preserve spatial resolution (stride=1, dilation=2/4). We omit exact dilation details
        # here for a simplified "from scratch" toy implementation, but it can be added.
        
        # PPM: Original in_channels = 2048 from resnet.layer4
        self.ppm = PPM(in_dim=2048, reduction_dim=512, bins=(1, 2, 3, 6))
        
        self.cls = nn.Sequential(
            nn.Conv2d(2048 + 512*4, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(512, num_classes, kernel_size=1)
        )
        
    def forward(self, x):
        size = x.size()
        
        # Backbone
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # PSP Module
        x = self.ppm(x)
        
        # Classifier
        x = self.cls(x)
        
        # Upsample to original size target
        x = nn.functional.interpolate(x, size=size[2:], mode='bilinear', align_corners=True)
        return x

if __name__ == '__main__':
    model = PSPNetScratch(num_classes=21)
    dummy_input = torch.randn(2, 3, 256, 256)
    out = model(dummy_input)
    print("Scratch Output shape:", out.shape)
