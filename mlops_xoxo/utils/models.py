import torch
import torch.nn.functional as F
import torchvision
from torchvision import models
import torch.nn as nn
import yaml
from torchvision.models import resnet18, ResNet18_Weights

class MobileFace(torch.nn.Module):
    def __init__(self, emb_size=512):
        super().__init__()
        backbone = models.mobilenet_v2(weights="IMAGENET1K_V1")
        self.backbone = backbone.features
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(backbone.last_channel, emb_size)
        self.bn = torch.nn.BatchNorm1d(emb_size)

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x).flatten(1)
        x = self.fc(x)
        x = self.bn(x)
        return F.normalize(x)
    
with open('pipelines/age_gender/params.yaml', 'r') as f:
    age_gender_params = yaml.safe_load(f)

class SimpleModel(nn.Module):
    """Lightweight MobileNetV2-based model"""
    def __init__(self, num_outputs):
        super().__init__()
        self.backbone = models.mobilenet_v2(weights='IMAGENET1K_V1')
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=age_gender_params['model']['dropout']),
            nn.Linear(self.backbone.last_channel, num_outputs)
        )
    
    def forward(self, x):
        out = self.backbone(x)
        return out.squeeze() if out.shape[1] == 1 else out
    
def build_resnet18(num_classes: int, pretrained: bool = True, in_channels: int = 1):
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    if in_channels != 3:
        old_conv = model.conv1
        model.conv1 = nn.Conv2d(in_channels, old_conv.out_channels,
                                kernel_size=old_conv.kernel_size,
                                stride=old_conv.stride,
                                padding=old_conv.padding,
                                bias=(old_conv.bias is not None))
        if pretrained and getattr(old_conv, "weight", None) is not None:
            with torch.no_grad():
                model.conv1.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def build_model(backbone: str, num_classes: int,
                pretrained: bool = True, freeze_backbone: bool = False) -> nn.Module:
    if backbone == "mobilenet_v3_small":
        weights = torchvision.models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        m = torchvision.models.mobilenet_v3_small(weights=weights)
        in_feat = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_feat, num_classes)
        head_names = ["classifier.3", "classifier.1"]
    elif backbone == "resnet18":
        weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        m = torchvision.models.resnet18(weights=weights)
        in_feat = m.fc.in_features
        m.fc = nn.Linear(in_feat, num_classes)
        head_names = ["fc"]
    elif backbone == "efficientnet_b0":
        weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        m = torchvision.models.efficientnet_b0(weights=weights)
        in_feat = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_feat, num_classes)
        head_names = ["classifier.1", "classifier.3"]
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    if freeze_backbone:
        for name, p in m.named_parameters():
            p.requires_grad = any(k in name for k in head_names)
    return m