import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class FaceModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.backbone = resnet18(pretrained=False)
        self.backbone.fc = nn.Identity()

    def forward(self, input, **kwargs):
        out = self.backbone(input)
        return out