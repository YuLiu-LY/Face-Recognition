import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class FaceModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.backbone = resnet18(pretrained=False)
        hidden_dim = self.backbone.fc.weight.shape[1]

        # projection head
        proj_dim = args.projection_dim
        self.backbone.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True), 
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim, bias=False),
            nn.BatchNorm1d(proj_dim, affine=False)
        ) 

        # prediction head
        pred_dim = args.prediction_dim
        self.predictor = nn.Sequential(
            nn.Linear(proj_dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),
            nn.Linear(pred_dim, proj_dim)
        ) 
        
        self.criterion = nn.CosineSimilarity(dim=1)

    def forward(self, x):
        # x: [B, 2, C, H, W]
        x1 = x[:, 0]
        x2 = x[:, 1]
        z1 = self.backbone(x1)
        z2 = self.backbone(x2)
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        z1 = z1.detach()
        z2 = z2.detach()
        return z1, z2, p1, p2

    def loss(self, x):
        # [B, D]
        z1, z2, p1, p2 = self.forward(x)
        similarity = 0.5 * (self.criterion(z1, p2) + self.criterion(z2, p1))
        loss = -similarity.mean()
        return loss

    def predict(self, x, threshold=0.5):
        # x: [B, 2, C, H, W]
        z1, z2, p1, p2 = self.forward(x)
        similarity = 0.5 * (self.criterion(z1, p2) + self.criterion(z2, p1))
        pred = (similarity > threshold).int()
        return pred, similarity