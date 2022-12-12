import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


class FaceModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.backbone = resnet50(pretrained=False)
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

        # # prediction head
        # pred_dim = args.prediction_dim
        # self.predictor = nn.Sequential(
        #     nn.Linear(proj_dim, pred_dim, bias=False),
        #     nn.BatchNorm1d(pred_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(pred_dim, proj_dim)
        # ) 
        
        # self.criterion = nn.CrossEntropyLoss()
        # self.tau = 1

        # self.margin = 0.5
        self.positive_margin = 0.5
        self.negitive_margin = 2


    def forward(self, x):
        # x: [B, 2, C, H, W]
        x1 = x[:, 0]
        x2 = x[:, 1]
        z1 = self.backbone(x1)
        z2 = self.backbone(x2)
        # p1 = self.predictor(z1)
        # p2 = self.predictor(z2)
        # z1 = z1.detach()
        # z2 = z2.detach()
        return z1, z2

    def loss(self, x):
        # [B, D]
        # z1, z2, p1, p2 = self.forward(x)
        # loss = 0.5 * (self.contrastive_loss(z1, p2) + self.contrastive_loss(z2, p1))
        # return loss
        z1, z2 = self.forward(x)
        loss = self.triplet_loss(z1, z2)
        return loss.mean()
    
    def triplet_loss(self, z1, z2):
        # compute the euclidean distance between all embeddings
        dist = torch.cdist(z1, z2) # [B, B]
        dist1 = dist + torch.eye(dist.shape[0], device=dist.device) * 1e6
        negative = torch.min(dist1, dim=1)[0] # [B]
        positive = torch.diagonal(dist) # [B]
        # compute the triplet loss
        # loss = torch.clamp(self.margin + positive - negative, min=0.0)
        loss = torch.clamp(positive - self.positive_margin, min=0.0) + torch.clamp(self.negitive_margin - negative, min=0.0)
        return loss.mean()

    def predict(self, x):
        # x: [B, 2, C, H, W]
        # z1, z2, p1, p2 = self.forward(x)
        # loss = 0.5 * (self.contrastive_loss(z1, p2) + self.contrastive_loss(z2, p1))
        # pred = nn.CosineSimilarity(dim=1)(z1, z2) > self.margin
        # return pred.int(), loss
        z1, z2 = self.forward(x)
        loss = self.triplet_loss(z1, z2)
        pred = F.pairwise_distance(z1, z2) < 1
        return pred.int(), loss.mean()
    
    def contrastive_loss(self, p, z):
        # normalize
        p = F.normalize(p, dim=1)
        z = F.normalize(z, dim=1)
        logits = torch.einsum('nc, mc->nm', [p, z]) / self.tau
        N = logits.shape[0]
        labels = torch.arange(N, dtype=torch.long).to(logits.device)
        return self.criterion(logits, labels) * (2 * self.tau)
