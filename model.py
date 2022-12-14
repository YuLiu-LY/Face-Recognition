import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34


class FaceModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.backbone = FaceNet(args.N_layer, args.projection_dim)

        self.contras_weight = args.contras_weight
        self.triplet_weight = args.triplet_weight
        self.predict_mode = args.predict_mode # 'cosine' or 'euclidean'
        
        self.s = args.scale
        if args.learn_scale:
            self.s = nn.Parameter(torch.tensor(self.s).log())

    def forward(self, x):
        # x: [B, 2, C, H, W]
        x1 = x[:, 0]
        x2 = x[:, 1]
        z1 = self.backbone(x1)
        z2 = self.backbone(x2)
        return z1, z2

    def loss(self, x, margin=0):
        # [B, D]
        z1, z2 = self.forward(x)
        loss = 0
        if self.contras_weight > 0:
            loss1 = self.contrastive_loss(z1, z2, m=margin)
            loss += loss1 * self.contras_weight
        if self.triplet_weight > 0:
            loss2 = self.triplet_loss(z1, z2, m=margin)
            loss += loss2 * self.triplet_weight
        return loss
    
    def triplet_loss(self, z1, z2, m=0):
        # compute the euclidean distance between all embeddings
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        dist = torch.cdist(z1, z2) # [B, B]
        positive = torch.diagonal(dist) # [B]
        negtive_mask = torch.ones_like(dist) - torch.eye(dist.shape[0]).to(dist.device)
        negative = dist * negtive_mask + 1e6 * (1 - negtive_mask) # [B, B] 
        hard_negative = negative.min(dim=1)[0] # [B]
        # compute the triplet loss
        loss = torch.relu(m + positive - hard_negative)
        return loss.mean()
    
    def contrastive_loss(self, p, z, m=0):
        # normalize
        p = F.normalize(p, dim=1)
        z = F.normalize(z, dim=1)
        logits = torch.einsum('nc, mc->nm', [p, z]) 
        logits = logits - m * torch.eye(logits.shape[0]).to(logits.device)
        logits = self.s * logits
        loss = - F.log_softmax(logits, dim=1).diag()
        return loss.mean()
    
    def predict(self, x):
        # x: [B, 2, C, H, W]
        z1, z2 = self.forward(x)
        if self.predict_mode == 'cosine':
            dist = - F.cosine_similarity(z1, z2)
        elif self.predict_mode == 'euclidean':
            z1 = F.normalize(z1, dim=1)
            z2 = F.normalize(z2, dim=1)
            dist = F.pairwise_distance(z1, z2)
        return dist


class FaceNet(nn.Module):
    def __init__(self, N_layer=64, fc_dim=512):
        super().__init__()
        if N_layer == 36:
            # 36-Layer CNN
            N_blocks = [2, 4, 8, 2]
        elif N_layer == 64:
            # 64-layer CNN
            N_blocks = [3, 8, 16, 3]
        else:
            raise ValueError('N_layer must be 36 or 64')
        blocks = [
            nn.Sequential(
                DownBlock(3, 64),
                *[ConvBlock(64, 64) for _ in range(N_blocks[0])]
            )
        ]
        ch_in = 64
        for i in range(1, len(N_blocks)):
            n = N_blocks[i]
            ch_out = ch_in * 2
            blocks.append(nn.Sequential(
                DownBlock(ch_in, ch_out),
                *[ConvBlock(ch_out, ch_out) for _ in range(n)]
            ))
            ch_in = ch_out
        self.conv_blocks = nn.Sequential(*blocks)
        self.fc = nn.Linear(512, fc_dim)
    
    def forward(self, x):
        x = self.conv_blocks(x)
        x = F.adaptive_avg_pool2d(x, 1).squeeze()
        x = self.fc(x)
        return x


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 2, 1),
            nn.BatchNorm2d(out_ch),
            nn.PReLU(out_ch),
        )
    
    def forward(self, x):
        return self.block(x)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.PReLU(out_ch),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.PReLU(out_ch),
        )
    
    def forward(self, x):
        return x + self.block(x)