import math
import torch
import pytorch_lightning as pl
from torch import optim
from torchvision import utils as vutils
from torch.nn import functional as F

import os
import sys
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)


class FaceMethod(pl.LightningModule):
    def __init__(self, model, datamodule: pl.LightningDataModule, args):
        super().__init__()
        self.model = model
        self.datamodule = datamodule
        self.args = args
        self.val_iter = iter(self.datamodule.val_dataloader())
        self.sample_num = 0
        self.empty_cache = True

    def forward(self, input, **kwargs):
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx):
        batch_img = batch['image']
        loss = self.model.loss(batch_img)
        logs = {'loss': loss}
        self.log_dict(logs, sync_dist=True)
        return {'loss': loss}

    def sample_images(self):
        if self.sample_num % (len(self.val_iter) - 1) == 0:
            self.val_iter = iter(self.datamodule.val_dataloader())
        self.sample_num += 1

        batch = next(self.val_iter)
        batch_img = batch['image'][:self.args.n_samples]
        label = batch['label'][:self.args.n_samples] # [B]
        if self.args.gpus > 0:
            batch_img = batch_img.to(self.device)

        pred, _ = self.model.predict(batch_img, label)
        pred = pred.reshape(-1, 1, 1, 1, 1).expand(-1, 1, 3, 128, 128)
        out = torch.cat([batch_img, pred], dim=1)
        B, _, C, H, W = batch_img.shape
        images = vutils.make_grid(
            out.reshape(-1, C, H, W), normalize=False, nrow=2,
            padding=3, pad_value=0,
        )

        return images

    def validation_step(self, batch, batch_idx):
        if self.empty_cache:
            torch.cuda.empty_cache()
            self.empty_cache = False

        batch_img = batch['image']
        label = batch['label'] # [B]
        pred, loss = self.model.predict(batch_img, label)
        acc = (pred == label).float().mean()

        output = {
            'loss': loss,
            'acc': acc,
        }
       
        return output

    def validation_epoch_end(self, outputs):
        self.empty_cache = True
        keys = outputs[0].keys()
        logs = {}
        for k in keys:
            v = torch.stack([x[k] for x in outputs]).mean()
            logs['avg_' + k] = v
        self.log_dict(logs, sync_dist=True)
        print("; ".join([f"{k}: {v.item():.6f}" for k, v in logs.items()]))
    
    def test_step(self, batch, batch_idx):
        if self.empty_cache:
            torch.cuda.empty_cache()
            self.empty_cache = False

        # batch_img = batch['image']
        # pred, loss = self.model.predict(batch_img)
        
        # output = {
        #     'loss': loss,
        # }
        batch_img = batch['image']
        label = batch['label'] # [B]
        pred, loss = self.model.predict(batch_img, label)
        acc = (pred == label).float().mean()

        output = {
            'loss': loss,
            'acc': acc,
        }
       
        return output

    def test_epoch_end(self, outputs):
        self.empty_cache = True
        keys = outputs[0].keys()
        logs = {}
        for k in keys:
            v = torch.stack([x[k] for x in outputs]).mean()
            logs['avg_' + k] = v
        self.log_dict(logs)
        print("; ".join([f"{k}: {v.item():.6f}" for k, v in logs.items()]))


    def configure_optimizers(self):
        params = self.model.parameters()
        optimizer = optim.AdamW(params, lr=self.args.lr, weight_decay=1e-4)

        warmup_steps = self.args.warmup_rate * self.args.max_steps
        decay_steps = self.args.decay_rate * self.args.max_steps
        
        def lr_scheduler_main(step: int):
            if step < warmup_steps:
                factor = step / warmup_steps
            else:
                factor = 1
            factor *= 0.5 ** (step // decay_steps)
            return factor

        scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=[lr_scheduler_main])

        return (
            [optimizer],
            [{"scheduler": scheduler, "interval": "step",}],
        )

    