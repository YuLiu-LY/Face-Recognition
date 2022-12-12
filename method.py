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
        if self.args.gpus > 0:
            batch_img = batch_img.to(self.device)

        pred, _ = self.model.predict(batch_img) * batch_img[:, 0:1]
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
        pred, dist = self.model.predict(batch_img)
        loss = - dist.mean()
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

        batch_img = batch['image']
        pred, dist = self.model.predict(batch_img)
        loss = - dist.mean()
        
        output = {
            'loss': loss,
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
        optimizer = optim.SGD(params, lr=self.args.lr, momentum=0.9, weight_decay=1e-4)
        
        def lr_scheduler_main(epoch: int):
            factor = 0.5 * (1. + math.cos(math.pi * epoch / self.args.max_epochs))
            return factor

        scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=[lr_scheduler_main])

        return (
            [optimizer],
            [{"scheduler": scheduler, "interval": "epoch",}],
        )

    def cosine_anneal(self, step, final_step, start_step=0, start_value=1.0, final_value=0.1):
    
        assert start_value >= final_value
        assert start_step <= final_step
        
        if step < start_step:
            value = start_value
        elif step >= final_step:
            value = final_value
        else:
            a = 0.5 * (start_value - final_value)
            b = 0.5 * (start_value + final_value)
            progress = (step - start_step) / (final_step - start_step)
            value = a * math.cos(math.pi * progress) + b
        return value

    