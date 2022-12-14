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
        self.threshold = 0.5
        self.margin = 0

    def forward(self, input, **kwargs):
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx):
        if self.global_step >= self.args.m_warmup_steps:
            self.margin = self.args.margin
        else:
            self.margin = self.global_step / self.args.m_warmup_steps * self.args.margin
        batch_img = batch['image']
        loss = self.model.loss(batch_img, margin=self.margin)
        logs = {'loss': loss}
        self.log_dict(logs, sync_dist=True)
        return {'loss': loss}

    def sample_images(self):
        if self.sample_num % (len(self.val_iter) - 1) == 0:
            self.val_iter = iter(self.datamodule.val_dataloader())
        self.sample_num += 1

        batch = next(self.val_iter)
        batch_img = batch['image'][:self.args.n_samples]
        label = batch['label'][:self.args.n_samples]
        if self.args.gpus > 0:
            batch_img = batch_img.to(self.device)
            label = label.to(self.device)

        B, _, C, H, W = batch_img.shape

        dist = self.model.predict(batch_img)
        pred = dist < self.threshold
        pred = pred.reshape(-1, 1, 1, 1, 1).expand(-1, 1, 3, H, W)
        label = label.reshape(-1, 1, 1, 1, 1).expand(-1, 1, 3, H, W)
        label = label.float()
        batch_img = batch_img / 2 + 0.5
        out = torch.cat([batch_img, pred, label], dim=1)
        images = vutils.make_grid(
            out.reshape(-1, C, H, W), normalize=False, nrow=4,
            padding=3, pad_value=0,
        )

        return images

    def validation_step(self, batch, batch_idx):
        if self.empty_cache:
            torch.cuda.empty_cache()
            self.empty_cache = False

        batch_img = batch['image']
        label = batch['label'] # [B]
        dist = self.model.predict(batch_img)
       
        return dist, label

    def validation_epoch_end(self, outputs):
        self.empty_cache = True
        if self.args.predict_mode == 'cosine':
            thresholds = torch.linspace(-1, 0, 200) 
        elif self.args.predict_mode == 'euclidean':
            thresholds = torch.linspace(0, 1.5, 200)
        logs = {}
        dists = []
        labels = []
        for dist, label in outputs:
            dists.append(dist)
            labels.append(label)
        dists = torch.cat(dists, dim=0)
        labels = torch.cat(labels, dim=0)
        accs = []
        for threshold in thresholds:
            pred = dists < threshold
            acc = (pred == labels).float().mean()
            accs.append(acc)
        accs = torch.stack(accs, dim=0)
        logs['avg_acc'] = accs.max()
        best_threshold = thresholds[accs.argmax()]
        self.threshold = best_threshold
        print(f"Best threshold: {best_threshold.item():.6f}")
        print(f"Best acc for validation: {accs.max().item():.6f}")
        self.log_dict(logs, sync_dist=True)
    
    def test_step(self, batch, batch_idx):
        if self.empty_cache:
            torch.cuda.empty_cache()
            self.empty_cache = False

        batch_img = batch['image']
        label = batch['label'] # [B]
        dist = self.model.predict(batch_img)
       
        return dist, label

    def test_epoch_end(self, outputs):
        self.empty_cache = True
        if self.args.predict_mode == 'cosine':
            thresholds = torch.linspace(-1, 1, 400) 
        elif self.args.predict_mode == 'euclidean':
            thresholds = torch.linspace(0, 2, 400)
        logs = {}
        dists = []
        labels = []
        for dist, label in outputs:
            dists.append(dist)
            labels.append(label)
        dists = torch.cat(dists, dim=0)
        labels = torch.cat(labels, dim=0)
        accs = []
        for threshold in thresholds:
            pred = dists < threshold
            acc = (pred == labels).float().mean()
            accs.append(acc)
        accs = torch.stack(accs, dim=0)
        logs['avg_acc'] = accs.max()
        best_threshold = thresholds[accs.argmax()]
        self.threshold = best_threshold
        print(f"Best threshold: {best_threshold.item():.6f}")
        print(f"Best acc for validation: {accs.max().item():.6f}")


    def configure_optimizers(self):
        params = self.model.parameters()
        optimizer = optim.SGD(params, lr=self.args.lr, weight_decay=5e-4)

        warmup_steps = self.args.warmup_rate * self.args.max_steps
        decay_steps = self.args.decay_rate * self.args.max_steps
        decay1 = 16000
        decay2 = 24000
        decay3 = 28000
        
        # def lr_scheduler_main(step: int):
        #     if step < warmup_steps:
        #         factor = step / warmup_steps
        #     else:
        #         factor = 1
        #     factor *= 0.5 ** (step / decay_steps)
        #     return factor
        if self.args.lr_mode == 'cosine':
            def lr_scheduler_main(step: int):
                factor = self.cosine_anneal(step, decay3, 0, 1, 0.001)
                return factor
        elif self.args.lr_mode == 'step':
            def lr_scheduler_main(step: int):
                if step < decay1:
                    factor = 1
                elif step < decay2:
                    factor = 0.1
                elif step < decay3:
                    factor = 0.01
                else:
                    factor = 0.001
                return factor

        scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=[lr_scheduler_main])

        return (
            [optimizer],
            [{"scheduler": scheduler, "interval": "step",}],
        )

    def find_best_threshold(self):
        print("Finding best threshold...")
        print('Predict mode: ', self.args.predict_mode)
        if self.args.gpus > 0:
            self.model.to(self.device)
        if self.args.predict_mode == 'cosine':
            thresholds = torch.linspace(-1, 1, 400) 
        elif self.args.predict_mode == 'euclidean':
            thresholds = torch.linspace(0, 2, 400)
        self.model.eval()
        dataloader = self.datamodule.val_dataloader()
        with torch.no_grad():
            dists = []
            labels = []
            for batch in dataloader:
                batch_img = batch['image']
                label = batch['label']
                if self.args.gpus > 0:
                    batch_img = batch_img.to(self.device)
                    label = label.to(self.device)
                dist = self.model.predict(batch_img)
                dists.append(dist)
                labels.append(label)
            dists = torch.cat(dists, dim=0)
            labels = torch.cat(labels, dim=0)

            accs = []
            for threshold in thresholds:
                pred = dists < threshold
                acc = (pred == labels).float().mean()
                accs.append(acc)
            accs = torch.stack(accs, dim=0)
            best_threshold = thresholds[accs.argmax()]
            self.threshold = best_threshold
            print(f"Best threshold: {best_threshold.item():.6f}")
            print(f"Best acc for validation: {accs.max().item():.6f}")

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