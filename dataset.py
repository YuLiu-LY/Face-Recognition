import os
import sys
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)

import torch
import random
from PIL import Image
from glob import glob
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms


class FaceDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        split:str,
    ):
        super().__init__()
        self.data_root = data_root
        self.split = split

        self.get_files()

        self.T1 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.T2 = transforms.Compose([
            transforms.RandomHorizontalFlip(1),
            transforms.RandomApply([transforms.ColorJitter(0.3, 0.15, 0.1, 0.1)], p=0.5),
            transforms.RandomApply([transforms.GaussianBlur(31, 2)], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
    def __getitem__(self, index: int):
        out = {}
        img_paths = self.img_files[index]
        if self.split ==  'train':
            if len(img_paths) < 2:
                img1 = Image.open(img_paths[0]).convert("RGB")
                img_pair= [self.T1(img1), self.T2(img1)]
            else:
                random.shuffle(img_paths)
                img1 = Image.open(img_paths[0]).convert("RGB")
                img2 = Image.open(img_paths[1]).convert("RGB")
                img_pair = [self.T1(img1), self.T1(img2)]
        else:
            img1 = Image.open(img_paths[0]).convert("RGB")
            img2 = Image.open(img_paths[1]).convert("RGB")
            img_pair = [self.T1(img1), self.T1(img2)]
            if self.split == 'val':
                label = self.labels[index]
                out['label'] = torch.tensor(label).int()
        img_pair = torch.stack(img_pair, dim=0)
        out['image'] = img_pair
        return out

    def __len__(self):
        return len(self.img_files)
    
    def get_files(self):
        with open(f'{self.data_root}/{self.split}.txt', 'r') as f:
            lines = f.read().splitlines()
            if self.split == 'train':
                img_dirs = lines
                self.img_files = [sorted(glob(f'{dir}/*_a.jpg')) for dir in img_dirs]
            elif self.split == 'val':
                pairs = [line.split(',') for line in lines]
                self.img_files = [[pair[0], pair[1]] for pair in pairs]
                self.labels = [int(pair[2]) for pair in pairs]
            else:
                pairs = [line.split(',') for line in lines]
                self.img_files = [[pair[0], pair[1]] for pair in pairs]


class FaceDataModule(pl.LightningDataModule):
    def __init__(
        self,
        args,
    ):
        super().__init__()
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

        self.train_dataset = FaceDataset(args.data_root, 'train')
        self.val_dataset = FaceDataset(args.data_root, 'val')
        self.test_dataset = FaceDataset(args.data_root, 'val')

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


'''test'''
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.data_root = '/scratch/generalvision/SlotAttention/Face'
    args.use_rescale = False
    args.batch_size = 20
    args.num_workers = 0

    datamodule = FaceDataModule(args)
    dl = datamodule.val_dataloader()
    it = iter(dl)
    batch = next(it)
    print(batch['image'].shape)
    print(batch['label'])
    
