import pytorch_lightning.loggers as pl_loggers
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.nn.modules import SyncBatchNorm

import os
import sys
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)


import tensorboard

import argparse

from dataset import FaceDataModule
from model import FaceModel
from method import FaceMethod
from utils import set_random_seed, state_dict_ckpt, ImageLogCallback


parser = argparse.ArgumentParser()

parser.add_argument('--data_root', default='/home/yuliu/Dataset/Face')
parser.add_argument('--log_name', default='test')
parser.add_argument('--log_path', default='/home/yuliu/Projects/Face/results/')
parser.add_argument('--ckpt_path', default='.ckpt')
parser.add_argument('--test_ckpt_path', default='ckpt.pt.tar')

parser.add_argument('--monitor', type=str, default='avg_acc', help='avg_acc')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--num_sanity_val_steps', type=int, default=1)
parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
parser.add_argument('--n_samples', type=int, default=16)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--gpus', type=int, default=0)
parser.add_argument('--device', type=str, default='1')
parser.add_argument('--grad_clip', type=float, default=0)

parser.add_argument('--is_logger_enabled', default=False, action='store_true')
parser.add_argument('--load_from_ckpt', default=False, action='store_true')

parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--warmup_rate', type=float, default=0.1)
parser.add_argument('--decay_rate', type=float, default=0.4)
parser.add_argument('--max_steps', type=int, default=50000)

parser.add_argument('--projection_dim', type=int, default=512)
parser.add_argument('--prediction_dim', type=int, default=512)

parser.add_argument('--test', action='store_true')
parser.add_argument('--contras_weight', type=float, default=1)
parser.add_argument('--triplet_weight', type=float, default=0)
parser.add_argument('--predict_mode', type=str, default='cosine', help='cosine, euclidean')


def main(args):
    print(args)
    set_random_seed(args.seed)
    # set device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    datamodule = FaceDataModule(args)
    model = FaceModel(args)
    if args.gpus > 1:
        model = SyncBatchNorm.convert_sync_batchnorm(model)
    if args.test:
        ckpt = state_dict_ckpt(args.test_ckpt_path)
        model.load_state_dict(ckpt)
    method = FaceMethod(model=model, datamodule=datamodule, args=args)
    method.hparams = args

    if args.is_logger_enabled:
        logger = pl_loggers.TensorBoardLogger(args.log_path, name=args.log_name) 
        arg_str_list = ['{}={}'.format(k, v) for k, v in vars(args).items()]
        arg_str = '__'.join(arg_str_list)
        log_dir = os.path.join(args.log_path, args.log_name)
        print(log_dir)
        logger.experiment.add_text('hparams', arg_str)
        callbacks = [LearningRateMonitor("step"), ImageLogCallback(), ModelCheckpoint(monitor=args.monitor, save_top_k=1, save_last=True, mode='max')]
    else:
        logger = False
        callbacks = []

    trainer = Trainer(
        resume_from_checkpoint=args.ckpt_path if args.load_from_ckpt else None,
        logger=logger,
        default_root_dir=args.log_path,
        accelerator="ddp" if args.gpus > 1 else None,
        num_sanity_val_steps=args.num_sanity_val_steps,
        gpus=args.gpus,
        max_epochs=100000,
        max_steps=args.max_steps,
        log_every_n_steps=50,
        callbacks=callbacks,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        gradient_clip_val=args.grad_clip,
    )
    if args.test:
        # method.find_best_threshold()
        trainer.test(method)
    else:
        trainer.fit(method)

if __name__ == "__main__":
    args = parser.parse_args()
    # args.batch_size = 64
    # args.projection_dim = 256
    # # args.test = True
    # args.gpus = 1
    # args.predict_mode = 'euclidean'
    # args.test_ckpt_path = '/home/yuliu/Projects/Face/results/d256_b512/version_4/checkpoints/last.ckpt'
    if args.gpus > 1:
        args.batch_size = args.batch_size // args.gpus
    main(args)

