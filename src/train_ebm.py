import os
import argparse
from tqdm import tqdm
import json
import pickle 
import random
import torch
import wandb
import numpy as np
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CelebA
from torch.nn.utils import clip_grad_norm
import pytorch_lightning as pl
from lightning.pytorch.utilities import rank_zero_only

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from model.model import ebm_lightning_module
from model.data import get_data,get_data_zero_shot,get_data_specific_subset
from model.utils import clamp_x, ReplayBuffer, ReservoirBuffer
from model.callbacks import SaveReplayBufferCallback
import matplotlib.pyplot as plt
from model.constant import CONFLICTS_LIST, ATTRIBUTE_LIST

def main(args):

    # data settings
    train_loader, val_loader, test_loader, label_dim, indexs = get_data_zero_shot(args)
    args.label_dim = label_dim
    args.label_list = indexs
    
    # model settings
    model = ebm_lightning_module(args)

    # load replay buffer
    if args.buffer_path:
        with open(args.buffer_path,"rb") as file:
            model.sampler = pickle.load(file)

    # callback settings

    if rank_zero_only.rank == 0:
        wandb.init(
            project=f"{args.proj_name}", 
            name=f"{args.exp_name}",
            config=args
        )
    else:
        wandb.logger = None

    # Initialize WandB logger for PyTorch Lightning (This is passed to the trainer)
    #wandb_logger = WandbLogger(log_model="all") if rank_zero_only.rank == 0 else None
    wandb_logger = WandbLogger(log_model="all")

    # callback settings

    # Initialize ModelCheckpoint callback to save checkpoint after each epoch
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.save_dir,  # Directory to save checkpoints
        filename="epoch-{epoch:02d}",  # Save every epoch with filename `epoch-XX.ckpt`
        save_top_k=-1,  # Save all checkpoints (default is to save the best one only)
        save_weights_only=False,  # Save only the model weights, not the entire model
        verbose=True  # Enable printing when saving checkpoints
    )

    callback_list = [
        SaveReplayBufferCallback(),
        checkpoint_callback
    ]

    # trainer settings

    trainer = pl.Trainer(max_epochs=args.epoch_num,
    default_root_dir=args.save_dir,
    accelerator="gpu",
    devices=args.gpus,
    strategy=args.strategy,
    callbacks=callback_list,
    logger=wandb_logger
    )

    if trainer.local_rank == 0:  # 只有rank 0的进程才会打印
        print(json.dumps(vars(args), indent=4))
        print("attribute considered")
        for index in indexs:
            print(ATTRIBUTE_LIST[index])

    if args.check_point_path:
        trainer.fit(model, train_loader, val_loader, ckpt_path=args.check_point_path)
    else:
        trainer.fit(model, train_loader, val_loader)
   
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--proj_name', type=str, help='project name')
    parser.add_argument('--exp_name', type=str, help='experiment name')
    # distributed training hyperparameters
    parser.add_argument('--gpus', type=int, default=1,help='number of gpus to use')
    parser.add_argument('--strategy', type=str, default='ddp',help='strategy to use for distributed training')

    # general experiment settings
    #experiment settings
    parser.add_argument('--check_point_path', type=str, default=None,help='path to load checkpoint')
    parser.add_argument('--buffer_path', type=str, default=None,help='path to load replay buffer')
    parser.add_argument('--save_dir', type=str, default=os.getcwd(),help='directory to save logs and checkpoints')

    # data settings
    parser.add_argument('--dataset',type=str, default='cifar10',choices=['mnist', 'cifar10', 'celeba'],help='dataset to use')
    parser.add_argument('--celeba_drop_infreq', type=float, default=0.13)
    parser.add_argument('--img_sigma', type = float, default=0.01)
    parser.add_argument('--dataset_transform', action='store_true', default=False, help='whether to apply dataset transformations')
    parser.add_argument('--label_smoothing', action='store_true', default=False, help='whether to apply label smoothing')
    parser.add_argument('--mix_up', action='store_true', default=False, help='whether to apply mix up')
    parser.add_argument('--epoch_num', type=int, default=10000,help='number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=128,help='batch size')
    parser.add_argument('--num_workers', type=int, default=4,help='number of workers to use for data loading')

    parser.add_argument('--val_interval', type=int, default=1000,help='interval between validation steps')

    # optimizer settings
    parser.add_argument('--lr', type=float, default=2e-4,help='learning rate')
    parser.add_argument('--beta1', type=float, default=0.0,help='beta1 for adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.9,help='beta2 for adam optimizer')

    parser.add_argument('--clip_gradients', action='store_true', default=False, help='whether clip the gradient while training')
    parser.add_argument('--gradient_clip_val', type=float, default=0.5,help='gradient norm clipping for model parameters')

    #loss settings
    parser.add_argument('--kl_loss', action='store_true', default=False,help='whether to use kl divergence loss')

    # setting for MCMC sampling
    parser.add_argument('--buffer_size', type=int, default=10000,help='size of the buffer')
    parser.add_argument('--replace_prob', type=float, default=0.05,help='probability of replacing a sample in the buffer')
    parser.add_argument('--buffer_transform', action='store_true', help='whether to apply dataset transformations to the buffer')
    parser.add_argument('--sgld_steps', type=int, default=40,help='number of steps for MCMC sampling')
    parser.add_argument('--sgld_lr', type=float, default=100.0,help='learning rate for the step size of the MCMC sampler')
    parser.add_argument('--sgld_sigma', type=float, default=0.001,help='noise scale for the MCMC sampler')

    parser.add_argument('--clamp_grad', action='store_true', default=False,help='whether to clamp the gradient for MCMC sampling')

    # architecture settings
    parser.add_argument('--deep', action='store_true', default=False, help='whether to use deeep residual networks')
    parser.add_argument('--multiscale', action='store_true', default=False,help='whether to use multiscale architecture')
    parser.add_argument('--self-attn', action='store_true', default=False,help='whether to use self-attention')

    parser.add_argument('--filter_dim', type=int, default=64,help='number of filters in the model')
    parser.add_argument('--img_size', type=int, default=64,help='size of the image')
    parser.add_argument('--spec_norm', action='store_true', default=False,help='whether to use spectral normalization')

    # ema settings
    parser.add_argument('--use_ema', action='store_true', default=False, help='whether to use ema')
    parser.add_argument('--ema_decay', type=float, default=0.9999,help='ema decay rate')
    parser.add_argument('--ema_eval', action='store_true', default=False,help='whether to use ema for validation')
    
    args = parser.parse_args()
    
    main(args)