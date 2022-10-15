# Author(s): Justice Mason
# Projects: DEVS/RODEN
# Package: Training Script for DEVS/RODEN
# Date: 12/17/21

import os
import sys
sys.path.append('.')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse
import torch, torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from mlflow import log_metric, log_param, log_artifacts
from glob import glob
from time import time, strftime

# User-defined function imports
# import helper
from models import lie_tools
from models.devs_models import DEVS, DEVS_SO3
from models.autoencoder import EncoderNet, DecoderNet, EncoderNetNP, DecoderNetNP, EncoderNetSO3, DecoderNetSO3
from data.dataset_classes import DEVSdataset
from data.data_utils import load_data, generate_devs_dl
from utils.math_utils import pd_matrix
from utils.train_utils import latest_checkpoint, load_checkpoint, train, devs_loss, devs_loss_LRAN, devs_loss_plus_energy, init_weights
from utils.physics_utils import true_dyn_model
from utils.visualization_utils import latent_eval, plot_loss, recon_eval

def get_args():
    """
    Get arguments for argparse instance.
    
    ...
    
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',
                        action='store',
                        default=0,
                        type=int, help='int : random seed')
    parser.add_argument('--tune',
                        action='store_false',
                        default=False,
                        help='Ray Tune flag during training')
    parser.add_argument('--single_gpu',
                        action='store_true',
                        help='flag for single-gpu training')
    parser.add_argument('--use_cuda',
                        action='store_true',
                        help='flag for gpu training')
    parser.add_argument('--gpu',
                        action='store',
                        default='cuda',
                        type=str, help='gpu used for training the model')
    parser.add_argument('--data_dir',
                        action='store',
                        type=str,
                        help='data directory')
    parser.add_argument('--log_dir',
                        action='store',
                        type=str,
                        help='log directory')
    parser.add_argument('--checkpoint_dir',
                        action='store',
                        type=str,
                        help='checkpoint directory')
    parser.add_argument('--test_split',
                        action='store',
                        default=0.2,
                        type=float, help='ratio of raw data in test set')
    parser.add_argument('--val_split',
                        action='store',
                        default=0.1,
                        type=float, help='ratio of non-test data used in validation set')
    parser.add_argument('--n_epochs',
                        action='store',
                        default=100,
                        type=int, help='number of epochs during training')
    parser.add_argument('--batch_size',
                        action='store',
                        dest='bs',
                        default=32, type=int, help='training batch size')
    parser.add_argument('--learning_rate_ae',
                        action='store',
                        default=1e-3,
                        type=float, dest='lr_ae', help='learning rate for training autoencoder')
    parser.add_argument('--learning_rate_dyn',
                        action='store',
                        default=1e-3,
                        type=float, dest='lr_dyn', help='learning rate for training the dynamics model')
    parser.add_argument('--weight_decay',
                        action='store',
                        default=1e-5,
                        type=float, dest='wd', help='weight decay for training the model')
    parser.add_argument('--eval_freq',
                        action='store',
                        default=100,
                        type=int, help='model evaluation during training')
    parser.add_argument('--log_freq',
                        action='store',
                        default=100,
                        type=int, help='checkpoint frequency during training')
    parser.add_argument('--obs_len',
                        action='store',
                        default=1,
                        type=int, help='sequence length of input for encoder, not used for SO(3) net')
    parser.add_argument('--seq_len', 
                        action='store',
                        default=5,
                        type=int, help='prediction sequence length for dynamics')
    parser.add_argument('--time_step',
                        action='store',
                        default=1e-2,
                        type=float, dest='dt', help='time interval between states for training data')
    parser.add_argument('--loss_gamma',
                        action='store',
                        nargs="+",
                        default=[1., 1., 1., 1., 1., 1.], type=float, dest='gamma', help='weights for recon, dyn, and latent losses.')
    
    args = parser.parse_args()
    return args

def main():
    """
    Main function that initiates an entire training cycle.
    
    ...
    
    Returns
    -------
    model : torch.nn.Module
        trained model
        
    stats : dict
        Dictionary of training statistics (losses, etc.)
        
    Returns
    -------
    model : torch.nn.Module
        Trained model to be returned for analysis
    Notes
    -----
    
    """
    args = get_args()
    torch.set_num_threads(4)
    torch.get_num_threads()
    torch.manual_seed(args.seed)
    device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")

    print('\n Loading dataloaders ... \n')
    devs_traindl, devs_testdl, devs_valdl = generate_devs_dl(args=args, data_dir=args.data_dir)

    print('\n Loading model ... \n')
    devs_encoder = EncoderNetSO3(obs_len=args.obs_len)
    devs_decoder = DecoderNetSO3(in_channels=6)
    devs = DEVS_SO3(device=device, encoder=devs_encoder, decoder=devs_decoder)
    
    devs.apply(init_weights)
    
    print('\n Initializing optimizer and loss function... \n')
    devs_optim = optim.Adam(params=[{'params': devs.parameters()},\
                                    {'params': devs.moi_diag, 'lr': args.lr_dyn, 'weight_decay': args.wd},\
                                    {'params': devs.moi_off_diag, 'lr': args.lr_dyn, 'weight_decay': args.wd}],\
                            lr=args.lr_ae, weight_decay=args.wd)
    
    decayRate = 0.01 ** (4e-5)
    devs_lr_sched = torch.optim.lr_scheduler.ExponentialLR(optimizer=devs_optim, gamma=decayRate)
    devs_lf = devs_loss_plus_energy
    
    print('\n Checking logs for model checkpoint ... \n')
    devs, devs_optim, devs_stats, start_epoch = latest_checkpoint(model=devs, optimizer=devs_optim, checkpointdir=args.checkpoint_dir)
        
    if args.single_gpu:
        print('\n Training on single GPU: {} ... \n'.format(args.gpu))
        devs = devs.to(device)
        
    else:
        if torch.cuda.device_count() > 1:
            print('\n Training on multiple GPUS: {}, {}, {} ... \n'.format(0, 1, 2))
            devs = nn.DataParallel(devs, device_ids=[0, 1, 2])
            devs.to(f'cuda:{devs.device_ids[0]}')
    
    torch.backends.cudnn.benchmark = True
    
    print('\n Training model ... \n')
    model, stats = train(args=args,\
                         traindata_dl=devs_traindl,\
                         valdata_dl=devs_testdl,\
                         optim=devs_optim,\
                         lr_scheduler=devs_lr_sched,\
                         stats=devs_stats, model=devs, \
                         loss_fcn=devs_lf,\
                         start_epoch=start_epoch,\
                         num_epoch=args.n_epochs,\
                         eval_freq=args.eval_freq,\
                         log_freq=args.log_freq)
    
    print('\n Done training model ... \n')
    return model, stats

if __name__ == "__main__":
    
    torch.cuda.empty_cache()
    args = get_args()
    torch.manual_seed(args.seed)
    model, stats = main()
    
    if not args.single_gpu:
        model = model.module
    
    est_moi_inv = pd_matrix(diag=model.moi_diag, off_diag=model.moi_off_diag)
    print('\n Estimated Inverse of MOI: {} \n', est_moi_inv)
    print('\n JOB DONE! \n')
    