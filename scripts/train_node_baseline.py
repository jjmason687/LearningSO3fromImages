# Author(s): Justice Mason
# Projects: DEVS/RODEN
# Package: Training Script for DEVS/RODEN
# Date: 07/29/22

from typing import Tuple
import os
import sys
sys.path.append('.')

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision import transforms
from glob import glob
from torch.utils.data import DataLoader,random_split
from torch.utils.tensorboard import SummaryWriter
from torchdiffeq import odeint_adjoint as odeint

# User-defined function imports
from models import lie_tools
from data.dataset_classes import DEVSdataset
from data.data_utils import load_data, generate_devs_dl
from utils.math_utils import pd_matrix
from utils.train_utils import init_weights, ae_loss, dynamics_loss, data_preprocess, data_postprocess
from utils.visualization_utils import latent_eval, plot_loss, recon_eval


class EncoderNet(nn.Module):
    """
    Implementation of an convolutional encoder network for DEVS to map to the SO(3) latent space.
    
    ...
    
    """
    def __init__(self,
                in_channels: int = 3,
                obs_len: int = 2,
                latent_dim = 6) -> None:
        """
        Parameters
        ----------
        in_channels : int
            Number of channels on the input layer of the encoder network.
            
        obs_len : int, default=3
            Observation length of each input in the train/test dataset.
            
        latent_dim : int, default=9
            Latent dimension size. Chosen to be 9 to reconstruct the state (R, \Pi) \in T*SO(3).
            
        Notes
        -----
        
        """
        super(EncoderNet, self).__init__()
        self.obs_len = obs_len

        self.conv1 = nn.Conv2d(in_channels= in_channels * self.obs_len, out_channels=16, kernel_size=3)
        self.relu1 = nn.ELU()
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3)
        self.relu2 = nn.ELU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.relu3 = nn.ELU()
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.relu4 = nn.ELU()
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.bn4 =  nn.BatchNorm2d(32)
        self.flatten4 = nn.Flatten(start_dim=1)
        self.linear5 = nn.Linear(in_features=32*4*4, out_features=120)
        self.relu5 = nn.ELU()
        self.bn5 = nn.BatchNorm1d(120)
        self.linear6 = nn.Linear(in_features=120, out_features=84)
        self.relu6 = nn.ELU()
        self.linear7 = nn.Linear(in_features=84, out_features= latent_dim)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the encoding for a sequence of images.
        
        ...
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor for encoding.
        
        Notes
        -----
        
        """
        assert torch.any(x.isnan()) == False and torch.any(x.isinf()) == False
        
        batch_size, obs_len, channels, w, h = x.shape
        x_ = x.reshape(batch_size, obs_len*channels, w, h)
        
        h1 = self.relu1(self.conv1(x_))
        h2, indices2 = self.maxpool2(self.relu2(self.conv2(h1)))
        h2 = self.bn2(h2)
        h3 = self.relu3(self.conv3(h2))
        h4, indices4 =  self.maxpool4(self.relu4(self.conv4(h3)))
        h4 = self.flatten4(self.bn4(h4))
        h5 = self.bn5(self.relu5(self.linear5(h4)))
        h6 = self.relu6(self.linear6(h5))
        x_enc = self.linear7(h6)
        
        return x_enc, indices2, indices4

    
class DecoderNet(nn.Module):
    """
    Implementation of an deconvolutional decoder network for DEVS to map from the desired latent space to image space.
    
    ...
    
    """
    def __init__(self,
                in_channels: int = 6) -> None:
        """
        Parameters
        ----------
        in_channels : int
            Number of channels on the input layer of the encoder network.
            
        Notes
        -----
        
        """
        super(DecoderNet, self).__init__()
        
        self.linear7 = nn.Linear(in_features=in_channels, out_features=84)
        self.relu6 = nn.ELU()
        self.linear6 = nn.Linear(in_features=84, out_features=120)
        self.bn5 = nn.BatchNorm1d(120)
        self.relu5 = nn.ELU()
        self.linear5 = nn.Linear(in_features=120, out_features=32*4*4)
        self.unflatten4 = nn.Unflatten(dim=1, unflattened_size=(32, 4, 4))
        self.bn4 = nn.BatchNorm2d(32)
        self.maxunpool4 = nn.MaxUnpool2d(2, 2)
        self.relu4 = nn.ELU()
        self.conv4 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3)
        self.relu3 = nn.ELU()
        self.conv3 = nn.ConvTranspose2d(32, 16, 3)
        self.bn2 = nn.BatchNorm2d(16)
        self.maxunpool2 = nn.MaxUnpool2d(2, 2)
        self.relu2 = nn.ELU()
        self.conv2 = nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=3)
        self.relu1 = nn.ELU()
        self.conv1 = nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=3)
                
    def forward(self, x: torch.Tensor, indices2, indices4) -> torch.Tensor:
        """
        Compute the encoding for a sequence of images.
        
        ...
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor for encoding.
            
        Returns
        -------
        x_dec : torch.Tensor
            Encoded tensor.
        
        Notes
        -----
        
        """
        assert torch.any(x.isnan()) == False and torch.any(x.isinf()) == False
        
        x_ = x.reshape(-1, 6)
        
        h7 = self.linear7(x_)
        h6 = self.linear6(self.relu6(h7))
        h5 = self.linear5(self.relu5(self.bn5(h6)))
        h4 = self.bn4(self.unflatten4(h5))
        
        h4 = self.conv4(self.relu4(self.maxunpool4(h4, indices4)))
        h3 = self.conv3(self.relu3(h4))
        h2 = self.bn2(h3)
        h2 = self.conv2(self.relu2(self.maxunpool2(h2, indices2)))
        x_dec = self.conv1(self.relu1(h2))
        return x_dec

    
class Baseline(torch.nn.Module):
    """
    Model class for baseline training.
    
    ...
    
    """
    def __init__(self, encoder: torch.nn.Module, decoder: torch.nn.Module, baseline: torch.nn.Module, mlp: torch.nn.Module = None):
        super(Baseline, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.model = baseline
        self.mlp = mlp
        self.indices2 = None
        self.indices4 = None
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encoder method.
        
        ...
        
        Parameters
        ----------
        x : torch.Tensor
            Input observation.
            
        Returns
        -------
        z_enc : torch.Tensor
            Encoded latent state.
            
        Notes
        -----
        
        """
        z_enc, indices2, indices4 = self.encoder(x)
        
        self.indices2 = indices2
        self.indices4 = indices4
        
        return z_enc
    
    def decode(self, z: torch.Tensor):
        """
        Decoder method.
        
        ...
        
        Parameters
        ----------
        z : torch.Tensor
            Latent representation for input observation.
            
        Returns
        -------
        z_enc : torch.Tensor
            Encoded latent state.
            
        Notes
        -----
        
        """
        if self.indices2.shape[0] != z.shape[0]:
            batch_size = int(z.shape[0]/self.indices2.shape[0])
            indices2 = self.indices2.repeat([batch_size, 1, 1, 1])
        else:
            indices2 = self.indices2
        
        if self.indices4.shape[0] != z.shape[0]:
            batch_size = int(z.shape[0]/self.indices4.shape[0])
            indices4 = self.indices4.repeat([batch_size, 1, 1, 1])
        else:
            indices4 = self.indices4
            
        return self.decoder(z, indices2, indices4)
    
    def state_rollout(self, z_seq: torch.Tensor, dt: float = 1e-3, seq_len: int = None) -> torch.Tensor:
        """
        Latent state prediction method for both the LSTM and NeuralODE method.
        
        ...
        
        Parameters
        ----------
        z0 : torch.Tensor
            Input latent state.
            
        seq_len : int, default=None
            Prediction sequence lenght.
            
        Returns
        -------
        
        Notes
        -----
        
        """
        if not seq_len:
            output_, _ = self.model(z_seq) # bs, L, 100
            output = self.mlp(output_)
            
        else:
            z0 = z_seq[:, 0, ...]
            t = torch.linspace(0., seq_len * dt, seq_len, device=z0.device)
            output = odeint(func=self.model, y0=z0, t=t, method='rk4', options=dict(step_size=dt))
            
        return output
    
    def forward(self, x: torch.Tensor,
                obs_len: int = 1,
                seq_len: int = None, 
                dt: float = 1e-3) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward method for baseline model.
        
        ...
        
        Parameters
        ----------
        x : torch.Tensor
            Input image sequence for model.
            
        obs_len : int, default=1
            Desired observation length, default to 1 for A2 architecture.
            
        seq_len : int, default=3
            Desired sequence length for training. There is a minimum sequence length of 3.
            
        dt : float, default=1e-3
            Timestep used for forward rollout.
            
        Returns
        -------
        xhat_recon : torch.Tensor
            Image sequence reconstructed using auto-encoder.
        
        xhat_pred : torch. Tensor
            Image sequence reconstructed using dynamics prediction.
        
        z_enc : torch.Tensor
            Encoded latent state given by encoder side of network.
        
        z_pred : torch.Tensor
            Predicted latent state given by learned dynamics.
        
        Notes
        -----
        
        """
        assert torch.any(x.isnan()) == False and torch.any(x.isinf()) == False
        bs, slen, _, _, _ = x.shape
        
        # encode
        x_obs = data_preprocess(data=x, observation_len=obs_len)
        z_enc = self.encode(x=x_obs) # pbs, 6
        z_rs = z_enc.reshape(bs, -1, 6)
        
        # predict
        z_pred = self.state_rollout(z_seq=z_rs, seq_len=seq_len)
        z_pred_rs = z_pred.reshape(-1, 6)
        
        # decode
        # import pdb; pdb.set_trace()
        xhat_dec = self.decode(z=z_enc)
        xhat_recon = data_postprocess(data=xhat_dec, batch_size=bs, seq_len=slen+1)
        
        xhat_pred_dec = self.decode(z=z_pred_rs)
        xhat_pred = data_postprocess(data=xhat_pred_dec, batch_size=bs, seq_len=slen+1)
        
        return xhat_recon, xhat_pred, z_enc, z_pred_rs
    
class Neural_ODE_Baseline(torch.nn.Module):
    """
    """
    def __init__(self, in_dim: int = 6, out_dim: int = 6, hidden_dim: int = 100):
        super(Neural_ODE_Baseline, self).__init__()
        
        self.relu = torch.nn.ELU()
        self.lin1 = torch.nn.Linear(in_features=in_dim, out_features=hidden_dim)
        self.lin2 = torch.nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.lin3 = torch.nn.Linear(in_features=hidden_dim, out_features=out_dim)
        
    def forward(self, t: torch.Tensor = None, x: torch.Tensor = None) -> torch.Tensor:
        """
        """
        h1 = self.relu(self.lin1(x))
        h2 = self.relu(self.lin2(h1))
        h3 = self.lin3(h2)
        
        return h3
    
    
def save_checkpoint(model,
                    stats: list,
                    optimizer,
                    epoch: int ,
                    loss: float,
                    path: str):
    """
    Saves checkpoint after every number of epochs.
    
    ...
    
    Notes
    -----
    
    """
    if not os.path.exists(path):
        print('\n Directory not found at "{}"; creating directory ... \n'.format(path))
        os.makedirs(path)
        
    ckpt_path = path + '/checkpoint-{:06}.pth'.format(epoch)
    checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'stats': stats,
            }
    torch.save(checkpoint, ckpt_path)

def latest_checkpoint(model,
                      optimizer,
                      checkpointdir):
    """
    Function to load the latest checkpoint of model and optimizer in given checkpoint directory.
    
    ...
    
    Parameters
    ----------
    model : torch.nn.Module
        Untrained model
    
    optimizer : torch.nn.optim
        Optimizer
    
    checkpointdir : str
        Checkpoint directory
        
    Returns
    -------
    model : torch.nn.Module
        Untrained model
    
    optimizer : torch.nn.optim
        Optimizer
    
    checkpointdir : str
        Checkpoint directory
    
    stats : list
        List of statistics for analysis.
        
    start_epoch : int
        Starting epoch for training.

    Notes
    -----
    
    """
    # device = model.device
    
    if not os.path.exists(checkpointdir):
        print('\n Log directory does not exist. Making it ... \n')
        os.makedirs(checkpointdir)
        
    filenames = glob(checkpointdir + '*.pth')
    
    if not filenames:
        latest = 'not-found'
    else:
        latest = sorted(filenames)[-1]
    
    model, optimizer, stats, start_epoch = load_checkpoint(model, optimizer, latest)
    
    return model, optimizer, stats, start_epoch

def load_checkpoint(model,
                    optimizer,
                    filename):
    """
    Function to load checkpoints.
    
    ...
    
    Parameters
    ----------
    model
        Neural network work model used for training.
        
    optimizer
        Optimizer used during training.
        
    filename : str
        String that contains the filename of the checkpoint.
        
    Returns
    -------
    model
        Neural network work model used for training with updated states.
        
    optimizer
         Optimizer used during training with updated states.
         
    start_epoch : int
        Starting epoch for training.
        
    Notes
    -----
    
    """
    start_epoch = 0
    stats = []
    
    if os.path.isfile(filename):
        print("\n Loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename, map_location='cpu')
        
        start_epoch = checkpoint['epoch']
        print('\n Starting at epoch {} ... \n'.format(start_epoch))
        
        stats = checkpoint['stats']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(args.device)
        print("\n Loaded checkpoint '{}' (epoch {}) ... \n".format(filename, checkpoint['epoch']))
    else:
        print("\n No checkpoint found at '{}' ... \n".format(filename))
    
    return model, optimizer, stats, start_epoch

def baseline_loss_fcn(x: torch.Tensor,
              xhat_recon: torch.Tensor,
              xhat_pred: torch.Tensor,
              z_pred: torch.Tensor,
              z_enc: torch.Tensor, 
              gamma: list = [1., 1., 1.]) -> torch.Tensor:
    """
    Loss based on LagNetViP including conservation of energy regularizer.
    
    ...
    
    
    Notes
    -----
    
    """
    loss_ae = ae_loss(x=x, xhat_recon=xhat_recon)
    loss_dyn = dynamics_loss(x=x, xhat_pred=xhat_pred)
    loss_latent = ((z_pred - z_enc)**2).sum()
    
    loss_total = (gamma[0] * loss_ae) + (gamma[1] * loss_dyn) + (gamma[2] * loss_latent)
    return loss_total, loss_ae, loss_dyn, loss_latent

def train_epoch(args, model, dataloader, optimizer):
    """
    """
    model.train()
    train_loss = 0.0
    
    for i, x in enumerate(dataloader): 
        x = x.to(args.device)
        xhat_recon, xhat_pred, z_enc, z_pred = model(x.float(), dt=args.dt, seq_len=args.seq_len)
        # import pdb; pdb.set_trace()
        train_loss, train_ae_loss, train_dyn_loss, train_latent_loss = baseline_loss_fcn(x=x.float(),\
                                                                                xhat_recon=xhat_recon.float(),\
                                                                                xhat_pred=xhat_pred.float(),\
                                                                                z_pred=z_pred.float(),\
                                                                                z_enc=z_enc.float(),\
                                                                                gamma=args.gamma)

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        train_loss.backward()
        optimizer.step()
        
    return train_loss, train_ae_loss, train_dyn_loss, train_latent_loss

def val_epoch(args, model, dataloader, optimizer):
    """
    """
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for i, x in enumerate(dataloader): 
            x = x.to(args.device)
            xhat_recon, xhat_pred, z_enc, z_pred = model(x.float(), dt=args.dt, seq_len=args.seq_len)
            val_loss, val_ae_loss, val_dyn_loss, val_latent_loss = baseline_loss_fcn(x=x.float(),\
                                                                                    xhat_recon=xhat_recon.float(),\
                                                                                    xhat_pred=xhat_pred.float(),\
                                                                                    z_pred=z_pred.float(),\
                                                                                    z_enc=z_enc.float())
    
    return val_loss, val_ae_loss, val_dyn_loss, val_latent_loss

def train_loop(args, writer, model, optimizer, train_dl, val_dl, stats=None):
    """
    """
    if not stats:
        stats = {'train loss': [], 'train ae loss': [], 'train dyn loss': [], 'train latent loss': [],\
             'val loss': [], 'val ae loss': [], 'val dyn loss': [], 'val latent loss': []}
        
    for epoch in range(args.n_epochs):
        train_loss, train_ae_loss, train_dyn_loss, train_latent_loss = train_epoch(args=args, model=model, dataloader=train_dl, optimizer=optimizer)
        val_loss, val_ae_loss, val_dyn_loss, val_latent_loss = val_epoch(args=args, model=model, dataloader=val_dl, optimizer=optimizer)

        writer.add_scalar('Loss/Train/Total', train_loss, epoch)
        writer.add_scalar('Loss/Train/Autoencoder', train_ae_loss, epoch)
        writer.add_scalar('Loss/Train/Dynamics', train_dyn_loss, epoch)
        writer.add_scalar('Loss/Train/Latent', train_latent_loss, epoch)
        
        stats['train loss'].append(train_loss.detach())
        stats['train ae loss'].append(train_ae_loss.detach())
        stats['train dyn loss'].append(train_dyn_loss.detach())
        stats['train latent loss'].append(train_latent_loss.detach())
        
        writer.add_scalar('Loss/Val/Total', val_loss, epoch)
        writer.add_scalar('Loss/Val/Autoencoder', val_ae_loss, epoch)
        writer.add_scalar('Loss/Val/Dynamics', val_dyn_loss, epoch)
        writer.add_scalar('Loss/Val/Latent', val_latent_loss, epoch)
        
        stats['val loss'].append(val_loss.detach())
        stats['val ae loss'].append(val_ae_loss.detach())
        stats['val dyn loss'].append(val_dyn_loss.detach())
        stats['val latent loss'].append(val_latent_loss.detach())

        if epoch % args.eval_freq == (args.eval_freq - 1):
            print("\n Epoch: {}, Training Loss: {:.4e}, Validation Loss: {:.4e}".format(epoch + 1, train_loss.item(), val_loss.item()))
            axes_pred = plot_pred_output(args, model, val_dl)
            writer.add_figure('ImageRecon/Pred', axes_pred.figure, epoch)
            
            axes_recon = plot_recon_output(args, model, val_dl)
            writer.add_figure('ImageRecon/Recon', axes_recon.figure, epoch)
            
        if epoch % args.checkpoint_freq == (args.checkpoint_freq - 1):
            print("\n Saving Checkpoint for Epoch {} ... \n".format(epoch))
            save_checkpoint(model, stats, optimizer, epoch, train_loss.item(), args.checkpoint_dir)
            
    return model, stats


def plot_pred_output(args, model, test_dataloader):
    plt.figure(figsize=(16,4.5))
    
    test_data = next(iter(test_dataloader))
    x = test_data[0, ...].unsqueeze(0).to(args.device)
    
    model.eval()
    with torch.no_grad():
        _, x_pred, _, _ = model(x.float(), dt=args.dt, seq_len=args.seq_len)
    
    seq_len = x.shape[1]
    for i in range(seq_len):
        ax = plt.subplot(2, seq_len, i + 1)
        img = x[:, i, ...]
        plt.imshow(img.cpu().squeeze().permute(1, 2, 0).numpy()) #, cmap='gist_gray') plt.
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        if i == seq_len//2:
            ax.set_title('Original Images')
            
        ax = plt.subplot(2, seq_len, i + 1 + seq_len)
        img_pred = x_pred[:, i, ...]
        plt.imshow(img_pred.cpu().squeeze().permute(1, 2, 0).numpy()) #, cmap='gist_gray')   plt.
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)  
        
        if i == seq_len//2:
            ax.set_title('Predicted Reconstructed Images')
    
    plt.show()
    return ax

def plot_recon_output(args, model, test_dataloader):
    plt.figure(figsize=(16,4.5))
    
    test_data = next(iter(test_dataloader))
    x = test_data[0, ...].unsqueeze(0).to(args.device)
    
    model.eval()
    with torch.no_grad():
        x_recon, _, _, _ = model(x.float(), dt=args.dt, seq_len=args.seq_len)
    
    seq_len = x.shape[1]
    for i in range(seq_len):
        ax = plt.subplot(2, seq_len, i + 1)
        img = x[:, i, ...]
        plt.imshow(img.cpu().squeeze().permute(1, 2, 0).numpy()) #, cmap='gist_gray') plt.
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        if i == seq_len//2:
            ax.set_title('Original Images')
            
        ax = plt.subplot(2, seq_len, i + 1 + seq_len)
        img_recon = x_recon[:, i, ...]
        plt.imshow(img_recon.cpu().squeeze().permute(1, 2, 0).numpy()) #, cmap='gist_gray')   plt.
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)  
        
        if i == seq_len//2:
            ax.set_title('AE Reconstructed Images')
    
    plt.show()
    return ax


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
                        default=256, type=int, help='training batch size')
    parser.add_argument('--learning_rate',
                        action='store',
                        default=1e-3,
                        type=float, dest='lr', help='learning rate for training autoencoder')
    parser.add_argument('--weight_decay',
                        action='store',
                        default=1e-5,
                        type=float, dest='wd', help='weight decay for training the model')
    parser.add_argument('--eval_freq',
                        action='store',
                        default=10,
                        type=int, help='model evaluation during training')
    parser.add_argument('--checkpoint_freq',
                        action='store',
                        default=10,
                        type=int, help='checkpoint frequency during training')
    parser.add_argument('--obs_len',
                        action='store',
                        default=1,
                        type=int, help='sequence length of input for encoder, not used for SO(3) net')
    parser.add_argument('--seq_len', 
                        action='store',
                        default=10,
                        type=int, help='prediction sequence length for dynamics')
    parser.add_argument('--time_step',
                        action='store',
                        default=1e-3,
                        type=float, dest='dt', help='time interval between states for training data')
    parser.add_argument('--loss_gamma',
                        action='store',
                        nargs="+",
                        default=[1., 1., 1.], type=float, dest='gamma', help='weights for recon, dyn, and latent losses.')
    
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
    torch.manual_seed(args.seed)
    device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")
    args.device = device
    
    #import pdb; pdb.set_trace()
    writer = SummaryWriter(log_dir=args.log_dir) # './runs/lstm/non_uniform_cube/' 

    print('\n Loading dataloaders ... \n')
    traindl, testdl, valdl = generate_devs_dl(args=args, data_dir=args.data_dir)

    print('\n Loading model ... \n')
    encoder = EncoderNet(obs_len=1)
    decoder = DecoderNet()
    neural_ode = Neural_ODE_Baseline()

    baseline = Baseline(encoder=encoder, decoder=decoder, baseline=neural_ode)
    baseline.apply(init_weights)
    
    num_params = 0
    for p in baseline.model.parameters():
        num_params += np.prod(p.shape)

    print('\n Model has {} number of parameters... \n'.format(num_params))

    print('\n Initializing optimizer and loss function... \n')
    bl_optim = optim.Adam(params=[{'params': baseline.parameters()}], lr=args.lr, weight_decay=args.wd)

    decayRate = 0.01 ** (4e-5)
    bl_lr_sched = torch.optim.lr_scheduler.ExponentialLR(optimizer=bl_optim, gamma=decayRate)
    bl_lf = baseline_loss_fcn

    print('\n Checking logs for model checkpoint ... \n')
    baseline, bl_optim, bl_stats, start_epoch = latest_checkpoint(model=baseline, optimizer=bl_optim, checkpointdir=args.checkpoint_dir)
    
    print('\n Training on single GPU: {} ... \n'.format(args.device))
    baseline = baseline.to(args.device)
    
    print('\n Reproducibility for CNNs...\n')
    torch.backends.cudnn.benchmark = True
    
    print('\n Training model ... \n')
    model, stats = train_loop(args=args,\
                              writer=writer,\
                              model=baseline,\
                              optimizer=bl_optim,\
                              train_dl=traindl,\
                              val_dl=valdl,\
                              stats=bl_stats)
    
    print('\n Done training model ... \n')
    return model, stats

if __name__ == "__main__":
    
    torch.cuda.empty_cache()
    args = get_args()
    torch.manual_seed(args.seed)
    model, stats = main()
    
    print('\n JOB DONE! \n')
    