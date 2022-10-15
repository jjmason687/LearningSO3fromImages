# Author(s): Justice Mason
# Project: DEVS/RODEN 
# Package: Training Utilities
# Date: 12/04/21

from __future__ import print_function
import os
import sys
# sys.path.append('.')

import numpy as np
import torch
import torch.nn as nn

#from ray import tune
#from ray.tune import CLIReporter
#from ray.tune.schedulers import ASHAScheduler
from glob import glob

from utils.math_utils import pd_matrix, distance_so3, rotate 
from data.data_utils import generate_devs_dl
from models.lie_tools import s2s1rodrigues
from utils.visualization_utils import  plot_recon_output, plot_pred_output

"""

TRAINING FUNCTION


"""
def train_epoch(args, model, dataloader, optimizer, loss_fcn, lr_scheduler):
    """
    """
    model.train()
    train_loss = 0.0
    
    for i, x in enumerate(dataloader):
        x = x.to(args.device)
        xhat_recon, xhat_pred, z_enc, z_pred, pi_enc, pi_pred  = model(x.float(), seq_len=args.seq_len, dt=args.dt)
        
        train_loss, train_ae_loss, train_dyn_loss, train_latent_R_loss, train_latent_pi_loss, train_energy = loss_fcn(x=x.float(),\
                                                                                                                                xhat_recon=xhat_recon.float(),\
                                                                                                                                xhat_pred=xhat_pred.float(),\
                                                                                                                                z_pred=z_pred.float(),\
                                                                                                                                z_enc=z_enc.float(),\
                                                                                                                                pi_enc=pi_enc.float(),\
                                                                                                                                pi_pred=pi_pred.float(),\
                                                                                                                                model=model,\
                                                                                                                                seed=args.seed,\
                                                                                                                                gamma=args.gamma)
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        train_loss.backward()
        
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
        optimizer.step()
        lr_scheduler.step()
            
    return train_loss, train_ae_loss, train_dyn_loss, train_latent_R_loss, train_latent_pi_loss, train_energy

def val_epoch(args, model, dataloader, loss_fcn, optimizer):
    """
    """
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for i, x in enumerate(dataloader):
            x = x.to(args.device)
            xhat_recon, xhat_pred, z_enc, z_pred, pi_enc, pi_pred  = model(x.float(), seq_len=args.seq_len, dt=args.dt)
        
            val_loss, val_ae_loss, val_dyn_loss, val_latent_R_loss, val_latent_pi_loss, val_energy = loss_fcn(x=x.float(),\
                                                                                                xhat_recon=xhat_recon.float(),\
                                                                                                xhat_pred=xhat_pred.float(),\
                                                                                                z_pred=z_pred.float(),\
                                                                                                z_enc=z_enc.float(),\
                                                                                                pi_enc=pi_enc.float(),\
                                                                                                pi_pred=pi_pred.float(),\
                                                                                                model=model,\
                                                                                                seed=args.seed,\
                                                                                                gamma=args.gamma)
    
    return val_loss, val_ae_loss, val_dyn_loss, val_latent_R_loss, val_latent_pi_loss, val_energy

def train_loop(args, writer, model, optimizer, train_dl, val_dl, lr_scheduler, loss_fcn, start_epoch=0, stats=None):
    """
    """
    if not stats:
        stats = {'train loss': [], 'train ae loss': [],\
                 'train dyn loss': [], 'train latent R loss': [],\
                 'train latent pi loss': [], 'train energy loss': [],\
             'val loss': [], 'val ae loss': [], 'val dyn loss': [],\
                 'val latent R loss': [], 'val latent pi loss': [], 'val energy loss': []}
        
    for epoch in range(start_epoch, start_epoch + args.n_epochs):
        train_loss, train_ae_loss, train_dyn_loss, train_latent_R_loss, train_latent_pi_loss, train_energy = train_epoch(args=args,\
                                                                                                                         model=model,\
                                                                                                                         dataloader=train_dl,\
                                                                                                                         lr_scheduler=lr_scheduler,\
                                                                                                                         optimizer=optimizer, loss_fcn=loss_fcn)
        
        val_loss, val_ae_loss, val_dyn_loss, val_latent_R_loss, val_latent_pi_loss, val_energy = val_epoch(args=args,\
                                                                                                           model=model,\
                                                                                                           dataloader=val_dl,\
                                                                                                           optimizer=optimizer,\
                                                                                                           loss_fcn=loss_fcn)

        writer.add_scalar('Loss/Train/Total', train_loss, epoch)
        writer.add_scalar('Loss/Train/Autoencoder', train_ae_loss, epoch)
        writer.add_scalar('Loss/Train/Dynamics', train_dyn_loss, epoch)
        writer.add_scalar('Loss/Train/Latent-R', train_latent_R_loss, epoch)
        writer.add_scalar('Loss/Train/Latent-pi', train_latent_pi_loss, epoch)
        writer.add_scalar('Loss/Train/Energy', train_energy, epoch)
        
        stats['train loss'].append(train_loss.detach())
        stats['train ae loss'].append(train_ae_loss.detach())
        stats['train dyn loss'].append(train_dyn_loss.detach())
        stats['train latent R loss'].append(train_latent_R_loss.detach())
        stats['train latent pi loss'].append(train_latent_pi_loss.detach())
        stats['train energy loss'].append(train_energy.detach())
        
        writer.add_scalar('Loss/Val/Total', val_loss, epoch)
        writer.add_scalar('Loss/Val/Autoencoder', val_ae_loss, epoch)
        writer.add_scalar('Loss/Val/Dynamics', val_dyn_loss, epoch)
        writer.add_scalar('Loss/Val/Latent-R', val_latent_R_loss, epoch)
        writer.add_scalar('Loss/Val/Latent-pi', val_latent_pi_loss, epoch)
        writer.add_scalar('Loss/Val/Energy', val_energy, epoch)
        
        stats['val loss'].append(val_loss.detach())
        stats['val ae loss'].append(val_ae_loss.detach())
        stats['val dyn loss'].append(val_dyn_loss.detach())
        stats['val latent R loss'].append(val_latent_R_loss.detach())
        stats['val latent pi loss'].append(val_latent_pi_loss.detach())
        stats['val energy loss'].append(val_energy.detach())
        

        if epoch % args.log_freq == (args.log_freq - 1):
            print("\n Epoch: {}, Training Loss: {:.4e}, Validation Loss: {:.4e}".format(epoch + 1, train_loss.item(), val_loss.item()))
            axes_pred = plot_pred_output(args, model, val_dl)
            writer.add_figure('ImageRecon/Pred', axes_pred.figure, epoch)
            
            axes_recon = plot_recon_output(args, model, val_dl)
            writer.add_figure('ImageRecon/Recon', axes_recon.figure, epoch)
            
        if epoch % args.checkpoint_freq == (args.checkpoint_freq - 1):
            print("\n Saving Checkpoint for Epoch {} ... \n".format(epoch))
            save_checkpoint(model, stats, optimizer, epoch, train_loss.item(), args.checkpoint_dir)
            
    return model, stats

def train(args, 
          traindata_dl,
          valdata_dl,
          optim,
          lr_scheduler,
          stats: list,
          model: nn.Module,
          loss_fcn: nn.Module,
          start_epoch: int = 0,
          num_epoch: int = 1000,
          log_freq: int = 200,
          eval_freq: int = 200):
    
    """
    Function that runs training loop.
    
    ...
    
    Parameters
    ----------
    traindata_dl
        Training dataset dataloader.
        
    valdata_dl
        Validation dataset dataloader.
    
    optim
        Optimizer for training model.
    
    lr_scheduler
        Learning rate scheduler.
        
    stats : list
        List of important statistics (e.g. loss, etc.) from previously save checkpoints.
        
    model : torch.nn.Module
        Model to be trained
        
    loss_fcn : torch.nn.Module
        Loss function for training model
        
    start_epoch : int, default=0
        Starting epoch given by checkpoint or defaulted to 0
        
    num_epoch : int, default=1000
        Number of epochs for training
        
    log_freq : int, default=200,
        Checkpoint logging frequency
        
    eval_freq : int, default=200
        Evaluation frequency on validation dataset
        
    Returns
    -------
    model : torch.nn.Module
        Trained model
    
    stats : dict
        Complete list of statistics
        
    Notes
    -----
    
    """
    device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")
    dt = args.dt
    sl = args.seq_len
    bs = args.bs
    
    if not stats:
        stats = {'train loss': [], 'train ae loss': [], 'train dyn loss': [], 'train latent R loss': [], 'train latent pi loss': [], 'train energy loss': [],\
             'val loss': [], 'val ae loss': [], 'val dyn loss': [], 'val latent R loss': [], 'val latent pi loss': [], 'val energy loss': []}
    
    for epoch in range(start_epoch, start_epoch + num_epoch):
        train_loss = 0.0
        train_running_loss = 0.0
        train_steps = 0
        for i, data in enumerate(traindata_dl):
            # print('iteration number: {}'.format(i))
            x_img = data.to(device)
            
            xhat_recon, xhat_pred, z_enc, z_pred, pi_enc, pi_pred  = model(x_img.float(), seq_len=sl, dt=dt)
            train_loss, train_ae_loss, train_dyn_loss, train_latent_R_loss, train_energy = loss_fcn(x=x_img.float(),\
                                                                                                                                        xhat_recon=xhat_recon.float(),\
                                                                                                                                        xhat_pred=xhat_pred.float(),\
                                                                                                                                        z_pred=z_pred.float(),\
                                                                                                                                        z_enc=z_enc.float(),\
                                                                                                                                        pi_enc=pi_enc.float(),\
                                                                                                                                        pi_pred=pi_pred.float(),\
                                                                                                                                        model=model,\
                                                                                                                                        seed=args.seed,\
                                                                                                                                        gamma=args.gamma)
            
            optim.zero_grad(set_to_none=True)
            train_loss.backward()
            optim.step()
            lr_scheduler.step()
            
            stats['train loss'].append(train_loss.detach())
            stats['train ae loss'].append(train_ae_loss.detach())
            stats['train dyn loss'].append(train_dyn_loss.detach())
            stats['train latent R loss'].append(train_latent_R_loss.detach())
            stats['train energy loss'].append(train_energy.detach())
            
            train_running_loss += train_loss.detach()
            train_steps += 1
            if i % args.eval_freq == (args.eval_freq - 1):  
                print("[%d, %5d] running train loss: %.3f" % (epoch + 1, i + 1,
                                                train_running_loss / train_steps))
                train_running_loss = 0.0
        
        val_running_loss = 0.0
        val_steps = 0    
        with torch.no_grad():
            for i, data_val in enumerate(valdata_dl):
                y_img = y.to(device)
            
                yhat_recon, yhat_pred, zy_enc, zy_pred, piy_enc, piy_pred = model(y_img.float(), seq_len=sl, dt=dt)
                val_loss, val_ae_loss, val_dyn_loss, val_latent_R_loss, val_energy = loss_fcn(x=y_img.float(),\
                                                                                                                                xhat_recon=yhat_recon.float(),\
                                                                                                                                xhat_pred=yhat_pred.float(),\
                                                                                                                                z_pred=zy_pred.float(),\
                                                                                                                                z_enc=zy_enc.float(),\
                                                                                                                                pi_enc=piy_enc,\
                                                                                                                                pi_pred=piy_pred,\
                                                                                                                                model=model,\
                                                                                                                                seed=args.seed,\
                                                                                                                                gamma=args.gamma)

                stats['val loss'].append(val_loss.detach())
                stats['val ae loss'].append(val_ae_loss.detach())
                stats['val dyn loss'].append(val_dyn_loss.detach())
                stats['val latent R loss'].append(val_latent_R_loss.detach())
                # stats['val latent pi loss'].append(val_latent_pi_loss.detach())
                stats['val energy loss'].append(val_energy.detach())
                
                val_running_loss += val_loss.detach()
                val_steps += 1
                
            if epoch % args.eval_freq == (args.eval_freq - 1):
                print("\n Epoch: {}, Training Loss: {:.4e}, Validation Loss: {:.4e}".format(epoch + 1, train_loss.data, val_loss.data))
            
            if epoch % args.save_freq == (args.save_freq - 1):
                print("\n Saving Checkpoint for Epoch {} ... \n".format(epoch))
                save_checkpoint(model, stats, optim, epoch, train_loss.data, args.checkpoint_dir)
                
    return model, stats

"""

AUXILARY FUNCTIONS FOR TRAINING


"""

def init_weights(m):
    """
    Weight initialization for training using the Xavier initialization.
    
    ...
    
    Parameters
    ----------
    m : torch.nn.Module
        Model
        
    Returns
    -------
    
    Notes
    -----
    
    """
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)
        
def expand_dim(x, n, dim=0):
    """
    Expand dims function used in the equivariance regularizer.
    
    ...
    
    """
    if dim < 0:
        dim = x.dim()+dim+1
    return x.unsqueeze(dim).expand(*[-1]*dim, n, *[-1]*(x.dim()-dim))

"""

AUXILARY FUNCTIONS FOR LOSS FUNCTIONS


"""

def calc_momentum_torch(data: torch.Tensor):
    """
    Function to calculate the total angular momentum within a trajectory in Pytorch.
    
    ...
    
    Parameters
    ----------
    data : torch.Tensor
        Input trajectory data array.
    
    Returns
    -------
    total_momentum : torch.Tensor
        Torch tensor for the total angular momentum of each timestep of the trajectory.
        
    Notes
    -----
    
    """

    pi_vec = data[..., :3].float()
    total_momentum = torch.einsum('bni, bni -> bn', pi_vec, pi_vec)
    
    return total_momentum

def momentum_conservation_loss(momentum: torch.Tensor) -> torch.Tensor:
    """
    Function to calculate momentum loss.
    
    ...
    
    Parameters
    ----------
    momentum : torch.Tensor
        Tensor of the total angular momentum values throughout the trajectory.
        
    Returns
    -------
    loss : torch.Tensor
        Loss value for the batch.
        
    Notes
    -----
    
    """
    diff = torch.std(momentum, axis=-1).squeeze()
    loss = diff.view(-1).pow(2).mean()
    return loss


def calc_energy_torch(data: torch.Tensor, moi_diag: torch.Tensor, moi_off_diag: torch.Tensor = torch.Tensor([0., 0., 0.])):
    """
    Function to calculate the (kinetic) energy within a trajectory in Pytorch.
    
    ...
    
    Parameters
    ----------
    data : torch.Tensor
        Input trajectory data array.
    
    moi_diag : torch.Tensor
        Diagonal element of the moment of inertia tensor.
        
    moi_off_diag : torch.Tensor, default=torch.Tensor([0., 0., 0.])
        Off-diagonal element of the moment of inertia tensor.
        
    Returns
    -------
    kin_energy : torch.Tensor
        Torch tensor for the kinetic energy of each timestep of the trajectory.
        
    Notes
    -----
    The moment of inertia (MOI) is assumed to be diagonal.
    """
    pi_vec = data[..., :3].float()
    moi_inv = pd_matrix(diag=moi_diag, off_diag=moi_off_diag).float().to(pi_vec.device)
    
    kin_energy = 0.5 * torch.einsum('bnj, bnj -> bn', pi_vec, torch.einsum('ij, bnj -> bni ', moi_inv, pi_vec))
    
    return kin_energy

def energy_conservation_loss(energy: torch.Tensor) -> torch.Tensor:
    """
    Function to calculate energy loss.
    
    ...
    
    Parameters
    ----------
    energy : torch.Tensor
        Tensor of the energy values throughout the trajectory.
        
    Returns
    -------
    loss : torch.Tensor
        Loss value for the batch.
        
    Notes
    -----
    
    """
    diff = torch.std(energy, axis=-1).squeeze()
    loss = diff.view(-1).pow(2).mean()
    return loss

"""

LOSS FUNCTION DEFINITIONS


"""

def ae_loss(x: torch.Tensor,
            xhat_recon: torch.Tensor) -> torch.Tensor:
    """
    Loss function for the auto-encodering neural network.
    
    ...
    
    Parameters
    ----------
    x : torch.Tensor
        Input image array.
        
    xhat_recon : torch.Tensor
        Reconstructed input image array.
    
    Returns
    -------
    loss_val : torch.Tensor
        Output loss value 
        
    Notes
    -----
    
    """
    bs, seq_len, _, _, _ = xhat_recon.shape
    
    loss = torch.nn.MSELoss()
    loss_val = loss(xhat_recon, x[:, :seq_len, ...])
    return loss_val
 
def dynamics_loss(x: torch.Tensor,
                  xhat_pred: torch.Tensor) -> torch.Tensor:
    """
    Loss generated between the ground-truth image sequence and the those decoded preditions given by the learned dynamics.
    
    ...
    
    Parameters
    ----------
    x : torch.Tensor
        Input image array.
        
    xhat_pred : torch.Tensor
        Reconstructed images of the given image array predicted from dynamics.
    
    Returns
    -------
    loss_val : torch.Tensor
        Output loss value 
        
    Notes
    -----
    
    """
    # assert torch.any(x.isnan()) == False and torch.any(x.isinf()) == False
    # assert torch.any(xhat_pred.isnan()) == False and torch.any(xhat_pred.isinf()) == False
    
    bs, seq_len, _, _, _ = xhat_pred.shape
    
    loss = torch.nn.MSELoss()
    loss_val = loss(xhat_pred[:, 1:, ...], x[:, 1:, ...])
    return loss_val

# @torch.jit.script
def latent_loss_so3(z_enc: torch.Tensor,
                    z_pred: torch.Tensor) -> torch.Tensor:
    """
    """
    # assert torch.any(z_enc.isnan()) == False and torch.any(z_enc.isinf()) == False
    # assert torch.any(z_pred.isnan()) == False and torch.any(z_pred.isinf()) == False
    
    I = torch.eye(3).to(z_enc.device)
    I = I[None, ...]
    I = I.repeat(z_enc.shape[0], 1, 1)
    
    loss = torch.nn.MSELoss()
    loss_val = loss(I, torch.einsum('bij, bik -> bjk', z_enc, z_pred))
    return loss_val

@torch.jit.script
def latent_loss_pi(pi_enc: torch.Tensor, 
                   pi_pred: torch.Tensor) -> torch.Tensor:
    """
    Loss function for latent space -- angular momentum.
    
    ...
    
    Parameters
    ----------
    pi_enc : torch.Tensor,
        Estimated angular momentum using the encodings from the network.
        
    pi_pred : torch.Tensor
        Predicted angular momentum given by the learned dynamics.
        
    Returns
    -------
    loss_val : torch.Tensor
        Output loss value 
        
    Notes
    -----
    
    """
    n = pi_enc.shape[0]
    
    loss_n = (pi_enc - pi_pred).pow(2).view(n, -1).sum(-1)
    loss_val = loss_n.mean()
    return loss_val


"""

WEIGHTED LOSS FUNCTION INSPIRED BY OTTO AND ROWLEY, 2017


"""

@torch.jit.script
def recon_loss_LRAN(x: torch.Tensor,
               xhat_recon: torch.Tensor,
               delta: float = 1.0,
               epsilon: float = 1e-3) -> torch.Tensor:
    """
    Reconstruction loss function based on LRAN loss functions.
    
    ...
    
    Parameters
    ----------
    x : torch.Tensor
        Input image array.
        
    xhat_recon : torch.Tensor
        Reconstructed input image array.
        
    delta : float, default=1.0
        Discounting factor/weight for successive timesteps.
        
    epsilon : float, default=1E-3
        Small non-zero number added to denom to prevent divide-by-zero errors.
    
    Returns
    -------
    loss_val : torch.Tensor
        Output loss value 
        
    Notes
    -----
    
    """
    bs, seq_len, _, _, _ = x.shape
    n = bs * seq_len
    
    # define normalized discount factors
    tau_range = torch.arange(start=0, end=seq_len, device=x.device)
    delta_t = torch.pow(delta * torch.ones(seq_len, device=x.device), tau_range)
    
    N_delta = delta_t.sum()
    delta_tvec = delta_t/N_delta
    
    # calculate square differences and norm
    sq_diff = (x - xhat_recon).pow(2).reshape(bs, seq_len, -1).sum(-1)
    x_norm = x.pow(2).reshape(bs, seq_len, -1).sum(-1) + epsilon
    sq_diff_norm = torch.div(sq_diff, x_norm)
    
    loss_val = torch.einsum('bt, t -> b', sq_diff_norm, delta_tvec).mean()
    
    return loss_val

@torch.jit.script
def latent_loss_R_LRAN(R_enc: torch.Tensor,
                  R_pred: torch.Tensor,
                  delta: float = 1.):
    """
    Discounted latent loss over SO(3) for rotation matrices.
    
    ...
    
    Parameters
    ----------
    R_enc : torch.Tensor
        Encoded latent state.
        
    R_pred : torch.Tensor
        Predicted latent state.
        
    delta : float, default=1.0
        Discounting factor/weight for successive timesteps.
        
    Returns
    -------
    loss_val : torch.Tensor
        Output loss value 
        
    Notes
    -----
    
    """ 
    bs, seq_len, _, _ = R_enc.shape
    
    # define product element and identity
    R_prod = torch.einsum('btij, btkj -> btik', R_enc[:, 1:, ...], R_pred[:, 1:, ...])
    I = torch.eye(3, device=R_enc.device)
    I = I[None, None, ...].repeat(bs, seq_len, 1, 1)
    
    # define normalized discount factors
    tau_range = torch.arange(start=0, end=seq_len, device=R_enc.device)
    delta_t = torch.pow(delta * torch.ones(seq_len, device=R_enc.device), tau_range)
    N_delta = delta_t.sum()
    delta_tvec = delta_t/N_delta
    
    # calculate square differences and norm
    sq_norm = (I - R_prod).pow(2).view(bs, seq_len, -1).sum(-1) # torch.linalg.matrix_norm((I - R_prod)).pow(2)
    loss_val = torch.einsum('bt, t -> b', sq_norm, delta_tvec[1:]).mean()
    
    return loss_val

@torch.jit.script
def latent_loss_pi_LRAN(pi_enc: torch.Tensor,
                        pi_pred: torch.Tensor,
                        delta: float = 1.,
                        epsilon: float = 1e-3):
    """
    Reconstruction loss function based on LRAN loss functions.
    
    ...
    
    Parameters
    ----------
    pi_enc : torch.Tensor
        Estimated angular momentum from the encoded latent states.
        
    pi_pred : torch.Tensor
        Predicted angular momentum from the learned dynamics.
        
    delta : float, default=1.0
        Discounting factor/weight for successive timesteps.
    
    epsilon : float, default=1E-3
        Small non-zero number added to denom to prevent divide-by-zero errors.
        
    Returns
    -------
    loss_val : torch.Tensor
        Output loss value 
        
    Notes
    -----
    
    """ 
    bs, seq_len, _ = pi_enc.shape
    
    # define normalized discount factors
    tau_range = torch.arange(start=0, end=seq_len, device=pi_enc.device)
    delta_t = torch.pow(delta * torch.ones(seq_len, device=pi_enc.device), tau_range)
    N_delta = delta_t.sum()
    delta_tvec = delta_t/N_delta
    
    # calculate square differences and norm
    sq_diff = (pi_enc[:, 1:, ...] - pi_pred[:, 1:-1, ...]).pow(2).sum(-1)
    pi_norm = pi_enc[:, 1:, ...].pow(2).sum(-1) + epsilon
    sq_diff_norm = torch.div(sq_diff, pi_norm)
    
    loss_val = torch.einsum('bt, t -> b', sq_diff_norm, delta_tvec[:-1]).mean()
    
    return loss_val

"""

TOTAL LOSS FUNCTIONS


"""

def devs_loss(x: torch.Tensor,
              xhat_recon: torch.Tensor,
              xhat_pred: torch.Tensor,
              z_pred: torch.Tensor,
              z_enc: torch.Tensor,
              pi_pred: torch.Tensor,
              pi_enc: torch.Tensor,
              model,
              seed: int, 
              gamma: list = [1., 1., 1., 1.]) -> torch.Tensor:
    """
    Loss based on LagNetViP.
    
    ...
    
    Parameters
    ----------
    x : torch.Tensor
        Input image array.
        
    xhat_recon : torch.Tensor
        Reconstructed input image array.
    
    xhat_pred : torch.Tensor
        Reconstructed images of the given image array predicted from dynamics.
    
    z_pred : torch.Tensor
        Predicted latent state vector array.
        
    z_enc_norm : torch.Tensor
        Encoded latent state vector array of given image sequence.
    
    pi_enc : torch.Tensor
        Estimated angular momentum from the encoded latent states.
        
    pi_pred : torch.Tensor
        Predicted angular momentum from the learned dynamics.
    
    model : torch.nn.Module
        Trained model
        
    seed : int
        Random seed
        
    gamma : list, default=n*[1.]
        List of discount factors for each loss term. 
        
    Returns
    -------
    loss_total : float
        Total loss.
        
    loss_ae : float
        Autoencoder loss.
        
    loss_dyn : float
        Dynamics (learned) loss  from recon. images.
        
    loss_latent : float
        Latent loss.
    
    continuity : torch.Tensor
        Continuity regularization term.
        
    equivariance: torch.Tensor
        Equaivariance regularization term.
        
    Notes
    -----
    
    """
    loss_ae = ae_loss(x=x, xhat_recon=xhat_recon)
    loss_dyn = dynamics_loss(x=x, xhat_pred=xhat_pred)
    loss_latent_R = latent_loss_so3(z_pred=z_pred, z_enc=z_enc)
    # loss_latent_pi = latent_loss_pi_LRAN(pi_enc=pi_enc, pi_pred=pi_pred)
    
    # continuity = continuity_reg(encodings=z_enc, bs=x.shape[0], seq_len=x.shape[1])
    # equivariance = equivariance_reg(model=model, img=x, encoding=z_enc, seed=seed)
    
    loss_total = (gamma[0] * loss_ae) + (gamma[1] * loss_dyn) + (gamma[2] * loss_latent_R) # + (gamma[3] * loss_latent_pi) # + (gamma[4] * continuity) + (gamma[5] * equivariance)
    return loss_total, loss_ae, loss_dyn, loss_latent_R, # loss_latent_pi, continuity, equivariance

def devs_loss_plus_energy(x: torch.Tensor,
              xhat_recon: torch.Tensor,
              xhat_pred: torch.Tensor,
              z_pred: torch.Tensor,
              z_enc: torch.Tensor,
              pi_pred: torch.Tensor,
              pi_enc: torch.Tensor,
              model,
              seed: int, 
              gamma: list = [1., 1., 1., 1.]) -> torch.Tensor:
    """
    Loss based on LagNetViP including conservation of energy regularizer.
    
    ...
    
    Parameters
    ----------
    x : torch.Tensor
        Input image array.
        
    xhat_recon : torch.Tensor
        Reconstructed input image array.
    
    xhat_pred : torch.Tensor
        Reconstructed images of the given image array predicted from dynamics.
    
    z_pred : torch.Tensor
        Predicted latent state vector array.
        
    z_enc_norm : torch.Tensor
        Encoded latent state vector array of given image sequence.
    
    pi_enc : torch.Tensor
        Estimated angular momentum from the encoded latent states.
        
    pi_pred : torch.Tensor
        Predicted angular momentum from the learned dynamics.
        
    model : torch.nn.Module
        Trained model
        
    seed : int
        Random seed
    
    gamma : list, default=n*[1.]
        List of discount factors for each loss term. 
        
    Returns
    -------
    loss_total : torch.Tensor
        Total loss.
        
    loss_ae : torch.Tensor
        Autoencoder loss.
        
    loss_dyn : torch.Tensor
        Dynamics (learned) loss  from recon. images.
        
    loss_latent_R : torch.Tensor
        Latent loss for latent states on SO(3).
        
    loss_latent_pi : torch.Tensor
        Latent loss for angular moemntum.
        
    continuity : torch.Tensor
        Continuity regularization term.
        
    equivariance: torch.Tensor
        Equaivariance regularization term.
        
    energy : torch.Tensor
        Conservation of energy term.
        
    Notes
    -----
    
    """
    loss_ae = ae_loss(x=x, xhat_recon=xhat_recon)
    loss_dyn = dynamics_loss(x=x, xhat_pred=xhat_pred)
    loss_latent_R = latent_loss_so3(z_pred=z_pred, z_enc=z_enc)
    loss_latent_pi = latent_loss_pi_LRAN(pi_enc=pi_enc, pi_pred=pi_pred)
    
    # continuity = continuity_reg(encodings=z_enc, bs=x.shape[0], seq_len=x.shape[1])
    # equivariance = equivariance_reg(model=model, img=x, encoding=z_enc, seed=seed)
    
    energy = calc_energy_torch(data=pi_enc, moi_diag=model.moi_diag, moi_off_diag=model.moi_off_diag)
    energy_loss = energy_conservation_loss(energy=energy)
    
    loss_total = (gamma[0] * loss_ae) + (gamma[1] * loss_dyn) + (gamma[2] * loss_latent_R) + (gamma[3] * loss_latent_pi) + (gamma[4] * energy_loss)
    return loss_total, loss_ae, loss_dyn, loss_latent_R, loss_latent_pi, energy_loss

def devs_loss_LRAN(x: torch.Tensor,
                   xhat_recon: torch.Tensor,
                   xhat_pred: torch.Tensor,
                   z_pred: torch.Tensor,
                   z_enc: torch.Tensor,
                   pi_pred: torch.Tensor,
                   pi_enc: torch.Tensor,
                   gamma: list = [1., 1., 1., 1., 1., 1.],
                   delta: float = 1.,
                   epsilon: float = 1e-3) -> torch.Tensor:
    """
    Total loss inspired by LRAN (Otto and Crowley, 2017).
    
    ...
    
    Parameters
    ----------
    x : torch.Tensor
        Input image array.
        
    xhat_recon : torch.Tensor
        Reconstructed input image array.
    
    xhat_pred : torch.Tensor
        Reconstructed images of the given image array predicted from dynamics.
    
    z_pred : torch.Tensor
        Predicted latent state vector array.
        
    z_enc_norm : torch.Tensor
        Encoded latent state vector array of given image sequence.
    
    pi_enc : torch.Tensor
        Estimated angular momentum from the encoded latent states.
        
    pi_pred : torch.Tensor
        Predicted angular momentum from the learned dynamics.
        
    gamma : list, default=n*[1.]
        List of discount factors for each loss term.
    
    delta : float, default=1.0
        Discounting factor/weight for successive timesteps.
    
    epsilon : float, default=1E-3
        Small non-zero number added to denom to prevent divide-by-zero errors.
        
    Returns
    -------
    loss_total : torch.Tensor
        Total loss.
        
    loss_ae : torch.Tensor
        Autoencoder loss.
        
    loss_dyn : torch.Tensor
        Dynamics (learned) loss  from recon. images.
        
    loss_latent_R : torch.Tensor
        Latent loss for latent states on SO(3).
        
    loss_latent_pi : torch.Tensor
        Latent loss for angular moemntum.

    Notes
    -----
    
    """
    bs, seq_len, _, _, _ = x.shape
    R_enc = z_enc.reshape(bs, seq_len, 3, 3)
    R_pred = z_pred.reshape(bs, seq_len, 3, 3)
    
    loss_ae = ae_loss(x=x, xhat_recon=xhat_recon)
    loss_dyn = recon_loss_LRAN(x=x[:, 1:, ...], xhat_recon=xhat_pred[:, 1:, ...], delta=delta, epsilon=epsilon)
    loss_latent_R = latent_loss_R_LRAN(R_pred=R_pred, R_enc=R_enc, delta=delta)
    loss_latent_pi = latent_loss_pi_LRAN(pi_enc=pi_enc, pi_pred=pi_pred, delta=delta, epsilon=epsilon)
    
    # continuity = continuity_reg(encodings=z_enc, bs=x.shape[0], seq_len=x.shape[1])
    
    loss_total = (gamma[0] * loss_ae) + (gamma[1] * loss_dyn) + (gamma[2] * loss_latent_R) + (gamma[3] * loss_latent_pi) # + (0.1 * continuity)
    return loss_total, loss_ae, loss_dyn, loss_latent_R, loss_latent_pi #, continuity

"""

ADDITIONAL REGULARIZERS INSPIRED BY FALORSI ET AL., 2018


"""

@torch.jit.script
def continuity_reg(encodings: torch.Tensor,
                   bs: int,
                   seq_len:int) -> torch.Tensor:
    """
    Encoder continuity regularization.
    
    ...
    
    Parameters
    ----------
    encodings : torch.Tensor
        Encodings given from the encoding side of the network.
        
    bs : int
        Batch size used during training.
        
    seq_len : int
        Length of image seqeuences used during training.
        
    Returns
    -------
    mean : torch.Tensor
        MSE value
        
    Notes
    -----
    
    """
    encodings = encodings.squeeze()
    encodings = encodings.reshape(bs, seq_len, 3, 3)
    encodings_rs = torch.diff(encodings[..., :2].reshape(bs, seq_len, -1), dim=1)
  
    encodings_ = encodings_rs.view(encodings_rs.shape[0] * encodings_rs.shape[1], 6)
    diffs = encodings_.pow(2).view(encodings_rs.shape[0] * encodings_rs.shape[1], -1).sum(-1)
    mean = diffs.mean()
    
    return mean


def equivariance_reg(model,
                     img: torch.Tensor,
                     encoding: torch.Tensor, seed: int) -> torch.Tensor:
    """
    Regularization loss based on the equivariance in SO(3) -- adapted from Falorsi et al., 2018.
    
    ...
    
    Parameters
    ----------
    model : torch.nn.Module
        Trained model
        
    img : torch.Tensor
        Input images
        
    encodings : torch.Tensor
        Encodings given from the encoding side of the network.
        
    Returns
    -------
    reg_val : torch.Tensor
        Output regularization value 
        
    Notes
    -----
    
    """
    
    torch.manual_seed(seed)
    
    bs, sl, C, W, H = img.shape
    img_obs = img.reshape(-1, C, W, H)
    
    n = bs * sl
    theta = torch.rand(n, device=encoding.device)
    v = torch.tensor([1, 0, 0], dtype=torch.float32, device=encoding.device)
    
    s1 = torch.stack((torch.cos(theta), torch.sin(theta)), 1)
    g = s2s1rodrigues(expand_dim(v, n), s1)
    
    enc_rot = g.bmm(encoding)
    img_rot = rotate(img_obs, theta)
    img_rot_ = img_rot[:, None, ...] 
    img_rot_enc = model.encode(img_rot_)
    
    # loss difference for between the rotations
    I = torch.eye(3, device=encoding.device)
    I = I[None, ...]
    I = I.repeat(enc_rot.shape[0], 1, 1)
    
    loss_n = (I - torch.einsum('bij, bkj -> bjk', enc_rot, img_rot_enc)).pow(2).view(n, -1).sum(-1)
    reg_val = loss_n.mean()
    
    return reg_val

"""

DATA PROCESSING


"""

def data_preprocess(data: torch.Tensor,
                    observation_len: int = 2) -> torch.Tensor:
    """
    Function for pre-processing the input data to the correct observation length BEFORE encoding.
    
    ...
    
    Parameters
    ----------
    data : torch.Tensor
        Input data for encoding neural network model --shape: (batch size, sequence length, C, W, H).
        
    observation_len : int, default=2
        Desired observation lenght for the input of the neural network.
    
    Returns
    -------
    data_reshaped : torch.Tensor
        Reshaped input data -- shape: (psuedo batch size, observation len, C, W, H).
        
    Notes
    -----
    
    """
    batch_size, seq_len, C, W, H = data.shape
    
    if observation_len > 1:
        psuedo_bs = batch_size * (seq_len - 1)
    else:
        psuedo_bs = batch_size * seq_len
    
    data_unfolded = data.unfold(dimension=1, size=observation_len, step=1)
    data_unfolded = data_unfolded.permute(0, 1, 5, 2, 3, 4)
    
    data_rs = data_unfolded.reshape(psuedo_bs, observation_len, C, W, H)
    return data_rs

def data_postprocess(data: torch.Tensor,
                     batch_size: int,
                     seq_len: int = 2) -> torch.Tensor:
    """
    Function to post-process the output data AFTER decoding images.
    
    ...
    
    Parameters
    ----------
    data : torch.Tensor
        Output data from decoding neural network model --shape: (psuedo batch size, C, W, H).
    
    batch_size : int
        Batch size
        
    seq_len : int, default=2
        Length of the trajectory sequence.
        
    Returns
    -------
    data_rs : torch.Tensor
        Data reshaped to the shape (bs, seq_len, C, W, H).
        
    Notes
    -----
    
    """
    psuedo_bs, C, W, H = data.shape
    
    data_rs = data.reshape(batch_size, seq_len-1, C, W, H)
    return data_rs

"""

CHECKPOINT LOADING AND SAVING 


"""

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
        model.moi_diag = checkpoint['moi_diag']
        model.moi_off_diag = checkpoint['moi_off_diag']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(model.device)
        print("\n Loaded checkpoint '{}' (epoch {}) ... \n".format(filename, checkpoint['epoch']))
    else:
        print("\n No checkpoint found at '{}' ... \n".format(filename))
    
    return model, optimizer, stats, start_epoch

def save_checkpoint(model,
                    stats: list,
                    optimizer,
                    epoch: int ,
                    loss: float,
                    path: str):
    """
    Saves checkpoint after every number of epochs.
    
    ...
    
    Parameters
    ----------
    model
    
    stats : list
    
    optimizer
    
    epoch : int
        Epoch of the checkpoint.
        
    loss : float
        Loss value for checkpoint.
        
    path : str
        Directory for saving checkpoints.
        
    Returns
    -------
    
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
            'moi_diag': model.moi_diag,
            'moi_off_diag': model.moi_off_diag,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'stats': stats,
            }
    torch.save(checkpoint, ckpt_path)

def load_data(args: dict,
              data_dir: str):
    """
    Function for loading data.
    
    ...
    
    """
    print('\n Loading dataloaders ... \n')
    devs_traindl, devs_testdl, devs_valdl = generate_devs_dl(args=args, data_dir=data_dir)
    
    return devs_traindl, devs_testdl, devs_valdl

"""    

TUNE UTILS


"""
def run_tune(params, model, epoch, optim, path, ):
    """
    Function to report to tune for Ray Tune.
    
    ...
    
    Parameters
    ----------
    Returns
    -------
    Notes
    -----
    """
    # Condition : GPU is set to multiple gpus
    if params.gpu == 'cuda':
        model_ = model.module

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            save_checkpoint(model_, stats, optim, epoch, train_loss.item(), path)

        print('\n Reporting to Tune... \n')
        tune.report(loss=val_loss.item())
        
    else:
        
        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            save_checkpoint(model, stats, optim, epoch, train_loss.item(), path)

        print('\n Reporting to Tune... \n')    
        tune.report(loss=val_loss.item())
        