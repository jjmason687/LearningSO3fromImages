# Author(s): Justice Mason
# Projects: DEVS/RODEN
# Package: Training Script for DEVS/RODEN
# Date: 08/01/22

import os
import sys
sys.path.append('.')
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse
import torch, torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from glob import glob
from time import time, strftime
from torch.utils.tensorboard import SummaryWriter

# User-defined function imports
# import helper
from models import lie_tools
from models.devs_models import DEVS_SO3
from models.autoencoder import EncoderNetSO3, DecoderNetSO3
from data.dataset_classes import DEVSdataset
from data.data_utils import load_data, generate_devs_dl, window_split
from utils.math_utils import pd_matrix, project_so3, quat_omega
from utils.train_utils import latest_checkpoint, load_checkpoint, train_loop, devs_loss_plus_energy, init_weights, data_preprocess, data_postprocess
from utils.physics_utils import true_dyn_model, true_dyn_model, rotational_kinematics, euler_step, rk4_step, estimate_ang_vel
from utils.visualization_utils import latent_eval, plot_loss, recon_eval, plot_recon_output, plot_pred_output

class DEVS_SO3(nn.Module):
    """
    DEVS model.
    
    ...
    
    Notes
    -----
    
    """
    def __init__(self,
                 device,
                 encoder: torch.nn.Module,
                 decoder: torch.nn.Module,
                 potential=None) -> None:
        
        super(DEVS_SO3, self).__init__()
        self.device = device
        self.encoder = encoder
        self.decoder = decoder
        self.moi_diag = torch.rand(3, requires_grad=True, dtype=torch.float)
        self.moi_off_diag = torch.rand(3, requires_grad=True, dtype=torch.float)
        self.potential = None
        self.indices2 = None
        self.indices4 = None
    
    def get_moi_inv(self) -> torch.Tensor:
        """
        Function to calculate the inverse MOI.
        
        ...
        
        """
        # assert torch.any(self.moi_diag.isnan()) == False and torch.any(self.moi_diag.isinf()) == False
        # assert torch.any(self.moi_off_diag.isnan()) == False and torch.any(self.moi_off_diag.isinf()) == False
        
        return pd_matrix(diag=self.moi_diag, off_diag=self.moi_off_diag).to(self.device)
    
    def encode(self,
               x: torch.Tensor) -> torch.Tensor:
        """
        Encoder network.
        
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
        # assert torch.any(x.isnan()) == False and torch.any(x.isinf()) == False
        
        z_enc, indices2, indices4 = self.encoder(x)
        
        self.indices2 = indices2
        self.indices4 = indices4
        
        return z_enc
    
    def decode(self,
               z: torch.Tensor) -> torch.Tensor:
        """
        Decoder network.
        
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
    
    def forward(self,
                x: torch.Tensor,
                obs_len: int = 1,
                seq_len: int = 3,
                dt: float = 1e-3) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward method for DEVS.
        
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
        
        pi_enc : torch.Tensor
            Estimated angular momentum using the encoded latent state given by encoder side of network.
        
        pi_pred : torch.Tensor
            Predicted angular momentum given by learned dynamics.
        
        Notes
        -----
        
        """
        # assert torch.any(x.isnan()) == False and torch.any(x.isinf()) == False
        bs, slen, _, _, _ = x.shape
        
        moi_inv = self.get_moi_inv()
        
        
        x_obs = data_preprocess(data=x, observation_len=obs_len)
        z_enc = self.encode(x=x_obs) 
        z_rs = z_enc.reshape(bs, -1, 3 ,3)
        
    
        z_rs_slide_ = window_split(x=z_rs, window_size=2, stride=1, keep_short_tails=False)
        z_rs_slide = torch.stack(z_rs_slide_, dim=1)
        
        z1 = z_rs_slide[:, :, 0, ...].reshape(-1, 3, 3)
        z2 = z_rs_slide[:, :, 1, ...].reshape(-1, 3, 3)
        
        
        # estimate angular velocity
        w_enc = estimate_ang_vel(R1=z1, R2=z2, dt=dt).reshape(bs, -1, 3)
        moi = torch.linalg.inv(moi_inv)
        
        
        # estimate angular momentum 
        pi_enc = torch.einsum('ij, btj -> bti', moi, w_enc)
        pi_enc_rs = pi_enc.reshape(bs, slen-1, 3)
        
        
        # define initial conditions for R and pi
        pi0 = pi_enc_rs[:, 0, ...]
        R0 = z_rs[:, 0, ...]
        
        # forward dynamics prediction
        R_pred, pi_pred = self.state_rollout(x0=R0, pi0=pi0, moi_inv=moi_inv, seq_len=seq_len, dt=dt)
        
        # reshape predictions into pseudo batch form
        z_pred = R_pred.reshape(-1, 3, 3)
        pi_pred_rs = pi_pred.reshape(-1, 3)
        
        # decode images from autoencoder
        xhat_dec = self.decode(z_enc)
        xhat_recon = data_postprocess(data=xhat_dec, batch_size=bs, seq_len=seq_len+1)
        
        # decode images from predicted states
        xhat_pred_dec = self.decode(z=z_pred)
        xhat_pred = data_postprocess(data=xhat_pred_dec, batch_size=bs, seq_len=seq_len+1)
        
        return xhat_recon, xhat_pred, z_enc, z_pred, pi_enc_rs, pi_pred
    
    def check_kine_constraint(self,
                              C: torch.Tensor,
                              threshold: float = 1e-3) -> torch.Tensor:
        """
        Function to check if matrix satisfies the kinematic constraint on SO(3)/project it on SO(3).
        
        ...
        
        Parameters
        ----------
        C : torch.Tensor
            Input tensor of shape (bs, 3, 3).
            
        threshold : float, default=1E-3
            Threshold value for SO(3) constraint.
        
        Returns
        -------
        R_next : torch.Tensor
            Output tensor on SO(3) of shape (bs, 3, 3).
        
        Notes
        -----
        
        """
        # assert torch.any(C.isnan()) == False and torch.any(C.isinf()) == False
        
        loss = torch.nn.MSELoss()
        I = torch.eye(3, device=self.device)[None, ...]
        constraint  = loss(torch.bmm(C.permute(0, 2, 1), C), I)
        
        if constraint > threshold:
            R_next = project_so3(C)
        else:
            R_next = C
    
        return R_next
    
    def hamiltonian(self,
                    x: torch.Tensor,
                    moi_inv: torch.Tensor) -> torch.Tensor:
        """
        Function for calculating the Hamiltonian.
        
        ...
        
        Parameters
        ----------
        x : torch.Tensor
            Input angular momentum tensor of shape (bs, 1, 3)

        moi_inv : torch.Tensor
            Moment of inertia tensor of shape (3, 3)
        
        V : torch.Tensor
            Potential energy.
            
        Returns
        -------
        hamiltonian : torch.Tensor
        
        Notes
        -----
        
        """
        # assert torch.any(x.isnan()) == False and torch.any(x.isinf()) == False
        # assert torch.any(moi_inv.isnan()) == False and torch.any(moi_inv.isinf()) == False
        
        kinetic = 0.5 * torch.einsum('bi, bi -> b', x, torch.einsum('ij, bj -> bi', moi_inv, x))
        potential = self.potential
        
        if self.potential is not None:
            hamiltonian = kinetic.squeeze() + potential 
        else:
            hamiltonian = kinetic.squeeze()
        
        return hamiltonian
    
    def calc_gradH(self,
                   x: torch.Tensor,
                   moi_inv: torch.Tensor,
                   gradHtype: str = 'ANALYTICAL',
                   epsilon: float = 1e-4) -> torch.Tensor:
        """
        Functon to calculate time derivative of the Hamiltonian.
        
        ...
    
        """
        # assert torch.any(x.isnan()) == False and torch.any(x.isinf()) == False
        # assert torch.any(moi_inv.isnan()) == False and torch.any(moi_inv.isinf()) == False
        
        if gradHtype.upper() == 'FD':
            
            e = torch.eye(3, device=self.device, dtype=torch.float32)
            H_i = self.hamiltonian(x=x, moi_inv=moi_inv)
            gradH = torch.zeros(size=(H_i.shape[0], 3), dtype=torch.float32, device=self.device)
            
            gradH[:,0] = (self.hamiltonian(x = (x + epsilon * e[:, 0].T), moi_inv=moi_inv) - H_i )/epsilon
            gradH[:,1] = (self.hamiltonian(x = (x + epsilon * e[:, 1].T), moi_inv=moi_inv) - H_i )/epsilon
            gradH[:,2] = (self.hamiltonian(x = (x + epsilon * e[:, 2].T), moi_inv=moi_inv) - H_i )/epsilon
            
            gradH = gradH
            
        elif gradHtype.upper() == 'ANALYTICAL':
            gradH = torch.einsum('ij, bj -> bi', moi_inv, x)
            
        else:
            raise ValueError('Must choose either "FD" or "ANALYTICAL" for gradHtype.')
        
        return gradH
    
    def dynamics_update(self,
                        x: torch.Tensor,
                        moi_inv: torch.Tensor,
                        gradHtype: str = 'ANALYTICAL',
                        epsilon: float = 1e-1) -> torch.Tensor:
        """
        Update function for RB dynamics on SO(3).
        
        ...
        
        Parameters
        ----------
        x : torch.Tensor
        moi_inv : torch.Tensor
        gradHtype : str, default='FD'
        epsilon : float, default=1e-3
        
        Returns
        -------
        dx : torch. Tensor
        
        Notes
        -----
        
        """
        # assert torch.any(x.isnan()) == False and torch.any(x.isinf()) == False
        # assert torch.any(moi_inv.isnan()) == False and torch.any(moi_inv.isinf()) == False
        
        gradH = self.calc_gradH(x=x, moi_inv=moi_inv, gradHtype=gradHtype, epsilon=epsilon)
        
        
        dx = torch.cross(x, gradH)
        
        return dx
    
    def kinematics_update(self,
                          x: torch.Tensor,
                          moi_inv: torch.Tensor,
                          ang_mom: torch.Tensor,
                          dt: float = 1e-3) -> torch.Tensor:
        """
        Update function for RB kinematic on SO(3).
        
        ...
        
        Parameters
        ----------
        x : torch.Tensor 
        moi_inv : torch.Tensor
        ang_mom : torch.Tensor
        dt : float, default=1e-3
        
        Returns
        -------
        R_next: torch.Tensor
        
        Notes
        -----
        The function first interpolates the next value of R using angular momentum, then projects onto SO(3) if the resulting matrix
        doesn't satisfy the constraints of SO(3).
        
        """
        # assert torch.any(x.isnan()) == False and torch.any(x.isinf()) == False
        # assert torch.any(moi_inv.isnan()) == False and torch.any(moi_inv.isinf()) == False
        # assert torch.any(ang_mom.isnan()) == False and torch.any(ang_mom.isinf()) == False
        
        R = x
        pi = ang_mom
        
        # interpolation on SO(3)
        w = torch.squeeze(torch.matmul(moi_inv, pi))
        exp_R = lie_tools.rodrigues(v=dt* w)
        C = torch.bmm(exp_R, R)
        
        # constraint-based projection onto SO(3)
        R_next = self.check_kine_constraint(C=C)
        
        return R_next
    
    def kinematics_quat_update(self,
                               x: torch.Tensor,
                               moi_inv: torch.Tensor,
                               ang_mom: torch.Tensor,
                               dt: float = 1e-3) -> torch.Tensor:
        """
        Update function for RB kinematic on SO(3).

        ...

        Parameters
        ----------
        x : torch.Tensor 
        moi_inv : torch.Tensor
        ang_mom : torch.Tensor
        dt : float, default=1e-3

        Returns
        -------
        R_next: torch.Tensor

        Notes
        -----
        The function first interpolates the next value of R using angular momentum, then projects onto SO(3) if the resulting matrix
        doesn't satisfy the constraints of SO(3).

        """
        # assert torch.any(x.isnan()) == False and torch.any(x.isinf()) == False
        # assert torch.any(moi_inv.isnan()) == False and torch.any(moi_inv.isinf()) == False
        # assert torch.any(ang_mom.isnan()) == False and torch.any(ang_mom.isinf()) == False
        
        R = x
        pi = ang_mom
        
        q = lie_tools.group_matrix_to_quaternions(R)

        # interpolation on SO(3)
        w = torch.einsum('ij, bj -> bi', moi_inv, pi)
        
        dq = lambda q : torch.einsum('bij, bj -> bi', quat_omega(w=w), q)
        q_next = rk4_step(func=dq, x0=q, dt=dt)
        q_next = q_next/q_next.norm(p=2, dim=-1,keepdim=True).clamp(min=1E-5)
        
        R_next = lie_tools.quaternions_to_group_matrix(q_next)

        return R_next
    
    def state_rollout(self,
                      x0: torch.Tensor,
                      pi0: torch.Tensor,
                      moi_inv: torch.Tensor,
                      seq_len: int = 2,
                      dt: float = 1e-3) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Function to perform state roll-out given the dynamics and kinematic updates using the first-order Euler step.
        
        ...
        
        Parameters
        ----------
        x0 : torch.Tensor
            Initial condition for orientation matrix
            
        pi0 : torch.Tensor
            Initial condition for angular momentum vector
            
        moi_inv : torch.Tensor 
            Moment of inertia tensor
            
        seq_len : int, default=2
            Prediction sequence length
            
        dt : float, default=1e-3
            Timestep 
        
        Returns
        -------
        R_pred : torch.Tensor
        pi_pred : torch.Tensor
        
        Notes
        -----
        
        """
        # assert torch.any(x0.isnan()) == False and torch.any(x0.isinf()) == False
        # assert torch.any(pi0.isnan()) == False and torch.any(pi0.isinf()) == False
        # assert torch.any(moi_inv.isnan()) == False and torch.any(moi_inv.isinf()) == False
        
        output_R = [x0]
        output_pi = [pi0]
        
        for t in range(1, seq_len):
            R = output_R[-1]
            pi = output_pi[-1]
            
            dynamics_update = lambda x : self.dynamics_update(x=x, moi_inv=moi_inv)

            R_next = self.kinematics_quat_update(x=R, moi_inv=moi_inv, ang_mom=pi, dt=dt)
            pi_next = rk4_step(func=dynamics_update, x0=pi, dt=dt)
            
            output_R.append(R_next)
            output_pi.append(pi_next)
        
        R_pred = torch.stack(output_R, axis=1)
        pi_pred = torch.stack(output_pi, axis=1)
        
        return R_pred, pi_pred

def angularvel_estimator_tensor_v2(R_i: torch.Tensor, R_ii: torch.Tensor) -> torch.Tensor:
    """
    """
    #torch.set_default_dtype(torch.double)
    
    batch_size = R_i.shape[0]
    Rprod = torch.einsum('bij, bkj -> bik', R_ii, R_i)
    u_skew = Rprod - Rprod.permute(0, 2, 1)

    theta = torch.arccos((torch.einsum('bii', Rprod) - 1)/2)
    
    u_norm = torch.zeros((u_skew.shape[0], 3), device=R_i.device)
    
    for elem in range(batch_size):
        u_skew_el = u_skew[elem, ...]
        theta_el = theta[elem, ...]
        
        if torch.all(u_skew_el == torch.zeros(3, device=u_skew_el.device)) and theta_el == 0.0:
            u_norm[elem, ...] = u_skew_el
    
        elif torch.all(u_skew_el == torch.zeros(3, device=u_skew_el.device)) and theta_el == np.pi:
            u_ = Rprod[elem, ...] + torch.eye(3, device=u_skew_el.device)
            u = u_[:, 0]
            u_norm[elem, ...] = u/(torch.linalg.norm(u))

        else:
            u = lie_tools.map_to_lie_vector(X=u_skew_el.unsqueeze(0)).squeeze()
            u_norm[elem, ...] = u/(torch.linalg.norm(u))
    
    return u_norm, theta 

def avel_est_traj_tensor(R_traj: torch.Tensor, timestep: float = 1e-3) -> torch.Tensor:
    """
    """
    traj_len = R_traj.shape[1]
    angvel_list = []
    
    for t in range(traj_len-1):
        Ri = R_traj[:, t, ...]
        Rii = R_traj[:, t+1, ...]
        
        axis, angle = angularvel_estimator_tensor_v2(R_i=Ri, R_ii=Rii)
        mag_est = (angle.unsqueeze(1)/timestep).clamp(max=1E2)
        angvel_est = torch.einsum('bji, bj -> bi', Ri, (mag_est * axis))
        angvel_list.append(angvel_est)
    
    angvel_vec = torch.stack(angvel_list, axis=1)
    return angvel_vec


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
    parser.add_argument('--checkpoint_freq',
                        action='store',
                        default=10,
                        type=int, help='model evaluation during training')
    parser.add_argument('--log_freq',
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
    torch.manual_seed(args.seed)
    device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")
    args.device = device
    
    writer = SummaryWriter(log_dir=args.log_dir)
    
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
        
    
    print('\n Training on GPU: {} ... \n'.format(args.gpu))
    devs = devs.to(device)
    
    torch.backends.cudnn.benchmark = True
    
    print('\n Training model ... \n')
    model, stats = train_loop(args=args,\
                                writer=writer,\
                                train_dl=devs_traindl,\
                                val_dl=devs_valdl,\
                                optimizer=devs_optim,\
                                lr_scheduler=devs_lr_sched,\
                                stats=devs_stats, model=devs, \
                                loss_fcn=devs_lf,\
                                start_epoch=start_epoch)
    
    print('\n Done training model ... \n')
    return model, stats

if __name__ == "__main__":
    
    torch.cuda.empty_cache()
    args = get_args()
    torch.manual_seed(args.seed)
    model, stats = main()
    
    est_moi_inv = pd_matrix(diag=model.moi_diag, off_diag=model.moi_off_diag)
    print('\n Estimated Inverse of MOI: {} \n', est_moi_inv)
    print('\n JOB DONE! \n')
    