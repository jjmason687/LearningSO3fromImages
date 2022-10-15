# Author(s): Justice Mason
# Projects: DEVS/RODEN
# Date: 12/04/21
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from models import lie_tools
from utils.physics_utils import rotational_kinematics, euler_step, rk4_step, estimate_ang_vel
from utils.math_utils import pd_matrix, project_so3, quat_omega
from utils.train_utils import data_preprocess, data_postprocess
from data.data_utils import window_split

class DEVS_SO3(nn.Module):
    """
    DEVS model.
    
    ...
    Attributes
    ----------
    device
    encoder
    decoder
    moi_diag
    moi_off_diag
    potential
    indices2
    indices4
    
    Methods
    -------
    get_moi_inv()
        Calculates the inverse moment of inertia tensor.
        
    encode(x)
        Runs input through the encoder.
        
    decode(z)
        Runs latent state through the decoder.
    
    forward(x, obs_len, seq_len, dt)
        Calls encode, the dynamics neural network, and decode.
        
    check_kine_constraint(C, threshold)
        Checks the kinematic constraint for elements of SO(3).
        
    hamiltonian(x, moi_inv)
        Calculates the hamiltonian given the state and inverser moment of inertia.
        
    calc_gradH(x, moi_inv, gradHtype, epsilon)
        Calculated the gradient (time) of the Hamiltonian.
        
    dynamics_update(x, moi_inv, gradHtype, epsilon)
        Computes the next state given the current state, inverser moi, and gradtype.
    
    kinematics_update(x, moi_inv, ang_mom, dt)
        Kinematics update in SO(3) for the next state given the current state, angular momentum, inverse MOI, and dt.
        
    kinematics_quat_update(x, moi_inv, ang_mom, dt)
        Kinematics update in quaternions for the next state given the current state, angular momentum, inverse MOI, and dt.
        
    state_rollout(x0, pi0, moi_inv, seq_len, dt)
        Full trajectory rollout for orientation and angular momentum given the initial values.
    
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
        self.moi_diag = torch.randn(3, requires_grad=True, dtype=torch.float)
        self.moi_off_diag = torch.randn(3, requires_grad=True, dtype=torch.float)
        self.potential = None
        self.indices2 = None
        self.indices4 = None
    
    def get_moi_inv(self) -> torch.Tensor:
        """
        Function to calculate the inverse MOI.
        
        ...
        
        """
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
        assert torch.any(x.isnan()) == False and torch.any(x.isinf()) == False
        
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
        assert torch.any(x.isnan()) == False and torch.any(x.isinf()) == False
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
        w_enc = estimate_ang_vel(R1=z1, R2=z2, dt=dt)
        moi = torch.linalg.inv(moi_inv)
        
        # estimate angular momentum 
        pi_enc = torch.einsum('ij, bj -> bi', moi, w_enc)
        pi_enc_rs = pi_enc.reshape(bs, slen-1, 3)
        
        # define initial conditions for R and pi
        pi0 = pi_enc_rs[:, 0, ...]
        R0 = z_rs_slide[:, 0, 0, ...]
        
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
        assert torch.any(x.isnan()) == False and torch.any(x.isinf()) == False
        assert torch.any(moi_inv.isnan()) == False and torch.any(moi_inv.isinf()) == False
        
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
                   gradHtype: str = 'FD',
                   epsilon: float = 1e-3) -> torch.Tensor:
        """
        Functon to calculate time derivative of the Hamiltonian.
        
        ...
    
        """
        assert torch.any(x.isnan()) == False and torch.any(x.isinf()) == False
        assert torch.any(moi_inv.isnan()) == False and torch.any(moi_inv.isinf()) == False
        
        if gradHtype.upper() == 'FD':
            
            e = torch.eye(3, device=self.device, dtype=torch.float32)
            H_i = self.hamiltonian(x=x, moi_inv=moi_inv)
            gradH = torch.zeros(size=(H_i.shape[0], 3), dtype=torch.float32, device=self.device)
            
            gradH[:,0] = (self.hamiltonian(x = (x + epsilon * e[:, 0].T), moi_inv=moi_inv) - H_i )/epsilon
            gradH[:,1] = (self.hamiltonian(x = (x + epsilon * e[:, 1].T), moi_inv=moi_inv) - H_i )/epsilon
            gradH[:,2] = (self.hamiltonian(x = (x + epsilon * e[:, 2].T), moi_inv=moi_inv) - H_i )/epsilon
            
            gradH = gradH
            
        elif gradHtype.upper() == 'ANALYTICAL':
            gradH = torch.matmul(moi_inv, x).permute(0, 2, 1).to(self.device)
            
        else:
            raise ValueError('Must choose either "FD" or "ANALYTICAL" for gradHtype.')
        
        return gradH
    
    def dynamics_update(self,
                        x: torch.Tensor,
                        moi_inv: torch.Tensor,
                        gradHtype: str = 'FD',
                        epsilon: float = 1e-3) -> torch.Tensor:
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
        assert torch.any(x.isnan()) == False and torch.any(x.isinf()) == False
        assert torch.any(moi_inv.isnan()) == False and torch.any(moi_inv.isinf()) == False
        
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
        assert torch.any(x.isnan()) == False and torch.any(x.isinf()) == False
        assert torch.any(moi_inv.isnan()) == False and torch.any(moi_inv.isinf()) == False
        
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
        assert torch.any(x.isnan()) == False and torch.any(x.isinf()) == False
        assert torch.any(moi_inv.isnan()) == False and torch.any(moi_inv.isinf()) == False
        
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
        assert torch.any(x0.isnan()) == False and torch.any(x0.isinf()) == False
        assert torch.any(pi0.isnan()) == False and torch.any(pi0.isinf()) == False
        assert torch.any(moi_inv.isnan()) == False and torch.any(moi_inv.isinf()) == False
        
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