import os
import sys

sys.path.append('/home/jjmason/DEVS')

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import plotly.express as px
from torch.utils.data import Dataset, DataLoader

from data.data_utils import generate_devs_dl
from utils.train_utils import latest_checkpoint, load_checkpoint, data_preprocess, data_postprocess, latent_loss_so3
from utils.visualization_utils import animate_trajectory, plot_traj_on_sphere
from utils.integrator_utils import Integrator, integrate_trajectory
from utils.physics_utils import eulers_driver, rk4_step, estimate_ang_vel
from utils.math_utils import distance_so3, project_so3
from models.autoencoder import EncoderNetSO3, DecoderNetSO3
from models.devs_models import DEVS_SO3
from models.lie_tools import group_matrix_to_quaternions, quaternions_to_group_matrix

import models.lie_tools

class VariationalIntegrator():
    """
    
    """
    def __init__(self):
        super(VariationalIntegrator, self).__init__()
        
    def skew(self, v):
        
        S = np.zeros([3, 3])
        S[0, 1] = -v[2]
        S[1, 0] = v[2]
        S[0, 2] = v[1]
        S[2, 0] = -v[1]
        S[1, 2] = -v[0]
        S[2, 1] = v[0]
    
        return S
    
    def cayley_transx(self, fc):
        """
        """
        F = (np.eye(3) + self.skew(fc)) @ np.linalg.inv(np.eye(3) - self.skew(fc))
        return F
    
    def calc_fc_init(self, pi_vec, moi):
        """
        """
        fc_init = np.linalg.inv(2 * moi - self.skew(pi_vec)) @ pi_vec
        return fc_init
    
    def calc_Ac(self, a_vec: np.ndarray, moi: np.ndarray, fc: np.ndarray) -> np.ndarray:
        """
        """
        
        Ac = a_vec + (self.skew(a_vec) @ fc) + (fc * (a_vec.T @ fc)) - (2 * (moi @ fc))
        return Ac
        
    def calc_grad_Ac(self, a_vec: np.ndarray, moi: np.ndarray, fc: np.ndarray) -> np.ndarray:
        """
        """
        grad_Ac = self.skew(a_vec) + ((a_vec.T @ fc) * np.eye(3)) + (fc * a_vec.T) - (2 * moi)
        return grad_Ac
    
    def optimize_fc(self, pi_vec: np.ndarray, moi: np.ndarray, fc_list: list = [], timestep: float = 1e-3, max_iter: int = 1000, tol: float = 1e-3):
        """
        """
        it = 0
        eps = np.inf
        
        if not fc_list:
            fc_list.append(self.calc_fc_init(pi_vec=pi_vec, moi=moi))
        
        while  eps > tol and it < max_iter:
            
            fc_i = fc_list[-1]
            a_vec = timestep * pi_vec
            
            Ac = self.calc_Ac(a_vec=a_vec, moi=moi, fc=fc_i)
            grad_Ac = self.calc_grad_Ac(a_vec=a_vec, moi=moi, fc=fc_i)
            
            
            fc_ii = fc_i - (np.linalg.inv(grad_Ac) @ Ac)
            
            eps = np.linalg.norm(fc_ii - fc_i)
            fc_list.append(fc_ii)
            it += 1
        
        return fc_list
    
    def step(self, R_i: np.ndarray, pi_i: np.ndarray, moi: np.ndarray, fc_list: list = [], timestep: float = 1e-3):
        """
        """
        fc_list = self.optimize_fc(pi_vec=pi_i, moi=moi, timestep=timestep, fc_list=fc_list)
        
        fc_opt = fc_list[-1]
        F_i = self.cayley_transx(fc=fc_opt)
        
        R_ii = R_i @ F_i
        pi_ii = F_i.T @ pi_i
        
        return R_ii, pi_ii, fc_list
    
    def integrate(self, pi_init: np.ndarray, R_init: np.ndarray, moi: np.ndarray, timestep: float = 1e-3, traj_len: int = 100):
        """
        """
        pi_list = [pi_init]
        R_list = [R_init]
        fc_list = []
        
        for it in range(1, traj_len):
            R_i = R_list[-1]
            pi_i = pi_list[-1]
            
            R_ii, pi_ii, fc_list = self.step(R_i=R_i, pi_i=pi_i, moi=moi, fc_list=fc_list, timestep=timestep)
            
            R_list.append(R_ii)
            pi_list.append(pi_ii)
        
        R_traj = np.stack(R_list, axis=0)
        pi_traj = np.stack(pi_list, axis=0)
        return R_traj, pi_traj
    
    
VI = VariationalIntegrator()

moi = np.diag([3., 2., 1.])
R_0 = np.eye(3)
pi_ = np.random.rand(3)
pi_0 = (40 * pi_/(np.linalg.norm(pi_))).reshape(3, 1)

R, pi = VI.integrate(pi_init=pi_0, R_init=R_0, moi=moi, traj_len=1000)