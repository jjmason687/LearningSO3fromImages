# Author(s) : Justice Mason
# Project : DEVS/RODEN
# Package : Data Processing Utility Functions 
# Date : 12/18/21

import os
import sys
sys.path.append('.')

# Pre-defined imports
import numpy as np
import torch
import pickle
from torch.utils.data import random_split

# User-defined imports
from models.lie_tools import rodrigues
from data.dataset_classes import DEVSdataset

def MSS_uniform(radius: float, seed: int = 0, num_traj: int = 1000) -> np.ndarray:
    """
    Function to generate an initial condition for angular momentum trajectories by "uniformly" sampling of the
    momentum sphere.
    
    ...
    
    Parameters
    ----------
    radius : float 
        Radius of the angular momentum sphere.
        
    seed : int, default=0
        RV seed for randomization.
        
    num_traj : int, default=1000
        Number of initial conditions/trajectories to be generated.
    
    Returns
    -------
    data : np.ndarray
        Array of initial conditions for angular momentum vector. [SHAPE: (num_traj, 3)]
    
    Notes
    -----
    source: http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/
    
    """
    np.random.seed(seed) 
    size = (num_traj, 3)
    
    data = np.random.normal(0, 1, size)  
    norm = np.linalg.norm(data, axis=1)  
    data = radius * (data.T/norm).T 
    
    return data

def generate_random_am_normal(n_trajectory: int = 1, radius_ams: float = 0.0) -> np.ndarray:
    """
    Function to generate random angular momentum vector.
    
    ...
    
    Parameters
    ----------
    
    Returns
    -------
    
    Notes
    -----
    
    """
    assert radius_ams > 0, "Radius of angular momentum sphere radius_ams must be greater than zero."
    assert n_trajectory > 0, "Number of trajectories must be greater than zero."
    
    data_ = np.randn((n_trajectory, 3))
    data_norm = np.linalg.norm(data_, dim=-1)
    data = radius_ams * np.divide(data_.T, data_norm)
    
    return data.T

def generate_random_am_uniform(n_trajectory: int = 1, radius_ams: float = 0.0) -> torch.Tensor:
    """
    Function to generate random angular momentum vector uniformly.
    
    ...
    
    Parameters
    ----------
    
    Returns
    -------
    
    Notes
    -----
    
    """
    assert radius_ams > 0, "Radius of angular momentum sphere radius_ams must be greater than zero."
    assert n_trajectory > 0, "Number of trajectories must be greater than zero."
    
    u = torch.rand((n_trajectory, 1))
    v = torch.rand((n_trajectory, 1))
    
    phi = torch.acos(2 * v - 1)
    theta = 2 * np.pi * u
    
    x = radius_ams * torch.sin(phi) * torch.sin(theta)
    y = radius_ams * torch.sin(phi) * torch.cos(theta)
    z = radius_ams * torch.cos(phi)
    
    data = torch.cat([x, y, z], dim=-1)
    
    return data.T

def sample_sphere_along_axis(n_trajectory: int = 1, radius_ams: float = 0.0, axis: str = None, direction: str = "positive") -> torch.Tensor:
    """
    Function to generate random angular momentum vector along a specified axis.
    
    ...
    
    Parameters
    ----------
    
    Returns
    -------
    
    Notes
    -----
    
    """
    assert radius_ams > 0, "Radius of angular momentum sphere radius_ams must be greater than zero."
    assert n_trajectory > 0, "Number of trajectories must be greater than zero."
    assert axis is not None, "Desired sampling axis must be provided."
    
    if direction.lower() == "postive":
        ax = 1.
    elif direction.lower() == "negative":
        ax = -1.
    else:
        raise ValueError("Must choose direction {} or {}.".format("positive", "negative"))
    
    if axis.upper() == "X":
        
        x = torch.ones((n_trajectory, 1)) * ax
        vec = torch.randn((n_trajectory, 2))
        data_ = torch.cat([x, vec], dim=-1)
        
    elif axis.upper() == "Y": 
        
        y = torch.ones((n_trajectory, 1)) * ax
        vec = torch.randn((n_trajectory, 2))
        
        x = vec[:, 0]
        z = vec[:, 1]
        data_ = torch.cat([x, y, z], dim=-1)
        
    elif axis.upper() == "Z":
        
        z = torch.ones((n_trajecotry, 1)) * ax
        vec = torch.randn((n_trajectory, 2))
        data_ = torch.cat([vec, z], dim=-1)
        
    else:
        raise ValueError("Must choose a valid axis: {}, {}, or {}.".format("X", "Y", "Z"))
    
    data_norm = torch.linalg.norm(data_, dim=-1)
    data = radius_ams * torch.div(data_.T, data_norm)
    
    return data.T

def euler_eigvec(MOI: torch.Tensor, radius: float) -> torch.Tensor:
    """
    Function to calculate the eigenvectors of the Euler dynamics, linearized about the intermediate axis.
    
    ...
    
    Parameters
    ----------
    MOI : torch.Tensor
        Moment of intertia tensor for the system.
        
    radius : float
        Radius of the angular momentum sphere.
    
    Returns
    -------
    eigvec : torch.Tensor
        Eigenvectors correpsonding to the dynamics after they're linearized about the intermediate axis.
        
    Notes
    -----
    
    """
    assert radius > 0., "Radius must be greater than zero."
    
    beta = (MOI[0, 0] - MOI[1, 1])/(MOI[0, 0] * MOI[1, 1]) # factor used for linearization
    gamma = (MOI[1, 1] - MOI[2, 2])/(MOI[1, 1] * MOI[2, 2]) # factor used for linearization
    
    euler_umatrix = torch.Tensor([[0, 0, beta * radius], [0, 0, 0], [gamma * radius, 0 , 0]]) # linearize dyns
    eigval, eigvec = torch.linalg.eig(euler_umatrix) # calculate the eigenvalues and eigenvectors 
    
    return eigvec

def sample_ms_unstable(MOI: np.ndarray, radius: float, seed: int = 0, num_traj: int = 1000) -> np.ndarray:
    """
    Function to generate an initial condition for angular momentum trajectories by sampling from the momentum sphere 
    along the unstable axes.
    
    ...
    
    Parameters
    ----------
    MOI : np.ndarray
        Moment of intertia tensor for the system.
        
    radius : float
        Radius of the angular momentum sphere.
        
    seed : int, default=0
        RV seed for randomization.
        
    num_traj : int, default=1000
        Number of initial conditions/trajectories to be generated.
    
    Returns
    -------
    data : np.ndarray
        Array of initial conditions for angular momentum vector. [SHAPE: (num_traj, 3)]
        
    Notes
    -----
    
    """
    np.random.seed(seed) # seed 
    size = (num_traj, 1)
    
    ev = euler_eigvec(MOI=MOI, radius=radius) # calc eigenvectors for unstable linearizations
    v1 = np.expand_dims(ev[:, 0], axis=0) # eigenvector 1 as column vector 
    v2 = np.expand_dims(ev[:, 1], axis=0) # eigenvector 2 as column vector 
    v3 = np.expand_dims(ev[:, 2], axis=0) # eigenvector 3 as column vector 
    
    lc_1 = np.random.uniform(-1.0, 1.0, size) # randomized weigth for linear combination 
    lc_2 = np.random.uniform(-1.0, 1.0, size) # randomized weigth for linear combination
    
    data = lc_1 @ v1 + v3
    
    norm = np.linalg.norm(data, axis=1) # calculate the norm of each data point
    data = radius * (data.T/norm).T # normalize and scale each point to be on the momentum sphere
    
    return data

def load_data(data_dir: str = "./data") -> np.ndarray:
    """
    Function to load data.
    
    ...
    
    Parameters
    ----------
    PIK : str
        Path to data file.
        
    Returns
    -------
    load_dict : dict
        Loaded dictionary of data.
        
    Notes
    -----
    
    """
    
    with open(data_dir, "rb") as file:
        load_dict = np.load(file, allow_pickle=True) #, mmap_mode='r+')
    
    return load_dict

def generate_devs_dl(args, eval_model: bool = False, data_dir: str = './data'): 
    """
    Function for generating the training, validation, and testing datasets from the raw dataset.
    
    ...
    
    Parameters
    ----------
    params : dict
        
    raw_data 
    Returns
    -------
    
    Notes
    -----
    
    """
    np.random.seed(args.seed)
    bs = args.bs
    raw_data = load_data(data_dir=data_dir)
    num_traj, traj_len, _, _, _ = raw_data.shape
    
    test_split = args.test_split
    val_split = args.val_split
    
    train_kwargs = {'batch_size': args.bs}
    val_kwargs = {'batch_size': args.bs}
    test_kwargs = {'batch_size': args.bs}
    
    if args.use_cuda:
        cuda_kwargs = {'num_workers': 4,
                       'pin_memory': True,
                       'shuffle': False}
        
        train_kwargs.update(cuda_kwargs)
        val_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    
    test_len = int(test_split * num_traj)
    val_len = int((1. - test_split) * val_split * num_traj)
    train_len = int((1. - test_split) * (1. - val_split) * num_traj)
    
    np.random.shuffle(raw_data)
        
    rd_split = np.split(raw_data.astype(float), [train_len, train_len + val_len, train_len + val_len + test_len], axis=0)
    
    train_dataset = rd_split[0]
    val_dataset = rd_split[1]
    test_dataset = rd_split[2]
    
    trainds_devs = DEVSdataset(data=train_dataset, seq_len=args.seq_len)
    testds_devs = DEVSdataset(data=test_dataset, seq_len=args.seq_len)
    valds_devs = DEVSdataset(data=val_dataset, seq_len=args.seq_len)
    
    
    devs_traindl = torch.utils.data.DataLoader(trainds_devs, **train_kwargs, drop_last=True)
    devs_testdl = torch.utils.data.DataLoader(testds_devs, **val_kwargs, drop_last=True)
    devs_valdl = torch.utils.data.DataLoader(valds_devs, **test_kwargs, drop_last=True)
    
    return devs_traindl, devs_testdl, devs_valdl

def window_split(x, window_size=100, stride=50, keep_short_tails=True):
    length = x.size(1)
    splits = []

    if keep_short_tails:
        for slice_start in range(0, length, stride):
          slice_end = min(length, slice_start + window_size)
          splits.append(x[:, slice_start:slice_end, ...])
    else:
        for slice_start in range(0, length - window_size + 1, stride):
          slice_end = slice_start + window_size
          splits.append(x[:, slice_start:slice_end, ...])
            
    return splits