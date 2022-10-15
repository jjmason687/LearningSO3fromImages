# Author(s): Justice Mason
# Project: DEVS/RODEN 
# Package: Math Utilities
# Date: 12/17/21

import numpy as np
import torch
import torch.nn.functional as F

from models.lie_tools import log_map, map_to_lie_algebra
from utils.physics_utils import rk4_step

def distance_so3(R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
    """
    Distance function for SO(3).
    
    ...
    
    Parameters
    ----------
    R1
    R2
    
    Returns
    -------
    distance
    
    Notes
    -----
    
    """
    product = torch.einsum('bij, bkj -> bik', R2, R1)
    trace = torch.einsum('bii', product)
    
    dist = 0.5 * (trace - 1)
    eps = 1e-2
    
    dist_ = dist.clamp(min=-1+eps, max=1-eps)
    distance = torch.acos(dist_)
    
    return distance

def torch_matrix_power(X: torch.Tensor, n) -> torch.Tensor:
    """
    Funciton to calculate the matrix power.
    
    ...
    
    Parameters
    ----------
    X : torch.Tensor
        Input matrix of shape (bs, n, n).
        
    n: float or in
        Power to raise matrix.
        
    Returns
    -------
    Xn : torch.Tensor
        Matrix raised to power n.
    
    Notes
    -----
    
    """
    evals, evecs = torch.linalg.eig(X)  # get eigendecomposition
    evals = evals.real # get real part of (real) eigenvalues
    evecs = evecs.real
    
    evpow = evals**n # raise eigenvalues to fractional power
    Xn = torch.matmul(evecs, torch.matmul(torch.diag_embed(evpow), torch.linalg.inv(evecs)))
    
    return Xn

def project_so3(R: torch.Tensor):
    """
    Function that projects R^{3 \times 3} on to SO(3).
    
    ...
    
    """
    assert torch.any(R.isnan()) == False and torch.any(R.isinf()) == False
    
    prod = torch.bmm(R, R.permute(0, 2, 1))
    C = torch.bmm(torch_matrix_power(X=prod, n=0.5), R)
    
    return C

def quat_omega(w: torch.Tensor) -> torch.Tensor:
    """
    Function to generate the \Omega(\omega) matrix in the kinematic differential equations for quaternions.
    
    ...
    
    Parameters
    ----------
    w : torch.Tensor
        Angular velocity
        
    Returns
    -------
    Q : torch.Tensor
        Matrix for KDEs of quaternions
        
    Notes
    -----
    Q = \Omega(w) = \[-S(w)  w \] \in su(2)
                    \[-w^{T} 0 \]
                    
    """
    bs, _, = w.shape
    S_w = map_to_lie_algebra(v=w)
    
    Q = torch.zeros((bs, 4, 4), device=w.device)
    Q[:, :3, :3] = -S_w
    Q[:, -1, :3] = -w
    Q[:, :3, -1] = w
    
    return Q
    
def pd_matrix(diag: torch.Tensor, off_diag: torch.Tensor) -> torch.Tensor:
    """
    Function constructing postive-definite matrix from diag/off-diag entries.
    
    ...
    
    Parameters
    ----------
    diag : torch.Tensor
        Diagonal elements of PD matrix.
        
    off-diag: torch.Tensor
        Off-diagonal elements of PD matrix.
        
    Returns
    -------
    matrix_pd : torch.Tensor
        Calculated PD matrix.
        
    Notes
    -----
    
    """
    diag_dim = diag.shape[0]
    
    L = torch.diag_embed(diag)
    ind = np.tril_indices(diag_dim, k=-1)
    flat_ind  = np.ravel_multi_index(ind, (diag_dim, diag_dim))
    
    L = torch.flatten(L, start_dim=0)
    L[flat_ind] = off_diag
    L = torch.reshape(L, (diag_dim, diag_dim))
    
    matrix_pd = L @ L.T + (1 * torch.eye(3, device=diag.device))
    
    return matrix_pd

def symmetric_matrix(diag: np.ndarray, off_diag: np.ndarray) -> np.ndarray:
    """
    Function to make symmetric matrix from diagonal and off-diagonal elements.
    
    ...
    
    Parameters
    ----------
    diag : np.ndarray
        Diagonal entries
        
    off_diag : np.ndarray
        Off-diagonal entries
        
    Returns
    -------
    sym_matrix : np.ndarray
        Symmetric matrix
    Notes
    -----
    
    """
    lt_id = np.tril_indices(3, k=-1)
    ut_id = np.triu_indices(3, k=1)
    A = np.zeros((3,3))
    
    A[ut_id] = off_diag
    A[lt_id] = off_diag
    
    sym_matrix = A + np.diag(diag)
    return sym_matrix

def sample_points_on_sphere(radius: float, theta_arr: np.ndarray, phi_arr: np.ndarray) -> np.ndarray:
    """
    Function to sample points on sphere given radius and angles \phi and \theta.
    
    ...
    
    Parameters
    ----------
    radius : float
        Radius of desired sphere.
        
    tehta_arr : np.ndarray
        Array of theta values to be sampled on given sphere.
        
    phi_arr : np.ndarray
        Array of phi values to be sampled on given sphere.
        
    Returns
    -------
    samples : np.ndarray
        Array of sampled points from given sphere.
        
    Notes
    -----
    Using (r, \theta, \phi) \rightarrow (x, y, z) convention.
    
    """
    
    cos = np.cos
    sin = np.sin
    r = radius
    
    assert r > 0.0, "sphere's radius must be postive-value and greater than zero."
    
    x = r * np.multiply(sin(phi_arr), cos(theta_arr))
    y = r * np.multiply(sin(phi_arr), sin(theta_arr))
    z = r * sin(phi_arr)
    
    assert (y.shape == z.shape and x.shape == y.shape), "XYZ coordinate shapes are not matching."
    
    samples = np.concatenate((x, y, z), axis=0)
    return samples

def rotate(img, theta):
    """
    Rotation function used in equivariance loss from Falorsi et al., 2018.
    
    ...
    
    """
    cos = torch.cos(theta)
    sin = torch.sin(theta)
    zero = torch.zeros_like(theta)
    affine = torch.stack([cos, -sin, zero, sin, cos, zero], 1).view(-1, 2, 3)
    grid = F.affine_grid(affine, img.size(), align_corners=True)
    return F.grid_sample(img, grid, align_corners=True)
