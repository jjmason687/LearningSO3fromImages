# Author(s): Justice Mason
# Project: DEVS/RODEN 
# Package: Physics Utilities 
# Date: 12/17/21

import numpy as np
import torch
import pandas as pd

from models import lie_tools
from utils.integrator_utils import integrate_trajectory

"""

ODE DRIVERS


"""

def euler_driver(pi: torch.Tensor,
                 moi_inv: torch.Tensor) -> torch.Tensor:
    """
    Driver for Euler's rigid body dynamics for Pytorch.
    
    ...
    
    Parameters
    ----------
    pi : torch.Tensor
        Angular momentum state vector.
        
    moi_inv : torch.Tensor
        Inverse moment of inertia tensor.
        
    Returns
    -------
    pi_dot : torch.Tensor
        Angular momentum state vector derivative.
        
    Notes
    -----
    
    """
    grad_H = torch.matmul(moi_inv, pi.permute(0, 2, 1))
    pi_dot = torch.cross(pi, grad_H.permute(0, 2, 1))
    
    return pi_dot

def eulers_driver(t,
                  y: np.ndarray,
                  moi_diag: np.ndarray = np.array([3., 2., 1.]),
                  moi_off_diag: np.ndarray = np.array([0., 0., 0.])) -> np.ndarray:
    """
    Driver for integrating the Euler dynamic equations for Numpy.
    
    ...
    
    Parameters
    ----------
    t : np.ndarray
        Input time array
        
    y : np.ndarray
        Input state vector as an array
        
    moi : np.ndarray
        Moment of inertia tensor
        
    Returns
    -------
    ydot : np.ndarray
        Time derivative of the state vector
        
    Notes
    -----
    ASSUMPTIONS: 
    
        1. The control input vector u(t) is assumed constant, so its grad is hardcoded as a column vector of zeros.
        
        2. The Euler's equations are calculated in the princinple axis reference frame, so the moment of inertia
        is assumed diagnal (in addition to symmetric and PD). 
    
    """
    I_inv = symmetric_matrix(diag = moi_diag, off_diag=moi_off_diag)
    y = np.expand_dims(y, axis=1)
    pi, u = np.split(y, indices_or_sections=2, axis=0) 
    
    pi_dot = np.cross(pi.T, (I_inv @ pi).T) + u.T # Euler's equations for rigid bodies 
    pi_dot = pi_dot.T 
    u_dot = np.zeros_like(pi_dot)
    
    return np.squeeze(np.concatenate((pi_dot, u_dot), axis=0))

"""

INTEGRATION FUNCTIONS


"""

def euler_step(dy: torch.Tensor,
               y0: torch.Tensor,
               dt: float = 0.01) -> torch.Tensor:
    """
    First-order Euler step function.
    
    ...
    
    Parameters
    ----------
    dy : torch.Tensor
        Time derivative of state vector.
        
    y0 : torch.Tensor
        Initial condition for state vector.
        
    dt : float, default=0.01
        Constant value used for time step.
        
    Returns
    -------
    y1 : torch.Tensor
        State corresponding to the next time step.
        
    Notes
    -----
    
    """
    y1 = y0 + (dt * dy)
    
    return y1

def rk4_step(func,
             x0: torch.Tensor,
             dt: float = 0.01,
             **kwargs) -> torch.Tensor:
    """
    Function to calculate the Runge-Kutta fourth-roder step.
    
    ...
    
    Parameters
    ----------
    func :
        Lambda function for calculating the derivative of the dynamical system.
        
    x0 : np.ndarray
        Initial state array.
        
    dt : float, default=1e-3
        Timestep used for the integration.
        
    Returns
    -------
    x1 : np.ndarray
        Next step estimated from the RK-4 step.
        
    Notes
    -----
    ::math::
    
    k_{1} = f(x_{n})
    k_{2} = f(x_{n} + (dt * 0.5 * k_{1}))
    k_{3} = f(x_{n} + (dt * 0.5 * k_{2}))
    k_{4} = f(x_{n} + (dt * k_{3}))
    
    x_{n+1} = x_{n} + (dt/6) * (k_{1} + 2 * k_{2} + 2 * k_{3} + k_{4})
    Reference: https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
    
    """
    k1 = func(x0)
    k2 = func(x0 + (dt * 0.5 * k1))
    k3 = func(x0 + (dt * 0.5 * k2))
    k4 = func(x0 + (dt * k3))
    
    x1 = x0 + ((dt/6) * (k1 + (2 * k2) + (2 * k3) + k4))
    
    return x1

"""

DYNAMICS AND KINEMATICS FUNCTIONS


"""

def true_dyn_model(x0: torch.Tensor,
                   moi_inv: torch.Tensor,
                   seq_len: int = 2,
                   dt: float = 0.01) -> torch.Tensor:
    """
    True dynamics model.
    
    ...
    
    Parameters
    ----------
    x0 : torch.Tensor
        Initial state vector.
        
    moi_inv : torch.Tensor
        Inverse moment of inertia tensor.
        
    seq_len : int, default=2
        Prediction sequence length.
        
    dt : float, default=0.01
        Constant-valued time step.
        
    Returns
    -------
    pi_pred : torch.Tensor
        Predicted state vector sequence.
    
    Notes
    -----
    
    """
    output = [x0]
    
    for t in range(1, seq_len):
        x = output[-1]
        pi_next = euler_step(dy=euler_driver(pi=x, moi_inv=moi_inv), y0=x, dt=dt)
        output.append(pi_next)              
    
    pi_pred = torch.stack(output, axis=1)
    return pi_pred

def rotational_kinematics(R0: torch.Tensor,
                          pi_array: torch.Tensor,
                          moi_inv:torch.Tensor,
                          seq_len: int = 2,
                          dt: float = 0.01) -> torch.Tensor:
    """
    Function to calculate (orientation) state trajectories using rotational kinematics.
    
    ...
    
    Parameters
    ----------
    R0 : torch.Tensor
        Initial rotation matrix -- orientation.
        
    pi_array : torch.Tensor
        Array of sequential angular momentum state vectors.
    
    moi_inv : torch.Tensor
        Inverse moment of inertia tensor.
        
    seq_len : int, default=2
        Prediction sequence length.
        
    dt : float, default=0.01
        Constant-valued time step.
        
    Returns
    -------
    R_pred : torch.Tensor
        Array of predicted sequential rotation matrices -- orientaions.
        
    Notes
    -----
    
    """
    output = [R0]
    
    for t in range(1, seq_len):
        
        R = output[-1]
        pi = pi_array[:, t-1, ...]
        
        w = torch.squeeze(torch.matmul(moi_inv, pi.permute(0, 2, 1)))
        exp_R = lie_tools.rodrigues(v=dt* w)
        R_next = torch.bmm(exp_R, R)
        
        output.append(R_next)
    
    R_pred = torch.stack(output, axis=1)
    return R_pred

def generate_R_traj(R0: torch.Tensor,
                    pi_array: torch.Tensor,
                    moi_inv: torch.Tensor,
                    dt: float = 0.01) -> torch.Tensor:
    """
    Function to generate rotation trajectories given angular momentum trajectories.
    
    ...
    
    Parameters
    ----------
    R0 : torch.Tensor
        Initial rotation matrix -- orientation.
        
    pi_array : torch.Tensor
        Array of sequential angular momentum state vectors.
    
    moi_inv : torch.Tensor
        Inverse moment of inertia tensor.
        
    dt : float, default=0.01
        Constant-valued time step.
        
    Returns
    -------
    output : torch.Tensor
        Array of predicted sequential rotation matrices -- orientaions.
        
    Notes
    -----
    
    """
    output_ = [R0]
    bs, traj_len, _ = pi_array.shape
    
    for i in range(1, traj_len):
        x = output_[-1]
        w = torch.squeeze(torch.matmul(moi_inv, torch.unsqueeze(pi_array[:, i-1, :], axis=2)))
        exp_wdt = rodrigues(w * dt)
        output_.append(x.bmm(exp_wdt))
    
    output = torch.stack(output_, axis=1)
    return output

def similarity_metric_so3(R1, R2):
    """
    Similarity metric between two elements of SO(3).
    
    ...
    
    Parameters
    ----------
    R1 : torch.Tensor
        Initial orientation tensor
        
    R2 : torch.Tensor
        Final orientation tensor
        
    Returns
    -------
    similar : torch.Tensor
        
    Notes
    -----
    
    """
    Rprod = torch.einsum('bji, bjk -> bik', R2, R1)
    I = torch.eye(3, device=Rprod.device)[None, ...].repeat(Rprod.shape[0], 1, 1)
    loss = torch.mean((I - Rprod)**2, dim=(1, 2))
    
    similar = loss < 1e-3    
    return similar

def check_axis(R1: torch.Tensor,
               R2: torch.Tensor,
               mag: torch.Tensor,
               axis: torch.Tensor):
    """
    Function to check if the assigned axis is correct using the similarity metric.
    
    ...
    
    Parameters
    ----------
    R1 : torch.Tensor
        Initial orientation tensor
        
    R2 : torch.Tensor
        Final orientation tensor
        
    mag : torch.Tensor
        Magnitude of estimate angular velocity tensor.
        
    axis : torch.Tensor
        Unit direction of estimated angular velocity tensor.
    
    Returns
    -------
    
    Notes
    -----
    
    """
    Rprod_gt = torch.einsum('bij, bkj -> bik', R1, R2)
    Rprod_est = lie_tools.rodrigues(v=torch.einsum('bi, b -> bi', axis, mag))
    
    return similarity_metric_so3(Rprod_gt, Rprod_est)

def estimate_ang_vel(R1: torch.Tensor,
                     R2: torch.Tensor,
                     dt: float = 1e-3) -> torch.Tensor:
    """
    Function to calculate the angular velocity vector from two sequential orientation matrices in SO(3).
    
    ...
    
    Parameters
    ---------
    R1 : torch.Tensor
        Initial orientation tensor
        
    R2 : torch.Tensor
        Final orientation tensor
        
    dt : float, default=1e-3
        Constant-valued time step.
    
    Returns
    -------
    angular_vel
    
    Notes
    -----
    Adapted from Barfoot et al p.256.
    
    """
    assert torch.any(R1.isnan()) == False and torch.any(R1.isinf()) == False
    assert torch.any(R2.isnan()) == False and torch.any(R2.isinf()) == False
    assert dt > 0, "Timestep must be greater than 0."
    
    bs, _, _ = R1.shape
    R_prod = torch.einsum('bij, bkj -> bik', R2, R1)
    
    e, v = torch.linalg.eig(R_prod)
    a = e.imag == 0.
    a = a[:, None, :].repeat(1, 3, 1)
    v = v[a].reshape(bs, 3).real
    
    eps=1e-7
    
    # issues with phi being to close to zero
    cos = 0.5 * (torch.einsum('bii', R_prod) - 1)
    phi_int = torch.acos(cos.clamp(min=eps-1, max=1-eps))
    
    # ensure phi is not too close to zero but maintains correct sign
    phi = phi_int
    check_idx = check_axis(R1=R1, R2=R2, axis=v, mag=phi)
    phi[~check_idx] = - phi[~check_idx]
    
    angular_vel = torch.einsum('bj, b -> bj', v, phi/(2 * dt)) # put back the 2
    return angular_vel

def integrate_trajectories(moi_inv: torch.Tensor, x0: torch.Tensor, trajectory_length: int = 20, dt: float = 0.01) -> torch.Tensor:
    """
    Function to integrate trajectories given initial condition and trajectory length according to Euler's RBD.
    
    ...
    
    Parameters
    ----------
    moi_inv : torch.Tensor
        Inverse moment of inertia tensor.
        
    x0 : torch.Tensor
        Initial angular momentum state vector.
    
    trajectory_length : int, defualt=20
       Trajectory length. 
    
    dt : float, default=0.01
        Constant value time step.
        
    Returns
    -------
    output : torch.Tensor
        Generated state trajectories.
        
    Notes
    -----
    
    """
    output_ = [x0]
    for i in range(1, trajectory_length):
        x = output_[-1]
        dx = eulers_driver(moi_inv, x)
        output_.append(x + (dt * dx))
    
    output = torch.stack(output_, axis=1)
    return output

"""

NUMPY FUNCTIONS


"""

def generate_groundtruth(data_dir: str,
                         moi_diag: np.ndarray,
                         moi_off_diag: np.ndarray,
                         integrator,
                         timesteps: int):
    """
    Function to generate the ground-truth quaternion and angular velocity vectors for a given dataset.
    ...
    
    """
    # read in csv of initial conditions
    data_df = pd.read_csv(data_dir)
    
    # create initial conditions
    data_np = data_df.to_numpy()
    pi_vec = data_np[:, 4:7]
    state_vec = np.concatenate([pi_vec, np.zeros_like(pi_vec)], axis=1)
    
    euler_func = lambda x, t=0, moi_diag=np.array(moi_diag), moi_off_diag=np.array(moi_off_diag): \
                                                            eulers_driver(t=t, y=x, moi_diag=moi_diag, moi_off_diag=moi_off_diag)
    
    # integrate angular momentum
    data_final = []
    for i in range(state_vec.shape[0]):
        data = integrate_trajectory(integrator=integrator, func=euler_func, x0=state_vec[i, :], dt=1e-3, timesteps=timesteps)
        data_final.append(data.T)

    momentum_final = np.stack(data_final, axis=0)
    
    #integrate quaternion
    q0 = data_np[:, :4]
    quat_final = integrate_quat(q0=q0, pi_array=momentum_final[..., :3], moi_inv=np.diag([4., 2., 1.]), dt=1e-3)
    
    return quat_final, momentum_final

def quat_driver(q: np.ndarray, pi: np.ndarray, moi_inv: np.ndarray, dt: float = 1e-3) -> np.array:
    """
    """

    # interpolation on SO(3)
    w = np.einsum('ij, bi -> bj', moi_inv, pi)

    dq = lambda q : np.einsum('bij, bj -> bi', quat_omega(w=w), q)
    q_next = rk4_step(func=dq, x0=q, dt=dt)
    q_next = q_next/np.linalg.norm(q_next, ord=2, axis=-1,keepdims=True).clip(min=1E-5)
    
    return q_next

def quat_omega(w: np.ndarray):
    """
    """
    bs, _= w.shape
    S_w = map_to_lie_algebra(v=w)
    
    Q = np.zeros((bs, 4, 4))
    Q[:, :3, :3] = -S_w
    Q[:, -1, :3] = -w
    Q[:, :3, -1] = w
    
    return Q

def integrate_quat(q0: np.ndarray, pi_array: np.ndarray, moi_inv: np.ndarray, dt: float = 1e-3) -> np.ndarray:
    """
    """
    data_quat = [q0]
    n_timesteps = pi_array.shape[1]
    
    for i in range(n_timesteps-1):
        q = data_quat[-1]
        pi = pi_array[:, i, ...]
        q_next = quat_driver(q=q, pi=pi, moi_inv=moi_inv, dt=dt)
        
        data_quat.append(q_next)
    
    data_quat_ = np.stack(data_quat, axis=1)
    return data_quat_

def map_to_lie_algebra(v):
    """
    """
    S = np.zeros([v.shape[0], 3, 3])
    S[:, 0, 1] = -v[:, 2]
    S[:, 1, 0] = v[:, 2]
    S[:, 0, 2] = v[:, 1]
    S[:, 2, 0] = -v[:, 1]
    S[:, 1, 2] = -v[:, 0]
    S[:, 2, 1] = v[:, 0]
    
    return S

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
