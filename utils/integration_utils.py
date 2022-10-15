# Author(s): Justice Mason
# Project: DEVS/RODE
# package: Integration Utilities
# Date: 12/17/21

import numpy as np
import torch

from utils.physics_utils import eulers_driver

class Integrator:
    """
    Integrator class for computing trajectories.
    
    ...
    
    Attributes
    ----------
    dt : float, default=1e-3
            Timestep used for the integration.
    
    method : str, default='euler'
        Integration method used.
        
    Methods
    -------
    euler_step
    rk4_step
    
    Notes
    -----
    
    """
    METHODS = ['euler', 'rk4']
    
    def __init__(self, dt: float = 1e-3, method: str = 'euler') -> None:
        super(Integrator, self).__init__()
        self.dt = dt
        
        if method.lower() not in self.METHODS:
            msg = method.lower() + " is not in our  METHODS."
            msg +=  " Choose one of these: " + str(self.METHODS) + " ."
            raise KeyError(msg)
            
        else:
            self.method = method
            
    def euler_step(self, func, x0: np.ndarray) -> np.ndarray:
        """
        Calculates the next step using explicit first-order Euler method.
    
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
            Next step estimated from the Euler step.

        Notes
        -----
        ::math::
        y_{n+1} = y_{n} + dt * f(y_{n}, t_{n})

        Reference: https://en.wikipedia.org/wiki/Euler_method

        """
        x1 = x0 + (self.dt * func(x0))
        return x1
    
    def rk4_step(self, func, x0: np.ndarray) -> np.ndarray:
        """
        Function to calculate the Runge-Kutta fourth-roder step.

        ...

        Parameters
        ----------
        func :
            Lambda function for calculating the derivative of the dynamical system.

        x0 : np.ndarray
            Initial state array.

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
        k2 = func(x0 + (self.dt * 0.5 * k1))
        k3 = func(x0 + (self.dt * 0.5 * k2))
        k4 = func(x0 + (self.dt * k3))

        x1 = x0 + (self.dt/6) * (k1 + (2 * k2) + (2 * k3) + k4)

        return x1
        
    def step(self, eval_fcn, x0: np.ndarray) -> np.ndarray:
        """
        "Forward" method to calculate the next step.

        ...

        Parameters
        ----------
        eval_fcn :
            Lambda function for calculating the derivative of the dynamical system.

        x0 : np.ndarray
            Initial state array.

        Returns
        -------
        x1 : np.ndarray
            Next step estimated from the RK-4 step.

        """

        if self.method == 'euler':
            x1 = self.euler_step(func=eval_fcn, x0=x0)
        elif self.method == 'rk4':
            x1 = self.rk4_step(func=eval_fcn, x0=x0)
            
        return x1
    

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

def trajectory_integrate(func, y0: torch.Tensor, dt: float = 0.01, timesteps: int = 20, **kwargs) -> torch.Tensor:
    """
    """
    
    output_ = [y0]
    for t in range(timesteps):
        y = output_[-1]
        output_.append(func(y0=y0))
    
    y = torch.stack(output_, axis=1)
    
    return y