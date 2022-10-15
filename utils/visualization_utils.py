# Author(s): Justice Mason
# Project : DEVS/RODEN
#

import sys
import os
import glob

import numpy as np
import torch
import pandas as pd
import plotly as py
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.spatial import Delaunay

from utils.math_utils import sample_points_on_sphere, pd_matrix
from utils.physics_utils import true_dyn_model
from data.data_utils import load_data, generate_devs_dl

# Plotly Express Functions

def dataset_np_to_df(data: np.ndarray):
    """
    Function to convert numpy array to pandas dataframe.
    
    ...
    
    Parameters
    ----------
    data : np.ndarray
    
    Returns
    -------
    df_final : pd.DataFrame
    
    Notes
    -----
    
    """
    df_data = []

    for i in range(data.shape[0]):
        df = pd.DataFrame(data= {"x": data[i, :, 0],
                                 "y": data[i, :, 1],
                                 "z": data[i, :, 2],
                                 "n_sample": ['traj-' + str(i)] * data.shape[1]})
        df_data.append(df)

    df_final = pd.concat(df_data)
    return df_final

def render_am_sphere(radius: float, title_str: str = "Angular Momentum Sphere" ):
    """
    Function to generate visualization for the angular momentum sphere.
    
    ...
    
    Paramters
    ---------
    radius : float
        Radius of the angular momentum sphere
    
    title_str : str, default='Angular Momentum Sphere'
        Title string
    
    Returns
    -------
    fig
    
    Notes
    -----
    
    """
    
    # angular momentum
    u = np.linspace(-np.pi/2, np.pi/2, 60)
    v = np.linspace(0, 2 * np.pi, 60)
    u, v = np.meshgrid(u, v)
    u = u.flatten()
    v = v.flatten()

    x_s = radius * np.cos(u) * np.sin(v)
    y_s = radius * np.sin(u) * np.sin(v)
    z_s = radius * np.cos(v)
    
    points2D = np.vstack([u, v]).T
    tri = Delaunay(points2D)
    simplices = tri.simplices
    
    fig = ff.create_trisurf(x=x_s, y=y_s, z=z_s,
                            colormap="rgb(40, 40, 40)",
                            show_colorbar=False,
                            simplices=simplices,
                            plot_edges=False,
                            title=title_str)
    
    fig.update_traces(opacity=0.25)

    # fixed points
    max_x = np.max(x_s)
    max_y = np.max(y_s)
    max_z = np.max(z_s)

    min_x = np.min(x_s)
    min_y = np.min(y_s)
    min_z = np.min(z_s)
    
    df_fixed = pd.DataFrame(data={"x": [0., 0., 0., 0., max_x, min_x],
                                  "y": [0., 0., max_y, min_y, 0., 0.],
                                  "z":[max_z, min_z, 0., 0., 0., 0.],
                                  'stability': ['stable', 'stable', 'unstable', \
                                                   'unstable', 'stable', 'stable']})
    
    fig_fixed = px.scatter_3d(df_fixed,  x="x", y="y", z="z",
                              color='stability',
                              color_discrete_map={'stable': 'lightgreen',\
                                              'unstable': 'orangered'})
    
    fig_fixed.update_traces(marker=dict(size=10, symbol='square'))
    fig.add_trace(fig_fixed.data[0])
    fig.add_trace(fig_fixed.data[1])
    
    return fig

def visualize_dataset(data: np.ndarray, radius: float):
    """
    Function to visualize the initial conditions of the angular momentum on the momentum sphere from the dataset.
    
    ...
    
    Parameters
    ----------
    data : np.ndarray
        Data from dataset to be visualized.
    
    radius : float,
        Radius of the angular momentum sphere.
    
    Notes
    -----
    """
    
    # create momentum sphere object
    fig = render_am_sphere(radius=radius, title_str='Dataset Visualized on Angular Momentum Sphere')
    
    # create dataset objects
    df = pd.DataFrame(data)
    df.rename(columns={0:"x", 1:"y", 2:"z"}, inplace=True)
    
    fig_scatter = px.scatter_3d(df, x="x", y="y", z="z")
    fig_scatter.update_traces(marker=dict(size=5,
                                          color= 'peachpuff'))
    
    # visualize
    fig.add_trace(fig_scatter.data[0])
    return fig

def visualize_multidataset(radius: float, train_dataset: np.ndarray, test_dataset: np.ndarray = None, val_dataset: np.ndarray = None):
    """
    Function to visualize the initial conditions of the angular momentum on the momentum sphere from the dataset.
    
    ...
    
    Parameters
    ----------
    data : np.ndarray
        Data from dataset to be visualized.
    
    radius : float,
        Radius of the angular momentum sphere.
    
    Notes
    -----
    """
    
    # create momentum sphere and fixed points
    fig = render_am_sphere(radius=radius, title_str='ICs on Angular Momentum Sphere')
    df_list = []
    
    # create dataset objects
    df_train = pd.DataFrame(train_dataset)
    df_train.rename(columns={0:"x", 1:"y", 2:"z"}, inplace=True)
    df_train.insert(3, 'dataset', ['train'] * len(train_dataset), True)
    
    df_list.append(df_train)
    
    if test_dataset is not None:
        df_test = pd.DataFrame(test_dataset)
        df_test.rename(columns={0:"x", 1:"y", 2:"z"}, inplace=True)
        df_test.insert(3, 'dataset', ['test'] * len(test_dataset), True)
        
        df_list.append(df_test)
    
    if val_dataset is not None:
        df_val = pd.DataFrame(val_dataset)
        df_val.rename(columns={0:"x", 1:"y", 2:"z"}, inplace=True)
        df_val.insert(3, 'dataset', ['val'] * len(val_dataset), True)
        
        df_list.append(df_val)
    
    df = pd.concat(df_list)
    fig_scatter = px.scatter_3d(df, x="x", y="y", z="z",
                                color='dataset',
                                color_discrete_map={'train': 'skyblue', 'test': 'orange', 'val': 'yellow'})
    
    fig_scatter.update_traces(marker=dict(size=5))
    
    for i in range(len(df_list)):
        fig.add_trace(fig_scatter.data[i])
        
    return fig 

def plot_traj_on_sphere(data: np.ndarray, radius: float):
    """
    Function to plot collection of trajectories on momentum sphere.
    
    ...
    
    Parameters
    ----------
    data : np.ndarray
        Data from dataset to be visualized.
    
    radius : float,
        Radius of the angular momentum sphere.
    
    Notes
    -----
    
    """
    # momentum sphere and fixed points
    fig = render_am_sphere(radius=radius, title_str='Trajectory Dataset on Angular Momentum Sphere') 

    # trajectories
    df_data = dataset_np_to_df(data=data)
    fig_traj = px.line_3d(df_data, x="x", y="y", z="z", color='n_sample', line_dash_sequence=['dash'])
    
    # render figure
    for i in range(data.shape[0]):
        fig.add_trace(fig_traj.data[i])
    
    return fig

def animate_trajectory(data: np.ndarray):
    """
    Function to generate an animation of the dataset.
    
    ...
    
    Parameters
    ----------
    data : np.ndarray
        Numpy data array for the image dataset.
    
    Returns
    -------
    fig
        Plotly figure for animation.
        
    Notes
    -----
    
    """
    data_t = data.transpose((1, 0, 3, 4, 2))
    fig = px.imshow(data_t, animation_frame=0, facet_col=1, labels=dict(animation_frame="timestep", facet_col="n_traj"))
    fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 30
    fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 5
    
    return fig

def plot_loss_px(stats: dict):
    """
    Function for plotting the loss curves after training in Plotly Express.
    
    ...
    
    Parameters
    ----------
    Returns
    -------
    Notes
    -----
    
    """
    fig = make_subplots(rows=2, cols=2, subplot_titles=("DEVS Training Loss",
                                                        "DEVS Validation Loss",
                                                        "DEVS Separated Training Loss",
                                                        "DEVS Separated Validation Loss"))
    
    # define train dataframes
    
    train_tot_df = pd.DataFrame(data = {'loss' : stats['train loss'], 'type': ['total'] * len(stats['train loss'])})
    train_ae_df = pd.DataFrame(data = {'loss' : stats['train ae loss'], 'type': ['auto-encoder'] * len(stats['train ae loss'])})
    train_dyn_df = pd.DataFrame(data = {'loss' : stats['train dyn loss'], 'type': ['dynamics'] * len(stats['train dyn loss'])})
    train_latent_df = pd.DataFrame(data = {'loss' : stats['train latent loss'], 'type': ['latent'] * len(stats['train latent loss'])})
    train_df = pd.concat([train_ae_df, train_dyn_df, train_latent_df])
    
    # define validation dataframes
    
    val_tot_df = pd.DataFrame(data = {'loss' : stats['val loss'], 'dataset': ['total'] * len(stats['val loss'])})
    val_ae_df = pd.DataFrame(data = {'loss' : stats['val ae loss'], 'dataset': ['auto-encoder'] * len(stats['val ae loss'])})
    val_dyn_df = pd.DataFrame(data = {'loss' : stats['val dyn loss'], 'dataset': ['dynamics'] * len(stats['val dyn loss'])})
    val_latent_df = pd.DataFrame(data = {'loss' : stats['val latent loss'], 'dataset': ['latent'] * len(stats['val latent loss'])})
    val_df = pd.concat([val_ae_df, val_dyn_df, val_latent_df])
    
    # add trace to subplot figure
                       
    fig.add_trace(px.line(train_tot_df, log_y=True), row=1, col=1)
    fig.add_trace(px.line(train_df, color='type', log_y=True), row=2, col=1)
    fig.add_trace(px.line(val_tot_df, log_y=True), row=1, col=2)
    fig.add_trace(px.line(val_df, color='type', log_y=True), row=2, col=2)
    
    return fig

# Matplotlib Functions

def plot_loss(stats: dict):
    """
    Function for plotting the loss curves after training.
    
    ...
    
    Parameters
    ----------
    Returns
    -------
    Notes
    -----
    
    """
    fig, ax = plt.subplots(nrows=2, ncols=2, constrained_layout=True)
    
    ax[0, 0].plot(np.log(stats['train loss']))
    ax[0, 0].grid()
    ax[0, 0].set_xlabel('Iterations [-]')
    ax[0, 0].set_ylabel('Log Loss [-]')
    ax[0, 0].set_title('DEVS Training Loss')
    
    ax[0, 1].plot(np.log(stats['val loss']))
    ax[0, 1].grid()
    ax[0, 1].set_xlabel('Iterations [-]')
    ax[0, 1].set_ylabel('Log Loss [-]')
    ax[0, 1].set_title('DEVS Validation Loss')
    
    ax[1, 0].plot(np.log(stats['train ae loss']), label='Autoencoder')
    ax[1, 0].plot(np.log(stats['train dyn loss']), label='Dynamics')
    ax[1, 0].plot(np.log(stats['train latent loss']), label='Latent')
    ax[1, 0].grid()
    ax[1, 0].set_xlabel('Iterations [-]')
    ax[1, 0].set_ylabel('Log Loss [-]')
    ax[1, 0].set_title("DEVS Separated Training Loss")
    ax[1, 0].legend(loc = 'upper right')
    
    ax[1, 1].plot(np.log(stats['val ae loss']), label='Autoencoder')
    ax[1, 1].plot(np.log(stats['val dyn loss']), label='Dynamics')
    ax[1, 1].plot(np.log(stats['val latent loss']), label='Latent')
    ax[1, 1].grid()
    ax[1, 1].set_xlabel('Iterations [-]')
    ax[1, 1].set_ylabel('Log Loss [-]')
    ax[1, 1].set_title("DEVS Separated Validation Loss")
    ax[1, 1].legend(loc = 'upper right')
    
    return fig, ax

def latent_eval(model, true_moi_inv: torch.Tensor, seq_len: int = 15, dt: float = 1e-3):
    """
    Function to visualize/evaluate the model's learned latent space dynamics on known trajectories.
    
    ...
    
    Parameters
    ----------
    model
    true_moi_inv
    seq_len
    dt
    
    Returns
    -------
    fig
    
    Notes
    -----
    
    """
    theta_eps = 0.001 * (2 * np.pi)
    phi_eps = 0.001 * (np.pi)
    
    radius = 40.0
    theta_arr = np.array([[0.0 + theta_eps, np.pi/2 + theta_eps, np.pi/2 + theta_eps]])
    phi_arr = np.array([[np.pi/2 + phi_eps, np.pi/2 + phi_eps, 0.0 + phi_eps]])
    
    eval_pts = sample_points_on_sphere(radius=radius, theta_arr=theta_arr, phi_arr=phi_arr)
    eval_tensor = torch.Tensor(eval_pts)
    
    # generate ground-truth trajectories in angular-momentum-space
    
    N = eval_tensor.shape[0]
    true_latent_traj = torch.zeros(N, seq_len, 3) 
    
    for n in range(N):
        x0_n = eval_tensor[n, ...].unsqueeze(0).unsqueeze(0)
        x_traj_n = true_dyn_model(x0 = x0_n, moi_inv=true_moi_inv, seq_len=seq_len, dt=dt)
        true_latent_traj[n, ...] = x_traj_n.squeeze()
    
    # generate estimates for trajectories in angular-momentum-space using learned dynamics
    
    est_latent_traj = torch.zeros(N, seq_len, 3)
    est_moi_inv = pd_matrix(diag=model.moi_diag, off_diag=model.moi_off_diag)
    print('Estimated Inverse of MOI: ', est_moi_inv)
    
    model.eval()
    
    with torch.no_grad():
        for m in range(N):
            x0_m = eval_tensor[m, ...].unsqueeze(0).unsqueeze(0)
            x_traj_m = model.dyn_net(x0=x0_m, moi_inv=est_moi_inv, seq_len=seq_len, dt=dt)
            est_latent_traj[m, ...] = x_traj_m.squeeze()
    
    pi_true = true_latent_traj.detach().numpy()
    pi_est = est_latent_traj.detach().numpy()
    
    print('shape of ')
    fig = plt.figure(constrained_layout=True)
    fig.suptitle('Learned Latent Space Evaluation')
    
    ax1 = fig.add_subplot(131,projection='3d')
    ax2 = fig.add_subplot(132,projection='3d')
    ax3 = fig.add_subplot(133,projection='3d')
    
    ax1.plot(pi_true[0, :, 0], pi_true[0, :, 1], pi_true[0, :, 2], label='True Dynamics')
    ax1.plot(pi_est[0, :, 0], pi_est[0, :, 1], pi_est[0, :, 2], label='Learned Dynamics')
    ax1.grid()
    ax1.set_xlabel('$\Pi_{1}$')
    ax1.set_ylabel('$\Pi_{2}$')
    ax1.set_zlabel('$\Pi_{3}$')
    ax1.set_xlim(-1.5*radius, 1.5*radius)
    ax1.set_ylim(-1.5*radius, 1.5*radius)
    ax1.set_zlim(-1.5*radius, 1.5*radius)
    ax1.set_title('$\Pi_{1}$ Fixed Pt')
    ax1.legend()
    
    ax2.plot(pi_true[1, :, 0], pi_true[1, :, 1], pi_true[1, :, 2], label='True Dynamics')
    ax2.plot(pi_est[1, :, 0], pi_est[1, :, 1], pi_est[1, :, 2], label='Learned Dynamics')
    ax2.grid()
    ax2.set_xlabel('$\Pi_{1}$')
    ax2.set_ylabel('$\Pi_{2}$')
    ax2.set_zlabel('$\Pi_{3}$')
    ax2.set_xlim(-1.5*radius, 1.5*radius)
    ax2.set_ylim(-1.5*radius, 1.5*radius)
    ax2.set_zlim(-1.5*radius, 1.5*radius)
    ax2.set_title('$\Pi_{2}$ Fixed Pt')
    ax2.legend()
    
    ax3.plot(pi_true[2, :, 0], pi_true[2, :, 1], pi_true[2, :, 2], label='True Dynamics')
    ax3.plot(pi_est[2, :, 0], pi_est[2, :, 1], pi_est[2, :, 2], label='Learned Dynamics')
    ax3.grid()
    ax3.set_xlabel('$\Pi_{1}$')
    ax3.set_ylabel('$\Pi_{2}$')
    ax3.set_zlabel('$\Pi_{3}$')
    ax3.set_xlim(-1.5*radius, 1.5*radius)
    ax3.set_ylim(-1.5*radius, 1.5*radius)
    ax3.set_zlim(-1.5*radius, 1.5*radius)
    ax3.set_title('$\Pi_{3}$ Fixed Pt')
    ax3.legend()
    
    return fig

def recon_eval(params, model, x_train, x_val, x_test):
    """
    Function to visualize/evaluate the trained model's ability to reconstruct and predict image sequences.
    
    ...
    
    Parameters
    ----------
    model
    seq_len
    dt
    
    Returns
    -------
    fig
    
    Notes
    -----
    
    """
    torch.manual_seed(params.seed)
    device = torch.device(params.gpu if torch.cuda.is_available() else "cpu")
    
    # load dataloaders
    print('\n Loading data for model evaluation ... \n')
    #devs_traindl, devs_testdl, devs_valdl = generate_devs_dl(args=params, data_dir=params.data_dir)
    
    #x_train = next(iter(devs_traindl)).to(device)
    #x_val = next(iter(devs_valdl)).to(device)
    #x_test = next(iter(devs_testdl)).to(device)
    
    # switch model to evaluation mode
    model.eval()
    
    x_tr_recon, x_tr_pred, _, _, _, _ = model(x_train.float(), seq_len=int(1.0 * params.seq_len), dt=params.dt)
    x_val_recon, x_val_pred, _, _, _, _ = model(x_val.float(), seq_len=int(1.0 * params.seq_len), dt=params.dt)
    x_ts_recon, x_ts_pred, _, _, _, _ = model(x_test.float(), seq_len=int(1.0 * params.seq_len), dt=params.dt)
    
    # setting up the figure
    n_row = 9
    n_col = int(1.0 * params.seq_len) - 1 # params.seq_len-1
    fig_width = 3 * n_col
    fig_height = 3 * n_row
    
    fig, axes = plt.subplots(n_row, n_col, figsize=(fig_width, fig_height))
    
    # organizing trajectories and converting them to numpy arrays
    t1 = x_train[0, ...].cpu().detach().numpy()
    t1_r = x_tr_recon[0, ...].cpu().detach().numpy()
    t1_p = x_tr_pred[0, ...].cpu().detach().numpy()
    
    t2 = x_val[0, ...].cpu().detach().numpy()
    t2_r = x_val_recon[0, ...].cpu().detach().numpy()
    t2_p = x_val_pred[0, ...].cpu().detach().numpy()
    
    t3 = x_test[0, ...].cpu().detach().numpy()
    t3_r = x_ts_recon[0, ...].cpu().detach().numpy()
    t3_p = x_ts_pred[0, ...].cpu().detach().numpy()
    
    trajs = [t1, t1_r, t1_p, t2, t2_r, t2_p, t3, t3_r, t3_p]
    
    # each row will be a trajectory. Here, we iterate through all the rows
    for row_idx, ax_row in enumerate(axes):

        # extract trajectory
        traj = trajs[row_idx]

        # now, we iterate through each column and plot
        for col_idx, ax in enumerate(ax_row):
            img = traj[col_idx]

            # ax.imshow() will display the image on the given <AxesSubplot> object
            ax.imshow(img.transpose((1,2,0)))

            # we can also customize on an axis by axis basis.
            # here, we can turn off the ticks and grids
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            
    # now just because we iterated through it doesn't mean each axis is gone. They're still stored in
    # axes. So we can go back and change some things. For example, suppose we wanted to add titles for the
    # top and left of the rows, we can follow the guide here:
    # ref: https://stackoverflow.com/questions/25812255/row-and-column-headers-in-matplotlibs-subplots 
    
    pad = 5
    fontsize = 20

    # lists of labels to use
    row_names = ['tr_gt', 'tr_recon', 'tr_pred', 'val_gt', 'val_recon', 'val_pred', 'ts_gt', 'ts_recon', 'ts_pred'] #[f'traj={idx}' for idx in range(axes.shape[0])]
    col_names = [f't={idx}' for idx in range(axes.shape[1])]

    # create the column labels
    for col_name, ax in zip(col_names, axes[0]):
        ax.annotate(col_name, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size=fontsize, ha='center', va='baseline')

    # create the row labels
    for row_name, ax in zip(row_names, axes[:,0]):
        ax.annotate(row_name, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size=fontsize, ha='right', va='center')

    return fig, axes

def plot_pred_output(args, model, test_dataloader):
    plt.figure(figsize=(16,4.5))
    
    test_data = next(iter(test_dataloader))
    x = test_data[0, ...].unsqueeze(0).to(args.device)
    
    model.eval()
    with torch.no_grad():
        _, xhat_pred, _, _, _, _  = model(x.float(), seq_len=args.seq_len, dt=args.dt)
    
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
        img_pred = xhat_pred[:, i, ...]
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
        xhat_recon, _, _, _, _, _  = model(x.float(), seq_len=args.seq_len, dt=args.dt)
    
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
        img_recon = xhat_recon[:, i, ...]
        plt.imshow(img_recon.cpu().squeeze().permute(1, 2, 0).numpy()) #, cmap='gist_gray')   plt.
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)  
        
        if i == seq_len//2:
            ax.set_title('AE Reconstructed Images')
    
    plt.show()
    return ax

