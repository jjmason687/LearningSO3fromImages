import argparse
import torch
import numpy as np
from tempfile import NamedTemporaryFile
from subprocess import call, PIPE

from models.lie_tools import random_group_matrices, rodrigues, group_matrix_to_quaternions
from utils.integration_utils import integrate_trajectories
from data.data_utils import generate_random_am, generate_R_traj

def generate(n_traj, radius, moi_inv, step_size, traj_len, dir, size=64, tmppath=None, silent=True):
    """
    """
    moi_inv = torch.diag(torch.Tensor(moi_inv))
    pi0 = generate_random_am(n_trajectories=n_traj, radius_ams=radius)
    pi = integrate_trajectories(moi_inv=moi_inv, x0=pi0, trajectory_length=traj_len, dt=step_size)
    
    r0 = random_group_matrices(n_traj)
    r = generate_R_traj(R0=r0, pi_array=pi, moi_inv=moi_inv, dt=step_size)
    q = group_matrix_to_quaternions(r)

    names = [['{:06}_{:06}_{:.4f}_{:.4f}_{:.4f}_{:.4f}'.format(i+1, j+1, *q[i, j])
              for j in range(traj_len)] for i in range(n_traj)]
    names = np.array(names)

    data = np.zeros((n_traj, traj_len), dtype=[('quaternion', 'f4', (4,)), ('name', 'a50')])
    data['quaternion'] = q
    data['name'] = names
    data = data.flatten()

    datafile = NamedTemporaryFile(dir=tmppath)
    np.save(datafile, data)
    datafile.flush()

    call(['blender', '--background', '--python', 'blender_spherecube.py',
          '--', str(n_traj), dir, '--quaternions', datafile.name, '--size', str(size)],
         stdout=(PIPE if silent else None))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_traj', type=int)
    parser.add_argument('--radius', type=float)
    parser.add_argument('--traj_len', type=int)
    parser.add_argument('--dir')
    parser.add_argument('--moi_inv',nargs='+', type=float,
                    help='Array of floats')
    parser.add_argument('--step_size', type=float, default=2 * np.pi / 60)
    parser.add_argument('--size', type=int, default=64)
    args = parser.parse_args()
    generate(n_traj=args.n_traj, radius=args.radius, moi_inv=args.moi_inv, step_size=args.step_size, traj_len=args.traj_len, dir=args.dir, size=args.size, silent=False)


if __name__ == '__main__':
    main()