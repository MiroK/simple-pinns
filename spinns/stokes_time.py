import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

from calculus import div, grad, transpose, dot
from utils import sample_hypercube

import matplotlib.pyplot as plt
import numpy as np
import signal

# Space time Stokes where we solve
#
# du/dt = Delta u - grad(p)  in (0, 1)^2 x (0, 1)
# div(u) = 0
#
# With some initial and boundary conditions

def split(tensor):
    '''Split and glue space time points'''
    # Input to our networks are space time points xt. However, the way I 
    # eval derivative it is convenient to talk about xt as (x, t), i.e. 
    # referring to x and t. The problem with this is that the compute graph
    # looks like this and so dN(xt)/dx is None for pytorch
    #
    #
    # To avoid the issue we glue xt from x,t, that is,
    #
    #        x >-\
    #            xt --> N(xt)
    #        t >-/
    #
    tensor_x, tensor_t = tensor[..., [0, 1]], tensor[..., 2]
    tensor_x.requires_grad, tensor_t.requires_grad = True, True
    tensor = torch.column_stack([tensor_x.squeeze(0), tensor_t.squeeze(0)]).unsqueeze(0)

    return tensor, tensor_x, tensor_t


torch.manual_seed(46)
np.random.seed(46)

# Only take space part
vel_data = lambda x: torch.zeros_like(x[..., [0, 1]])

pLeft = lambda x: torch.ones_like(x[..., 0])
pRight = lambda x: torch.zeros_like(x[..., 0])

pressure_data = {1: pLeft, 2: pRight}
velocity_data = {3: vel_data, 4: vel_data}

# --------------------------------------------------------------------

class VelocityNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # 2 + 1 for space time
        self.lin1 = nn.Linear(3, 48)
        self.lin2 = nn.Linear(48, 48)
        self.lin3 = nn.Linear(48, 32)        
        self.lin4 = nn.Linear(32, 2)
 
    def forward(self, xt):
        y = self.lin1(xt)
        y = torch.tanh(y)
        y = self.lin2(y)
        y = torch.tanh(y)
        y = self.lin3(y)
        y = torch.tanh(y)        
        y = self.lin4(y)

        return y

    
class PressureNetwork(nn.Module):
    def __init__(self):
        super().__init__()
         # 2 + 1 for space time
        self.lin1 = nn.Linear(3, 48)
        self.lin2 = nn.Linear(48, 32)
        self.lin3 = nn.Linear(32, 32)        
        self.lin4 = nn.Linear(32, 1)
 
    def forward(self, xt):
        y = self.lin1(xt)
        y = torch.tanh(y)
        y = self.lin2(y)
        y = torch.tanh(y)
        y = self.lin3(y)
        y = torch.tanh(y)        
        y = self.lin4(y)

        y = y.squeeze(2)
        
        return y
    
u = VelocityNetwork()
u.double()

p = PressureNetwork()
p.double()

params = list(u.parameters()) + list(p.parameters())

maxiter = 1_000
optimizer = optim.LBFGS(params, max_iter=maxiter,
                        history_size=1000, tolerance_change=1e-12,
                        line_search_fn="strong_wolfe")

nvol_pts = 2_000
# Interior space time points
volume_xt = sample_hypercube(dim=2+1, npts=nvol_pts)
volume_xt, volume_x, volume_t = split(volume_xt)

# Space boundary of space time
nbdry_pts = 100
# We have space as the first 2 axis
bdry_xts = [sample_hypercube(dim=2+1, npts=nbdry_pts, fixed=[fixed])
           for fixed in ((0, 0), (0, 1), (1, 0), (1, 1))]
[setattr(bdry_xt, 'requires_grad', True) for bdry_xt in bdry_xts]
left_xt, right_xt, bot_xt, top_xt = bdry_xts

nic_pts = 100
# For initial condition; time is the final axis
ic_xt = sample_hypercube(dim=2+1, npts=nic_pts, fixed=[(2, 0)])
ic_xt.requires_grad = True


epoch_loss = []
def closure(history=epoch_loss):
    optimizer.zero_grad()
    # Space time
    U, P = u(volume_xt), p(volume_xt)

    # Space
    grad_u = grad(U, volume_x)
    delta_u = div(grad_u, volume_x)

    grad_p = grad(P, volume_x)
    div_u = div(U, volume_x)

    # Time
    du_dt = grad(U, volume_t)

    pde_loss = (
        ((du_dt + delta_u - grad_p)**2).mean()
        + ((div_u)**2).mean()
    )
    
    bdry_loss = (
        ((u(bot_xt) - vel_data(bot_xt))**2).mean() + 
        ((u(top_xt) - vel_data(top_xt))**2).mean() + 
        ((p(left_xt) - pLeft(left_xt))**2).mean() +
        ((p(right_xt) - pRight(right_xt))**2).mean()
    )

    # Start from 0 velocity and pressure
    ic_loss = (
        ((u(ic_xt) - vel_data(ic_xt))**2).mean()
        + ((p(ic_xt) - pLeft(ic_xt))**2).mean()
    )
    
    loss = pde_loss + bdry_loss + ic_loss

    print(f'Loss @ {len(history)} = {float(loss)}')
    loss.backward()

    history.append((float(loss), ))
    
    return loss


def viz(u, p, history):
    xi = torch.linspace(0, 1, 100, dtype=torch.float64)
    X, Y = torch.meshgrid(xi, xi)
    X_, Y_ = X.flatten(), Y.flatten()
    # We are plotting at t = 1
    grid = torch.column_stack([X_, Y_, torch.ones_like(X_)])
    grid = grid.unsqueeze(0)

    with torch.no_grad():
        vals_p = p(grid)
        vals_p = vals_p.reshape(X.shape)

        vals_u = u(grid)
        vals_u0 = vals_u[..., 0].reshape(X.shape)
        vals_u1 = vals_u[..., 1].reshape(X.shape)        

    X, Y, vals_p, vals_u0, vals_u1 = (thing.numpy()
                                      for thing in (X, Y, vals_p, vals_u0, vals_u1))

    fig, ax = plt.subplots(1, 2)
    ax[0].pcolor(X, Y, np.sqrt(vals_u0**2 + vals_u1**2))
    ax[0].quiver(X, Y, vals_u0, vals_u1)

    ax[1].pcolor(X, Y, vals_p)

    fig, ax = plt.subplots()
    ax.semilogy(np.arange(1, len(history)+1), history)
    
    plt.show()

# For some strange reason (I suspect gmsh threads are involved) it is
# hard to catch KeryboardInterrupt here so
def interuppt_handler(signum, frame):
   print('Caught CTRL+C!!!')
   raise AssertionError

# Interpret CTRL+Z 
def sleep_handler(signum, frame, nn=(u, p), history=epoch_loss):
   viz(u, p, history)

try:
    epoch_loss.clear()

    signal.signal(signal.SIGINT, interuppt_handler)
    signal.signal(signal.SIGTSTP, sleep_handler)
    
    epochs = 1
    for epoch in range(epochs):
        print(f'Epoch = {epoch}')
        optimizer.step(closure)
        
except AssertionError:
    pass

closure()

viz(u, p, epoch_loss)

print('Done')
