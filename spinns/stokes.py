import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

from calculus import div, grad, transpose, dot
from utils import sample_hypercube

import matplotlib.pyplot as plt
import numpy as np
import signal

torch.manual_seed(46)
np.random.seed(46)

# We are solving
#   -Delta u + grad(p) = 0
#   div(u)             = 0 in (0, 1)^2
#
# With boundary conditions u = u0 on some boundary piece and p = p0 elsewhere

# Specifically we will have noslip so u0 is x -> 0*x (to get the shape)
vel_data = lambda x: torch.zeros_like(x)
# And we impose pressure gradient 
pLeft = lambda x: torch.ones_like(x[..., 0])
pRight = lambda x: torch.zeros_like(x[..., 0])

# --------------------------------------------------------------------

class VelocityNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Takes 2d spatial points
        self.lin1 = nn.Linear(2, 48)
        self.lin2 = nn.Linear(48, 48)
        self.lin3 = nn.Linear(48, 32)        
        self.lin4 = nn.Linear(32, 2)  # And outputs a vector
 
    def forward(self, x):
        y = self.lin1(x)
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
        # Takes 2d spatial points
        self.lin1 = nn.Linear(2, 48)
        self.lin2 = nn.Linear(48, 32)
        self.lin3 = nn.Linear(32, 32)        
        self.lin4 = nn.Linear(32, 1)  # And outputs a scalar
 
    def forward(self, x):
        y = self.lin1(x)
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

# We are optimizing for parameters of both networks
params = list(u.parameters()) + list(p.parameters())

maxiter = 1000
optimizer = optim.LBFGS(params, max_iter=maxiter,
                        history_size=1000, tolerance_change=1e-12,
                        line_search_fn="strong_wolfe")

# Our loss function will be the PDE residual evaluated inside the domain
# and the residual in the boundary conditions
nvol_pts = 1000
# So we sample (0, 1)^2 to get points for the PDE residual evaluation
volume_x = sample_hypercube(2, nvol_pts)
volume_x.requires_grad = True

nbdry_pts = 100
# And analogously for the boundary
bdry_xs = [sample_hypercube(2, nbdry_pts, fixed=[fixed])
           for fixed in ((0, 0), (0, 1), (1, 0), (1, 1))]
[setattr(bdry_x, 'requires_grad', True) for bdry_x in bdry_xs]
# Expand
left_x, right_x, bot_x, top_x = bdry_xs

epoch_loss = []
# Steps of the optimizer compute the forward pass to compute the loss
# and then backpropagate to get the gradient wrt to weights
def closure(history=epoch_loss):
    optimizer.zero_grad()

    grad_u = grad(u(volume_x), volume_x)
    delta_u = div(grad_u, volume_x)

    grad_p = grad(p(volume_x), volume_x)
    div_u = div(u(volume_x), volume_x)
    
    pde_loss = (((-delta_u + grad_p)**2).mean()
                + ((div_u)**2).mean())
    
    bdry_loss = (((u(bot_x) - vel_data(bot_x))**2).mean()
                 + ((u(top_x) - vel_data(top_x))**2).mean()
                 + ((p(left_x) - pLeft(left_x))**2).mean()
                 + ((p(right_x) - pRight(right_x))**2).mean())

    # NOTE: these terms could be differently weighted
    loss = pde_loss + bdry_loss

    print(f'Loss @ {len(history)} = {float(loss)}')
    loss.backward()

    history.append((float(loss), ))
    
    return loss


def viz(u, p, history):
    '''Plot the solution on a uniform grid'''
    xi = torch.linspace(0, 1, 100, dtype=torch.float64)
    X, Y = torch.meshgrid(xi, xi)
    X_, Y_ = X.flatten(), Y.flatten()
    grid = torch.column_stack([X_, Y_])
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

# Interpret CTRL+Z: 
def sleep_handler(signum, frame, nn=(u, p), history=epoch_loss):
   viz(u, p, history)

try:
    epoch_loss.clear()

    signal.signal(signal.SIGINT, interuppt_handler)
    # CTRL+Z will plot the solution at this stage of training and then
    # we can resume
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
