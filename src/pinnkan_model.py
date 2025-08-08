import torch
import torch.nn as nn
from src.utils import *
from kan import *

class PyKAN_PINN(nn.Module):
    """
    PINN using PyKAN library - most popular and well-tested KAN implementation
    """
    def __init__(self, inputDim, arch, outputDim, bc, 
                 grid=3, k=3, seed=0, device='cuda'):
        super(PyKAN_PINN, self).__init__()
        
        self.bc = bc
        self.device = device
        
        # Convert architecture to KAN format
        # PyKAN expects [input_dim, hidden1, hidden2, ..., output_dim]
        kan_arch = [inputDim]
        
        # Extract only the integer dimensions from arch (skip activation strings)
        for layer_spec in arch:
            if isinstance(layer_spec, int):
                kan_arch.append(layer_spec)
        
        kan_arch.append(outputDim)
        
        # Create KAN model
        self.model = KAN(
            width=kan_arch,
            grid=grid,
            k=k,
            seed=seed,
            device=device
        )
        
        # Move to device
        self.to(device)
    
    def forward(self, x):
        # Get KAN output
        out = self.model(x)
        u = torch.zeros(len(x), 5, device=self.device)
        
        # Apply boundary conditions (same as original PINN)
        if self.bc == 'ls':  # left side clamp 
            trial_func = x[:, 0] + 0.5
        elif self.bc == 'rs':  # right side clamp
            trial_func = x[:, 0] - 0.5
        elif self.bc == 'ls-rs':
            trial_func = (x[:, 0] + 0.5) * (x[:, 0] - 0.5)
        elif self.bc == 'fc':
            trial_func = (x[:, 0]**2 - 0.25) * (x[:, 1]**2 - 0.25)
        elif self.bc == 'scordelis_lo':
            u[:, 0] = out[:, 0] * (x[:, 0]**2 + x[:, 1]**2)
            u[:, 1] = out[:, 1] * (x[:, 0] + 0.5) * (x[:, 0] - 0.5)
            u[:, 2] = out[:, 2] * (x[:, 0] + 0.5) * (x[:, 0] - 0.5)
            u[:, 3] = out[:, 3]
            u[:, 4] = out[:, 4]
            return u
        elif self.bc == 'fc_circular':
            trial_func = (1. - x[:, 0] * x[:, 0] - x[:, 1] * x[:, 1]) / 2.
        else:
            raise ValueError('Missing Dirichlet boundary conditions.')

        # Broadcast over all five fields
        u = out * trial_func[:, None]
        return u

