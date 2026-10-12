import torch
import torch.nn as nn

from . import functional as mplcF

class PhaseMask(nn.Module):

    def __init__(self, mask_shape, device, init_scale = 2*torch.pi*1e-2):

        super().__init__()

        self.mask_shape = mask_shape
        init_theta = torch.randn(mask_shape, device = device)*init_scale
        self.theta = nn.Parameter(init_theta)

        
    def get_theta(self, field_shape, method = 'linear'):
        
        interpolated_theta = mplcF.interpolate_phase(self.theta, field_shape)
        return interpolated_theta


    def forward(self, field):

        phase_mask = torch.exp(1j*self.get_theta(field.shape[-1]))
        return field*phase_mask