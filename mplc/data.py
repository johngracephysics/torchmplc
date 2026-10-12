import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import fft2, fftshift, ifft2, ifftshift
from torch.nn.functional import pad
from torch.utils.data import Dataset, DataLoader
import random

from . import functional as mplcF

def get_basis_fields(D, shape, spacing, beam_waist, device, separation = 1):

    def kmode(m, shape, spacing, beam_waist, device, separation):

        Sx = shape
        m = torch.tensor(m)
        x = torch.arange(-Sx//2,Sx//2, device = device)*spacing

        kmode = torch.exp(-(x**2)/beam_waist**2)*torch.exp((2j*torch.pi/beam_waist)*m*x)+1e-5
        norm = torch.sqrt(torch.sum(torch.conj(kmode)*kmode))
        kmode = kmode / norm

        return kmode.to(dtype = torch.complex64)

    if D % 2 == 0:
        mlist = (np.delete(np.arange(-(D//2), (D//2)+1), D//2) + np.array([-0.5 if i >= D//2 else 0.5 for i in range(D)]))*separation
    if D % 2 == 1:
        mlist = np.arange(-(D//2), (D//2)+1)*separation

    basis_fields = [mplcF.normalize_field(kmode(m, shape, spacing, beam_waist, device, separation)) for m in mlist]

    return torch.stack(basis_fields, dim=0)

def encode(x, basis_fields):
    return torch.einsum('bd, dx -> bx', x, basis_fields)

def decode(field, basis_fields):

    return torch.einsum('bx, dx -> bd', field, torch.conj(basis_fields))

def get_sic_kets(D):
            from qbism import sic_povm
            
            povm = sic_povm(D)
            np_kets = np.zeros((D**2, D), dtype=np.complex64)

            for i, effect in enumerate(povm):
                mat = effect.full()
                evals, evecs = np.linalg.eigh(mat)
                idx = int(np.argmax(evals))
                v = evecs[:, idx]
                norm = np.linalg.norm(v)
                v = v / norm
                np_kets[i, :] = v.astype(np.complex128)
            kets = torch.from_numpy(np_kets)
            return kets


class FieldDataset(Dataset):

    def __init__(self, target, basis_fields):

        super().__init__()
        self.target = target
        self.dimension = target.shape[0]

        input_kets = self.get_inputs()
        output_kets = torch.einsum('ij, bj -> bi', target, input_kets)
        self.inputs = mplcF.normalize_field(encode(input_kets, basis_fields))
        self.outputs = mplcF.normalize_field(encode(output_kets, basis_fields))
    
    def get_inputs(self):

        return torch.eye(self.dimension)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]

    def __len__(self):
        return len(self.inputs)

    def get_dataloader(self, batch_size, shuffle = True):
        return DataLoader(self, batch_size = batch_size, shuffle = shuffle)

class SICFieldDataset(FieldDataset):

    def get_inputs(self):

        return get_sic_kets(self.dimension)

class RandomFieldDataset(FieldDataset):

    def __init__(self, target, basis_fields, num_inputs, seed = None):

        self.num_inputs = num_inputs
        self.seed = seed
        super().__init__(target, basis_fields)

    def get_inputs(self):

        if not self.seed is None:
            generator = torch.Generator()
            generator.manual_seed(self.seed)
            kets = torch.randn((self.num_inputs, self.dimension), generator = generator) + 1j * torch.randn((self.num_inputs, self.dimension), generator = generator)
        else:
            kets = torch.randn(self.num_inputs, self.dimension) + 1j * torch.randn(self.num_inputs, self.dimension)

        
        return kets / torch.linalg.vector_norm(kets, dim=1, keepdim=True)