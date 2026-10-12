import torch
import torch.nn as nn

class Propagator(nn.Module):

    def __init__(self, shape, spacing, propagation_distance, wavelength, device):
        
        super().__init__()

        self.shape = shape
        self.spacing = spacing
        self.register_buffer('propagation_distance', propagation_distance)
        self.wavelength = wavelength
        self.device = device
        
    def forward(self, field):

        return self.asm(field)


    def asm(self, field):
        
        S = field.shape[-1]

        dx = self.spacing
        lam = self.wavelength
        z = self.propagation_distance

        # Zero-pad to suppress periodic wrap-around
        pad = S // 2
        field_padded = torch.nn.functional.pad(field, (pad, pad))
        N = field_padded.shape[-1]

        fx = torch.fft.fftfreq(
            N,
            d=dx,
            device=field.device,
            dtype=field.real.dtype,
        )
        kx = 2 * torch.pi * fx

        k = 2 * torch.pi / lam
        kz = torch.sqrt((k**2 - kx**2).to(field.dtype))

        H = torch.exp(1j * kz * z)

        propagated = torch.fft.ifft(
            torch.fft.fft(field_padded, dim=-1) * H,
            dim=-1,
        )

        # Return original observation window
        return propagated[..., pad:pad + S]