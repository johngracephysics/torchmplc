import torch
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors

def field_norm(field):

    return torch.sqrt(torch.abs(field_inner(field, field)))

def field_inner(field1, field2):

    return torch.einsum('...x, ...x -> ...', field1, torch.conj(field2))

def field_overlap_loss(field1, field2):

        return 1 - torch.mean(torch.abs(torch.einsum('bx, bx ->b', field1, torch.conj(field2)))**2)


def normalize_field(field):
    norm = torch.sqrt(torch.sum(torch.abs(field) ** 2, dim= -1, keepdim=True))
    return field / norm

def phase_gradient(theta, dx = 1):
    
    dtheta = theta[..., 1:] - theta[..., :-1]
    dtheta_wrapped = torch.atan2(torch.sin(dtheta), torch.cos(dtheta))
    return dtheta_wrapped/dx

def gradient(y, dx = 1):
    dy = y[..., 1:] - y[..., :-1]
    return dy/dx


def phase_laplacian(theta, dx=1):
    # Wrapped phase differences
    dtheta = theta[..., 1:] - theta[..., :-1]
    dtheta_wrapped = torch.atan2(torch.sin(dtheta), torch.cos(dtheta))

    # Second derivative: change in the wrapped phase gradient
    return (dtheta_wrapped[..., 1:] - dtheta_wrapped[..., :-1]) / dx**2

def hadamard(D):

    D = torch.tensor(D)
    hadamard_matrix = torch.zeros((D,D), dtype = torch.complex64)
    for i in range(D):
        for j in range(D):
            hadamard_matrix[i][j] = 1/torch.sqrt(D)*torch.exp(1j*2*torch.pi*i*j/D)

    return hadamard_matrix

def interpolate_phase(phase, field_shape):
    # Differences between adjacent samples
    delta = phase[1:] - phase[:-1]

    # Choose each difference modulo 2π with minimum magnitude
    delta = torch.remainder(delta + torch.pi, 2 * torch.pi) - torch.pi

    # Reconstruct a continuous, unwrapped phase
    unwrapped = torch.cat([
        phase[:1],
        phase[:1] + torch.cumsum(delta, dim=0),
    ])

    # Linearly interpolate the unwrapped phase
    interpolated_phase_unwrapped = F.interpolate(
        unwrapped[None, None, :],
        size=field_shape,
        mode="linear",
        align_corners=True,
    )[0, 0]

    interpolated_phase_wrapped = torch.remainder(
        interpolated_phase_unwrapped + torch.pi,
        2 * torch.pi,
    ) - torch.pi

    # Wrap back into [-π, π)
    return interpolated_phase_wrapped

@torch.no_grad()
def pad_to_square(tensor):
    t = tensor.detach().clone()

    rows, cols = t.shape

    pad_top = pad_bottom = 0
    pad_left = pad_right = 0

    if rows < cols:
        diff = cols - rows
        pad_top = diff // 2
        pad_bottom = diff - pad_top

    elif cols < rows:
        diff = rows - cols
        pad_left = diff // 2
        pad_right = diff - pad_left

    padded_tensor = F.pad(
        t,
        (pad_left, pad_right, pad_top, pad_bottom),
        mode="constant",
        value=0,
    )

    crop = (
        slice(pad_top, pad_top + rows),
        slice(pad_left, pad_left + cols),
    )

    return padded_tensor, crop

def plot_field(field):
    
    field = np.asarray(field)
    if field.ndim != 1:
        raise ValueError("field must be a 1D complex array")

    x = np.arange(field.size)
    mag = np.abs(field)
    phase = np.angle(field)

    fig, ax = plt.subplots(figsize=(10, 4.5))

    # --- Main magnitude plot ---
    ax.plot(x, mag, linewidth=2, color = 'black')
    ax.set_xlabel("Position")
    ax.set_ylabel("Field Amplitude")
    ax.grid(alpha=0.2)

    # --- Ribbon geometry ---
    mag_max = mag.max() if mag.size else 1.0
    if mag_max == 0:
        mag_max = 1.0

    ribbon_h = 0.08 * mag_max
    ribbon_gap = 0.06 * mag_max
    ribbon_y1 = -ribbon_gap
    ribbon_y0 = ribbon_y1 - ribbon_h

    # --- Phase colormap ---
    cmap = cm.get_cmap("hsv")
    norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
    rgba = cmap(norm(phase))   # shape (N, 4)

    # --- Fade out ribbon where magnitude is very small ---
    # Phase is only visually emphasized where |psi| is significant.
    cutoff = 0.03 * mag.max() if mag.max() > 0 else 0.0
    alpha = np.clip((mag - cutoff) / (mag.max() - cutoff + 1e-12), 0, 1)
    alpha = alpha**0.7  # slightly smoother ramp
    rgba[:, 3] = alpha

    # Build a 2-pixel-high RGBA image for the ribbon
    ribbon_img = np.repeat(rgba[np.newaxis, :, :], 2, axis=0)

    # Light gray background under the ribbon so "empty" regions stay visible
    ax.axhspan(ribbon_y0, ribbon_y1, color="0.92", zorder=0)

    ax.imshow(
        ribbon_img,
        extent=[x.min(), x.max(), ribbon_y0, ribbon_y1],
        aspect="auto",
        interpolation="nearest",
        zorder=1,
    )

    # --- Limits and cosmetics ---
    ax.set_ylim(ribbon_y0 - 0.03 * mag_max, mag.max() + 0.05 * mag_max)
    #ax.text(x.min(), 0.5 * (ribbon_y0 + ribbon_y1), " phase",
    #        va="center", ha="left")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- Colorbar for phase ---
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Phase")
    cbar.set_ticks(
        [-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
        labels=[r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"]
    )

    fig.tight_layout()
    return fig, ax

def fft_convolve_1d(x, h):
    S = h.shape[-1]
    n = 2 * S - 1
    y = torch.fft.ifft(torch.fft.fft(h, n=n, dim=-1) * torch.fft.fft(x, n=n))
    return y[..., :S]