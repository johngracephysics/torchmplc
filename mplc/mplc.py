import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import fft2, fftshift, ifft2, ifftshift
from torch.nn.functional import pad
from torch.utils.data import Dataset, DataLoader

from IPython.display import clear_output, display
import matplotlib.pyplot as plt
from matplotlib import colors

from pathlib import Path
import shutil
import numpy as np
import time

from .propagation import Propagator
from .modulation import PhaseMask
from . import functional as mplcF

class MPLC(nn.Module):

    def __init__(self, 
        shape, 
        spacing, 
        mask_spacing, 
        num_masks, 
        propagation_distance, 
        beam_waist, 
        wavelength, 
        device, 
        smooth_weight = 1e-1,
        init_scale = 1e-1): 

        super().__init__()

        self.shape = torch.tensor(shape, device = device)
        self.spacing = torch.tensor(spacing, device = device)
        self.mask_spacing = torch.tensor(mask_spacing, device = device)
        self.mask_shape = (self.shape*self.spacing//self.mask_spacing).to(torch.int)
        self.num_masks = num_masks
        self.propagation_distance = torch.tensor(propagation_distance, device = device)
        self.beam_waist = torch.tensor(beam_waist, device = device)
        self.wavelength = torch.tensor(wavelength, device = device)
        self.device = device
        self.register_buffer('smooth_weight', torch.tensor(smooth_weight, device = device))
        self.init_scale = init_scale

        self.layers = nn.ModuleList()
        self.add_mplc_stack()
        self.sequence = nn.Sequential(*self.layers)

        self.loss_history = []
        self.time_history = []
        self.epochs = 0
        
    def add_mplc_stack(self):

        self.phase_masks = nn.ModuleList()
        self.propagators = nn.ModuleList()

        for i in range(self.num_masks):

            phase_mask = PhaseMask(self.mask_shape, self.device, self.init_scale)
            self.layers.append(phase_mask)
            self.phase_masks.append(phase_mask)

            propagator = Propagator(self.shape, self.spacing, self.propagation_distance, self.wavelength, self.device)
            self.layers.append(propagator)
            self.propagators.append(propagator)

    def forward(self, field):
        
        return self.sequence(field)
    
    def smoothness_penalty(self):

        laplacians = torch.stack([
            mplcF.phase_laplacian(mask.theta)
            for mask in self.phase_masks
        ])
        return self.smooth_weight * torch.mean(torch.abs(laplacians))

    def train_mplc(self, num_epochs, dataloader, optimizer, device, loss_func):
    
        def train_epoch():

            self.train()
            total_loss = 0
            count = 0
            for x_batch, y_batch in dataloader:

                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()
                y_pred = self.forward(x_batch)
                
                loss =  loss_func(y_pred, y_batch) + self.smoothness_penalty()
                loss.backward()
                optimizer.step()

                batch_size = x_batch.size(0)
                total_loss += loss.item() * batch_size
            
                count += batch_size

            epoch_loss = total_loss / count

            return epoch_loss

        @torch.no_grad()
        def wrap_phases():
        
            for mask in self.phase_masks:
                mask.theta.copy_(((mask.theta + torch.pi) % (2 * torch.pi)) - torch.pi)


        for i in range(num_epochs):
            start_time = time.time()
            loss = train_epoch()
            wrap_phases()
            self.time_history.append(time.time() - start_time)
            self.loss_history.append(loss)
            self.epochs += 1
            clear_output(wait=True)
            print(f"Epoch {self.epochs:02d} | Loss: {loss:.6f}")
            #fig = self.visualize_model(show = False)
            #display(fig)
            #plt.close(fig)

    @torch.no_grad()
    def test(self, dataloader, loss_func):
        
        self.eval()
        total_loss = 0
        count = 0
        for x_batch, y_batch in dataloader:

            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            y_pred = self.forward(x_batch)
            loss = loss_func(y_pred, y_batch)

            batch_size = x_batch.size(0)
            total_loss += loss.item() * batch_size
            count += batch_size

        test_loss = total_loss / count

        return test_loss

    def visualize_model(self, show = False):
    
        fig_width = 6 + self.num_masks * 4  # Base width + width per mask
        fig, axes = plt.subplots(1, self.num_masks + 1, figsize=(fig_width, 4))

        # plot loss
        ax = axes[0]
        ax.plot(np.arange(self.epochs) + 1, self.loss_history, '-o', label='Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_ylim(0,1)
        ax.grid(True)
        ax.legend()

        # plot phase masks
        for i, mask in enumerate(self.phase_masks):
            ax = axes[i + 1]
            theta = np.array([mask.get_theta(self.shape).detach().cpu() for i in range(self.shape)])
            im = ax.imshow(theta, vmin=-np.pi, vmax=np.pi, cmap='twilight', origin='lower')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Phase [rad]')
            ax.set_title(f'Phase Mask {i + 1}')

        fig.tight_layout()

        if show:
            plt.show()

        return fig 

    @torch.no_grad()
    def visualize_field(self, input_fields, show=True, far_field=False):
        """
        Visualize 1D complex fields through the MPLC.

        Rows:
            One row per input field.

        Columns:
            Input field, then one column after each mask + propagation step.

        Each subplot shows:
            - field magnitude as a black line
            - field phase as a ribbon underneath

        If far_field=True:
            - performs a regular length-S FFT
            - crops afterwards around the expected Gaussian diffraction modes
            - assumes basis modes of the form

                exp(-x^2 / w0^2) * exp(2π i m x / w0)

            whose Fourier peaks are centred at f = m / w0.
        """

        # ------------------------------------------------------------
        # Extract mask / propagation layers
        # ------------------------------------------------------------
        phasemask_layers = [
            layer for layer in self.sequence
            if isinstance(layer, PhaseMask)
        ]

        propagation_layers = [
            layer for layer in self.sequence
            if isinstance(layer, Propagator)
        ]

        n_fields = len(input_fields)

        n_steps = min(
            len(phasemask_layers),
            len(propagation_layers),
        )

        n_cols = n_steps + 1

        # ------------------------------------------------------------
        # Figure layout
        #
        # Last narrow column contains one colourbar per row.
        # ------------------------------------------------------------
        fig = plt.figure(
            figsize=(
                max(4.5 * n_cols + 0.8, 7),
                max(3.2 * n_fields, 3.5),
            )
        )

        gs = fig.add_gridspec(
            n_fields,
            n_cols + 1,
            width_ratios=[1] * n_cols + [0.045],
            wspace=0.30,
            hspace=0.40,
        )

        axes = np.empty(
            (n_fields, n_cols),
            dtype=object,
        )

        caxes = []

        for row in range(n_fields):

            for col in range(n_cols):
                axes[row, col] = fig.add_subplot(
                    gs[row, col]
                )

            caxes.append(
                fig.add_subplot(gs[row, -1])
            )

        # ------------------------------------------------------------
        # Phase colormap
        # ------------------------------------------------------------
        cmap = plt.get_cmap("hsv")

        phase_norm = colors.Normalize(
            vmin=-np.pi,
            vmax=np.pi,
        )

        phase_map = plt.cm.ScalarMappable(
            norm=phase_norm,
            cmap=cmap,
        )

        phase_map.set_array([])

        # ------------------------------------------------------------
        # Maximum expected mode index
        # ------------------------------------------------------------
        if hasattr(self, "m_max"):
            m_max = abs(self.m_max)

        elif hasattr(self, "modes"):
            m_max = max(
                abs(float(m))
                for m in self.modes
            )

        else:
            # Fallback: assume the input fields span symmetric modes
            # about zero.
            m_max = (n_fields - 1) / 2

        # ------------------------------------------------------------
        # Convert field into whatever should actually be displayed
        # ------------------------------------------------------------
        def get_display_field(field):

            S = field.shape[-1]

            # ========================================================
            # Near field
            # ========================================================
            if not far_field:

                x = np.arange(S)

                return field, x

            # ========================================================
            # Far field
            # ========================================================

            # Regular length-S centered FFT.
            fft_field = torch.fft.fftshift(
                torch.fft.fft(
                    torch.fft.ifftshift(
                        field,
                        dim=-1,
                    ),
                    dim=-1,
                    norm="ortho",
                ),
                dim=-1,
            )

            # Spatial-frequency coordinate [cycles / unit length].
            fx = torch.fft.fftshift(
                torch.fft.fftfreq(
                    S,
                    d=self.spacing,
                    device=field.device,
                    dtype=field.real.dtype,
                )
            )

            w0 = self.beam_waist

            # --------------------------------------------------------
            # For
            #
            #   exp(-x²/w0²) exp(2π i m x/w0)
            #
            # the Fourier-space Gaussian is centred at
            #
            #   f_m = m/w0
            #
            # and has characteristic amplitude width
            #
            #   Δf = 1/(π w0).
            #
            # Crop two widths beyond the outermost expected mode.
            # --------------------------------------------------------
            f_outer = m_max / w0

            f_margin = 2 / (torch.pi * w0) * 3

            f_max = f_outer + f_margin

            keep = torch.abs(fx) <= f_max

            return (
                fft_field[keep],
                fx[keep].detach().cpu().numpy(),
            )

        # ------------------------------------------------------------
        # Plot a single complex 1D field
        # ------------------------------------------------------------
        def plot_field(ax, field):

            display_field, x = get_display_field(
                field
            )

            field_np = (
                display_field
                .detach()
                .cpu()
                .numpy()
            )

            if field_np.ndim != 1:
                raise ValueError(
                    "Expected a 1D field, "
                    f"got shape {field_np.shape}"
                )

            mag = np.abs(field_np)
            phase = np.angle(field_np)

            # --------------------------------------------------------
            # Magnitude
            # --------------------------------------------------------
            ax.plot(
                x,
                mag,
                color="black",
                linewidth=2,
            )

            ax.grid(alpha=0.18)

            ax.set_xlabel(
                r"Spatial frequency $f_x$"
                if far_field
                else "Position"
            )

            # --------------------------------------------------------
            # Vertical scale
            # --------------------------------------------------------
            if mag.size > 0:
                mag_max = float(mag.max())
            else:
                mag_max = 0.0

            scale = (
                mag_max
                if mag_max > 0
                else 1.0
            )

            # --------------------------------------------------------
            # Phase ribbon geometry
            # --------------------------------------------------------
            ribbon_h = 0.08 * scale
            ribbon_gap = 0.055 * scale

            ribbon_y1 = -ribbon_gap
            ribbon_y0 = ribbon_y1 - ribbon_h

            # --------------------------------------------------------
            # Phase -> colour
            # --------------------------------------------------------
            rgba = cmap(
                phase_norm(phase)
            )

            # Fade the phase where amplitude is negligible.
            cutoff = 0.03 * mag_max

            alpha = np.clip(
                (
                    mag - cutoff
                )
                /
                (
                    mag_max
                    - cutoff
                    + 1e-12
                ),
                0,
                1,
            )

            alpha = alpha ** 0.7

            rgba[:, 3] = alpha

            ribbon_img = np.repeat(
                rgba[np.newaxis, :, :],
                2,
                axis=0,
            )

            # Neutral ribbon background.
            ax.axhspan(
                ribbon_y0,
                ribbon_y1,
                color="0.92",
                zorder=0,
            )

            # --------------------------------------------------------
            # Draw phase ribbon
            # --------------------------------------------------------
            if len(x) > 1:

                ax.imshow(
                    ribbon_img,
                    extent=[
                        x[0],
                        x[-1],
                        ribbon_y0,
                        ribbon_y1,
                    ],
                    aspect="auto",
                    interpolation="nearest",
                    zorder=1,
                )

                ax.set_xlim(
                    x[0],
                    x[-1],
                )

            ax.set_ylim(
                ribbon_y0 - 0.025 * scale,
                mag_max + 0.06 * scale,
            )

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # --------------------------------------------------------
            # Total power in the actual field
            # --------------------------------------------------------
            total_power = (
                torch.abs(field)
                .square()
                .sum()
                .real
                .item()
            )

            ax.text(
                0.97,
                0.94,
                f"$P={total_power:.3f}$",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
            )

        # ============================================================
        # One row for each input field
        # ============================================================
        for row, field in enumerate(input_fields):

            current_field = (
                field.detach().clone()
            )

            # --------------------------------------------------------
            # Input field
            # --------------------------------------------------------
            ax = axes[row, 0]

            plot_field(
                ax,
                current_field,
            )

            ax.set_ylabel(
                f"Input {row + 1}\nAmplitude"
            )

            if row == 0:
                ax.set_title(
                    "Input"
                    + (" — FFT" if far_field else "")
                )

            # --------------------------------------------------------
            # Propagate through each mask + propagator pair
            # --------------------------------------------------------
            for step in range(n_steps):

                current_field = phasemask_layers[
                    step
                ](
                    current_field
                )

                current_field = propagation_layers[
                    step
                ](
                    current_field
                )

                ax = axes[
                    row,
                    step + 1,
                ]

                plot_field(
                    ax,
                    current_field,
                )

                if row == 0:

                    ax.set_title(
                        f"After Mask {step + 1}"
                        + (
                            " — FFT"
                            if far_field
                            else ""
                        )
                    )

                if step > 0:
                    ax.set_ylabel("")

            # --------------------------------------------------------
            # Full-height phase colourbar for this row
            # --------------------------------------------------------
            cbar = fig.colorbar(
                phase_map,
                cax=caxes[row],
            )

            cbar.set_ticks(
                [
                    -np.pi,
                    -np.pi / 2,
                    0,
                    np.pi / 2,
                    np.pi,
                ],
                labels=[
                    r"$-\pi$",
                    r"$-\pi/2$",
                    "0",
                    r"$\pi/2$",
                    r"$\pi$",
                ],
            )

            cbar.set_label(
                "Phase",
                rotation=270,
                labelpad=15,
            )

        if show:
            plt.show()

        return fig

    def vizualize_phase_gradient(self):

        phases = [mask.get_theta(self.shape).detach() for mask in self.phase_masks]
        phase_grads = [mplcF.phase_gradient(phase, self.spacing) for phase in phases]

        for i, grad in enumerate(phase_grads):
            plt.plot(grad, label = f'Phase Mask {i+1}')

        plt.xlabel('x-postition')
        plt.ylabel('Phase Gradient')
        threshold = np.pi/50e-6

        plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
        plt.axhline(y=-threshold, color='r', linestyle='--')
        plt.ylim(-threshold-10000, threshold+10000)

        plt.legend(loc = 'center right')
        plt.show()

    def export_masks(self, filename):

        phases = [mask.get_theta(self.shape).detach() for mask in self.phase_masks]
        
        folder = Path(filename)

        if folder.exists():
            confirm = input(
                f"Folder '{folder}' already exists. Overwrite it? [y/N]: "
            )

            if confirm.lower() not in ("y", "yes"):
                print("Save cancelled.")
                return

            shutil.rmtree(folder)

        folder.mkdir(parents=True)

        for i, phase in enumerate(phases):
            if phase.ndim != 1:
                raise ValueError(f"phases[{i}] must be a 1D tensor.")

            np.savetxt(
                folder / f"phase_{i + 1}.csv",
                phase.detach().cpu().numpy(),
                delimiter=",",
            )

        print(f"Saved {len(phases)} phases to '{folder}'.")
    
        
    

        