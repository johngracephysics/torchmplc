import torch

from mplc import MPLC
from mplc.data import RandomFieldDataset


def test_mplc_forward_shape():
    model = MPLC(
        shape=16,
        spacing=1e-5,
        mask_spacing=2e-5,
        num_masks=2,
        propagation_distance=0.01,
        beam_waist=1e-4,
        wavelength=810e-9,
        device="cpu",
    )
    input_field = torch.ones((1, 16), dtype=torch.complex64)
    output = model(input_field)
    assert output.shape == input_field.shape
    assert torch.is_complex(output)
