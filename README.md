# torchmplc

PyTorch-based tools for designing phase masks for multi-plane light conversion (MPLC).

## Installation

With Python 3.9 or newer, install from the GitHub repository:

```bash
python -m pip install git+https://github.com/YOUR_USERNAME/torchmplc.git
```

For SIC-POVM datasets, also install the optional `sic` extra:

```bash
python -m pip install "torchmplc[sic] @ git+https://github.com/YOUR_USERNAME/torchmplc.git"
```

For local development:

```bash
python -m pip install -e ".[sic]"
```

The SIC-POVM functionality relies on the `qbism` package (which in turn uses QuTiP). Verify its compatibility with your Python environment.

## Usage

```python
from mplc import MPLC
from mplc.data import get_basis_fields, RandomFieldDataset, SICFieldDataset
import mplc.functional as mplcF
```

See [`example.ipynb`](example.ipynb) for a full training and visualization example.

The Python **distribution** is named `torchmplc`, while the **import** is named `mplc`.

## Testing

```bash
python -m pip install pytest
python -m pytest
```

The smoke test covers basic imports and the shape of an MPLC forward pass. It does not validate the full scientific simulation or the SIC-POVM dependency.
