# fourier_pkg

A teaching package for understanding Fourier Transform with visualization.

## Installation

### Development Installation (Recommended for this course)

```bash
cd fourier_pkg
pip install -e .
```

The `-e` flag installs in "editable" mode, meaning changes to the source code
take effect immediately without reinstalling.

### Standard Installation

```bash
pip install .
```

## Package Structure

```
fourier_pkg/
├── fourier_pkg/           # Main package directory
│   ├── __init__.py        # Package initialization & public API
│   ├── core.py            # Core FFT functions
│   ├── utils.py           # Utility functions & SignalGenerator
│   └── visualize.py       # Visualization functions
├── pyproject.toml         # Modern package configuration
├── setup.py               # Setup script (backwards compatible)
└── README.md              # This file
```

## Key Concepts: Python Package Management

### 1. What is a Package?

A Python package is a directory containing Python modules along with an
`__init__.py` file. This file is executed when the package is imported
and defines what gets exposed as the public API.

```
mypackage/
├── __init__.py     # Makes this a package
├── module1.py      # A module
└── module2.py      # Another module
```

### 2. Why `__init__.py`?

The `__init__.py` file serves several purposes:

- **Marks the directory as a package**: Without it, Python 2 won't recognize the directory as a package.
- **Controls imports**: You can expose only specific functions/classes as the public API.
- **Performs initialization**: Code in `__init__.py` runs when the package is imported.

### 3. The `__all__` Variable

The `__all__` list in `__init__.py` defines the public API:

```python
__all__ = ['function1', 'class2', 'CONSTANT3']
```

When someone does `from mypackage import *`, only these names are imported.

### 4. Relative vs Absolute Imports

**Absolute import:**
```python
from fourier_pkg.core import compute_fft
```

**Relative import (inside the package):**
```python
from .core import compute_fft  # . means "current package"
```

### 5. Installing in Development Mode

```bash
pip install -e .
```

- `-e` = editable/development mode
- Changes to source code take effect immediately
- Package appears in `pip list`
- Can be uninstalled with `pip uninstall`

## Usage Example

```python
import numpy as np
from fourier_pkg import (
    generate_composite_signal,
    compute_fft,
    find_dominant_frequencies,
    plot_combined_analysis,
)

# Generate a signal
sample_rate = 1000
t = np.linspace(0, 1, sample_rate)
signal = generate_composite_signal(t, [5, 15, 50], [1.0, 0.5, 0.3], [0, 0, 0])

# Analyze it
freqs, mags = compute_fft(signal, sample_rate)
peaks = find_dominant_frequencies(freqs, mags, top_n=3)

# Visualize
import matplotlib.pyplot as plt
fig = plot_combined_analysis(t, signal, sample_rate, true_frequencies=[5, 15, 50])
plt.show()
```

## For Package Developers

### Building a Distribution

```bash
# Install build tool
pip install build

# Build source and wheel
python -m build
```

This creates:
- `dist/fourier_pkg-0.1.0.tar.gz` (source)
- `dist/fourier_pkg-0.1.0-py3-none-any.whl` (wheel)

### Uploading to PyPI

```bash
pip install twine
twine upload dist/*
```

## License

MIT License
