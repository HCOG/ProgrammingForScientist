"""
fourier_pkg - A teaching package for understanding Fourier Transform
"""

__version__ = "0.1.0"
__author__ = "ProgrammingForScientist"

from .core import (
    generate_composite_signal,
    compute_fft,
    find_dominant_frequencies,
    nyquist_frequency,
)

from .visualize import (
    plot_time_domain,
    plot_frequency_spectrum,
    plot_signal_decomposition,
    plot_combined_analysis,
)

from .utils import (
    add_noise,
    SignalGenerator,
    estimate_snr,
)

__all__ = [
    # Core
    "generate_composite_signal",
    "compute_fft",
    "find_dominant_frequencies",
    "nyquist_frequency",
    # Visualize
    "plot_time_domain",
    "plot_frequency_spectrum",
    "plot_signal_decomposition",
    "plot_combined_analysis",
    # Utils
    "add_noise",
    "SignalGenerator",
    "estimate_snr",
]
