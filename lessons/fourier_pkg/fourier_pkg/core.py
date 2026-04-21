"""
Core FFT functions for fourier_pkg.
"""

import numpy as np


def nyquist_frequency(sample_rate, n_samples):
    """
    Calculate the Nyquist frequency (maximum analysable frequency).
    
    Parameters
    ----------
    sample_rate : float
        Sampling rate in Hz.
    n_samples : int
        Number of samples.
    
    Returns
    -------
    float
        Nyquist frequency in Hz.
    
    Notes
    -----
    The Nyquist frequency is half the sampling rate. Frequencies above
    this cannot be correctly reconstructed from discrete samples.
    
    Examples
    --------
    >>> nyquist_frequency(1000, 1024)
    500.0
    """
    return sample_rate / 2


def generate_composite_signal(t, frequencies, amplitudes, phases):
    """
    Generate a composite sinusoidal signal.
    
    Parameters
    ----------
    t : array_like
        Time array in seconds.
    frequencies : list of float
        List of frequencies in Hz.
    amplitudes : list of float
        List of amplitudes for each frequency component.
    phases : list of float
        List of initial phases in radians.
    
    Returns
    -------
    ndarray
        Composite signal as sum of sinusoids.
    
    Examples
    --------
    >>> t = np.linspace(0, 1, 1000)
    >>> signal = generate_composite_signal(t, [5, 10], [1.0, 0.5], [0, np.pi/4])
    
    Notes
    -----
    The signal is computed as:
    s(t) = Σ Aᵢ · sin(2πfᵢt + φᵢ)
    """
    frequencies = np.asarray(frequencies)
    amplitudes = np.asarray(amplitudes)
    phases = np.asarray(phases)

    if not (len(frequencies) == len(amplitudes) == len(phases)):
        raise ValueError(
            "frequencies, amplitudes, and phases must have the same length"
        )

    composite = np.zeros_like(t)
    for freq, amp, phase in zip(frequencies, amplitudes, phases):
        composite += amp * np.sin(2 * np.pi * freq * t + phase)
    return composite


def compute_fft(signal, sample_rate, return_full=False):
    """
    Compute the Fast Fourier Transform and return positive frequencies.
    
    Parameters
    ----------
    signal : array_like
        Input signal in time domain.
    sample_rate : float
        Sampling rate in Hz.
    return_full : bool, default False
        If True, return full spectrum (both positive and negative).
    
    Returns
    -------
    tuple
        (frequencies, magnitudes) where frequencies is in Hz.
    
    Notes
    -----
    The FFT output is scaled by 2/N for single-sided spectrum to account
    for the energy in negative frequencies.
    
    Examples
    --------
    >>> t = np.linspace(0, 1, 1000)
    >>> signal = np.sin(2*np.pi*5*t)
    >>> freqs, mags = compute_fft(signal, 1000)
    >>> dominant_freq = freqs[np.argmax(mags[1:]) + 1]  # skip DC component
    """
    signal = np.asarray(signal)
    n = len(signal)
    
    # Compute FFT
    fft_result = np.fft.fft(signal)
    fft_freqs = np.fft.fftfreq(n, 1 / sample_rate)
    
    if return_full:
        return fft_freqs, np.abs(fft_result) * 2 / n
    
    # Return only positive frequencies (single-sided spectrum)
    positive_mask = fft_freqs >= 0
    magnitudes = np.abs(fft_result[positive_mask]) * 2 / n
    # Set DC component (index 0) to not doubled
    magnitudes[0] = magnitudes[0] / 2
    
    return fft_freqs[positive_mask], magnitudes


def find_dominant_frequencies(freqs, magnitudes, top_n=5, exclude_dc=True):
    """
    Find the dominant frequency components in a spectrum.
    
    Parameters
    ----------
    freqs : array_like
        Frequency array in Hz.
    magnitudes : array_like
        Magnitude spectrum.
    top_n : int, default 5
        Number of top frequencies to return.
    exclude_dc : bool, default True
        Exclude the DC (0 Hz) component from results.
    
    Returns
    -------
    list of tuple
        List of (frequency, magnitude) tuples sorted by magnitude.
    
    Examples
    --------
    >>> freqs, mags = compute_fft(signal, sample_rate)
    >>> dominant = find_dominant_frequencies(freqs, mags, top_n=3)
    >>> print(f"Largest component: {dominant[0][0]:.1f} Hz")
    """
    freqs = np.asarray(freqs)
    magnitudes = np.asarray(magnitudes)
    
    if exclude_dc and len(magnitudes) > 0:
        magnitudes = magnitudes.copy()
        magnitudes[0] = 0  # Zero out DC component
    
    # Get indices of top N magnitudes
    sorted_indices = np.argsort(magnitudes)[::-1]
    top_indices = sorted_indices[:top_n]
    
    return [(freqs[i], magnitudes[i]) for i in top_indices]


def compute_power_spectrum(signal, sample_rate):
    """
    Compute the power spectrum (magnitude squared).
    
    Parameters
    ----------
    signal : array_like
        Input signal.
    sample_rate : float
        Sampling rate in Hz.
    
    Returns
    -------
    tuple
        (frequencies, power) arrays.
    """
    freqs, magnitudes = compute_fft(signal, sample_rate)
    return freqs, magnitudes ** 2
