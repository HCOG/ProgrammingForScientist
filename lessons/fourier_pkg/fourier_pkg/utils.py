"""
Utility functions for fourier_pkg.
"""

import numpy as np


def add_noise(signal, noise_level=0.1, seed=None):
    """
    Add Gaussian white noise to a signal.
    
    Parameters
    ----------
    signal : array_like
        Input signal.
    noise_level : float, default 0.1
        Standard deviation of the noise.
    seed : int, optional
        Random seed for reproducibility.
    
    Returns
    -------
    ndarray
        Noisy signal.
    
    Examples
    --------
    >>> clean = np.sin(np.linspace(0, 2*np.pi, 100))
    >>> noisy = add_noise(clean, noise_level=0.05, seed=42)
    """
    if seed is not None:
        np.random.seed(seed)
    
    noise = noise_level * np.random.randn(len(signal))
    return np.asarray(signal) + noise


def estimate_snr(signal, noisy_signal):
    """
    Estimate the Signal-to-Noise Ratio (SNR).
    
    Parameters
    ----------
    signal : array_like
        Clean reference signal.
    noisy_signal : array_like
        Signal with noise.
    
    Returns
    -------
    float
        SNR in decibels (dB).
    
    Notes
    -----
    SNR = 10 * log10(P_signal / P_noise)
    where P is the power (variance for zero-mean signals).
    
    Examples
    --------
    >>> snr = estimate_snr(clean_signal, noisy_signal)
    >>> print(f"SNR: {snr:.2f} dB")
    """
    signal = np.asarray(signal)
    noisy_signal = np.asarray(noisy_signal)
    
    # Estimate noise as difference between signals
    noise = noisy_signal - signal
    
    # Power of signal and noise
    psignal = np.var(signal)
    pnoise = np.var(noise)
    
    if pnoise == 0:
        return float('inf')
    
    snr_linear = psignal / pnoise
    snr_db = 10 * np.log10(snr_linear)
    
    return snr_db


class SignalGenerator:
    """
    A class for generating various types of test signals.
    
    This class provides a convenient way to create common test signals
    used in signal processing demonstrations.
    
    Attributes
    ----------
    sample_rate : float
        Sampling rate in Hz.
    duration : float
        Duration of generated signals in seconds.
    
    Examples
    --------
    >>> gen = SignalGenerator(sample_rate=1000, duration=1.0)
    >>> signal = gen.sine(frequency=5, amplitude=1.0)
    >>> signal = gen.square(frequency=2, amplitude=0.5)
    >>> signal = gen.sawtooth(frequency=1, amplitude=1.0)
    """
    
    def __init__(self, sample_rate=1000, duration=1.0):
        """
        Initialize the signal generator.
        
        Parameters
        ----------
        sample_rate : float, default 1000
            Sampling rate in Hz.
        duration : float, default 1.0
            Duration in seconds.
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self._t = np.arange(0, duration, 1 / sample_rate)
    
    @property
    def time(self):
        """Get the time array."""
        return self._t.copy()
    
    def sine(self, frequency=1.0, amplitude=1.0, phase=0.0):
        """
        Generate a sine wave.
        
        Parameters
        ----------
        frequency : float, default 1.0
            Frequency in Hz.
        amplitude : float, default 1.0
            Amplitude.
        phase : float, default 0.0
            Initial phase in radians.
        
        Returns
        -------
        ndarray
            Sine wave signal.
        """
        return amplitude * np.sin(2 * np.pi * frequency * self._t + phase)
    
    def cosine(self, frequency=1.0, amplitude=1.0, phase=0.0):
        """Generate a cosine wave."""
        return amplitude * np.cos(2 * np.pi * frequency * self._t + phase)
    
    def square(self, frequency=1.0, amplitude=1.0, phase=0.0, duty=0.5):
        """
        Generate a square wave.
        
        Parameters
        ----------
        duty : float, default 0.5
            Duty cycle (fraction of period that is positive).
        """
        return amplitude * (2 * (self.sine(frequency, 1, phase) > duty - 0.5) - 1)
    
    def sawtooth(self, frequency=1.0, amplitude=1.0, phase=0.0):
        """Generate a sawtooth (ramp) wave."""
        period = 1 / frequency
        phase_normalized = (phase / (2 * np.pi)) % 1
        shifted_t = self._t + phase_normalized * period
        return amplitude * (2 * ((shifted_t * frequency) % 1) - 1)
    
    def impulse(self, position=0.5):
        """
        Generate an impulse (delta) signal.
        
        Parameters
        ----------
        position : float, default 0.5
            Position of impulse as fraction of duration.
        """
        impulse_idx = int(len(self._t) * position)
        signal = np.zeros_like(self._t)
        signal[impulse_idx] = 1.0
        return signal
    
    def step(self, step_time=None):
        """
        Generate a step signal.
        
        Parameters
        ----------
        step_time : float, optional
            Time of step transition. Defaults to 50% of duration.
        """
        if step_time is None:
            step_time = self.duration / 2
        return np.where(self._t >= step_time, 1.0, 0.0)
    
    def chirp(self, f_start=1.0, f_end=10.0, method='linear'):
        """
        Generate a frequency sweep (chirp) signal.
        
        Parameters
        ----------
        f_start : float
            Starting frequency in Hz.
        f_end : float
            Ending frequency in Hz.
        method : {'linear', 'quadratic'}, default 'linear'
            Frequency sweep method.
        """
        if method == 'linear':
            return amplitude * np.sin(
                2 * np.pi * (f_start + (f_end - f_start) * self._t / (2 * self.duration) * self._t)
            )
        elif method == 'quadratic':
            t_normalized = self._t / self.duration
            phase = 2 * np.pi * (f_start * self._t + 
                                (f_end - f_start) * t_normalized**2 * self._t / 2)
            return np.sin(phase)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def white_noise(self, amplitude=1.0, seed=None):
        """Generate white noise."""
        if seed is not None:
            np.random.seed(seed)
        return amplitude * np.random.randn(len(self._t))
    
    def brownian_noise(self, amplitude=1.0):
        """Generate Brownian (red) noise by integrating white noise."""
        white = self.white_noise(amplitude)
        return np.cumsum(white) / self.sample_rate


def periodogram(signal, sample_rate):
    """
    Compute the periodogram (power spectral density estimate).
    
    Parameters
    ----------
    signal : array_like
        Input signal.
    sample_rate : float
        Sampling rate in Hz.
    
    Returns
    -------
    tuple
        (frequencies, power_spectral_density).
    """
    n = len(signal)
    fft_result = np.fft.fft(signal)
    psd = (np.abs(fft_result) ** 2) / n
    freqs = np.fft.fftfreq(n, 1 / sample_rate)
    
    # Return positive frequencies only
    positive_mask = freqs >= 0
    return freqs[positive_mask], psd[positive_mask] * 2
