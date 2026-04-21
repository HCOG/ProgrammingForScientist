"""
Visualization functions for fourier_pkg.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_time_domain(t, signal, title="Time Domain Signal", 
                     xlabel="Time (s)", ylabel="Amplitude",
                     figsize=(12, 4), ax=None):
    """
    Plot a signal in the time domain.
    
    Parameters
    ----------
    t : array_like
        Time array.
    signal : array_like
        Signal values.
    title : str, default "Time Domain Signal"
        Plot title.
    xlabel : str, default "Time (s)"
        X-axis label.
    ylabel : str, default "Amplitude"
        Y-axis label.
    figsize : tuple, default (12, 4)
        Figure size.
    ax : Axes, optional
        Axes object to plot on. If None, creates a new figure.
    
    Returns
    -------
    Axes
        The matplotlib Axes object.
    
    Examples
    --------
    >>> t = np.linspace(0, 1, 1000)
    >>> signal = np.sin(2*np.pi*5*t)
    >>> plot_time_domain(t, signal, title="5 Hz Sine Wave")
    >>> plt.show()
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(t, signal, linewidth=1.2, color='#2E86AB')
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([t.min(), t.max()])
    
    return ax


def plot_frequency_spectrum(freqs, magnitudes, title="Frequency Spectrum",
                            xlabel="Frequency (Hz)", ylabel="Magnitude",
                            xlim=None, ylim=None,
                            figsize=(12, 4), ax=None, 
                            marker_style='o', mark_peaks=True,
                            top_n=3):
    """
    Plot the frequency spectrum.
    
    Parameters
    ----------
    freqs : array_like
        Frequency array in Hz.
    magnitudes : array_like
        Magnitude spectrum.
    title : str, default "Frequency Spectrum"
        Plot title.
    xlabel : str, default "Frequency (Hz)"
        X-axis label.
    ylabel : str, default "Magnitude"
        Y-axis label.
    xlim : tuple, optional
        X-axis limits (min, max).
    ylim : tuple, optional
        Y-axis limits (min, max).
    figsize : tuple, default (12, 4)
        Figure size.
    ax : Axes, optional
        Axes object to plot on.
    marker_style : str, default 'o'
        Marker style for peaks.
    mark_peaks : bool, default True
        Whether to mark the top N peaks.
    top_n : int, default 3
        Number of peaks to mark.
    
    Returns
    -------
    Axes
        The matplotlib Axes object.
    
    Examples
    --------
    >>> freqs, mags = compute_fft(signal, sample_rate)
    >>> plot_frequency_spectrum(freqs, mags, title="Spectrum of Composite Signal")
    >>> plt.show()
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Plot stem for bar-like appearance
    ax.stem(freqs, magnitudes, linefmt='-', markerfmt=' ', basefmt='k-', 
            use_line_collection=True)
    ax.fill_between(freqs, magnitudes, alpha=0.3, color='#2E86AB')
    
    # Mark top peaks
    if mark_peaks and len(freqs) > 0:
        from .core import find_dominant_frequencies
        peaks = find_dominant_frequencies(freqs, magnitudes, top_n=top_n)
        
        for i, (f, m) in enumerate(peaks):
            if m > 0:  # Skip zero magnitude
                ax.annotate(f'{f:.1f} Hz\n({m:.2f})', 
                           xy=(f, m), 
                           xytext=(f, m * 1.15),
                           ha='center', va='bottom',
                           fontsize=9,
                           arrowprops=dict(arrowstyle='->', color='#E94F37', lw=1.5),
                           color='#E94F37', fontweight='bold')
    
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    
    return ax


def plot_signal_decomposition(t, signal, frequencies, amplitudes, phases,
                               sample_rate, figsize=(14, 10)):
    """
    Plot a composite signal and its decomposition into individual components.
    
    This visualization shows:
    1. The original composite signal
    2. Individual frequency components
    3. The frequency spectrum
    
    Parameters
    ----------
    t : array_like
        Time array.
    signal : array_like
        Composite signal.
    frequencies : list of float
        List of component frequencies.
    amplitudes : list of float
        List of component amplitudes.
    phases : list of float
        List of component phases.
    sample_rate : float
        Sampling rate in Hz.
    figsize : tuple, default (14, 10)
        Figure size.
    
    Returns
    -------
    Figure
        The matplotlib Figure object.
    
    Examples
    --------
    >>> from fourier_pkg import generate_composite_signal, plot_signal_decomposition
    >>> t = np.linspace(0, 1, 1000)
    >>> signal = generate_composite_signal(t, [5, 10, 15], [1, 0.5, 0.3], [0, 0, 0])
    >>> plot_signal_decomposition(t, signal, [5, 10, 15], [1, 0.5, 0.3], [0, 0, 0], 1000)
    >>> plt.show()
    """
    from .core import compute_fft
    
    n_components = len(frequencies)
    n_rows = n_components + 2
    
    fig, axes = plt.subplots(n_rows, 2, figsize=figsize)
    
    # Color cycle for components
    colors = ['#E94F37', '#44AF69', '#2E86AB', '#F18F01', '#8338EC']
    
    # Row 1: Original composite signal
    ax_composite = axes[0, 0] if n_components > 0 else axes[0]
    ax_composite.plot(t, signal, color='#2E86AB', linewidth=1)
    ax_composite.set_title("Composite Signal (Sum of All Components)", fontweight='bold')
    ax_composite.set_xlabel("Time (s)")
    ax_composite.set_ylabel("Amplitude")
    ax_composite.grid(True, alpha=0.3)
    
    # Row 1, right: Frequency spectrum
    freqs, mags = compute_fft(signal, sample_rate)
    axes[0, 1].stem(freqs, mags, linefmt='-', markerfmt=' ', basefmt='k-',
                    use_line_collection=True)
    axes[0, 1].fill_between(freqs, mags, alpha=0.3, color='#2E86AB')
    axes[0, 1].set_title("Frequency Spectrum", fontweight='bold')
    axes[0, 1].set_xlabel("Frequency (Hz)")
    axes[0, 1].set_ylabel("Magnitude")
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Individual components
    for i, (freq, amp, phase) in enumerate(zip(frequencies, amplitudes, phases)):
        component = amp * np.sin(2 * np.pi * freq * t + phase)
        
        # Time domain
        ax_time = axes[i + 1, 0] if n_components > 0 else axes[i + 1]
        ax_time.plot(t, component, color=colors[i % len(colors)], linewidth=1)
        ax_time.set_title(f"Component {i+1}: {freq} Hz (A={amp}, φ={phase:.2f} rad)", 
                          fontweight='bold')
        ax_time.set_xlabel("Time (s)")
        ax_time.set_ylabel("Amplitude")
        ax_time.grid(True, alpha=0.3)
        ax_time.set_ylim([-max(amplitudes) * 1.2, max(amplitudes) * 1.2])
        
        # Spectrum of single component
        freqs_single, mags_single = compute_fft(component, sample_rate)
        axes[i + 1, 1].stem(freqs_single, mags_single, linefmt='-', 
                            markerfmt=' ', basefmt='k-', use_line_collection=True)
        axes[i + 1, 1].fill_between(freqs_single, mags_single, alpha=0.3, 
                                    color=colors[i % len(colors)])
        axes[i + 1, 1].set_title(f"Spectrum: {freq} Hz Component", fontweight='bold')
        axes[i + 1, 1].set_xlabel("Frequency (Hz)")
        axes[i + 1, 1].set_ylabel("Magnitude")
        axes[i + 1, 1].grid(True, alpha=0.3, axis='y')
        axes[i + 1, 1].set_xlim([0, max(frequencies) * 2])
    
    plt.tight_layout()
    return fig


def plot_combined_analysis(t, signal, sample_rate, 
                            true_frequencies=None, figsize=(14, 10)):
    """
    Create a comprehensive 4-panel analysis of a signal.
    
    Panel layout:
    1. Time domain signal
    2. Frequency spectrum with detected peaks
    3. Noisy vs clean comparison (if noise present)
    4. Spectrogram (time-frequency representation)
    
    Parameters
    ----------
    t : array_like
        Time array.
    signal : array_like
        Input signal.
    sample_rate : float
        Sampling rate in Hz.
    true_frequencies : list of float, optional
        True frequencies for comparison.
    figsize : tuple, default (14, 10)
        Figure size.
    
    Returns
    -------
    Figure
        The matplotlib Figure object.
    
    Examples
    --------
    >>> plot_combined_analysis(t, signal, 1000, true_frequencies=[5, 10, 15])
    >>> plt.show()
    """
    from .core import compute_fft, find_dominant_frequencies
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Panel 1: Time Domain
    ax1 = axes[0, 0]
    ax1.plot(t, signal, color='#2E86AB', linewidth=0.8)
    ax1.set_title("Time Domain", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([t.min(), t.max()])
    
    # Panel 2: Frequency Spectrum
    freqs, mags = compute_fft(signal, sample_rate)
    peaks = find_dominant_frequencies(freqs, mags, top_n=5)
    
    ax2 = axes[0, 1]
    ax2.stem(freqs, mags, linefmt='-', markerfmt=' ', basefmt='k-',
             use_line_collection=True)
    ax2.fill_between(freqs, mags, alpha=0.3, color='#2E86AB')
    
    # Mark detected peaks
    for f, m in peaks:
        if m > 0:
            ax2.annotate(f'{f:.1f} Hz', xy=(f, m), xytext=(f, m * 1.2),
                        ha='center', fontsize=9, color='#E94F37', fontweight='bold')
    
    # Mark true frequencies if provided
    if true_frequencies:
        for tf in true_frequencies:
            ax2.axvline(x=tf, color='#44AF69', linestyle='--', alpha=0.7, 
                       label=f'True: {tf} Hz')
    
    ax2.set_title("Frequency Spectrum (with Peak Detection)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Magnitude")
    ax2.grid(True, alpha=0.3, axis='y')
    
    if true_frequencies:
        ax2.legend(loc='upper right', fontsize=8)
    
    # Panel 3: Phase spectrum
    ax3 = axes[1, 0]
    fft_result = np.fft.fft(signal)
    positive_mask = np.fft.fftfreq(len(signal), 1/sample_rate) >= 0
    phases = np.angle(fft_result[positive_mask])
    
    ax3.plot(freqs, phases, color='#F18F01', linewidth=1)
    ax3.set_title("Phase Spectrum", fontsize=12, fontweight='bold')
    ax3.set_xlabel("Frequency (Hz)")
    ax3.set_ylabel("Phase (radians)")
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([-np.pi, np.pi])
    ax3.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax3.set_yticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
    
    # Panel 4: Spectrogram
    ax4 = axes[1, 1]
    from matplotlib.colors import LogNorm
    
    # Compute spectrogram
    window = int(sample_rate * 0.1)  # 100ms window
    noverlap = window // 2
    f_spec, t_spec, Sxx = plt.specgram(signal, Fs=sample_rate, NFFT=window,
                                         noverlap=noverlap, cmap='viridis')
    
    im = ax4.pcolormesh(t_spec, f_spec, 10 * np.log10(Sxx + 1e-10), 
                        shading='gouraud', cmap='viridis')
    ax4.set_title("Spectrogram (Time-Frequency)", fontsize=12, fontweight='bold')
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Frequency (Hz)")
    cbar = plt.colorbar(im, ax=ax4)
    cbar.set_label('Power/Frequency (dB/Hz)', fontsize=9)
    
    plt.tight_layout()
    return fig


def plot_nyquist_demo(sample_rate, signal_freq, figsize=(12, 5)):
    """
    Visualize the Nyquist sampling theorem.
    
    Shows how undersampling creates aliasing artifacts.
    
    Parameters
    ----------
    sample_rate : float
        Sampling rate in Hz.
    signal_freq : float
        Signal frequency in Hz.
    figsize : tuple, default (12, 5)
        Figure size.
    
    Examples
    --------
    >>> plot_nyquist_demo(sample_rate=100, signal_freq=60)  # Undersampling
    >>> plt.show()
    """
    duration = 0.1  # 100ms
    t = np.arange(0, duration, 1 / sample_rate)
    
    # Original continuous signal (high sample rate for visualization)
    t_continuous = np.linspace(0, duration, 10000)
    continuous_signal = np.sin(2 * np.pi * signal_freq * t_continuous)
    
    # Sampled signal
    sampled_signal = np.sin(2 * np.pi * signal_freq * t)
    
    # Alias frequency
    alias_freq = abs(2 * (sample_rate / 2 - signal_freq)) if signal_freq > sample_rate / 2 else signal_freq
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Left: Time domain comparison
    ax1 = axes[0]
    ax1.plot(t_continuous * 1000, continuous_signal, 'b-', alpha=0.5, 
             linewidth=2, label='True signal')
    ax1.stem(t * 1000, sampled_signal, linefmt='r-', markerfmt='ro', 
             basefmt='k-', use_line_collection=True, label='Samples')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title(f'Sampling at {sample_rate} Hz\nSignal: {signal_freq} Hz', 
                  fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, duration * 1000])
    
    # Right: Frequency domain
    from .core import compute_fft
    freqs, mags = compute_fft(sampled_signal, sample_rate)
    
    ax2 = axes[1]
    ax2.stem(freqs, mags, linefmt='-', markerfmt=' ', basefmt='k-',
             use_line_collection=True)
    ax2.fill_between(freqs, mags, alpha=0.3, color='#2E86AB')
    ax2.axvline(x=sample_rate / 2, color='r', linestyle='--', 
                label=f'Nyquist: {sample_rate/2} Hz')
    
    if signal_freq > sample_rate / 2:
        ax2.axvline(x=alias_freq, color='orange', linestyle=':', 
                   label=f'Alias: {alias_freq:.0f} Hz')
    
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Magnitude')
    ax2.set_title('Frequency Spectrum', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xlim([0, sample_rate])
    
    plt.tight_layout()
    return fig
