import numpy as np
from scipy.signal import welch

# EEG bands in Hz
eeg_bands = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 12),
    'beta': (12, 30),
    'gamma': (30, 45)
}

def compute_psd(signal, fs=256):
    
    freqs, psd = welch(signal, fs=fs, nperseg=fs*2)
    return freqs, psd

def bandpower(freqs, psd, band):
 
    idx = np.logical_and(freqs >= band[0], freqs <= band[1])
    return np.trapz(psd[idx], freqs[idx])

def extract_bandpowers(signal, fs=256):

    freqs, psd = compute_psd(signal, fs)
    return {band: bandpower(freqs, psd, limits) for band, limits in eeg_bands.items()}
