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
def shannon_entropy(signal: np.ndarray, bins: int = 64, value_range=None, base: float = 2.0) -> float:
    signal = np.asarray(signal).ravel()
    if signal.size == 0 or np.all(signal == signal[0]):
        return 0.0
    counts, _ = np.histogram(signal, bins=bins, range=value_range, density=False)
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    
    p_nz = p[p > 0]
    return float(-(p_nz * (np.log(p_nz) / np.log(base))).sum())

def line_length(signal: np.ndarray) -> float:
    x = np.asarray(signal).ravel()
    if x.size < 2: 
        return 0.0
    return float(np.abs(np.diff(x)).sum())

def zero_crossings(signal: np.ndarray) -> int:
    x = np.asarray(signal).ravel()
    if x.size < 2:
        return 0
    # Count sign changes in the signal
    x = np.where(x == 0, 1e-12, x)
    return int(np.sum(np.sign(x[:-1]) != np.sign(x[1:])))

def hjorth_parameters(signal: np.ndarray) -> dict:
    x = np.asarray(signal).ravel()
    if x.size < 2:
        return {"hjorth_activity": 0.0, "hjorth_mobility": 0.0, "hjorth_complexity": 0.0}

    dx = np.diff(x)
    var_x = float(np.var(x))
    var_dx = float(np.var(dx))

    activity = var_x
    mobility = np.sqrt(var_dx / var_x) if var_x > 0 else 0.0

    ddx = np.diff(dx) if dx.size > 1 else np.array([0.0])
    var_ddx = float(np.var(ddx)) if ddx.size > 1 else 0.0
    mobility_dx = np.sqrt(var_ddx / var_dx) if var_dx > 0 else 0.0

    complexity = (mobility_dx / mobility) if mobility > 0 else 0.0

    return {
        "hjorth_activity": activity,
        "hjorth_mobility": float(mobility),
        "hjorth_complexity": float(complexity),
    }

def extract_advanced_features(signal: np.ndarray, fs: int = 256) -> dict:
    feats = {}
    feats["entropy"] = shannon_entropy(signal, bins=64, value_range=None, base=2.0)
    feats["line_length"] = line_length(signal)
    feats["zero_crossings"] = float(zero_crossings(signal))  
    feats.update(hjorth_parameters(signal))
    return feats

def extract_all_features(signal: np.ndarray, fs: int = 256) -> tuple[np.ndarray, list]:
    """
    Returns:
        features_vector: np.ndarray of shape (n_features,)
        feature_names: list[str] in the exact order used
    """
    bp = extract_bandpowers(signal, fs)               # dict of band powers
    adv = extract_advanced_features(signal, fs)       # dict of advanced features

    
    band_names = ["delta", "theta", "alpha", "beta", "gamma"]
    adv_names  = ["entropy", "line_length", "zero_crossings", 
                  "hjorth_activity", "hjorth_mobility", "hjorth_complexity"]

    feature_names = band_names + adv_names
    features = [bp[name] for name in band_names] + [adv[name] for name in adv_names]

    features = np.array(features, dtype=float)
    if not np.all(np.isfinite(features)):
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    return features, feature_names
