import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter, welch
import pywt

FS = 128

# Bandpass filter
def bandpass_filter(data, low=0.5, high=45):
    b, a = butter(4, [low/(FS/2), high/(FS/2)], btype='band')
    return lfilter(b, a, data, axis=0)

# Notch filter (50Hz)
def notch_filter(data):
    from scipy.signal import iirnotch
    b, a = iirnotch(50, 30, FS)
    return lfilter(b, a, data, axis=0)

# Normalize
def normalize(data):
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)

# Sliding window
def sliding_window(data, window_size=2, overlap=0.5):
    step = int(window_size * FS * (1 - overlap))
    size = int(window_size * FS)
    windows = []
    for i in range(0, len(data) - size, step):
        windows.append(data[i:i+size])
    return np.array(windows)

# Differential Entropy
def compute_de(window):
    return 0.5 * np.log(2 * np.pi * np.e * np.var(window, axis=0))

# PSD
def compute_psd(window):
    freqs, psd = welch(window, FS, axis=0)
    return np.mean(psd, axis=0)

# DWT
def compute_dwt(window):
    coeffs = pywt.wavedec(window[:,0], 'db4', level=4)
    energy = [np.sum(np.square(c)) for c in coeffs]
    return np.array(energy)

# Load Excel
def load_excel(file):
    df = pd.read_excel(file)
    return df.values