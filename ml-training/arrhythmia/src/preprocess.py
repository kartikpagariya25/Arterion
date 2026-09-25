import numpy as np
from scipy.signal import butter, sosfiltfilt, resample_poly
from fractions import Fraction

from config import SAMPLING_RATE, BEAT_WINDOW, PRE_R, POST_R, MIN_PEAK_DISTANCE_MS
from config import BANDPASS_LOW, BANDPASS_HIGH, BANDPASS_ORDER


def resample_to_target(signal: np.ndarray, orig_fs: int, target_fs: int = SAMPLING_RATE) -> np.ndarray:
    if orig_fs == target_fs:
        return signal
    frac = Fraction(target_fs, orig_fs).limit_denominator(1000)
    return resample_poly(signal, frac.numerator, frac.denominator)


def bandpass_filter(signal: np.ndarray, fs: int = SAMPLING_RATE) -> np.ndarray:
    sos = butter(BANDPASS_ORDER, [BANDPASS_LOW, BANDPASS_HIGH], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, signal)


def detect_r_peaks(cleaned_signal: np.ndarray, fs: int = SAMPLING_RATE) -> np.ndarray:
    import neurokit2 as nk

    _, info = nk.ecg_peaks(cleaned_signal, sampling_rate=fs, method="neurokit")
    peaks = info["ECG_R_Peaks"]

    min_distance = int(fs * MIN_PEAK_DISTANCE_MS / 1000)
    filtered = []
    last = -min_distance
    for p in peaks:
        if p - last >= min_distance:
            filtered.append(p)
            last = p
    return np.array(filtered)


def extract_beat_window(signal: np.ndarray, r_index: int) -> np.ndarray | None:
    start = r_index - PRE_R
    end = r_index + POST_R
    if start < 0 or end > len(signal):
        return None
    return signal[start:end]


def normalize_beat(beat: np.ndarray) -> np.ndarray:
    mean = beat.mean()
    std = beat.std()
    return (beat - mean) / (std + 1e-8)


def compute_rr_features(r_peaks: np.ndarray, index: int, fs: int = SAMPLING_RATE) -> np.ndarray:
    n = len(r_peaks)

    if index == 0:
        rr_next = (r_peaks[1] - r_peaks[0]) / fs if n > 1 else 0.8
        rr_prev = rr_next
    else:
        rr_prev = (r_peaks[index] - r_peaks[index - 1]) / fs

    if index == n - 1:
        rr_next = rr_prev
    elif index > 0:
        rr_next = (r_peaks[index + 1] - r_peaks[index]) / fs

    window_start = max(0, index - 10)
    local_intervals = np.diff(r_peaks[window_start:index + 1]) / fs
    rr_local = local_intervals.mean() if len(local_intervals) > 0 else rr_prev

    rr_ratio = rr_prev / rr_local if rr_local > 0 else 1.0

    return np.array([rr_prev, rr_next, rr_local, rr_ratio], dtype=np.float32)
