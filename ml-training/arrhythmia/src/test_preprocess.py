import numpy as np
from preprocess import extract_beat_window, compute_rr_features, normalize_beat, resample_to_target


def test_beat_window_shape_and_r_position():
    signal = np.arange(1000, dtype=np.float32)
    r_index = 200
    window = extract_beat_window(signal, r_index)
    assert window.shape == (256,)
    assert window[90] == r_index


def test_beat_window_returns_none_near_boundary():
    signal = np.arange(100, dtype=np.float32)
    assert extract_beat_window(signal, 5) is None
    assert extract_beat_window(signal, 95) is None


def test_normalize_beat_zero_mean_unit_std():
    beat = np.random.randn(256) * 5 + 3
    normalized = normalize_beat(beat)
    assert abs(normalized.mean()) < 1e-5
    assert abs(normalized.std() - 1.0) < 1e-5


def test_rr_features_first_and_last_beat():
    r_peaks = np.array([100, 460, 820, 1180])
    fs = 360

    first = compute_rr_features(r_peaks, 0, fs)
    assert first[0] == first[1]

    last = compute_rr_features(r_peaks, len(r_peaks) - 1, fs)
    assert last[0] == last[1]


def test_rr_ratio_formula():
    r_peaks = np.array([0, 360, 720, 1080, 1440])
    fs = 360
    rr = compute_rr_features(r_peaks, 2, fs)
    rr_prev, rr_next, rr_local, rr_ratio = rr
    assert abs(rr_ratio - (rr_prev / rr_local)) < 1e-5


def test_resample_length_scales_correctly():
    signal = np.random.randn(250)
    resampled = resample_to_target(signal, orig_fs=250, target_fs=360)
    expected_len = int(250 * 360 / 250)
    assert abs(len(resampled) - expected_len) <= 1
