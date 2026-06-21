import numpy as np


def _ensure_2d(sequence: np.ndarray) -> np.ndarray:
    arr = np.asarray(sequence, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D [seq_len, features], got shape={arr.shape}")
    return arr


def _fft_features(series: np.ndarray, top_k: int = 8) -> np.ndarray:
    # rfft is used instead of fft because the input is real-valued;
    # it returns only the non-redundant positive-frequency components.
    # Taking magnitudes discards phase, keeping only spectral energy per frequency.
    spec = np.fft.rfft(series)
    mag = np.abs(spec).astype(np.float32)
    if len(mag) < top_k:
        padded = np.zeros(top_k, dtype=np.float32)
        padded[: len(mag)] = mag
        return padded
    # The lowest-frequency bins carry the most energy for typical market series;
    # keeping only the top-k bins is a compact spectral summary.
    return mag[:top_k]


def _haar_detail_energy(series: np.ndarray, levels: int = 3) -> np.ndarray:
    # Single-level Haar: split into pairwise averages (approximation 'a')
    # and pairwise differences (detail 'd').  Energy of 'd' at each level
    # measures high-frequency variation at that scale.
    x = np.asarray(series, dtype=np.float32).copy()
    energies: list[float] = []
    for _ in range(levels):
        if len(x) < 2:
            energies.append(0.0)
            continue
        if len(x) % 2 == 1:
            x = x[:-1]
        a = (x[0::2] + x[1::2]) / 2.0  # approximation (low frequency)
        d = (x[0::2] - x[1::2]) / 2.0  # detail (high frequency at this scale)
        energies.append(float(np.mean(d * d)))
        x = a  # recurse on approximation to capture coarser scales
    return np.asarray(energies, dtype=np.float32)


def compute_transform_features(
    sequence: np.ndarray, fft_top_k: int = 8, wavelet_levels: int = 3
) -> np.ndarray:
    arr = _ensure_2d(sequence)
    outputs = []
    for col in range(arr.shape[1]):
        series = arr[:, col]
        fft_repr = _fft_features(series, top_k=fft_top_k)
        wavelet_repr = _haar_detail_energy(series, levels=wavelet_levels)
        outputs.append(np.concatenate([fft_repr, wavelet_repr], axis=0))
    return np.concatenate(outputs, axis=0).astype(np.float32)


def transform_feature_dim(n_cols: int = 5, fft_top_k: int = 8, wavelet_levels: int = 3) -> int:
    """
    Return the output dimension of ``compute_transform_features`` without
    running it.  Use this to size the aggregator's ``branch_dims`` entry.

    >>> transform_feature_dim(n_cols=5, fft_top_k=8, wavelet_levels=3)
    55
    """
    return n_cols * (fft_top_k + wavelet_levels)


def batch_transform_features(
    sequences: np.ndarray, fft_top_k: int = 8, wavelet_levels: int = 3
) -> np.ndarray:
    arr = np.asarray(sequences, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D [N, seq_len, features], got shape={arr.shape}")
    return np.stack(
        [
            compute_transform_features(seq, fft_top_k=fft_top_k, wavelet_levels=wavelet_levels)
            for seq in arr
        ],
        axis=0,
    )