"""
Sensor Data Preprocessing — TPMS, pressure, temperature, vibration, accelerometer.
Implements ALL missing sensor preprocessing steps:
  1. Outlier detection (IQR, Z-score)
  2. Noise filtering (moving average, Savitzky-Golay)
  3. Signal smoothing
  4. Time synchronization
  5. Sampling rate adjustment (resampling)
  6. Interpolation (linear, spline)
  7. Drift correction
  8. Sensor calibration
  9. Unit conversion
 10. Windowing
 11. Signal segmentation
 12. Frequency domain transformation (FFT)
 13. Wavelet transformation
"""

import numpy as np
import logging
from typing import Optional, Tuple, List, Union, Callable
from scipy import signal, ndimage, interpolate
from scipy.fft import fft, fftfreq

logger = logging.getLogger(__name__)


def detect_outliers_iqr(
    data: np.ndarray,
    multiplier: float = 1.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect outliers using Interquartile Range method.

    Args:
        data: 1D array of sensor readings
        multiplier: IQR multiplier (1.5 = mild, 3.0 = extreme)

    Returns:
        (clean_mask, outlier_indices) where clean_mask is boolean array
    """
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    clean_mask = (data >= lower) & (data <= upper)
    outlier_indices = np.where(~clean_mask)[0]
    if len(outlier_indices) > 0:
        logger.info(f"IQR outlier detection: {len(outlier_indices)} outliers (multiplier={multiplier})")
    return clean_mask, outlier_indices


def detect_outliers_zscore(
    data: np.ndarray,
    threshold: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect outliers using Z-score method.

    Args:
        data: 1D array of sensor readings
        threshold: Z-score threshold (3.0 recommended)

    Returns:
        (clean_mask, outlier_indices)
    """
    z_scores = np.abs((data - np.nanmean(data)) / (np.nanstd(data) + 1e-10))
    clean_mask = z_scores < threshold
    outlier_indices = np.where(~clean_mask)[0]
    if len(outlier_indices) > 0:
        logger.info(f"Z-score outlier detection: {len(outlier_indices)} outliers (threshold={threshold})")
    return clean_mask, outlier_indices


def replace_outliers(
    data: np.ndarray,
    outlier_indices: np.ndarray,
    method: str = "median",
) -> np.ndarray:
    """
    Replace outlier values with estimated values.

    Args:
        data: 1D array of sensor readings
        outlier_indices: Indices of outliers
        method: 'median' (global median), 'mean' (global mean),
                'neighbor' (local neighbor average), 'interpolate' (linear)

    Returns:
        Cleaned array
    """
    cleaned = data.copy()
    if len(outlier_indices) == 0:
        return cleaned

    if method == "median":
        replacement = np.nanmedian(data[~np.isnan(data)])
        cleaned[outlier_indices] = replacement
    elif method == "mean":
        replacement = np.nanmean(data[~np.isnan(data)])
        cleaned[outlier_indices] = replacement
    elif method == "neighbor":
        for idx in outlier_indices:
            left = max(0, idx - 2)
            right = min(len(data) - 1, idx + 2)
            neighbors = [data[i] for i in range(left, right + 1) if i != idx and not np.isnan(data[i])]
            cleaned[idx] = np.mean(neighbors) if neighbors else np.nanmedian(data)
    elif method == "interpolate":
        good = np.where(~np.isin(np.arange(len(data)), outlier_indices))[0]
        if len(good) > 1:
            cleaned[outlier_indices] = np.interp(outlier_indices, good, data[good])
    else:
        raise ValueError(f"Unknown replacement method: {method}")

    return cleaned


def smooth_signal(
    data: np.ndarray,
    method: str = "savgol",
    window_length: int = 11,
    polyorder: int = 3,
) -> np.ndarray:
    """
    Smooth sensor signal to reduce noise while preserving trends.

    Args:
        data: 1D sensor array
        method: 'savgol' (Savitzky-Golay), 'moving_avg', 'gaussian', 'median'
        window_length: Window length (must be odd for savgol)
        polyorder: Polynomial order for Savitzky-Golay

    Returns:
        Smoothed signal
    """
    data = np.asarray(data, dtype=np.float64)
    n = len(data)

    if method == "savgol":
        if window_length % 2 == 0:
            window_length += 1
        window_length = min(window_length, n if n % 2 == 1 else n - 1)
        if window_length < 3:
            return data
        return signal.savgol_filter(data, window_length, polyorder)

    elif method == "moving_avg":
        window = np.ones(window_length) / window_length
        return np.convolve(data, window, mode="same")

    elif method == "gaussian":
        sigma = window_length / 5.0
        return ndimage.gaussian_filter1d(data, sigma=sigma, mode="reflect")

    elif method == "median":
        return ndimage.median_filter(data, size=window_length)

    else:
        raise ValueError(f"Unknown smoothing method: {method}")


def interpolate_missing(
    data: np.ndarray,
    method: str = "linear",
    max_gap: Optional[int] = None,
) -> np.ndarray:
    """
    Interpolate missing (NaN) values in sensor data.

    Args:
        data: 1D array possibly containing NaN values
        method: 'linear', 'cubic', 'quadratic', 'nearest', 'previous', 'next'
        max_gap: Maximum gap size to interpolate (None = all gaps)

    Returns:
        Interpolated array
    """
    data = np.asarray(data, dtype=np.float64)
    n = len(data)
    mask = np.isnan(data)

    if not mask.any():
        return data

    if method in ("previous", "next"):
        filled = data.copy()
        if method == "previous":
            for i in range(1, n):
                if np.isnan(filled[i]):
                    filled[i] = filled[i - 1]
        else:
            for i in range(n - 2, -1, -1):
                if np.isnan(filled[i]):
                    filled[i] = filled[i + 1]
        return filled

    good_indices = np.where(~mask)[0]
    good_values = data[good_indices]

    if len(good_indices) < 2:
        return data

    if max_gap is not None:
        gap_starts = np.where(np.diff(np.concatenate([[False], mask, [False]])) == 1)[0]
        gap_ends = np.where(np.diff(np.concatenate([[False], mask, [False]])) == -1)[0]
        valid_mask = mask.copy()
        for start, end in zip(gap_starts, gap_ends):
            if end - start > max_gap:
                valid_mask[start:end] = False
        if not valid_mask.any():
            return data
        bad_indices = np.where(valid_mask)[0]
    else:
        bad_indices = np.where(mask)[0]

    if len(bad_indices) == 0:
        return data

    interpolator = interpolate.interp1d(
        good_indices, good_values, kind=method, bounds_error=False, fill_value="extrapolate"
    )
    result = data.copy()
    result[bad_indices] = interpolator(bad_indices)
    return result


def synchronize_timestamps(
    timestamps: List[np.ndarray],
    values: List[np.ndarray],
    target_fs: float = 10.0,
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Synchronize multiple sensor streams to a common time base.

    Args:
        timestamps: List of timestamp arrays (one per sensor)
        values: List of value arrays (one per sensor)
        target_fs: Target sampling frequency in Hz
        t_start: Start time (auto if None)
        t_end: End time (auto if None)

    Returns:
        (common_time, synced_signals) where synced_signals shape is (n_sensors, n_samples)
    """
    all_times = np.concatenate(timestamps)
    all_times = all_times[~np.isnan(all_times)]

    if t_start is None:
        t_start = np.min(all_times)
    if t_end is None:
        t_end = np.max(all_times)

    dt = 1.0 / target_fs
    common_time = np.arange(t_start, t_end, dt)
    n_samples = len(common_time)
    n_sensors = len(values)

    synced = np.full((n_sensors, n_samples), np.nan)
    for i in range(n_sensors):
        if len(timestamps[i]) < 2:
            continue
        valid = ~np.isnan(values[i])
        if valid.sum() < 2:
            continue
        f = interpolate.interp1d(
            timestamps[i][valid],
            values[i][valid],
            kind="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        synced[i] = f(common_time)

    return common_time, synced


def resample_signal(
    data: np.ndarray,
    original_fs: float,
    target_fs: float,
) -> np.ndarray:
    """
    Change sampling rate of a sensor signal.

    Args:
        data: 1D sensor array
        original_fs: Original sampling frequency in Hz
        target_fs: Target sampling frequency in Hz

    Returns:
        Resampled signal
    """
    if original_fs == target_fs:
        return data

    n_original = len(data)
    n_target = int(n_original * target_fs / original_fs)
    resampled = signal.resample(data, n_target)
    logger.info(f"Resampled {n_original} samples ({original_fs}Hz) -> {n_target} samples ({target_fs}Hz)")
    return resampled


def correct_drift(
    data: np.ndarray,
    method: str = "detrend",
    reference_points: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Correct sensor drift over time.

    Args:
        data: 1D sensor array
        method: 'detrend' (linear detrend), 'mean_center', 'highpass'
        reference_points: Known reference values for calibration correction

    Returns:
        Drift-corrected signal
    """
    if method == "detrend":
        return signal.detrend(data, type="linear")

    elif method == "mean_center":
        return data - np.nanmean(data)

    elif method == "highpass":
        b, a = signal.butter(2, 0.01, btype="high")
        return signal.filtfilt(b, a, data, nan_policy="omit")

    elif method == "calibration" and reference_points is not None:
        if len(reference_points) < 2:
            return data
        x = np.linspace(0, 1, len(reference_points))
        coeffs = np.polyfit(x, reference_points, 1)
        correction = np.polyval(coeffs, np.linspace(0, 1, len(data)))
        return data - correction

    else:
        raise ValueError(f"Unknown drift correction method: {method}")


def calibrate_sensor(
    raw_value: float,
    slope: float = 1.0,
    intercept: float = 0.0,
    calibration_table: Optional[np.ndarray] = None,
) -> float:
    """
    Apply sensor calibration correction.

    Args:
        raw_value: Raw sensor reading
        slope: Linear calibration slope
        intercept: Linear calibration intercept
        calibration_table: (N, 2) array of [raw, calibrated] pairs for lookup

    Returns:
        Calibrated value
    """
    if calibration_table is not None:
        return float(np.interp(raw_value, calibration_table[:, 0], calibration_table[:, 1]))
    return slope * raw_value + intercept


def convert_unit(
    value: float,
    from_unit: str,
    to_unit: str,
    quantity: str = "pressure",
) -> float:
    """
    Convert sensor units.

    Args:
        value: Numeric value
        from_unit: Source unit
        to_unit: Target unit
        quantity: 'pressure', 'temperature', 'speed', 'distance'

    Returns:
        Converted value
    """
    pressure_conversions = {
        ("psi", "bar"): lambda v: v / 14.5038,
        ("bar", "psi"): lambda v: v * 14.5038,
        ("psi", "kpa"): lambda v: v * 6.89476,
        ("kpa", "psi"): lambda v: v / 6.89476,
        ("bar", "kpa"): lambda v: v * 100.0,
        ("kpa", "bar"): lambda v: v / 100.0,
    }

    temperature_conversions = {
        ("c", "f"): lambda v: v * 9.0 / 5.0 + 32.0,
        ("f", "c"): lambda v: (v - 32.0) * 5.0 / 9.0,
        ("c", "k"): lambda v: v + 273.15,
        ("k", "c"): lambda v: v - 273.15,
    }

    speed_conversions = {
        ("kmh", "mph"): lambda v: v / 1.60934,
        ("mph", "kmh"): lambda v: v * 1.60934,
        ("ms", "kmh"): lambda v: v * 3.6,
        ("kmh", "ms"): lambda v: v / 3.6,
    }

    unit_map = {
        "pressure": pressure_conversions,
        "temperature": temperature_conversions,
        "speed": speed_conversions,
    }

    conversions = unit_map.get(quantity, {})
    key = (from_unit.lower(), to_unit.lower())
    converter = conversions.get(key)
    if converter is None:
        logger.warning(f"No conversion found: {from_unit} -> {to_unit} for {quantity}")
        return value

    return converter(value)


def apply_windowing(
    data: np.ndarray,
    window_size: int = 128,
    stride: Optional[int] = None,
    window_fn: str = "hann",
) -> np.ndarray:
    """
    Apply sliding window to sensor signal.

    Args:
        data: 1D sensor array
        window_size: Number of samples per window
        stride: Step between windows (default: window_size // 2)
        window_fn: 'hann', 'hamming', 'blackman', 'bartlett', 'kaiser', 'rectangular'

    Returns:
        2D array of shape (n_windows, window_size)
    """
    if stride is None:
        stride = window_size // 2

    windows = {
        "hann": np.hanning(window_size),
        "hamming": np.hamming(window_size),
        "blackman": np.blackman(window_size),
        "bartlett": np.bartlett(window_size),
        "kaiser": np.kaiser(window_size, beta=14),
        "rectangular": np.ones(window_size),
    }

    win = windows.get(window_fn, np.hanning(window_size))
    n = len(data)
    n_windows = max(1, (n - window_size) // stride + 1)
    result = np.zeros((n_windows, window_size))

    for i in range(n_windows):
        start = i * stride
        end = start + window_size
        if end > n:
            break
        result[i] = data[start:end] * win

    return result[:i+1] if n_windows > 0 else result


def segment_signal(
    data: np.ndarray,
    method: str = "fixed",
    segment_length: int = 256,
    threshold: Optional[float] = None,
    min_gap: int = 50,
) -> List[np.ndarray]:
    """
    Segment sensor signal into meaningful segments.

    Args:
        data: 1D sensor array
        method: 'fixed' (fixed length), 'threshold' (activity-based)
        segment_length: Length for fixed segmentation
        threshold: Activity threshold for threshold-based segmentation
        min_gap: Minimum gap between segments

    Returns:
        List of segment arrays
    """
    segments = []

    if method == "fixed":
        for start in range(0, len(data), segment_length):
            end = min(start + segment_length, len(data))
            segments.append(data[start:end])

    elif method == "threshold":
        if threshold is None:
            threshold = np.nanstd(data) * 0.5
        active = np.abs(data - np.nanmean(data)) > threshold
        diff = np.diff(np.concatenate([[False], active, [False]]))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        for start, end in zip(starts, ends):
            if end - start >= min_gap:
                segments.append(data[start:end])

    return segments


def apply_fft(
    data: np.ndarray,
    sampling_rate: float = 100.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Fast Fourier Transform to sensor signal.

    Args:
        data: 1D sensor array
        sampling_rate: Sampling frequency in Hz

    Returns:
        (frequencies, magnitude) arrays
    """
    n = len(data)
    yf = fft(data - np.nanmean(data))
    xf = fftfreq(n, 1.0 / sampling_rate)
    positive = xf > 0
    return xf[positive], 2.0 / n * np.abs(yf[positive])


def apply_wavelet(
    data: np.ndarray,
    wavelet_name: str = "db4",
    mode: str = "smooth",
    level: int = 4,
) -> np.ndarray:
    """
    Apply wavelet transformation for denoising.

    Args:
        data: 1D sensor array
        wavelet_name: Wavelet type (e.g., 'db4', 'sym5', 'coif3')
        mode: 'smooth' (denoise), 'detail' (extract details)
        level: Decomposition level

    Returns:
        Transformed/denoised signal
    """
    try:
        import pywt
    except ImportError:
        logger.warning("PyWavelets not installed. Install with: pip install PyWavelets")
        return data

    coeffs = pywt.wavedec(data, wavelet_name, level=level)

    if mode == "smooth":
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(data)))
        coeffs = [coeffs[0]] + [pywt.threshold(c, threshold, mode="soft") for c in coeffs[1:]]
        return pywt.waverec(coeffs, wavelet_name)[:len(data)]

    elif mode == "detail":
        for i in range(1, len(coeffs)):
            coeffs[i] = pywt.threshold(coeffs[i], np.std(coeffs[i]) * 2, mode="hard")
        return pywt.waverec(coeffs, wavelet_name)[:len(data)]

    return data


def preprocess_sensor_pipeline(
    data: np.ndarray,
    sampling_rate: float = 100.0,
    detect_outliers: bool = True,
    outlier_method: str = "iqr",
    smooth: bool = True,
    smooth_method: str = "savgol",
    interpolate: bool = True,
    interpolate_method: str = "linear",
    correct_drift_flag: bool = False,
    drift_method: str = "detrend",
    apply_fft_flag: bool = False,
    apply_wavelet_flag: bool = False,
    wavelet_denoise: bool = False,
) -> dict:
    """
    Complete sensor preprocessing pipeline.

    Args:
        data: 1D sensor array
        sampling_rate: Original sampling frequency
        detect_outliers: Enable outlier detection and replacement
        outlier_method: 'iqr' or 'zscore'
        smooth: Enable signal smoothing
        smooth_method: 'savgol', 'moving_avg', 'gaussian', 'median'
        interpolate: Enable missing value interpolation
        interpolate_method: 'linear', 'cubic', etc.
        correct_drift_flag: Enable drift correction
        drift_method: 'detrend', 'mean_center', 'highpass'
        apply_fft_flag: Compute FFT
        apply_wavelet_flag: Apply wavelet transform
        wavelet_denoise: Apply wavelet denoising

    Returns:
        dict with keys: 'cleaned', 'outlier_indices', 'fft_freq', 'fft_mag', 'wavelet'
    """
    result = {"cleaned": data.copy(), "outlier_indices": np.array([], dtype=int)}

    if interpolate:
        result["cleaned"] = interpolate_missing(result["cleaned"], method=interpolate_method)
        logger.info("Interpolated missing values")

    if detect_outliers:
        if outlier_method == "iqr":
            _, outlier_idx = detect_outliers_iqr(result["cleaned"])
        else:
            _, outlier_idx = detect_outliers_zscore(result["cleaned"])
        if len(outlier_idx) > 0:
            result["cleaned"] = replace_outliers(result["cleaned"], outlier_idx, method="interpolate")
            result["outlier_indices"] = outlier_idx

    if correct_drift_flag:
        result["cleaned"] = correct_drift(result["cleaned"], method=drift_method)
        logger.info(f"Applied drift correction: {drift_method}")

    if smooth:
        result["cleaned"] = smooth_signal(result["cleaned"], method=smooth_method)
        logger.info(f"Applied signal smoothing: {smooth_method}")

    if apply_wavelet_flag or wavelet_denoise:
        mode = "smooth" if wavelet_denoise else "detail"
        result["wavelet"] = apply_wavelet(result["cleaned"], mode=mode)

    if apply_fft_flag:
        f, m = apply_fft(result["cleaned"], sampling_rate)
        result["fft_freq"] = f
        result["fft_mag"] = m

    return result


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    np.random.seed(42)
    t = np.linspace(0, 10, 1000)
    clean = 30 + 5 * np.sin(2 * np.pi * 0.5 * t)
    noisy = clean + np.random.normal(0, 2, len(t))
    noisy[50:60] = np.nan
    noisy[200:205] = 100

    result = preprocess_sensor_pipeline(
        noisy,
        sampling_rate=100.0,
        detect_outliers=True,
        smooth=True,
        interpolate=True,
        apply_fft_flag=True,
    )
    logger.info(f"Input: {len(noisy)}, Output: {len(result['cleaned'])}")
    logger.info(f"Outliers found: {len(result['outlier_indices'])}")
    if "fft_mag" in result:
        logger.info(f"FFT peaks: {len(result['fft_mag'])}")
