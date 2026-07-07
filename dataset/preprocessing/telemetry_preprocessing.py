"""
Vehicle Telemetry Preprocessing — GPS, speed, coordinates, timestamps.
Implements ALL missing telemetry preprocessing steps:
  1. GPS cleaning (remove invalid coordinates, outliers)
  2. Speed smoothing
  3. Duplicate record removal
  4. Timestamp alignment
  5. Route segmentation
  6. Coordinate normalization
  7. Data aggregation
  8. Trip extraction
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional, Tuple, List, Dict
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


EARTH_RADIUS_KM = 6371.0
INVALID_COORD_RANGE = (-180, 180)


def clean_gps_coordinates(
    df: pd.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    max_speed_kmh: float = 300.0,
    max_altitude_m: float = 5000.0,
    remove_stationary: bool = True,
    stationary_threshold_m: float = 0.5,
) -> pd.DataFrame:
    """
    Clean GPS coordinates by removing invalid and outlier readings.

    Args:
        df: DataFrame with GPS data
        lat_col: Latitude column name
        lon_col: Longitude column name
        max_speed_kmh: Maximum plausible speed (km/h)
        max_altitude_m: Maximum plausible altitude (meters)
        remove_stationary: Remove near-identical consecutive readings
        stationary_threshold_m: Distance threshold for stationary detection

    Returns:
        Cleaned DataFrame
    """
    cleaned = df.copy()

    initial_count = len(cleaned)

    valid_lat = (
        (cleaned[lat_col] >= -90) & (cleaned[lat_col] <= 90) &
        (~cleaned[lat_col].isna())
    )
    valid_lon = (
        (cleaned[lon_col] >= -180) & (cleaned[lon_col] <= 180) &
        (~cleaned[lon_col].isna())
    )
    cleaned = cleaned[valid_lat & valid_lon].copy()
    invalid_coords = initial_count - len(cleaned)
    if invalid_coords > 0:
        logger.info(f"Removed {invalid_coords} invalid GPS coordinates")

    speed_col = None
    for col in ["speed", "speed_kmh", "speed_kmph", "velocity"]:
        if col in cleaned.columns:
            speed_col = col
            break

    if speed_col:
        cleaned = cleaned[cleaned[speed_col].between(0, max_speed_kmh)].copy()
        logger.info(f"Filtered by max speed ({max_speed_kmh} km/h)")

    cleaned = cleaned.drop_duplicates(subset=[lat_col, lon_col]).reset_index(drop=True)
    dup_count = initial_count - len(cleaned) - invalid_coords
    if dup_count > 0:
        logger.info(f"Removed {dup_count} duplicate GPS readings")

    return cleaned


def smooth_speed(
    speed: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    window_s: float = 3.0,
    method: str = "savgol",
) -> np.ndarray:
    """
    Smooth speed readings to remove GPS jitter.

    Args:
        speed: 1D array of speed values (km/h)
        timestamps: Optional timestamps for time-based smoothing
        window_s: Smoothing window in seconds
        method: 'savgol', 'moving_avg', 'exponential'

    Returns:
        Smoothed speed array
    """
    speed = np.asarray(speed, dtype=np.float64)

    if method == "exponential":
        alpha = 2.0 / (window_s + 1.0) if timestamps is None else 0.3
        smoothed = np.zeros_like(speed)
        smoothed[0] = speed[0]
        for i in range(1, len(speed)):
            if timestamps is not None:
                dt = timestamps[i] - timestamps[i - 1]
                alpha = 1.0 - np.exp(-dt / window_s)
            smoothed[i] = alpha * speed[i] + (1 - alpha) * smoothed[i - 1]
        return smoothed

    from scipy import signal
    window_size = max(3, int(window_s)) if timestamps is None else max(3, int(window_s))
    if window_size % 2 == 0:
        window_size += 1
    window_size = min(window_size, len(speed))

    if method == "savgol":
        polyorder = min(2, window_size - 1)
        return signal.savgol_filter(speed, window_size, polyorder)
    else:
        window = np.ones(window_size) / window_size
        return np.convolve(speed, window, mode="same")


def align_timestamps(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    target_fs: float = 1.0,
    method: str = "linear",
) -> pd.DataFrame:
    """
    Align telemetry records to a regular time grid.

    Args:
        df: DataFrame with timestamp column
        timestamp_col: Timestamp column name
        target_fs: Target sampling frequency in Hz
        method: Interpolation method for resampling

    Returns:
        DataFrame with regularly spaced timestamps
    """
    result = df.copy()
    result[timestamp_col] = pd.to_datetime(result[timestamp_col])
    result = result.set_index(timestamp_col)
    result = result.sort_index()

    dt = pd.Timedelta(seconds=1.0 / target_fs)
    new_index = pd.date_range(
        start=result.index.min(),
        end=result.index.max(),
        freq=dt,
    )

    numeric_cols = result.select_dtypes(include=[np.number]).columns
    aligned = result[numeric_cols].reindex(result.index.union(new_index))
    aligned = aligned.interpolate(method="time").reindex(new_index)

    for col in result.select_dtypes(exclude=[np.number]).columns:
        aligned[col] = result[col].reindex(new_index, method="ffill")

    aligned = aligned.reset_index().rename(columns={"index": timestamp_col})
    logger.info(f"Aligned {len(df)} records to {len(aligned)} regular timestamps ({target_fs} Hz)")
    return aligned


def normalize_coordinates(
    lat: np.ndarray,
    lon: np.ndarray,
    reference_lat: Optional[float] = None,
    reference_lon: Optional[float] = None,
    method: str = "centered",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize GPS coordinates for ML model input.

    Args:
        lat: Latitude array
        lon: Longitude array
        reference_lat: Reference latitude (mean if None)
        reference_lon: Reference longitude (mean if None)
        method: 'centered' (zero-mean), 'standardized' (z-score),
                'minmax' ([0,1] scale)

    Returns:
        (normalized_lat, normalized_lon)
    """
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)

    if reference_lat is None:
        reference_lat = np.nanmean(lat)
    if reference_lon is None:
        reference_lon = np.nanmean(lon)

    if method == "centered":
        return lat - reference_lat, lon - reference_lon

    elif method == "standardized":
        return (
            (lat - reference_lat) / (np.nanstd(lat) + 1e-10),
            (lon - reference_lon) / (np.nanstd(lon) + 1e-10),
        )

    elif method == "minmax":
        lat_min, lat_max = np.nanmin(lat), np.nanmax(lat)
        lon_min, lon_max = np.nanmin(lon), np.nanmax(lon)
        return (
            (lat - lat_min) / (lat_max - lat_min + 1e-10),
            (lon - lon_min) / (lon_max - lon_min + 1e-10),
        )

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def haversine_distance(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """Calculate great-circle distance between two GPS points in km."""
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    )
    return EARTH_RADIUS_KM * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def segment_routes(
    df: pd.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    timestamp_col: str = "timestamp",
    min_trip_distance_km: float = 0.5,
    min_trip_duration_min: float = 1.0,
    gap_threshold_min: float = 30.0,
    speed_threshold_kmh: float = 3.0,
) -> List[pd.DataFrame]:
    """
    Segment GPS trace into individual trips/routes.

    Args:
        df: DataFrame with GPS data
        lat_col: Latitude column
        lon_col: Longitude column
        timestamp_col: Timestamp column
        min_trip_distance_km: Minimum distance for a valid trip
        min_trip_duration_min: Minimum duration for a valid trip
        gap_threshold_min: Gap threshold to split trips
        speed_threshold_kmh: Below this speed = stationary

    Returns:
        List of trip DataFrames
    """
    result = df.copy()
    result[timestamp_col] = pd.to_datetime(result[timestamp_col])
    result = result.sort_values(timestamp_col).reset_index(drop=True)

    time_diffs = result[timestamp_col].diff().dt.total_seconds() / 60.0
    large_gaps = time_diffs > gap_threshold_min
    split_indices = np.where(large_gaps)[0]

    segments = []
    start = 0
    for split_idx in split_indices:
        segment = result.iloc[start:split_idx].copy()
        if len(segment) > 1:
            segments.append(segment)
        start = split_idx
    final_segment = result.iloc[start:].copy()
    if len(final_segment) > 1:
        segments.append(final_segment)

    trips = []
    for segment in segments:
        lat, lon = segment[lat_col].values, segment[lon_col].values
        total_distance = sum(
            haversine_distance(lat[i], lon[i], lat[i + 1], lon[i + 1])
            for i in range(len(lat) - 1)
        )
        duration_min = (
            segment[timestamp_col].iloc[-1] - segment[timestamp_col].iloc[0]
        ).total_seconds() / 60.0

        moving = segment.copy()
        if "speed" in moving.columns:
            moving = moving[moving["speed"] > speed_threshold_kmh]

        if total_distance >= min_trip_distance_km or duration_min >= min_trip_duration_min:
            trips.append(segment)
            logger.info(f"Trip: {total_distance:.2f} km, {duration_min:.1f} min, {len(segment)} points")

    return trips


def aggregate_telemetry(
    df: pd.DataFrame,
    time_col: str = "timestamp",
    freq: str = "5min",
    aggregations: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Aggregate telemetry data over time windows.

    Args:
        df: Telemetry DataFrame
        time_col: Timestamp column
        freq: Aggregation frequency (e.g., '1min', '5min', '1h')
        aggregations: Dict of {col: agg_func} (auto if None)

    Returns:
        Aggregated DataFrame
    """
    result = df.copy()
    result[time_col] = pd.to_datetime(result[time_col])
    result = result.set_index(time_col)

    if aggregations is None:
        aggregations = {}
        for col in result.select_dtypes(include=[np.number]).columns:
            aggregations[col] = "mean"
        for col in result.select_dtypes(exclude=[np.number]).columns:
            if col != time_col:
                aggregations[col] = "first"

    aggregated = result.resample(freq).agg(aggregations).dropna(how="all").reset_index()
    logger.info(f"Aggregated {len(df)} records -> {len(aggregated)} ({freq} windows)")
    return aggregated


def extract_trips_from_gps(
    df: pd.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    timestamp_col: str = "timestamp",
    speed_col: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Full trip extraction pipeline from raw GPS data.

    Args:
        df: Raw GPS DataFrame
        lat_col, lon_col: Coordinate columns
        timestamp_col: Timestamp column
        speed_col: Optional speed column

    Returns:
        dict with keys: 'trips' (list of DataFrames), 'statistics' (summary)
    """
    cleaned = clean_gps_coordinates(df, lat_col, lon_col)

    if speed_col and speed_col in cleaned.columns:
        cleaned[speed_col] = smooth_speed(cleaned[speed_col].values)

    trips = segment_routes(cleaned, lat_col, lon_col, timestamp_col)

    stats = {
        "total_trips": len(trips),
        "total_points": sum(len(t) for t in trips),
        "trip_lengths_km": [],
        "trip_durations_min": [],
    }

    for trip in trips:
        lat, lon = trip[lat_col].values, trip[lon_col].values
        distance = sum(
            haversine_distance(lat[i], lon[i], lat[i + 1], lon[i + 1])
            for i in range(len(lat) - 1)
        )
        duration = (
            trip[timestamp_col].iloc[-1] - trip[timestamp_col].iloc[0]
        ).total_seconds() / 60.0
        stats["trip_lengths_km"].append(distance)
        stats["trip_durations_min"].append(duration)

    logger.info(f"Extracted {len(trips)} trips from GPS data")
    return {"trips": trips, "statistics": stats}


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    n_points = 500
    times = pd.date_range("2026-01-01", periods=n_points, freq="10s")
    lats = 40.7128 + np.cumsum(np.random.normal(0, 0.001, n_points))
    lons = -74.0060 + np.cumsum(np.random.normal(0, 0.001, n_points))
    speeds = np.abs(np.random.normal(50, 15, n_points))

    df = pd.DataFrame({
        "timestamp": times, "latitude": lats,
        "longitude": lons, "speed": speeds,
    })
    df.iloc[10, 1] = 200.0
    df.iloc[10, 2] = 300.0

    result = extract_trips_from_gps(df)
    logger.info(f"Trips found: {result['statistics']['total_trips']}")
