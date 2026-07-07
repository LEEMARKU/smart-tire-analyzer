"""
Weather and Environmental Data Preprocessing.
Implements ALL missing weather preprocessing steps:
  1. Missing weather value imputation
  2. Temporal alignment
  3. Feature encoding
  4. Weather categorization
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional, Dict, List, Tuple, Union
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

WEATHER_CATEGORIES: Dict[str, List[str]] = {
    "clear": ["clear", "sunny", "fair"],
    "cloudy": ["cloudy", "overcast", "partly cloudy", "mostly cloudy"],
    "rain": ["rain", "drizzle", "light rain", "moderate rain", "heavy rain", "showers"],
    "storm": ["thunderstorm", "thunder", "severe", "hail"],
    "snow": ["snow", "flurries", "blizzard", "sleet", "ice"],
    "fog": ["fog", "mist", "haze", "foggy"],
    "wind": ["windy", "wind", "gust"],
}

PRESSURE_TRENDS = ["rising", "falling", "stable"]


def impute_missing_weather(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "temporal",
    max_gap_hours: float = 6.0,
) -> pd.DataFrame:
    """
    Impute missing weather data values.

    Args:
        df: Weather DataFrame with datetime index
        columns: Columns to impute (all numeric if None)
        method: 'temporal' (time-based interpolation), 'forward' (ffill),
                'backward' (bfill), 'mean' (global mean), 'nearest_station'
        max_gap_hours: Maximum gap to impute

    Returns:
        DataFrame with imputed values
    """
    result = df.copy()

    if columns is None:
        columns = result.select_dtypes(include=[np.number]).columns.tolist()

    for col in columns:
        if col not in result.columns:
            continue

        missing = result[col].isna().sum()
        if missing == 0:
            continue

        if method == "temporal":
            result[col] = result[col].interpolate(method="time", limit=int(max_gap_hours * 4))
        elif method == "forward":
            result[col] = result[col].ffill(limit=int(max_gap_hours * 4))
        elif method == "backward":
            result[col] = result[col].bfill(limit=int(max_gap_hours * 4))
        elif method == "mean":
            result[col] = result[col].fillna(result[col].mean())
        else:
            raise ValueError(f"Unknown imputation method: {method}")

        still_missing = result[col].isna().sum()
        if still_missing > 0:
            result[col] = result[col].fillna(result[col].mean())

        filled = missing - still_missing
        if filled > 0:
            logger.info(f"Imputed {filled} missing values in '{col}' ({method})")

    return result


def align_weather_temporal(
    weather_df: pd.DataFrame,
    target_timestamps: pd.DatetimeIndex,
    columns: Optional[List[str]] = None,
    method: str = "nearest",
    tolerance_minutes: int = 30,
) -> pd.DataFrame:
    """
    Align weather data to target timestamps.

    Args:
        weather_df: Weather DataFrame with datetime index
        target_timestamps: Target timestamps to align to
        columns: Columns to include (all if None)
        method: 'nearest' (nearest neighbor), 'interpolate' (linear),
                'forward' (ffill)
        tolerance_minutes: Max gap for matching

    Returns:
        Aligned DataFrame
    """
    if columns is None:
        columns = weather_df.columns.tolist()

    aligned = pd.DataFrame(index=target_timestamps)

    for col in columns:
        if col not in weather_df.columns:
            continue

        if method == "nearest":
            idx = weather_df.index.get_indexer(target_timestamps, method="nearest", tolerance=pd.Timedelta(minutes=tolerance_minutes))
            valid = idx != -1
            aligned.loc[valid, col] = weather_df[col].iloc[idx[valid]].values
        elif method == "interpolate":
            combined = pd.concat([weather_df[col], pd.Series(index=target_timestamps, dtype=float)])
            combined = combined[~combined.index.duplicated(keep="first")].sort_index()
            aligned[col] = combined.interpolate(method="time").reindex(target_timestamps)
        elif method == "forward":
            aligned[col] = weather_df[col].reindex(target_timestamps, method="ffill", tolerance=pd.Timedelta(minutes=tolerance_minutes))

    return aligned


def encode_weather_features(
    df: pd.DataFrame,
    condition_col: str = "weather_condition",
    temperature_col: str = "temperature_c",
    humidity_col: str = "humidity_pct",
    wind_speed_col: str = "wind_speed_ms",
    pressure_col: str = "pressure_hpa",
) -> pd.DataFrame:
    """
    Encode weather features for ML model input.

    Args:
        df: Weather DataFrame
        condition_col: Weather condition text column
        temperature_col: Temperature column
        humidity_col: Humidity column
        wind_speed_col: Wind speed column
        pressure_col: Pressure column

    Returns:
        DataFrame with encoded weather features
    """
    result = df.copy()

    if condition_col in result.columns:
        result["weather_encoded"] = result[condition_col].apply(_categorize_weather)
        weather_dummies = pd.get_dummies(result["weather_encoded"], prefix="weather", dtype=float)
        result = pd.concat([result, weather_dummies], axis=1)

    if temperature_col in result.columns:
        result["temp_normalized"] = (result[temperature_col] - result[temperature_col].mean()) / (
            result[temperature_col].std() + 1e-10
        )

    if humidity_col in result.columns:
        result["humidity_normalized"] = result[humidity_col] / 100.0

    if wind_speed_col in result.columns:
        result["wind_normalized"] = (result[wind_speed_col] - result[wind_speed_col].min()) / (
            result[wind_speed_col].max() - result[wind_speed_col].min() + 1e-10
        )

    if pressure_col in result.columns:
        result["pressure_trend"] = _compute_pressure_trend(result[pressure_col].values)

    # Combined risk index
    risk_cols = [c for c in result.columns if c.startswith("weather_")]
    if risk_cols:
        weather_risk_map = {
            "weather_clear": 0.0, "weather_cloudy": 0.2,
            "weather_rain": 0.6, "weather_storm": 1.0,
            "weather_snow": 0.8, "weather_fog": 0.7, "weather_wind": 0.5,
        }
        numeric_risk_cols = []
        for col in risk_cols:
            if col in result.columns:
                result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0).astype(float)
                risk = weather_risk_map.get(col, 0.5)
                result[col] = result[col] * risk
                numeric_risk_cols.append(col)
        if numeric_risk_cols:
            result["weather_risk_score"] = result[numeric_risk_cols].sum(axis=1, numeric_only=True).fillna(0.0)

    return result


def _categorize_weather(condition: str) -> str:
    """Map raw weather condition text to category."""
    if pd.isna(condition):
        return "clear"
    condition_lower = str(condition).lower()
    for category, keywords in WEATHER_CATEGORIES.items():
        if any(kw in condition_lower for kw in keywords):
            return category
    return "clear"


def _compute_pressure_trend(pressure: np.ndarray) -> np.ndarray:
    """Compute pressure trend indicator."""
    if len(pressure) < 3:
        return np.full_like(pressure, 0, dtype=float)
    diff = np.diff(pressure)
    trend = np.zeros(len(pressure), dtype=float)
    trend[1:] = diff
    trend[0] = diff[0] if len(diff) > 0 else 0
    return np.where(trend > 0.5, 1.0, np.where(trend < -0.5, -1.0, 0.0))


def preprocess_weather_pipeline(
    df: pd.DataFrame,
    target_timestamps: Optional[pd.DatetimeIndex] = None,
    impute: bool = True,
    align: bool = True,
    encode: bool = True,
    categorize: bool = True,
) -> pd.DataFrame:
    """
    Complete weather data preprocessing pipeline.

    Args:
        df: Raw weather DataFrame
        target_timestamps: Optional target timestamps for alignment
        impute: Impute missing values
        align: Align to target timestamps
        encode: Encode categorical features
        categorize: Categorize weather conditions

    Returns:
        Preprocessed weather DataFrame
    """
    result = df.copy()

    if "timestamp" in result.columns:
        result = result.set_index(pd.to_datetime(result["timestamp"]))

    if impute:
        result = impute_missing_weather(result)

    if target_timestamps is not None and align:
        result = align_weather_temporal(result, target_timestamps)
        logger.info(f"Aligned weather to {len(target_timestamps)} target timestamps")

    if categorize and "weather_condition" in result.columns:
        result["weather_category"] = result["weather_condition"].apply(_categorize_weather)
        logger.info(f"Weather categories: {result['weather_category'].value_counts().to_dict()}")

    if encode:
        result = encode_weather_features(result)

    return result


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    dates = pd.date_range("2026-01-01", periods=100, freq="1h")
    df = pd.DataFrame({
        "timestamp": dates,
        "temperature_c": np.random.normal(20, 5, 100),
        "humidity_pct": np.random.uniform(30, 90, 100),
        "weather_condition": np.random.choice(
            ["clear", "rain", "cloudy", "fog"], 100
        ),
    })
    df.iloc[10:15, 1] = np.nan

    result = preprocess_weather_pipeline(df)
    logger.info(f"Shape: {result.shape}")
    logger.info(f"Weather risk cols: {[c for c in result.columns if 'weather_' in c]}")
