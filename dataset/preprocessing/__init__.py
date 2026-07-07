"""
Smart Tire Analyzer — Comprehensive Preprocessing Package

Modules:
    clean_dataset         — Dataset cleaning (tread imputation, brand/size normalization, dedup)
    split_dataset         — Stratified train/val/test splitting
    validate_images       — Image corruption, resolution, and quality checks
    image_enhancement     — Bilateral filtering, illumination correction, histogram equalization,
                            segmentation, morphological operations
    sensor_preprocessing  — Outlier detection, signal smoothing, interpolation, FFT, wavelet,
                            time sync, resampling, drift correction, calibration, windowing
    telemetry_preprocessing — GPS cleaning, speed smoothing, route segmentation,
                              coordinate normalization, trip extraction
    text_preprocessing     — Tokenization, stopword removal, lemmatization, stemming,
                             NER preprocessing, spell correction
    weather_preprocessing  — Missing value imputation, temporal alignment, feature encoding,
                             weather categorization
    feature_engineering    — Feature selection, PCA, correlation analysis, label/one-hot encoding,
                             SMOTE, data splitting, sequence generation
    pipeline              — Unified orchestration pipeline combining all modules
"""

from dataset.preprocessing.clean_dataset import clean_dataset, fix_tread_zeros, normalize_brand_names, remove_duplicates  # noqa: F401
from dataset.preprocessing.split_dataset import stratified_split, split_and_save  # noqa: F401
from dataset.preprocessing.validate_images import validate_all, validate_image  # noqa: F401
from dataset.preprocessing.image_enhancement import (  # noqa: F401
    bilateral_filter, correct_illumination, global_histogram_equalization,
    segment_tire_region, morphological_operations, apply_all_image_enhancements,
)
from dataset.preprocessing.sensor_preprocessing import (  # noqa: F401
    detect_outliers_iqr, detect_outliers_zscore, smooth_signal,
    interpolate_missing, synchronize_timestamps, resample_signal,
    correct_drift, calibrate_sensor, convert_unit, apply_windowing,
    segment_signal, apply_fft, apply_wavelet, preprocess_sensor_pipeline,
)
from dataset.preprocessing.telemetry_preprocessing import (  # noqa: F401
    clean_gps_coordinates, smooth_speed, align_timestamps,
    normalize_coordinates, segment_routes, aggregate_telemetry,
    extract_trips_from_gps, haversine_distance,
)
from dataset.preprocessing.text_preprocessing import (  # noqa: F401
    clean_text, lowercase_text, remove_stopwords, tokenize_text,
    lemmatize_text, stem_tokens, preprocess_for_ner, spell_correct_text,
    preprocess_text_pipeline,
)
from dataset.preprocessing.weather_preprocessing import (  # noqa: F401
    impute_missing_weather, align_weather_temporal, encode_weather_features,
    preprocess_weather_pipeline,
)
from dataset.preprocessing.feature_engineering import (  # noqa: F401
    select_features_variance, select_features_mutual_info, apply_pca,
    correlation_analysis, label_encode, one_hot_encode, apply_smote,
    split_data, build_tread_sequences, feature_engineering_pipeline,
)
from dataset.preprocessing.pipeline import run_full_preprocessing  # noqa: F401
from dataset.preprocessing.deblurring import (  # noqa: F401
    wiener_deconvolution, regularized_deconvolution, laplacian_sharpen,
    texture_enhancement, multi_scale_sharpen, add_gaussian_sharpen,
    try_deblur, deblur_pipeline,
)
from dataset.preprocessing.heic_converter import (  # noqa: F401
    convert_heic_to_jpeg, batch_convert_heic_to_jpeg,
    heic_files_in_dir, ensure_images_are_jpeg,
)

__all__ = [
    # Cleaning
    "clean_dataset", "fix_tread_zeros", "normalize_brand_names", "remove_duplicates",
    # Splitting
    "stratified_split", "split_and_save",
    # Validation
    "validate_all", "validate_image",
    # Image Enhancement
    "bilateral_filter", "correct_illumination", "global_histogram_equalization",
    "segment_tire_region", "morphological_operations", "apply_all_image_enhancements",
    # Sensor
    "detect_outliers_iqr", "detect_outliers_zscore", "smooth_signal",
    "interpolate_missing", "synchronize_timestamps", "resample_signal",
    "correct_drift", "calibrate_sensor", "convert_unit", "apply_windowing",
    "segment_signal", "apply_fft", "apply_wavelet", "preprocess_sensor_pipeline",
    # Telemetry
    "clean_gps_coordinates", "smooth_speed", "align_timestamps",
    "normalize_coordinates", "segment_routes", "aggregate_telemetry",
    "extract_trips_from_gps", "haversine_distance",
    # Text
    "clean_text", "lowercase_text", "remove_stopwords", "tokenize_text",
    "lemmatize_text", "stem_tokens", "preprocess_for_ner", "spell_correct_text",
    "preprocess_text_pipeline",
    # Weather
    "impute_missing_weather", "align_weather_temporal", "encode_weather_features",
    "preprocess_weather_pipeline",
    # Feature Engineering
    "select_features_variance", "select_features_mutual_info", "apply_pca",
    "correlation_analysis", "label_encode", "one_hot_encode", "apply_smote",
    "split_data", "build_tread_sequences", "feature_engineering_pipeline",
    # Deblurring
    "wiener_deconvolution", "regularized_deconvolution", "laplacian_sharpen",
    "texture_enhancement", "multi_scale_sharpen", "add_gaussian_sharpen",
    "try_deblur", "deblur_pipeline",
    # HEIC Converter
    "convert_heic_to_jpeg", "batch_convert_heic_to_jpeg",
    "heic_files_in_dir", "ensure_images_are_jpeg",
    # Pipeline
    "run_full_preprocessing",
]
