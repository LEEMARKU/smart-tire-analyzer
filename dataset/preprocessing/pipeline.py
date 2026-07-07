"""
Unified Preprocessing Pipeline — Orchestrates ALL preprocessing steps end-to-end.
Combines image, sensor, telemetry, text, weather, and feature engineering
preprocessing into a single configurable pipeline.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
from datetime import datetime

logger = logging.getLogger(__name__)

PIPELINE_VERSION = "2.0.0"


def run_full_preprocessing(
    config: Optional[Dict[str, Any]] = None,
    data_dir: Union[str, Path] = "dataset",
    output_dir: Union[str, Path] = "dataset/processed",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run the complete preprocessing pipeline across all data types.

    This orchestrates ALL preprocessing categories:
      1. Image preprocessing (existing + enhancement)
      2. Sensor data preprocessing
      3. Vehicle telemetry preprocessing
      4. Text preprocessing
      5. Weather data preprocessing
      6. Feature engineering

    Args:
        config: Pipeline configuration dict with per-module flags
        data_dir: Root data directory
        output_dir: Output directory for processed data
        verbose: Enable detailed logging

    Returns:
        Dict with keys: 'status', 'report', 'output_paths', 'statistics'
    """
    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if config is None:
        config = _default_config()

    report = {
        "pipeline_version": PIPELINE_VERSION,
        "started_at": datetime.now().isoformat(),
        "modules_run": [],
        "errors": [],
        "warnings": [],
    }

    results: Dict[str, Any] = {}

    # ── 0. HEIC to JPEG Conversion (pre-step) ──
    if config.get("image", {}).get("convert_heic", True):
        try:
            img_dir = data_dir / "raw" / "tread_images"
            if img_dir.exists():
                from dataset.preprocessing.heic_converter import ensure_images_are_jpeg
                converted, skipped = ensure_images_are_jpeg(
                    img_dir,
                    recursive=True,
                    overwrite=config.get("image", {}).get("overwrite_heic", False),
                    remove_original=config.get("image", {}).get("remove_heic_original", False),
                )
                if converted:
                    logger.info(f"Converted {len(converted)} HEIC images to JPEG")
                    report["modules_run"].append("heic_conversion")
                if skipped:
                    report["warnings"].append(f"{len(skipped)} HEIC files could not be converted")
        except Exception as e:
            report["warnings"].append(f"HEIC conversion error: {e}")

    # ── 1. Image Preprocessing (with deblurring) ──
    if config.get("image", {}).get("enabled", False):
        try:
            logger.info("=" * 60)
            logger.info("PHASE 1: Image Preprocessing (with Deblurring)")
            logger.info("=" * 60)
            from dataset.preprocessing.image_enhancement import apply_all_image_enhancements
            from dataset.preprocessing.deblurring import deblur_pipeline
            img_dir = data_dir / "raw" / "tread_images"
            if img_dir.exists():
                import cv2
                enhanced_dir = output_dir / "enhanced_images"
                enhanced_dir.mkdir(parents=True, exist_ok=True)
                img_count = 0
                deblur_count = 0
                blur_threshold = config["image"].get("blur_threshold", 100.0)
                accepted_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
                for img_path in img_dir.rglob("*"):
                    if img_path.suffix.lower() not in accepted_exts:
                        continue
                    try:
                        img = cv2.imread(str(img_path))
                        if img is None:
                            continue

                        # ── Blur detection + deblurring ──
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                        if blur_score < blur_threshold:
                            deblurred, new_score, recovered = deblur_pipeline(
                                img, blur_score, threshold=blur_threshold,
                                max_attempts=config["image"].get("deblur_attempts", 3),
                            )
                            if recovered:
                                deblur_count += 1
                                img = deblurred
                            else:
                                report["warnings"].append(
                                    f"Could not deblur {img_path.name} "
                                    f"(score: {blur_score:.1f} -> {new_score:.1f})"
                                )

                        enhanced = apply_all_image_enhancements(
                            img,
                            do_bilateral=config["image"].get("bilateral", True),
                            do_rotation_correction=config["image"].get("rotation_correction", True),
                            do_shadow_removal=config["image"].get("shadow_removal", True),
                            do_illumination=config["image"].get("illumination", True),
                            do_hist_equal=config["image"].get("histogram_equalization", False),
                            do_segmentation=config["image"].get("segmentation", False),
                            do_morphology=config["image"].get("morphology", None),
                        )
                        out_path = enhanced_dir / img_path.relative_to(img_dir).parent / img_path.name
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(out_path), enhanced)
                        img_count += 1
                    except Exception as e:
                        report["warnings"].append(f"Failed to process {img_path.name}: {e}")
                results["enhanced_images"] = img_count
                results["deblurred_images"] = deblur_count
                logger.info(f"Processed {img_count} images (deblurred {deblur_count})")
                report["modules_run"].append("image_enhancement")
        except Exception as e:
            report["errors"].append(f"Image preprocessing failed: {e}")
            logger.error(f"Image preprocessing error: {e}")

    # ── 2. Sensor Data Preprocessing ──
    if config.get("sensor", {}).get("enabled", False):
        try:
            logger.info("=" * 60)
            logger.info("PHASE 2: Sensor Data Preprocessing")
            logger.info("=" * 60)
            from dataset.preprocessing.sensor_preprocessing import preprocess_sensor_pipeline
            sensor_config = config["sensor"]
            sensor_data_path = data_dir / "raw" / "sensor_data.csv"
            if sensor_data_path.exists():
                sensor_df = pd.read_csv(sensor_data_path)
                if sensor_config.get("preprocess_all", True):
                    for col in sensor_df.select_dtypes(include=[np.number]).columns:
                        if col in ("timestamp", "id"):
                            continue
                        values = sensor_df[col].values.astype(float)
                        sensor_result = preprocess_sensor_pipeline(
                            values,
                            sampling_rate=sensor_config.get("sampling_rate", 100.0),
                            detect_outliers=sensor_config.get("detect_outliers", True),
                            outlier_method=sensor_config.get("outlier_method", "iqr"),
                            smooth=sensor_config.get("smooth", True),
                            smooth_method=sensor_config.get("smooth_method", "savgol"),
                            interpolate=sensor_config.get("interpolate", True),
                            interpolate_method=sensor_config.get("interpolate_method", "linear"),
                            correct_drift_flag=sensor_config.get("drift_correction", False),
                            apply_fft_flag=sensor_config.get("fft", False),
                            wavelet_denoise=sensor_config.get("wavelet", False),
                        )
                        sensor_df[col] = sensor_result["cleaned"]
                sensor_df.to_csv(output_dir / "sensor_data_cleaned.csv", index=False)
                results["sensor_rows"] = len(sensor_df)
                logger.info(f"Preprocessed {len(sensor_df)} sensor records")
                report["modules_run"].append("sensor_preprocessing")
            else:
                report["warnings"].append(f"Sensor data not found: {sensor_data_path}")
        except Exception as e:
            report["errors"].append(f"Sensor preprocessing failed: {e}")

    # ── 3. Telemetry Preprocessing ──
    if config.get("telemetry", {}).get("enabled", False):
        try:
            logger.info("=" * 60)
            logger.info("PHASE 3: Telemetry Preprocessing")
            logger.info("=" * 60)
            from dataset.preprocessing.telemetry_preprocessing import (
                clean_gps_coordinates, smooth_speed, align_timestamps,
                normalize_coordinates, extract_trips_from_gps,
            )
            telemetry_config = config["telemetry"]
            telemetry_path = data_dir / "raw" / "telemetry.csv"
            if telemetry_path.exists():
                telemetry_df = pd.read_csv(telemetry_path)
                if "latitude" in telemetry_df.columns and "longitude" in telemetry_df.columns:
                    telemetry_df = clean_gps_coordinates(
                        telemetry_df,
                        max_speed_kmh=telemetry_config.get("max_speed", 300.0),
                    )
                if telemetry_config.get("align_timestamps", False) and "timestamp" in telemetry_df.columns:
                    telemetry_df = align_timestamps(
                        telemetry_df,
                        target_fs=telemetry_config.get("target_fs", 1.0),
                    )
                telemetry_df.to_csv(output_dir / "telemetry_cleaned.csv", index=False)
                results["telemetry_rows"] = len(telemetry_df)
                logger.info(f"Preprocessed {len(telemetry_df)} telemetry records")
                report["modules_run"].append("telemetry_preprocessing")
        except Exception as e:
            report["errors"].append(f"Telemetry preprocessing failed: {e}")

    # ── 4. Text Preprocessing ──
    if config.get("text", {}).get("enabled", False):
        try:
            logger.info("=" * 60)
            logger.info("PHASE 4: Text Preprocessing")
            logger.info("=" * 60)
            from dataset.preprocessing.text_preprocessing import preprocess_text_pipeline
            text_config = config["text"]
            labels_path = data_dir / "processed" / "labels.csv"
            if labels_path.exists():
                labels_df = pd.read_csv(labels_path)
                text_cols = labels_df.select_dtypes(include=["object"]).columns.tolist()
                for col in text_cols:
                    if col in ("image_id", "tire_size"):
                        continue
                    text_results = labels_df[col].apply(
                        lambda x: preprocess_text_pipeline(
                            str(x) if pd.notna(x) else "",
                            do_clean=text_config.get("clean", True),
                            do_lowercase=text_config.get("lowercase", True),
                            do_tokenize=False,
                            do_remove_stopwords=False,
                            do_spell_correct=text_config.get("spell_correct", False),
                        )
                    )
                    if text_config.get("spell_correct", False):
                        labels_df[col] = text_results.apply(
                            lambda r: r.get("spell_corrected", r["cleaned"])
                        )
                    else:
                        labels_df[col] = text_results.apply(lambda r: r["cleaned"])
                labels_df.to_csv(output_dir / "labels_text_cleaned.csv", index=False)
                results["text_rows"] = len(labels_df)
                logger.info(f"Preprocessed text in {len(text_cols)} columns")
                report["modules_run"].append("text_preprocessing")
        except Exception as e:
            report["errors"].append(f"Text preprocessing failed: {e}")

    # ── 5. Weather Preprocessing ──
    if config.get("weather", {}).get("enabled", False):
        try:
            logger.info("=" * 60)
            logger.info("PHASE 5: Weather Preprocessing")
            logger.info("=" * 60)
            from dataset.preprocessing.weather_preprocessing import preprocess_weather_pipeline
            weather_config = config["weather"]
            weather_path = data_dir / "raw" / "weather_data.csv"
            if weather_path.exists():
                weather_df = pd.read_csv(weather_path)
                weather_result = preprocess_weather_pipeline(
                    weather_df,
                    impute=weather_config.get("impute", True),
                    align=False,
                    encode=weather_config.get("encode", True),
                    categorize=weather_config.get("categorize", True),
                )
                weather_result.to_csv(output_dir / "weather_processed.csv", index=False)
                results["weather_rows"] = len(weather_result)
                logger.info(f"Preprocessed {len(weather_result)} weather records")
                report["modules_run"].append("weather_preprocessing")
        except Exception as e:
            report["errors"].append(f"Weather preprocessing failed: {e}")

    # ── 6. Feature Engineering ──
    if config.get("feature_engineering", {}).get("enabled", False):
        try:
            logger.info("=" * 60)
            logger.info("PHASE 6: Feature Engineering")
            logger.info("=" * 60)
            from dataset.preprocessing.feature_engineering import feature_engineering_pipeline
            fe_config = config["feature_engineering"]
            merged_path = output_dir / "labels_text_cleaned.csv"
            if not merged_path.exists():
                merged_path = data_dir / "processed" / "cleaned_dataset.csv"

            if merged_path.exists():
                fe_df = pd.read_csv(merged_path)
                fe_result = feature_engineering_pipeline(
                    fe_df,
                    target_col=fe_config.get("target_col"),
                    do_variance_selection=fe_config.get("variance_selection", True),
                    do_mutual_info=fe_config.get("mutual_info", False),
                    do_pca=fe_config.get("pca", False),
                    do_correlation=fe_config.get("correlation", False),
                    do_label_encode=fe_config.get("label_encode", True),
                    do_one_hot=fe_config.get("one_hot", False),
                    do_smote=fe_config.get("smote", False),
                    do_split=fe_config.get("split", False),
                )
                fe_df_result = fe_result.get("features", fe_df)
                if isinstance(fe_df_result, pd.DataFrame):
                    fe_df_result.to_csv(output_dir / "features_engineered.csv", index=False)
                results["feature_cols"] = (
                    fe_df_result.shape[1] if isinstance(fe_df_result, pd.DataFrame) else 0
                )
                results["feature_engineering"] = {
                    k: v for k, v in fe_result.items()
                    if k not in ("features", "correlation")
                }
                logger.info(f"Feature engineering complete")
                report["modules_run"].append("feature_engineering")
        except Exception as e:
            report["errors"].append(f"Feature engineering failed: {e}")

    report["completed_at"] = datetime.now().isoformat()
    report["results"] = results
    report["status"] = "completed_with_errors" if report["errors"] else "completed"

    _print_summary(report, output_dir)
    return report


def _default_config() -> Dict[str, Any]:
    """Return default pipeline configuration with all steps enabled."""
    return {
        "image": {
            "enabled": True,
            "convert_heic": True,
            "overwrite_heic": False,
            "remove_heic_original": False,
            "blur_threshold": 100.0,
            "deblur_attempts": 3,
            "bilateral": True,
            "rotation_correction": True,
            "shadow_removal": True,
            "illumination": True,
            "histogram_equalization": False,
            "segmentation": False,
            "morphology": None,
        },
        "sensor": {
            "enabled": True,
            "sampling_rate": 100.0,
            "detect_outliers": True,
            "outlier_method": "iqr",
            "smooth": True,
            "smooth_method": "savgol",
            "interpolate": True,
            "interpolate_method": "linear",
            "drift_correction": False,
            "fft": False,
            "wavelet": False,
        },
        "telemetry": {
            "enabled": True,
            "max_speed": 300.0,
            "align_timestamps": False,
            "target_fs": 1.0,
        },
        "text": {
            "enabled": True,
            "clean": True,
            "lowercase": True,
            "spell_correct": True,
        },
        "weather": {
            "enabled": True,
            "impute": True,
            "encode": True,
            "categorize": True,
        },
        "feature_engineering": {
            "enabled": True,
            "variance_selection": True,
            "mutual_info": False,
            "pca": False,
            "correlation": True,
            "label_encode": True,
            "one_hot": False,
            "smote": False,
            "split": False,
        },
    }


def _print_summary(report: Dict[str, Any], output_dir: Path) -> None:
    """Print pipeline execution summary."""
    print("\n" + "=" * 72)
    print(f"UNIFIED PREPROCESSING PIPELINE — {PIPELINE_VERSION}")
    print("=" * 72)
    print(f"Status: {report['status']}")
    print(f"Modules run ({len(report['modules_run'])}): {', '.join(report['modules_run'])}")
    print(f"Output directory: {output_dir}")

    if report["results"]:
        print("\nResults:")
        for key, value in report["results"].items():
            if isinstance(value, dict) or key == "feature_engineering":
                continue
            print(f"  {key}: {value}")

    print(f"\nErrors: {len(report['errors'])}")
    for err in report["errors"][:5]:
        print(f"  ! {err}")

    print(f"Warnings: {len(report['warnings'])}")
    for warn in report["warnings"][:5]:
        print(f"  ? {warn}")

    print("=" * 72 + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run full preprocessing pipeline")
    parser.add_argument("--data-dir", default="dataset", help="Data directory")
    parser.add_argument("--output-dir", default="dataset/processed", help="Output directory")
    parser.add_argument("--module", nargs="*", help="Specific modules to run (image, sensor, telemetry, text, weather, feature)")
    args = parser.parse_args()

    config = _default_config()
    if args.module:
        for key in config:
            config[key]["enabled"] = key in args.module

    run_full_preprocessing(config, args.data_dir, args.output_dir)
