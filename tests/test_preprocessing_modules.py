"""
Test all new preprocessing modules import and basic functionality.
"""

import sys
import numpy as np

sys.path.insert(0, '.')


def test_imports():
    from dataset.preprocessing import (
        image_enhancement,
        sensor_preprocessing,
        telemetry_preprocessing,
        text_preprocessing,
        weather_preprocessing,
        feature_engineering,
        pipeline,
        deblurring,
        heic_converter,
    )
    print("ALL modules imported successfully")


def test_image_enhancement():
    import cv2
    from dataset.preprocessing.image_enhancement import (
        bilateral_filter,
        correct_illumination,
        global_histogram_equalization,
        segment_tire_region,
        morphological_operations,
        apply_all_image_enhancements,
    )

    test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    bf = bilateral_filter(test_img)
    assert bf.shape == test_img.shape
    print(f"bilateral_filter: {bf.shape}")

    ic = correct_illumination(test_img)
    assert ic.shape == test_img.shape
    print(f"correct_illumination: {ic.shape}")

    he = global_histogram_equalization(test_img)
    assert he.shape == test_img.shape
    print(f"global_histogram_equalization: {he.shape}")

    mask, seg = segment_tire_region(test_img)
    assert mask.shape == test_img.shape[:2]
    assert seg.shape == test_img.shape
    print(f"segment_tire_region: mask={mask.shape}, seg={seg.shape}")

    morph = morphological_operations(test_img, operation="close")
    assert len(morph.shape) == 2
    print(f"morphological_operations: {morph.shape}")

    enhanced = apply_all_image_enhancements(test_img)
    assert enhanced.shape == test_img.shape
    print(f"apply_all_image_enhancements: {enhanced.shape}")


def test_sensor_preprocessing():
    from dataset.preprocessing.sensor_preprocessing import (
        detect_outliers_iqr,
        detect_outliers_zscore,
        smooth_signal,
        interpolate_missing,
        preprocess_sensor_pipeline,
    )

    t = np.linspace(0, 10, 200)
    signal = 30 + 5 * np.sin(2 * np.pi * 0.5 * t) + np.random.normal(0, 2, 200)
    signal[20:25] = np.nan
    signal[100:103] = 100.0

    mask, outliers = detect_outliers_iqr(signal[~np.isnan(signal)])
    print(f"detect_outliers_iqr: {len(outliers)} outliers")

    mask2, outliers2 = detect_outliers_zscore(signal[~np.isnan(signal)])
    print(f"detect_outliers_zscore: {len(outliers2)} outliers")

    smoothed = smooth_signal(np.nan_to_num(signal), method="savgol")
    assert len(smoothed) == len(signal)
    print(f"smooth_signal: {smoothed.shape}")

    interpolated = interpolate_missing(signal)
    assert not np.any(np.isnan(interpolated))
    print(f"interpolate_missing: all NaN filled")

    result = preprocess_sensor_pipeline(signal)
    assert len(result["cleaned"]) == len(signal)
    print(f"preprocess_sensor_pipeline: cleaned={len(result['cleaned'])}, outliers={len(result['outlier_indices'])}")


def test_text_preprocessing():
    from dataset.preprocessing.text_preprocessing import (
        preprocess_text_pipeline,
        tokenize_text,
        remove_stopwords,
        spell_correct_text,
    )

    sample = "The tire tread depth is 3.2mm and the sidewall shows sign of cracking."

    result = preprocess_text_pipeline(sample)
    assert "cleaned" in result
    assert "tokens" in result
    print(f"preprocess_text_pipeline: tokens={result['tokens'][:5]}")

    tokens = tokenize_text(sample)
    assert len(tokens) > 0
    print(f"tokenize_text: {len(tokens)} tokens")

    filtered = remove_stopwords(tokens)
    assert len(filtered) < len(tokens)
    print(f"remove_stopwords: {len(tokens)} -> {len(filtered)} tokens")

    corrected = spell_correct_text("tyre tread deph is low")
    print(f"spell_correct_text: {corrected}")


def test_weather_preprocessing():
    import pandas as pd
    from dataset.preprocessing.weather_preprocessing import (
        preprocess_weather_pipeline,
    )

    dates = pd.date_range("2026-01-01", periods=50, freq="1h")
    df = pd.DataFrame({
        "timestamp": dates,
        "temperature_c": np.random.normal(20, 5, 50),
        "humidity_pct": np.random.uniform(30, 90, 50),
        "weather_condition": np.random.choice(["clear", "rain", "cloudy", "fog"], 50),
    })
    df.iloc[5:8, 1] = np.nan

    result = preprocess_weather_pipeline(df)
    assert result is not None
    weather_cols = [c for c in result.columns if "weather_" in c]
    print(f"preprocess_weather_pipeline: {result.shape}, weather cols: {weather_cols}")


def test_feature_engineering():
    import pandas as pd
    from dataset.preprocessing.feature_engineering import (
        feature_engineering_pipeline,
        select_features_variance,
        correlation_analysis,
    )

    df = pd.DataFrame({
        "tread_1": np.random.uniform(1, 8, 100),
        "tread_2": np.random.uniform(1, 8, 100),
        "tread_3": np.random.uniform(1, 8, 100),
        "tread_4": np.random.uniform(1, 8, 100),
        "brand": np.random.choice(["A", "B", "C"], 100),
        "wear_pattern": np.random.choice(["even", "center", "edge"], 100),
        "constant_col": np.ones(100),
    })

    sel = select_features_variance(df)
    print(f"select_features_variance: {sel.shape}")

    corr = correlation_analysis(df, target_col="wear_pattern")
    print(f"correlation_analysis: {len(corr['high_corr_pairs'])} high-correlation pairs")

    result = feature_engineering_pipeline(df, target_col="wear_pattern")
    assert "features" in result
    print(f"feature_engineering_pipeline: features={result['features'].shape}")


def test_telemetry_preprocessing():
    import pandas as pd
    from dataset.preprocessing.telemetry_preprocessing import (
        clean_gps_coordinates,
        smooth_speed,
        haversine_distance,
    )

    df = pd.DataFrame({
        "latitude": 40.7128 + np.cumsum(np.random.normal(0, 0.001, 100)),
        "longitude": -74.0060 + np.cumsum(np.random.normal(0, 0.001, 100)),
        "speed": np.abs(np.random.normal(50, 15, 100)),
    })
    df.iloc[5, 0] = 200.0
    df.iloc[5, 1] = 300.0

    cleaned = clean_gps_coordinates(df)
    assert len(cleaned) <= len(df)
    print(f"clean_gps_coordinates: {len(df)} -> {len(cleaned)}")

    smoothed = smooth_speed(cleaned["speed"].values, method="moving_avg")
    assert len(smoothed) == len(cleaned)
    print(f"smooth_speed: {smoothed.shape}")

    dist = haversine_distance(40.7128, -74.0060, 40.7580, -73.9855)
    print(f"haversine_distance: {dist:.2f} km")


def test_pipeline():
    from dataset.preprocessing.pipeline import _default_config, run_full_preprocessing

    config = _default_config()
    # Disable all modules for dry-run test
    for key in config:
        config[key]["enabled"] = False
    config["feature_engineering"]["enabled"] = True

    print(f"Pipeline config created: {len(config)} modules")
    # Verify new deblur config keys exist
    assert "blur_threshold" in config["image"]
    assert "deblur_attempts" in config["image"]
    assert "convert_heic" in config["image"]
    print(f"Image config keys: {list(config['image'].keys())}")


def test_deblurring():
    import cv2
    import numpy as np
    from dataset.preprocessing.deblurring import (
        wiener_deconvolution,
        regularized_deconvolution,
        laplacian_sharpen,
        texture_enhancement,
        multi_scale_sharpen,
        add_gaussian_sharpen,
        try_deblur,
        deblur_pipeline,
    )

    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    # All methods should return same-shaped images
    result = wiener_deconvolution(img)
    assert result.shape == img.shape, f"wiener: {result.shape}"
    print(f"wiener_deconvolution: {result.shape}")

    result = regularized_deconvolution(img)
    assert result.shape == img.shape, f"regularized: {result.shape}"
    print(f"regularized_deconvolution: {result.shape}")

    result = laplacian_sharpen(img, strength=2.0)
    assert result.shape == img.shape, f"laplacian: {result.shape}"
    print(f"laplacian_sharpen: {result.shape}")

    result = texture_enhancement(img)
    assert result.shape == img.shape, f"texture: {result.shape}"
    print(f"texture_enhancement: {result.shape}")

    result = multi_scale_sharpen(img)
    assert result.shape == img.shape, f"multi_scale: {result.shape}"
    print(f"multi_scale_sharpen: {result.shape}")

    result = add_gaussian_sharpen(img)
    assert result.shape == img.shape, f"gaussian: {result.shape}"
    print(f"add_gaussian_sharpen: {result.shape}")

    # Simulate a blurry image to test try_deblur / deblur_pipeline
    blurred = cv2.GaussianBlur(img, (15, 15), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    result_img, new_score, method = try_deblur(blurred, blur_score)
    assert result_img.shape == blurred.shape
    print(f"try_deblur: {blur_score:.1f} -> {new_score:.1f} ({method})")

    result_img, final_score, recovered = deblur_pipeline(blurred, blur_score)
    assert result_img.shape == blurred.shape
    print(f"deblur_pipeline: {blur_score:.1f} -> {final_score:.1f}, recovered={recovered}")


def test_heic_converter():
    import tempfile
    from pathlib import Path
    from dataset.preprocessing.heic_converter import (
        convert_heic_to_jpeg,
        batch_convert_heic_to_jpeg,
        heic_files_in_dir,
        ensure_images_are_jpeg,
    )

    # Test heic_files_in_dir with empty directory
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        files = heic_files_in_dir(tmp_dir)
        assert files == []
        print(f"heic_files_in_dir (empty): {len(files)} files")

    # Test conversion fails gracefully with non-existent file
    result = convert_heic_to_jpeg(Path("/nonexistent/file.heic"))
    assert result is None
    print("convert_heic_to_jpeg (missing file): returns None")

    # Test batch convert with no HEIC files
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        # Create a .jpg file (should be ignored)
        jpg_file = tmp_dir / "test.jpg"
        jpg_file.write_text("not an image")
        converted = batch_convert_heic_to_jpeg(tmp_dir, recursive=False)
        assert converted == []
        print(f"batch_convert_heic_to_jpeg (no heic): {len(converted)} files")

    # Test ensure_images_are_jpeg with no HEIC files
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        converted, skipped = ensure_images_are_jpeg(tmp_dir)
        assert converted == []
        assert skipped == []
        print("ensure_images_are_jpeg (empty dir): no files")


if __name__ == "__main__":
    test_imports()
    print()
    test_image_enhancement()
    print()
    test_sensor_preprocessing()
    print()
    test_text_preprocessing()
    print()
    test_weather_preprocessing()
    print()
    test_feature_engineering()
    print()
    test_telemetry_preprocessing()
    print()
    test_pipeline()
    print()
    test_deblurring()
    print()
    test_heic_converter()
    print("\nALL TESTS PASSED")
