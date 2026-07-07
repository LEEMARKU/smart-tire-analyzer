# Dataset

## Structure

```
dataset/
  raw/
    tread_images/           # Tire photos organized by condition
      safe/                 # Safe tires (> 3mm tread depth)
      moderate/             # Moderately worn (1.6-3mm)
      replace/              # Below legal limit (< 1.6mm)
    sidewall_images/        # Optional sidewall brand photos
    spreadsheet/            # Label spreadsheets (XLSX/CSV)
    sensor_data.csv         # Optional sensor measurements
    telemetry.csv           # Optional vehicle telemetry
    weather_data.csv        # Optional weather context
  processed/                # Pipeline output directory
    enhanced_images/        # Preprocessed images
    enhanced_previews/      # Preview JPEGs for inspection
  splits/                   # Train/val/test splits
    train/
    validation/
    test/
  continuous_learning/      # User feedback collection
    new_images/
    user_feedback/
    retrain_dataset/
  preprocessing/            # Preprocessing pipeline modules
    pipeline.py             # Unified orchestrator
    image_enhancement.py    # Rotation, shadow, bilateral, etc.
    deblurring.py           # Multi-method deblurring
    sensor_preprocessing.py
    telemetry_preprocessing.py
    text_preprocessing.py
    weather_preprocessing.py
    feature_engineering.py
    validate_images.py
```

## Label Format

Each image in `tread_images/{condition}/` has a corresponding label. The
dataset spreadsheet (`raw/spreadsheet/dataset.xlsx`) should contain:

| Column | Description |
|--------|-------------|
| image_id | Unique image identifier |
| image_path | Relative path to image file |
| tread_1..4 | Tread depth at 4 positions (mm) |
| wear_pattern | One of: uniform, center, edge, patchy, one_side, cupping |
| condition | One of: safe, moderate, replace |
| health_norm | Normalized health score (0-1) |

## Preprocessing Pipeline

Run the full pipeline with:
```bash
python scripts/prepare_dataset.py
```

Or use individual modules via `dataset/preprocessing/pipeline.py`:
```bash
python -m dataset.preprocessing.pipeline --module image
```
