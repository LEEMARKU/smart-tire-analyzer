"""
Add Raw Tire Data — just provide tread depths + front tire image,
everything else is auto-calculated.

CSV mode:  python scripts/add_raw_data.py --csv data.csv --image-dir ./photos
Interactive: python scripts/add_raw_data.py

CSV format (only 5 required columns):
  image_id,tread_1,tread_2,tread_3,tread_4
"""

import argparse
import csv
import logging
import shutil
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("add_raw_data")

BASE_DIR = Path(__file__).resolve().parent.parent
IMAGES_DIR = BASE_DIR / "dataset" / "raw" / "tread_images"
PROCESSED_DIR = BASE_DIR / "dataset" / "processed"
LABELS_CSV = PROCESSED_DIR / "labels.csv"
CLEANED_CSV = PROCESSED_DIR / "cleaned_dataset.csv"
SPLITS_DIR = BASE_DIR / "dataset" / "splits"

TREAD_MAX = 12.0
HEALTH_MAX = 10.0
MAX_KM = 80000.0
MIN_LEGAL_MM = 1.6

WEAR_CLASS_MAP = {
    "center_wear": 0, "edge_wear": 1, "uneven_wear": 2,
    "even": 3, "one_sided_wear": 4, "critical_wear": 5,
}
CONDITION_MAP = {"safe": 0, "moderate": 1, "replace": 2}


def classify_condition(avg: float) -> tuple[str, int]:
    if avg >= 4.0:
        return "safe", 0
    if avg >= MIN_LEGAL_MM:
        return "moderate", 1
    return "replace", 2


def estimate_wear_pattern(treads: list[float]) -> tuple[str, int]:
    t1, t2, t3, t4 = treads
    avg = sum(treads) / 4
    max_diff = max(treads) - min(treads)
    center = (t2 + t3) / 2
    edge = (t1 + t4) / 2

    if avg < MIN_LEGAL_MM:
        label, cls = "critical_wear", 5
    elif max_diff < 0.5:
        label, cls = "even", 3
    elif max_diff > 2.0:
        label, cls = "uneven_wear", 2
    elif abs(center - edge) > 0.8:
        if center > edge:
            label, cls = "edge_wear", 1
        else:
            label, cls = "center_wear", 0
    elif max(abs(t1 - avg), abs(t4 - avg)) > 0.6:
        label, cls = "edge_wear", 1
    elif max(abs(t2 - avg), abs(t3 - avg)) > 0.6:
        label, cls = "center_wear", 0
    else:
        label, cls = "uneven_wear", 2
    return label, cls


def urgency_info(condition: str) -> tuple[str, int, str, bool]:
    table = {
        "safe":     ("Medium", 20, "LOW",     False),
        "moderate": ("Immediate", 30, "MODERATE", True),
        "replace":  ("Immediate", 85, "CRITICAL", True),
    }
    return table.get(condition, ("Unknown", 0, "UNKNOWN", False))


def build_row(image_id: str, treads: list[float]) -> dict[str, object]:
    t1, t2, t3, t4 = treads
    avg = sum(treads) / 4.0
    condition, cond_id = classify_condition(avg)
    wear_label, wear_class_id = estimate_wear_pattern(treads)
    health = round(avg / TREAD_MAX * HEALTH_MAX, 4)
    health_norm = round(health / HEALTH_MAX, 4)
    life_km = round(max(0, (avg - MIN_LEGAL_MM) / (TREAD_MAX - MIN_LEGAL_MM) * MAX_KM), 2)
    life_norm = round(life_km / MAX_KM, 6)
    life_pred = round(life_km / 1000, 4) if life_km > 0 else 0
    urgency, risk_score, risk_level, replace_rec = urgency_info(condition)
    tread_spread = round(max_diff := max(treads) - min(treads), 6)
    feature_ts = round(max_diff / avg if avg > 0 else 0, 6)
    center = (t2 + t3) / 2
    edge = (t1 + t4) / 2
    feature_ce_gap = round((center - edge) / avg if avg > 0 else 0, 6)

    return {
        "image_id": image_id,
        "tread_1": t1, "tread_2": t2, "tread_3": t3, "tread_4": t4,
        "tread_average": round(avg, 4),
        "condition": condition, "condition_id": cond_id,
        "wear_pattern": wear_label, "wear_class_6": wear_class_id,
        "health_score": health, "health_norm": health_norm,
        "remaining_life_km": life_km, "remaining_life_norm": life_norm,
        "remaining_life_pred": life_pred,
        "replacement_urgency": urgency, "risk_score": risk_score,
        "risk_level": risk_level, "replace_recommended": replace_rec,
        "image_path": str(IMAGES_DIR / image_id),
        "front_image_path": str(IMAGES_DIR / image_id),
        "has_image": True, "has_sidewall_image": False,
        "feature_tread_spread": feature_ts,
        "feature_center_edge_gap": feature_ce_gap,
    }


def validate_image(img_path: Path) -> tuple[bool, str]:
    try:
        import cv2
        import numpy as np
    except ImportError:
        logger.error("OpenCV required: pip install opencv-python")
        sys.exit(1)
    size_mb = img_path.stat().st_size / (1024 * 1024)
    if size_mb == 0:
        return False, "Empty file"
    if size_mb > 20.0:
        return False, f"File too large ({size_mb:.1f}MB > 20MB)"
    img = cv2.imread(str(img_path))
    if img is None:
        return False, "Corrupted or unreadable"
    h, w = img.shape[:2]
    if h < 224 or w < 224:
        return False, f"Too small ({w}x{h} < 224x224)"
    if len(img.shape) < 3 or img.shape[2] < 3:
        return False, "Not a color image"
    if float(img.std()) < 2.0:
        return False, "Image appears blank"
    return True, "OK"


def copy_images(image_paths: list[Path]) -> list[str]:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    copied = []
    for src in image_paths:
        dst = IMAGES_DIR / src.name
        if dst.exists():
            logger.warning(f"  Overwriting: {dst.name}")
        shutil.copy2(str(src), str(dst))
        copied.append(src.name)
        logger.info(f"  Copied: {src.name}")
    return copied


COLUMNS = [
    "image_id", "tread_1", "tread_2", "tread_3", "tread_4",
    "tread_average", "condition", "condition_id",
    "wear_pattern", "wear_class_6",
    "health_score", "health_norm",
    "remaining_life_km", "remaining_life_norm", "remaining_life_pred",
    "replacement_urgency", "risk_score", "risk_level", "replace_recommended",
    "image_path", "front_image_path",
    "has_image", "has_sidewall_image",
    "feature_tread_spread", "feature_center_edge_gap",
]


def save_labels(rows: list[dict]):
    LABELS_CSV.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if LABELS_CSV.exists():
        with open(LABELS_CSV, newline="", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))

    seen = {r["image_id"] for r in existing}
    deduped = []
    for row in rows:
        if row["image_id"] in seen:
            logger.warning(f"  Skipping duplicate: {row['image_id']}")
            continue
        seen.add(row["image_id"])
        deduped.append(row)

    all_rows = existing + deduped
    with open(LABELS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k, "") for k in COLUMNS})
    logger.info(f"Saved {len(all_rows)} rows to {LABELS_CSV}")


def run_pipeline():
    logger.info("Running dataset cleaning + split...")
    from dataset.preprocessing.clean_dataset import clean_dataset
    from dataset.preprocessing.split_dataset import split_and_save
    clean_dataset(str(LABELS_CSV), str(CLEANED_CSV))
    split_and_save()


def load_csv(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        logger.error("CSV is empty")
        sys.exit(1)
    missing = [c for c in ["image_id", "tread_1", "tread_2", "tread_3", "tread_4"]
               if c not in rows[0]]
    if missing:
        logger.error(f"Missing columns: {missing}")
        sys.exit(1)
    return rows


def interactive():
    print("\n=== Add Raw Tire Data ===\n")
    img_dir = input("Path to folder with tire images: ").strip()
    image_files = []
    if img_dir:
        p = Path(img_dir)
        if p.is_dir():
            exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            image_files = sorted(f for f in p.iterdir() if f.suffix.lower() in exts)
            logger.info(f"Found {len(image_files)} images")
        else:
            logger.warning("Directory not found")

    rows = []
    while True:
        print("-" * 50)
        name = input("Image filename (or 'done'): ").strip()
        if name.lower() == "done":
            break
        if not name:
            continue
        try:
            t1 = float(input("  Tread depth 1 (mm): "))
            t2 = float(input("  Tread depth 2 (mm): "))
            t3 = float(input("  Tread depth 3 (mm): "))
            t4 = float(input("  Tread depth 4 (mm): "))
        except ValueError:
            logger.error("Tread depths must be numbers")
            continue
        rows.append(build_row(name, [t1, t2, t3, t4]))

    if not rows:
        sys.exit(0)

    if image_files:
        for row in rows:
            match = next((f for f in image_files if f.name == row["image_id"]), None)
            if match is None:
                alt = next((f for f in image_files if f.stem == Path(row["image_id"]).stem), None)
                if alt:
                    row["image_id"] = alt.name

    return rows, image_files


def main():
    parser = argparse.ArgumentParser(description="Add raw tire data to the dataset")
    parser.add_argument("--csv", help="CSV with image_id,tread_1..4")
    parser.add_argument("--image-dir", help="Directory with tire images")
    parser.add_argument("--skip-split", action="store_true",
                        help="Skip regenerating train/val/test splits")
    args = parser.parse_args()

    if args.csv:
        raw_rows = load_csv(Path(args.csv))
    else:
        raw_rows, _ = interactive()

    image_files = []
    if args.image_dir:
        d = Path(args.image_dir)
        if d.is_dir():
            exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            image_files = sorted(f for f in d.iterdir() if f.suffix.lower() in exts)

    rows = []
    img_map = {f.name: f for f in image_files}
    for r in raw_rows:
        img_file = img_map.get(r["image_id"])
        if img_file is None:
            alt = next((f for f in image_files if f.stem == Path(r["image_id"]).stem), None)
            if alt:
                r["image_id"] = alt.name
                img_file = alt
        treads = [float(r[c]) for c in ["tread_1", "tread_2", "tread_3", "tread_4"]]
        rows.append(build_row(r["image_id"], treads))
        if img_file:
            image_files.append(img_file)  # already in list

    # Deduplicate image_files
    seen_paths = set()
    unique_images = []
    for f in image_files:
        if f.resolve() not in seen_paths:
            seen_paths.add(f.resolve())
            unique_images.append(f)
    image_files = unique_images

    # Validate + copy images
    if image_files:
        logger.info("Step 1: Validating images")
        valid = []
        for p in image_files:
            ok, reason = validate_image(p)
            if ok:
                valid.append(p)
                logger.info(f"  OK: {p.name}")
            else:
                logger.warning(f"  SKIP: {p.name} — {reason}")
        logger.info("Step 2: Copying images")
        copy_images(valid)

    save_labels(rows)

    if not args.skip_split:
        logger.info("Step 3: Regenerating train/val/test splits")
        run_pipeline()

    logger.info("\nDone! Added %d tire(s) to the dataset.", len(rows))


if __name__ == "__main__":
    main()
