"""
Add labeled front-view tire samples for continuous learning.

Single sample:
    python scripts/add_learning_samples.py --image C:\\tires\\front1.jpg --tread-1 5.2 --tread-2 5.1 --tread-3 5.0 --tread-4 5.1 --brand MRF --tire-size "165/80 R14" --prepare

Batch CSV:
    python scripts/add_learning_samples.py --csv C:\\tires\\new_samples.csv --prepare

CSV columns:
    image_path,tread_1,tread_2,tread_3,tread_4,brand,tire_model,tire_size,sidewall_image_path,ocr_text,tube_type
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_DATASET_ROOT = ROOT / "dataset"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif"}
LABEL_COLUMNS = [
    "image_id",
    "front_image_reference",
    "sidewall_image_id",
    "tread_1",
    "tread_2",
    "tread_3",
    "tread_4",
    "tread_average",
    "brand",
    "tire_model",
    "tire_size",
    "ocr_text",
    "tube_type",
]

FRONT_PATH_COLUMNS = ("image_path", "front_image_path", "front_view_path", "front_image", "front")
SIDEWALL_PATH_COLUMNS = ("sidewall_image_path", "sidewall_path", "sidewall_image", "sidewall")


@dataclass(frozen=True)
class LearningSample:
    front_image: Path
    tread_1: float
    tread_2: float
    tread_3: float
    tread_4: float
    image_id: str = ""
    brand: str = ""
    tire_model: str = ""
    tire_size: str = ""
    sidewall_image: Path | None = None
    ocr_text: str = ""
    tube_type: str = ""

    @property
    def tread_average(self) -> float:
        return round((self.tread_1 + self.tread_2 + self.tread_3 + self.tread_4) / 4.0, 4)


def _clean_text(value: object) -> str:
    return str(value or "").strip()


def _safe_stem(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-")
    return cleaned[:120] or "sample"


def _resolve_path(value: str, base_dir: Path | None = None) -> Path:
    text = _clean_text(value)
    if not text:
        raise ValueError("Missing image path")
    path = Path(text).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image type for {path.name}; expected {sorted(IMAGE_EXTENSIONS)}")
    return path


def _parse_depth(value: object, column: str) -> float:
    text = _clean_text(value)
    if text == "":
        raise ValueError(f"{column} is required")
    try:
        depth = float(text)
    except ValueError as exc:
        raise ValueError(f"{column} must be a number: {text}") from exc
    if depth < 0.0 or depth > 12.0:
        raise ValueError(f"{column} must be between 0 and 12 mm")
    return depth


def _first_value(row: dict[str, str], candidates: tuple[str, ...]) -> str:
    normalized = {key.strip().lower(): value for key, value in row.items()}
    for candidate in candidates:
        value = normalized.get(candidate)
        if _clean_text(value):
            return _clean_text(value)
    return ""


def _copy_image(source: Path, target_dir: Path, target_name: str) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / target_name
    if target.exists():
        try:
            if target.samefile(source):
                return target
        except OSError:
            pass
    shutil.copy2(source, target)
    return target


def _sample_to_row(sample: LearningSample, dataset_root: Path) -> dict[str, str]:
    learning_root = dataset_root / "raw" / "learning_samples"
    front_dir = learning_root / "front_view"
    sidewall_dir = learning_root / "sidewall"

    image_stem = _safe_stem(sample.image_id or sample.front_image.stem)
    front_target = _copy_image(sample.front_image, front_dir, f"{image_stem}{sample.front_image.suffix.lower()}")

    sidewall_id = ""
    if sample.sidewall_image is not None:
        sidewall_target = _copy_image(
            sample.sidewall_image,
            sidewall_dir,
            f"{image_stem}_sidewall{sample.sidewall_image.suffix.lower()}",
        )
        sidewall_id = sidewall_target.name

    return {
        "image_id": front_target.name,
        "front_image_reference": front_target.name,
        "sidewall_image_id": sidewall_id,
        "tread_1": f"{sample.tread_1:.4g}",
        "tread_2": f"{sample.tread_2:.4g}",
        "tread_3": f"{sample.tread_3:.4g}",
        "tread_4": f"{sample.tread_4:.4g}",
        "tread_average": f"{sample.tread_average:.4g}",
        "brand": sample.brand,
        "tire_model": sample.tire_model,
        "tire_size": sample.tire_size,
        "ocr_text": sample.ocr_text,
        "tube_type": sample.tube_type,
    }


def _read_existing_labels(labels_path: Path) -> list[dict[str, str]]:
    if not labels_path.exists():
        return []
    with labels_path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_labels(labels_path: Path, rows: list[dict[str, str]]) -> None:
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    with labels_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LABEL_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in LABEL_COLUMNS})


def add_samples(samples: list[LearningSample], dataset_root: Path) -> Path:
    labels_path = dataset_root / "raw" / "learning_samples" / "labels.csv"
    existing = _read_existing_labels(labels_path)
    by_image_id = {row.get("image_id", ""): row for row in existing if row.get("image_id")}

    for sample in samples:
        row = _sample_to_row(sample, dataset_root)
        by_image_id[row["image_id"]] = row

    _write_labels(labels_path, list(by_image_id.values()))
    return labels_path


def sample_from_args(args: argparse.Namespace) -> LearningSample:
    missing = [
        option
        for option, value in {
            "--tread-1": args.tread_1,
            "--tread-2": args.tread_2,
            "--tread-3": args.tread_3,
            "--tread-4": args.tread_4,
        }.items()
        if value is None
    ]
    if missing:
        raise ValueError(f"Single-image mode requires: {', '.join(missing)}")

    sidewall = _resolve_path(args.sidewall_image) if args.sidewall_image else None
    return LearningSample(
        front_image=_resolve_path(args.image),
        image_id=_clean_text(args.image_id),
        tread_1=_parse_depth(args.tread_1, "tread_1"),
        tread_2=_parse_depth(args.tread_2, "tread_2"),
        tread_3=_parse_depth(args.tread_3, "tread_3"),
        tread_4=_parse_depth(args.tread_4, "tread_4"),
        brand=_clean_text(args.brand),
        tire_model=_clean_text(args.tire_model),
        tire_size=_clean_text(args.tire_size),
        sidewall_image=sidewall,
        ocr_text=_clean_text(args.ocr_text),
        tube_type=_clean_text(args.tube_type),
    )


def samples_from_csv(csv_path: Path) -> list[LearningSample]:
    csv_path = csv_path.resolve()
    samples: list[LearningSample] = []
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader, start=2):
            front_value = _first_value(row, FRONT_PATH_COLUMNS)
            sidewall_value = _first_value(row, SIDEWALL_PATH_COLUMNS)
            try:
                sidewall = _resolve_path(sidewall_value, csv_path.parent) if sidewall_value else None
                samples.append(
                    LearningSample(
                        front_image=_resolve_path(front_value, csv_path.parent),
                        image_id=_clean_text(row.get("image_id")),
                        tread_1=_parse_depth(row.get("tread_1"), "tread_1"),
                        tread_2=_parse_depth(row.get("tread_2"), "tread_2"),
                        tread_3=_parse_depth(row.get("tread_3"), "tread_3"),
                        tread_4=_parse_depth(row.get("tread_4"), "tread_4"),
                        brand=_clean_text(row.get("brand")),
                        tire_model=_clean_text(row.get("tire_model")),
                        tire_size=_clean_text(row.get("tire_size")),
                        sidewall_image=sidewall,
                        ocr_text=_clean_text(row.get("ocr_text")),
                        tube_type=_clean_text(row.get("tube_type")),
                    )
                )
            except Exception as exc:
                raise ValueError(f"{csv_path}:{index}: {exc}") from exc
    if not samples:
        raise ValueError(f"No samples found in {csv_path}")
    return samples


def write_template(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "image_path",
            "tread_1",
            "tread_2",
            "tread_3",
            "tread_4",
            "brand",
            "tire_model",
            "tire_size",
            "sidewall_image_path",
            "ocr_text",
            "tube_type",
        ])
        writer.writeheader()
        writer.writerow({
            "image_path": "front_001.jpg",
            "tread_1": "5.2",
            "tread_2": "5.1",
            "tread_3": "5.0",
            "tread_4": "5.1",
            "brand": "MRF",
            "tire_model": "ZLX",
            "tire_size": "165/80 R14",
            "sidewall_image_path": "",
            "ocr_text": "",
            "tube_type": "TUBELESS",
        })


def run_prepare_dataset(dataset_root: Path) -> None:
    import importlib

    module = importlib.import_module("scripts.prepare_dataset")
    manifest = module.prepare_dataset(paths=module.DatasetPaths(root=dataset_root))
    module.print_summary(manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add labeled tire images to the continuous-learning intake set.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--csv", type=Path, help="Batch CSV with image_path and tread_1..tread_4 columns.")
    parser.add_argument("--image", type=str, help="Single front-view tire image.")
    parser.add_argument("--image-id", type=str, default="", help="Optional stable ID/name for this sample.")
    parser.add_argument("--sidewall-image", type=str, default="", help="Optional sidewall image for the same tire.")
    parser.add_argument("--tread-1", type=float)
    parser.add_argument("--tread-2", type=float)
    parser.add_argument("--tread-3", type=float)
    parser.add_argument("--tread-4", type=float)
    parser.add_argument("--brand", type=str, default="")
    parser.add_argument("--tire-model", type=str, default="")
    parser.add_argument("--tire-size", type=str, default="")
    parser.add_argument("--ocr-text", type=str, default="")
    parser.add_argument("--tube-type", type=str, default="")
    parser.add_argument("--prepare", action="store_true", help="Run scripts/prepare_dataset.py after adding samples.")
    parser.add_argument("--write-template", type=Path, help="Write a starter CSV template and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.write_template:
        write_template(args.write_template)
        print(f"Template written: {args.write_template}")
        return

    if bool(args.csv) == bool(args.image):
        raise SystemExit("Choose exactly one input mode: --csv or --image")

    dataset_root = args.dataset_root.resolve()
    samples = samples_from_csv(args.csv) if args.csv else [sample_from_args(args)]
    labels_path = add_samples(samples, dataset_root)
    print(f"Added/updated {len(samples)} learning sample(s).")
    print(f"Labels: {labels_path}")
    print(f"Front images: {dataset_root / 'raw' / 'learning_samples' / 'front_view'}")

    if args.prepare:
        run_prepare_dataset(dataset_root)
    else:
        print("Next: python scripts/prepare_dataset.py")


if __name__ == "__main__":
    main()
