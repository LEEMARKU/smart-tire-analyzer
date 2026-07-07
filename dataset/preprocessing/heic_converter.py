"""
HEIC to JPEG Converter — Convert .heic images to JPEG format for processing.
Uses pillow-heif to read HEIC files and converts them to standard JPEG.
"""

import logging
from pathlib import Path
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)

try:
    from PIL import Image
    import pillow_heif
    pillow_heif.register_heif_opener()
    HAVE_PILLOW_HEIF = True
except ImportError:
    HAVE_PILLOW_HEIF = False
    Image = None


def convert_heic_to_jpeg(
    heic_path: Path,
    output_path: Optional[Path] = None,
    quality: int = 95,
    overwrite: bool = False,
) -> Optional[Path]:
    """
    Convert a single HEIC file to JPEG.

    Args:
        heic_path: Path to .heic file
        output_path: Desired output path. If None, replaces .heic with .jpg in same dir.
        quality: JPEG quality (1-100, default 95)
        overwrite: Overwrite existing output file

    Returns:
        Path to output JPEG, or None if conversion failed
    """
    if not HAVE_PILLOW_HEIF:
        logger.error("pillow-heif not installed. Install with: pip install pillow-heif")
        return None

    if not heic_path.exists():
        logger.error(f"HEIC file not found: {heic_path}")
        return None

    if heic_path.suffix.lower() not in {".heic", ".heif", ".hif"}:
        logger.warning(f"Not a HEIC file: {heic_path}")
        return None

    if output_path is None:
        output_path = heic_path.with_suffix(".jpg")

    if output_path.exists() and not overwrite:
        logger.info(f"Output already exists, skipping: {output_path}")
        return output_path

    try:
        image = Image.open(heic_path)
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")
        image.save(output_path, "JPEG", quality=quality)
        logger.info(f"Converted: {heic_path.name} -> {output_path.name}")
        return output_path
    except Exception as e:
        logger.error(f"Failed to convert {heic_path.name}: {e}")
        return None


def batch_convert_heic_to_jpeg(
    source_dir: Path,
    output_dir: Optional[Path] = None,
    quality: int = 95,
    recursive: bool = True,
    overwrite: bool = False,
    remove_original: bool = False,
) -> List[Path]:
    """
    Batch convert all HEIC files in a directory to JPEG.

    Args:
        source_dir: Directory to search for HEIC files
        output_dir: Output directory. If None, saves alongside originals.
        quality: JPEG quality
        recursive: Search subdirectories
        overwrite: Overwrite existing JPEGs
        remove_original: Delete original HEIC after successful conversion

    Returns:
        List of paths to converted JPEG files
    """
    if not HAVE_PILLOW_HEIF:
        logger.error("pillow-heif not installed.")
        return []

    source_dir = Path(source_dir)
    if not source_dir.is_dir():
        logger.error(f"Source directory not found: {source_dir}")
        return []

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    pattern = "**/*" if recursive else "*"
    heic_files = [
        p for p in source_dir.glob(f"{pattern}.heic")
    ] + [
        p for p in source_dir.glob(f"{pattern}.HEIC")
    ] + [
        p for p in source_dir.glob(f"{pattern}.heif")
    ] + [
        p for p in source_dir.glob(f"{pattern}.HEIF")
    ]

    if not heic_files:
        logger.info(f"No HEIC files found in {source_dir}")
        return []

    converted = []
    for heic_path in heic_files:
        if output_dir:
            rel_path = heic_path.relative_to(source_dir)
            out_path = output_dir / rel_path.with_suffix(".jpg")
            out_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            out_path = heic_path.with_suffix(".jpg")

        result = convert_heic_to_jpeg(heic_path, out_path, quality, overwrite)
        if result:
            converted.append(result)
            if remove_original:
                try:
                    heic_path.unlink()
                    logger.info(f"Removed original: {heic_path.name}")
                except Exception as e:
                    logger.warning(f"Failed to remove {heic_path.name}: {e}")

    logger.info(f"Converted {len(converted)}/{len(heic_files)} HEIC files to JPEG")
    return converted


def heic_files_in_dir(directory: Path, recursive: bool = True) -> List[Path]:
    """List all HEIC/HEIF files in a directory."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    prefix = "**/" if recursive else ""
    files = []
    for ext in ("*.heic", "*.HEIC", "*.heif", "*.HEIF", "*.hif", "*.HIF"):
        files.extend(directory.glob(f"{prefix}{ext}"))
    return sorted(set(files))


def ensure_images_are_jpeg(
    image_dir: Path,
    output_dir: Optional[Path] = None,
    quality: int = 95,
    recursive: bool = True,
    overwrite: bool = False,
    remove_original: bool = False,
) -> Tuple[List[Path], List[Path]]:
    """
    Ensure all images in a directory are JPEG format.
    Converts HEIC files to JPEG; leaves existing JPEGs untouched.

    Args:
        image_dir: Directory containing images
        output_dir: Output directory for converted JPEGs
        quality: JPEG quality
        recursive: Search subdirectories
        overwrite: Overwrite existing converted files
        remove_original: Delete HEIC originals after conversion

    Returns:
        (converted_jpegs, skipped_jpegs) paths
    """
    heic_files = heic_files_in_dir(image_dir, recursive=recursive)
    if not heic_files:
        logger.info(f"No HEIC files found in {image_dir}")
        return [], []

    converted = batch_convert_heic_to_jpeg(
        source_dir=image_dir,
        output_dir=output_dir,
        quality=quality,
        recursive=recursive,
        overwrite=overwrite,
        remove_original=remove_original,
    )

    skipped = [f for f in heic_files if f.with_suffix(".jpg") not in converted]
    return converted, skipped


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Convert HEIC images to JPEG format"
    )
    parser.add_argument("source", type=str, help="Source directory or file")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality (1-100)")
    parser.add_argument("--no-recursive", action="store_true", help="Do not search subdirectories")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing JPEGs")
    parser.add_argument("--remove-original", action="store_true", help="Delete HEIC after conversion")
    args = parser.parse_args()

    source = Path(args.source)
    if source.is_file():
        convert_heic_to_jpeg(source, quality=args.quality)
    elif source.is_dir():
        batch_convert_heic_to_jpeg(
            source_dir=source,
            output_dir=Path(args.output_dir) if args.output_dir else None,
            quality=args.quality,
            recursive=not args.no_recursive,
            overwrite=args.overwrite,
            remove_original=args.remove_original,
        )
    else:
        logger.error(f"Source not found: {source}")
