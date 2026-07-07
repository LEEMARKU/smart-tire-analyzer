"""
Preprocess all tire images with the complete image preprocessing pipeline.
Runs both existing steps + new enhancement steps on every image.
"""

import sys, os, logging, time
sys.path.insert(0, '.')
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from ai_model.cnn.preprocessing import (
    detect_blur, detect_and_crop_tire, correct_perspective,
    reduce_noise, apply_clahe, sharpen_image, compute_edge_channel,
    resize_image, normalize_image, downscale_for_processing,
    TARGET_SIZE, BLUR_THRESHOLD,
)
from dataset.preprocessing.image_enhancement import (
    bilateral_filter, correct_illumination, global_histogram_equalization,
    segment_tire_region, morphological_operations,
    auto_rotate_correction, remove_shadows,
)

OUTPUT_DIR = Path("dataset/processed/enhanced_images")
PREVIEW_DIR = Path("dataset/processed/enhanced_previews")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)


def process_single_image(img_path: str, idx: int, total: int) -> dict:
    """Apply full image preprocessing pipeline to a single image."""
    result = {"path": img_path, "status": "ok", "error": None, "steps": {}}
    try:
        img = cv2.imread(img_path)
        if img is None:
            return {**result, "status": "skipped", "error": "corrupt/unreadable"}

        h, w = img.shape[:2]
        if h < 100 or w < 100:
            return {**result, "status": "skipped", "error": f"too small ({w}x{h})"}

        # Step 1: Blur detection (reject bad inputs)
        is_blurry, blur_score = detect_blur(img)
        result["steps"]["blur_score"] = round(blur_score, 1)
        if is_blurry:
            return {**result, "status": "rejected", "error": f"blurry (score={blur_score:.1f})"}

        img = downscale_for_processing(img)

        # Step 2: Auto rotation correction (deskew)
        img = auto_rotate_correction(img)
        result["steps"]["rotation_corrected"] = True

        # Step 3: Shadow removal
        img = remove_shadows(img)
        result["steps"]["shadow_removed"] = True

        # Step 4: Tire detection & crop (with GrabCut refinement)
        img = detect_and_crop_tire(img, use_grabcut=True)
        result["steps"]["tire_cropped"] = img.shape[:2]

        # Step 5: Perspective correction
        img = correct_perspective(img)

        # Step 6: Noise reduction (bilateral for edge-preserving)
        img_bl = bilateral_filter(img)
        img_gb = reduce_noise(img, use_median=False)
        img = cv2.addWeighted(img_bl, 0.5, img_gb, 0.5, 0)

        # Step 7: Illumination correction
        img = correct_illumination(img)

        # Step 8: CLAHE contrast enhancement
        img = apply_clahe(img)

        # Step 9: Sharpening
        img = sharpen_image(img)

        # Step 10: Edge detection
        edges = compute_edge_channel(img)

        # Step 11: Resize
        img_224 = resize_image(img)
        edges_224 = resize_image(np.stack([edges, edges, edges], axis=-1))[:, :, 0]

        # Step 12: Normalization (ImageNet)
        norm = normalize_image(img_224, use_imagenet=True)
        edge_norm = edges_224.astype(np.float32) / 255.0
        final = np.concatenate([norm, edge_norm[:, :, np.newaxis]], axis=-1)

        # Step 13: Morphological cleanup on edge channel
        morph_edges = morphological_operations(edges, operation="close", kernel_size=3)
        morph_norm = morph_edges.astype(np.float32) / 255.0

        # Save the 4-channel result
        stem = Path(img_path).stem
        out_path = OUTPUT_DIR / f"{stem}_preprocessed.npy"
        np.save(out_path, final.astype(np.float32))

        # Save a preview (first 3 RGB channels denormalized for viewing)
        preview_rgb = ((norm[:, :, :3] * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])) * 255
        preview_rgb = preview_rgb[:, :, ::-1].clip(0, 255).astype(np.uint8)
        preview_path = PREVIEW_DIR / f"{stem}_preview.jpg"
        cv2.imwrite(str(preview_path), preview_rgb)

        result["steps"]["output_shape"] = final.shape
        result["steps"]["output_file"] = str(out_path)

        if idx % 50 == 0:
            logger.info(f"[{idx}/{total}] {stem} -> {final.shape}")

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)

    return result


def main():
    df = pd.read_csv("dataset/processed/features.csv")
    paths = df["image_path"].dropna().unique().tolist()
    total = len(paths)
    logger.info(f"Found {total} unique image paths")

    # Filter to existing files
    existing = [p for p in paths if os.path.exists(p)]
    logger.info(f"{len(existing)} images exist on disk, {total - len(existing)} missing")

    if not existing:
        logger.error("No images found to preprocess!")
        return

    stats = {"ok": 0, "rejected": 0, "skipped": 0, "error": 0, "blur_scores": [], "times": []}
    max_workers = min(8, os.cpu_count() or 4)

    logger.info(f"Processing {len(existing)} images with {max_workers} workers...")
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_image, p, i + 1, len(existing)): p
            for i, p in enumerate(existing)
        }
        for future in as_completed(futures):
            r = future.result()
            stats[r["status"]] += 1
            if "blur_score" in r.get("steps", {}):
                stats["blur_scores"].append(r["steps"]["blur_score"])

    elapsed = time.time() - t0

    print("\n" + "=" * 72)
    print("IMAGE PREPROCESSING COMPLETE")
    print("=" * 72)
    print(f"Total images:     {len(existing)}")
    print(f"Processed (ok):   {stats['ok']}")
    print(f"Rejected (blurry): {stats['rejected']}")
    print(f"Skipped:          {stats['skipped']}")
    print(f"Errors:           {stats['error']}")
    if stats["blur_scores"]:
        print(f"Avg blur score:   {np.mean(stats['blur_scores']):.1f}")
    print(f"Time:             {elapsed:.1f}s ({elapsed/len(existing):.2f}s per image)")
    print(f"Output:           {OUTPUT_DIR}/")
    print(f"Previews:         {PREVIEW_DIR}/")
    print("=" * 72)

    # Save processing log
    log_df = df[["image_id", "image_path"]].copy()
    log_df["processed"] = log_df["image_path"].apply(
        lambda p: os.path.exists(OUTPUT_DIR / f"{Path(p).stem}_preprocessed.npy")
        if pd.notna(p) and os.path.exists(p) else False
    )
    log_df.to_csv("dataset/processed/image_preprocessing_log.csv", index=False)
    logger.info(f"Processing log saved to dataset/processed/image_preprocessing_log.csv")


if __name__ == "__main__":
    main()
