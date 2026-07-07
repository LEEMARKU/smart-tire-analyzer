"""
Re-process blurry/rejected images with deblurring techniques.
Tries to recover images that were previously rejected.
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
    TARGET_SIZE,
)
from dataset.preprocessing.image_enhancement import (
    bilateral_filter, correct_illumination, morphological_operations,
)
from dataset.preprocessing.deblurring import deblur_pipeline

OUTPUT_DIR = Path("dataset/processed/enhanced_images")
PREVIEW_DIR = Path("dataset/processed/enhanced_previews")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

BLUR_THRESHOLD = 100.0
DEBLUR_ATTEMPTS = 3


def process_with_deblur(img_path: str, idx: int, total: int) -> dict:
    """Process image with deblurring recovery for borderline cases."""
    result = {"path": img_path, "status": "ok", "error": None, "deblur_used": False}
    try:
        img = cv2.imread(img_path)
        if img is None:
            return {**result, "status": "skipped", "error": "corrupt"}

        h, w = img.shape[:2]
        if h < 100 or w < 100:
            return {**result, "status": "skipped", "error": f"too small ({w}x{h})"}

        # Step 1: Blur detection
        is_blurry, blur_score = detect_blur(img)
        result["blur_score_before"] = round(blur_score, 1)

        # Step 2: If blurry, try deblurring
        if is_blurry:
            img, new_score, recovered = deblur_pipeline(
                img, blur_score, threshold=BLUR_THRESHOLD, max_attempts=DEBLUR_ATTEMPTS
            )
            result["blur_score_after"] = round(new_score, 1)
            result["deblur_used"] = True
            if not recovered:
                return {**result, "status": "rejected", "error": f"deblur failed ({blur_score:.0f}->{new_score:.0f})"}
        else:
            result["blur_score_after"] = result["blur_score_before"]

        # Step 3: Downscale
        img = downscale_for_processing(img)

        # Step 4: Tire detection & crop
        img = detect_and_crop_tire(img)

        # Step 5: Perspective correction
        img = correct_perspective(img)

        # Step 6: Noise reduction + bilateral
        img_bl = bilateral_filter(img)
        img_gb = reduce_noise(img, use_median=False)
        img = cv2.addWeighted(img_bl, 0.5, img_gb, 0.5, 0)

        # Step 7: Illumination correction
        img = correct_illumination(img)

        # Step 8: CLAHE
        img = apply_clahe(img)

        # Step 9: Sharpening
        img = sharpen_image(img)

        # Step 10: Edge detection
        edges = compute_edge_channel(img)

        # Step 11: Resize
        img_224 = resize_image(img)
        edges_224 = resize_image(np.stack([edges, edges, edges], axis=-1))[:, :, 0]

        # Step 12: Normalization
        norm = normalize_image(img_224, use_imagenet=True)
        edge_norm = edges_224.astype(np.float32) / 255.0
        final = np.concatenate([norm, edge_norm[:, :, np.newaxis]], axis=-1)

        # Step 13: Morphological cleanup on edge channel
        morph_edges = morphological_operations(edges, operation="close", kernel_size=3)
        morph_norm = morph_edges.astype(np.float32) / 255.0

        # Save
        stem = Path(img_path).stem
        out_path = OUTPUT_DIR / f"{stem}_preprocessed.npy"
        np.save(out_path, final.astype(np.float32))

        preview_rgb = ((norm[:, :, :3] * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])) * 255
        preview_rgb = preview_rgb[:, :, ::-1].clip(0, 255).astype(np.uint8)
        preview_path = PREVIEW_DIR / f"{stem}_preview.jpg"
        cv2.imwrite(str(preview_path), preview_rgb)

        result["output_shape"] = final.shape

        if idx % 25 == 0:
            deblur_tag = " [deblurred]" if result["deblur_used"] else ""
            logger.info(f"[{idx}/{total}] {stem}{deblur_tag} -> {final.shape}")

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)

    return result


def main():
    df = pd.read_csv("dataset/processed/features.csv")
    paths = df["image_path"].dropna().unique().tolist()
    existing = [p for p in paths if os.path.exists(p)]
    total = len(existing)

    logger.info(f"Processing {total} images with deblurring recovery...")

    stats = {
        "ok": 0, "rejected": 0, "skipped": 0, "error": 0,
        "deblurred": 0, "blur_scores_before": [], "blur_scores_after": [],
    }
    max_workers = min(8, os.cpu_count() or 4)
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_with_deblur, p, i + 1, total): p
            for i, p in enumerate(existing)
        }
        for future in as_completed(futures):
            r = future.result()
            stats[r["status"]] += 1
            if r.get("deblur_used") and r["status"] == "ok":
                stats["deblurred"] += 1
            if "blur_score_before" in r:
                stats["blur_scores_before"].append(r["blur_score_before"])
            if "blur_score_after" in r:
                stats["blur_scores_after"].append(r["blur_score_after"])

    elapsed = time.time() - t0

    print("\n" + "=" * 72)
    print("IMAGE PREPROCESSING WITH DEBLURRING — RESULTS")
    print("=" * 72)
    print(f"Total images:           {total}")
    print(f"Processed (ok):         {stats['ok']}")
    print(f"  Recovered via deblur: {stats['deblurred']}")
    print(f"Rejected (still blurry): {stats['rejected']}")
    print(f"Skipped:                {stats['skipped']}")
    print(f"Errors:                 {stats['error']}")
    print(f"Time:                   {elapsed:.1f}s")
    if stats["blur_scores_before"]:
        before = np.array(stats["blur_scores_before"])
        after = np.array(stats["blur_scores_after"])
        recovered_mask = (before < BLUR_THRESHOLD) & (after >= BLUR_THRESHOLD)
        print(f"\nBlur score stats:")
        print(f"  Before: mean={before.mean():.1f}, min={before.min():.1f}")
        print(f"  After:  mean={after.mean():.1f}, min={after.min():.1f}")
        print(f"  Recovered from blurry: {recovered_mask.sum()} images")
    print(f"\nOutput: {OUTPUT_DIR}/")
    print("=" * 72)


if __name__ == "__main__":
    main()
