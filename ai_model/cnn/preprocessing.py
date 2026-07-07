"""
Smart Tire Analyzer — Image Preprocessing Pipeline
Implements ALL 10 required preprocessing techniques in the correct order:
  Phase 1: Blur Detection (input validation)
  Phase 2: Tire Detection & Crop, Perspective Correction
  Phase 3: Noise Reduction, CLAHE, Sharpening, Edge Detection
  Phase 4: Resize (224×224), Normalization
  Phase 5: Data Augmentation (training only)
"""

import cv2
import numpy as np
import albumentations as A
from typing import Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)

# ImageNet normalization constants (MobileNetV2 pretrained)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

TARGET_SIZE = (224, 224)
BLUR_THRESHOLD = 100.0  # Laplacian variance — reject below this
MAX_PROCESSING_DIM = 768


# ---------------------------------------------------------------------------
# Step 10 — Blur Detection (Phase 1: run FIRST, reject bad inputs)
# ---------------------------------------------------------------------------
def detect_blur(image: np.ndarray, threshold: float = BLUR_THRESHOLD) -> Tuple[bool, float]:
    """
    Detect blurry images using Laplacian variance.
    Returns (is_blurry: bool, score: float).
    Images below threshold are rejected before any further processing.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    is_blurry = bool(score < threshold)
    if is_blurry:
        logger.warning(f"Image rejected: blur score {score:.2f} < threshold {threshold}")
    return is_blurry, score


# ---------------------------------------------------------------------------
# Step 4 — Tire Detection & Cropping (Phase 2)
# ---------------------------------------------------------------------------
def grabcut_background_removal(
    image: np.ndarray,
    rect: Optional[Tuple[int, int, int, int]] = None,
    mask_init: Optional[np.ndarray] = None,
    iter_count: int = 5,
) -> np.ndarray:
    """
    Remove background using GrabCut segmentation.
    Produces a cleaner tire mask than Hough circle cropping alone.

    Args:
        image: BGR image
        rect: (x, y, w, h) bounding rectangle for GrabCut initialization.
              If None, uses the full image minus 10px border.
        mask_init: Optional initial mask (GC_FGD/GC_BGD/GC_PR_FGD/GC_PR_BGD).
                   If None, uses rect for initialization.
        iter_count: Number of GrabCut iterations (5 is usually sufficient)

    Returns:
        BGR image with background set to black (pixels=0)
    """
    if rect is None:
        h, w = image.shape[:2]
        rect = (10, 10, w - 20, h - 20)

    mask = np.zeros(image.shape[:2], np.uint8) if mask_init is None else mask_init.copy()
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    if mask_init is None:
        cv2.grabCut(image, mask, rect, bgd_model, fgd_model, iter_count, cv2.GC_INIT_WITH_RECT)
    else:
        cv2.grabCut(image, mask, None, bgd_model, fgd_model, iter_count, cv2.GC_INIT_WITH_MASK)

    foreground_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    result = image.copy()
    result[foreground_mask == 0] = 0
    return result


def detect_and_crop_tire(image: np.ndarray, padding: int = 10, use_grabcut: bool = True) -> np.ndarray:
    """
    Detect tire region using OpenCV Hough Circle Transform.
    Falls back to full image if no circle found.
    For production: replace with YOLOv8 nano detection.

    Args:
        image: BGR image
        padding: Extra pixels around detected region

    Returns:
        Cropped BGR image containing only the tire
    """
    h, w = image.shape[:2]
    scale = 1.0
    detect_image = image
    if max(h, w) > 1280:
        scale = 1280.0 / float(max(h, w))
        detect_image = cv2.resize(
            image,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )

    gray = cv2.cvtColor(detect_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    detect_h, detect_w = detect_image.shape[:2]
    min_radius = max(10, min(detect_h, detect_w) // 6)
    max_radius = max(min_radius + 5, min(detect_h, detect_w) // 2)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min(detect_h, detect_w) // 2,
        param1=80,
        param2=40,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        cx, cy, r = circles[0]  # Use largest/first detected circle
        if scale != 1.0:
            cx = int(round(cx / scale))
            cy = int(round(cy / scale))
            r = int(round(r / scale))
        x1 = max(0, cx - r - padding)
        y1 = max(0, cy - r - padding)
        x2 = min(w, cx + r + padding)
        y2 = min(h, cy + r + padding)
        cropped = image[y1:y2, x1:x2]
        logger.debug(f"Tire detected: center=({cx},{cy}), radius={r}")

        if use_grabcut and cropped.shape[0] >= 50 and cropped.shape[1] >= 50:
            refined = grabcut_background_removal(cropped, rect=(5, 5, cropped.shape[1] - 10, cropped.shape[0] - 10))
            fg_mask = np.any(refined > 0, axis=-1)
            if fg_mask.sum() > 0.1 * fg_mask.size:
                cropped = refined

        return cropped

    # Fallback: contour-based detection + GrabCut
    logger.warning("No tire circle detected — trying contour-based detection + GrabCut")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, cw, ch = cv2.boundingRect(largest)
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w, x + cw + padding)
        y2 = min(h, y + ch + padding)
        cropped = image[y1:y2, x1:x2]
        if use_grabcut and cropped.shape[0] >= 50 and cropped.shape[1] >= 50:
            mask_init = np.zeros(cropped.shape[:2], np.uint8)
            mask_init[padding:-padding, padding:-padding] = cv2.GC_PR_FGD
            refined = grabcut_background_removal(cropped, mask_init=mask_init)
            fg_mask = np.any(refined > 0, axis=-1)
            if fg_mask.sum() > 0.1 * fg_mask.size:
                cropped = refined
        return cropped

    logger.warning("No tire region found — using full image")
    return image


# ---------------------------------------------------------------------------
# Step 9 — Perspective Correction (Phase 2: before resize)
# ---------------------------------------------------------------------------
def correct_perspective(
    image: np.ndarray,
    src_pts: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Apply perspective warp to get flat frontal view of tire face.
    If no source points provided, applies auto-detection using contours.

    Args:
        image: BGR image (cropped tire)
        src_pts: 4 corner points (float32 array shape (4,2)), or None for auto

    Returns:
        Perspective-corrected BGR image
    """
    h, w = image.shape[:2]

    if src_pts is None:
        # Auto-detect corners from largest contour
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return image  # No contour found — skip correction

        cnt = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        src_pts = np.float32(box)

    dst_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(image, M, (w, h))
    return warped


# ---------------------------------------------------------------------------
# Step 3 — Noise Reduction (Phase 3)
# ---------------------------------------------------------------------------
def reduce_noise(image: np.ndarray, use_median: bool = False) -> np.ndarray:
    """
    Reduce sensor noise from phone cameras.

    Args:
        image: BGR image
        use_median: Use median blur (better for salt-and-pepper / dirty tires)
                    Use Gaussian blur for general noise (default)

    Returns:
        Denoised BGR image
    """
    if use_median:
        return cv2.medianBlur(image, 3)
    return cv2.GaussianBlur(image, (3, 3), 0)


# ---------------------------------------------------------------------------
# Step 5 — Contrast Enhancement (CLAHE, Phase 3)
# ---------------------------------------------------------------------------
def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) in LAB space.
    Critical for black rubber tires — makes grooves visible to CNN.

    Args:
        image: BGR image
        clip_limit: Controls noise amplification (2.0 recommended)
        tile_size: Grid size for local equalization (8×8 for tire face)

    Returns:
        Contrast-enhanced BGR image
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return enhanced


# ---------------------------------------------------------------------------
# Step 7 — Sharpening (Phase 3)
# ---------------------------------------------------------------------------
def sharpen_image(image: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """
    Apply unsharp masking to sharpen tread groove edges.
    WARNING: Keep strength subtle to avoid amplifying noise.

    Args:
        image: BGR image
        strength: Multiplier for sharpening kernel (1.0 = standard)

    Returns:
        Sharpened BGR image
    """
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32) * strength
    kernel[1, 1] = 5 * strength  # Center weight
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened


# ---------------------------------------------------------------------------
# Step 6 — Edge Detection (Phase 3: adds 4th channel)
# ---------------------------------------------------------------------------
def compute_edge_channel(image: np.ndarray) -> np.ndarray:
    """
    Compute Sobel edge map to use as 4th CNN input channel.
    Do NOT replace the image — blend as [R, G, B, edges].

    Args:
        image: BGR image (should be after CLAHE + sharpening)

    Returns:
        Edge map as uint8 single-channel image (0-255)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edges = cv2.magnitude(sobelx, sobely)
    edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return edges


# ---------------------------------------------------------------------------
# Step 1 — Resize to 224×224 (Phase 4)
# ---------------------------------------------------------------------------
def pad_to_square(image: np.ndarray) -> np.ndarray:
    """Pad image to square with black border WITHOUT stretching aspect ratio."""
    h, w = image.shape[:2]
    size = max(h, w)
    padded = np.zeros((size, size, image.shape[2] if len(image.shape) == 3 else 1), dtype=image.dtype)
    if len(image.shape) == 2:
        padded = np.zeros((size, size), dtype=image.dtype)
    offset_h = (size - h) // 2
    offset_w = (size - w) // 2
    padded[offset_h:offset_h + h, offset_w:offset_w + w] = image
    return padded


def resize_image(image: np.ndarray, size: Tuple[int, int] = TARGET_SIZE) -> np.ndarray:
    """
    Resize to 224×224 after padding to square.
    Uses INTER_AREA for downscaling, INTER_LINEAR for upscaling.
    """
    padded = pad_to_square(image)
    h, w = padded.shape[:2]
    interp = cv2.INTER_AREA if h > size[0] else cv2.INTER_LINEAR
    resized = cv2.resize(padded, size, interpolation=interp)
    return resized


def downscale_for_processing(image: np.ndarray, max_dim: int = MAX_PROCESSING_DIM) -> np.ndarray:
    """Shrink very large source images to keep preprocessing responsive."""
    h, w = image.shape[:2]
    largest = max(h, w)
    if largest <= max_dim:
        return image

    scale = max_dim / float(largest)
    resized = cv2.resize(
        image,
        (max(1, int(w * scale)), max(1, int(h * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized


# ---------------------------------------------------------------------------
# Step 2 — Normalization (Phase 4)
# ---------------------------------------------------------------------------
def normalize_image(image: np.ndarray, use_imagenet: bool = True) -> np.ndarray:
    """
    Normalize pixel values for neural network input.

    Args:
        image: uint8 BGR image (224×224)
        use_imagenet: Use ImageNet mean/std (True) or simple [0,1] (False)

    Returns:
        float32 array normalized for model input
    """
    img = image.astype(np.float32) / 255.0
    if use_imagenet:
        # Convert BGR→RGB for ImageNet stats
        img = img[:, :, ::-1] if img.shape[-1] >= 3 else img
        img[..., :3] = (img[..., :3] - IMAGENET_MEAN) / IMAGENET_STD
    return img


# ---------------------------------------------------------------------------
# Step 8 — Data Augmentation (Phase 5: training only)
# ---------------------------------------------------------------------------
def build_augmentation_pipeline(training: bool = True) -> A.Compose:
    """
    Build Albumentations augmentation pipeline.
    Only applied during TRAINING — not at inference time.

    Safe transforms: brightness, contrast, horizontal flip, small rotation
    Careful transforms: zoom (directional tread), vertical flip
    Avoided: 180° rotation (unnatural tire appearance)
    """
    if not training:
        return A.Compose([])  # No-op for validation/inference

    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.5),
        A.Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_REFLECT),
        A.RandomScale(scale_limit=0.1, p=0.3),
        A.GaussNoise(var_limit=(5.0, 30.0), p=0.3),
        A.CoarseDropout(max_holes=4, max_height=16, max_width=16, p=0.2),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=15, val_shift_limit=15, p=0.3),
        A.ElasticTransform(alpha=30, sigma=5, p=0.15),
        A.GridDistortion(num_steps=3, distort_limit=0.2, p=0.15),
    ])


# ---------------------------------------------------------------------------
# Master Pipeline — runs all 10 steps in correct order
# ---------------------------------------------------------------------------
def run_preprocessing_pipeline(
    image: np.ndarray,
    training: bool = False,
    include_edge_channel: bool = True,
    blur_threshold: float = BLUR_THRESHOLD,
) -> Optional[np.ndarray]:
    """
    Master preprocessing function — runs all 10 techniques in the correct order.

    Order:
      1. Blur detection (reject bad inputs)      [Step 10]
      2. Tire detection & crop                   [Step 4]
      3. Perspective correction                  [Step 9]
      4. Noise reduction                         [Step 3]
      5. CLAHE contrast enhancement              [Step 5]
      6. Sharpening                              [Step 7]
      7. Edge detection (4th channel)            [Step 6]
      8. Resize 224×224                          [Step 1]
      9. Normalization                           [Step 2]
     10. Data augmentation (training only)       [Step 8]

    Args:
        image: Raw BGR image from camera/file
        training: If True, applies augmentation
        include_edge_channel: If True, outputs 4-channel (RGBE) array
        blur_threshold: Laplacian score below this rejects image

    Returns:
        float32 numpy array of shape (224, 224, 4) or (224, 224, 3),
        or None if image is rejected (too blurry)
    """
    # Step 10: Blur detection — run FIRST
    is_blurry, score = detect_blur(image, blur_threshold)
    if is_blurry:
        return None

    image = downscale_for_processing(image)

    # Step 4: Tire detection & crop
    image = detect_and_crop_tire(image)

    # Step 9: Perspective correction
    image = correct_perspective(image)

    # Step 3: Noise reduction
    image = reduce_noise(image)

    # Step 5: CLAHE contrast enhancement
    image = apply_clahe(image)

    # Step 7: Sharpening
    image = sharpen_image(image)

    # Step 6: Edge detection (compute before resize for best quality)
    if include_edge_channel:
        edges = compute_edge_channel(image)

    # Step 1: Resize to 224×224
    image = resize_image(image)
    if include_edge_channel:
        edges = resize_image(np.stack([edges, edges, edges], axis=-1))[:, :, 0]

    # Step 2: Normalization
    normalized = normalize_image(image, use_imagenet=True)

    # Add edge channel (4th channel)
    if include_edge_channel:
        edge_norm = edges.astype(np.float32) / 255.0
        normalized = np.concatenate([normalized, edge_norm[:, :, np.newaxis]], axis=-1)

    # Step 8: Augmentation (training only)
    if training:
        aug = build_augmentation_pipeline(training=True)
        # Augment on uint8 (3 channels), then re-normalize
        img_uint8 = image.astype(np.uint8)
        augmented = aug(image=img_uint8)["image"]
        augmented = resize_image(augmented)
        aug_norm = normalize_image(augmented, use_imagenet=True)
        if include_edge_channel:
            aug_edges = compute_edge_channel(augmented)
            edge_norm = aug_edges.astype(np.float32) / 255.0
            normalized = np.concatenate([aug_norm, edge_norm[:, :, np.newaxis]], axis=-1)
        else:
            normalized = aug_norm

    return normalized.astype(np.float32)


if __name__ == "__main__":
    # Quick smoke test
    import logging
    logging.basicConfig(level=logging.DEBUG)
    test_img = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
    result = run_preprocessing_pipeline(test_img, training=False)
    if result is not None:
        logger.debug("Output shape: %s, dtype: %s", result.shape, result.dtype)
        logger.debug("Value range: [%.3f, %.3f]", result.min(), result.max())
    else:
        logger.debug("Image rejected (too blurry)")
