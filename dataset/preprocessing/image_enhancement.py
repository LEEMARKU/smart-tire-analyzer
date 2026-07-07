"""
Image Enhancement — Missing image preprocessing steps:
  1. Bilateral filtering (edge-preserving denoising)
  2. Illumination correction (Gaussian-based)
  3. Histogram equalization (standalone global)
  4. Image segmentation (threshold-based tire region)
  5. Morphological operations (erosion, dilation, opening, closing)
  6. Auto rotation correction (deskew via Hough lines)
  7. Shadow removal (LAB illumination estimation)
"""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def bilateral_filter(
    image: np.ndarray,
    d: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0,
) -> np.ndarray:
    """
    Edge-preserving noise reduction using bilateral filtering.
    Smoothes tire surface noise while preserving tread groove edges.

    Args:
        image: BGR image
        d: Diameter of pixel neighborhood (9 recommended)
        sigma_color: Filter sigma in color space (75 for tire rubber)
        sigma_space: Filter sigma in coordinate space (75)

    Returns:
        Denoised BGR image with preserved edges
    """
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def correct_illumination(
    image: np.ndarray,
    kernel_size: int = 61,
) -> np.ndarray:
    """
    Correct uneven illumination using Gaussian-based background estimation.
    Essential for tire images with directional lighting or shadows.

    Args:
        image: BGR image
        kernel_size: Size of Gaussian kernel for background estimation (odd)

    Returns:
        Illumination-corrected BGR image
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    illumination = blurred.astype(np.float32)
    corrected = gray.astype(np.float32) - illumination
    corrected = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(corrected, cv2.COLOR_GRAY2BGR)


def global_histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    Apply global histogram equalization on the value channel.
    Complement to CLAHE for overall contrast adjustment.

    Args:
        image: BGR image

    Returns:
        Histogram-equalized BGR image
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def segment_tire_region(
    image: np.ndarray,
    method: str = "otsu",
    morph_kernel_size: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segment tire region from background using thresholding.
    Provides a binary mask for ROI extraction.

    Args:
        image: BGR image
        method: 'otsu' for Otsu threshold, 'adaptive' for adaptive threshold
        morph_kernel_size: Size of morphological kernel for cleanup

    Returns:
        (mask, segmented_image) where mask is binary uint8, segmented is BGR
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if method == "otsu":
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        mask = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5
        )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(mask)
        cv2.drawContours(mask, [largest], -1, 255, -1)

    segmented = cv2.bitwise_and(image, image, mask=mask)
    return mask, segmented


def morphological_operations(
    image: np.ndarray,
    operation: str = "close",
    kernel_size: int = 5,
    iterations: int = 1,
) -> np.ndarray:
    """
    Apply morphological operations for tire groove analysis.
    Useful for cleaning binary masks and separating tread patterns.

    Args:
        image: Grayscale or binary image
        operation: 'erode', 'dilate', 'open', 'close', 'gradient', 'tophat', 'blackhat'
        kernel_size: Size of structuring element
        iterations: Number of iterations

    Returns:
        Processed image
    """
    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    ops = {
        "erode": cv2.erode,
        "dilate": cv2.dilate,
        "open": cv2.morphologyEx,
        "close": cv2.morphologyEx,
        "gradient": cv2.morphologyEx,
        "tophat": cv2.morphologyEx,
        "blackhat": cv2.morphologyEx,
    }

    op = ops.get(operation)
    if op is None:
        raise ValueError(f"Unknown morphological operation: {operation}")

    if operation in ("erode", "dilate"):
        result = op(gray, kernel, iterations=iterations)
    else:
        morph_type = getattr(cv2, f"MORPH_{operation.upper()}")
        result = op(gray, morph_type, kernel, iterations=iterations)

    return result


def auto_rotate_correction(
    image: np.ndarray,
    max_angle: float = 45.0,
    min_line_length: int = 50,
) -> np.ndarray:
    """
    Auto-deskew/rotation correction using Hough line detection.
    Finds the dominant angle of lines in the image and rotates to straighten.

    Works well on tire images where tread grooves form parallel lines.
    The dominant line angle is computed via Hough Transform, then the
    image is rotated to align the dominant lines to horizontal.

    Args:
        image: BGR image
        max_angle: Maximum rotation angle to consider (degrees)
        min_line_length: Minimum line length for Hough transform

    Returns:
        Rotation-corrected BGR image
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=min_line_length,
        maxLineGap=10,
    )

    if lines is None or len(lines) < 5:
        logger.debug("Not enough lines detected for rotation correction — skipping")
        return image

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) < max_angle:
            angles.append(angle)

    if not angles:
        return image

    median_angle = np.median(angles)

    if abs(median_angle) < 1.5:
        return image

    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT,
    )
    logger.debug(f"Rotation corrected by {median_angle:.2f} degrees")
    return rotated


def remove_shadows(
    image: np.ndarray,
    kernel_size: int = 61,
    shadow_threshold: float = 0.85,
    boost_strength: float = 1.4,
) -> np.ndarray:
    """
    Remove/reduce shadows from tire images using illumination estimation in LAB space.

    Works by:
      1. Converting to LAB color space
      2. Estimating local illumination via large Gaussian blur on L channel
      3. Identifying shadow regions (L significantly below local illumination)
      4. Boosting L in shadow regions while preserving color (AB channels)

    This is more advanced than correct_illumination() because it operates
    in color space and selectively boosts only shadow regions instead of
    globally subtracting illumination.

    Args:
        image: BGR image
        kernel_size: Gaussian kernel for illumination estimation (odd, large)
        shadow_threshold: Fraction of local illumination below which is shadow
        boost_strength: Multiplier for shadow region brightness boost

    Returns:
        Shadow-reduced BGR image
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    l, a, b = cv2.split(lab)

    illumination = cv2.GaussianBlur(l, (kernel_size, kernel_size), 0)
    illumination = np.maximum(illumination, 1.0)

    shadow_mask = (l / illumination) < shadow_threshold

    l_corrected = l.copy()
    l_corrected[shadow_mask] = np.minimum(
        l[shadow_mask] * boost_strength,
        illumination[shadow_mask],
    )

    l_corrected = np.clip(l_corrected, 0, 255)
    corrected = cv2.merge([l_corrected, a, b]).astype(np.uint8)
    return cv2.cvtColor(corrected, cv2.COLOR_LAB2BGR)


def apply_all_image_enhancements(
    image: np.ndarray,
    do_bilateral: bool = True,
    do_rotation_correction: bool = True,
    do_shadow_removal: bool = True,
    do_illumination: bool = True,
    do_hist_equal: bool = False,
    do_segmentation: bool = False,
    do_morphology: Optional[str] = None,
) -> np.ndarray:
    """
    Apply all image enhancement steps in optimal order.

    Order:
      1. Auto rotation correction (deskew)
      2. Shadow removal (illumination estimation in LAB)
      3. Bilateral filtering (edge-preserving denoising)
      4. Illumination correction (Gaussian background)
      5. Global histogram equalization
      6. Segmentation (returns mask-separated)
      7. Morphological operations

    Args:
        image: BGR image
        do_bilateral: Apply bilateral filter
        do_rotation_correction: Apply auto rotation correction
        do_shadow_removal: Apply shadow removal
        do_illumination: Apply illumination correction
        do_hist_equal: Apply global histogram equalization
        do_segmentation: Apply tire region segmentation
        do_morphology: Morphological operation name or None

    Returns:
        Enhanced BGR image
    """
    result = image.copy()

    if do_rotation_correction:
        result = auto_rotate_correction(result)
        logger.debug("Applied auto rotation correction")

    if do_shadow_removal:
        result = remove_shadows(result)
        logger.debug("Applied shadow removal")

    if do_bilateral:
        result = bilateral_filter(result)
        logger.debug("Applied bilateral filtering")

    if do_illumination:
        result = correct_illumination(result)
        logger.debug("Applied illumination correction")

    if do_hist_equal:
        result = global_histogram_equalization(result)
        logger.debug("Applied global histogram equalization")

    if do_segmentation:
        _, result = segment_tire_region(result)
        logger.debug("Applied tire region segmentation")

    if do_morphology:
        result = morphological_operations(result, operation=do_morphology)
        logger.debug(f"Applied morphological operation: {do_morphology}")

    return result


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG)
    test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    out = apply_all_image_enhancements(test_img)
    logger.info(f"Output shape: {out.shape}, dtype: {out.dtype}")
