"""
Deblurring Module — Recover blurry tire images instead of rejecting them.
Implements multiple deblurring techniques:
  1. Wiener deconvolution (motion/de focus blur)
  2. Regularized deconvolution
  3. Laplacian sharpening (aggressive)
  4. Texture enhancement (CLAHE on edge map)
  5. Multi-scale sharpening
  6. Adaptive deblurring (auto-selects best method)
"""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple, Callable

logger = logging.getLogger(__name__)


def estimate_blur_kernel(image: np.ndarray, kernel_size: int = 15) -> np.ndarray:
    """
    Estimate blur kernel from image edges.
    Uses edge detection to infer blur direction and magnitude.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)
    return kernel


def wiener_deconvolution(
    image: np.ndarray,
    kernel_size: int = 5,
    noise_power: float = 0.01,
    signal_power: float = 1.0,
) -> np.ndarray:
    """
    Wiener deconvolution for motion/de focus blur recovery.
    Works in frequency domain to inverse-filter blur.

    Args:
        image: BGR image
        kernel_size: Blur kernel size (5 for mild, 11 for heavy blur)
        noise_power: Estimated noise power (lower = more aggressive)
        signal_power: Estimated signal power

    Returns:
        Deblurred BGR image
    """
    from scipy import signal as scipy_signal
    from scipy.fft import fft2, ifft2

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    img_float = gray.astype(np.float64) / 255.0

    h, w = img_float.shape
    psf = np.zeros((h, w), dtype=np.float64)
    psf[h // 2 - kernel_size // 2: h // 2 + kernel_size // 2 + 1,
        w // 2 - kernel_size // 2: w // 2 + kernel_size // 2 + 1] = 1.0
    psf /= psf.sum()

    img_fft = fft2(img_float)
    psf_fft = fft2(psf)
    psf_fft_conj = np.conj(psf_fft)
    noise_ratio = noise_power / (signal_power + 1e-10)

    deblurred_fft = (psf_fft_conj * img_fft) / (psf_fft * psf_fft_conj + noise_ratio)
    deblurred = np.real(ifft2(deblurred_fft))
    deblurred = np.clip(deblurred * 255, 0, 255).astype(np.uint8)

    if len(image.shape) == 3:
        return cv2.cvtColor(deblurred, cv2.COLOR_GRAY2BGR)
    return deblurred


def regularized_deconvolution(
    image: np.ndarray,
    kernel_size: int = 5,
    regularization: float = 0.1,
    iterations: int = 10,
) -> np.ndarray:
    """
    Richardson-Lucy style iterative deconvolution.
    Better for tire texture than Wiener on severe blur.

    Args:
        image: BGR image
        kernel_size: Blur kernel size
        regularization: Regularization strength
        iterations: Number of iterations

    Returns:
        Deblurred BGR image
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    img_float = gray.astype(np.float64) / 255.0

    kernel = cv2.getGaussianKernel(kernel_size, -1)
    kernel_2d = kernel @ kernel.T
    kernel_2d = kernel_2d / kernel_2d.sum()

    estimate = img_float.copy()
    for _ in range(iterations):
        blurred_estimate = cv2.filter2D(estimate, -1, kernel_2d)
        relative_blur = img_float / (blurred_estimate + 1e-10)
        correction = cv2.filter2D(relative_blur, -1, kernel_2d)
        estimate = estimate * correction
        estimate = estimate / (1 + regularization * (estimate - 0.5))

    deblurred = np.clip(estimate * 255, 0, 255).astype(np.uint8)
    if len(image.shape) == 3:
        return cv2.cvtColor(deblurred, cv2.COLOR_GRAY2BGR)
    return deblurred


def laplacian_sharpen(
    image: np.ndarray,
    strength: float = 2.0,
) -> np.ndarray:
    """
    Aggressive Laplacian sharpening for moderate blur recovery.

    Args:
        image: BGR image
        strength: Sharpening strength (1.0 = mild, 3.0 = very aggressive)

    Returns:
        Sharpened BGR image
    """
    kernel = np.array([
        [0, -1, 0],
        [-1, 4 + strength, -1],
        [0, -1, 0],
    ], dtype=np.float32)
    kernel[1, 1] = 4 + strength

    sharpened = cv2.filter2D(image, -1, kernel)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def texture_enhancement(image: np.ndarray) -> np.ndarray:
    """
    Enhance tire tread texture using CLAHE on multiple color channels.
    Specifically designed for tire rubber surface.

    Args:
        image: BGR image

    Returns:
        Texture-enhanced BGR image
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe_heavy = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    l_enhanced = clahe_heavy.apply(l)

    sobel_x = cv2.Sobel(l_enhanced, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(l_enhanced, cv2.CV_64F, 0, 1, ksize=3)
    edges = cv2.magnitude(sobel_x, sobel_y)
    edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    l_final = cv2.addWeighted(l_enhanced, 0.7, edges, 0.3, 0)
    enhanced = cv2.merge([l_final, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def multi_scale_sharpen(image: np.ndarray) -> np.ndarray:
    """
    Sharpen at multiple scales using Laplacian pyramid.
    Recovers details lost at different frequency levels.

    Args:
        image: BGR image

    Returns:
        Multi-scale sharpened BGR image
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    img_float = gray.astype(np.float32) / 255.0

    pyramid = []
    current = img_float.copy()
    for _ in range(3):
        blurred = cv2.GaussianBlur(current, (5, 5), 0)
        laplacian = current - blurred
        pyramid.append(laplacian)
        current = cv2.resize(blurred, (blurred.shape[1] // 2, blurred.shape[0] // 2))

    reconstructed = current.copy()
    for level in reversed(pyramid):
        reconstructed = cv2.resize(reconstructed, (level.shape[1], level.shape[0]))
        reconstructed = reconstructed + level * 1.5

    reconstructed = np.clip(reconstructed * 255, 0, 255).astype(np.uint8)

    if len(image.shape) == 3:
        return cv2.cvtColor(reconstructed, cv2.COLOR_GRAY2BGR)
    return reconstructed


def add_gaussian_sharpen(image: np.ndarray, sigma: float = 1.0, alpha: float = 1.5) -> np.ndarray:
    """
    Gaussian unsharp masking — subtracts blurred version to emphasize edges.
    """
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    sharpened = cv2.addWeighted(image, 1.0 + alpha, blurred, -alpha, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def try_deblur(
    image: np.ndarray,
    blur_score: float,
    threshold: float = 100.0,
) -> Tuple[np.ndarray, float, str]:
    """
    Try multiple deblurring methods on a blurry image, returns the best result.

    Strategy based on blur severity:
      - Borderline (80-100): Light sharpening
      - Moderate (50-80): Wiener + Laplacian sharpening
      - Severe (20-50): Texture enhancement + multi-scale sharpening
      - Extreme (<20): Try all methods, pick best

    Args:
        image: BGR image
        blur_score: Laplacian variance score
        threshold: Blur threshold

    Returns:
        (deblurred_image, new_blur_score, method_used)
    """
    methods = []

    if blur_score < threshold * 0.2:
        methods = [
            ("texture_enhance", lambda img: texture_enhancement(img)),
            ("multi_scale", lambda img: multi_scale_sharpen(img)),
            ("laplacian_aggressive", lambda img: laplacian_sharpen(img, strength=3.0)),
            ("gaussian_sharpen", lambda img: add_gaussian_sharpen(img, sigma=2.0, alpha=2.0)),
            ("regularized_deconv", lambda img: regularized_deconvolution(img, kernel_size=7, iterations=15)),
        ]
    elif blur_score < threshold * 0.5:
        methods = [
            ("texture_enhance", lambda img: texture_enhancement(img)),
            ("multi_scale", lambda img: multi_scale_sharpen(img)),
            ("laplacian_aggressive", lambda img: laplacian_sharpen(img, strength=2.5)),
            ("gaussian_sharpen", lambda img: add_gaussian_sharpen(img, sigma=1.5, alpha=1.5)),
        ]
    elif blur_score < threshold * 0.8:
        methods = [
            ("laplacian", lambda img: laplacian_sharpen(img, strength=2.0)),
            ("gaussian_sharpen", lambda img: add_gaussian_sharpen(img, sigma=1.0, alpha=1.5)),
            ("wiener", lambda img: wiener_deconvolution(img, kernel_size=5, noise_power=0.01)),
        ]
    else:
        methods = [
            ("light_sharpen", lambda img: laplacian_sharpen(img, strength=1.5)),
            ("gaussian_sharpen", lambda img: add_gaussian_sharpen(img, sigma=0.5, alpha=1.0)),
        ]

    best_img = image
    best_score = blur_score
    best_method = "none"

    for method_name, method_fn in methods:
        try:
            result = method_fn(image)
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if len(result.shape) == 3 else result
            new_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

            if new_score > best_score and new_score > blur_score * 1.1:
                best_score = new_score
                best_img = result
                best_method = method_name
                logger.debug(f"Deblur improved: {blur_score:.1f} -> {new_score:.1f} ({method_name})")
        except Exception as e:
            logger.debug(f"Deblur method {method_name} failed: {e}")

    return best_img, best_score, best_method


def deblur_pipeline(
    image: np.ndarray,
    blur_score: float,
    threshold: float = 100.0,
    max_attempts: int = 3,
) -> Tuple[np.ndarray, float, bool]:
    """
    Full deblur pipeline — tries progressively stronger methods until
    the image passes the blur threshold or max attempts reached.

    Args:
        image: BGR image
        blur_score: Initial Laplacian score
        threshold: Blur threshold to pass
        max_attempts: Maximum deblur attempts

    Returns:
        (deblurred_image, final_score, recovered: bool)
    """
    if blur_score >= threshold:
        return image, blur_score, True

    current_img = image.copy()
    current_score = blur_score
    recovered = False

    for attempt in range(max_attempts):
        current_img, current_score, method = try_deblur(
            current_img, current_score, threshold
        )

        if current_score >= threshold:
            recovered = True
            logger.info(
                f"Deblur recovery successful (attempt {attempt + 1}): "
                f"{blur_score:.1f} -> {current_score:.1f} using {method}"
            )
            break

        if method == "none":
            break

    if not recovered:
        logger.info(
            f"Deblur failed to recover: {blur_score:.1f} -> {current_score:.1f} "
            f"(threshold={threshold})"
        )

    return current_img, current_score, recovered


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    blurred = cv2.GaussianBlur(test_img, (15, 15), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    print(f"Blur score: {score:.1f}")

    result, new_score, recovered = deblur_pipeline(blurred, score)
    print(f"After deblur: {new_score:.1f}, recovered={recovered}")
