"""Lightweight aircraft image preprocessing and degradation utilities.

The preprocessing pipeline is intentionally conservative:
- estimate blur and contrast first
- denoise mildly only when needed
- apply CLAHE only on luminance
- sharpen with a small, edge-preserving amount
- optionally apply a lightweight deblurring approximation

The module returns both the original and enhanced image so the caller can
choose original-only, enhanced-only, or fused behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


@dataclass
class AircraftEnhancementConfig:
    target_long_side: int = 768
    keep_aspect_ratio: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)
    blur_var_threshold: float = 100.0
    low_contrast_std_threshold: float = 32.0
    denoise_strength: float = 3.0
    sharpen_min: float = 0.10
    sharpen_max: float = 0.45
    enable_deblur: bool = False
    deblur_threshold: float = 0.55
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass
class PreprocessResult:
    original_image: np.ndarray
    enhanced_image: np.ndarray
    normalized_image: np.ndarray
    blur_variance: float
    contrast_std: float
    blur_severity: float
    contrast_severity: float
    applied_deblur: bool
    scale_factor: float


class AircraftPreprocessor:
    """Quality-aware preprocessing for degraded aircraft images."""

    def __init__(self, config: Optional[AircraftEnhancementConfig] = None):
        self.config = config or AircraftEnhancementConfig()

    @staticmethod
    def _ensure_bgr(image: np.ndarray) -> np.ndarray:
        if image is None:
            raise ValueError("image is None")
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a numpy array")
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        if image.ndim == 3 and image.shape[2] == 3:
            return image
        raise ValueError(f"Unsupported image shape: {image.shape}")

    def resize_preserve_aspect(self, image: np.ndarray, target_long_side: Optional[int] = None) -> Tuple[np.ndarray, float]:
        image = self._ensure_bgr(image)
        long_side = int(target_long_side or self.config.target_long_side)
        if long_side <= 0:
            raise ValueError("target_long_side must be positive")

        height, width = image.shape[:2]
        current_long = max(height, width)
        if current_long <= long_side:
            return image.copy(), 1.0

        scale = long_side / float(current_long)
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return resized, scale

    def estimate_blur(self, image: np.ndarray) -> Tuple[float, float]:
        image = self._ensure_bgr(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        threshold = max(self.config.blur_var_threshold, 1e-6)
        blur_severity = float(np.clip(1.0 - (variance / threshold), 0.0, 1.0))
        return variance, blur_severity

    def estimate_contrast(self, image: np.ndarray) -> Tuple[float, float]:
        image = self._ensure_bgr(image)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lightness = lab[:, :, 0]
        contrast_std = float(lightness.std())
        threshold = max(self.config.low_contrast_std_threshold, 1e-6)
        contrast_severity = float(np.clip(1.0 - (contrast_std / threshold), 0.0, 1.0))
        return contrast_std, contrast_severity

    def apply_mild_denoise(self, image: np.ndarray, blur_severity: float, contrast_severity: float) -> np.ndarray:
        image = self._ensure_bgr(image)
        strength = self.config.denoise_strength
        strength *= 1.0 + 0.35 * max(blur_severity, contrast_severity)
        strength = float(np.clip(strength, 1.0, 6.0))

        if image.ndim == 3 and image.shape[2] == 3:
            return cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, None, strength, 7, 21)
        return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)

    def apply_clahe_luminance(self, image: np.ndarray, contrast_severity: float, blur_severity: float = 0.0) -> np.ndarray:
        image = self._ensure_bgr(image)
        clip_limit = self.config.clahe_clip_limit
        clip_limit *= 1.0 + 0.7 * contrast_severity
        clip_limit *= 1.0 - 0.15 * blur_severity
        clip_limit = float(np.clip(clip_limit, 1.0, 4.0))

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=self.config.clahe_tile_grid_size)
        l_enhanced = clahe.apply(l_channel)
        merged = cv2.merge((l_enhanced, a_channel, b_channel))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    def apply_edge_preserving_sharpen(self, image: np.ndarray, blur_severity: float) -> np.ndarray:
        image = self._ensure_bgr(image)
        strength = self.config.sharpen_min + (self.config.sharpen_max - self.config.sharpen_min) * blur_severity
        strength = float(np.clip(strength, self.config.sharpen_min, self.config.sharpen_max))

        blurred = cv2.GaussianBlur(image, (0, 0), 1.2)
        sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)

        try:
            sharpened = cv2.bilateralFilter(sharpened, d=5, sigmaColor=24, sigmaSpace=24)
        except Exception:
            pass

        return sharpened

    def apply_optional_deblur(self, image: np.ndarray, blur_severity: float) -> np.ndarray:
        image = self._ensure_bgr(image)
        if not self.config.enable_deblur or blur_severity < self.config.deblur_threshold:
            return image

        kernels = [
            np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32),
            np.array([[0, -0.5, 0], [-0.5, 3.0, -0.5], [0, -0.5, 0]], dtype=np.float32),
        ]

        best_image = image
        best_score = self.estimate_blur(image)[0]
        for kernel in kernels:
            candidate = cv2.filter2D(image, -1, kernel)
            candidate_score = self.estimate_blur(candidate)[0]
            if candidate_score > best_score:
                best_image = candidate
                best_score = candidate_score

        return best_image

    def normalize_for_cnn(self, image: np.ndarray) -> np.ndarray:
        image = self._ensure_bgr(image).astype(np.float32) / 255.0
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mean = np.array(self.config.normalize_mean, dtype=np.float32).reshape(1, 1, 3)
        std = np.array(self.config.normalize_std, dtype=np.float32).reshape(1, 1, 3)
        normalized = (rgb - mean) / std
        return np.transpose(normalized, (2, 0, 1)).astype(np.float32)

    def preprocess(
        self,
        image: np.ndarray,
        target_long_side: Optional[int] = None,
        enable_deblur: Optional[bool] = None,
    ) -> PreprocessResult:
        """Run the full enhancement pipeline and return both original and enhanced outputs."""
        original = self._ensure_bgr(image)
        resized, scale = self.resize_preserve_aspect(original, target_long_side)
        blur_variance, blur_severity = self.estimate_blur(resized)
        contrast_std, contrast_severity = self.estimate_contrast(resized)

        denoised = self.apply_mild_denoise(resized, blur_severity, contrast_severity)
        clahe = self.apply_clahe_luminance(denoised, contrast_severity, blur_severity)
        sharpened = self.apply_edge_preserving_sharpen(clahe, blur_severity)

        original_deblur_flag = self.config.enable_deblur if enable_deblur is None else bool(enable_deblur)
        self.config.enable_deblur = original_deblur_flag
        enhanced = self.apply_optional_deblur(sharpened, blur_severity)
        normalized = self.normalize_for_cnn(enhanced)

        return PreprocessResult(
            original_image=original,
            enhanced_image=enhanced,
            normalized_image=normalized,
            blur_variance=blur_variance,
            contrast_std=contrast_std,
            blur_severity=blur_severity,
            contrast_severity=contrast_severity,
            applied_deblur=bool(original_deblur_flag and blur_severity >= self.config.deblur_threshold),
            scale_factor=scale,
        )

    def enhance_for_generalist(self, image: np.ndarray) -> PreprocessResult:
        """Generalist-stage enhancement is conservative and quality-aware."""
        return self.preprocess(image, target_long_side=self.config.target_long_side)

    def enhance_for_roi(self, image: np.ndarray) -> PreprocessResult:
        """ROI enhancement can be slightly stronger because the crop is already localized."""
        config = self.config
        roi_preprocessor = AircraftPreprocessor(
            AircraftEnhancementConfig(
                target_long_side=config.target_long_side,
                keep_aspect_ratio=config.keep_aspect_ratio,
                clahe_clip_limit=min(3.0, config.clahe_clip_limit + 0.4),
                clahe_tile_grid_size=config.clahe_tile_grid_size,
                blur_var_threshold=max(60.0, config.blur_var_threshold * 0.75),
                low_contrast_std_threshold=max(24.0, config.low_contrast_std_threshold * 0.9),
                denoise_strength=min(5.0, config.denoise_strength + 0.5),
                sharpen_min=config.sharpen_min,
                sharpen_max=min(0.55, config.sharpen_max + 0.1),
                enable_deblur=config.enable_deblur,
                deblur_threshold=config.deblur_threshold,
                normalize_mean=config.normalize_mean,
                normalize_std=config.normalize_std,
            )
        )
        return roi_preprocessor.preprocess(image, target_long_side=config.target_long_side)


class AircraftDegradationAugmentor:
    """Synthetic degradations for training robustness."""

    @staticmethod
    def _ensure_bgr(image: np.ndarray) -> np.ndarray:
        return AircraftPreprocessor._ensure_bgr(image)

    @staticmethod
    def gaussian_blur(image: np.ndarray, ksize: int = 7, sigma: float = 0.0) -> np.ndarray:
        image = AircraftDegradationAugmentor._ensure_bgr(image)
        ksize = max(3, int(ksize) | 1)
        return cv2.GaussianBlur(image, (ksize, ksize), sigma)

    @staticmethod
    def motion_blur(image: np.ndarray, kernel_size: int = 9, angle: float = 0.0) -> np.ndarray:
        image = AircraftDegradationAugmentor._ensure_bgr(image)
        kernel_size = max(3, int(kernel_size) | 1)
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        kernel[kernel_size // 2, :] = 1.0
        center = (kernel_size / 2.0 - 0.5, kernel_size / 2.0 - 0.5)
        rotation = cv2.getRotationMatrix2D(center, angle, 1.0)
        kernel = cv2.warpAffine(kernel, rotation, (kernel_size, kernel_size))
        kernel /= max(kernel.sum(), 1e-6)
        return cv2.filter2D(image, -1, kernel)

    @staticmethod
    def jpeg_compression(image: np.ndarray, quality: int = 40) -> np.ndarray:
        image = AircraftDegradationAugmentor._ensure_bgr(image)
        quality = int(np.clip(quality, 5, 100))
        ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            return image
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        return image if decoded is None else decoded

    @staticmethod
    def fog_haze(image: np.ndarray, strength: float = 0.35) -> np.ndarray:
        image = AircraftDegradationAugmentor._ensure_bgr(image).astype(np.float32)
        strength = float(np.clip(strength, 0.05, 0.75))
        haze = np.full_like(image, 255.0)
        foggy = image * (1.0 - strength) + haze * strength
        foggy = cv2.GaussianBlur(foggy, (0, 0), 2.0)
        return np.clip(foggy, 0, 255).astype(np.uint8)

    @staticmethod
    def low_light(image: np.ndarray, gamma: float = 1.8, brightness: int = -25) -> np.ndarray:
        image = AircraftDegradationAugmentor._ensure_bgr(image)
        table = np.array([(i / 255.0) ** gamma * 255 for i in range(256)]).astype("uint8")
        adjusted = cv2.LUT(image, table)
        adjusted = cv2.convertScaleAbs(adjusted, alpha=1.0, beta=brightness)
        return adjusted

    @staticmethod
    def sensor_noise(image: np.ndarray, sigma: float = 12.0) -> np.ndarray:
        image = AircraftDegradationAugmentor._ensure_bgr(image).astype(np.float32)
        noise = np.random.normal(0.0, sigma, image.shape).astype(np.float32)
        noisy = image + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

    @staticmethod
    def downsample_upscale(image: np.ndarray, scale: float = 0.6) -> np.ndarray:
        image = AircraftDegradationAugmentor._ensure_bgr(image)
        height, width = image.shape[:2]
        scale = float(np.clip(scale, 0.25, 0.95))
        small = cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def partial_blur(image: np.ndarray, frac: float = 0.45) -> np.ndarray:
        image = AircraftDegradationAugmentor._ensure_bgr(image)
        height, width = image.shape[:2]
        frac = float(np.clip(frac, 0.15, 0.85))
        blurred = cv2.GaussianBlur(image, (11, 11), 0)
        mask = np.zeros((height, width), dtype=np.float32)
        split = int(width * frac)
        mask[:, :split] = 1.0
        mask = cv2.GaussianBlur(mask, (0, 0), 19)
        mask_3 = cv2.merge([mask, mask, mask])
        return np.clip(image * (1.0 - mask_3) + blurred * mask_3, 0, 255).astype(np.uint8)

    @staticmethod
    def atmospheric_distortion(image: np.ndarray) -> np.ndarray:
        image = AircraftDegradationAugmentor._ensure_bgr(image).astype(np.float32)
        height, width = image.shape[:2]
        x = np.linspace(0, 1, width, dtype=np.float32)
        y = np.linspace(0, 1, height, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        ripple = 0.08 * np.sin(8 * np.pi * xx) * np.cos(6 * np.pi * yy)
        distorted = image * (1.0 - 0.08) + 255.0 * 0.08
        distorted[..., 0] += 255.0 * ripple
        distorted[..., 1] += 255.0 * ripple * 0.6
        distorted[..., 2] += 255.0 * ripple * 0.4
        return np.clip(cv2.GaussianBlur(distorted, (0, 0), 1.2), 0, 255).astype(np.uint8)

    @classmethod
    def generate_training_variants(cls, image: np.ndarray) -> Dict[str, np.ndarray]:
        image = cls._ensure_bgr(image)
        return {
            "gaussian_blur": cls.gaussian_blur(image, ksize=7),
            "motion_blur": cls.motion_blur(image, kernel_size=9, angle=15.0),
            "jpeg": cls.jpeg_compression(image, quality=35),
            "fog": cls.fog_haze(image, strength=0.35),
            "low_light": cls.low_light(image),
            "sensor_noise": cls.sensor_noise(image, sigma=10.0),
            "down_up": cls.downsample_upscale(image, scale=0.6),
            "partial_blur": cls.partial_blur(image, frac=0.45),
            "atmospheric": cls.atmospheric_distortion(image),
        }


def load_and_preprocess_image(image_path: str, config: Optional[AircraftEnhancementConfig] = None) -> PreprocessResult:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    return AircraftPreprocessor(config).preprocess(image)
