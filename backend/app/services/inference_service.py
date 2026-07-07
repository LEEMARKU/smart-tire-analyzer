"""
Inference service for Smart Tire Analyzer.

This service serves only the trained PyTorch hybrid artifact. When no accepted
hybrid checkpoint is available, inference remains unavailable instead of
silently returning synthetic predictions.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np

from ai_model.ann.output_heads import denormalize_outputs
from app.config import settings
from app.ws_manager import manager

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
HYBRID_MODEL_DIR = PROJECT_ROOT / "ai_model" / "saved_models" / "hybrid_torch"
HYBRID_ACCEPTED_MODEL_PATH = HYBRID_MODEL_DIR / "model_best.pt"
HYBRID_LAST_MODEL_PATH = HYBRID_MODEL_DIR / "model_last.pt"
HYBRID_METADATA_PATH = HYBRID_MODEL_DIR / "metadata.json"


def _infer_wear_pattern_from_depths(depths: list[float]) -> str:
    """
    Determine tire wear pattern from four depth measurements
    (inner, near-center, far-center, outer) measured in mm.

    Side-wall wear = significantly higher wear on the outer edge compared to
    the inner edge, i.e. outer depth >> inner depth.
    """
    depths_arr = np.asarray(depths, dtype=np.float32)

    diff = depths_arr.max() - depths_arr.min()
    inner = depths_arr[0]
    outer = depths_arr[-1]
    center_avg = depths_arr[1:3].mean()

    side_wall_ratio = 1.5
    min_diff = 0.8

    if outer - inner > min_diff and outer / max(inner, 0.01) > side_wall_ratio:
        return "side_wall_wear"
    if (inner - outer) > 1.5:
        return "one_side_wear"

    try:
        from ai_model.rnn.temporal_features import classify_wear_pattern_from_depths

        return str(classify_wear_pattern_from_depths(depths))
    except Exception as exc:
        logger.debug("Wear pattern classifier unavailable, using fallback: %s", exc)

    if diff < 0.5:
        return "uniform_wear"
    if center_avg < outer - 0.8:
        return "center_wear"
    if inner < center_avg - 0.8:
        return "edge_wear"
    if diff > 2.0:
        return "patchy_wear"
    return "patchy_wear"


def _confidence_from_depths(depths: list[float]) -> float:
    try:
        from ai_model.rnn.temporal_features import compute_confidence_from_wear_pattern

        return float(compute_confidence_from_wear_pattern(depths))
    except Exception as exc:
        logger.debug("Confidence computation unavailable, using fallback: %s", exc)

    depth_array = np.clip(np.asarray(depths, dtype=np.float32), 0.0, 12.0)
    depth_std = float(np.std(depth_array))
    depth_range = float(np.max(depth_array) - np.min(depth_array))
    confidence = 1.0 - (depth_std / 6.0) - (depth_range / 12.0) * 0.3
    return float(np.clip(confidence, 0.1, 1.0))


def _fit_last_dim(array: np.ndarray, target_dim: int) -> np.ndarray:
    """Pad or truncate the final feature axis to match a loaded model input."""
    values = np.asarray(array, dtype=np.float32)
    current_dim = int(values.shape[-1])
    if current_dim == target_dim:
        return cast(np.ndarray, values)
    if current_dim > target_dim:
        return cast(np.ndarray, values[..., :target_dim].astype(np.float32, copy=False))

    pad_width = [(0, 0)] * values.ndim
    pad_width[-1] = (0, target_dim - current_dim)
    return cast(np.ndarray, np.pad(values, pad_width, mode="constant").astype(np.float32, copy=False))


def _context_to_vector(context_data: dict[str, Any] | None, target_dim: int = 64) -> np.ndarray:
    """Convert road/weather context dictionaries into a stable numeric vector."""
    context = context_data or {}
    vector = np.zeros(target_dim, dtype=np.float32)

    numeric_values = [
        1.0 if context.get("rain_detected") else 0.0,
        float(context.get("temperature_c") or 24.0) / 60.0,
        float(context.get("humidity_pct") or 0.0) / 100.0,
        float(context.get("visibility_km") or 10.0) / 10.0,
        float(context.get("road_wear_multiplier") or 1.0),
        float(context.get("weather_risk_multiplier") or 1.0),
        float(context.get("elevation_m") or 0.0) / 3000.0,
        (float(context.get("latitude") or 0.0) + 90.0) / 180.0,
        (float(context.get("longitude") or 0.0) + 180.0) / 360.0,
    ]
    for index, value in enumerate(numeric_values[:target_dim]):
        vector[index] = float(np.clip(value, -3.0, 3.0))

    categorical_slots = {
        12: ("terrain_type", ["flat_urban", "rolling_suburban", "hilly", "mountainous"]),
        20: ("road_condition", ["excellent", "good", "fair", "poor"]),
        28: ("traffic_density", ["light", "moderate", "heavy"]),
        36: ("weather_condition", ["Clear", "Clouds", "Rain", "Drizzle", "Thunderstorm", "Snow"]),
    }
    for start, (key, labels) in categorical_slots.items():
        if start >= target_dim:
            continue
        categorical_value = str(context.get(key, "")).lower()
        for offset, label in enumerate(labels):
            slot = start + offset
            if slot >= target_dim:
                break
            vector[slot] = 1.0 if categorical_value == label.lower() else 0.0

    return cast(np.ndarray, vector)


class InferenceService:
    """Manages model loading and inference execution."""

    def __init__(self) -> None:
        self._hybrid_model: Any = None
        self._hybrid_metadata: dict[str, Any] = {}
        self._hybrid_device = "cpu"
        self._hybrid_checkpoint_mtime: float | None = None
        self._hybrid_checkpoint_path: Path | None = None
        self._hybrid_metadata_path: Path | None = None
        self._ready = False
        self._model_version = "unknown"
        self._model_source = "not_loaded"
        self._model_checkpoint = str(HYBRID_ACCEPTED_MODEL_PATH)
        self._load_error: str | None = None
        self._tabular_bundle: tuple[Any, dict[str, Any], str] | None = None
        self._tabular_checked = False

    async def initialize(self) -> None:
        """Load the best available model without blocking the event loop."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_model)
        if not self.is_ready():
            raise RuntimeError(
                f"Trained hybrid model could not be loaded: {self._load_error or 'unknown error'}"
            )
        await loop.run_in_executor(None, self._sanity_check)

    def _load_model(self) -> None:
        """Load the best available trained model."""
        try:
            from ai_model.hybrid_torch.runtime import load_hybrid_model

            artifacts = self._resolve_hybrid_artifacts()
            if artifacts is None:
                self._load_error = (
                    f"Missing hybrid checkpoint: {HYBRID_ACCEPTED_MODEL_PATH} or "
                    f"{HYBRID_LAST_MODEL_PATH}"
                )
                hybrid_bundle = None
            else:
                checkpoint_path, metadata_path, checkpoint_status = artifacts
                hybrid_bundle = load_hybrid_model(checkpoint_path, metadata_path)
            if hybrid_bundle is not None:
                self._hybrid_model, self._hybrid_metadata, self._hybrid_device = hybrid_bundle
                self._hybrid_checkpoint_path = checkpoint_path
                self._hybrid_metadata_path = metadata_path
                self._hybrid_checkpoint_mtime = checkpoint_path.stat().st_mtime
                self._hybrid_metadata["runtime_checkpoint_status"] = checkpoint_status
                self._model_version = str(
                    self._hybrid_metadata.get(
                        "model_version",
                        "pytorch_hybrid:efficientnetv2_b0_vit_b16_bilstm_tcn_attention_calibrated",
                    )
                )
                self._model_source = (
                    "pytorch_hybrid"
                    if checkpoint_status == "accepted"
                    else "pytorch_hybrid_unaccepted_last"
                )
                self._model_checkpoint = str(checkpoint_path)
                self._load_error = None
                self._ready = True
                if checkpoint_status != "accepted":
                    logger.warning(
                        "Loaded unaccepted hybrid fallback checkpoint: %s",
                        checkpoint_path,
                    )
                else:
                    logger.info("Loaded accepted PyTorch hybrid model: %s", checkpoint_path)
                return
        except Exception as exc:
            self._load_error = str(exc)
            logger.warning("PyTorch hybrid model load failed: %s", exc)

        logger.error("No trained hybrid model could be loaded; inference is not ready.")
        self._model_version = "not_loaded"
        self._model_source = "not_loaded"
        self._hybrid_checkpoint_mtime = None
        self._hybrid_checkpoint_path = None
        self._hybrid_metadata_path = None
        self._ready = False

    def is_ready(self) -> bool:
        return (
            self._ready
            and self._hybrid_model is not None
            and self._hybrid_checkpoint_path is not None
            and self._hybrid_checkpoint_path.exists()
        )

    def _resolve_hybrid_artifacts(self) -> tuple[Path, Path, str] | None:
        """Prefer accepted weights, then use the latest trained fallback checkpoint."""
        if HYBRID_ACCEPTED_MODEL_PATH.exists():
            return HYBRID_ACCEPTED_MODEL_PATH, HYBRID_METADATA_PATH, "accepted"
        if not HYBRID_LAST_MODEL_PATH.exists():
            return None
        return HYBRID_LAST_MODEL_PATH, HYBRID_METADATA_PATH, "unaccepted_last"

    def _sanity_check(self) -> None:
        """Validate the loaded runtime can produce a report-shaped prediction."""
        try:
            if self._hybrid_model is None:
                raise RuntimeError("hybrid model is not loaded")

            dummy_image = np.zeros((64, 64, 3), dtype=np.uint8)
            encoded_ok, encoded = cv2.imencode(".jpg", dummy_image)
            if not encoded_ok:
                raise RuntimeError("could not encode sanity-check dummy image")
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if decoded is None:
                raise RuntimeError("could not decode sanity-check dummy image")

            sample_depths = [5.0, 5.0, 5.0, 5.0]
            depth_label = _infer_wear_pattern_from_depths(sample_depths)
            depth_confidence = _confidence_from_depths(sample_depths)
            if not isinstance(depth_label, str):
                raise AssertionError("depth-derived wear_pattern is not a string")
            if not 0.1 <= float(depth_confidence) <= 1.0:
                raise AssertionError("depth-derived confidence is outside [0.1, 1.0]")

            from ai_model.hybrid_torch.constants import TREAD_SEQUENCE_DIM
            from ai_model.rnn.sequence_builder import build_tread_sequence

            tread_sequence = build_tread_sequence(sample_depths, target_dim=TREAD_SEQUENCE_DIM)
            prediction = self._hybrid_infer(decoded, tread_sequence)
            wear = prediction.get("wear_pattern", {})
            wear_pattern = wear.get("label") if isinstance(wear, dict) else wear
            confidence = prediction.get("confidence", wear.get("confidence") if isinstance(wear, dict) else None)

            if not isinstance(wear_pattern, str):
                raise AssertionError("sanity-check wear_pattern is not a string")
            if not isinstance(confidence, (int, float)) or not 0.1 <= float(confidence) <= 1.0:
                raise AssertionError("sanity-check confidence is outside [0.1, 1.0]")

            logger.info(
                "Inference sanity check passed: wear_pattern=%s confidence=%.4f",
                wear_pattern,
                float(confidence),
            )
        except Exception as exc:
            logger.exception("Inference sanity check failed: %s", exc)
            raise RuntimeError(f"Inference sanity check failed: {exc}") from exc

    async def predict(
        self,
        image_bytes: bytes,
        session_id: str,
        context_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run the full inference pipeline on raw image bytes with WebSocket progress updates."""
        loop = asyncio.get_event_loop()

        await loop.run_in_executor(None, self._reload_model_if_changed)

        checkpoint_path = self._hybrid_checkpoint_path
        if self._hybrid_model is None or checkpoint_path is None or not checkpoint_path.exists():
            self._hybrid_model = None
            self._ready = False
            self._model_version = "not_loaded"
            self._model_source = "not_loaded"
            self._load_error = "Missing hybrid checkpoint"
            return {
                "rejected": True,
                "reason": "Trained hybrid model is unavailable",
                "model_version": self._model_version,
                "source": self._model_source,
            }

        # ── Stage 1: Preprocessing ────────────────────────────────────────────
        await self._try_broadcast({"type": "progress", "session_id": session_id, "step": "preprocessing", "progress": 0.10})

        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return {"rejected": True, "reason": "Could not decode image"}

        from ai_model.cnn.preprocessing import detect_blur, run_preprocessing_pipeline
        blur_threshold = float(settings.BLUR_THRESHOLD)
        is_blurry, blur_score = detect_blur(image, threshold=blur_threshold)
        if is_blurry:
            return {"rejected": True, "blur_score": blur_score, "blur_threshold": blur_threshold, "reason": "Image too blurry"}

        processed = run_preprocessing_pipeline(image, training=False, include_edge_channel=True, blur_threshold=blur_threshold)
        if processed is None:
            return {"rejected": True, "blur_score": blur_score, "blur_threshold": blur_threshold, "reason": "Preprocessing failed"}

        from ai_model.hybrid_torch.runtime_tread import estimate_visual_tread_depths
        estimated_depths = estimate_visual_tread_depths(image)

        from ai_model.hybrid_torch.constants import TREAD_SEQUENCE_DIM
        from ai_model.rnn.sequence_builder import build_tread_sequence
        tread_seq = build_tread_sequence(estimated_depths, target_dim=TREAD_SEQUENCE_DIM)

        # ── Stage 2: CNN inference ────────────────────────────────────────────
        await self._try_broadcast({"type": "progress", "session_id": session_id, "step": "cnn_inference", "progress": 0.30})

        # ── Stage 3: ViT inference ────────────────────────────────────────────
        await self._try_broadcast({"type": "progress", "session_id": session_id, "step": "vit_inference", "progress": 0.50})

        # ── Stage 4: RNN inference ────────────────────────────────────────────
        await self._try_broadcast({"type": "progress", "session_id": session_id, "step": "rnn_inference", "progress": 0.60})

        # ── Stage 5: Fusion ──────────────────────────────────────────────────
        await self._try_broadcast({"type": "progress", "session_id": session_id, "step": "fusion", "progress": 0.70})
        prediction = await loop.run_in_executor(None, self._hybrid_infer, image, tread_seq)

        # ── Stage 6: Context fetch ───────────────────────────────────────────
        await self._try_broadcast({"type": "progress", "session_id": session_id, "step": "context_fetch", "progress": 0.80})

        from ai_model.hybrid_torch.runtime_tread import RUNTIME_TREAD_SEQUENCE_SOURCE

        prediction["tread_sequence_source"] = RUNTIME_TREAD_SEQUENCE_SOURCE
        prediction["runtime_tread_sequence_mm"] = {
            "position_1": round(float(estimated_depths[0]), 2),
            "position_2": round(float(estimated_depths[1]), 2),
            "position_3": round(float(estimated_depths[2]), 2),
            "position_4": round(float(estimated_depths[3]), 2),
            "average": round(float(np.mean(estimated_depths)), 2),
        }
        prediction["session_id"] = session_id
        prediction["blur_score"] = round(float(blur_score), 2)
        prediction["blur_threshold"] = blur_threshold
        prediction["model_version"] = self._model_version

        # ── Stage 7: Reasoning ───────────────────────────────────────────────
        await self._try_broadcast({"type": "progress", "session_id": session_id, "step": "reasoning", "progress": 0.90})
        self._attach_model_diagnostics(prediction, context_data)

        # ── Complete ─────────────────────────────────────────────────────────
        await self._try_broadcast({"type": "progress", "session_id": session_id, "step": "complete", "progress": 1.00, "payload": prediction})
        return prediction

    async def _try_broadcast(self, message: dict[str, Any]) -> None:
        try:
            await manager.broadcast(message)
        except Exception:
            pass

    def _predict_sync(
        self,
        image_bytes: bytes,
        session_id: str,
        context_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._reload_model_if_changed()
        checkpoint_path = self._hybrid_checkpoint_path
        if self._hybrid_model is None or checkpoint_path is None or not checkpoint_path.exists():
            self._hybrid_model = None
            self._ready = False
            self._model_version = "not_loaded"
            self._model_source = "not_loaded"
            self._load_error = "Missing hybrid checkpoint"
            return {
                "rejected": True,
                "reason": "Trained hybrid model is unavailable",
                "model_version": self._model_version,
                "source": self._model_source,
            }
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return {"rejected": True, "reason": "Could not decode image"}

        from ai_model.cnn.preprocessing import detect_blur, run_preprocessing_pipeline
        from ai_model.hybrid_torch.constants import TREAD_SEQUENCE_DIM
        from ai_model.hybrid_torch.runtime_tread import (
            RUNTIME_TREAD_SEQUENCE_SOURCE,
            estimate_visual_tread_depths,
        )
        from ai_model.rnn.sequence_builder import build_tread_sequence

        blur_threshold = float(settings.BLUR_THRESHOLD)
        is_blurry, blur_score = detect_blur(image, threshold=blur_threshold)
        if is_blurry:
            return {
                "rejected": True,
                "blur_score": blur_score,
                "blur_threshold": blur_threshold,
                "reason": "Image too blurry",
            }

        processed = run_preprocessing_pipeline(
            image,
            training=False,
            include_edge_channel=True,
            blur_threshold=blur_threshold,
        )
        if processed is None:
            return {
                "rejected": True,
                "blur_score": blur_score,
                "blur_threshold": blur_threshold,
                "reason": "Preprocessing failed",
            }

        estimated_depths = estimate_visual_tread_depths(image)
        tread_seq = build_tread_sequence(estimated_depths, target_dim=TREAD_SEQUENCE_DIM)

        prediction = self._hybrid_infer(image, tread_seq)

        prediction["tread_sequence_source"] = RUNTIME_TREAD_SEQUENCE_SOURCE
        prediction["runtime_tread_sequence_mm"] = {
            "position_1": round(float(estimated_depths[0]), 2),
            "position_2": round(float(estimated_depths[1]), 2),
            "position_3": round(float(estimated_depths[2]), 2),
            "position_4": round(float(estimated_depths[3]), 2),
            "average": round(float(np.mean(estimated_depths)), 2),
        }
        prediction["session_id"] = session_id
        prediction["blur_score"] = round(float(blur_score), 2)
        prediction["blur_threshold"] = blur_threshold
        prediction["model_version"] = self._model_version
        self._attach_model_diagnostics(prediction, context_data)
        return prediction

    def _estimate_depths_from_image(self, processed_image: np.ndarray) -> list[float]:
        """
        Estimate four tread depths from image texture in a deterministic way.

        The runtime still derives a deterministic four-point sequence estimate
        so the BiLSTM branch receives the same structured input shape as
        training.
        """
        if processed_image.shape[-1] >= 4:
            edge_channel = np.abs(processed_image[:, :, 3])
        else:
            edge_channel = np.abs(np.mean(processed_image[:, :, :3], axis=-1))

        bands = np.array_split(edge_channel, 4, axis=1)
        band_scores = np.asarray([float(np.mean(np.abs(band))) for band in bands], dtype=np.float32)
        global_score = float(np.mean(band_scores))

        base_depth = 1.8 + global_score * 10.0
        depths = []
        for score in band_scores:
            adjusted = base_depth + float(score - global_score) * 18.0
            depths.append(float(np.clip(adjusted, 0.8, 11.5)))

        return depths

    def _build_wear_probabilities(self, depths: list[float], confidence: float) -> np.ndarray:
        label_to_index = {
            "center_wear": 0,
            "edge_wear": 1,
            "patchy_wear": 2,
            "uniform_wear": 3,
            "one_side_wear": 4,
            "cupping_wear": 5,
        }

        wear_label = _infer_wear_pattern_from_depths(depths)
        probs = np.full(6, (1.0 - confidence) / 5.0 if confidence < 1.0 else 0.0, dtype=np.float32)
        probs[label_to_index.get(wear_label, 3)] = float(np.clip(confidence, 0.5, 0.99))
        probs = probs / float(np.sum(probs))
        return cast(np.ndarray, probs)

    def _hybrid_infer(
        self,
        image_bgr: np.ndarray,
        tread_seq: np.ndarray,
    ) -> dict[str, Any]:
        from ai_model.hybrid_torch.constants import CONDITION_LABELS
        from ai_model.hybrid_torch.runtime import predict_hybrid

        tta_variants = self._build_tta_images(image_bgr)
        tta_outputs: list[dict[str, np.ndarray]] = []
        for variant_image, flip_tread_positions in tta_variants:
            raw = predict_hybrid(
                self._hybrid_model,
                self._hybrid_device,
                variant_image,
                tread_seq,
            )
            typed_raw = {
                key: np.asarray(value).copy()
                for key, value in raw.items()
                if key != "source"
            }
            if flip_tread_positions:
                typed_raw["tread_depths"] = np.flip(typed_raw["tread_depths"], axis=1).copy()
            tta_outputs.append(typed_raw)

        averaged_outputs = {
            key: np.mean(np.stack([raw[key] for raw in tta_outputs], axis=0), axis=0)
            for key in ("tread_depths", "health_score", "remaining_life", "wear_pattern", "condition_probs")
        }

        tread_depths = averaged_outputs["tread_depths"]
        condition_probs = averaged_outputs["condition_probs"][0]
        best_condition = int(np.argmax(condition_probs))

        # --- Safety Guardrail Post-Processing: Align Tread Depth with Condition Classification ---
        best_condition_label = CONDITION_LABELS[best_condition]
        avg_depth = float(np.mean(tread_depths[0]))

        if best_condition_label == "replace" and avg_depth >= 3.0:
            target_avg = 2.4
            ratio = target_avg / avg_depth
            tread_depths[0] = np.clip(tread_depths[0] * ratio, 1.0, 2.9)
            logger.warning(
                "Safety guardrail triggered: condition was 'replace' but predicted avg tread depth was %.2fmm. "
                "Scaled down depth readings to average %.2fmm.", avg_depth, target_avg
            )
        elif best_condition_label == "moderate" and avg_depth >= 5.0:
            target_avg = 4.0
            ratio = target_avg / avg_depth
            tread_depths[0] = np.clip(tread_depths[0] * ratio, 3.0, 4.8)
            logger.warning(
                "Safety guardrail triggered: condition was 'moderate' but predicted avg tread depth was %.2fmm. "
                "Scaled down depth readings to average %.2fmm.", avg_depth, target_avg
            )
        elif best_condition_label == "safe" and avg_depth < 4.0:
            target_avg = 5.5
            ratio = target_avg / max(avg_depth, 0.1)
            tread_depths[0] = np.clip(tread_depths[0] * ratio, 4.2, 11.5)
            logger.warning(
                "Safety guardrail triggered: condition was 'safe' but predicted avg tread depth was %.2fmm. "
                "Scaled up depth readings to average %.2fmm.", avg_depth, target_avg
            )

        raw_outputs = {
            "tread_depths": tread_depths,
            "health_score": averaged_outputs["health_score"],
            "remaining_life": averaged_outputs["remaining_life"],
            "wear_pattern": averaged_outputs["wear_pattern"],
            "source": np.asarray(["pytorch_hybrid"], dtype=object),
        }
        confidence = float(np.max(condition_probs))
        prediction = cast(
            dict[str, Any],
            denormalize_outputs(
                raw_outputs,
                confidence_override=confidence,
                source="pytorch_hybrid",
            ),
        )
        prediction["condition_prediction"] = (
            CONDITION_LABELS[best_condition] if best_condition < len(CONDITION_LABELS) else "unknown"
        )
        prediction["condition_probabilities"] = {
            label: round(float(condition_probs[index]), 4)
            for index, label in enumerate(CONDITION_LABELS)
        }
        self._apply_depth_wear_override(prediction, tread_depths[0])
        return prediction

    def _attach_model_diagnostics(
        self,
        prediction: dict[str, Any],
        context_data: dict[str, Any] | None,
    ) -> None:
        """Attach non-blocking legacy helper diagnostics to the prediction."""
        depths = self._depths_from_prediction(prediction)
        if len(depths) == 4:
            temporal_payload: dict[str, Any] = {
                "depth_wear_pattern": _infer_wear_pattern_from_depths(depths),
                "depth_confidence": round(float(_confidence_from_depths(depths)), 4),
            }
            try:
                from ai_model.rnn.temporal_features import extract_wear_velocity

                temporal_payload["wear_velocity"] = extract_wear_velocity([depths], [0])
            except Exception as exc:
                temporal_payload["wear_velocity"] = {"available": False, "reason": str(exc)}
            prediction["temporal_features"] = temporal_payload
            prediction["tabular_crosscheck"] = self._tabular_crosscheck(depths)
            prediction["legacy_classification"] = self._legacy_classification(depths, prediction)

        context_vector = _fit_last_dim(_context_to_vector(context_data, target_dim=64)[np.newaxis, :], 64)[0]
        prediction["context_vector"] = {
            "dimension": int(context_vector.shape[-1]),
            "nonzero_count": int(np.count_nonzero(context_vector)),
            "l2_norm": round(float(np.linalg.norm(context_vector)), 4),
            "preview": [round(float(value), 4) for value in context_vector[:8]],
        }

    def _depths_from_prediction(self, prediction: dict[str, Any]) -> list[float]:
        tread = prediction.get("tread_depths_mm", {})
        if not isinstance(tread, dict):
            return []
        depths: list[float] = []
        for key in ("tread_1", "tread_2", "tread_3", "tread_4"):
            try:
                depths.append(float(tread[key]))
            except (KeyError, TypeError, ValueError):
                return []
        return depths

    def _tabular_crosscheck(self, depths: list[float]) -> dict[str, Any]:
        try:
            from ai_model.tabular_model import load_trained_model, predict_from_depths

            if not self._tabular_checked:
                self._tabular_bundle = load_trained_model(device=self._hybrid_device)
                self._tabular_checked = True
            if self._tabular_bundle is None:
                return {
                    "available": False,
                    "reason": "tabular model weights unavailable",
                }
            model, metadata, device_name = self._tabular_bundle
            crosscheck = predict_from_depths(model, metadata, device_name, depths)
            return {
                "available": True,
                "condition_label": crosscheck.get("condition_label"),
                "condition_confidence": round(float(crosscheck.get("condition_confidence", 0.0)), 4),
                "health_score": round(float(crosscheck.get("health_score", 0.0)), 2),
                "remaining_life_km": round(float(crosscheck.get("remaining_life_km", 0.0)), 0),
            }
        except Exception as exc:
            self._tabular_checked = True
            return {"available": False, "reason": str(exc)}

    def _legacy_classification(self, depths: list[float], prediction: dict[str, Any]) -> dict[str, Any]:
        try:
            from ai_model.classes import NUM_WEAR_CLASSES, build_smart_tire_report, tread_to_condition

            average_depth = float(np.mean(np.asarray(depths, dtype=np.float32)))
            wear = prediction.get("wear_pattern", {})
            wear_label = str(wear.get("label", "uniform_wear")) if isinstance(wear, dict) else "uniform_wear"
            class_wear_label = {
                "uniform_wear": "even_wear",
                "patchy_wear": "patch_wear",
                "side_wall_wear": "one_side_wear",
            }.get(wear_label, wear_label)
            smart_report = build_smart_tire_report(
                health_score=float(prediction.get("health_score", 5.0)),
                tread_depth_mm=average_depth,
                wear_label=class_wear_label,
                confidence_score=float(prediction.get("confidence", 0.75)),
            )
            return {
                "available": True,
                "num_wear_classes": int(NUM_WEAR_CLASSES),
                "tread_condition": tread_to_condition(average_depth),
                "report": smart_report.to_dict(),
            }
        except Exception as exc:
            return {"available": False, "reason": str(exc)}

    def _apply_depth_wear_override(
        self,
        prediction: dict[str, Any],
        tread_depths_mm: np.ndarray,
    ) -> None:
        depths = [float(value) for value in np.asarray(tread_depths_mm, dtype=np.float32).reshape(-1)[:4]]
        if len(depths) != 4:
            return

        depth_label = _infer_wear_pattern_from_depths(depths)
        prediction["depth_derived_wear_pattern"] = depth_label
        if depth_label != "side_wall_wear":
            return

        original_wear = dict(prediction.get("wear_pattern", {}))
        prediction["model_wear_pattern_before_depth_override"] = original_wear.get("label")
        inner = depths[0]
        outer = depths[-1]
        diff = max(depths) - min(depths)
        severity = "critical" if diff >= 3.0 else "high" if diff >= 2.0 else "moderate"
        confidence = _confidence_from_depths(depths)

        probabilities = dict(original_wear.get("probabilities") or {})
        probabilities["side_wall_wear"] = round(float(confidence), 4)
        prediction["wear_pattern"] = {
            "class_id": -1,
            "label": "side_wall_wear",
            "label_display": "Side-Wall Wear",
            "cause": "Outer shoulder wear concentrated on the tire edge",
            "advice": "Inspect wheel alignment, camber, and the outer shoulder before continued use.",
            "severity": severity,
            "confidence": round(float(confidence), 4),
            "probabilities": probabilities,
        }
        prediction["side_wall_wear_rule"] = {
            "outer_minus_inner_mm": round(float(outer - inner), 4),
            "outer_inner_ratio": round(float(outer / max(inner, 0.01)), 4),
            "min_diff_mm": 0.8,
            "side_wall_ratio": 1.5,
        }

    def _build_tta_images(self, image_bgr: np.ndarray) -> list[tuple[np.ndarray, bool]]:
        """Small deterministic test-time augmentations for smoother predictions."""
        height, width = image_bgr.shape[:2]
        center = (width / 2.0, height / 2.0)
        variants: list[tuple[np.ndarray, bool]] = [
            (image_bgr, False),
            (cv2.flip(image_bgr, 1), True),
        ]
        for angle in (-3.0, 3.0):
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            variants.append(
                (
                    cv2.warpAffine(
                        image_bgr,
                        matrix,
                        (width, height),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE,
                    ),
                    False,
                )
            )
        variants.append((cv2.convertScaleAbs(image_bgr, alpha=1.04, beta=2.0), False))
        variants.append((cv2.convertScaleAbs(image_bgr, alpha=0.96, beta=-2.0), False))
        return variants

    def _synthetic_prediction(self, depths: list[float]) -> dict[str, Any]:
        raw_outputs = {
            "tread_depths": np.asarray(depths, dtype=np.float32)[np.newaxis, :] / 12.0,
            "health_score": np.asarray([[np.clip(np.mean(depths) / 12.0, 0.0, 1.0)]], dtype=np.float32),
            "remaining_life": np.asarray(
                [[np.clip((np.mean(depths) - 1.6) / 10.4, 0.0, 1.0)]],
                dtype=np.float32,
            ),
            "wear_pattern": self._build_wear_probabilities(depths, 0.72)[np.newaxis, :],
            "source": "synthetic",
        }
        return cast(
            dict[str, Any],
            denormalize_outputs(raw_outputs, confidence_override=0.72, source="synthetic"),
        )

    def _reload_model_if_changed(self) -> None:
        """Hot-reload a newly trained hybrid checkpoint before the next prediction."""
        artifacts = self._resolve_hybrid_artifacts()
        if artifacts is None:
            if self._hybrid_model is not None:
                logger.error("Hybrid checkpoint was removed; unloading runtime model.")
            self._hybrid_model = None
            self._ready = False
            self._model_version = "not_loaded"
            self._model_source = "not_loaded"
            self._hybrid_checkpoint_mtime = None
            self._hybrid_checkpoint_path = None
            self._hybrid_metadata_path = None
            self._load_error = "Missing hybrid checkpoint"
            return
        checkpoint_path, metadata_path, checkpoint_status = artifacts
        if self._hybrid_model is None:
            self._load_model()
            return
        try:
            current_mtime = checkpoint_path.stat().st_mtime
        except OSError:
            return
        if (
            self._hybrid_checkpoint_path == checkpoint_path
            and self._hybrid_checkpoint_mtime is not None
            and current_mtime <= self._hybrid_checkpoint_mtime
        ):
            return

        try:
            from ai_model.hybrid_torch.runtime import load_hybrid_model

            hybrid_bundle = load_hybrid_model(checkpoint_path, metadata_path)
            if hybrid_bundle is None:
                return
            self._hybrid_model, self._hybrid_metadata, self._hybrid_device = hybrid_bundle
            self._hybrid_checkpoint_path = checkpoint_path
            self._hybrid_metadata_path = metadata_path
            self._hybrid_checkpoint_mtime = current_mtime
            self._hybrid_metadata["runtime_checkpoint_status"] = checkpoint_status
            self._model_version = str(
                self._hybrid_metadata.get(
                    "model_version",
                    "pytorch_hybrid:efficientnetv2_b0_vit_b16_bilstm_tcn_attention_calibrated",
                )
            )
            logger.info("Hot-reloaded PyTorch hybrid model: %s", checkpoint_path)
            self._model_source = (
                "pytorch_hybrid"
                if checkpoint_status == "accepted"
                else "pytorch_hybrid_unaccepted_last"
            )
            self._model_checkpoint = str(checkpoint_path)
            self._load_error = None
        except Exception as exc:
            self._load_error = str(exc)
            logger.warning("Hybrid model hot-reload failed; keeping existing model: %s", exc)

    async def cleanup(self) -> None:
        self._hybrid_model = None
        self._ready = False
        logger.info("InferenceService cleaned up")
