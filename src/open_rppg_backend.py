from typing import Dict, Optional

import numpy as np


class OpenRPPGBackend:
    """Thin wrapper around open-rppg for chunk-level face tensor inference."""

    def __init__(self, model_name: str = "FacePhys.rlap") -> None:
        try:
            import rppg
        except Exception as exc:
            raise RuntimeError(
                "open-rppg is not installed. Install with: python3 -m pip install open-rppg"
            ) from exc

        self.model_name = model_name
        self._model = rppg.Model(model_name)

    def estimate_chunk(self, face_tensor: np.ndarray, fps: float) -> Dict[str, Optional[float]]:
        """Estimate HR-related metrics for a single chunk.

        Args:
            face_tensor: uint8 array [T, H, W, 3] in RGB.
            fps: chunk frame rate.

        Returns:
            Dict with bpm, sqi, latency_sec, respiratory_rate_brpm.
        """
        result = self._model.process_faces_tensor(face_tensor, fps=float(fps))
        bpm = result.get("hr") if isinstance(result, dict) else None
        sqi = result.get("SQI") if isinstance(result, dict) else None
        latency_sec = result.get("latency") if isinstance(result, dict) else None

        rr = None
        hrv = result.get("hrv") if isinstance(result, dict) else None
        if isinstance(hrv, dict):
            rr = hrv.get("breathingrate")

        return {
            "bpm": float(bpm) if bpm is not None else None,
            "sqi": float(sqi) if sqi is not None else None,
            "latency_sec": float(latency_sec) if latency_sec is not None else None,
            "respiratory_rate_brpm": float(rr) if rr is not None else None,
        }
