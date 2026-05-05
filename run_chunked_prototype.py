import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from scipy import signal

from src.open_rppg_backend import OpenRPPGBackend
from src.rppg_toolbox_pos import pos_wang


@dataclass
class ChunkResult:
    chunk_index: int
    start_sec: float
    end_sec: float
    bpm: Optional[float]
    confidence: float
    face_coverage: float
    process_time_ms: float


def bandpass_filter(x: np.ndarray, fs: float, low_hz: float, high_hz: float, order: int = 3) -> np.ndarray:
    nyq = 0.5 * fs
    low = max(low_hz / nyq, 1e-5)
    high = min(high_hz / nyq, 0.999)
    if low >= high:
        return x
    b, a = signal.butter(order, [low, high], btype="band")
    return signal.filtfilt(b, a, x)


def extract_bvp_pos(rgb_trace: np.ndarray) -> np.ndarray:
    # POS-like projection on normalized RGB traces.
    eps = 1e-8
    c = rgb_trace.astype(np.float64)
    c = (c / (np.mean(c, axis=0, keepdims=True) + eps)) - 1.0

    x = c[:, 1] - c[:, 2]  # G - B
    y = c[:, 1] + c[:, 2] - 2.0 * c[:, 0]  # G + B - 2R
    alpha = np.std(x) / (np.std(y) + eps)
    bvp = x + alpha * y
    bvp = bvp - np.mean(bvp)
    return bvp


def estimate_rate_from_psd(
    x: np.ndarray,
    fs: float,
    low_hz: float,
    high_hz: float,
    to_per_minute: bool = True,
) -> Tuple[Optional[float], float]:
    if len(x) < max(int(fs * 2), 16):
        return None, 0.0

    nperseg = min(len(x), int(fs * 4))
    freqs, pxx = signal.welch(x, fs=fs, nperseg=nperseg)
    band = (freqs >= low_hz) & (freqs <= high_hz)
    if not np.any(band):
        return None, 0.0

    bf = freqs[band]
    bp = pxx[band]
    idx = int(np.argmax(bp))
    peak_f = float(bf[idx])
    peak_p = float(bp[idx])

    noise_floor = float(np.median(bp) + 1e-10)
    confidence = float(np.clip(peak_p / noise_floor, 0.0, 50.0))

    value = peak_f * 60.0 if to_per_minute else peak_f
    return value, confidence


def robust_weighted_average(values: List[float], weights: List[float]) -> Optional[float]:
    if not values:
        return None

    arr = np.array(values, dtype=np.float64)
    w = np.array(weights, dtype=np.float64)

    if len(arr) == 1:
        return float(arr[0])

    # Confidence-aware outlier rejection:
    # 1) remove the lowest-confidence quarter when enough chunks exist
    # 2) remove BPM values far from the robust center using MAD
    if len(arr) >= 4:
        confidence_floor = np.percentile(w, 25)
        keep_conf = w >= confidence_floor
    else:
        keep_conf = np.ones_like(w, dtype=bool)

    arr = arr[keep_conf]
    w = w[keep_conf]

    if len(arr) == 0:
        return None

    median = np.median(arr)
    mad = np.median(np.abs(arr - median)) + 1e-8
    keep_mad = np.abs(arr - median) <= 1.75 * 1.4826 * mad

    arr = arr[keep_mad]
    w = w[keep_mad]

    if len(arr) == 0:
        return None

    if np.sum(w) <= 0:
        return float(np.median(arr))

    weighted = float(np.sum(arr * w) / np.sum(w))
    return weighted


class FaceRPPGChunkProcessor:
    def __init__(
        self,
        video_path: Path,
        chunk_sec: float = 5.0,
        min_face_area_ratio: float = 0.01,
        detect_every_n: int = 5,
        model_backend: str = "rppg-toolbox-pos",
        roi_size: int = 36,
        open_rppg_model_name: str = "FacePhys.rlap",
    ) -> None:
        self.video_path = video_path
        self.chunk_sec = chunk_sec
        self.min_face_area_ratio = min_face_area_ratio
        self.detect_every_n = detect_every_n
        self.model_backend = model_backend
        self.roi_size = roi_size
        self.open_rppg_model_name = open_rppg_model_name
        self.open_rppg_backend = None
        if self.model_backend == "open-rppg":
            self.open_rppg_backend = OpenRPPGBackend(model_name=self.open_rppg_model_name)

        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def _detect_face(self, frame_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
        )
        if len(faces) == 0:
            return None
        areas = [w * h for (_, _, w, h) in faces]
        return tuple(map(int, faces[int(np.argmax(areas))]))

    @staticmethod
    def _forehead_roi(face_box: Tuple[int, int, int, int], fw: int, fh: int) -> Tuple[int, int, int, int]:
        x, y, w, h = face_box
        rx = x + int(0.2 * w)
        ry = y + int(0.15 * h)
        rw = int(0.6 * w)
        rh = int(0.25 * h)

        rx = max(0, min(rx, fw - 1))
        ry = max(0, min(ry, fh - 1))
        rw = max(1, min(rw, fw - rx))
        rh = max(1, min(rh, fh - ry))
        return rx, ry, rw, rh

    def run(self) -> dict:
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration_sec = (frame_count / fps) if frame_count > 0 else 0.0
        chunk_size = max(1, int(round(self.chunk_sec * fps)))

        chunk_rgb: List[np.ndarray] = []
        chunk_rois: List[np.ndarray] = []
        chunk_results: List[ChunkResult] = []
        all_bvp_parts: List[np.ndarray] = []
        all_valid_rois: List[np.ndarray] = []
        rr_estimates: List[float] = []
        rr_weights: List[float] = []

        frame_idx = 0
        detect_counter = 0
        face_box: Optional[Tuple[int, int, int, int]] = None
        used_face_frames = 0

        t0 = time.perf_counter()
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            h, w = frame.shape[:2]
            detect_counter += 1

            if face_box is None or detect_counter >= self.detect_every_n:
                detect_counter = 0
                candidate = self._detect_face(frame)
                if candidate is not None:
                    fx, fy, fw, fh = candidate
                    if (fw * fh) >= int(self.min_face_area_ratio * w * h):
                        face_box = candidate

            if face_box is not None:
                rx, ry, rw, rh = self._forehead_roi(face_box, w, h)
                roi = frame[ry : ry + rh, rx : rx + rw, :]
                if roi.size > 0:
                    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    rgb_mean = roi_rgb.reshape(-1, 3).mean(axis=0)
                    roi_small = cv2.resize(
                        roi_rgb,
                        (self.roi_size, self.roi_size),
                        interpolation=cv2.INTER_AREA,
                    )
                    chunk_rgb.append(rgb_mean)
                    chunk_rois.append(roi_small)
                    all_valid_rois.append(roi_small)
                    used_face_frames += 1
                else:
                    chunk_rgb.append(np.array([np.nan, np.nan, np.nan], dtype=np.float64))
                    chunk_rois.append(np.zeros((self.roi_size, self.roi_size, 3), dtype=np.uint8))
            else:
                chunk_rgb.append(np.array([np.nan, np.nan, np.nan], dtype=np.float64))
                chunk_rois.append(np.zeros((self.roi_size, self.roi_size, 3), dtype=np.uint8))

            frame_idx += 1

            if len(chunk_rgb) == chunk_size:
                c_start = time.perf_counter()
                start_sec = (frame_idx - chunk_size) / fps
                end_sec = frame_idx / fps

                chunk_arr = np.array(chunk_rgb, dtype=np.float64)
                valid = ~np.isnan(chunk_arr).any(axis=1)
                face_coverage = float(np.mean(valid)) if len(valid) else 0.0

                bpm = None
                confidence = 0.0

                if np.sum(valid) >= int(0.7 * chunk_size):
                    if self.model_backend == "open-rppg":
                        roi_arr = np.array(chunk_rois, dtype=np.uint8)
                        valid_rois = roi_arr[valid]
                        model_out = self.open_rppg_backend.estimate_chunk(valid_rois, fps=fps)
                        bpm = model_out["bpm"]
                        sqi = model_out["sqi"]
                        confidence = float(max(1e-3, (sqi if sqi is not None else 0.0) * 20.0))
                        rr_chunk = model_out["respiratory_rate_brpm"]
                        if rr_chunk is not None:
                            rr_estimates.append(float(rr_chunk))
                            rr_weights.append(confidence)
                    elif self.model_backend == "rppg-toolbox-pos":
                        roi_arr = np.array(chunk_rois, dtype=np.uint8)
                        valid_rois = roi_arr[valid]
                        bvp = pos_wang(valid_rois, fs=fps)
                        bvp = bandpass_filter(bvp, fps, 0.7, 3.5)
                        bpm, confidence = estimate_rate_from_psd(bvp, fps, 0.7, 3.5, to_per_minute=True)
                        if bpm is not None:
                            all_bvp_parts.append(bvp)
                    else:
                        valid_trace = chunk_arr[valid]
                        bvp = extract_bvp_pos(valid_trace)
                        bvp = bandpass_filter(bvp, fps, 0.7, 3.5)
                        bpm, confidence = estimate_rate_from_psd(bvp, fps, 0.7, 3.5, to_per_minute=True)
                        if bpm is not None:
                            all_bvp_parts.append(bvp)

                proc_ms = (time.perf_counter() - c_start) * 1000.0
                chunk_results.append(
                    ChunkResult(
                        chunk_index=len(chunk_results),
                        start_sec=round(start_sec, 3),
                        end_sec=round(end_sec, 3),
                        bpm=round(float(bpm), 2) if bpm is not None else None,
                        confidence=round(confidence, 3),
                        face_coverage=round(face_coverage, 3),
                        process_time_ms=round(proc_ms, 3),
                    )
                )

                chunk_rgb = []
                chunk_rois = []

        cap.release()
        elapsed = time.perf_counter() - t0

        valid_bpms = [c.bpm for c in chunk_results if c.bpm is not None]
        confs = [max(c.confidence, 1e-3) for c in chunk_results if c.bpm is not None]
        overall_bpm = robust_weighted_average(valid_bpms, confs)

        overall_rr = None
        rr_conf = 0.0
        if rr_estimates:
            overall_rr = robust_weighted_average(rr_estimates, rr_weights)
            rr_conf = float(np.mean(rr_weights)) if rr_weights else 0.0
        elif all_bvp_parts:
            full_bvp = np.concatenate(all_bvp_parts)
            rr_signal = bandpass_filter(full_bvp, fps, 0.1, 0.5)
            rr, rr_conf = estimate_rate_from_psd(rr_signal, fps, 0.1, 0.5, to_per_minute=True)
            overall_rr = rr
        elif all_valid_rois:
            full_roi_arr = np.array(all_valid_rois, dtype=np.uint8)
            full_bvp = pos_wang(full_roi_arr, fs=fps)
            rr_signal = bandpass_filter(full_bvp, fps, 0.1, 0.5)
            rr, rr_conf = estimate_rate_from_psd(rr_signal, fps, 0.1, 0.5, to_per_minute=True)
            overall_rr = rr

        chunk_times = [c.process_time_ms for c in chunk_results]
        output = {
            "video": str(self.video_path),
            "model_backend": self.model_backend,
            "open_rppg_model": self.open_rppg_model_name if self.model_backend == "open-rppg" else None,
            "fps": round(fps, 3),
            "total_frames": frame_idx,
            "duration_sec": round(duration_sec if duration_sec > 0 else frame_idx / fps, 3),
            "chunk_sec": self.chunk_sec,
            "chunk_size_frames": chunk_size,
            "chunk_results": [asdict(c) for c in chunk_results],
            "overall": {
                "bpm": round(overall_bpm, 2) if overall_bpm is not None else None,
                "respiratory_rate_brpm": round(float(overall_rr), 2) if overall_rr is not None else None,
                "rr_confidence": round(rr_conf, 3),
                "valid_chunk_count": len(valid_bpms),
                "total_chunk_count": len(chunk_results),
            },
            "metrics": {
                "wall_time_sec": round(elapsed, 3),
                "avg_chunk_compute_ms": round(float(np.mean(chunk_times)) if chunk_times else 0.0, 3),
                "p95_chunk_compute_ms": round(float(np.percentile(chunk_times, 95)) if chunk_times else 0.0, 3),
                "effective_pipeline_fps": round(frame_idx / elapsed, 3) if elapsed > 0 else None,
                "realtime_factor_x": round((frame_idx / fps) / elapsed, 3) if elapsed > 0 and fps > 0 else None,
                "face_detection_coverage": round(used_face_frames / frame_idx, 3) if frame_idx > 0 else 0.0,
            },
        }
        return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Near real-time chunked rPPG prototype")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--chunk-sec", type=float, default=5.0, help="Chunk duration in seconds")
    parser.add_argument(
        "--json-out",
        default="notes/chunked_rppg_output.json",
        help="Where to write JSON output",
    )
    parser.add_argument(
        "--model-backend",
        default="open-rppg",
        choices=["open-rppg", "rppg-toolbox-pos", "legacy-pos"],
        help="rPPG model backend for chunk processing",
    )
    parser.add_argument(
        "--open-rppg-model",
        default="FacePhys.rlap",
        help="Open-rppg model name, e.g. FacePhys.rlap, PhysNet.pure, TSCAN.rlap",
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Input video does not exist: {video_path}")

    backend = args.model_backend if args.model_backend != "legacy-pos" else "legacy-pos"
    result = FaceRPPGChunkProcessor(
        video_path=video_path,
        chunk_sec=args.chunk_sec,
        model_backend=backend,
        open_rppg_model_name=args.open_rppg_model,
    ).run()

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("=== Chunk-level BPM (5-second windows) ===")
    for c in result["chunk_results"]:
        print(
            f"Chunk {c['chunk_index']:02d} [{c['start_sec']:.1f}-{c['end_sec']:.1f}s] "
            f"BPM={c['bpm']} conf={c['confidence']} face_cov={c['face_coverage']} "
            f"latency_ms={c['process_time_ms']}"
        )

    print("\n=== Overall ===")
    print(f"BPM: {result['overall']['bpm']}")
    print(f"Respiratory Rate (brpm): {result['overall']['respiratory_rate_brpm']}")

    print("\n=== Runtime Metrics ===")
    for k, v in result["metrics"].items():
        print(f"{k}: {v}")

    print(f"\nSaved JSON output to: {out_path}")


if __name__ == "__main__":
    main()
