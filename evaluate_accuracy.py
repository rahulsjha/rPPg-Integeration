import argparse
import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ChunkEval:
    chunk_index: int
    start_sec: float
    end_sec: float
    pred_bpm: Optional[float]
    ref_bpm: Optional[float]
    abs_error: Optional[float]
    signed_error: Optional[float]


def load_prediction(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_reference_csv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    times = []
    bpms = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"time_sec", "bpm"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError("Reference CSV must include columns: time_sec,bpm")

        for row in reader:
            try:
                t = float(row["time_sec"])
                b = float(row["bpm"])
            except (TypeError, ValueError):
                continue
            if np.isfinite(t) and np.isfinite(b):
                times.append(t)
                bpms.append(b)

    if not times:
        raise ValueError("Reference CSV has no valid rows")

    t_arr = np.array(times, dtype=np.float64)
    b_arr = np.array(bpms, dtype=np.float64)
    order = np.argsort(t_arr)
    return t_arr[order], b_arr[order]


def ref_mean_for_window(times: np.ndarray, bpms: np.ndarray, start_sec: float, end_sec: float) -> Optional[float]:
    mask = (times >= start_sec) & (times < end_sec)
    if not np.any(mask):
        return None
    return float(np.mean(bpms[mask]))


def safe_corr(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    if len(x) < 2:
        return None
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def compute_error_metrics(pred: np.ndarray, ref: np.ndarray) -> Dict[str, Optional[float]]:
    err = pred - ref
    abs_err = np.abs(err)

    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(err**2)))

    nonzero = np.abs(ref) > 1e-8
    mape = float(np.mean(abs_err[nonzero] / np.abs(ref[nonzero])) * 100.0) if np.any(nonzero) else None

    within_3 = float(np.mean(abs_err <= 3.0) * 100.0)
    within_5 = float(np.mean(abs_err <= 5.0) * 100.0)
    within_10 = float(np.mean(abs_err <= 10.0) * 100.0)

    corr = safe_corr(pred, ref)

    return {
        "mae_bpm": round(mae, 4),
        "rmse_bpm": round(rmse, 4),
        "mape_percent": round(mape, 4) if mape is not None else None,
        "pearson_r": round(corr, 4) if corr is not None else None,
        "within_3bpm_percent": round(within_3, 3),
        "within_5bpm_percent": round(within_5, 3),
        "within_10bpm_percent": round(within_10, 3),
        "sample_count": int(len(pred)),
    }


def compute_no_gt_quality(prediction: Dict) -> Dict[str, Optional[float]]:
    chunks = prediction.get("chunk_results", [])
    bpms = [c.get("bpm") for c in chunks if c.get("bpm") is not None]
    confs = [c.get("confidence", 0.0) for c in chunks if c.get("bpm") is not None]

    if not bpms:
        return {
            "valid_chunk_count": 0,
            "median_confidence": None,
            "bpm_std": None,
            "bpm_iqr": None,
            "outlier_chunk_ratio": None,
            "note": "No valid BPM chunks available",
        }

    arr = np.array(bpms, dtype=np.float64)
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1
    med = np.median(arr)

    outlier_mask = np.abs(arr - med) > max(10.0, 1.5 * iqr)
    outlier_ratio = float(np.mean(outlier_mask))

    return {
        "valid_chunk_count": int(len(arr)),
        "median_confidence": round(float(np.median(confs)), 4) if confs else None,
        "bpm_std": round(float(np.std(arr)), 4),
        "bpm_iqr": round(float(iqr), 4),
        "outlier_chunk_ratio": round(outlier_ratio, 4),
        "note": "Proxy quality metrics only (no reference ground truth provided)",
    }


def evaluate(prediction: Dict, ref_times: np.ndarray, ref_bpms: np.ndarray) -> Dict:
    chunk_evals: List[ChunkEval] = []

    for c in prediction.get("chunk_results", []):
        pred_bpm = c.get("bpm")
        start_sec = float(c.get("start_sec", 0.0))
        end_sec = float(c.get("end_sec", 0.0))

        ref_bpm = ref_mean_for_window(ref_times, ref_bpms, start_sec, end_sec)

        if pred_bpm is None or ref_bpm is None:
            chunk_evals.append(
                ChunkEval(
                    chunk_index=int(c.get("chunk_index", -1)),
                    start_sec=start_sec,
                    end_sec=end_sec,
                    pred_bpm=pred_bpm,
                    ref_bpm=ref_bpm,
                    abs_error=None,
                    signed_error=None,
                )
            )
            continue

        err = float(pred_bpm) - float(ref_bpm)
        chunk_evals.append(
            ChunkEval(
                chunk_index=int(c.get("chunk_index", -1)),
                start_sec=start_sec,
                end_sec=end_sec,
                pred_bpm=float(pred_bpm),
                ref_bpm=float(ref_bpm),
                abs_error=abs(err),
                signed_error=err,
            )
        )

    valid = [e for e in chunk_evals if e.pred_bpm is not None and e.ref_bpm is not None]
    if not valid:
        raise ValueError("No overlapping valid chunk predictions and reference data")

    pred = np.array([e.pred_bpm for e in valid], dtype=np.float64)
    ref = np.array([e.ref_bpm for e in valid], dtype=np.float64)

    metrics = compute_error_metrics(pred, ref)

    sample_count = int(len(ref))
    small_sample_warning = None
    if sample_count < 10:
        small_sample_warning = (
            f"Reference set contains only {sample_count} chunk-aligned samples; "
            "treat accuracy values as a workflow validation signal, not a stable benchmark."
        )

    overall_pred = prediction.get("overall", {}).get("bpm")
    overall_ref = float(np.mean(ref)) if len(ref) else None
    overall_abs_error = abs(float(overall_pred) - overall_ref) if overall_pred is not None and overall_ref is not None else None

    return {
        "mode": "with_ground_truth",
        "chunk_accuracy": metrics,
        "warnings": [small_sample_warning] if small_sample_warning else [],
        "overall": {
            "pred_overall_bpm": overall_pred,
            "ref_overall_bpm": round(overall_ref, 4) if overall_ref is not None else None,
            "overall_abs_error_bpm": round(float(overall_abs_error), 4) if overall_abs_error is not None else None,
        },
        "chunk_details": [asdict(e) for e in chunk_evals],
    }


def evaluate_rr(prediction: Dict, ref_rr: float) -> Dict[str, Optional[float]]:
    pred_rr = prediction.get("overall", {}).get("respiratory_rate_brpm")
    if pred_rr is None:
        return {
            "pred_rr_brpm": None,
            "ref_rr_brpm": ref_rr,
            "rr_abs_error_brpm": None,
        }
    err = abs(float(pred_rr) - float(ref_rr))
    return {
        "pred_rr_brpm": float(pred_rr),
        "ref_rr_brpm": float(ref_rr),
        "rr_abs_error_brpm": round(float(err), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate rPPG chunk prediction accuracy")
    parser.add_argument("--pred", required=True, help="Prediction JSON path (from run_chunked_prototype.py)")
    parser.add_argument(
        "--ref-bpm-csv",
        default=None,
        help="Optional reference HR CSV with columns: time_sec,bpm",
    )
    parser.add_argument(
        "--ref-rr",
        type=float,
        default=None,
        help="Optional reference respiratory rate in breaths/min",
    )
    parser.add_argument(
        "--out",
        default="notes/accuracy_report.json",
        help="Output JSON path for evaluation report",
    )
    args = parser.parse_args()

    pred_path = Path(args.pred)
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction JSON not found: {pred_path}")

    prediction = load_prediction(pred_path)

    report: Dict[str, object]
    if args.ref_bpm_csv is None:
        report = {
            "mode": "no_ground_truth",
            "proxy_quality": compute_no_gt_quality(prediction),
            "overall_prediction": prediction.get("overall", {}),
            "pipeline_metrics": prediction.get("metrics", {}),
        }
    else:
        ref_csv = Path(args.ref_bpm_csv)
        if not ref_csv.exists():
            raise FileNotFoundError(f"Reference CSV not found: {ref_csv}")
        ref_t, ref_b = load_reference_csv(ref_csv)
        report = evaluate(prediction, ref_t, ref_b)

    if args.ref_rr is not None:
        report["respiratory_rate_evaluation"] = evaluate_rr(prediction, args.ref_rr)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=== Accuracy Evaluation ===")
    print(f"Mode: {report['mode']}")

    if report["mode"] == "with_ground_truth":
        m = report["chunk_accuracy"]
        o = report["overall"]
        for warning in report.get("warnings", []):
            print(f"WARNING: {warning}")
        print(f"Chunk MAE (bpm): {m['mae_bpm']}")
        print(f"Chunk RMSE (bpm): {m['rmse_bpm']}")
        print(f"Chunk MAPE (%): {m['mape_percent']}")
        print(f"Within 5 bpm (%): {m['within_5bpm_percent']}")
        print(f"Pearson r: {m['pearson_r']}")
        print(f"Overall BPM abs error: {o['overall_abs_error_bpm']}")
    else:
        q = report["proxy_quality"]
        print(f"Valid chunks: {q['valid_chunk_count']}")
        print(f"Median confidence: {q['median_confidence']}")
        print(f"BPM std: {q['bpm_std']}")
        print(f"Outlier chunk ratio: {q['outlier_chunk_ratio']}")

    if "respiratory_rate_evaluation" in report:
        rr = report["respiratory_rate_evaluation"]
        print(f"RR abs error (brpm): {rr['rr_abs_error_brpm']}")

    print(f"Saved evaluation report to: {out_path}")


if __name__ == "__main__":
    main()
