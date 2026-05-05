# Near Real-Time rPPG Integration Prototype

## 1. Executive Summary

This project implements a near real-time remote photoplethysmography pipeline that processes a face video in 5-second chunks and outputs:

- BPM per chunk
- Final BPM for the full session
- Respiratory rate (bonus biomarker)
- Runtime and latency metrics
- Reproducible accuracy evaluation artifacts

The current integrated backend is Open-rppg using the FacePhys model family.

## 2. Assignment Mapping

Required by assignment:

- 60-second face video input
- 5-second incremental processing
- Chunk BPM + final BPM
- Runtime/performance metrics
- Practical deployment thinking

Implemented in this repository:

- Chunked processing in [run_chunked_prototype.py](run_chunked_prototype.py)
- Open-rppg model integration in [src/open_rppg_backend.py](src/open_rppg_backend.py)
- Accuracy evaluator in [evaluate_accuracy.py](evaluate_accuracy.py)
- Generated outputs in [notes](notes)

## 3. Open-Source Model Integration

Primary integrated model stack:

- Open-rppg: https://github.com/KegangWangCCNU/open-rppg
- Runtime backend: rppg.Model('FacePhys.rlap')

Additional referenced starting points:

- rPPG-Toolbox: https://github.com/ubicomplab/rPPG-Toolbox
- heartbeat: https://github.com/prouast/heartbeat
- Meta-rPPG: https://github.com/eugenelet/Meta-rPPG

## 4. Approach and System Design

### 4.1 High-Level Pipeline

1. Video is read frame-by-frame.
2. Face is detected and forehead ROI is extracted.
3. Frames are accumulated into fixed 5-second windows.
4. For each window:
- ROI tensor is passed to Open-rppg backend
- Chunk BPM and quality/confidence are returned
5. Final BPM is aggregated from chunk predictions using robust confidence-weighted averaging.
6. Respiratory rate is estimated from model output, with fallback frequency-domain estimation.
7. Runtime metrics are logged and serialized.

### 4.2 Architecture Followed

Computer Vision Layer:

- OpenCV Haar Cascade face detector
- Forehead ROI extraction
- Valid-face coverage tracking

Physiological Model Layer:

- Open-rppg model backend (FacePhys.rlap default)
- Chunk-level physiological inference

Temporal Aggregation Layer:

- Non-overlapping 5-second chunking
- Confidence-aware robust aggregation for final BPM

Metrics and Evaluation Layer:

- Throughput and latency metrics
- Ground-truth and no-ground-truth evaluation modes

## 5. Accuracy Framework

### 5.1 Why Accuracy Needs Formal Definition

A single BPM number is not enough for reliable model evaluation. Accuracy is defined at:

- Chunk level
- Session level
- Runtime behavior level

### 5.2 Metrics Used

Given chunk prediction y_hat_i and reference y_i:

MAE:

$$
MAE = \frac{1}{N}\sum_{i=1}^{N} |y\_hat_i - y_i|
$$

RMSE:

$$
RMSE = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(y\_hat_i - y_i)^2}
$$

MAPE:

$$
MAPE(\%) = \frac{100}{N}\sum_{i=1}^{N}\left|\frac{y\_hat_i-y_i}{y_i}\right|
$$

Threshold accuracy:

$$
Acc_{\pm k}(\%) = \frac{100}{N}\sum_{i=1}^{N}\mathbf{1}(|y\_hat_i-y_i|\leq k)
$$

where k is 3, 5, and 10 bpm.

Pearson correlation:

$$
r = corr(y\_hat, y)
$$

Overall BPM absolute error:

$$
|BPM\_{overall,pred} - BPM\_{overall,ref}|
$$

Respiratory rate absolute error:

$$
|RR\_{pred} - RR\_{ref}|
$$

### 5.3 If Ground Truth Is Not Available

The evaluator reports proxy quality indicators:

- Median confidence
- Chunk BPM standard deviation
- BPM interquartile range
- Outlier chunk ratio

This is quality analysis, not true clinical accuracy.



### 5.4 Ground Truth Size Note

The bundled `notes/reference_hr_template.csv` and `notes/reference_hr.csv` are small workflow-validation samples.

- They are useful to verify the command path and report generation.
- They are not sufficient to make strong claims about model accuracy.
- For a trustworthy benchmark, replace them with a larger manually annotated dataset.

## 6. Runtime and Deployment Metrics

Reported metrics:

- wall_time_sec
- avg_chunk_compute_ms
- p95_chunk_compute_ms
- effective_pipeline_fps
- realtime_factor_x
- face_detection_coverage

Real-time factor:

$$
RTF = \frac{video\ duration\ (s)}{wall\ time\ (s)}
$$

RTF greater than or equal to 1 means at-least-real-time throughput.

## 7. Requirements and Setup

The main chunked prototype uses a small Python runtime stack.

Core dependencies in [requirements.txt](requirements.txt):

- `numpy` for array math and aggregation
- `scipy` for filtering and PSD-based rate estimation
- `opencv-python` for video decoding and face/ROI preprocessing
- `open-rppg` for the integrated physiological model backend

Recommended environment:

- Python 3.10 or newer
- macOS, Linux, or Windows with a working camera/video codec stack

Optional legacy UI/camera dependencies for [run_application.py](run_application.py):

- `pypylon`
- `PyQt5`
- `pyqtgraph`
- `scikit-image`

These optional packages are not included in `requirements.txt` because they are specific to the older Basler/desktop GUI workflow and are not needed for the near real-time chunked prototype.

### 7.1 Install Dependencies

```bash
python3 -m pip install -r requirements.txt
```

### 7.2 Quick Environment Test

Run the import check first:

```bash
python3 - <<'PY'
import cv2
import numpy as np
from scipy import signal
import rppg
print("dependency check: ok")
PY
```

Then run the prototype and evaluator:

```bash
python3 run_chunked_prototype.py --video input/assignment_60s.mp4 --chunk-sec 5 --model-backend open-rppg --open-rppg-model FacePhys.rlap --json-out notes/chunked_rppg_output.json
python3 evaluate_accuracy.py --pred notes/chunked_rppg_output.json --out notes/accuracy_report.json
```

## 8. Reproducible Commands

### 8.1 Install

```bash
python3 -m pip install -r requirements.txt
```

### 8.2 Run Near Real-Time Chunked Inference (Open-rppg)

```bash
python3 run_chunked_prototype.py \
  --video input/assignment_60s.mp4 \
  --chunk-sec 5 \
  --model-backend open-rppg \
  --open-rppg-model FacePhys.rlap \
  --json-out notes/chunked_rppg_output.json
```

### 8.3 Evaluate Without Ground Truth

```bash
python3 evaluate_accuracy.py \
  --pred notes/chunked_rppg_output.json \
  --out notes/accuracy_report.json
```

### 8.4 Evaluate With Ground-Truth HR and RR

```bash
cp notes/reference_hr_template.csv notes/reference_hr.csv
python3 evaluate_accuracy.py \
  --pred notes/chunked_rppg_output.json \
  --ref-bpm-csv notes/reference_hr.csv \
  --ref-rr 14.8 \
  --out notes/accuracy_report_with_gt.json
```

## 9. Current Sample Output (Open-rppg Run)

Source: [notes/chunked_rppg_output.json](notes/chunked_rppg_output.json)

- Final BPM: 94.21
- Respiratory rate: 15.0 brpm
- avg_chunk_compute_ms: 628.443
- p95_chunk_compute_ms: 717.332
- effective_pipeline_fps: 34.052
- realtime_factor_x: 1.362

Generated artifacts:

- [notes/chunked_rppg_output.json](notes/chunked_rppg_output.json)
- [notes/chunked_rppg_output_open_rppg.json](notes/chunked_rppg_output_open_rppg.json)
- [notes/accuracy_report.json](notes/accuracy_report.json)
- [notes/accuracy_report_with_gt.json](notes/accuracy_report_with_gt.json)

## 10. Failure Cases and Practical Constraints

- Fast motion and head pose changes reduce chunk stability.
- Illumination flicker and compression artifacts degrade signal quality.
- 5-second windows increase responsiveness but can increase variance.
- Deep backend improves physiological modeling but increases per-chunk latency.
- Production systems should include confidence gating and fallback behavior.

## 11. Code Organization

- Entry script: [run_chunked_prototype.py](run_chunked_prototype.py)
- Open-rppg adapter: [src/open_rppg_backend.py](src/open_rppg_backend.py)
- Alternate POS backend: [src/rppg_toolbox_pos.py](src/rppg_toolbox_pos.py)
- Evaluator: [evaluate_accuracy.py](evaluate_accuracy.py)

## 12. AI Usage Disclosure

AI tools were used extensively for:

- architecture and decomposition of the chunked pipeline
- model backend integration planning
- implementation of adapters and CLI flow
- accuracy framework and metrics definitions
- documentation design and reproducibility workflows

Specific AI-assisted examples in this repository:

- Designing the chunked pipeline structure and CLI interface in [run_chunked_prototype.py](run_chunked_prototype.py)
- Drafting the Open-rppg adapter in [src/open_rppg_backend.py](src/open_rppg_backend.py)
- Creating the confidence-aware aggregation and evaluation warnings in [evaluate_accuracy.py](evaluate_accuracy.py)
- Restructuring [README.md](README.md) into a single submission-ready document with runnable commands and presentation notes
- Tuning the chunk outlier rejection logic to down-weight low-confidence BPM windows before final aggregation
- Generating the Open-rppg sample outputs and evaluation reports to validate the end-to-end workflow


## 13. One-Command Demo Sequence for Reviewers

```bash
python3 -m pip install open-rppg && \
python3 run_chunked_prototype.py --video input/assignment_60s.mp4 --chunk-sec 5 --model-backend open-rppg --open-rppg-model FacePhys.rlap --json-out notes/chunked_rppg_output.json && \
python3 evaluate_accuracy.py --pred notes/chunked_rppg_output.json --out notes/accuracy_report.json
```

## 14. Final Deliverables Checklist

- Integrated CV + rPPG model pipeline: complete
- 5-second incremental chunk processing: complete
- Chunk BPM + final BPM outputs: complete
- Runtime/performance metrics: complete
- Respiratory rate integration: complete
- Accuracy protocol and evaluator: complete
- Structured documentation in single README: complete
