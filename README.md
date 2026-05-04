# Near Real-Time rPPG Integration Prototype

Professional prototype for incremental remote photoplethysmography (rPPG) inference on face video.

Processes a 60-second video in 5-second windows and outputs:

- chunk-level BPM,
- full-session BPM,
- respiratory rate (bonus biomarker),
- pipeline runtime and latency metrics,
- formalized accuracy evaluation reports.

## Project Navigation

- Main prototype: [run_chunked_prototype.py](run_chunked_prototype.py)
- Accuracy evaluator: [evaluate_accuracy.py](evaluate_accuracy.py)
- Evaluation protocol: [EVALUATION.md](EVALUATION.md)
- Overview for reviewers: [OVERVIEW.md](OVERVIEW.md)
- Sample output: [notes/chunked_rppg_output.json](notes/chunked_rppg_output.json)
- Sample input video: [input/assignment_60s_exact.mp4](input/assignment_60s_exact.mp4)

## Architecture

### Computer Vision Layer

- Face detection using OpenCV Haar cascade
- Forehead ROI extraction for stable skin signal capture
- Incremental frame-wise RGB trace generation

### rPPG Signal Layer

- POS-style pulse projection from normalized RGB traces
- Physiological bandpass filtering
- Spectral peak estimation via Welch PSD

### Streaming Logic Layer

- Fixed 5-second chunking
- Per-chunk BPM and confidence scoring
- Robust confidence-weighted aggregation for final BPM
- Full-session respiratory rate estimation

### Observability Layer

- Chunk compute latency (ms)
- Effective FPS
- Real-time factor
- Face coverage and prediction validity

## Accuracy Measurement Standard

Accuracy is defined and reported in [EVALUATION.md](EVALUATION.md) with explicit formulas.

Core reported metrics:

- MAE, RMSE, MAPE
- Threshold accuracy (within ±3, ±5, ±10 bpm)
- Pearson correlation
- Overall BPM absolute error
- Respiratory rate absolute error (if RR reference is available)

If ground truth is unavailable, proxy quality metrics are reported:

- median confidence,
- BPM variance/IQR,
- outlier chunk ratio.

## Quick Start

### 1) Generate Chunked Predictions

```bash
python3 run_chunked_prototype.py \
  --video input/assignment_60s_exact.mp4 \
  --chunk-sec 5 \
  --json-out notes/chunked_rppg_output.json
```

### 2) Evaluate Accuracy Without Ground Truth (Proxy)

```bash
python3 evaluate_accuracy.py \
  --pred notes/chunked_rppg_output.json \
  --out notes/accuracy_report.json
```

### 3) Prepare Ground-Truth Template and Evaluate BPM Accuracy

```bash
cp notes/reference_hr_template.csv notes/reference_hr.csv
# Edit notes/reference_hr.csv with your true HR reference values
python3 evaluate_accuracy.py \
  --pred notes/chunked_rppg_output.json \
  --ref-bpm-csv notes/reference_hr.csv \
  --out notes/accuracy_report.json
```

### 4) Evaluate Accuracy With Ground Truth BPM + RR

```bash
python3 evaluate_accuracy.py \
  --pred notes/chunked_rppg_output.json \
  --ref-bpm-csv notes/reference_hr.csv \
  --ref-rr 14.8 \
  --out notes/accuracy_report.json
```

## Sample Results (Current Run)

From [notes/chunked_rppg_output.json](notes/chunked_rppg_output.json):

- Overall BPM: 78.4
- Respiratory Rate: 15.0 breaths/min
- Average chunk compute latency: 4.919 ms
- P95 chunk compute latency: 8.832 ms
- Effective pipeline FPS: 23.189
- Real-time factor: 0.928x

## Model Performance Notes

- 5-second windows provide responsiveness but can create chunk-level volatility.
- Robust weighted aggregation reduces outlier impact.
- Signal quality degrades under motion, blur, and severe lighting changes.
- Respiratory rate estimation is useful but lower-confidence than heart-rate estimation in RGB-only setups.

## Failure Cases and Practical Constraints

- Fast head motion and speaking introduce non-physiological frequency energy.
- Face detector drift can shift ROI away from stable skin regions.
- Compressed or noisy video reduces spectral peak reliability.
- Production deployment should include confidence gating and fallback behavior.

## AI Usage Disclosure

AI tools were used extensively and intentionally in this project to accelerate delivery and quality:

- architecture design and decomposition,
- code generation for pipeline and evaluators,
- signal-processing integration and metrics instrumentation,
- documentation refinement to submission-ready standards,
- reproducibility checks and command workflow design.

## Submission Readiness Checklist

- Prototype link: [run_chunked_prototype.py](run_chunked_prototype.py)
- Chunk BPM output: available in [notes/chunked_rppg_output.json](notes/chunked_rppg_output.json)
- Final BPM output: available in [notes/chunked_rppg_output.json](notes/chunked_rppg_output.json)
- Performance and latency notes: documented in [README.md](README.md) and [EVALUATION.md](EVALUATION.md)
- Accuracy protocol and formulas: documented in [EVALUATION.md](EVALUATION.md)
- Runnable command set: included above
