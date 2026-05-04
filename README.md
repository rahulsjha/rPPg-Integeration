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
- Sample input video: [input/assignment_60s.mp4](input/assignment_60s.mp4)

## Open-Source rPPG Starting Points Referenced

- open-rppg (Heart Rate + Respiratory Rate): https://github.com/KegangWangCCNU/open-rppg
- rPPG-Toolbox (Heart Rate + Respiratory Rate): https://github.com/ubicomplab/rPPG-Toolbox
- heartbeat (Heart Rate): https://github.com/prouast/heartbeat
- Meta-rPPG (Heart Rate): https://github.com/eugenelet/Meta-rPPG

Implementation note:
- It is aligned with the assignment requirement to integrate a CV model into an incremental rPPG pipeline and report HR + RR + runtime behavior.
- It can be extended to directly plug deep models from the repositories above as the signal-estimation backend.

Selected integrated model for this submission:

- Open-rppg deep model backend via [src/open_rppg_backend.py](src/open_rppg_backend.py), using `rppg.Model('FacePhys.rlap')` by default.

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
python3 -m pip install open-rppg

python3 run_chunked_prototype.py \
  --video input/assignment_60s.mp4 \
  --chunk-sec 5 \
  --model-backend open-rppg \
  --open-rppg-model FacePhys.rlap \
  --json-out notes/chunked_rppg_output.json
```

Optional comparison against previous internal baseline:

```bash
python3 run_chunked_prototype.py \
  --video input/assignment_60s.mp4 \
  --chunk-sec 5 \
  --model-backend rppg-toolbox-pos \
  --json-out notes/chunked_rppg_output_rppg_toolbox_pos.json

python3 run_chunked_prototype.py \
  --video input/assignment_60s.mp4 \
  --chunk-sec 5 \
  --model-backend legacy-pos \
  --json-out notes/chunked_rppg_output_legacy_pos.json
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

- Overall BPM: 103.0
- Respiratory Rate: 15.0 breaths/min
- Average chunk compute latency: 701.735 ms
- P95 chunk compute latency: 887.348 ms
- Effective pipeline FPS: 33.477
- Real-time factor: 1.339x

## Model Performance Notes

- 5-second windows provide responsiveness but can create chunk-level volatility.
- Robust weighted aggregation reduces outlier impact.
- Signal quality degrades under motion, blur, and severe lighting changes.
- Open-rppg backend improves model-level physiological feature extraction but costs more per-chunk latency than simple analytical baselines.
- Respiratory rate is provided from model HRV breathing-rate output with fallback spectral estimation.

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
