# Executive Overview

## Prototype Link

- Entry point: [run_chunked_prototype.py](run_chunked_prototype.py)

## Deliverables Status

- Chunk-level BPM per 5-second window: Complete
- Overall BPM for full 60-second sequence: Complete
- Runtime and latency metrics: Complete
- Respiratory rate estimation: Complete
- Formal accuracy protocol and formulas: Complete

## Output Artifacts

- Prediction output JSON: [notes/chunked_rppg_output.json](notes/chunked_rppg_output.json)
- Accuracy report JSON: [notes/accuracy_report.json](notes/accuracy_report.json)
- Accuracy protocol: [EVALUATION.md](EVALUATION.md)

## Runnable Commands

### Generate Prediction

```bash
python3 run_chunked_prototype.py \
  --video input/assignment_60s_exact.mp4 \
  --chunk-sec 5 \
  --json-out notes/chunked_rppg_output.json
```

### Evaluate Without Ground Truth

```bash
python3 evaluate_accuracy.py \
  --pred notes/chunked_rppg_output.json \
  --out notes/accuracy_report.json
```

### Evaluate With Ground Truth

```bash
cp notes/reference_hr_template.csv notes/reference_hr.csv
# Edit notes/reference_hr.csv with your true reference values
python3 evaluate_accuracy.py \
  --pred notes/chunked_rppg_output.json \
  --ref-bpm-csv notes/reference_hr.csv \
  --ref-rr 14.8 \
  --out notes/accuracy_report.json
```

## Current Sample Results

- Overall BPM: 78.4
- Respiratory Rate: 15.0 breaths/min
- Avg chunk latency: 4.919 ms
- P95 chunk latency: 8.832 ms
- Real-time factor: 0.928x

## Accuracy Measurement Definition

- True accuracy requires external reference HR values across time.
- This project reports MAE, RMSE, MAPE, threshold accuracy, correlation, and overall error when reference data is provided.
- Without ground truth, it reports proxy stability and confidence metrics only.

See [EVALUATION.md](EVALUATION.md) for full formulas and acceptance criteria.

## AI Usage Summary

AI was used heavily for architecture design, implementation, metric design, evaluator development, and documentation polishing.
