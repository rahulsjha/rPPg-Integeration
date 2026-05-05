# AI Usage Summary

This document explicitly details how AI tools were used in building this Near Real-Time rPPG Integration prototype.

## 1. Architecture & Design (AI-Assisted)

**Prompt to AI:** "Design a near real-time rPPG pipeline that processes 60-second face videos in 5-second chunks, outputs chunk-level BPM and final BPM with confidence metrics, and includes respiratory rate estimation as a bonus feature."

**AI Contributions:**
- Suggested chunking strategy (non-overlapping 5-sec windows for simplicity, confidence weighting for aggregation)
- Recommended face detection + forehead ROI extraction over chest detection (more practical for laptop webcams)
- Proposed robust weighted averaging with MAD-based outlier removal
- Suggested POS-style BVP extraction as fallback when deep models fail
- Recommended Welch PSD for frequency-domain heart rate extraction

**Implementation:** [run_chunked_prototype.py](run_chunked_prototype.py) lines 1-100

---

## 2. Evaluation Framework Design (AI-Designed)

**Prompt to AI:** "Design an accuracy evaluation framework for rPPG systems that works both with and without ground truth, including chunk-level and session-level metrics, robustness indicators, and runtime constraints."

**AI Contributions:**
- Formal metric definitions with mathematical formulas (MAE, RMSE, MAPE, correlation)
- Threshold-based accuracy (within 3/5/10 bpm) for clinical relevance
- Proxy quality metrics when GT unavailable (std, IQR, outlier ratio)
- Chunk-level error tracking with temporal alignment
- Real-time factor calculation for deployment feasibility

**Implementation:** [EVALUATION.md](EVALUATION.md), [evaluate_accuracy.py](evaluate_accuracy.py)

---

## 3. Open-rPPG Backend Integration (AI-Assisted)

**Prompt to AI:** "How do I wrap the open-rppg Python library into a reusable backend class that extracts chunk-level heart rate, quality score (SQI), and respiratory rate from face tensors?"

**AI Contributions:**
- Suggested interface design with clear input/output contracts
- Error handling for missing/invalid model outputs
- Extraction of SQI as confidence metric
- Respiratory rate extraction from HRV dictionary (bonus)

**Implementation:** [src/open_rppg_backend.py](src/open_rppg_backend.py)

---

## 4. Signal Processing Pipeline (AI-Generated)

**Prompt to AI:** "Generate Python functions for bandpass filtering BVP signals (0.7-3.5 Hz for HR, 0.1-0.5 Hz for RR) and robust PSD-based frequency extraction using Welch method."

**AI Contributions:**
- Bandpass filter design using scipy.signal.butter with proper nyquist handling
- Welch PSD computation with adaptive window sizing
- Peak detection in frequency domain with confidence from noise floor ratio
- Robust weighted averaging with MAD-based outlier trimming

**Implementation:** [run_chunked_prototype.py](run_chunked_prototype.py) lines 15-90

---

## 5. Aggregation & Confidence Weighting (AI-Designed)

**Prompt to AI:** "Design a robust aggregation method for combining chunk-level heart rate estimates into a session-level estimate that handles outliers, weights by confidence, and maintains interpretability."

**AI Contributions:**
- Confidence scores derived from model SQI and PSD peak-to-noise ratio
- Median-Absolute-Deviation (MAD) based outlier detection
- Weighted average only on non-outliers
- Fallback to simple mean if all confidence scores fail

**Implementation:** [run_chunked_prototype.py](run_chunked_prototype.py) lines 72-90, 268-298

---

## 6. Respiratory Rate Estimation Fallback (AI-Designed)

**Prompt to AI:** "How should I estimate respiratory rate from rPPG signals when the model doesn't provide it? Include a fallback that uses full-session BVP concatenation."

**AI Contributions:**
- Tiered strategy: use model output first, fall back to full-session BVP analysis
- Frequency-domain RR extraction (0.1-0.5 Hz band, typical breathing 12-30 breaths/min)
- Confidence scoring consistent with BPM extraction

**Implementation:** [run_chunked_prototype.py](run_chunked_prototype.py) lines 288-301

---

## 7. Data Structures & Type Hints (AI-Generated)

**Prompt to AI:** "Generate Python dataclasses for structured output that serializes to JSON for both chunk results and evaluation metrics."

**AI Contributions:**
- ChunkResult dataclass with all necessary fields
- Validation and bounds checking
- Consistent serialization to JSON

**Implementation:** [run_chunked_prototype.py](run_chunked_prototype.py) lines 8-15, [evaluate_accuracy.py](evaluate_accuracy.py) lines 14-17

---

## 8. Evaluation Metrics Implementation (AI-Implemented)

**Prompt to AI:** "Write Python functions to compute MAE, RMSE, MAPE, correlation, and threshold accuracy (within N bpm) from prediction and reference arrays."

**AI Contributions:**
- Proper statistical implementations with edge case handling
- MAPE computation avoiding division by zero
- Pearson correlation with std check
- Threshold calculation (within 3/5/10 bpm)

**Implementation:** [evaluate_accuracy.py](evaluate_accuracy.py) lines 31-55

---

## 9. Command-Line Interface (AI-Generated)

**Prompt to AI:** "Generate a production-style argparse CLI with proper error handling, type validation, and informative help text."

**AI Contributions:**
- Argument validation (file existence checks)
- Model backend selection with defaults
- Flexible output path handling
- Clear help text for all parameters

**Implementation:** [run_chunked_prototype.py](run_chunked_prototype.py) lines 355-379

---

## 10. Documentation & Markdown (AI-Polish)

**Prompts to AI:** 
- "Write comprehensive README for an rPPG project including architecture, usage, and evaluation framework"
- "Create mathematical notation for evaluation metrics using proper formatting"
- "Document system design decisions with deployment considerations"

**AI Contributions:**
- Executive summary with clear mapping to requirements
- Architecture diagrams in text form
- Formal mathematical definitions (KaTeX formatted)
- Deployment considerations section

**Implementation:** [README.md](README.md), [EVALUATION.md](EVALUATION.md), [OVERVIEW.md](OVERVIEW.md)

---

## 11. Error Handling & Robustness (AI-Designed)

**Prompts to AI:**
- "What are failure modes in real-time rPPG systems and how do I handle them?"
- "How should I handle face detection failures, invalid ROIs, and low-quality frames?"

**AI Contributions:**
- Face detection retry logic with N-frame intervals
- Face coverage monitoring (70% threshold for chunk validity)
- NaN handling for missing frames
- Fallback mechanisms for all signal processing stages

**Implementation:** [run_chunked_prototype.py](run_chunked_prototype.py) lines 155-200

---

## 12. Model Backend Abstraction (AI-Designed)

**Prompt to AI:** "Design a backend abstraction pattern so different rPPG models (open-rppg, rppg-toolbox, etc.) can be swapped without changing core pipeline logic."

**AI Contributions:**
- Strategy pattern for backend selection
- Consistent interface across backends
- Graceful degradation when model fails

**Implementation:** [run_chunked_prototype.py](run_chunked_prototype.py) lines 220-250

---

## Summary

**Total Lines of Code Generated/Assisted by AI:** ~600 lines  
**Total Documentation Lines:** ~400 lines  
**AI Tools Used:** GitHub Copilot (inline completions and chat suggestions)  

**Key Insight:** AI was most valuable for:
1. Designing the overall system architecture (time saved: ~4 hours)
2. Implementing signal processing correctly (time saved: ~3 hours)
3. Generating evaluation metrics and testing harness (time saved: ~2 hours)
4. Creating production-quality documentation (time saved: ~2 hours)

**Total Time Saved:** ~11 hours by AI assistance
