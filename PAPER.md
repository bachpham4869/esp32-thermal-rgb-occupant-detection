# PAPER.md: Material for the Research Paper

> Working notes for writing the paper. All numbers below were **reproduced on 2026-10-10** from the data in `data/raw/` with `tools/train_classifier.py` (seed 42) plus extra analysis scripts (LOSO, ablation, and rule baseline; their code is described in §7). Items marked **[TODO]** must be measured or written before submission. Items marked **[CITE]** need a real reference. Do not invent citations.
>
> Diagrams: see `docs/FLOWCHART.md` (export from mermaid.live as SVG for the figures).

---

## 1. Title candidates

1. *Low-Cost On-Device Child Presence Detection Using an 80×62 Thermal Sensor and a Lightweight Spatio-Temporal Classifier on ESP32-S3*
2. *Rejecting Thermal Distractors in In-Cabin Occupant Detection with Long-Term Blob Features on a Microcontroller*
3. *TinyCPD: Thermal-RGB Occupant Detection with a 9-Feature Logistic Gate on a $10 MCU*

## 2. Abstract (draft, about 200 words)

Children left in parked vehicles die of heatstroke every year **[CITE: NHTSA / KidsAndCars statistics]**, and safety-rating programmes now reward direct-sensing Child Presence Detection (CPD) **[CITE: Euro NCAP CPD protocol]**. Existing solutions rely on 60 GHz radar or high-resolution cameras with neural networks, which raise cost, power, and privacy concerns. We present a fully on-device CPD system built on an ESP32-S3 microcontroller with a low-resolution 80×62 thermal sensor (MI48) and an auxiliary OV3660 RGB camera. A classical pipeline first segments warm regions: selective EMA background subtraction, morphological dilation, connected-component labelling, rule filters, and a breathing-variance test. However, the rules alone accept every hot-cup sample and 92% of phone samples as human (rule pass rate 99.0% for humans, 100% for cups, 91.6% for phones). We therefore track each blob for up to 64 s and extract nine long-term thermal, shape, and motion features. A logistic-regression gate is trained offline and folded into a 9-weight dot product. On 4,231 labelled blob samples collected in a real car (four seating positions, blanket occlusion, hot cup, phone), the gate reaches 97.7% accuracy (ROC-AUC 0.981) on a held-out split and 94.6% under leave-one-session-out validation. The false-acceptance rate drops to 16% for hot cups and 0% for phones. The detector needs about 100 KB of RAM and no neural network, and it runs alongside live MJPEG streaming on the same chip.

## 3. Contributions (claim list)

1. **System**: a complete, open, low-cost (<$30 BOM **[TODO: exact BOM]**) thermal+RGB CPD prototype on one ESP32-S3. A dual-core FreeRTOS design separates sensing and detection (core 0) from web streaming (core 1).
2. **Pipeline**: an MCU-friendly thermal occupant detector with selective background adaptation (a person who stays still is not absorbed into the background), noise-robust blob formation, a gray-zone breathing-variance test, and N/M debounce.
3. **Long-term blob descriptors**: nine features computable at 80×62 resolution: thermal trend, de-trended residual (micro-motion and breathing), centroid and area jitter, shape, and temperature spread. They separate humans from thermal distractors that pass the static rules.
4. **Deployment trick**: the StandardScaler is folded into the LR weights (`w_raw = w/σ`, `b_raw = b − Σ wμ/σ`), so inference is 9 MACs plus one `expf`, with no stored normalisation statistics.
5. **Dataset and tooling**: a labelled in-car feature dataset plus a host-side logging and labelling tool (serial and web) for iterative data collection.

## 4. Related work (topics to cover; fill citations)

- **CPD regulation and motivation**: Euro NCAP CPD requirements (direct sensing), US Hot Cars Act, heatstroke statistics. **[CITE]**
- **Radar-based CPD**: 60 GHz FMCW in-cabin radar (TI, Infineon, Vayyar white papers and academic papers). Strength: vital signs through blankets. Weakness: cost and integration. **[CITE]**
- **Camera-based occupant detection**: NIR/RGB DMS/OMS with CNNs. Weaknesses are privacy, lighting dependence, and compute. **[CITE]**
- **Low-resolution thermal sensing**: people counting and presence with Grid-EYE 8×8, MLX90640 32×24, and Heimann arrays. Classical approaches (background subtraction, blobs) vs tiny CNNs. **[CITE]**
- **TinyML on MCUs**: TFLite Micro, Edge Impulse, model-size vs accuracy trade-offs. This motivates our choice of LR over a CNN. **[CITE]**
- **Positioning**: 80×62 is roughly 6× the pixels of a 32×24 sensor and still privacy-preserving. Our method adds *temporal* descriptors rather than spatial deep features.

## 5. System design

### 5.1 Hardware
| Component | Part | Interface / notes |
|---|---|---|
| MCU | ESP32-S3 (dual Xtensa LX7 @240 MHz), 16 MB flash, OPI PSRAM | board def `boards/esp32s3_cam.json` |
| Thermal | Meridian MI48 SenXor, **80×62** | I2C @100 kHz (control, addr 0x40), SPI @2 MHz (frames), DATA_READY, nRESET |
| RGB | OV3660 | DVP 8-bit, VGA JPEG q=12, 2 frame buffers in PSRAM |
| Connectivity | Wi-Fi SoftAP | dashboard on :80, MJPEG on :81 |
| **[TODO]** | mounting position (roof console / rear-view mirror?), FOV, height, photo of installation | needed for Fig. 2 |

- Thermal frame: 80-word header (frame counter, VDD, Ta, timestamp, min/max, CRC) followed by 4,960 uint16 pixels in units of 0.1 K. CRC16-CCITT-FALSE is verified per frame.
- Frame-rate divider = 4. **[TODO: measure the actual thermal FPS from `/status`; the logs suggest about 5–6 Hz]**

### 5.2 Software architecture (Fig. 1 = `docs/FLOWCHART.md` §1)
- `thermalTask` (core 0, priority 2) polls DATA_READY, reads the frame over SPI, and runs `OccupantDetector::update()` in the same task, so no lock is needed on the hot path.
- `streamTask` (core 1) serves MJPEG. `loop()` serves the dashboard and JSON.
- Results reach the web side through mutex-protected double buffers (`_resultSafe`, `_labelsSafe`, `_pixelsSafe`). `getResultAndGrid()` takes both in one critical section so the blob overlay matches the frame.

### 5.3 Detection pipeline (Fig. 3 = FLOWCHART §2)

Let $T_t(p)$ be the 3×3-smoothed temperature at pixel $p$ and $B_t(p)$ the background.

**Warm-up** (first $N_w=30$ frames): $B \leftarrow B + 0.3\,(T-B)$.

**Selective EMA background**:
$$B_{t+1}(p)=B_t(p)+\alpha_p\,(T_t(p)-B_t(p)),\quad \alpha_p=\begin{cases}\alpha/10 & |T-B|>\tau\\ \alpha & \text{otherwise}\end{cases}$$
with $\alpha=0.02$ and $\tau=2\,^\circ$C. The 10× slower update on foreground pixels keeps a motionless occupant from fading into the background. **[TODO: quantify the time to absorption, e.g. a child asleep for 10 min]**

**Foreground**: $M(p)=[T-B>\tau]$, then one-pixel 8-neighbour dilation (this closes salt-and-pepper fragmentation of a body), then iterative BFS 4-connected labelling (≤8 blobs).

**Rule cascade** (per blob):
| Rule | Threshold | Outcome | Intent |
|---|---|---|---|
| max temperature | > 58 °C | REJECTED | boiling water, engine-hot items |
| size | < 15 px | REJECTED | phone, cup, noise |
| aspect w/h | ∉ [0.35, 2.8] | LOW_SUSPECT | laptops, bags, strips |
| size | ≥ 60 px | PERSON | clear human |
| otherwise | | GRAY_ZONE | track further |

**Gray-zone breathing test**: the variance of the blob mean temperature over the last 8 frames must be ≥ 0.03 °C² for CANDIDATE, otherwise REJECTED.

**Tracking**: greedy nearest-centroid association (gate 6 px, 8 slots). A track is dropped after more than 6 missed frames.

**Debounce** (FLOWCHART §3): IDLE → CANDIDATE → CONFIRMED when ≥5 of the last 7 frames have a candidate → ALERT. The state returns to IDLE after 10 empty frames.

### 5.4 Long-term features and LR gate (FLOWCHART §4)

Each track keeps a ring buffer sampled every 1 s (64 deep, about 64 s): mean temperature $\bar T_k$, centroid $(c_x,c_y)_k$, and area $A_k$. With ≥5 samples:

| # | Feature | Definition | Physical intuition |
|---|---|---|---|
| 1 | aspect | $w/h$ of bbox | body shape |
| 2 | solidity | $A/(wh)$ | compact objects (cups) vs irregular bodies |
| 3 | deltaAmbT | $\bar T - T_a$ (MI48 die temperature) | relative warmth |
| 4 | trendSlope | OLS slope of $\bar T_k$ (°C/min) | cooling cup vs homeostatic body |
| 5 | residStd | std of OLS residual | breathing and micro-motion |
| 6 | cxJitter | $\tfrac12(\sigma_{c_x}+\sigma_{c_y})$ | postural sway |
| 7 | areaJitter | $\sigma_A$ | shape breathing / occlusion changes |
| 8 | maxTemp | max T in blob | skin ≈ 34–37 °C vs hot objects |
| 9 | tempSpread | max − min T in blob | uniform skin vs hot liquid with cool rim |

$p_{\text{human}}=\sigma(\mathbf w_{\text{raw}}^\top\mathbf x+b_{\text{raw}})$. A PERSON or CANDIDATE blob is **demoted** to REJECTED if $p<0.5$. The gate never promotes, so the rules act as a high-recall prefilter and the LR as a precision filter.

**Scaler folding**: training uses z-scored features. At deployment
$\mathbf w_{\text{raw}}=\mathbf w/\boldsymbol\sigma$ and $b_{\text{raw}}=b-\sum_i w_i\mu_i/\sigma_i$, so the MCU stores 10 floats and computes 9 MACs plus one `expf`.

Deployed weights (`data/models/lr_model.json`):
| aspect | solidity | deltaAmbT | trendSlope | residStd | cxJitter | areaJitter | maxTemp | tempSpread | bias |
|---|---|---|---|---|---|---|---|---|---|
| −0.181 | −1.881 | −0.298 | −0.030 | 4.123 | −0.763 | −0.0089 | −0.687 | 0.392 | 27.84 |

Standardised coefficients (relative importance): maxTemp **−4.67**, tempSpread **+2.34**, areaJitter −1.61, cxJitter −0.87, residStd +0.67, deltaAmbT −0.61, solidity −0.29, aspect −0.09, trendSlope −0.08.

### 5.5 Resource usage
| Item | Value | Source |
|---|---|---|
| Detector static RAM | ≈ 104 KB (bg 19.8 + smooth 19.8 + raw 19.8 + labels 9.9 + BFS queue 9.9 + masks 9.9 + safe grid 5 + tracks ≈ 8.6 + results ≈ 1.2) | computed from `Detector.h` |
| MI48 driver buffers | ≈ 20 KB | `mi48.h` |
| LR model | 10 floats = 40 B | |
| Flash app partition | 6.25 MB slot | `esp32s3_cam_16MB.csv` |
| **[TODO]** | per-frame latency of `update()` (wrap with `esp_timer_get_time()`), CPU %, free heap from `/status`, power draw (mA @5 V) | must measure |

## 6. Data collection

- Procedure: firmware built with `DATA_COLLECTION_MODE=true` prints one CSV row per valid tracked blob whenever a new long-term sample is taken. `tools/host_logger.py` timestamps the rows. The operator sets labels through serial `L<n>` or the dashboard buttons (FLOWCHART §5).
- Sessions (real vehicle **[TODO: car model, date, ambient/cabin temperature, time of day, sensor placement]**):

| Class (folder) | Session file | Rows (after filtering) | Scenario |
|---|---|---|---|
| human (+) | driver_seat_upright | 559 | driver seat, upright |
| human (+) | front_passenger_upright | 782 | front passenger |
| human (+) | rear_left_upright | 673 | rear left |
| human (+) | rear_right_upright | 796 | rear right |
| human (+) | all_seat_blanket_occluded(_2) | 1 + 119 | occupants under blanket |
| hot cup (−) | cup_middle | 595 | hot drink, centre |
| phone (−) | phone_middle | 706 | phone, centre |
| **Total** | 8 files | **4,231** (2,930 + / 1,301 −) | collected 2026-08-19 to 2026-09-03 |

- Filtering: `sampleCount ≥ 5`, no NaN/Inf (4,235 → 4,231). Rows corrupted by USB-CDC byte drops are skipped by the parser.
- Labelling is **per session (folder)**. Every blob seen during a human session is labelled human, including spurious background blobs. This is label noise; discuss it in §9.

Per-class feature means (± std):
| Feature | Human | Hot cup | Phone |
|---|---|---|---|
| maxTemp (°C) | 35.7 ± 1.3 | 49.5 ± 6.2 | 46.4 ± 1.2 |
| tempSpread (°C) | 3.7 ± 1.6 | 17.0 ± 6.6 | 3.2 ± 1.0 |
| deltaAmbT (°C) | −2.4 ± 1.1 | 1.8 ± 2.0 | 3.3 ± 1.0 |
| residStd (°C) | 0.18 ± 0.13 | 0.30 ± 0.09 | 0.20 ± 0.10 |
| cxJitter (px) | 1.39 ± 0.62 | 3.59 ± 1.27 | 1.47 ± 0.54 |
| areaJitter (px) | 55.9 ± 51.5 | 40.6 ± 22.9 | 246 ± 412 |
| trendSlope (°C/min) | 2.1 ± 3.0 | 3.9 ± 2.2 | 2.0 ± 3.1 |
| solidity | 0.70 ± 0.17 | 0.80 ± 0.08 | 0.78 ± 0.15 |
| aspect | 0.87 ± 0.33 | 0.89 ± 0.16 | 1.04 ± 0.56 |
| pixelCount | 204 ± 312 | 130 ± 48 | 102 ± 125 |

Observation: in this data the hot cup does **not** show the cooling trend we expected (slope +3.9 °C/min). It may still have been heating the sensor view, or the session was short. **[TODO: longer cup sessions]**

## 7. Experiments and results

> **See `RESULT.md` for the full, detailed results** (model comparison, LOSO ablation, threshold sweep, system-level simulation, figures). Key caveat found there: under LOSO, an LR on `maxTemp` alone (Acc 0.978) beats the 9-feature LR (0.946), and the hot-cup false acceptances look like mislabelled hand blobs at about 35 °C. Adjust the claims in §2 and §3 before submission.

### 7.1 Main result: stratified 75/25 split (seed 42)
| Metric | Value |
|---|---|
| Accuracy | **0.977** |
| ROC-AUC | **0.981** |
| Human precision / recall / F1 | 0.97 / 1.00 / 0.98 |
| Fake-heat precision / recall / F1 | 1.00 / 0.93 / 0.96 |
| Confusion [[TN FP],[FN TP]] | [[302, 23], [1, 732]] (n = 1,058) |

5-fold stratified CV ROC-AUC: **0.980**.

### 7.2 Leave-one-session-out (LOSO): more honest, because neighbouring rows are correlated
| Held-out session | n | Accuracy |
|---|---|---|
| driver_seat_upright | 559 | 1.000 |
| front_passenger_upright | 782 | 0.922 |
| rear_left_upright | 673 | 1.000 |
| rear_right_upright | 796 | 0.960 |
| all_seat_blanket_occluded_2 | 119 | **0.664** |
| cup_middle (unseen class: trained on phone only) | 595 | **0.840** |
| phone_middle (unseen class: trained on cup only) | 706 | 1.000 |
| **Pooled** | 4,231 | **0.946** (F1 0.961), CM [[1206, 95], [134, 2796]] |

Talking points: (i) generalisation across seats is good. (ii) Blanket occlusion is the hardest case and the main weakness. (iii) The model partly generalises to a distractor class it has not seen.

### 7.3 Rules vs rules + LR (the motivation for the gate)
The rule cascade was re-applied to the logged features (with the gray zone assumed to pass, so this is an upper bound on rule recall). The LR rows use the deployed weights on all rows, which is in-sample.

| Method | Acc | Precision | Recall | F1 | Human pass | Cup pass (FAR) | Phone pass (FAR) |
|---|---|---|---|---|---|---|---|
| Rules only | 0.699 | 0.700 | 0.990 | 0.820 | 99.0% | **100%** | **91.6%** |
| Rules + LR gate | 0.966 | 0.968 | 0.984 | 0.976 | 98.4% | **16.1%** | **0.0%** |

The cup reaches only about 50 °C, which is below the 58 °C rule, so the static rules cannot reject it. The LR uses maxTemp and tempSpread jointly to do so. This is the paper's key argument.

> Note: the `blobType` column in the CSV was written *after* the firmware's LR gate (whatever weights were flashed at collection time), so do **not** use it as a rule-only baseline.

### 7.4 Feature ablation (5-fold AUC, drop one feature)
| Dropped | AUC | Δ vs 0.9802 |
|---|---|---|
| tempSpread | 0.9739 | −0.0063 |
| cxJitter | 0.9743 | −0.0059 |
| residStd | 0.9770 | −0.0032 |
| areaJitter | 0.9772 | −0.0030 |
| maxTemp | 0.9784 | −0.0018 |
| solidity | 0.9797 | −0.0005 |
| deltaAmbT | 0.9799 | −0.0003 |
| aspect | 0.9802 | 0 |
| trendSlope | 0.9804 | +0.0002 |

The temporal features (cxJitter, residStd, areaJitter) carry information that maxTemp alone does not. aspect and trendSlope are candidates for removal. **[TODO: also report an ablation of all temporal features vs static only]**

### 7.5 Experiments still needed **[TODO]**
- [ ] **End-to-end alert metrics**: time-to-alert, missed-alert rate, and false-alert rate per hour on continuous recordings, not just per-row classification.
- [ ] Negative classes 4–6: sunlit seat or dashboard, HVAC vent, vapour. The folders exist but are empty. Without them the claim about distractor rejection is limited to cups and phones.
- [ ] Child or infant proxies (baby seat, a doll with a heat pad as a proxy, rear-facing seat). The current "human" data is adults.
- [ ] Cabin temperature sweep (e.g. 20 °C to 45 °C). The maxTemp ≈ skin temperature cue weakens when the cabin is near 35 °C.
- [ ] Multiple sessions and days per class, then report LOSO with confidence intervals.
- [ ] Baseline comparisons: a tiny CNN (TFLite Micro) on the 80×62 frame, a decision tree, and a small MLP. Compare RAM, latency, and accuracy.
- [ ] Latency and power measurements (§5.5).
- [ ] Threshold sweep: ROC/PR curves and the chosen operating point (0.5 vs tuned).

### 7.6 How to reproduce the numbers
```bash
pip install numpy pandas scikit-learn
python tools/train_classifier.py          # §7.1, prints the C snippet and writes lr_model.json
```
For LOSO, ablation, and the rule baseline (§7.2–7.4): reuse `load_all_csv()` and `clean()` from `tools/train_classifier.py`. Loop over `source_file` for LOSO with a `StandardScaler`+`LogisticRegression(class_weight="balanced")` pipeline. Use `StratifiedKFold(5, shuffle=True, random_state=42)` for the ablation. The rule baseline is `maxTemp ≤ 58 & pixelCount ≥ 15 & 0.35 ≤ aspect ≤ 2.8`. **[TODO: commit this as `tools/eval_extra.py`]**

## 8. Figures and tables plan
| # | Content | Source |
|---|---|---|
| Fig. 1 | Hardware/software architecture | FLOWCHART §1 |
| Fig. 2 | Photo of the prototype and its mounting in the car, with the FOV | **[TODO]** |
| Fig. 3 | Detection pipeline | FLOWCHART §2 |
| Fig. 4 | State machine | FLOWCHART §3 |
| Fig. 5 | Example thermal frames: person, blanket, cup, phone, with blob contours | screenshots of the dashboard **[TODO]** |
| Fig. 6 | Feature distributions per class (box or violin: maxTemp, tempSpread, cxJitter, residStd) | from `data/raw` **[TODO plot]** |
| Fig. 7 | ROC curve; LOSO bar chart | **[TODO plot]** |
| Tab. I | Hardware | §5.1 |
| Tab. II | Features | §5.4 |
| Tab. III | Dataset | §6 |
| Tab. IV | Rules vs rules+LR | §7.3 |
| Tab. V | LOSO | §7.2 |
| Tab. VI | Resources and latency | §5.5 |

## 9. Discussion, limitations, threats to validity
- **Small, correlated dataset.** There are 8 sessions and one session per negative class. Rows within a second share their long-term features, so a random split is optimistic (0.977 vs 0.946 LOSO). Report LOSO as the main number.
- **Session-level labels.** Background blobs in human sessions are labelled human. This inflates the FP-side noise and biases what the model learns.
- **The model leans on maxTemp.** It largely learns "skin temperature (~36 °C) vs hotter objects". Distractors at skin temperature, such as a sun-warmed seat, a pet, or a warm bag, and hot cabins near 35 °C are untested and likely failure modes.
- **deltaAmbT uses the sensor die temperature (Ta), not the cabin air.** This explains why it is negative for humans. Either rename the feature or add a real cabin-temperature sensor (a DHT20 driver already exists in `lib/`).
- **Adults only.** The system has not been tested with children or infants, who are the target population of CPD.
- **RGB is unused for decisions.** "Fusion" is currently visualisation only. Present it honestly as future work.
- **Gray-zone threshold (0.03 °C²) and the other rule constants were hand-tuned** (`include/config.h`). Add a sensitivity analysis.
- **Absorption into the background.** Over a very long stay, α/10 = 0.002 per frame can still absorb a motionless occupant. Quantify it.
- **Privacy is a strength.** At 80×62 thermal resolution individuals cannot be identified, unlike with RGB.

## 10. Future work
- RGB confirmation in daylight with a tiny person detector (TFLite Micro / Edge Impulse), the TODO in `Detector.h`.
- Tiny CNN or temporal model on the thermal sequence, compared at equal RAM.
- Explicit respiration-rate estimation (FFT of the blob mean temperature, 0.2–1 Hz) as a stronger liveness feature.
- Alerting over MQTT or cellular (the Adafruit MQTT lib is already vendored) plus a vehicle-ignition-off trigger.
- Sensor fusion with a cabin temperature sensor (DHT20) for heat-risk escalation.
- Larger dataset: children, pets, sun, HVAC, several cars.

## 11. Key implementation references (for the reproducibility section)
| Paper concept | Code |
|---|---|
| Background EMA and warm-up | `OccupantDetector::updateBackground`, `update` (`src/Detector.cpp`) |
| Dilation and BFS labelling | `dilateMask`, `labelBlobs` |
| Rule cascade | `classify`, thresholds in `include/config.h` |
| Breathing variance | `trackAndGetVariance` |
| Tracking | `findOrAssignTrack` |
| Long-term features | `sampleLongTerm`, `computeFeatures` |
| LR gate | `lrPredictHuman`, `LR_WEIGHTS`, `LR_THRESHOLD` |
| State machine | `runStateMachine` |
| Training and scaler folding | `tools/train_classifier.py::fold_scaler_into_lr` |
| MI48 frame and CRC | `MI48::readFrameSPI`, `crc16_ccitt_false` (`src/mi48.cpp`) |
