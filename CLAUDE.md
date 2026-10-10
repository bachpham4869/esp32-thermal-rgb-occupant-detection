# CLAUDE.md

This file gives Claude Code guidance for working in this repository.

## What this project is

Embedded **in-cabin occupant / Child Presence Detection (CPD)** on an **ESP32-S3-CAM** board. It fuses:
- a **Meridian MI48 (SenXor) 80×62 thermal sensor**, the primary detection channel, and
- an **OV3660 RGB camera**, used today only for live MJPEG viewing. RGB confirmation is a TODO; see `include/Detector.h`.

The detection pipeline runs fully on-device in classical CV: background subtraction, blobs, rule filters, tracking, and a debounce state machine. A **9-feature logistic-regression (LR) gate** is trained offline in Python and hard-coded into the firmware. It separates real humans from "fake heat" sources such as hot cups and phones.

Supporting docs: `docs/FLOWCHART.md` (Mermaid diagrams), `PAPER.md` (paper material), `RESULT.md` (detailed results). Regenerate the results and figures with `python tools/eval_extra.py`, which writes `docs/results/metrics.json` and `docs/figures/*.png`.

Code comments and UI strings are **Vietnamese without diacritics**. Keep that style when you edit firmware.

## Build / flash / monitor (PlatformIO)

```bash
pio run                       # build (env: esp32s3_cam, the only env)
pio run -t upload             # flash (921600 baud)
pio device monitor            # serial @115200 (USB-CDC on boot)
```
- Custom board def: `boards/esp32s3_cam.json`. Partition table: `esp32s3_cam_16MB.csv` (2×6.25 MB OTA app slots, 3.4 MB SPIFFS). PSRAM is OPI (`qio_opi`).
- `boards/m5stack-atom.json` and `boards/yolo_uno.json` are not referenced by `platformio.ini`.
- `lib/` has vendored libs (Adafruit MQTT/NeoPixel/SleepyDog, DHT20, HCSR04, LCD, Modbus) that the current firmware **does not use**. Nothing in `src/` includes them.
- There are no unit tests (`test/` holds only the PlatformIO README).

## Runtime usage

- The board starts a Wi-Fi **AP** with SSID `YOLO_FUSION` and password `12345678` (`include/config.h`). Browse to `http://192.168.4.1/`.
- Port 80 (`src/Task_web.cpp`): `/` serves the dashboard, `/frame` returns the raw thermal frame (4960×uint16 LE, units of 0.1 K), `/status` returns JSON (FPS counters, CRC, ambient temperature, detector state, best blob, `pHuman`), and `/setlabel?id=N` sets the data-collection label.
- Port 81: `/stream` serves the RGB MJPEG stream. It is handled in its own FreeRTOS task.
- Keep the background clear of people for about 30 frames after boot (warm-up) or after `resetBackground()`.

## Architecture

| File | Role |
|---|---|
| `src/main.cpp` | Setup: camera init → `occupant.begin()` → Wi-Fi AP → `web_init()` → creates `thermalTask` (core 0, prio 2) and `streamTask` (core 1, prio 1). `loop()` runs the port-80 server. Holds `DATA_COLLECTION_MODE` and the serial `L<n>` label parser. |
| `src/mi48.cpp`, `include/mi48.h` | MI48 driver. Uses I2C (`Wire`, regs 0xB0–0xE0) for control and SPI (2 MHz) for frames. Polls the DATA_READY pin, reads an 80-word header plus 4960 pixels, checks CRC16-CCITT-FALSE, and parses Ta/VDD/timestamp. Has a mutex-protected copy for the web task. |
| `src/Detector.cpp`, `include/Detector.h` | `OccupantDetector`, the whole detection pipeline plus the LR gate. Static buffers (no heap) take about 100 KB. |
| `src/Task_web.cpp` | The dashboard HTML/JS (one PROGMEM string), the endpoints above, and the MJPEG handler. |
| `include/config.h` | Pin map (camera and MI48) plus **all detector thresholds** (`DET_*`). Tune here. |
| `tools/host_logger.py` | Reads feature CSV from serial, prepends `hostTimestamp`, and forwards typed `L<n>` commands to the board. |
| `tools/train_classifier.py` | Loads `data/raw/<labelX_name>/*.csv`, trains `StandardScaler`+`LogisticRegression(class_weight=balanced)`, **folds the scaler into the raw weights**, then prints a C snippet and writes `data/models/lr_model.json`. |

### Detection pipeline (`OccupantDetector::update`, per thermal frame)
1. Convert raw values to °C (`raw/10 − 273.15`), then apply a 3×3 box blur.
2. **Warm-up** (`DET_WARMUP_FRAMES`=30): fast EMA (α=0.3) learns the background. No detection runs in this phase.
3. Background EMA: α=`DET_BG_ALPHA` on background pixels and α/10 on foreground pixels. The slow foreground rate keeps a still person from being absorbed into the background.
4. Foreground mask (`T − bg > 2 °C`), then 1-px 8-neighbour dilation, then BFS 4-connected labelling (max 8 blobs).
5. Rule filters in `classify()`: `maxT > 58 °C` → REJECTED; `<15 px` → REJECTED; aspect outside [0.35, 2.8] → LOW_SUSPECT; `≥60 px` → PERSON; otherwise → GRAY_ZONE.
6. GRAY_ZONE: the temperature variance over the last 8 frames (≥0.03 °C²) decides CANDIDATE or REJECTED.
7. Tracking: nearest-centroid match (<6 px) through `findOrAssignTrack()`, the **single** track-assignment point. A track is dropped after >6 missed frames.
8. Long-term buffer: one sample per 1 s, 64 samples kept. Once there are ≥5 samples, `computeFeatures()` produces the 9 features. The LR then **demotes** PERSON/CANDIDATE to REJECTED when `pHuman < 0.5`. It never promotes a blob.
9. Debounce state machine: IDLE → CANDIDATE → (≥5 hits in a 7-frame window) CONFIRMED → ALERT. The state returns to IDLE after 10 consecutive empty frames.
10. The result and label grid are copied under a mutex into `_resultSafe`/`_labelsSafe` for the web task. Use `getResultAndGrid()` for a consistent pair.

## The LR model: keep three places in sync

The feature order is **`aspect, solidity, deltaAmbT, trendSlope, residStd, cxJitter, areaJitter, maxTemp, tempSpread`** and must match exactly in:
1. `FEATURE_COLS` in `tools/train_classifier.py`
2. `x[]` in `OccupantDetector::lrPredictHuman()` (`src/Detector.cpp`)
3. `LR_WEIGHTS` / `LR_BIAS` in `src/Detector.cpp`. Paste the script output here and don't hand-edit the numbers. `data/models/lr_model.json` is the reference copy.

The CSV column order is fixed by `printFeaturesCSV()`, the header string in `thermalTask`, and the header in `host_logger.py`.

## Data collection workflow

1. Set `#define DATA_COLLECTION_MODE true` in `src/main.cpp`, then rebuild and flash.
2. Run `python tools/host_logger.py --port /dev/ttyACM0 --out session.csv`.
3. Set the label with `L<n>` in the logger, or with the dashboard buttons (`/setlabel`). The dashboard only has buttons for 0–4.
4. Move the CSV into `data/raw/label<k>_<name>/`. **The training class comes from the folder name** (`POSITIVE_LABEL_DIRS = {"label1_human"}`), not from the `label` column. Every other folder counts as negative.
5. Run `python tools/train_classifier.py` and paste the snippet into `Detector.cpp`.

Label convention: 0 = background, 1 = human, 2 = hot cup or bottle, 3 = phone or laptop, 4 = sunlit dashboard or seat, 5 = HVAC vent, 6 = vapor. Folders `label4/5/6` are still **empty**.

## Gotchas

- `requirements.txt` lists TensorFlow and Keras, which nothing uses. The real dependencies are `numpy pandas scikit-learn` for training and `pyserial` for the logger.
- `host_logger.py` opens its output in append mode and writes the header on every run, so reusing a file produces duplicate header rows.
- Some raw CSV rows are corrupted by USB-CDC byte drops. The trainer uses `on_bad_lines="warn"` to skip them, so don't "fix" that setting.
- `deltaAmbT` uses the MI48 die temperature (header Ta), not the cabin air temperature. It is negative for humans in the current data.
- The detector hard-codes 80×62 (`DET_W`/`DET_H`). `THERMAL_WIDTH/HEIGHT` in `config.h` are separate copies of the same numbers.
- `.gitignore` excludes `myenv/`, the local Python venv.
