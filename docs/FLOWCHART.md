# System Flowcharts

These are Mermaid diagrams. They render on GitHub, in VS Code (Markdown Preview Mermaid), or at https://mermaid.live, where you can export them as SVG or PNG for the paper.

## 1. Hardware & software architecture

```mermaid
flowchart LR
    subgraph Sensors
        MI48["MI48 SenXor<br/>80×62 thermal<br/>(I2C ctrl + SPI data)"]
        OV["OV3660 RGB<br/>VGA JPEG"]
    end

    subgraph ESP32-S3["ESP32-S3 (dual core, 8 MB PSRAM)"]
        direction TB
        subgraph Core0["Core 0"]
            TT["thermalTask (prio 2)<br/>MI48::poll → OccupantDetector::update"]
        end
        subgraph Core1["Core 1"]
            ST["streamTask (prio 1)<br/>MJPEG :81/stream"]
            LP["loop() → WebServer :80<br/>/ · /frame · /status · /setlabel"]
        end
        MX[("Mutex-protected<br/>_resultSafe / _labelsSafe<br/>_pixelsSafe")]
        TT --> MX
        MX --> LP
    end

    MI48 -- "DATA_READY + SPI 2 MHz" --> TT
    OV -- "DVP 8-bit" --> ST
    LP -- "Wi-Fi AP 'YOLO_FUSION'" --> Browser["Browser dashboard<br/>(alert banner, thermal canvas,<br/>RGB stream, pHuman)"]
    ST --> Browser
    TT -- "Serial CSV (DATA_COLLECTION_MODE)" --> Host["host_logger.py"]
```

## 2. Per-frame detection pipeline (`OccupantDetector::update`)

```mermaid
flowchart TD
    A([New MI48 frame<br/>4960 × uint16, 0.1 K]) --> B[Convert to °C<br/>T = raw/10 − 273.15]
    B --> C[3×3 box blur]
    C --> D{Warm-up?<br/>frames < 30}
    D -- yes --> D1[Background EMA α = 0.3] --> D2([state = WARMUP, no detection])
    D -- no --> E[Selective background EMA<br/>α = 0.02 bg, 0.002 fg]
    E --> F[Foreground mask<br/>T − bg > 2 °C]
    F --> G[Dilate 1 px, 8-neighbour]
    G --> H[BFS connected components<br/>≤ 8 blobs, bbox / avg / max / min T]
    H --> I[[For each blob]]

    I --> R1{maxT > 58 °C?}
    R1 -- yes --> REJ[REJECTED]
    R1 -- no --> R2{pixels < 15?}
    R2 -- yes --> REJ
    R2 -- no --> R3{aspect ∉ 0.35–2.8?}
    R3 -- yes --> LOW[LOW_SUSPECT]
    R3 -- no --> R4{pixels ≥ 60?}
    R4 -- yes --> PER[PERSON]
    R4 -- no --> GZ[GRAY_ZONE]
    GZ --> V{Var avgT over 8 frames<br/>≥ 0.03 °C²?}
    V -- yes --> CAN[CANDIDATE]
    V -- no --> REJ

    PER & CAN & LOW & REJ --> TR{pixels ≥ 15?}
    TR -- yes --> TK[Track: nearest centroid < 6 px<br/>long-term sample every 1 s, 64 deep]
    TK --> FE{≥ 5 long-term samples?}
    FE -- yes --> LR[Compute 9 features<br/>pHuman = σ w·x + b]
    LR --> GATE{type ∈ PERSON/CANDIDATE<br/>and pHuman < 0.5?}
    GATE -- yes --> REJ2[Demote → REJECTED]
    GATE -- no --> KEEP[keep type]
    FE -- no --> KEEP
    TR -- no --> KEEP

    REJ2 & KEEP --> HC{any PERSON / CANDIDATE?}
    HC --> SM[Debounce state machine]
    SM --> OUT([Publish result + label grid under mutex])
```

## 3. Debounce state machine

```mermaid
stateDiagram-v2
    [*] --> WARMUP
    WARMUP --> IDLE: 30 frames learned
    IDLE --> CANDIDATE: hasCandidate
    CANDIDATE --> CONFIRMED: ≥ 5 hits in last 7 frames
    CANDIDATE --> IDLE: 10 consecutive empty frames
    CONFIRMED --> ALERT: next frame
    ALERT --> IDLE: 10 consecutive empty frames
    IDLE --> WARMUP: resetBackground()
    CANDIDATE --> WARMUP: resetBackground()
    ALERT --> WARMUP: resetBackground()
```

## 4. Long-term feature extraction (per tracked blob)

```mermaid
flowchart LR
    S["Ring buffer (64 × 1 s)<br/>avgT, cx, cy, area"] --> L["Least-squares fit avgT vs t"]
    L --> F1["trendSlope (°C/min)"]
    L --> F2["residStd (°C)<br/>breathing / micro-motion"]
    S --> F3["cxJitter = ½(std cx + std cy)"]
    S --> F4["areaJitter = std(area)"]
    B["Current blob"] --> F5["aspect = w/h"]
    B --> F6["solidity = px / (w·h)"]
    B --> F7["maxTemp"]
    B --> F8["tempSpread = maxT − minT"]
    B --> F9["deltaAmbT = avgT − Ta(MI48)"]
    F1 & F2 & F3 & F4 & F5 & F6 & F7 & F8 & F9 --> X["x ∈ ℝ⁹ → LR → pHuman"]
```

## 5. Offline ML workflow (data → model → firmware)

```mermaid
flowchart TD
    A["Set DATA_COLLECTION_MODE = true<br/>flash firmware"] --> B["Stage scene in car<br/>(person / cup / phone ...)"]
    B --> C["host_logger.py<br/>L&lt;n&gt; or web /setlabel"]
    C --> D["session.csv<br/>hostTimestamp + 13 columns"]
    D --> E["data/raw/label&lt;k&gt;_&lt;name&gt;/"]
    E --> F["train_classifier.py<br/>filter sampleCount ≥ 5, drop NaN<br/>class = folder (label1_human = 1)"]
    F --> G["StandardScaler + LogisticRegression<br/>class_weight = balanced, 75/25 split"]
    G --> H["Fold scaler into weights<br/>w_raw = w/σ, b_raw = b − Σ wμ/σ"]
    H --> I["C snippet → Detector.cpp<br/>lr_model.json"]
    I --> J["Rebuild & flash"]
    J --> B
```
