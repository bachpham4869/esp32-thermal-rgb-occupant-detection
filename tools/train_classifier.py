#!/usr/bin/env python3
"""
train_classifier.py - Train logistic regression cho CPD (Child/Person Detection)
Phan biet nguoi that (label1_human) vs nhiet gia (label2_hotcup, label3_phone, label4_sunlit...)

Input: cac file CSV trong data/raw/<ten_folder_label>/*.csv, dung header:
label,blobType,pixelCount,aspect,solidity,deltaAmbT,trendSlope,residStd,cxJitter,areaJitter,maxTemp,tempSpread,sampleCount
(day chinh la header ma main.cpp in ra Serial khi DATA_COLLECTION_MODE=true)

Output:
- In accuracy / confusion matrix / classification report tren tap test
- In he so LR da "gap" scaler, dang C array de dan thang vao Detector.cpp
- Luu model ra data/models/lr_model.json (de doi chieu / train lai sau nay)

Chay:
    python3 tools/train_classifier.py
    python3 tools/train_classifier.py --raw-dir data/raw --test-size 0.25
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix,
                              classification_report, roc_auc_score)
from sklearn.preprocessing import StandardScaler

# Thu tu feature NAY QUYET DINH thu tu he so xuat ra - phai khop tay voi
# thu tu dot-product trong Detector.cpp. Khong tu y doi thu tu sau khi da
# hardcode, tru khi sua ca 2 ben dong bo.
FEATURE_COLS = [
    "aspect", "solidity", "deltaAmbT", "trendSlope",
    "residStd", "cxJitter", "areaJitter", "maxTemp", "tempSpread",
]

# Folder nao la nguoi that (positive=1). Con lai (label2_hotcup, label3_phone,
# label4_sunlit, ...) tu dong la negative=0 - khong can liet ke het,
# de them class negative moi chi can them thu muc, khong sua code.
POSITIVE_LABEL_DIRS = {"label1_human"}

# Khop voi computeFeatures() trong Detector.cpp: features.valid=false neu
# chua du 5 mau dai han -> loai bo cac dong chua du tin cay truoc khi train.
MIN_SAMPLE_COUNT = 5


def load_all_csv(raw_dir: str) -> pd.DataFrame:
    frames = []
    if not os.path.isdir(raw_dir):
        print(f"LOI: khong thay thu muc {raw_dir}")
        sys.exit(1)

    for label_dir in sorted(os.listdir(raw_dir)):
        full_dir = os.path.join(raw_dir, label_dir)
        if not os.path.isdir(full_dir):
            continue
        csv_files = sorted(glob.glob(os.path.join(full_dir, "*.csv")))
        if not csv_files:
            continue
        for f in csv_files:
            try:
                # on_bad_lines="warn": chi bo qua DUNG dong bi loi (thuong do
                # byte bi rot/lap khi truyen qua USB-CDC luc in nhieu blob lien
                # tiep), khong huy ca file - khac voi mac dinh cua pandas la
                # nem exception va mat toan bo file chi vi 1-2 dong hong.
                df = pd.read_csv(f, on_bad_lines="warn", engine="python")
            except Exception as e:
                print(f"[WARN] Bo qua {f}: {e}")
                continue
            if df.empty:
                print(f"[WARN] File rong: {f}")
                continue
            df["source_file"] = os.path.basename(f)
            df["label_dir"] = label_dir
            df["is_human"] = 1 if label_dir in POSITIVE_LABEL_DIRS else 0
            frames.append(df)
            print(f"[OK] {f}: {len(df)} dong")

    if not frames:
        print(f"LOI: khong tim thay CSV nao trong {raw_dir}")
        sys.exit(1)
    return pd.concat(frames, ignore_index=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    missing_cols = [c for c in FEATURE_COLS + ["sampleCount"] if c not in df.columns]
    if missing_cols:
        print(f"LOI: CSV thieu cot {missing_cols} - kiem tra lai header co dung format khong.")
        sys.exit(1)

    df = df[df["sampleCount"] >= MIN_SAMPLE_COUNT].copy()
    df = df.dropna(subset=FEATURE_COLS)
    for col in FEATURE_COLS:
        df = df[np.isfinite(df[col])]
    after = len(df)
    print(f"Loc du lieu: {before} -> {after} dong "
          f"(bo sampleCount<{MIN_SAMPLE_COUNT} va NaN/Inf)")
    return df


def print_class_balance(df: pd.DataFrame):
    print("\n=== Phan bo theo folder / class ===")
    print(df.groupby(["label_dir", "is_human"]).size().to_string())
    print("\n=== Tong positive(human) / negative(fake heat) ===")
    print(df["is_human"].value_counts().to_string())


def train(df: pd.DataFrame, test_size: float, seed: int):
    X = df[FEATURE_COLS].values
    y = df["is_human"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # class_weight="balanced": bu lai neu label1_human co nhieu dong hon
    # nhieu (~240KB) so voi label2/label3 (moi label 1 file) - tranh model
    # thien vi ve phia class nhieu du lieu hon.
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    clf.fit(X_train_s, y_train)

    y_pred = clf.predict(X_test_s)
    y_proba = clf.predict_proba(X_test_s)[:, 1]

    print("\n=== KET QUA TREN TAP TEST ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    if len(np.unique(y_test)) > 1:
        print("ROC AUC :", roc_auc_score(y_test, y_proba))
    print("\nConfusion matrix (hang=that, cot=du doan) [0=fake_heat, 1=human]:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["fake_heat", "human"]))

    return clf, scaler


def fold_scaler_into_lr(clf: LogisticRegression, scaler: StandardScaler):
    """
    Train tren du lieu da StandardScaler (mean=0, std=1) de LR hoi tu on dinh,
    nhung ESP32 KHONG nen chuan hoa runtime (tốn RAM luu mean/std + tinh toan).
    Gop truc tiep scaler vao he so LR de Detector.cpp chi can 1 phep dot-product
    tren feature tho:
        z_scaled = w . ((x - mean) / scale) + b
                 = (w/scale) . x + (b - sum(w*mean/scale))
                 = w_raw . x + b_raw
    """
    w = clf.coef_[0]
    b = clf.intercept_[0]
    mean = scaler.mean_
    scale = scaler.scale_

    w_raw = w / scale
    b_raw = b - np.sum(w * mean / scale)
    return w_raw, b_raw


def print_cpp_snippet(w_raw, b_raw):
    print("\n=== DAN DOAN NAY VAO Detector.cpp (xem huong dan tich hop kem theo) ===\n")
    print("// Sinh tu dong boi tools/train_classifier.py - KHONG sua tay so,")
    print("// chay lai script va dan lai neu doi/them data.")
    print(f"static const float LR_WEIGHTS[{len(FEATURE_COLS)}] = {{")
    for name, w in zip(FEATURE_COLS, w_raw):
        print(f"    {w:.8f}f,  // {name}")
    print("};")
    print(f"static const float LR_BIAS = {b_raw:.8f}f;")
    print(f"// Thu tu feature dua vao dot-product PHAI dung: {FEATURE_COLS}")


def save_model_json(w_raw, b_raw, out_path):
    payload = {
        "feature_order": FEATURE_COLS,
        "weights": list(map(float, w_raw)),
        "bias": float(b_raw),
        "formula": "z = dot(weights, features) + bias; p_human = sigmoid(z)",
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nDa luu model -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--out", default="data/models/lr_model.json")
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = load_all_csv(args.raw_dir)
    print_class_balance(df)
    df = clean(df)
    print_class_balance(df)

    if df["is_human"].nunique() < 2:
        print("\nLOI: chi con 1 class sau khi loc - khong the train binary classifier.")
        print("Kiem tra sampleCount trong CSV hoac thu them data class con thieu.")
        sys.exit(1)

    min_class_count = df["is_human"].value_counts().min()
    if min_class_count < 20:
        print(f"\nCANH BAO: class it nhat chi co {min_class_count} dong sau loc - "
              f"model se khong on dinh, nen thu them data truoc khi tin ket qua nay.")

    clf, scaler = train(df, args.test_size, args.seed)
    w_raw, b_raw = fold_scaler_into_lr(clf, scaler)
    print_cpp_snippet(w_raw, b_raw)
    save_model_json(w_raw, b_raw, args.out)


if __name__ == "__main__":
    main()