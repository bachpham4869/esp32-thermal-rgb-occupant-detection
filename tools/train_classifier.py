#!/usr/bin/env python3
"""
train_classifier.py - Train mot logistic regression NHO tren du lieu CSV thu
thap tu host_logger.py, roi xuat trong so ra file C header de nhung thang
vao ESP32-S3 (khong can TFLite Micro cho buoc dau tien - logistic regression
la 1 phep dot-product + sigmoid, ~10 he so, chay tuc thi tren MCU).

Cach dung:
    pip install pandas scikit-learn numpy
    python train_classifier.py session1.csv session2.csv ... --out skin_model.h

QUAN TRONG ve khoa hoc: chia train/test theo SESSION (tung lan thu thap), KHONG
theo tung dong, vi cac dong trong cung 1 session tuong quan cao (cung 1 vat/tay,
cung dieu kien) -> chia theo dong se bi ro ri du lieu (data leakage), lam accuracy
ao cao hon thuc te. O day dung tham so --out cho tung file rieng bieu thi 1 session.
"""
import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [
    "aspect", "solidity", "deltaAmbT", "trendSlope",
    "residStd", "cxJitter", "areaJitter", "maxTemp", "tempSpread",
]

# Nhan 1 = tay nguoi la lop duong (positive); tat ca nhan khac (0,2,3,4,5,6...) la lop am.
POSITIVE_LABEL = 1


def load_sessions(paths):
    frames = []
    for i, p in enumerate(paths):
        df = pd.read_csv(p)
        df["session"] = i  # dung de GroupKFold - moi file = 1 session doc lap
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_files", nargs="+", help="Cac file CSV tu host_logger.py (moi file = 1 session)")
    ap.add_argument("--out", default="skin_model.h")
    args = ap.parse_args()

    df = load_sessions(args.csv_files)
    df = df.dropna(subset=FEATURE_COLS)
    df = df[df["sampleCount"] >= 5]  # bo dong chua du du lieu dai han

    X = df[FEATURE_COLS].values
    y = (df["label"] == POSITIVE_LABEL).astype(int).values
    groups = df["session"].values

    print(f"Tong so dong: {len(df)}, so session: {df['session'].nunique()}")
    print(f"Ty le lop duong (tay nguoi): {y.mean():.2%}")
    if df["session"].nunique() < 3:
        print("CANH BAO: can it nhat 3 session (thu ly tuong ~10+) de cross-validate "
              "dang tin cay va tranh overfit vao dieu kien cu the. Ket qua duoi day "
              "chi mang tinh minh hoa.")

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # Danh gia bang GroupKFold (chia theo session) truoc khi fit tren toan bo du lieu
    n_splits = min(5, df["session"].nunique())
    if n_splits >= 2:
        gkf = GroupKFold(n_splits=n_splits)
        all_true, all_pred = [], []
        for train_idx, test_idx in gkf.split(Xs, y, groups):
            clf = LogisticRegression(class_weight="balanced", max_iter=1000)
            clf.fit(Xs[train_idx], y[train_idx])
            pred = clf.predict(Xs[test_idx])
            all_true.extend(y[test_idx])
            all_pred.extend(pred)
        print("\n=== Danh gia cross-session (GroupKFold) ===")
        print(classification_report(all_true, all_pred, target_names=["non-skin", "skin/hand"]))
        print("Confusion matrix:\n", confusion_matrix(all_true, all_pred))
    else:
        print("Chi co 1 session -> khong the cross-validate, chi fit va bao accuracy tren train (KHONG dang tin cay).")

    # Fit cuoi cung tren TOAN BO du lieu de xuat ra firmware
    final_clf = LogisticRegression(class_weight="balanced", max_iter=1000)
    final_clf.fit(Xs, y)

    export_c_header(final_clf, scaler, args.out)
    print(f"\nDa xuat trong so sang {args.out}")


def export_c_header(clf, scaler, out_path):
    w = clf.coef_[0]
    b = clf.intercept_[0]
    mean = scaler.mean_
    scale = scaler.scale_

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("#pragma once\n")
        f.write("// Tu dong sinh boi train_classifier.py - KHONG sua tay.\n")
        f.write("// Model: logistic regression, feature da chuan hoa (z-score) truoc khi dot voi weight.\n")
        f.write(f"// So feature: {len(w)}\n\n")
        f.write(f"#define SKIN_MODEL_NUM_FEATURES {len(w)}\n\n")
        f.write("static const float SKIN_MODEL_FEATURE_MEAN[SKIN_MODEL_NUM_FEATURES] = {\n    ")
        f.write(", ".join(f"{v:.6f}f" for v in mean))
        f.write("\n};\n\n")
        f.write("static const float SKIN_MODEL_FEATURE_SCALE[SKIN_MODEL_NUM_FEATURES] = {\n    ")
        f.write(", ".join(f"{v:.6f}f" for v in scale))
        f.write("\n};\n\n")
        f.write("static const float SKIN_MODEL_WEIGHTS[SKIN_MODEL_NUM_FEATURES] = {\n    ")
        f.write(", ".join(f"{v:.6f}f" for v in w))
        f.write("\n};\n\n")
        f.write(f"static const float SKIN_MODEL_BIAS = {b:.6f}f;\n\n")
        f.write("// Thu tu feature PHAI khop dung voi FEATURE_COLS trong train_classifier.py:\n")
        f.write("// aspect, solidity, deltaAmbT, trendSlope, residStd, cxJitter, areaJitter, maxTemp, tempSpread\n")


if __name__ == "__main__":
    sys.exit(main())
