#!/usr/bin/env python3
"""
host_logger.py - Ghi lai CSV dac trung tu ESP32 (qua Serial USB) ra file,
                  dong thoi cho phep doi nhan (label) ngay tu ban phim
                  ma khong can mo Serial Monitor rieng.

Cach dung:
    pip install pyserial
    python host_logger.py --port COM5 --out session1.csv
    (Linux/Mac: --port /dev/ttyUSB0 hoac /dev/ttyACM0)

Trong luc chay:
    - Go "L1" roi Enter  -> gui lenh doi nhan sang ESP32 (vd nhan 1 = tay nguoi)
    - Go "q" roi Enter   -> thoat chuong trinh
    - Moi dong CSV nhan duoc tu ESP32 se duoc in ra man hinh VA ghi vao --out
      (co them cot thoi gian may tinh o dau moi dong de doi chieu sau nay)

Quy uoc nhan (sua lai cho khop voi comment trong main.cpp neu doi):
    0 = khong xac dinh/nen      4 = bang tap-lo/ghe bi nang chieu
    1 = tay nguoi               5 = cua gio suoi/dieu hoa dang thoi nong
    2 = coc/chai nuoc nong      6 = hoi nuoc (vapor)
    3 = dien thoai/laptop
"""
import argparse
import datetime
import sys
import threading

import serial


def reader_thread(ser: serial.Serial, out_fh):
    while True:
        try:
            line = ser.readline().decode("utf-8", errors="replace").strip()
        except serial.SerialException:
            print("[host_logger] Mat ket noi Serial.")
            break
        if not line:
            continue
        ts = datetime.datetime.now().isoformat(timespec="milliseconds")
        print(line)
        # bo qua dong header CSV lap lai nhieu lan va dong log khac ([Detector], [Label]...)
        if line.startswith("[") or line.startswith("label,"):
            continue
        out_fh.write(f"{ts},{line}\n")
        out_fh.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True, help="Cong Serial, vd COM5 hoac /dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--out", required=True, help="File CSV de ghi")
    args = ap.parse_args()

    ser = serial.Serial(args.port, args.baud, timeout=1)

    write_header = True
    try:
        with open(args.out, "a", encoding="utf-8") as out_fh:
            if write_header:
                out_fh.write(
                    "hostTimestamp,label,blobType,pixelCount,aspect,solidity,"
                    "deltaAmbT,trendSlope,residStd,cxJitter,areaJitter,"
                    "maxTemp,tempSpread,sampleCount\n"
                )
                out_fh.flush()

            t = threading.Thread(target=reader_thread, args=(ser, out_fh), daemon=True)
            t.start()

            print("[host_logger] Dang chay. Go 'L<so>' de doi nhan, 'q' de thoat.")
            while True:
                cmd = input().strip()
                if cmd.lower() == "q":
                    break
                if cmd:
                    ser.write((cmd + "\n").encode("utf-8"))
    finally:
        ser.close()


if __name__ == "__main__":
    sys.exit(main())
