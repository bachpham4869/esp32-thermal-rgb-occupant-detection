#pragma once

// ============================================================
// Chan camera OV3660 (co dinh, khong doi vi board han cung)
// ============================================================
#define CAM_PIN_SIOD    4
#define CAM_PIN_SIOC    5
#define CAM_PIN_VSYNC   6
#define CAM_PIN_HREF    7
#define CAM_PIN_XCLK    15
#define CAM_PIN_Y9      16
#define CAM_PIN_Y8      17
#define CAM_PIN_Y7      18
#define CAM_PIN_Y4      8
#define CAM_PIN_Y3      9
#define CAM_PIN_Y5      10
#define CAM_PIN_Y2      11
#define CAM_PIN_Y6      12
#define CAM_PIN_PCLK    13
#define CAM_PIN_PWDN    -1
#define CAM_PIN_RESET   -1

// ============================================================
// Chan MI48 - TACH RIENG khoi camera (dung Wire1 / I2C1)
// ============================================================
#define MI48_SDA        42
#define MI48_SCL        41
#define MI48_MISO       14
#define MI48_MOSI       47
#define MI48_SCK        21
#define MI48_CS         1
#define MI48_READY      2
#define MI48_RESET      40
#define MI48_I2C_ADDR   0x40

// thermal size
#define THERMAL_WIDTH  80
#define THERMAL_HEIGHT 62

// WiFi - ESP32 tu phat Access Point
#define AP_SSID "YOLO_FUSION"
#define AP_PASS "12345678"

// ============================================================
// Nguong cho module phat hien nguoi/tre em bi bo quen (detector.h)
// Can hieu chinh thuc te tren xe that - day chi la gia tri khoi diem.
// ============================================================
#define DET_WARMUP_FRAMES      30      // so frame dau tien dung de hoc nen on dinh (khong detect)
#define DET_BG_ALPHA           0.02f   // toc do cap nhat nen (EMA) o vung "giong nen"
#define DET_FG_DIFF_THRESH_C   2.0f    // do lech nhiet (C) de coi 1 pixel la "khac nen"

#define DET_MAX_TEMP_C         58.0f   // tran nhiet - vuot qua thi loai (ca phe nong, binh nuoc soi)
#define DET_MIN_BLOB_PIXELS    15      // kich thuoc toi thieu (loai dien thoai, coc nuoc)
#define DET_MIN_PERSON_PIXELS  60      // kich thuoc de coi la "nguoi ro rang"

#define DET_ASPECT_MIN         0.35f   // ti le w/h duoi nguong nay -> hinh dang bat thuong
#define DET_ASPECT_MAX         2.8f    // ti le w/h tren nguong nay -> hinh dang bat thuong

#define DET_GRAY_HISTORY_LEN   8       // so frame luu de tinh phuong sai nhiet cua vung xam
#define DET_GRAY_VAR_THRESH    0.03f   // phuong sai nhiet (C^2) toi thieu de coi la "co dao dong giong tho"

#define DET_CONFIRM_N          5       // so frame co candidate can trong cua so M de CONFIRMED
#define DET_CONFIRM_M          7       // do rong cua so debounce (frame)
#define DET_RESET_EMPTY        10      // so frame lien tiep khong co candidate de quay ve IDLE

// ============================================================
// Buffer dac trung DAI HAN - phuc vu thu thap du lieu / phan biet
// tay nguoi vs vat nong khac (ML). Lay mau tho, chu ky cham (~1s/mau)
// khac voi _windowFrames (debounce, nhanh) va _tracked.history (gray-zone,
// ngan). Muc dich: bat xu huong nhiet dai han + do "run/jitter" vi tri.
// ============================================================
#define DET_LT_HISTORY_LEN     64      // so mau dai han luu tru (vd 64 mau x 1s = ~64s)
#define DET_LT_SAMPLE_MS       1000UL  // chu ky lay 1 mau dac trung dai han (ms)