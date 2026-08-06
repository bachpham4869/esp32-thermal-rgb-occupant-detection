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