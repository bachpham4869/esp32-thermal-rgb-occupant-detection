#include <Arduino.h>
#include <WiFi.h>
#include <WebServer.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include "esp_camera.h"

#include "config.h"
#include "mi48.h"
#include "task_web.h"

MI48 thermal;
WebServer streamServer(81);   // Server rieng, chi phuc vu MJPEG stream

TaskHandle_t thermalTaskHandle = nullptr;
TaskHandle_t streamTaskHandle  = nullptr;

#define FRAME_RATE_DIVIDER  4
#define USE_FRAME_HEADER    true
#define SPI_CLOCK_HZ        2000000UL
#define I2C_CLOCK_HZ        100000UL

bool initCamera() {
    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer   = LEDC_TIMER_0;
    config.pin_d0 = CAM_PIN_Y2;
    config.pin_d1 = CAM_PIN_Y3;
    config.pin_d2 = CAM_PIN_Y4;
    config.pin_d3 = CAM_PIN_Y5;
    config.pin_d4 = CAM_PIN_Y6;
    config.pin_d5 = CAM_PIN_Y7;
    config.pin_d6 = CAM_PIN_Y8;
    config.pin_d7 = CAM_PIN_Y9;
    config.pin_xclk = CAM_PIN_XCLK;
    config.pin_pclk = CAM_PIN_PCLK;
    config.pin_vsync = CAM_PIN_VSYNC;
    config.pin_href = CAM_PIN_HREF;
    config.pin_sccb_sda = CAM_PIN_SIOD;
    config.pin_sccb_scl = CAM_PIN_SIOC;
    config.pin_pwdn = CAM_PIN_PWDN;
    config.pin_reset = CAM_PIN_RESET;
    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_JPEG;

    if (psramFound()) {
        Serial.println("Da phat hien PSRAM.");
        config.frame_size = FRAMESIZE_VGA;
        config.jpeg_quality = 12;
        config.fb_count = 2;
        config.fb_location = CAMERA_FB_IN_PSRAM;
        config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
    } else {
        Serial.println("CANH BAO: khong phat hien PSRAM.");
        config.frame_size = FRAMESIZE_QVGA;
        config.jpeg_quality = 15;
        config.fb_count = 1;
        config.fb_location = CAMERA_FB_IN_DRAM;
    }

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("LOI: esp_camera_init that bai, ma loi 0x%x\n", err);
        return false;
    }
    return true;
}

void setupWiFi() {
    WiFi.mode(WIFI_AP);
    WiFi.softAP(AP_SSID, AP_PASS);
    Serial.println("Da bat Access Point.");
    Serial.print("SSID: "); Serial.println(AP_SSID);
    Serial.print("IP: "); Serial.println(WiFi.softAPIP());
}

void thermalTask(void *pvParameters) {
    Serial.println("[thermalTask] Dang khoi tao MI48...");

    bool ok = thermal.begin(MI48_I2C_ADDR,
                             MI48_SDA, MI48_SCL,
                             MI48_MISO, MI48_MOSI, MI48_SCK, MI48_CS,
                             MI48_READY,
                             MI48_RESET,
                             SPI_CLOCK_HZ, I2C_CLOCK_HZ);
    if (!ok) {
        Serial.println("[thermalTask] LOI: Khong doc duoc I2C MI48.");
    }

    if (!thermal.waitForBootup(3000)) {
        Serial.println("[thermalTask] CANH BAO: chua boot xong trong 3s.");
    }

    if (!thermal.startContinuousStream(FRAME_RATE_DIVIDER, USE_FRAME_HEADER)) {
        Serial.println("[thermalTask] LOI: khong bat duoc stream.");
    } else {
        Serial.println("[thermalTask] MI48 san sang.");
    }

    for (;;) {
        thermal.poll();
        vTaskDelay(1);
    }
}

// Task rieng, chi lo phuc vu video stream tren cong 81.
// Chay doc lap voi loop() chinh nen khong lam "ket" server port 80.
void streamTask(void *pvParameters) {
    for (;;) {
        streamServer.handleClient();
        vTaskDelay(1);
    }
}

void setup() {
    Serial.begin(115200);
    delay(300);
    Serial.println("\n=== ESP32-S3-CAM: OV3660 + MI48 Fusion ===");

    if (!initCamera()) {
        Serial.println("Dung lai vi camera khong khoi tao duoc.");
        while (true) delay(1000);
    }
    Serial.println("Camera OV3660 san sang.");

    setupWiFi();
    web_init();   // khoi tao ca server (80) va streamServer (81)

    xTaskCreatePinnedToCore(thermalTask, "thermalTask", 8192, nullptr, 2, &thermalTaskHandle, 0);
    xTaskCreatePinnedToCore(streamTask,  "streamTask",  4096, nullptr, 1, &streamTaskHandle,  1);
}

void loop() {
    web_loop();   // chi lo server port 80 (trang chinh + /frame + /status)
}