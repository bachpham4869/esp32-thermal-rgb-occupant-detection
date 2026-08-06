#pragma once
#include <Arduino.h>
#include <SPI.h>
#include <Wire.h>
#include <freertos/FreeRTOS.h>
#include <freertos/semphr.h>

#define MI48_REG_SENXOR_POWERUP    0xB0
#define MI48_REG_FRAME_MODE        0xB1
#define MI48_REG_FW_VERSION_1      0xB2
#define MI48_REG_FW_VERSION_2      0xB3
#define MI48_REG_FRAME_RATE        0xB4
#define MI48_REG_SLEEP_MODE        0xB5
#define MI48_REG_STATUS            0xB6
#define MI48_REG_CLK_SPEED         0xB7
#define MI48_REG_SENXOR_TYPE       0xBA
#define MI48_REG_SENSITIVITY       0xC2
#define MI48_REG_EMISSIVITY        0xCA
#define MI48_REG_OFFSET_CORR       0xCB
#define MI48_REG_SENXOR_ID         0xE0
#define MI48_REG_FILTER_CONTROL    0xD0
#define MI48_REG_FILTER_SETTING_1  0xD1
#define MI48_REG_FILTER_SETTING_2  0xD3
#define MI48_REG_USER_FLASH_CTRL   0xD8

#define MI48_SENXOR_POWERUP_VAL    0x13

#define MI48_GET_SINGLE_FRAME   (1 << 0)
#define MI48_CONTINUOUS_STREAM  (1 << 1)
#define MI48_NO_HEADER          (1 << 5)

#define MI48_STATUS_READOUT_TOO_SLOW (1 << 1)
#define MI48_STATUS_SENXOR_IF_ERROR  (1 << 2)
#define MI48_STATUS_CAPTURE_ERROR    (1 << 3)
#define MI48_STATUS_DATA_READY_BIT   (1 << 4)
#define MI48_STATUS_BOOTING_UP       (1 << 5)

#define MI48_SENXOR_WIDTH   80
#define MI48_SENXOR_HEIGHT  62
#define MI48_NUM_PIXELS     (MI48_SENXOR_WIDTH * MI48_SENXOR_HEIGHT)
#define MI48_HEADER_WORDS   80

#define MI48_HDR_FRCNT  0
#define MI48_HDR_VDD    1
#define MI48_HDR_TA     2
#define MI48_HDR_TIME   3
#define MI48_HDR_MAXV   5
#define MI48_HDR_MINV   6
#define MI48_HDR_CRC    7

class MI48 {
public:
    MI48();

    bool begin(uint8_t i2cAddr,
               int sdaPin, int sclPin,
               int misoPin, int mosiPin, int sckPin, int csPin,
               int dataReadyPin,
               int resetPin,
               uint32_t spiClockHz = 2000000,
               uint32_t i2cClockHz = 100000);

    bool waitForBootup(uint32_t timeoutMs = 3000);
    bool startContinuousStream(uint8_t frameRateDivider = 4, bool includeHeader = false);
    bool stopStream();

    bool poll();

    const uint16_t* pixels() const { return _pixels; }
    bool copyPixels(uint16_t *dst);

    uint16_t frameCounter() const { return _frameCounter; }
    uint16_t minRaw() const { return _minTemp; }
    uint16_t maxRaw() const { return _maxTemp; }

    bool headerEnabled() const { return _headerEnabled; }
    bool crcOk() const { return _crcOk; }
    uint16_t crcExpected() const { return _crcExpected; }
    uint16_t crcCalculated() const { return _crcCalculated; }

    // Chi co gia tri that khi startContinuousStream(..., includeHeader=true)
    uint32_t timestamp() const { return _timestamp; }
    float senxorVdd() const { return _senxorVdd; }
    float senxorTempC() const { return _senxorTempC; }

    bool writeReg(uint8_t reg, uint8_t val);
    bool readReg(uint8_t reg, uint8_t &val);

    static inline float rawToCelsius(uint16_t raw) {
        return (raw / 10.0f) - 273.15f;
    }

private:
    uint8_t  _i2cAddr;
    int      _csPin;
    int      _dataReadyPin;
    int      _resetPin = -1;
    uint32_t _spiClockHz;
    bool     _headerEnabled = false;

    uint16_t _pixels[MI48_NUM_PIXELS];
    uint16_t _pixelsSafe[MI48_NUM_PIXELS];
    uint16_t _headerBuf[MI48_HEADER_WORDS];
    uint16_t _frameCounter = 0;
    uint16_t _minTemp = 0;
    uint16_t _maxTemp = 0;
    bool     _hasFrame = false;

    bool     _crcOk = false;
    uint16_t _crcExpected = 0;
    uint16_t _crcCalculated = 0;

    uint32_t _timestamp = 0;
    float    _senxorVdd = 0;
    float    _senxorTempC = 0;

    SemaphoreHandle_t _mutex = nullptr;

    bool readFrameSPI();
    bool stopIfStreaming();
    bool senxorPowerUp();
    void hardwareReset();
    static uint16_t crc16_ccitt_false(const uint8_t *data, size_t len);
};