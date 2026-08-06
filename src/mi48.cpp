#include "mi48.h"

MI48::MI48() {}

void MI48::hardwareReset() {
    if (_resetPin < 0) return;
    pinMode(_resetPin, OUTPUT);
    digitalWrite(_resetPin, HIGH);
    delay(5);
    digitalWrite(_resetPin, LOW);
    delayMicroseconds(200);
    digitalWrite(_resetPin, HIGH);
    delay(50);
    Serial.println("[MI48] Da thuc hien hardware reset qua nRESET.");
}

bool MI48::begin(uint8_t i2cAddr,
                  int sdaPin, int sclPin,
                  int misoPin, int mosiPin, int sckPin, int csPin,
                  int dataReadyPin,
                  int resetPin,
                  uint32_t spiClockHz,
                  uint32_t i2cClockHz) {
    _i2cAddr = i2cAddr;
    _csPin = csPin;
    _dataReadyPin = dataReadyPin;
    _resetPin = resetPin;
    _spiClockHz = spiClockHz;

    if (_mutex == nullptr) {
        _mutex = xSemaphoreCreateMutex();
    }

    pinMode(_csPin, OUTPUT);
    digitalWrite(_csPin, HIGH);
    pinMode(_dataReadyPin, INPUT);

    hardwareReset();

    Wire.begin(sdaPin, sclPin);
    Wire.setClock(i2cClockHz);

    SPI.begin(sckPin, misoPin, mosiPin, csPin);

    uint8_t dummy;
    if (!readReg(MI48_REG_STATUS, dummy)) {
        return false;
    }

    stopIfStreaming();
    senxorPowerUp();

    return true;
}

bool MI48::stopIfStreaming() {
    uint8_t mode = 0;
    if (!readReg(MI48_REG_FRAME_MODE, mode)) return false;
    if (mode & (MI48_GET_SINGLE_FRAME | MI48_CONTINUOUS_STREAM)) {
        Serial.println("[MI48] Dang stream tu truoc -> dung lai...");
        writeReg(MI48_REG_FRAME_MODE, mode & ~(MI48_GET_SINGLE_FRAME | MI48_CONTINUOUS_STREAM));
        delay(100);
    }
    return true;
}

bool MI48::senxorPowerUp() {
    bool ok = writeReg(MI48_REG_SENXOR_POWERUP, MI48_SENXOR_POWERUP_VAL);
    delay(100);
    return ok;
}

bool MI48::writeReg(uint8_t reg, uint8_t val) {
    Wire.beginTransmission(_i2cAddr);
    Wire.write(reg);
    Wire.write(val);
    return (Wire.endTransmission() == 0);
}

bool MI48::readReg(uint8_t reg, uint8_t &val) {
    Wire.beginTransmission(_i2cAddr);
    Wire.write(reg);
    if (Wire.endTransmission(false) != 0) return false;
    if (Wire.requestFrom((int)_i2cAddr, 1) != 1) return false;
    val = Wire.read();
    return true;
}

bool MI48::waitForBootup(uint32_t timeoutMs) {
    uint32_t start = millis();
    uint8_t status = 0;
    while (millis() - start < timeoutMs) {
        if (readReg(MI48_REG_STATUS, status)) {
            if ((status & MI48_STATUS_BOOTING_UP) == 0) return true;
        }
        delay(20);
    }
    return false;
}

bool MI48::startContinuousStream(uint8_t frameRateDivider, bool includeHeader) {
    _headerEnabled = includeHeader;
    if (!writeReg(MI48_REG_FRAME_RATE, frameRateDivider)) return false;
    uint8_t mode = MI48_CONTINUOUS_STREAM;
    if (!includeHeader) mode |= MI48_NO_HEADER;
    return writeReg(MI48_REG_FRAME_MODE, mode);
}

bool MI48::stopStream() {
    uint8_t mode = _headerEnabled ? 0x00 : MI48_NO_HEADER;
    return writeReg(MI48_REG_FRAME_MODE, mode);
}

bool MI48::poll() {
    if (digitalRead(_dataReadyPin) != HIGH) return false;
    return readFrameSPI();
}

bool MI48::copyPixels(uint16_t *dst) {
    if (_mutex == nullptr || !_hasFrame) return false;
    if (xSemaphoreTake(_mutex, pdMS_TO_TICKS(50)) != pdTRUE) return false;
    memcpy(dst, _pixelsSafe, sizeof(_pixelsSafe));
    xSemaphoreGive(_mutex);
    return true;
}

uint16_t MI48::crc16_ccitt_false(const uint8_t *data, size_t len) {
    uint16_t crc = 0xFFFF;
    for (size_t i = 0; i < len; i++) {
        crc ^= (uint16_t)data[i] << 8;
        for (int b = 0; b < 8; b++) {
            if (crc & 0x8000) crc = (crc << 1) ^ 0x1021;
            else crc <<= 1;
        }
    }
    return crc;
}

bool MI48::readFrameSPI() {
    SPI.beginTransaction(SPISettings(_spiClockHz, MSBFIRST, SPI_MODE0));
    digitalWrite(_csPin, LOW);

    if (_headerEnabled) {
        for (int i = 0; i < MI48_HEADER_WORDS; i++) {
            _headerBuf[i] = SPI.transfer16(0x0000);
        }
    }
    for (int i = 0; i < MI48_NUM_PIXELS; i++) {
        _pixels[i] = SPI.transfer16(0x0000);
    }

    digitalWrite(_csPin, HIGH);
    SPI.endTransaction();

    if (_headerEnabled) {
        _frameCounter = _headerBuf[MI48_HDR_FRCNT];
        _maxTemp      = _headerBuf[MI48_HDR_MAXV];
        _minTemp      = _headerBuf[MI48_HDR_MINV];
        _crcExpected  = _headerBuf[MI48_HDR_CRC];
        _crcCalculated = crc16_ccitt_false((const uint8_t*)_pixels, sizeof(_pixels));
        _crcOk = (_crcCalculated == _crcExpected);

        _timestamp   = ((uint32_t)_headerBuf[MI48_HDR_TIME + 1] << 16) | _headerBuf[MI48_HDR_TIME];
        _senxorVdd   = _headerBuf[MI48_HDR_VDD] / 10000.0f;
        _senxorTempC = (_headerBuf[MI48_HDR_TA] / 100.0f) - 273.15f;
    } else {
        _frameCounter++;
        uint16_t mn = 0xFFFF, mx = 0;
        for (int i = 0; i < MI48_NUM_PIXELS; i++) {
            uint16_t v = _pixels[i];
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
        _minTemp = mn;
        _maxTemp = mx;
        _crcOk = true;
    }

    if (_mutex != nullptr) {
        xSemaphoreTake(_mutex, portMAX_DELAY);
        memcpy(_pixelsSafe, _pixels, sizeof(_pixels));
        _hasFrame = true;
        xSemaphoreGive(_mutex);
    }

    return true;
}