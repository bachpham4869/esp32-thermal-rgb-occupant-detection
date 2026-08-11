#include <Arduino.h>
#include "Detector.h"
#include <math.h>
#include <string.h>

OccupantDetector::OccupantDetector() {}

void OccupantDetector::begin() {
    if (_mutex == nullptr) {
        _mutex = xSemaphoreCreateMutex();
    }
    memset(_background, 0, sizeof(_background));
    memset(_tracked, 0, sizeof(_tracked));
    memset(&_result, 0, sizeof(_result));
    memset(&_resultSafe, 0, sizeof(_resultSafe));
    memset(_windowFrames, 0, sizeof(_windowFrames));
    memset(_labelsSafe, -1, sizeof(_labelsSafe));
    _bgInit = false;
    _warmupFrames = 0;
    _state = STATE_IDLE;
    _windowIdx = 0;
    _emptyStreak = 0;
}

void OccupantDetector::resetBackground() {
    _bgInit = false;
    _warmupFrames = 0;
    memset(_tracked, 0, sizeof(_tracked));
    _state = STATE_IDLE;
    memset(_windowFrames, 0, sizeof(_windowFrames));
    _emptyStreak = 0;
    Serial.println("[Detector] Da xoa nen - bat dau hoc lai (warmup).");
}

// ------------------------------------------------------------
// Loc nhieu 3x3 (box blur) - anh nhiet tho co nhieu dang "hat tieu",
// khong loc se lam blob bi vo thanh nhieu manh nho khong dat kich thuoc toi thieu.
// ------------------------------------------------------------
void OccupantDetector::smooth3x3(const float* src, float* dst) {
    for (int y = 0; y < DET_H; y++) {
        for (int x = 0; x < DET_W; x++) {
            float sum = 0.0f;
            int cnt = 0;
            for (int dy = -1; dy <= 1; dy++) {
                int ny = y + dy;
                if (ny < 0 || ny >= DET_H) continue;
                for (int dx = -1; dx <= 1; dx++) {
                    int nx = x + dx;
                    if (nx < 0 || nx >= DET_W) continue;
                    sum += src[ny * DET_W + nx];
                    cnt++;
                }
            }
            dst[y * DET_W + x] = sum / cnt;
        }
    }
}

// ------------------------------------------------------------
// Tru nen: EMA rieng cho tung pixel.
// learnFast=true (giai doan warmup): hoc gan nhu trung binh cong, hoi tu nhanh
// ve canh nen thuc te, khong quan tam vat the (gia dinh khung nhin luc nay trong/
// on dinh - nguoi dung nen bam nut "Xoa nen" khi khung nhin khong co nguoi).
// learnFast=false (hoat dong binh thuong): cap nhat rat cham o vung dang la
// "vat the moi" de khong nuot mat nguoi dung yen lau, cap nhat binh thuong o vung nen.
// ------------------------------------------------------------
void OccupantDetector::updateBackground(const float* tempC, bool learnFast) {
    if (!_bgInit) {
        memcpy(_background, tempC, sizeof(_background));
        _bgInit = true;
        return;
    }
    for (int i = 0; i < DET_NUM_PIXELS; i++) {
        float diff = tempC[i] - _background[i];
        float alpha;
        if (learnFast) {
            alpha = 0.3f;
        } else {
            alpha = (fabsf(diff) > DET_FG_DIFF_THRESH_C) ? (DET_BG_ALPHA * 0.1f) : DET_BG_ALPHA;
        }
        _background[i] += alpha * diff;
    }
}

void OccupantDetector::buildForegroundMask(const float* tempC) {
    for (int i = 0; i < DET_NUM_PIXELS; i++) {
        float diff = tempC[i] - _background[i];
        _fgMask[i] = (diff > DET_FG_DIFF_THRESH_C) ? 1 : 0;
    }
}

// ------------------------------------------------------------
// Dilation 1 pixel, 8-lang gieng: noi cac vung tien canh bi nhieu cat roi rac
// thanh 1 khoi lien tuc, truoc khi gom blob. Day la buoc quan trong nhat de
// khac phuc hien tuong "nguoi ngoi ngay truoc mat van khong nhan ra" do nhieu.
// ------------------------------------------------------------
void OccupantDetector::dilateMask(const uint8_t* src, uint8_t* dst) {
    for (int y = 0; y < DET_H; y++) {
        for (int x = 0; x < DET_W; x++) {
            int idx = y * DET_W + x;
            if (src[idx]) { dst[idx] = 1; continue; }
            uint8_t v = 0;
            for (int dy = -1; dy <= 1 && !v; dy++) {
                int ny = y + dy;
                if (ny < 0 || ny >= DET_H) continue;
                for (int dx = -1; dx <= 1; dx++) {
                    int nx = x + dx;
                    if (nx < 0 || nx >= DET_W) continue;
                    if (src[ny * DET_W + nx]) { v = 1; break; }
                }
            }
            dst[idx] = v;
        }
    }
}

// ------------------------------------------------------------
// Connected component labeling bang BFS lap (khong de quy), chay tren mask
// da dilate. Thong ke (kich thuoc/nhiet do) tinh tren toan bo vung da gom.
// ------------------------------------------------------------
int OccupantDetector::labelBlobs(const float* tempC, const uint8_t* mask, DetBlob* outBlobs, int maxBlobs) {
    memset(_labels, -1, sizeof(_labels));
    int numBlobs = 0;

    for (int start = 0; start < DET_NUM_PIXELS; start++) {
        if (numBlobs >= maxBlobs) break;
        if (_labels[start] != -1) continue;
        if (!mask[start]) continue;

        int qHead = 0, qTail = 0;
        _bfsQueue[qTail++] = (uint16_t)start;
        _labels[start] = numBlobs;

        int minX = start % DET_W, maxX = minX;
        int minY = start / DET_W, maxY = minY;
        int count = 0;
        float sumTemp = 0.0f, maxTemp = -1000.0f, minTemp = 1000.0f;

        while (qHead < qTail) {
            uint16_t idx = _bfsQueue[qHead++];
            int x = idx % DET_W, y = idx / DET_W;
            float t = tempC[idx];
            count++;
            sumTemp += t;
            if (t > maxTemp) maxTemp = t;
            if (t < minTemp) minTemp = t;
            if (x < minX) minX = x;
            if (x > maxX) maxX = x;
            if (y < minY) minY = y;
            if (y > maxY) maxY = y;

            static const int dx[4] = {1, -1, 0, 0};
            static const int dy[4] = {0, 0, 1, -1};
            for (int k = 0; k < 4; k++) {
                int nx = x + dx[k], ny = y + dy[k];
                if (nx < 0 || nx >= DET_W || ny < 0 || ny >= DET_H) continue;
                int nIdx = ny * DET_W + nx;
                if (_labels[nIdx] != -1) continue;
                if (!mask[nIdx]) continue;
                _labels[nIdx] = numBlobs;
                _bfsQueue[qTail++] = (uint16_t)nIdx;
            }
        }

        DetBlob &b = outBlobs[numBlobs];
        b.minX = (uint8_t)minX; b.maxX = (uint8_t)maxX;
        b.minY = (uint8_t)minY; b.maxY = (uint8_t)maxY;
        b.pixelCount = (uint16_t)count;
        b.avgTempC = sumTemp / count;
        b.maxTempC = maxTemp;
        b.minTempC = minTemp;
        b.type = BLOB_REJECTED;
        numBlobs++;
    }
    return numBlobs;
}

// ------------------------------------------------------------
// Bo loc 1 (nhiet tran) -> 2 (kich thuoc) -> 3 (hinh dang)
// ------------------------------------------------------------
void OccupantDetector::classify(DetBlob &b) {
    if (b.maxTempC > DET_MAX_TEMP_C) { b.type = BLOB_REJECTED; return; }
    if (b.pixelCount < DET_MIN_BLOB_PIXELS) { b.type = BLOB_REJECTED; return; }

    float w = (b.maxX - b.minX + 1);
    float h = (b.maxY - b.minY + 1);
    float aspect = w / h;
    bool aspectOk = (aspect >= DET_ASPECT_MIN && aspect <= DET_ASPECT_MAX);

    if (!aspectOk) { b.type = BLOB_LOW_SUSPECT; return; }

    if (b.pixelCount >= DET_MIN_PERSON_PIXELS) {
        b.type = BLOB_PERSON;       // size lon, ro rang giong nguoi
    } else {
        b.type = BLOB_GRAY_ZONE;    // size trung binh, hinh dang hop ly -> theo doi them
    }
}

// ------------------------------------------------------------
// Khop blob voi tracked blob gan nhat (theo centroid) va tinh
// phuong sai nhiet trung binh qua N frame gan nhat (giong tho).
// ------------------------------------------------------------
// ------------------------------------------------------------
// Khop blob voi tracked blob gan nhat (theo centroid), hoac cap phat
// track moi neu khong tim thay. DUY NHAT noi lam viec nay - ca
// trackAndGetVariance() (gray-zone, tan so cao) va computeFeatures()
// (dai han, ML) deu dung chung 1 track qua ham nay de tranh 2 blob
// cung mot vat the bi gan vao 2 track khac nhau.
// ------------------------------------------------------------
int OccupantDetector::findOrAssignTrack(DetBlob &b) {
    float cx = (b.minX + b.maxX) / 2.0f;
    float cy = (b.minY + b.maxY) / 2.0f;

    int best = -1;
    float bestDist = 6.0f;
    for (int i = 0; i < DET_MAX_BLOBS; i++) {
        if (!_tracked[i].active) continue;
        float ddx = _tracked[i].cx - cx, ddy = _tracked[i].cy - cy;
        float d = sqrtf(ddx * ddx + ddy * ddy);
        if (d < bestDist) { bestDist = d; best = i; }
    }
    if (best == -1) {
        for (int i = 0; i < DET_MAX_BLOBS; i++) {
            if (!_tracked[i].active) { best = i; break; }
        }
        if (best == -1) return -1;  // het slot track, bo qua blob nay
        _tracked[best].active = true;
        _tracked[best].histCount = 0;
        _tracked[best].histIdx = 0;
        _tracked[best].missedFrames = 0;
        _tracked[best].ltCount = 0;
        _tracked[best].ltIdx = 0;
        _tracked[best].lastLtSampleMs = 0;
    }

    _tracked[best].cx = cx;
    _tracked[best].cy = cy;
    _tracked[best].missedFrames = 0;
    return best;
}

float OccupantDetector::trackAndGetVariance(DetBlob &b) {
    int idx = findOrAssignTrack(b);
    if (idx < 0) return 0.0f;
    TrackedBlob &t = _tracked[idx];

    t.history[t.histIdx] = b.avgTempC;
    t.histIdx = (t.histIdx + 1) % DET_GRAY_HISTORY_LEN;
    if (t.histCount < DET_GRAY_HISTORY_LEN) t.histCount++;

    if (t.histCount < 3) return 0.0f;

    float mean = 0.0f;
    for (int i = 0; i < t.histCount; i++) mean += t.history[i];
    mean /= t.histCount;

    float var = 0.0f;
    for (int i = 0; i < t.histCount; i++) {
        float d = t.history[i] - mean;
        var += d * d;
    }
    var /= t.histCount;
    return var;
}

// ------------------------------------------------------------
// Lay 1 mau dai han (~1 mau/giay) vao ring buffer cua track idx.
// Chu ky cham hon nhieu so voi frame rate cam bien - muc dich la
// bat XU HUONG nhiet do va do "run" vi tri/kich thuoc qua hang chuc
// giay, khong phai dao dong tung frame.
// ------------------------------------------------------------
void OccupantDetector::sampleLongTerm(int idx, DetBlob &b, uint32_t nowMs) {
    TrackedBlob &t = _tracked[idx];
    if (t.lastLtSampleMs != 0 && (nowMs - t.lastLtSampleMs) < DET_LT_SAMPLE_MS) return;
    t.lastLtSampleMs = nowMs;

    t.ltTemp[t.ltIdx] = b.avgTempC;
    t.ltCx[t.ltIdx]   = (b.minX + b.maxX) / 2.0f;
    t.ltCy[t.ltIdx]   = (b.minY + b.maxY) / 2.0f;
    t.ltArea[t.ltIdx] = (float)b.pixelCount;
    t.ltIdx = (t.ltIdx + 1) % DET_LT_HISTORY_LEN;
    if (t.ltCount < DET_LT_HISTORY_LEN) t.ltCount++;
}

// ------------------------------------------------------------
// Tinh BlobFeatures tu buffer dai han. Hoi quy tuyen tinh dang closed-form
// (khong can ma tran) tren N mau, x = chi so thoi gian (don vi: mau,
// nhan voi DET_LT_SAMPLE_MS/60000 de doi ra C/phut).
// ------------------------------------------------------------
bool OccupantDetector::computeFeatures(int idx, DetBlob &b, float ambientTempC, BlobFeatures &out) {
    memset(&out, 0, sizeof(out));
    out.valid = false;

    float w = (b.maxX - b.minX + 1);
    float h = (b.maxY - b.minY + 1);
    out.aspectRatio = w / h;
    out.solidity = (float)b.pixelCount / (w * h);
    out.pixelCount = b.pixelCount;
    out.maxTempC = b.maxTempC;
    out.tempSpread = b.maxTempC - b.minTempC;
    out.deltaAmbientT = isnan(ambientTempC) ? NAN : (b.avgTempC - ambientTempC);

    if (idx < 0) return false;
    TrackedBlob &t = _tracked[idx];
    out.sampleCount = t.ltCount;
    if (t.ltCount < 5) return false;   // chua du du lieu de tin cay trend/jitter

    int N = t.ltCount;
    // Hoi quy tuyen tinh avgTempC theo thoi gian mau (x = 0..N-1, thu tu chen vao,
    // KHONG phai thu tu vong tron trong buffer - vi ta chi can do doc, thu tu tuong doi la du)
    float sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
    float cxVals[DET_LT_HISTORY_LEN], cyVals[DET_LT_HISTORY_LEN], areaVals[DET_LT_HISTORY_LEN];
    for (int i = 0; i < N; i++) {
        int bufIdx = (t.ltIdx - N + i + DET_LT_HISTORY_LEN) % DET_LT_HISTORY_LEN;
        float x = (float)i;
        float y = t.ltTemp[bufIdx];
        sumX += x; sumY += y; sumXY += x * y; sumXX += x * x;
        cxVals[i] = t.ltCx[bufIdx];
        cyVals[i] = t.ltCy[bufIdx];
        areaVals[i] = t.ltArea[bufIdx];
    }
    float denom = (N * sumXX - sumX * sumX);
    float slope = (fabsf(denom) > 1e-6f) ? (N * sumXY - sumX * sumY) / denom : 0.0f;
    float intercept = (sumY - slope * sumX) / N;
    // doi don vi: 1 don vi x = DET_LT_SAMPLE_MS ms -> quy ra C/phut
    out.tempTrendSlope = slope * (60000.0f / DET_LT_SAMPLE_MS);

    float residSumSq = 0;
    for (int i = 0; i < N; i++) {
        int bufIdx = (t.ltIdx - N + i + DET_LT_HISTORY_LEN) % DET_LT_HISTORY_LEN;
        float pred = intercept + slope * i;
        float r = t.ltTemp[bufIdx] - pred;
        residSumSq += r * r;
    }
    out.tempResidualStd = sqrtf(residSumSq / N);

    // jitter: std don gian (khong tru trend) cua vi tri va dien tich
    float meanCx = 0, meanCy = 0, meanArea = 0;
    for (int i = 0; i < N; i++) { meanCx += cxVals[i]; meanCy += cyVals[i]; meanArea += areaVals[i]; }
    meanCx /= N; meanCy /= N; meanArea /= N;
    float varCx = 0, varCy = 0, varArea = 0;
    for (int i = 0; i < N; i++) {
        varCx += (cxVals[i] - meanCx) * (cxVals[i] - meanCx);
        varCy += (cyVals[i] - meanCy) * (cyVals[i] - meanCy);
        varArea += (areaVals[i] - meanArea) * (areaVals[i] - meanArea);
    }
    varCx /= N; varCy /= N; varArea /= N;
    out.centroidJitterStd = (sqrtf(varCx) + sqrtf(varCy)) / 2.0f;
    out.areaJitterStd = sqrtf(varArea);

    out.valid = true;
    return true;
}

void OccupantDetector::printFeaturesCSV(uint8_t label) const {
    for (int i = 0; i < _result.numBlobs; i++) {
        const BlobFeatures &f = _result.features[i];
        if (!f.valid) continue;
        Serial.printf("%u,%d,%u,%.3f,%.3f,%.2f,%.3f,%.3f,%.3f,%.3f,%.2f,%.2f,%u\n",
            label,
            (int)_result.blobs[i].type,
            f.pixelCount,
            f.aspectRatio,
            f.solidity,
            f.deltaAmbientT,
            f.tempTrendSlope,
            f.tempResidualStd,
            f.centroidJitterStd,
            f.areaJitterStd,
            f.maxTempC,
            f.tempSpread,
            f.sampleCount);
    }
}

void OccupantDetector::runStateMachine(bool hasCandidate) {
    _windowFrames[_windowIdx] = hasCandidate ? 1 : 0;
    _windowIdx = (_windowIdx + 1) % DET_CONFIRM_M;

    uint8_t hits = 0;
    for (int i = 0; i < DET_CONFIRM_M; i++) hits += _windowFrames[i];

    _emptyStreak = hasCandidate ? 0 : (_emptyStreak + 1);

    switch (_state) {
        case STATE_IDLE:
            if (hasCandidate) _state = STATE_CANDIDATE;
            break;
        case STATE_CANDIDATE:
            if (hits >= DET_CONFIRM_N) _state = STATE_CONFIRMED;
            else if (_emptyStreak >= DET_RESET_EMPTY) _state = STATE_IDLE;
            break;
        case STATE_CONFIRMED:
            _state = STATE_ALERT;
            break;
        case STATE_ALERT:
            if (_emptyStreak >= DET_RESET_EMPTY) _state = STATE_IDLE;
            break;
        default:
            _state = STATE_IDLE;
            break;
    }
}

void OccupantDetector::update(const uint16_t* rawPixels, float ambientTempC) {
    static float tempRaw[DET_NUM_PIXELS];
    for (int i = 0; i < DET_NUM_PIXELS; i++) {
        tempRaw[i] = (rawPixels[i] / 10.0f) - 273.15f;
    }

    // Loc nhieu truoc khi lam bat ky viec gi khac - ca nen va tien canh deu
    // dung tren du lieu da lam min, giup on dinh nguong va giam vo blob.
    smooth3x3(tempRaw, _tempSmooth);

    bool warming = (_warmupFrames < DET_WARMUP_FRAMES);
    updateBackground(_tempSmooth, warming);

    if (warming) {
        _warmupFrames++;
        memset(_labelsSafe, -1, sizeof(_labelsSafe));
        _result.numBlobs = 0;
        _result.state = STATE_WARMUP;
        _result.alert = false;
        _result.warmingUp = true;
        _result.fgPixelCount = 0;
        _result.newLtSample = false;

        if (_mutex != nullptr && xSemaphoreTake(_mutex, pdMS_TO_TICKS(20)) == pdTRUE) {
            _resultSafe = _result;
            memcpy(_labelsSafe, _labels, sizeof(_labelsSafe)); // toan -1, khong co blob
            xSemaphoreGive(_mutex);
        }
        return; // khong chay detection trong luc warmup
    }

    buildForegroundMask(_tempSmooth);
    dilateMask(_fgMask, _fgMaskDilated);

    uint16_t fgCount = 0;
    for (int i = 0; i < DET_NUM_PIXELS; i++) fgCount += _fgMaskDilated[i];

    DetBlob blobs[DET_MAX_BLOBS];
    int n = labelBlobs(_tempSmooth, _fgMaskDilated, blobs, DET_MAX_BLOBS);

    uint32_t nowMs = millis();
    bool anyLtSample = false;
    BlobFeatures featuresOut[DET_MAX_BLOBS];

    bool hasCandidate = false;
    for (int i = 0; i < n; i++) {
        classify(blobs[i]);

        if (blobs[i].type == BLOB_GRAY_ZONE) {
            float var = trackAndGetVariance(blobs[i]);
            blobs[i].type = (var >= DET_GRAY_VAR_THRESH) ? BLOB_CANDIDATE : BLOB_REJECTED;
        }

        // NOTE (TODO): buoc xac nhan RGB khi du sang trong so do pipeline chua duoc
        // implement o day - xem ghi chu trong detector.h. Hien dung thang ket qua nhiet.

        if (blobs[i].type == BLOB_PERSON || blobs[i].type == BLOB_CANDIDATE) {
            hasCandidate = true;
        }

        // --- NEW: track + lay mau dai han + tinh feature cho MOI blob co kich thuoc
        // toi thieu (ke ca blob bi REJECTED/LOW_SUSPECT) - can log CA lop am (cup nong,
        // dien thoai...) de train classifier, khong chi log lop duong (nguoi).
        if (blobs[i].pixelCount >= DET_MIN_BLOB_PIXELS) {
            int idx = findOrAssignTrack(blobs[i]);
            if (idx >= 0) {
                sampleLongTerm(idx, blobs[i], nowMs);
                if (computeFeatures(idx, blobs[i], ambientTempC, featuresOut[i])) anyLtSample = true;
            } else {
                memset(&featuresOut[i], 0, sizeof(BlobFeatures));
            }
        } else {
            memset(&featuresOut[i], 0, sizeof(BlobFeatures));
        }
    }

    for (int i = 0; i < DET_MAX_BLOBS; i++) {
        if (_tracked[i].active) {
            _tracked[i].missedFrames++;
            if (_tracked[i].missedFrames > 6) _tracked[i].active = false;
        }
    }

    runStateMachine(hasCandidate);

    _result.numBlobs = (uint8_t)n;
    for (int i = 0; i < n; i++) {
        _result.blobs[i] = blobs[i];
        _result.features[i] = featuresOut[i];
    }
    _result.state = _state;
    _result.alert = (_state == STATE_ALERT);
    _result.warmingUp = false;
    _result.fgPixelCount = fgCount;
    _result.newLtSample = anyLtSample;

    if (_mutex != nullptr && xSemaphoreTake(_mutex, pdMS_TO_TICKS(20)) == pdTRUE) {
        _resultSafe = _result;
        for (int i = 0; i < DET_NUM_PIXELS; i++) {
            _labelsSafe[i] = (_labels[i] >= 0 && _labels[i] < DET_MAX_BLOBS) ? (int8_t)_labels[i] : -1;
        }
        xSemaphoreGive(_mutex);
    }
}

bool OccupantDetector::getResult(DetectionResult &out) {
    if (_mutex == nullptr) return false;
    if (xSemaphoreTake(_mutex, pdMS_TO_TICKS(20)) != pdTRUE) return false;
    out = _resultSafe;
    xSemaphoreGive(_mutex);
    return true;
}

bool OccupantDetector::getLabelGrid(int8_t* out) {
    if (_mutex == nullptr) return false;
    if (xSemaphoreTake(_mutex, pdMS_TO_TICKS(20)) != pdTRUE) return false;
    memcpy(out, _labelsSafe, DET_NUM_PIXELS);
    xSemaphoreGive(_mutex);
    return true;
}

bool OccupantDetector::getResultAndGrid(DetectionResult &out, int8_t* gridOut) {
    if (_mutex == nullptr) return false;
    if (xSemaphoreTake(_mutex, pdMS_TO_TICKS(20)) != pdTRUE) return false;
    out = _resultSafe;
    memcpy(gridOut, _labelsSafe, DET_NUM_PIXELS);
    xSemaphoreGive(_mutex);
    return true;
}