#pragma once
#include <stdint.h>
#include <freertos/FreeRTOS.h>
#include <freertos/semphr.h>
#include "config.h"

// ============================================================
// Module phat hien nguoi/tre em bi bo quen tu du lieu nhiet MI48
// Pipeline: lam min nhieu -> tru nen EMA (co warmup) -> dilation ->
//           connected component -> 3 bo loc -> tracking vung xam
//           (phuong sai nhiet) -> state machine debounce
// ============================================================

#define DET_W 80
#define DET_H 62
#define DET_NUM_PIXELS (DET_W * DET_H)
#define DET_MAX_BLOBS 8

enum DetState : uint8_t {
    STATE_WARMUP    = 4,   // dang hoc nen luc moi khoi dong / vua reset
    STATE_IDLE      = 0,
    STATE_CANDIDATE = 1,
    STATE_CONFIRMED = 2,
    STATE_ALERT     = 3
};

enum BlobType : uint8_t {
    BLOB_REJECTED    = 0,  // bi loai (qua nong / qua nho / da xac nhan la vat tinh)
    BLOB_LOW_SUSPECT = 1,  // hinh dang bat thuong (laptop, tui do) - nghi van thap
    BLOB_GRAY_ZONE   = 2,  // dang cho tracking de tinh phuong sai nhiet
    BLOB_CANDIDATE   = 3,  // vung xam co dao dong giong tho -> nghi co nguoi
    BLOB_PERSON      = 4   // size lon, hinh dang ro rang giong nguoi
};

struct DetBlob {
    uint8_t  minX, minY, maxX, maxY;  // bbox theo luoi 80x62 (van giu de debug/fallback)
    uint16_t pixelCount;
    float    avgTempC;
    float    maxTempC;
    float    minTempC;                // NEW: nhiet do thap nhat trong blob (dung cho tempSpread)
    BlobType type;
};

// ------------------------------------------------------------
// Vector dac trung dai han cho 1 blob duoc track - dung de log ra
// CSV thu thap du lieu, va sau nay lam input cho classifier nho
// (logistic regression / decision tree / MLP be) phan biet
// da nguoi vs vat nong khac. Tat ca deu tinh duoc voi 80x62 pixel,
// khong doi hoi do phan giai khong gian cao.
// ------------------------------------------------------------
struct BlobFeatures {
    bool     valid;             // false neu chua du mau dai han de tinh
    float    deltaAmbientT;     // avgTempC hien tai - nhiet do moi truong (senxorTempC)
    float    tempTrendSlope;    // do doc hoi quy tuyen tinh avgTempC theo thoi gian (C/phut)
    float    tempResidualStd;   // do lech chuan phan du sau khi tru trend (C)
    float    solidity;          // pixelCount / (bboxW * bboxH), 0..1
    float    aspectRatio;       // w/h cua bbox
    uint16_t pixelCount;
    float    centroidJitterStd; // trung binh std(cx), std(cy) qua cua so dai han (pixel)
    float    areaJitterStd;     // std(pixelCount) qua cua so dai han (pixel^2)
    float    maxTempC;
    float    tempSpread;        // maxTempC - minTempC trong blob (frame hien tai)
    uint8_t  sampleCount;       // so mau dai han da tich luy (de biet do tin cay)
};

struct DetectionResult {
    uint8_t       numBlobs;            // TAT CA blob tim duoc, ke ca loai BLOB_REJECTED (de debug)
    DetBlob       blobs[DET_MAX_BLOBS];
    BlobFeatures  features[DET_MAX_BLOBS]; // NEW: dac trung dai han tuong ung tung blobs[i] (neu track duoc)
    DetState      state;
    bool          alert;
    bool          warmingUp;
    uint16_t      fgPixelCount;        // tong so pixel vuot nguong nen (truoc khi gom blob) - de debug
    bool          newLtSample;         // NEW: true neu frame nay vua lay mau dai han moi (dung de quyet dinh khi nao log CSV)
    float         lrProb[DET_MAX_BLOBS];
};

class OccupantDetector {
public:
    OccupantDetector();
    // Tra ve xac suat [0..1] la nguoi that, dua tren BlobFeatures da tinh.
    // Chi goi khi features.valid == true (>=5 mau dai han).
   

    void begin();

    // Goi tu thermalTask moi khi co frame moi.
    // rawPixels: du lieu tho MI48 (don vi 0.1 Kelvin), kich thuoc DET_NUM_PIXELS
    // ambientTempC: nhiet do moi truong cam bien (MI48::senxorTempC(), can header bat).
    //               Neu khong co header, truyen NAN va deltaAmbientT se khong dang tin cay.
    void update(const uint16_t* rawPixels, float ambientTempC);

    // In ra Serial 1 dong CSV / blob dang co feature dai han hop le, kem nhan (label)
    // do nguoi van hanh truyen vao thu cong luc thu thap du lieu. Chi goi tu THERMAL TASK
    // (cung task voi update()) ngay sau khi update() tra ve - khong thread-safe voi task khac.
    // Header CSV tuong ung (in 1 lan luc bat dau thu thap):
    // label,blobType,pixelCount,aspect,solidity,deltaAmbT,trendSlope,residStd,cxJitter,areaJitter,maxTemp,tempSpread,sampleCount
    void printFeaturesCSV(uint8_t label) const;

    // True neu lan update() vua roi co lay them >=1 mau dai han moi (dung de goi
    // printFeaturesCSV() dung nhip, khong spam Serial moi frame). Chi doc an toan
    // tu THERMAL TASK ngay sau update(), khong dung tu task khac.
    bool newFeatureSample() const { return _result.newLtSample; }

    // An toan luong - goi tu task/handler khac (vd web server) de doc ket qua moi nhat
    bool getResult(DetectionResult &out);

    // Lay grid nhan blob (moi pixel -> chi so blob trong DetectionResult::blobs, hoac -1
    // neu khong thuoc blob nao). Dung de ve contour thay vi hinh chu nhat tren web.
    // out phai co kich thuoc DET_NUM_PIXELS.
    bool getLabelGrid(int8_t* out);

    // Lay CA result VA grid trong CUNG MOT lan khoa mutex - dung ham nay thay vi
    // goi getResult() + getLabelGrid() rieng le, vi 2 lan khoa tach biet co the bi
    // update() (chay o core khac) xen vao giua, lam result va grid lech nhau 1 frame
    // (bounding box khong khop vung to mau tren UI). gridOut phai co kich thuoc
    // DET_NUM_PIXELS.
    bool getResultAndGrid(DetectionResult &out, int8_t* gridOut);

    // Xoa nen da hoc, bat che do warmup lai tu dau. Goi khi can hieu chinh lai
    // (vd truoc khi test, dam bao khoang nhin cua cam bien dang trong/khong co nguoi).
    void resetBackground();

private:
    // --- nen thich nghi (EMA) ---
    float    _background[DET_NUM_PIXELS];
    bool     _bgInit = false;
    uint16_t _warmupFrames = 0;

    // --- buffer trung gian ---
    float    _tempSmooth[DET_NUM_PIXELS];   // sau loc nhieu 3x3
    uint8_t  _fgMask[DET_NUM_PIXELS];       // 1 = tien canh (truoc dilation)
    uint8_t  _fgMaskDilated[DET_NUM_PIXELS];// 1 = tien canh (sau dilation, dung de gom blob)

    // --- buffer cho connected component labeling (BFS, khong de quy) ---
    int16_t  _labels[DET_NUM_PIXELS];
    int8_t   _labelsSafe[DET_NUM_PIXELS];   // ban sao an toan luong, cho web doc ve ve contour
    uint16_t _bfsQueue[DET_NUM_PIXELS];

    // --- tracking cho "vung xam": khop blob qua cac frame de tinh phuong sai nhiet ---
    struct TrackedBlob {
        bool    active;
        float   cx, cy;
        float   history[DET_GRAY_HISTORY_LEN];
        uint8_t histCount;
        uint8_t histIdx;
        uint8_t missedFrames;

        // --- buffer dai han (NEW) - lay 1 mau moi ~DET_LT_SAMPLE_MS, dung cho
        // BlobFeatures (trend nhiet, jitter vi tri/kich thuoc). Doc lap voi
        // history[] o tren (history[] danh cho gray-zone variance, tan so cao hon).
        float    ltTemp[DET_LT_HISTORY_LEN];
        float    ltCx[DET_LT_HISTORY_LEN];
        float    ltCy[DET_LT_HISTORY_LEN];
        float    ltArea[DET_LT_HISTORY_LEN];
        uint8_t  ltCount;
        uint8_t  ltIdx;
        uint32_t lastLtSampleMs;
    };
    TrackedBlob _tracked[DET_MAX_BLOBS];

    // --- state machine debounce N/M ---
    DetState _state = STATE_IDLE;
    uint8_t  _windowFrames[DET_CONFIRM_M] = {0};
    uint8_t  _windowIdx = 0;
    uint8_t  _emptyStreak = 0;

    // --- ket qua, co mutex de web task doc an toan ---
    DetectionResult   _result;
    DetectionResult   _resultSafe;
    SemaphoreHandle_t _mutex = nullptr;

    void  smooth3x3(const float* src, float* dst);
    void  updateBackground(const float* tempC, bool learnFast);
    void  buildForegroundMask(const float* tempC);
    void  dilateMask(const uint8_t* src, uint8_t* dst);
    int   labelBlobs(const float* tempC, const uint8_t* mask, DetBlob* outBlobs, int maxBlobs);
    void  classify(DetBlob &b);
    float trackAndGetVariance(DetBlob &b);
    void  runStateMachine(bool hasCandidate);

    // --- NEW: tracking + feature dai han cho ML ---
    // Tra ve chi so trong _tracked[] duoc khop/gan cho blob b (giong logic trong
    // trackAndGetVariance nhung tach rieng de dung chung cho ca 2 muc dich).
    int   findOrAssignTrack(DetBlob &b);
    // Lay 1 mau vao buffer dai han cua track 'idx' neu du thoi gian trung binh
    // DET_LT_SAMPLE_MS da troi qua ke tu lan lay truoc.
    void  sampleLongTerm(int idx, DetBlob &b, uint32_t nowMs);
    // Tinh BlobFeatures tu buffer dai han cua track 'idx'. Tra ve false (features.valid=false)
    // neu chua co it nhat 5 mau.
    bool  computeFeatures(int idx, DetBlob &b, float ambientTempC, BlobFeatures &out);
     float lrPredictHuman(const BlobFeatures &f) const;
    // TODO (chua lam): xac nhan bang RGB khi du sang (theo so do pipeline).
    // Ly do bo qua: can mot detector hinh nguoi tren anh RGB (kieu YOLO/HOG) de khop
    // vi tri blob nhiet voi vung "giong nguoi" trong anh - khong kha thi chay realtime
    // tren ESP32-S3 khong co model. Hien tai he thong dung 100% ket qua nhiet ke ca
    // ban ngay. Neu can lam sau: Edge Impulse / TFLite Micro cho mot model nguoi nho.
};