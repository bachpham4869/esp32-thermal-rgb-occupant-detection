#include "task_web.h"

#include <Arduino.h>
#include <WebServer.h>
#include "esp_camera.h"

#include "config.h"
#include "mi48.h"
#include "Detector.h"

extern MI48 thermal;
extern WebServer streamServer;
extern OccupantDetector occupant;

// Dinh nghia trong main.cpp - dung chung 1 bien de dieu khien nhan CSV luc
// DATA_COLLECTION_MODE=true. Truoc day doi qua Serial ("L1" + Enter), gio
// doi truc tiep qua nut bam tren web (xem handleSetLabel() ben duoi).
extern volatile uint8_t g_currentLabel;

WebServer server(80);

// Dem so frame RGB da gui qua MJPEG stream - dung de client tinh FPS thuc te
// (khac voi frame counter cua MI48, ca 2 deu can de biet rieng RGB va Thermal
// co dang tut FPS hay khong, thay vi chi doan qua cam quan).
static volatile uint32_t g_rgbFrameCount = 0;

static const char* labelName(int v) {
    switch (v) {
        case 0: return "Nen / chua xac dinh";
        case 1: return "Nguoi that";
        case 2: return "Coc/chai nuoc nong";
        case 3: return "Dien thoai/laptop";
        case 4: return "Bang tap-lo/ghe nang chieu";
        case 5: return "Gio dieu hoa/suoi";
        case 6: return "Hoi nuoc";
        default: return "?";
    }
}

void root_page()
{
    static const char html[] PROGMEM = R"rawliteral(
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CPD Fusion Monitor</title>
<style>
:root{
  --bg:#0b0d10; --panel:#14171c; --panel2:#1a1e24; --border:#262b33;
  --text:#e6e8eb; --muted:#8b93a1; --accent:#3ddc97; --warn:#ffb454; --bad:#ff5d5d;
  --mono:'SF Mono',Consolas,'Roboto Mono',monospace;
}
*{box-sizing:border-box;}
body{
  margin:0; background:var(--bg); color:var(--text);
  font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
  padding:20px;
}
header{display:flex; align-items:baseline; gap:10px; margin-bottom:18px; flex-wrap:wrap;}
header h1{font-size:18px; font-weight:600; margin:0;}
header .sub{color:var(--muted); font-size:12px;}

#alertBanner{
  display:none; background:var(--bad); color:#0b0d10; font-weight:700;
  padding:10px 16px; border-radius:8px; margin-bottom:14px; text-align:center;
  max-width:900px;
}

.stats{
  display:grid; grid-template-columns:repeat(auto-fit, minmax(130px,1fr));
  gap:10px; margin-bottom:16px; max-width:900px;
}
.stat{
  background:var(--panel); border:1px solid var(--border); border-radius:10px;
  padding:12px 14px;
}
.stat .label{font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:.04em;}
.stat .value{font-family:var(--mono); font-size:22px; font-weight:600; margin-top:4px;}
.stat .value.small{font-size:15px;}
.pill{display:inline-block; padding:1px 8px; border-radius:999px; font-size:11px; font-weight:600;}
.pill.ok{background:rgba(61,220,151,.15); color:var(--accent);}
.pill.bad{background:rgba(255,93,93,.15); color:var(--bad);}

.labelPanel{
  background:var(--panel); border:1px solid var(--border); border-radius:12px;
  padding:14px; max-width:900px; margin-bottom:20px;
}
.labelPanel h3{margin:0 0 10px; font-size:13px; color:var(--muted); font-weight:600; text-transform:uppercase; letter-spacing:.04em;}
.labelBtns{display:flex; gap:8px; flex-wrap:wrap;}
.labelBtn{
  background:var(--panel2); color:var(--text); border:1px solid var(--border);
  border-radius:8px; padding:10px 14px; font-size:13px; cursor:pointer;
  transition:all .12s;
}
.labelBtn:hover{border-color:var(--accent);}
.labelBtn.active{background:var(--accent); color:#0b0d10; border-color:var(--accent); font-weight:700;}
.labelCurrent{margin-top:10px; font-size:12px; color:var(--muted);}
.labelCurrent b{color:var(--text); font-family:var(--mono);}

.grid{display:flex; gap:16px; flex-wrap:wrap; max-width:900px;}
.panel{
  background:var(--panel); border:1px solid var(--border); border-radius:12px;
  padding:14px; flex:1; min-width:340px;
}
.panel h3{margin:0 0 10px; font-size:13px; color:var(--muted); font-weight:600; text-transform:uppercase; letter-spacing:.04em;}
.viewport{
  width:100%; aspect-ratio:4/3; background:#000; border-radius:8px;
  border:1px solid var(--border); overflow:hidden; display:flex; align-items:center; justify-content:center;
}
.viewport img, .viewport canvas{width:100%; height:100%; object-fit:contain;}
.ctrl{margin-top:10px; display:flex; align-items:center; gap:8px; font-size:12px; color:var(--muted);}
select{
  background:var(--panel2); color:var(--text); border:1px solid var(--border);
  border-radius:6px; padding:5px 8px; font-size:12px;
}
footer{margin-top:16px; color:var(--muted); font-size:11px; max-width:900px;}
</style>
</head>
<body>
<header>
  <h1>CPD Fusion Monitor</h1>
  <span class="sub">ESP32-S3 &middot; OV3660 + MI48</span>
</header>

<div id="alertBanner">&#9888; CANH BAO: PHAT HIEN NGUOI TRONG XE</div>

<div class="labelPanel">
  <h3>Nhan dang thu (DATA_COLLECTION_MODE)</h3>
  <div class="labelBtns">
    <button class="labelBtn" data-id="0" onclick="setLabel(0)">0 - Nen/bg</button>
    <button class="labelBtn" data-id="1" onclick="setLabel(1)">1 - Nguoi that</button>
    <button class="labelBtn" data-id="2" onclick="setLabel(2)">2 - Coc nuoc nong</button>
    <button class="labelBtn" data-id="3" onclick="setLabel(3)">3 - Dien thoai/laptop</button>
    <button class="labelBtn" data-id="4" onclick="setLabel(4)">4 - Nang chieu</button>
  </div>
  <div class="labelCurrent">Dang ghi nhan: <b id="labelCurrentText">--</b></div>
</div>

<div class="stats">
  <div class="stat"><div class="label">Thermal FPS</div><div class="value" id="fpsTh">--</div></div>
  <div class="stat"><div class="label">RGB FPS</div><div class="value" id="fpsRgb">--</div></div>
  <div class="stat"><div class="label">CRC Nhiet</div><div class="value small" id="crc">--</div></div>
  <div class="stat"><div class="label">Nhiet MI48</div><div class="value small" id="ambient">--</div></div>
  <div class="stat"><div class="label">Heap ranh</div><div class="value small" id="heap">--</div></div>
  <div class="stat"><div class="label">Heap thap nhat</div><div class="value small" id="heapMin">--</div></div>
  <div class="stat"><div class="label">Trang thai</div><div class="value small" id="detState">--</div></div>
  <div class="stat"><div class="label">Doi tuong</div><div class="value small" id="detType">--</div></div>
  <div class="stat"><div class="label">Xac suat nguoi (LR)</div><div class="value small" id="detProb">--</div></div>
</div>

<div class="grid">
  <div class="panel">
    <h3>RGB (OV3660)</h3>
    <div class="viewport"><img id="rgb"></div>
    <div class="ctrl">
      Huong:
      <select id="rgbMode" onchange="applyRgb()">
        <option value="normal">Binh thuong</option>
        <option value="h">Lat ngang (guong)</option>
        <option value="v">Lat doc</option>
        <option value="180">Xoay 180</option>
      </select>
    </div>
  </div>
  <div class="panel">
    <h3>Thermal (MI48, 80x62)</h3>
    <div class="viewport"><canvas id="cam" width="320" height="248"></canvas></div>
    <div class="ctrl">
      Huong:
      <select id="thMode">
        <option value="normal">Binh thuong</option>
        <option value="h">Lat ngang (guong)</option>
        <option value="v">Lat doc</option>
        <option value="180">Xoay 180</option>
      </select>
      <span id="minmax" style="margin-left:auto;"></span>
    </div>
  </div>
</div>

<footer>Nhan bam tren cung se ghi thang vao CSV dang log (khong can Serial). Min/Max nhiet va thoi gian frame lay tu /status. Anh nhiet lay nhi phan tu /frame. Trang thai/Doi tuong/Xac suat lay tu OccupantDetector qua /status.</footer>

<script>
let rgbImg = document.getElementById("rgb");
let c = document.getElementById("cam");
let ctx = c.getContext("2d");

rgbImg.src = "http://" + location.hostname + ":81/stream";

function applyRgb(){
  let mode = document.getElementById("rgbMode").value;
  let t = "none";
  if (mode === "h")   t = "scaleX(-1)";
  if (mode === "v")   t = "scaleY(-1)";
  if (mode === "180") t = "scale(-1,-1)";
  rgbImg.style.transform = t;
}
applyRgb();

// --- NEW: chon nhan qua web, khong can Serial ---
function setLabel(id){
  fetch("/setlabel?id=" + id)
    .then(r => r.json())
    .then(() => updateStatus())
    .catch(e => console.error(e));
}
function highlightLabelBtn(id){
  document.querySelectorAll(".labelBtn").forEach(btn => {
    btn.classList.toggle("active", parseInt(btn.dataset.id) === id);
  });
}

const W=80, H=62;
function idxFor(x, y, mode){
  let xx = x, yy = y;
  if (mode === "h")   xx = W-1-x;
  if (mode === "v")   yy = H-1-y;
  if (mode === "180") { xx = W-1-x; yy = H-1-y; }
  return yy*W + xx;
}

function updateThermal(){
  let mode = document.getElementById("thMode").value;
  fetch("/frame")
    .then(r=>r.arrayBuffer())
    .then(buf=>{
      const data = new Uint16Array(buf);   // nhi phan, little-endian - khop ESP32
      let min=65535, max=0;
      for (let i=0;i<data.length;i++){
        if (data[i]<min) min=data[i];
        if (data[i]>max) max=data[i];
      }
      let range = Math.max(1, max-min);

      let img = ctx.createImageData(W,H);
      for (let y=0;y<H;y++){
        for (let x=0;x<W;x++){
          let srcIdx = idxFor(x,y,mode);
          let v = Math.round(((data[srcIdx]-min)/range)*255);
          let dstIdx = (y*W+x)*4;
          img.data[dstIdx]=v; img.data[dstIdx+1]=0; img.data[dstIdx+2]=255-v; img.data[dstIdx+3]=255;
        }
      }
      let tmp=document.createElement("canvas");
      tmp.width=W; tmp.height=H;
      tmp.getContext("2d").putImageData(img,0,0);
      ctx.drawImage(tmp,0,0,c.width,c.height);
    })
    .catch(e=>console.error(e));
}

// --- FPS: tinh tu delta counter that (frame cua MI48, frame cua RGB stream)
// qua thoi gian that giua 2 lan poll /status, KHONG phai uoc luong ---
let prevThFrame = null, prevRgbFrame = null, prevT = null;

function updateStatus(){
  fetch("/status").then(r=>r.json()).then(s=>{
    const now = performance.now();
    if (prevT !== null) {
      const dtSec = (now - prevT) / 1000;
      if (dtSec > 0) {
        const dTh  = (s.frame     - prevThFrame  + 65536) % 65536;
        const dRgb = (s.rgbFrames - prevRgbFrame + 4294967296) % 4294967296;
        document.getElementById("fpsTh").textContent  = (dTh  / dtSec).toFixed(1);
        document.getElementById("fpsRgb").textContent = (dRgb / dtSec).toFixed(1);
      }
    }
    prevThFrame = s.frame; prevRgbFrame = s.rgbFrames; prevT = now;

    const crcEl = document.getElementById("crc");
    crcEl.innerHTML = s.crcOk
      ? '<span class="pill ok">OK</span>'
      : '<span class="pill bad">LOI</span>';

    document.getElementById("ambient").textContent = s.senxorTempC.toFixed(1) + " C";
    document.getElementById("heap").textContent = (s.heapFree/1024).toFixed(1) + " KB";
    document.getElementById("heapMin").textContent = (s.heapMinFree/1024).toFixed(1) + " KB";
    document.getElementById("minmax").textContent =
      "Min/Max: " + s.minC.toFixed(1) + "C / " + s.maxC.toFixed(1) + "C";

    document.getElementById("detState").textContent = s.detStateLabel;
    document.getElementById("detType").textContent = s.numBlobs > 0 ? s.bestTypeLabel : "--";
    document.getElementById("detProb").textContent = s.bestPHuman >= 0
      ? (s.bestPHuman * 100).toFixed(0) + "%" : "--";

    const banner = document.getElementById("alertBanner");
    banner.style.display = s.detAlert ? "block" : "none";

    // --- NEW: dong bo nut nhan dang bam voi trang thai thuc te tren board ---
    document.getElementById("labelCurrentText").textContent =
      s.label + " - " + s.labelName;
    highlightLabelBtn(s.label);
  }).catch(()=>{});
}

setInterval(updateThermal, 250);
setInterval(updateStatus, 500);
updateThermal(); updateStatus();
</script>
</body>
</html>
)rawliteral";
    server.send_P(200, "text/html", html);
}

void handleStream() {
    WiFiClient client = streamServer.client();
    String response = "HTTP/1.1 200 OK\r\n";
    response += "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n";
    streamServer.sendContent(response);

    while (client.connected()) {
        camera_fb_t *fb = esp_camera_fb_get();
        if (!fb) break;
        client.print("--frame\r\n");
        client.print("Content-Type: image/jpeg\r\n");
        client.printf("Content-Length: %u\r\n\r\n", fb->len);
        client.write(fb->buf, fb->len);
        client.print("\r\n");
        esp_camera_fb_return(fb);
        g_rgbFrameCount++;
        if (!client.connected()) break;
        // QUAN TRONG: nhuong CPU moi frame. Thieu dong nay la nguyen nhan chinh
        // khien cac task khac tren cung core (WiFi, /frame, /status) bi doi dan
        // theo thoi gian - vong while nay truoc day chay lien tuc khong nghi.
        vTaskDelay(1);
    }
}

void frame_page()
{
    static uint16_t buf[MI48_NUM_PIXELS];

    if (!thermal.copyPixels(buf)) {
        server.send(503, "text/plain", "Chua co frame nhiet");
        return;
    }

    // Gui nhi phan tho (uint16 x 4960 = 9920 byte) thay vi JSON text (~25-30KB truoc
    // day) - giam ca chi phi CPU format chuoi tren ESP32 lan bang thong WiFi.
    server.send_P(200, "application/octet-stream", (const char*)buf, sizeof(buf));
}

// --- NEW: doi nhan qua web thay vi go Serial ---
// GET /setlabel?id=<0..255> -> set g_currentLabel, dung cho CSV logging
// (DATA_COLLECTION_MODE). Bam nut tren web goi thang endpoint nay.
void handleSetLabel() {
    if (server.hasArg("id")) {
        int v = server.arg("id").toInt();
        if (v >= 0 && v <= 255) {
            g_currentLabel = (uint8_t)v;
            Serial.printf("[Label] Doi qua web -> %u (%s)\n", g_currentLabel, labelName(g_currentLabel));
        }
    }
    String resp = "{\"ok\":true,\"label\":" + String((int)g_currentLabel) + "}";
    server.send(200, "application/json", resp);
}

// --- nhan dien nguoi/vat gia (OccupantDetector + LR) ---

static const char* detStateLabel(int s) {
    switch (s) {
        case STATE_IDLE:      return "IDLE";
        case STATE_CANDIDATE: return "CANDIDATE";
        case STATE_CONFIRMED: return "CONFIRMED";
        case STATE_ALERT:     return "ALERT";
        case STATE_WARMUP:    return "WARMUP";
        default:               return "--";
    }
}

static const char* blobTypeLabel(int t) {
    switch (t) {
        case BLOB_REJECTED:    return "Rejected (vat gia)";
        case BLOB_LOW_SUSPECT: return "Low suspect";
        case BLOB_GRAY_ZONE:   return "Gray zone";
        case BLOB_CANDIDATE:   return "Candidate";
        case BLOB_PERSON:      return "Person";
        default:                return "--";
    }
}

void status_page()
{
    DetectionResult res;
    bool gotRes = occupant.getResult(res);

    // Chon blob "dang chu y nhat": uu tien PERSON/CANDIDATE, pHuman cao nhat.
    // Neu khong co, van hien blob dau tien de debug (thay ro cai gi vua bi reject).
    int bestIdx = -1;
    for (int i = 0; i < res.numBlobs; i++) {
        if (res.blobs[i].type == BLOB_PERSON || res.blobs[i].type == BLOB_CANDIDATE) {
            if (bestIdx < 0 || res.lrProb[i] > res.lrProb[bestIdx]) bestIdx = i;
        }
    }
    if (bestIdx < 0 && res.numBlobs > 0) bestIdx = 0;

    int bestType = (bestIdx >= 0) ? (int)res.blobs[bestIdx].type : -1;
    float bestPHuman = (bestIdx >= 0) ? res.lrProb[bestIdx] : -1.0f;
    int curLabel = (int)g_currentLabel;

    char buf[800];
    snprintf(buf, sizeof(buf),
        "{\"frame\":%u,\"resolution\":\"%dx%d\",\"crcOk\":%s,"
        "\"timestamp\":%u,\"minC\":%.1f,\"maxC\":%.1f,\"senxorTempC\":%.2f,"
        "\"rgbFrames\":%u,\"heapFree\":%u,\"heapMinFree\":%u,\"uptimeMs\":%lu,"
        "\"detState\":%d,\"detStateLabel\":\"%s\",\"detAlert\":%s,"
        "\"numBlobs\":%d,\"bestType\":%d,\"bestTypeLabel\":\"%s\",\"bestPHuman\":%.3f,"
        "\"label\":%d,\"labelName\":\"%s\"}",
        thermal.frameCounter(),
        MI48_SENXOR_WIDTH, MI48_SENXOR_HEIGHT,
        thermal.crcOk() ? "true" : "false",
        thermal.timestamp(),
        MI48::rawToCelsius(thermal.minRaw()),
        MI48::rawToCelsius(thermal.maxRaw()),
        thermal.senxorTempC(),
        (unsigned)g_rgbFrameCount,
        (unsigned)ESP.getFreeHeap(),
        (unsigned)ESP.getMinFreeHeap(),
        (unsigned long)millis(),
        gotRes ? (int)res.state : -1,
        gotRes ? detStateLabel((int)res.state) : "--",
        (gotRes && res.alert) ? "true" : "false",
        gotRes ? (int)res.numBlobs : 0,
        bestType,
        blobTypeLabel(bestType),
        bestPHuman,
        curLabel,
        labelName(curLabel));
    server.send(200, "application/json", buf);
}

void web_init()
{
    server.on("/", root_page);
    server.on("/frame", frame_page);
    server.on("/status", status_page);
    server.on("/setlabel", handleSetLabel);
    server.begin();
    Serial.println("WebServer (port 80) started");

    streamServer.on("/stream", handleStream);
    streamServer.begin();
    Serial.println("Stream server (port 81) started");
}

void web_loop()
{
    server.handleClient();
    // streamServer.handleClient() da chay rieng trong streamTask (main.cpp)
}