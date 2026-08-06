#include "task_web.h"

#include <Arduino.h>
#include <WebServer.h>
#include "esp_camera.h"

#include "config.h"
#include "mi48.h"

extern MI48 thermal;
extern WebServer streamServer;

WebServer server(80);

void root_page()
{
    static const char html[] PROGMEM = R"rawliteral(
<html>
<head>
<title>Fusion RGB + Thermal</title>
<style>
body{background:#111;color:#eee;text-align:center;font-family:sans-serif;}
.row{display:flex;justify-content:center;gap:16px;flex-wrap:wrap;margin-top:10px;}
.box{display:flex;flex-direction:column;align-items:center;}
img,canvas{border:2px solid #444;width:320px;height:240px;object-fit:contain;background:#000;}
#info{margin-top:8px;font-size:14px;color:#aaa;line-height:1.6;}
.ctrl{margin-top:6px;}
select{padding:4px;font-size:12px;}
</style>
</head>
<body>
<h2>YOLO Fusion - RGB (OV3660) + Thermal (MI48)</h2>
<div class="row">
  <div class="box">
    <p>RGB</p>
    <img id="rgb">
    <div class="ctrl">
      Huong: <select id="rgbMode" onchange="applyRgb()">
        <option value="normal">Binh thuong</option>
        <option value="h">Lat ngang (guong)</option>
        <option value="v">Lat doc</option>
        <option value="180">Xoay 180</option>
      </select>
    </div>
  </div>
  <div class="box">
    <p>Thermal</p>
    <canvas id="cam" width="320" height="248"></canvas>
    <div class="ctrl">
      Huong: <select id="thMode">
        <option value="normal">Binh thuong</option>
        <option value="h">Lat ngang (guong)</option>
        <option value="v">Lat doc</option>
        <option value="180">Xoay 180</option>
      </select>
    </div>
  </div>
</div>
<div id="info">Dang tai...</div>
<script>
let rgbImg = document.getElementById("rgb");
let c = document.getElementById("cam");
let ctx = c.getContext("2d");
let info = document.getElementById("info");

// RGB dung stream that qua cong 81 - khong can polling, muot va nhe hon.
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

// Thermal: xu ly truc tiep tren du lieu diem anh, khong dung CSS.
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
    .then(r=>r.json())
    .then(data=>{
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

function updateStatus(){
  fetch("/status").then(r=>r.json()).then(s=>{
    info.innerHTML =
      "Do phan giai nhiet: "+s.resolution+" | Frame: "+s.frame+
      " | CRC: "+(s.crcOk?"OK":"LOI")+
      "<br>Nhiet MI48: "+s.senxorTempC.toFixed(1)+"C"+
      " | Min/Max: "+s.minC.toFixed(1)+"C / "+s.maxC.toFixed(1)+"C";
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
        if (!client.connected()) break;
    }
}

void frame_page()
{
    static uint16_t buf[MI48_NUM_PIXELS];
    static char jsonBuf[MI48_NUM_PIXELS * 6 + 8];

    if (!thermal.copyPixels(buf)) {
        server.send(503, "text/plain", "Chua co frame nhiet");
        return;
    }

    size_t pos = 0;
    jsonBuf[pos++] = '[';
    for (int i = 0; i < MI48_NUM_PIXELS; i++) {
        pos += snprintf(jsonBuf + pos, sizeof(jsonBuf) - pos, "%u", buf[i]);
        if (i < MI48_NUM_PIXELS - 1) jsonBuf[pos++] = ',';
    }
    jsonBuf[pos++] = ']';
    jsonBuf[pos] = '\0';

    server.send(200, "application/json", jsonBuf);
}

void status_page()
{
    char buf[400];
    snprintf(buf, sizeof(buf),
        "{\"frame\":%u,\"resolution\":\"%dx%d\",\"crcOk\":%s,"
        "\"timestamp\":%u,\"minC\":%.1f,\"maxC\":%.1f,\"senxorTempC\":%.2f}",
        thermal.frameCounter(),
        MI48_SENXOR_WIDTH, MI48_SENXOR_HEIGHT,
        thermal.crcOk() ? "true" : "false",
        thermal.timestamp(),
        MI48::rawToCelsius(thermal.minRaw()),
        MI48::rawToCelsius(thermal.maxRaw()),
        thermal.senxorTempC());
    server.send(200, "application/json", buf);
}

void web_init()
{
    server.on("/", root_page);
    server.on("/frame", frame_page);
    server.on("/status", status_page);
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