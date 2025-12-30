// ===============================================
// SignSpeak Frontend (DUAL MODE: CNN + LSTM)
// ===============================================
const API_BASE = "http://127.0.0.1:5000";

// --- DOM Elements ---
const videoElement = document.getElementById("video");
const canvasElement = document.getElementById("output");
const statusOverlay = document.getElementById("statusOverlay");

// Mode Buttons & Indicators
const btnStatic = document.getElementById("btnStatic");
const btnDynamic = document.getElementById("btnDynamic");
const modeBadge = document.getElementById("modeBadge");
const recordOverlay = document.getElementById("recordOverlay");
const btnRecord = document.getElementById("btnRecord");
const recordingStatus = document.getElementById("recordingStatus");

// Results
const predictResult = document.getElementById("predictResult");
const predictConfidence = document.getElementById("predictConfidence");
const statusText = document.getElementById("status");
const handCount = document.getElementById("handCount");
const handCountText = document.getElementById("handCountText");
const handDetectionStatus = document.getElementById("handDetectionStatus");
const fpsCounter = document.getElementById("fpsCounter");

// Settings
const intervalSlider = document.getElementById("intervalSlider");
const intervalValue = document.getElementById("intervalValue");
const confidenceSlider = document.getElementById("confidenceSlider");
const confidenceValue = document.getElementById("confidenceValue");
const handRequiredToggle = document.getElementById("handRequiredToggle");
const debugCropToggle = document.getElementById("debugCropToggle");
const debugContainer = document.getElementById("debugContainer");
const debugCrop = document.getElementById("debugCrop");
const debugCropCtx = debugCrop.getContext("2d");

// Tools
const wordOutput = document.getElementById("wordOutput");
const clearWordBtn = document.getElementById("clearWordBtn");
const addWordBtn = document.getElementById("addWordBtn");
const imageInput = document.getElementById("imageInput");
const predictImageBtn = document.getElementById("predictImageBtn");

const context = canvasElement.getContext("2d");

// --- GLOBAL STATE ---
let CURRENT_MODE = "STATIC"; // 'STATIC' (CNN) atau 'DYNAMIC' (LSTM)
let isRecording = false;
let handsDetected = false;
let lastHandLandmarks = null;
let cnnIntervalId = null;
let currentWord = "";
let lastPredictedLabel = "";
let lastPredictedConfidence = 0;
let backendReady = false;
let isProcessingCNN = false;
let consecutiveErrors = 0;

// Default Settings
let CNN_PREDICT_INTERVAL = 800;
let MIN_CONFIDENCE_DISPLAY = 0.30;
let MIN_CONFIDENCE_FOR_ADD = 0.70;
let REQUIRE_HAND_FOR_PREDICTION = true;
const MAX_CONSECUTIVE_ERRORS = 10;

// FPS
let frameCount = 0;
let lastFpsUpdate = Date.now();

// Canvas Config
const VIDEO_WIDTH = 640;
const VIDEO_HEIGHT = 480;
canvasElement.width = VIDEO_WIDTH;
canvasElement.height = VIDEO_HEIGHT;

// ======================================================
// 1. LOGIKA GANTI MODE (SWITCHER)
// ======================================================
function setMode(mode) {
    CURRENT_MODE = mode;
    
    if (mode === "STATIC") {
        // --- Masuk Mode ABJAD (CNN) ---
        // Update Tampilan Tombol
        btnStatic.className = "px-6 py-2.5 rounded-lg bg-indigo-600 font-bold text-sm text-white shadow-lg transition-all transform scale-105";
        btnDynamic.className = "px-6 py-2.5 rounded-lg text-gray-400 hover:text-white font-bold text-sm transition-all hover:bg-gray-800";
        
        // Update Badge & UI
        modeBadge.textContent = "MODE: ABJAD (CNN)";
        modeBadge.className = "text-indigo-400 font-bold text-xs tracking-wider";
        recordOverlay.classList.add("hidden"); // Sembunyikan tombol rekam
        
        // Reset Hasil
        predictResult.textContent = "-";
        predictConfidence.textContent = "Menunggu...";
        
        // Jalankan Loop CNN
        startCnnLoop(); 
        
    } else {
        // --- Masuk Mode GERAKAN (LSTM) ---
        // Update Tampilan Tombol
        btnDynamic.className = "px-6 py-2.5 rounded-lg bg-rose-600 font-bold text-sm text-white shadow-lg transition-all transform scale-105";
        btnStatic.className = "px-6 py-2.5 rounded-lg text-gray-400 hover:text-white font-bold text-sm transition-all hover:bg-gray-800";
        
        // Update Badge & UI
        modeBadge.textContent = "MODE: GERAKAN (LSTM)";
        modeBadge.className = "text-rose-400 font-bold text-xs tracking-wider";
        recordOverlay.classList.remove("hidden"); // Munculkan tombol rekam
        
        // Reset Hasil
        predictResult.textContent = "SIAP";
        predictConfidence.textContent = "Tekan Rekam";
        
        // Matikan Loop CNN (Biar tidak bentrok)
        stopCnnLoop(); 
    }
}

// Event Listener Tombol Header
btnStatic.onclick = () => setMode("STATIC");
btnDynamic.onclick = () => setMode("DYNAMIC");

// ======================================================
// 2. HELPER FUNCTIONS
// ======================================================
function showMessage(message, isError = false) {
    statusText.textContent = `Status: ${message}`;
    statusText.className = isError ? "text-red-400 text-xs font-mono" : "text-gray-400 text-xs font-mono";
}

function updateFPS() {
    frameCount++;
    const now = Date.now();
    const elapsed = now - lastFpsUpdate;
    if (elapsed >= 1000) {
        fpsCounter.textContent = Math.round((frameCount * 1000) / elapsed);
        frameCount = 0;
        lastFpsUpdate = now;
    }
}

// Fungsi Crop Kotak (Square) untuk Tangan
function getHandBoundingBox(landmarks) {
    let minX = 1, minY = 1, maxX = 0, maxY = 0;
    for (const lm of landmarks) {
        if (lm.x < minX) minX = lm.x;
        if (lm.y < minY) minY = lm.y;
        if (lm.x > maxX) maxX = lm.x;
        if (lm.y > maxY) maxY = lm.y;
    }
    const width = maxX - minX;
    const height = maxY - minY;
    const padding = 0.4;
    const size = Math.max(width, height) * (1 + padding);
    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;
    let x1 = Math.max(0, centerX - size / 2);
    let y1 = Math.max(0, centerY - size / 2);
    return { x: x1, y: y1, size: Math.min(size, 1-x1, 1-y1) };
}

async function fetchApi(url, options = {}) {
    try {
        const res = await fetch(url, options);
        if (!res.ok) throw new Error("API Error");
        return res.json();
    } catch (err) { throw err; }
}

// ======================================================
// 3. MEDIAPIPE & KAMERA
// ======================================================
const hands = new Hands({locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`});
hands.setOptions({maxNumHands: 2, modelComplexity: 1, minDetectionConfidence: 0.6, minTrackingConfidence: 0.5});

hands.onResults((results) => {
    context.save();
    context.clearRect(0, 0, canvasElement.width, canvasElement.height);
    context.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);

    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        handsDetected = true;
        lastHandLandmarks = results.multiHandLandmarks;
        
        handCount.classList.remove("hidden");
        handCountText.textContent = results.multiHandLandmarks.length;
        handDetectionStatus.innerHTML = `✅ Terdeteksi`;
        
        results.multiHandLandmarks.forEach((landmarks) => {
            drawConnectors(context, landmarks, HAND_CONNECTIONS, {color: "#00FF00", lineWidth: 2});
            drawLandmarks(context, landmarks, {color: "#FF0000", lineWidth: 1, radius: 2});
        });
    } else {
        handsDetected = false;
        lastHandLandmarks = null;
        handCount.classList.add("hidden");
        handDetectionStatus.innerHTML = "⏳ Menunggu tangan...";
    }
    context.restore();
    updateFPS();
});

const cam = new Camera(videoElement, {
    onFrame: async () => await hands.send({image: videoElement}),
    width: VIDEO_WIDTH, height: VIDEO_HEIGHT
});

cam.start().then(() => {
    statusOverlay.style.display = "none";
    showMessage("✅ Kamera Aktif");
    checkBackend(); // Cek koneksi ke Flask
});

// ======================================================
// 4. LOGIC CNN (STATIC / ABJAD)
// ======================================================
function startCnnLoop() {
    if (cnnIntervalId) clearInterval(cnnIntervalId);
    cnnIntervalId = setInterval(predictCnn, CNN_PREDICT_INTERVAL);
}

function stopCnnLoop() {
    if (cnnIntervalId) clearInterval(cnnIntervalId);
    cnnIntervalId = null;
}

async function predictCnn() {
    // Jangan prediksi jika bukan mode static atau sedang merekam
    if (CURRENT_MODE !== "STATIC" || isRecording) return;
    if (REQUIRE_HAND_FOR_PREDICTION && !handsDetected) return;

    try {
        const tempCanvas = document.createElement("canvas");
        const tCtx = tempCanvas.getContext("2d");
        tempCanvas.width = 224; tempCanvas.height = 224;

        // Smart Crop (Potong kotak tangan)
        if (handsDetected && lastHandLandmarks) {
            const bbox = getHandBoundingBox(lastHandLandmarks[0]);
            tCtx.drawImage(canvasElement, 
                bbox.x * VIDEO_WIDTH, bbox.y * VIDEO_HEIGHT, bbox.size * VIDEO_WIDTH, bbox.size * VIDEO_HEIGHT,
                0, 0, 224, 224);
        } else {
            // Center Crop (Kalau tidak ada tangan tapi dipaksa)
            const size = Math.min(VIDEO_WIDTH, VIDEO_HEIGHT);
            tCtx.drawImage(canvasElement, (VIDEO_WIDTH-size)/2, (VIDEO_HEIGHT-size)/2, size, size, 0, 0, 224, 224);
        }

        // Tampilkan Debug Crop (Pojok kiri bawah)
        if (debugCropToggle.checked) {
            debugCropCtx.drawImage(tempCanvas, 0, 0, 100, 100);
        }

        const base64 = tempCanvas.toDataURL("image/jpeg", 0.8).split(",")[1];
        
        // Kirim ke Backend (/predict)
        const res = await fetchApi(`${API_BASE}/predict`, {
            method: "POST", headers: {"Content-Type": "application/json"},
            body: JSON.stringify({image: base64})
        });

        if (res.label) {
            const conf = res.confidence;
            if (conf >= MIN_CONFIDENCE_DISPLAY) {
                predictResult.textContent = res.label;
                predictConfidence.textContent = `${(conf*100).toFixed(1)}% (CNN)`;
                predictResult.className = "text-6xl font-black text-white mb-2 tracking-tight drop-shadow-[0_0_15px_rgba(99,102,241,0.8)]";
                lastPredictedLabel = res.label;
                lastPredictedConfidence = conf;
            } else {
                predictResult.textContent = "?";
                predictConfidence.textContent = "Rendah";
                predictResult.className = "text-6xl font-black text-gray-600 mb-2";
            }
        }
    } catch (e) { console.error(e); }
}

// ======================================================
// 5. LOGIC LSTM (DYNAMIC / GERAKAN)
// ======================================================
btnRecord.onclick = async () => {
    if (isRecording) return;
    isRecording = true;
    
    // UI Updates
    recordingStatus.classList.remove("hidden");
    recordOverlay.classList.add("hidden");
    btnRecord.classList.add("scale-90"); // Efek tekan
    
    // Rekam Stream Canvas (Format WebM)
    const stream = canvasElement.captureStream(30); 
    const recorder = new MediaRecorder(stream, {mimeType: 'video/webm'});
    const chunks = [];
    
    recorder.ondataavailable = e => chunks.push(e.data);
    
    recorder.onstop = async () => {
        const blob = new Blob(chunks, {type: 'video/webm'});
        const reader = new FileReader();
        reader.readAsDataURL(blob);
        
        reader.onloadend = async () => {
            const base64 = reader.result.split(",")[1];
            predictResult.textContent = "...";
            predictConfidence.textContent = "Menganalisis...";
            
            try {
                // Kirim ke Backend (/predict_lstm)
                const res = await fetchApi(`${API_BASE}/predict_lstm`, {
                    method: "POST", headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({video: base64})
                });
                
                predictResult.textContent = res.label;
                predictConfidence.textContent = `${(res.confidence*100).toFixed(1)}% (LSTM)`;
                predictResult.className = "text-6xl font-black text-rose-500 mb-2 tracking-tight drop-shadow-lg";
                
                lastPredictedLabel = res.label;
                lastPredictedConfidence = res.confidence;
                
            } catch (e) {
                predictResult.textContent = "ERR";
                predictConfidence.textContent = "Gagal";
                showMessage(`Gagal: ${e.message}`, true);
            }
            
            // Reset UI
            isRecording = false;
            recordingStatus.classList.add("hidden");
            recordOverlay.classList.remove("hidden");
            btnRecord.classList.remove("scale-90");
        };
    };
    
    recorder.start();
    // Rekam selama 2 detik
    setTimeout(() => recorder.stop(), 2000); 
};

// ======================================================
// 6. SETTINGS & TOOLS HANDLERS
// ======================================================
intervalSlider.oninput = (e) => {
    CNN_PREDICT_INTERVAL = parseInt(e.target.value);
    intervalValue.textContent = CNN_PREDICT_INTERVAL + " ms";
    if (CURRENT_MODE === "STATIC") { stopCnnLoop(); startCnnLoop(); }
};
confidenceSlider.oninput = (e) => {
    MIN_CONFIDENCE_DISPLAY = e.target.value / 100;
    confidenceValue.textContent = e.target.value + "%";
};
handRequiredToggle.onchange = (e) => REQUIRE_HAND_FOR_PREDICTION = e.target.checked;
debugCropToggle.onchange = (e) => {
    if(e.target.checked) debugContainer.classList.remove("hidden");
    else debugContainer.classList.add("hidden");
};

// Word Builder (Tambah Kata)
addWordBtn.onclick = () => {
    if(!lastPredictedLabel || lastPredictedConfidence < MIN_CONFIDENCE_FOR_ADD) {
        showMessage("Hasil belum valid / confidence rendah", true);
        return;
    }
    // Tambah spasi jika mode gerakan (karena gerakan biasanya berupa kata)
    const spacer = CURRENT_MODE === "DYNAMIC" ? " " : "";
    currentWord += spacer + lastPredictedLabel;
    wordOutput.textContent = currentWord;
    showMessage(`Ditambahkan: ${lastPredictedLabel}`);
};

clearWordBtn.onclick = () => { 
    currentWord = ""; 
    wordOutput.textContent = "---"; 
    showMessage("Kata dihapus");
};

// Upload Gambar Manual
predictImageBtn.onclick = () => {
    const file = imageInput.files[0];
    if(!file) {
        showMessage("Pilih gambar dulu!", true);
        return;
    }
    const reader = new FileReader();
    reader.onload = async (e) => {
        const b64 = e.target.result.split(",")[1];
        try {
            const res = await fetchApi(`${API_BASE}/predict`, {
                method:"POST", headers:{"Content-Type":"application/json"},
                body: JSON.stringify({image:b64})
            });
            if(res.label) {
                predictResult.textContent = res.label;
                predictConfidence.textContent = "Upload";
                showMessage(`Hasil Upload: ${res.label}`);
            }
        } catch(e) {
            showMessage("Gagal prediksi gambar", true);
        }
    };
    reader.readAsDataURL(file);
};

// ======================================================
// 7. INITIALIZATION & HEALTH CHECK
// ======================================================
function checkBackend() {
    setInterval(() => {
        fetch(`${API_BASE}/health`)
            .then(r => r.json())
            .then(d => {
                // Update Status CNN
                document.getElementById("stCnn").className = d.cnn_ready ? "text-emerald-400" : "text-red-400";
                document.getElementById("stCnn").textContent = d.cnn_ready ? "CNN: ●" : "CNN: ○";
                
                // Update Status LSTM
                document.getElementById("stLstm").className = d.lstm_ready ? "text-emerald-400" : "text-red-400";
                document.getElementById("stLstm").textContent = d.lstm_ready ? "LSTM: ●" : "LSTM: ○";
                
                backendReady = d.cnn_ready;
            })
            .catch(() => {
                document.getElementById("stCnn").className = "text-red-400";
                document.getElementById("stLstm").className = "text-red-400";
            });
    }, 3000);
}

// Mulai di Mode Default (Static)
startCnnLoop();