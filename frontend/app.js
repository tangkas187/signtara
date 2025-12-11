// ===============================================
// SignSpeak Frontend (FINAL - CNN + KNN + FUSION)
// ===============================================
const API_BASE = "http://127.0.0.1:5000";

// --- DOM Elements ---
const videoElement = document.getElementById("video");
const canvasElement = document.getElementById("output");
const statusOverlay = document.getElementById("statusOverlay");
const predictResult = document.getElementById("predictResult");
const predictConfidence = document.getElementById("predictConfidence");
const statusText = document.getElementById("status");
const labelInput = document.getElementById("labelInput");
const addSampleButton = document.getElementById("addSample");
const trainModelButton = document.getElementById("trainModel");
const imageInput = document.getElementById("imageInput");
const predictImageBtn = document.getElementById("predictImageBtn");
const modeToggleBtn = document.getElementById("modeToggleBtn");
const wordOutput = document.getElementById("wordOutput");
const clearWordBtn = document.getElementById("clearWordBtn");
const addWordBtn = document.getElementById("addWordBtn");
const fusionVideoBtn = document.getElementById("fusionVideoBtn"); // tombol tambahan opsional

const context = canvasElement.getContext("2d");

// --- Global State ---
let lastLandmarks = null;
let currentPredictionMode = "CNN_REALTIME"; // CNN_REALTIME, KNN, FUSION
let intervalId = null;
let currentWord = "";
let lastPredictedLabel = "";
const MIN_CONFIDENCE_FOR_ADD = 0.75;
const PREDICT_INTERVAL = 200;

// --- API Routes ---
const CNN_PREDICT_URL = `${API_BASE}/predict`;
const FUSION_PREDICT_URL = `${API_BASE}/predict_fusion`;
const FUSION_VIDEO_URL = `${API_BASE}/predict_video_fusion`;
const KNN_ADD_SAMPLE_URL = `${API_BASE}/knn/add_sample`;
const KNN_TRAIN_URL = `${API_BASE}/knn/train`;
const KNN_PREDICT_URL = `${API_BASE}/knn/predict`;

// --- Canvas Setup ---
const VIDEO_WIDTH = 640;
const VIDEO_HEIGHT = 480;
canvasElement.width = VIDEO_WIDTH;
canvasElement.height = VIDEO_HEIGHT;

// ======================================================
// Helper Functions
// ======================================================
function showMessage(message, isError = false) {
  if (isError) {
    statusText.classList.add("text-red-500");
    statusText.classList.remove("text-gray-500");
    console.error("APP MESSAGE:", message);
  } else {
    statusText.classList.remove("text-red-500");
    statusText.classList.add("text-gray-500");
  }
  statusText.textContent = `Status: ${message}`;
}

function extractFeaturesSingle(handLandmarks) {
  if (!handLandmarks || handLandmarks.length === 0) return [];
  const base = handLandmarks[0];
  const features = [];
  for (const lm of handLandmarks) {
    features.push(lm.x - base.x, lm.y - base.y, lm.z - (base.z || 0));
  }
  return features;
}

function extractFeaturesMultiHands(multiHandLandmarks) {
  if (!multiHandLandmarks || multiHandLandmarks.length === 0) return [];
  let allFeatures = [];
  for (const hand of multiHandLandmarks) {
    allFeatures = allFeatures.concat(extractFeaturesSingle(hand));
  }
  return allFeatures;
}

async function fetchApi(url, options = {}) {
  try {
    const response = await fetch(url, options);
    if (!response.ok) {
      const errorBody = await response.json().catch(() => ({}));
      throw new Error(errorBody.error || response.statusText);
    }
    return response.json();
  } catch (err) {
    console.error("API Error:", err.message);
    showMessage("❌ Gagal konek ke Flask backend.", true);
    throw err;
  }
}

async function predictImageApi(imageBase64) {
  return fetchApi(FUSION_PREDICT_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: imageBase64.split(",")[1] }),
  });
}

async function predictLandmarkApi(features) {
  try {
    const response = await fetch(KNN_PREDICT_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ features }),
    });
    if (response.ok) return response.json();
  } catch {}
  return null;
}

// ======================================================
// MediaPipe Hands (2 Hands)
// ======================================================
const hands = new Hands({
  locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapip/hands/${file}`,
});
hands.setOptions({
  maxNumHands: 2,
  modelComplexity: 1,
  minDetectionConfidence: 0.7,
  minTrackingConfidence: 0.6,
});

hands.onResults(async (results) => {
  context.save();
  context.clearRect(0, 0, canvasElement.width, canvasElement.height);
  context.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);

  if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
    lastLandmarks = results.multiHandLandmarks;

    results.multiHandLandmarks.forEach((lm, index) => {
      const color = index === 0 ? "#00FF00" : "#FFC300";
      drawConnectors(context, lm, HAND_CONNECTIONS, { color, lineWidth: 3 });
      drawLandmarks(context, lm, { color: "#FF0000", lineWidth: 1 });
    });

    if (currentPredictionMode === "KNN") {
      const features = extractFeaturesMultiHands(results.multiHandLandmarks);
      const out = await predictLandmarkApi(features);
      if (out && out.label) {
        predictResult.textContent = out.label;
        predictConfidence.textContent = `${(out.confidence * 100).toFixed(2)}% (KNN)`;
        lastPredictedLabel = out.label;
      } else {
        predictResult.textContent = "?";
        predictConfidence.textContent = "Not Recognized (KNN)";
        lastPredictedLabel = "";
      }
    }
  } else {
    lastLandmarks = null;
  }
  context.restore();
});

// ======================================================
// Kamera & Loop Prediksi CNN/FUSION
// ======================================================
const cam = new Camera(videoElement, {
  onFrame: async () => await hands.send({ image: videoElement }),
  width: VIDEO_WIDTH,
  height: VIDEO_HEIGHT,
});

cam.start().then(() => {
  statusOverlay.style.opacity = "0";
  statusOverlay.style.display = "none";
  showMessage("✅ Kamera siap. Mode CNN Realtime aktif.");
  startPredictionLoop();
}).catch((err) => {
  showMessage(`❌ Gagal akses kamera: ${err.name}`, true);
});

function startPredictionLoop() {
  if (intervalId) return;
  intervalId = setInterval(captureAndPredict, PREDICT_INTERVAL);
  console.log("🧠 Loop prediksi dimulai.");
}

function stopPredictionLoop() {
  if (intervalId) {
    clearInterval(intervalId);
    intervalId = null;
    console.log("🛑 Loop prediksi dihentikan.");
  }
}

function captureAndPredict() {
  if (videoElement.paused || videoElement.ended) return;
  if (!["CNN_REALTIME", "FUSION"].includes(currentPredictionMode)) return;

  const tempCanvas = document.createElement("canvas");
  const ctx = tempCanvas.getContext("2d");
  tempCanvas.width = 128;
  tempCanvas.height = 128;
  ctx.drawImage(canvasElement, 0, 0, 128, 128);
  const dataURL = tempCanvas.toDataURL("image/jpeg", 0.8);

  predictImageApi(dataURL)
    .then((out) => {
      if (out && out.label) {
        const conf = parseFloat(out.confidence);
        const modeLabel = out.mode ? `(${out.mode.toUpperCase()})` : "(FUSION)";
        predictResult.textContent = out.label;
        predictConfidence.textContent = `${(conf * 100).toFixed(2)}% ${modeLabel}`;
        lastPredictedLabel = conf > 0.5 ? out.label : "";
      }
    })
    .catch(() => {
      predictResult.textContent = "ERR";
      predictConfidence.textContent = "API Down";
    });
}

// ======================================================
// 🎞️ Fusion Video Prediction
// ======================================================
async function recordAndPredictVideoFusion(seconds = 2) {
  showMessage("🎥 Merekam video untuk prediksi fusion...");
  const stream = videoElement.srcObject;
  if (!stream) return showMessage("❌ Kamera belum aktif.", true);

  const recorder = new MediaRecorder(stream, { mimeType: "video/webm" });
  const chunks = [];
  recorder.ondataavailable = (e) => chunks.push(e.data);
  recorder.start();

  await new Promise((resolve) => setTimeout(resolve, seconds * 1000));
  recorder.stop();

  recorder.onstop = async () => {
    const blob = new Blob(chunks, { type: "video/webm" });
    const reader = new FileReader();
    reader.onloadend = async () => {
      const base64Video = reader.result.split(",")[1];
      showMessage("🧠 Mengirim video ke backend untuk prediksi fusion...");
      try {
        const res = await fetchApi(FUSION_VIDEO_URL, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ video: base64Video }),
        });
        predictResult.textContent = res.label;
        predictConfidence.textContent = `${(res.confidence * 100).toFixed(2)}% (VIDEO-FUSION)`;
        showMessage(`🎯 Prediksi video: ${res.label}`);
      } catch (e) {
        showMessage(`❌ Gagal prediksi video: ${e.message}`, true);
      }
    };
    reader.readAsDataURL(blob);
  };
}

// ======================================================
// UI & Buttons
// ======================================================
modeToggleBtn.onclick = () => {
  if (currentPredictionMode === "CNN_REALTIME") {
    currentPredictionMode = "KNN";
    stopPredictionLoop();
    modeToggleBtn.textContent = "Aktif: MODE KNN";
    modeToggleBtn.classList.replace("bg-indigo-600", "bg-green-600");
    showMessage("✅ Mode KNN Aktif.");
  } else if (currentPredictionMode === "KNN") {
    currentPredictionMode = "FUSION";
    startPredictionLoop();
    modeToggleBtn.textContent = "Aktif: MODE FUSION (CNN+KNN)";
    modeToggleBtn.classList.replace("bg-green-600", "bg-yellow-600");
    showMessage("✅ Mode Fusion aktif (gabungan CNN+KNN).");
  } else {
    currentPredictionMode = "CNN_REALTIME";
    startPredictionLoop();
    modeToggleBtn.textContent = "Aktif: MODE CNN";
    modeToggleBtn.classList.replace("bg-yellow-600", "bg-indigo-600");
    showMessage("✅ Mode CNN Realtime aktif.");
  }
  predictResult.textContent = "-";
  predictConfidence.textContent = "-";
  lastPredictedLabel = "";
};

// Rekam dan prediksi video fusion
if (fusionVideoBtn) {
  fusionVideoBtn.onclick = async () => {
    stopPredictionLoop();
    await recordAndPredictVideoFusion(3); // durasi 3 detik
    startPredictionLoop();
  };
}

// Tambah sampel dua tangan (KNN)
addSampleButton.onclick = async () => {
  if (currentPredictionMode !== "KNN") return showMessage("⚠️ Pindah ke MODE KNN dulu.", true);
  if (!lastLandmarks || lastLandmarks.length === 0) return showMessage("⚠️ Tangan tidak terdeteksi.", true);

  const label = labelInput.value.trim().toUpperCase();
  if (!label) return showMessage("⚠️ Masukkan label huruf.", true);

  const features = extractFeaturesMultiHands(lastLandmarks);
  showMessage(`💾 Menyimpan sampel: ${label}`);
  try {
    const res = await fetchApi(KNN_ADD_SAMPLE_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ label, features }),
    });
    showMessage(`✅ Sampel disimpan (${label}). Total: ${res.total_samples}`);
  } catch (e) {
    showMessage(`❌ Gagal menyimpan sampel: ${e.message}`, true);
  }
};

// Latih model
trainModelButton.onclick = async () => {
  showMessage("🧠 Melatih model KNN...");
  try {
    const res = await fetchApi(KNN_TRAIN_URL, { method: "POST" });
    showMessage(`✅ Selesai latih. Akurasi ${(res.accuracy * 100).toFixed(1)}%.`);
  } catch (e) {
    showMessage(`❌ Gagal latih model: ${e.message}`, true);
  }
};

// Tambah huruf ke kata
addWordBtn.onclick = () => {
  const confText = predictConfidence.textContent;
  const match = confText.match(/([\d\.]+)%/);
  const conf = match ? parseFloat(match[1]) / 100 : 0;
  if (!lastPredictedLabel) return showMessage("⚠️ Tidak ada huruf terdeteksi.", true);
  if (conf < MIN_CONFIDENCE_FOR_ADD) return showMessage(`⚠️ Confidence rendah (${(conf * 100).toFixed(1)}%).`, true);
  if (lastPredictedLabel !== currentWord.slice(-1)) {
    currentWord += lastPredictedLabel;
    wordOutput.textContent = currentWord;
    showMessage(`✅ Huruf '${lastPredictedLabel}' ditambahkan.`);
  } else {
    showMessage(`⚠️ Huruf '${lastPredictedLabel}' sudah ada.`);
  }
};

// Bersihkan kata
clearWordBtn.onclick = () => {
  currentWord = "";
  wordOutput.textContent = "---";
  showMessage("✅ Kata dibersihkan.");
};

// Upload gambar manual
predictImageBtn.onclick = async () => {
  const file = imageInput.files[0];
  if (!file) return showMessage("⚠️ Pilih gambar dulu!", true);
  const reader = new FileReader();
  reader.onload = async (e) => {
    const base64 = e.target.result;
    try {
      const out = await predictImageApi(base64);
      predictResult.textContent = out.label;
      predictConfidence.textContent = `${(out.confidence * 100).toFixed(2)}% (Upload)`;
      showMessage(`✅ Prediksi: ${out.label}`);
    } catch {
      showMessage("❌ Gagal prediksi gambar.", true);
    }
  };
  reader.readAsDataURL(file);
};

// Backend check
window.addEventListener("load", async () => {
  try {
    const response = await fetch(`${API_BASE}/`);
    const data = await response.json();
    if (!data.cnn_loaded) showMessage("⚠️ Model CNN belum dimuat.", true);
  } catch (err) {
    showMessage("❌ Backend tidak terhubung.", true);
  }
});