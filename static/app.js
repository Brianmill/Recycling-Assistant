const videoEl = document.getElementById("video");
const overlayEl = document.getElementById("overlayCanvas");
const canvasEl = document.getElementById("snapshotCanvas");
const startCameraBtn = document.getElementById("startCameraBtn");
const analyzeBtn = document.getElementById("analyzeBtn");

const locationInput = document.getElementById("locationInput");
const sourceUrlInput = document.getElementById("sourceUrlInput");
const useLocationBtn = document.getElementById("useLocationBtn");
const loadRulesBtn = document.getElementById("loadRulesBtn");

const locationStatus = document.getElementById("locationStatus");
const detectStatus = document.getElementById("detectStatus");
const summaryBadge = document.getElementById("summaryBadge");
const resultCards = document.getElementById("resultCards");
const guidanceNotes = document.getElementById("guidanceNotes");

let stream;
let trackingTimer = null;
let trackingBusy = false;
const loadingSpinner = document.getElementById("loadingSpinner");
const sourceDropdown = document.getElementById("sourceDropdown");
const sourcesList = document.getElementById("sourcesList");
let loadingCount = 0;
let loadingTimer = null;
let loadingShownAt = 0;
const LOADING_DELAY_MS = 180;
const LOADING_MIN_VISIBLE_MS = 350;

function showLoading() {
  loadingCount += 1;
  if (loadingCount > 1) {
    return;
  }

  if (loadingTimer) {
    window.clearTimeout(loadingTimer);
  }

  loadingTimer = window.setTimeout(() => {
    loadingShownAt = Date.now();
    loadingSpinner.classList.remove("hidden");
    loadingTimer = null;
  }, LOADING_DELAY_MS);
}

function hideLoading() {
  loadingCount = Math.max(0, loadingCount - 1);
  if (loadingCount > 0) {
    return;
  }

  if (loadingTimer) {
    window.clearTimeout(loadingTimer);
    loadingTimer = null;
  }

  const elapsed = Date.now() - loadingShownAt;
  const hideNow = () => {
    loadingSpinner.classList.add("hidden");
    loadingShownAt = 0;
  };

  if (loadingShownAt && elapsed < LOADING_MIN_VISIBLE_MS) {
    window.setTimeout(hideNow, LOADING_MIN_VISIBLE_MS - elapsed);
  } else {
    hideNow();
  }
}

const STATUS_COLORS = {
  recyclable: "rgb(0, 180, 0)",
  not_recyclable: "rgb(220, 0, 0)",
  unknown: "rgb(255, 180, 0)",
};

function setStatus(el, text, cls = "") {
  el.textContent = text;
  el.className = `status ${cls}`.trim();
}

function setSummary(summary) {
  summaryBadge.textContent = `Summary: ${summary}`;
  summaryBadge.className = "badge";

  if (summary === "recycle") {
    summaryBadge.classList.add("ok");
  } else if (summary === "trash") {
    summaryBadge.classList.add("bad");
  } else {
    summaryBadge.classList.add("warn");
  }
}

function renderGuidance(guidance) {
  guidanceNotes.innerHTML = "";

  if (!guidance) {
    return;
  }

  const items = [];
  items.push(`Location: ${guidance.location}`);
  if (guidance.allowed.length) {
    items.push(`Allowed: ${guidance.allowed.join(", ")}`);
  }
  if (guidance.disallowed.length) {
    items.push(`Disallowed: ${guidance.disallowed.join(", ")}`);
  }
  if (guidance.disallowed_items && guidance.disallowed_items.length) {
    items.push(`Restricted items found: ${guidance.disallowed_items.join(", ")}`);
  }
  guidance.notes.forEach((note) => items.push(note));

  items.slice(0, 8).forEach((text) => {
    const li = document.createElement("li");
    li.textContent = text;
    guidanceNotes.appendChild(li);
  });

  // Render sources dropdown if sources exist
  if (guidance.sources && guidance.sources.length > 0) {
    sourceDropdown.classList.remove("hidden");
    sourcesList.innerHTML = "";
    guidance.sources.forEach((url) => {
      const li = document.createElement("li");
      const a = document.createElement("a");
      a.href = url;
      a.target = "_blank";
      a.textContent = url;
      li.appendChild(a);
      sourcesList.appendChild(li);
    });
  } else {
    sourceDropdown.classList.add("hidden");
  }
}

function renderDetections(detections) {
  resultCards.innerHTML = "";

  if (!detections.length) {
    const empty = document.createElement("p");
    empty.className = "meta";
    empty.textContent = "No item detected in this frame.";
    resultCards.appendChild(empty);
    return;
  }

  detections.forEach((det) => {
    const card = document.createElement("article");
    card.className = "card";

    const heading = document.createElement("h3");
    heading.textContent = det.label;

    const confidence = document.createElement("p");
    confidence.className = "meta";
    confidence.textContent = `Confidence: ${det.confidence}`;

    const status = document.createElement("p");
    status.className = `meta ${det.final_status === "recyclable" ? "ok" : det.final_status === "not_recyclable" ? "bad" : "warn"}`;
    status.textContent = `Decision: ${det.final_status} (base: ${det.base_status})`;

    const reason = document.createElement("p");
    reason.className = "meta";
    reason.textContent = det.local_reason;

    card.appendChild(heading);
    card.appendChild(confidence);
    card.appendChild(status);
    card.appendChild(reason);
    resultCards.appendChild(card);
  });
}

function drawOverlay(detections) {
  const ctx = overlayEl.getContext("2d");
  const width = videoEl.videoWidth || 640;
  const height = videoEl.videoHeight || 480;

  overlayEl.width = width;
  overlayEl.height = height;
  ctx.clearRect(0, 0, width, height);

  detections.forEach((det) => {
    if (!Array.isArray(det.box) || det.box.length !== 4) {
      return;
    }

    const [x1, y1, x2, y2] = det.box;
    const boxW = Math.max(0, x2 - x1);
    const boxH = Math.max(0, y2 - y1);
    const color = STATUS_COLORS[det.final_status] || STATUS_COLORS.unknown;

    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.strokeRect(x1, y1, boxW, boxH);

    let label;

    if (det.label === "plastic") {
      label = `${det.label} #1 | ${det.final_status}`;
    } else {
      label = `${det.label} | ${det.final_status}`;
    }
    
    ctx.font = "15px IBM Plex Mono";
    const textWidth = ctx.measureText(label).width;
    const textY = y1 > 26 ? y1 - 10 : y1 + 20;

    ctx.fillStyle = color;
    ctx.fillRect(x1, textY - 16, textWidth + 12, 22);
    ctx.fillStyle = "#ffffff";
    ctx.fillText(label, x1 + 6, textY);
  });
}

async function postJson(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  const data = await response.json();
  if (!response.ok || !data.ok) {
    throw new Error(data.error || "Request failed");
  }
  return data;
}

startCameraBtn.addEventListener("click", async () => {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    videoEl.srcObject = stream;
    setStatus(detectStatus, "Camera started.", "ok");
  } catch (err) {
    setStatus(detectStatus, `Camera error: ${err.message}`, "bad");
  }
});

useLocationBtn.addEventListener("click", async () => {
  if (!navigator.geolocation) {
    setStatus(locationStatus, "Geolocation is not supported in this browser.", "warn");
    return;
  }

  setStatus(locationStatus, "Resolving location from GPS...");

  navigator.geolocation.getCurrentPosition(
    async (position) => {
      try {
        const { latitude, longitude } = position.coords;
        const data = await postJson("/api/resolve-location", {
          lat: latitude,
          lon: longitude,
        });
        locationInput.value = data.location;
        setStatus(locationStatus, `Detected location: ${data.location}`, "ok");
      } catch (err) {
        setStatus(locationStatus, err.message, "bad");
      }
    },
    (err) => {
      setStatus(locationStatus, `Location error: ${err.message}`, "bad");
    },
    { timeout: 10000 }
  );
});

loadRulesBtn.addEventListener("click", async () => {
  const location = locationInput.value.trim();
  const sourceUrl = sourceUrlInput.value.trim();

  if (!location) {
    setStatus(locationStatus, "Enter or detect location first.", "warn");
    return;
  }

  setStatus(locationStatus, "Fetching local recycling guidance...");
  showLoading();

  try {
    const data = await postJson("/api/guidelines", {
      location,
      source_url: sourceUrl,
    });
    renderGuidance(data.guidance);
    const updatedCount = data.mapping_update ? data.mapping_update.updated_count : 0;
    setStatus(locationStatus, `Local rules loaded. Mapping updated with ${updatedCount} keys.`, "ok");
  } catch (err) {
    setStatus(locationStatus, `Guidance error: ${err.message}`, "bad");
  } finally {
    hideLoading();
  }
});

async function analyzeCurrentFrame() {
  if (!videoEl.srcObject) {
    setStatus(detectStatus, "Start the camera first.", "warn");
    return;
  }

  if (trackingBusy) {
    return;
  }

  trackingBusy = true;
  showLoading();

  const context = canvasEl.getContext("2d");
  canvasEl.width = videoEl.videoWidth || 640;
  canvasEl.height = videoEl.videoHeight || 480;
  context.drawImage(videoEl, 0, 0, canvasEl.width, canvasEl.height);

  const imageData = canvasEl.toDataURL("image/jpeg", 0.9);
  const location = locationInput.value.trim();
  const sourceUrl = sourceUrlInput.value.trim();

  // Debug: log frame/post size to help diagnose missing detections
  try {
    console.log("Posting frame to /api/detect-frame, image length:", imageData.length);
  } catch (e) {
    console.warn("Could not compute image length for debug log", e);
  }

  setStatus(detectStatus, "Tracking live frames...");

  try {
    const data = await postJson("/api/detect-frame", {
      image: imageData,
      location,
      source_url: sourceUrl,
    });
    console.log("/api/detect-frame response:", data);
    setSummary(data.summary);
    renderDetections(data.detections);
    drawOverlay(data.detections);
    renderGuidance(data.guidance);
    setStatus(detectStatus, "Live tracking active.", "ok");
  } catch (err) {
    console.error("Error posting frame or handling response:", err);
    setStatus(detectStatus, `Detection error: ${err.message}`, "bad");
  } finally {
    trackingBusy = false;
    hideLoading();
  }
}

function startLiveTracking() {
  if (trackingTimer) {
    return;
  }

  analyzeBtn.textContent = "Stop Live Tracking";
  setStatus(detectStatus, "Live tracking started.", "ok");
  analyzeCurrentFrame();
  trackingTimer = window.setInterval(analyzeCurrentFrame, 1000);
}

function stopLiveTracking() {
  if (!trackingTimer) {
    return;
  }

  window.clearInterval(trackingTimer);
  trackingTimer = null;
  overlayEl.getContext("2d").clearRect(0, 0, overlayEl.width, overlayEl.height);
  analyzeBtn.textContent = "Start Live Tracking";
  setStatus(detectStatus, "Live tracking stopped.", "warn");
}

analyzeBtn.addEventListener("click", () => {
  if (!videoEl.srcObject) {
    setStatus(detectStatus, "Start the camera first.", "warn");
    return;
  }

  if (trackingTimer) {
    stopLiveTracking();
  } else {
    startLiveTracking();
  }
});

document.addEventListener("visibilitychange", () => {
  if (document.hidden) {
    stopLiveTracking();
  }
});
