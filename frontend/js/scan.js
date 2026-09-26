const LOADING_LINES = [
  "Khol liya hota Dil agar Yaaro ke sath, toh nahi khulwana padta aujaaro ke sath.",
  "Stay fit, stay healthy, keep exercising.",
  "Dil hai toh dhadkan hai, dhadkan hai toh zindagi hai.",
  "A healthy heart doesn't ask for much — just some love, less oil, and daily walks.",
  "Tension kam lo, tension dil ke liye bura hai bhai.",
  "Your heart works 24/7 without a single day off — give it a break with healthy choices.",
  "Namak kam, pyaar zyada — dil khush rahega.",
  "Every 60 seconds your heart beats about 60-100 times. Respect the hustle.",
  "Chalna shuru karo, cardiologist ke paas jaana kam karo.",
  "Analyzing pixel by pixel, just like a cardiologist reads an angiogram.",
];

let currentFile = null;
let currentReport = null;
let loadingInterval = null;

function showError(msg) {
  const box = document.getElementById("errorBox");
  box.textContent = msg;
  box.style.display = "block";
}

function hideError() {
  document.getElementById("errorBox").style.display = "none";
}

function startLoadingAnimation() {
  const overlay = document.getElementById("loadingOverlay");
  const lineEl = document.getElementById("loadingLine");
  const heartEl = document.getElementById("loadingHeart");
  overlay.classList.add("active");

  let idx = Math.floor(Math.random() * LOADING_LINES.length);
  lineEl.textContent = LOADING_LINES[idx];
  loadingInterval = setInterval(() => {
    idx = (idx + 1) % LOADING_LINES.length;
    if (typeof gsap !== "undefined") {
      gsap.to(lineEl, {
        opacity: 0,
        duration: 0.25,
        onComplete: () => {
          lineEl.textContent = LOADING_LINES[idx];
          gsap.to(lineEl, { opacity: 1, duration: 0.25 });
        },
      });
    } else {
      lineEl.textContent = LOADING_LINES[idx];
    }
  }, 1800);

  if (typeof gsap !== "undefined" && heartEl) {
    gsap.to(heartEl, { scale: 1.12, duration: 0.5, repeat: -1, yoyo: true, ease: "sine.inOut", transformOrigin: "50% 50%" });
  }
}

function stopLoadingAnimation() {
  const overlay = document.getElementById("loadingOverlay");
  overlay.classList.remove("active");
  if (loadingInterval) clearInterval(loadingInterval);
}

function severityClass(band) {
  const b = (band || "").toLowerCase();
  if (b === "severe") return "sev-severe";
  if (b === "moderate") return "sev-moderate";
  return "sev-mild";
}

function renderLesions(lesions) {
  const container = document.getElementById("lesionList");
  if (!lesions || lesions.length === 0) {
    container.innerHTML = `<p style="color:var(--ink-soft); font-size:.9rem;">No lesions detected in this scan.</p>`;
    return;
  }
  container.innerHTML = lesions
    .map(
      (l, i) => `
      <div class="lesion-row">
        <span>Lesion ${i + 1} &mdash; ${l.stenosis_percent}% narrowing</span>
        <span>
          <span class="severity-tag ${severityClass(l.severity_band)}">${l.severity_band}</span>
          ${l.confidence != null ? `<span class="confidence-tag">${l.confidence}% confidence</span>` : ""}
        </span>
      </div>`
    )
    .join("");
}

function renderPatientReport(lang) {
  if (!currentReport) return;
  const block = currentReport.patient_report[lang];
  if (!block) return;
  document.getElementById("patientSummary").textContent = block.summary || "";
  document.getElementById("patientMeaning").textContent = block.what_it_means || "";
  const list = document.getElementById("patientSuggestions");
  list.innerHTML = (block.lifestyle_suggestions || []).map((s) => `<li>${s}</li>`).join("");
  document.getElementById("disclaimerText").textContent = currentReport.disclaimer[lang] || currentReport.disclaimer.english || "";
}

function renderDoctorReport() {
  if (!currentReport) return;
  const doc = currentReport.doctor_report || {};
  document.getElementById("doctorSummary").textContent = doc.clinical_summary || "";
  document.getElementById("doctorFindings").textContent = doc.findings || "";
  document.getElementById("doctorRecommendation").textContent = doc.recommendation || "";
}

async function analyzeImage(file) {
  hideError();
  startLoadingAnimation();
  const loadingStartedAt = Date.now();
  const MIN_LOADING_MS = 4200;
  const isVideo = file.type.startsWith("video/");

  try {
    const dataUrl = await fileToDataUrl(file);
    const fileUrl = await uploadAndGetUrl(file);

    let reportData;

    if (isVideo) {
      const predictRes = await fetch(`${API_BASE}/vessel-analysis/predict-video`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ file_url: fileUrl }),
      });
      if (!predictRes.ok) {
        throw new Error(`Server responded with an error (predict-video: ${predictRes.status})`);
      }
      const videoResult = await predictRes.json();
      currentReport = null;
      reportData = {
        original_image_base64: null,
        overlay_image_base64: videoResult.overlay_image_base64,
        summary: videoResult.summary,
        lesions: videoResult.lesions,
        viewer_role: null,
      };
    } else {
      const [predictRes, reportRes] = await Promise.all([
        fetch(`${API_BASE}/vessel-analysis/predict`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ file_url: fileUrl }),
        }),
        fetch(`${API_BASE}/vessel-analysis/report`, {
          method: "POST",
          headers: { "Content-Type": "application/json", ...authHeader() },
          body: JSON.stringify({ file_url: fileUrl }),
        }),
      ]);

      if (!predictRes.ok || !reportRes.ok) {
        throw new Error(`Server responded with an error (predict: ${predictRes.status}, report: ${reportRes.status})`);
      }

      reportData = await reportRes.json();
      currentReport = reportData.report;
    }

    const elapsed = Date.now() - loadingStartedAt;
    if (elapsed < MIN_LOADING_MS) {
      await new Promise((r) => setTimeout(r, MIN_LOADING_MS - elapsed));
    }

    document.getElementById("originalImg").src = reportData.original_image_base64
      ? `data:image/png;base64,${reportData.original_image_base64}`
      : (isVideo ? "" : dataUrl);
    document.getElementById("overlayImg").src = reportData.overlay_image_base64
      ? `data:image/png;base64,${reportData.overlay_image_base64}`
      : (isVideo ? "" : dataUrl);
    document.getElementById("summaryText").textContent = reportData.summary || "";
    renderLesions(reportData.lesions);

    const doctorCard = document.getElementById("doctorReportCard");
    const modeBanner = document.getElementById("viewerModeBanner");
    if (isVideo) {
      doctorCard.style.display = "none";
      modeBanner.className = "viewer-mode-banner mode-patient";
      modeBanner.style.display = "flex";
      modeBanner.innerHTML = '<i class="fa-solid fa-video"></i> Video analysis — showing the worst detected frame. Detailed doctor/patient reports are available for image scans.';
      document.querySelectorAll(".lang-tab").forEach((t) => t.classList.add("disabled"));
    } else {
      renderPatientReport("english");
      document.querySelectorAll(".lang-tab").forEach((t) => {
        t.classList.remove("disabled");
        t.classList.remove("active");
      });
      document.querySelector('.lang-tab[data-lang="english"]').classList.add("active");

      if (reportData.viewer_role === "doctor" && currentReport.doctor_report) {
        doctorCard.style.display = "block";
        renderDoctorReport();
        modeBanner.className = "viewer-mode-banner mode-doctor";
        modeBanner.style.display = "flex";
        modeBanner.innerHTML = '<i class="fa-solid fa-user-doctor"></i> Viewing as Doctor — full clinical report + simplified patient report below';
      } else if (reportData.viewer_role === "patient") {
        doctorCard.style.display = "none";
        modeBanner.className = "viewer-mode-banner mode-patient";
        modeBanner.style.display = "flex";
        modeBanner.innerHTML = '<i class="fa-solid fa-user"></i> Viewing as Patient — simplified report only. Clinical details are shared with your doctor.';
      } else {
        doctorCard.style.display = "none";
        modeBanner.style.display = "none";
      }
    }

    document.getElementById("previewWrap").style.display = "none";
    document.getElementById("uploadZone").style.display = "none";
    document.getElementById("resultsSection").style.display = "block";

    if (typeof onScanSaved === "function") onScanSaved();
    document.getElementById("resultsSection").scrollIntoView({ behavior: "smooth" });
  } catch (err) {
    showError(
      `Could not analyze the ${isVideo ? "video" : "image"}. ${err.message || "Please check that the Arterion backend is running and reachable, then try again."}`
    );
  } finally {
    stopLoadingAnimation();
  }
}

function fileToDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

// Backend endpoints in this project take a file_url (a reachable HTTP URL),
// not a raw upload. This helper posts the file to a lightweight local
// static-serving convention: it expects the backend or a sibling static
// server to expose an /uploads endpoint. If no such endpoint exists yet,
// this falls back to instructing the developer, since the browser cannot
// invent a public URL for a local file on its own.
async function uploadAndGetUrl(file) {
  const form = new FormData();
  form.append("file", file);

  try {
    const res = await fetch(`${API_BASE}/uploads`, { method: "POST", body: form });
    if (res.ok) {
      const data = await res.json();
      if (data.file_url) return data.file_url;
    }
  } catch (e) {
    // fall through to error below
  }

  throw new Error(
    "No file upload endpoint is configured on the backend yet. Add a POST /uploads route that saves the file and returns { file_url }, or serve uploaded files statically."
  );
}

document.addEventListener("DOMContentLoaded", () => {
  const loginPrompt = document.getElementById("loginPrompt");
  if (typeof isLoggedIn === "function" && !isLoggedIn()) {
    if (loginPrompt) loginPrompt.style.display = "block";
    document.getElementById("uploadZone").style.display = "none";
    return;
  }

  const uploadZone = document.getElementById("uploadZone");
  const fileInput = document.getElementById("fileInput");
  const previewWrap = document.getElementById("previewWrap");
  const previewImg = document.getElementById("previewImg");
  const analyzeBtn = document.getElementById("analyzeBtn");
  const resetBtn = document.getElementById("resetBtn");
  const scanAnotherBtn = document.getElementById("scanAnotherBtn");
  const downloadPdfBtn = document.getElementById("downloadPdfBtn");
  const doctorToggle = document.getElementById("doctorToggle");
  const doctorPanel = document.getElementById("doctorPanel");

  uploadZone.addEventListener("click", () => fileInput.click());

  uploadZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadZone.classList.add("dragover");
  });
  uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("dragover"));
  uploadZone.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadZone.classList.remove("dragover");
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length) handleFile(e.target.files[0]);
  });

  function handleFile(file) {
    const isImage = file.type.startsWith("image/");
    const isVideo = file.type.startsWith("video/");
    if (!isImage && !isVideo) {
      showError("Please upload an image (PNG/JPG) or a video (MP4) file.");
      return;
    }
    hideError();
    currentFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
      if (isVideo) {
        previewImg.style.display = "none";
        let previewVideo = document.getElementById("previewVideo");
        if (!previewVideo) {
          previewVideo = document.createElement("video");
          previewVideo.id = "previewVideo";
          previewVideo.controls = true;
          previewVideo.style.maxWidth = "100%";
          previewVideo.style.borderRadius = "var(--radius)";
          previewImg.insertAdjacentElement("afterend", previewVideo);
        }
        previewVideo.src = e.target.result;
        previewVideo.style.display = "block";
      } else {
        previewImg.style.display = "block";
        const previewVideo = document.getElementById("previewVideo");
        if (previewVideo) previewVideo.style.display = "none";
        previewImg.src = e.target.result;
      }
      uploadZone.style.display = "none";
      previewWrap.style.display = "block";
    };
    reader.readAsDataURL(file);
  }

  analyzeBtn.addEventListener("click", () => {
    if (currentFile) analyzeImage(currentFile);
  });

  resetBtn.addEventListener("click", () => {
    currentFile = null;
    fileInput.value = "";
    previewWrap.style.display = "none";
    uploadZone.style.display = "block";
    hideError();
  });

  scanAnotherBtn.addEventListener("click", () => {
    currentFile = null;
    currentReport = null;
    fileInput.value = "";
    document.getElementById("resultsSection").style.display = "none";
    uploadZone.style.display = "block";
    uploadZone.scrollIntoView({ behavior: "smooth" });
  });

  doctorToggle.addEventListener("click", () => {
    doctorToggle.classList.toggle("open");
    doctorPanel.classList.toggle("open");
  });

  downloadPdfBtn.addEventListener("click", () => {
    window.print();
  });

  const explainToggle = document.getElementById("explainToggle");
  const explainPanel = document.getElementById("explainPanel");
  explainToggle.addEventListener("click", () => {
    explainToggle.classList.toggle("open");
    explainPanel.classList.toggle("open");
  });

  document.querySelectorAll(".lang-tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".lang-tab").forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      renderPatientReport(tab.dataset.lang);
    });
  });
});