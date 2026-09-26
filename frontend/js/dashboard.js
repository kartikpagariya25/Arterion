const API_BASE = "http://127.0.0.1:8000";

let allScans = [];
let allPatients = [];

function severityClass(band) {
  const b = (band || "").toLowerCase();
  if (b === "severe") return "sev-severe";
  if (b === "moderate") return "sev-moderate";
  return "sev-mild";
}

function fmtDate(iso) {
  if (!iso) return "";
  const d = new Date(iso.includes("Z") || iso.includes("+") ? iso : iso + "Z");
  return d.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" }) +
    " · " + d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
}

function worstLesion(lesions) {
  if (!lesions || !lesions.length) return null;
  return lesions.reduce((a, b) => (a.stenosis_percent > b.stenosis_percent ? a : b));
}

/* ===================== SIDEBAR / NAVIGATION ===================== */

function setActivePanel(panelName) {
  document.querySelectorAll(".dash-nav-item").forEach((el) => el.classList.remove("active"));
  document.querySelector(`.dash-nav-item[data-panel="${panelName}"]`)?.classList.add("active");

  document.querySelectorAll(".dash-panel").forEach((el) => el.classList.remove("active"));
  document.getElementById(`panel-${panelName}`)?.classList.add("active");

  const titles = {
    overview: ["Overview", "Welcome back to Arterion."],
    newscan: ["New Scan", "Upload a coronary angiogram for AI analysis."],
    history: ["Scan History", "All your past scans, synced to your account."],
    secondary: [null, null],
    notifications: ["Notifications", "Updates from your recent scan activity."],
    settings: ["Settings", "Manage your account and password."],
  };

  const session = getSession();
  const isDoctor = session?.user?.role === "doctor";

  if (panelName === "secondary") {
    document.getElementById("pageTitle").textContent = isDoctor ? "My Patients" : "My Profile";
    document.getElementById("pageSubtitle").textContent = isDoctor
      ? "All registered patients and their latest scan status."
      : "Your personal and contact details.";
  } else if (titles[panelName]) {
    document.getElementById("pageTitle").textContent = titles[panelName][0];
    document.getElementById("pageSubtitle").textContent = titles[panelName][1];
  }

  if (panelName === "history") renderHistoryPanel();
  if (panelName === "secondary") isDoctor ? renderMyPatientsPanel() : renderProfilePanel();
  if (panelName === "notifications") renderNotificationsPanel();
  if (panelName === "settings") renderAccountInfo();
  if (panelName === "overview") renderOverviewPanel();

  document.getElementById("dashSidebar").classList.remove("open");
}

function initSidebarNav() {
  const session = getSession();
  const isDoctor = session?.user?.role === "doctor";

  document.getElementById("secondaryNavLabel").textContent = isDoctor ? "My Patients" : "Profile";
  document.querySelector('#secondaryNavItem i').className = isDoctor ? "fa-solid fa-user-group" : "fa-solid fa-id-badge";

  document.querySelectorAll(".dash-nav-item").forEach((item) => {
    item.addEventListener("click", () => setActivePanel(item.dataset.panel));
  });

  document.getElementById("dashLogout").addEventListener("click", logout);

  document.getElementById("mobileToggle")?.addEventListener("click", () => {
    document.getElementById("dashSidebar").classList.toggle("open");
  });
}

/* ===================== OVERVIEW ===================== */

function renderOverviewPanel() {
  const session = getSession();
  const isDoctor = session?.user?.role === "doctor";
  const statGrid = document.getElementById("overviewStats");
  const recentBox = document.getElementById("overviewRecent");

  if (isDoctor) {
    const totalPatients = allPatients.length;
    const totalScans = allPatients.reduce((sum, p) => sum + p.scan_count, 0);
    const severeCount = allPatients.filter((p) => p.latest_severity === "severe").length;

    statGrid.innerHTML = `
      <div class="stat-card"><div class="stat-icon blue"><i class="fa-solid fa-user-group"></i></div><div class="stat-value">${totalPatients}</div><div class="stat-label">Registered Patients</div></div>
      <div class="stat-card"><div class="stat-icon green"><i class="fa-solid fa-file-waveform"></i></div><div class="stat-value">${totalScans}</div><div class="stat-label">Total Scans Reviewed</div></div>
      <div class="stat-card"><div class="stat-icon red"><i class="fa-solid fa-triangle-exclamation"></i></div><div class="stat-value">${severeCount}</div><div class="stat-label">Patients Flagged Severe</div></div>
    `;
  } else {
    const totalScans = allScans.length;
    const worstOverall = allScans.reduce((worst, s) => {
      const w = worstLesion(s.lesions);
      if (!w) return worst;
      if (!worst || w.stenosis_percent > worst.stenosis_percent) return w;
      return worst;
    }, null);

    statGrid.innerHTML = `
      <div class="stat-card"><div class="stat-icon blue"><i class="fa-solid fa-file-waveform"></i></div><div class="stat-value">${totalScans}</div><div class="stat-label">Total Scans</div></div>
      <div class="stat-card"><div class="stat-icon ${worstOverall ? (worstOverall.severity_band === 'severe' ? 'red' : worstOverall.severity_band === 'moderate' ? 'amber' : 'green') : 'green'}">
        <i class="fa-solid fa-heart-pulse"></i></div>
        <div class="stat-value">${worstOverall ? worstOverall.severity_band.toUpperCase() : "—"}</div>
        <div class="stat-label">Most Recent Severity</div></div>
      <div class="stat-card"><div class="stat-icon amber"><i class="fa-solid fa-gauge"></i></div>
        <div class="stat-value">${worstOverall ? worstOverall.confidence + "%" : "—"}</div>
        <div class="stat-label">Model Confidence</div></div>
    `;
  }

  if (!allScans.length) {
    recentBox.innerHTML = `<div class="empty-state"><i class="fa-solid fa-inbox"></i>No scans yet. Head to New Scan to get started.</div>`;
    return;
  }

  recentBox.innerHTML = allScans.slice(0, 3).map((s) => {
    const worst = worstLesion(s.lesions);
    return `
      <div class="lesion-row">
        <span>${fmtDate(s.created_at)} &mdash; ${s.summary || "Scan analyzed"}</span>
        ${worst ? `<span class="severity-tag ${severityClass(worst.severity_band)}">${worst.severity_band}</span>` : ""}
      </div>`;
  }).join("");
}

/* ===================== HISTORY ===================== */

function renderHistoryPanel() {
  const grid = document.getElementById("historyGrid");
  if (!allScans.length) {
    grid.innerHTML = `<div class="empty-state" style="grid-column:1/-1;"><i class="fa-solid fa-clock-rotate-left"></i>No scans in your history yet.</div>`;
    return;
  }

  grid.innerHTML = allScans.map((s, i) => {
    const worst = worstLesion(s.lesions);
    return `
      <div class="history-card" data-idx="${i}">
        ${s.overlay_image_base64 ? `<img src="data:image/png;base64,${s.overlay_image_base64}" alt="Scan overlay">` : `<div style="height:150px; display:flex; align-items:center; justify-content:center; color:var(--pink-300);"><i class="fa-solid fa-image" style="font-size:2rem;"></i></div>`}
        <div class="history-card-body">
          <div class="history-date">${fmtDate(s.created_at)}</div>
          <div class="history-summary">${s.summary || "Scan analyzed"}</div>
          ${worst ? `<span class="severity-tag ${severityClass(worst.severity_band)}">${worst.severity_band}</span> <span class="confidence-tag">${worst.confidence}% confidence</span>` : ""}
        </div>
      </div>`;
  }).join("");

  grid.querySelectorAll(".history-card").forEach((card) => {
    card.addEventListener("click", () => openHistoryModal(allScans[parseInt(card.dataset.idx, 10)]));
  });
}

function openHistoryModal(scan) {
  const modal = document.getElementById("historyModal");
  const body = document.getElementById("historyModalBody");
  const report = scan.report;
  const patientBlock = report?.patient_report?.english;

  body.innerHTML = `
    <h3 style="font-size:1.2rem; font-weight:700; margin-bottom:6px;">${fmtDate(scan.created_at)}</h3>
    <p style="color:var(--ink-soft); font-size:.9rem; margin-bottom:18px;">${scan.summary || ""}</p>
    ${scan.overlay_image_base64 ? `<img src="data:image/png;base64,${scan.overlay_image_base64}" style="width:100%; border-radius:14px; margin-bottom:20px;">` : ""}
    ${patientBlock ? `
      <h4 style="font-size:.95rem; font-weight:700; color:var(--wine-800); margin-bottom:8px;">Patient Report</h4>
      <p style="font-size:.88rem; color:var(--ink-soft); margin-bottom:10px; line-height:1.6;">${patientBlock.summary || ""}</p>
      <p style="font-size:.88rem; color:var(--ink-soft); font-style:italic; margin-bottom:14px; line-height:1.6;">${patientBlock.what_it_means || ""}</p>
    ` : ""}
    ${report?.doctor_report ? `
      <h4 style="font-size:.95rem; font-weight:700; color:var(--wine-800); margin-bottom:8px;">Clinical Summary</h4>
      <p style="font-size:.88rem; color:var(--ink-soft); line-height:1.6;">${report.doctor_report.clinical_summary || ""}</p>
    ` : ""}
  `;
  modal.style.display = "flex";
}

document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("historyModalClose")?.addEventListener("click", () => {
    document.getElementById("historyModal").style.display = "none";
  });
  document.getElementById("historyModal")?.addEventListener("click", (e) => {
    if (e.target.id === "historyModal") e.target.style.display = "none";
  });
});

/* ===================== MY PATIENTS (doctor) ===================== */

function renderMyPatientsPanel() {
  const panel = document.getElementById("panel-secondary");

  if (!allPatients.length) {
    panel.innerHTML = `<div class="empty-state"><i class="fa-solid fa-user-group"></i>No patients registered yet.</div>`;
    return;
  }

  panel.innerHTML = `
    <table class="patients-table">
      <thead><tr><th>Name</th><th>Email</th><th>Age</th><th>Scans</th><th>Latest Severity</th><th>Last Scan</th></tr></thead>
      <tbody>
        ${allPatients.map((p) => `
          <tr>
            <td>${p.full_name}</td>
            <td>${p.email}</td>
            <td>${p.age ?? "—"}</td>
            <td>${p.scan_count}</td>
            <td>${p.latest_severity ? `<span class="severity-tag ${severityClass(p.latest_severity)}">${p.latest_severity}</span>` : "—"}</td>
            <td>${p.latest_scan_at ? fmtDate(p.latest_scan_at) : "—"}</td>
          </tr>`).join("")}
      </tbody>
    </table>
  `;
}

/* ===================== PROFILE (patient) ===================== */

function renderProfilePanel() {
  const panel = document.getElementById("panel-secondary");
  const session = getSession();
  const user = session?.user || {};

  panel.innerHTML = `
    <div class="form-card">
      <div class="form-row"><label>Full Name</label><input type="text" id="profileFullName" value="${user.full_name || ""}"></div>
      <div class="form-row"><label>Age</label><input type="number" id="profileAge" value="${user.age ?? ""}"></div>
      <div class="form-row"><label>Gender</label>
        <select id="profileGender">
          <option value="">Select</option>
          <option value="Male" ${user.gender === "Male" ? "selected" : ""}>Male</option>
          <option value="Female" ${user.gender === "Female" ? "selected" : ""}>Female</option>
          <option value="Other" ${user.gender === "Other" ? "selected" : ""}>Other</option>
        </select>
      </div>
      <div class="form-row"><label>Phone</label><input type="text" id="profilePhone" value="${user.phone || ""}"></div>
      <button class="btn btn-solid" id="saveProfileBtn">Save Changes</button>
      <div class="form-msg" id="profileMsg"></div>
    </div>
  `;

  document.getElementById("saveProfileBtn").addEventListener("click", async () => {
    const msg = document.getElementById("profileMsg");
    msg.className = "form-msg";
    try {
      const payload = {
        full_name: document.getElementById("profileFullName").value || undefined,
        age: document.getElementById("profileAge").value ? parseInt(document.getElementById("profileAge").value, 10) : undefined,
        gender: document.getElementById("profileGender").value || undefined,
        phone: document.getElementById("profilePhone").value || undefined,
      };
      const res = await fetch(`${API_BASE}/auth/me`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json", ...authHeader() },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error("Update failed");
      const updated = await res.json();
      saveSession(session.token, updated);
      msg.textContent = "Profile updated successfully.";
      msg.classList.add("success");
      document.getElementById("sidebarUserName").textContent = updated.full_name;
    } catch (e) {
      msg.textContent = "Could not update profile. Please try again.";
      msg.classList.add("error");
    }
  });
}

/* ===================== NOTIFICATIONS (derived from history) ===================== */

function renderNotificationsPanel() {
  const list = document.getElementById("notifList");
  if (!allScans.length) {
    list.innerHTML = `<div class="empty-state"><i class="fa-solid fa-bell-slash"></i>No notifications yet.</div>`;
    return;
  }

  list.innerHTML = allScans.slice(0, 10).map((s) => {
    const worst = worstLesion(s.lesions);
    const isSevere = worst?.severity_band === "severe";
    const icon = isSevere ? "fa-triangle-exclamation" : worst ? "fa-circle-info" : "fa-check";
    const iconClass = isSevere ? "red" : worst ? "amber" : "green";
    const text = worst
      ? `New scan analyzed &mdash; ${worst.severity_band} stenosis detected (${worst.confidence}% confidence)`
      : "New scan analyzed &mdash; no significant blockage found";

    return `
      <div class="notif-item">
        <div class="notif-icon stat-icon ${iconClass}"><i class="fa-solid ${icon}"></i></div>
        <div>
          <div class="notif-text">${text}</div>
          <div class="notif-time">${fmtDate(s.created_at)}</div>
        </div>
      </div>`;
  }).join("");
}

function updateNotifBadge() {
  const badge = document.getElementById("notifBadge");
  if (allScans.length > 0) {
    badge.style.display = "inline-block";
    badge.textContent = Math.min(allScans.length, 9);
  } else {
    badge.style.display = "none";
  }
}

/* ===================== SETTINGS ===================== */

function renderAccountInfo() {
  const session = getSession();
  const user = session?.user || {};
  const box = document.getElementById("accountInfoBox");
  box.innerHTML = `
    <div><strong>Name:</strong> ${user.full_name || "—"}</div>
    <div><strong>Email:</strong> ${user.email || "—"}</div>
    <div><strong>Role:</strong> ${user.role || "—"}</div>
  `;
}

document.addEventListener("DOMContentLoaded", () => {
  const passwordForm = document.getElementById("passwordForm");
  passwordForm?.addEventListener("submit", async (e) => {
    e.preventDefault();
    const msg = document.getElementById("passwordMsg");
    msg.className = "form-msg";
    try {
      const res = await fetch(`${API_BASE}/auth/change-password`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...authHeader() },
        body: JSON.stringify({
          current_password: document.getElementById("currentPassword").value,
          new_password: document.getElementById("newPassword").value,
        }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || "Could not change password");
      }
      msg.textContent = "Password updated successfully.";
      msg.classList.add("success");
      passwordForm.reset();
    } catch (err) {
      msg.textContent = err.message || "Could not change password.";
      msg.classList.add("error");
    }
  });
});

/* ===================== DATA LOADING ===================== */

async function loadScans() {
  try {
    const res = await fetch(`${API_BASE}/scans/mine`, { headers: authHeader() });
    if (res.ok) allScans = await res.json();
  } catch (e) {
    allScans = [];
  }
}

async function loadPatientsIfDoctor() {
  const session = getSession();
  if (session?.user?.role !== "doctor") return;
  try {
    const res = await fetch(`${API_BASE}/scans/all-patients`, { headers: authHeader() });
    if (res.ok) allPatients = await res.json();
  } catch (e) {
    allPatients = [];
  }
}

/* ===================== HOOK: called by scan.js after a scan completes ===================== */

async function onScanSaved() {
  await loadScans();
  updateNotifBadge();
}

/* ===================== INIT ===================== */

document.addEventListener("DOMContentLoaded", async () => {
  if (typeof isLoggedIn !== "function" || !isLoggedIn()) {
    window.location.href = "login.html";
    return;
  }

  const session = getSession();
  document.getElementById("sidebarUserName").textContent = session.user.full_name;
  document.getElementById("sidebarUserRole").textContent = session.user.role;

  initSidebarNav();

  await Promise.all([loadScans(), loadPatientsIfDoctor()]);

  updateNotifBadge();
  renderOverviewPanel();
});
