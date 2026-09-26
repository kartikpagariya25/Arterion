const SESSION_KEY = "Arterion_session";

function saveSession(token, user) {
  localStorage.setItem(SESSION_KEY, JSON.stringify({ token, user }));
}

function getSession() {
  try {
    const raw = localStorage.getItem(SESSION_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch (e) {
    return null;
  }
}

function clearSession() {
  localStorage.removeItem(SESSION_KEY);
}

function isLoggedIn() {
  return !!getSession()?.token;
}

function authHeader() {
  const session = getSession();
  return session?.token ? { Authorization: `Bearer ${session.token}` } : {};
}

function logout() {
  clearSession();
  window.location.href = "login.html";
}

function renderAuthNav(mountId) {
  const mount = document.getElementById(mountId);
  if (!mount) return;
  const session = getSession();

  if (session?.user) {
    mount.innerHTML = `
      <span class="auth-nav-name"><i class="fa-solid fa-circle-user"></i> ${session.user.full_name} <span class="auth-role-tag">${session.user.role}</span></span>
      <button class="auth-nav-btn" id="logoutBtn"><i class="fa-solid fa-right-from-bracket"></i></button>
    `;
    const logoutBtn = document.getElementById("logoutBtn");
    if (logoutBtn) logoutBtn.addEventListener("click", logout);
  } else {
    mount.innerHTML = `<a href="login.html" class="auth-nav-btn"><i class="fa-solid fa-right-to-bracket"></i> Log In</a>`;
  }
}
