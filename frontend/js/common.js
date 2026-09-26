// API base — change this if your backend runs on a different host/port
const API_BASE = "http://127.0.0.1:8000";

const NAV_ITEMS = [
  { href: "index.html", label: "Home" },
  { href: "scan.html", label: "Scan" },
  { href: "health-suggestions.html", label: "Health Tips" },
  { href: "cpr-survival.html", label: "CPR & Survival" },
  { href: "emergency-contacts.html", label: "Emergency" },
  { href: "get-in-touch.html", label: "Get In Touch" },
  { href: "disclaimers.html", label: "Disclaimers" },
  { href: "about-developers.html", label: "About Us" },
];

function currentPage() {
  const path = window.location.pathname.split("/").pop();
  return path === "" ? "index.html" : path;
}

function renderNav(mountId) {
  const mount = document.getElementById(mountId);
  if (!mount) return;
  const active = currentPage();
  const links = NAV_ITEMS.map(
    (item) =>
      `<li><a href="${item.href}" class="${item.href === active ? "active" : ""}">${item.label}</a></li>`
  ).join("");

  mount.innerHTML = `
    <div class="hero-nav">
      <a href="index.html" class="hero-logo">
        <span class="hero-logo-icon"><i class="fa-solid fa-heart-pulse"></i></span>
        <span class="hero-logo-name">Arterion</span>
      </a>
      <ul class="hero-nav-links" id="navLinks">${links}</ul>
      <div class="hero-nav-auth" id="authNavMount">
        <button class="hero-menu-toggle" id="menuToggle"><i class="fa-solid fa-bars"></i></button>
      </div>
    </div>
  `;

  if (typeof renderAuthNav === "function") {
    const authWrap = document.createElement("span");
    authWrap.id = "authNavInner";
    document.getElementById("authNavMount").prepend(authWrap);
    renderAuthNav("authNavInner");
  }

  const menuToggle = document.getElementById("menuToggle");
  const navLinks = document.getElementById("navLinks");
  if (menuToggle && navLinks) {
    menuToggle.addEventListener("click", () => {
      navLinks.classList.toggle("open");
      menuToggle.innerHTML = navLinks.classList.contains("open")
        ? '<i class="fa-solid fa-xmark"></i>'
        : '<i class="fa-solid fa-bars"></i>';
    });
    navLinks.querySelectorAll("a").forEach((a) =>
      a.addEventListener("click", () => {
        navLinks.classList.remove("open");
        menuToggle.innerHTML = '<i class="fa-solid fa-bars"></i>';
      })
    );
  }
}

function renderFooter(mountId) {
  const mount = document.getElementById(mountId);
  if (!mount) return;
  mount.innerHTML = `
    <div class="container">
      <div class="footer-grid">
        <div class="footer-brand-block">
          <a href="index.html" class="footer-brand"><i class="fa-solid fa-heart-pulse"></i> Arterion</a>
          <p>AI-assisted coronary vessel analysis &mdash; built to help clinicians and patients understand angiogram results faster and clearer. Not a replacement for professional medical judgment.</p>
        </div>
        <div class="footer-col">
          <h4>Product</h4>
          <ul>
            <li><a href="scan.html">Scan &amp; Analyze</a></li>
            <li><a href="health-suggestions.html">Health Tips</a></li>
            <li><a href="cpr-survival.html">CPR &amp; Survival</a></li>
          </ul>
        </div>
        <div class="footer-col">
          <h4>Support</h4>
          <ul>
            <li><a href="emergency-contacts.html">Emergency Contacts</a></li>
            <li><a href="get-in-touch.html">Get In Touch</a></li>
          </ul>
        </div>
        <div class="footer-col">
          <h4>Legal</h4>
          <ul>
            <li><a href="disclaimers.html">Disclaimers</a></li>
            <li><a href="about-developers.html">About the Team</a></li>
          </ul>
        </div>
      </div>
      <div class="footer-bottom">
        <span>&copy; 2026 Arterion. Built for Synapse Hackathon &mdash; Symbiosis Pune.</span>
        <span class="font-display" style="font-style:italic;">AI That Understands the Heart.</span>
      </div>
    </div>
  `;
}

function renderEmergencyBanner(mountId) {
  const mount = document.getElementById(mountId);
  if (!mount) return;
  mount.innerHTML = `
    <a href="tel:108" class="emergency-banner">
      <i class="fa-solid fa-phone-volume"></i> Emergency? Call 108
    </a>
  `;
}

function initRevealAnimations() {
  if (typeof gsap === "undefined") return;
  gsap.registerPlugin(ScrollTrigger);

  gsap.to(".top-card .reveal", { opacity: 1, y: 0, duration: 1, ease: "power3.out", stagger: 0.1, delay: 0.15 });

  gsap.utils.toArray(".section-pad .reveal").forEach((el) => {
    gsap.to(el, { opacity: 1, y: 0, duration: 0.85, ease: "power3.out", scrollTrigger: { trigger: el, start: "top 88%" } });
  });

  gsap.utils.toArray(".card-hover.reveal").forEach((el, i) => {
    gsap.fromTo(
      el,
      { opacity: 0, y: 24 },
      { opacity: 1, y: 0, duration: 0.6, ease: "power2.out", delay: (i % 4) * 0.07, scrollTrigger: { trigger: el, start: "top 92%" } }
    );
  });
}

document.addEventListener("DOMContentLoaded", () => {
  renderNav("nav-mount");
  renderFooter("footer-mount");
  renderEmergencyBanner("emergency-mount");
  initRevealAnimations();
});
