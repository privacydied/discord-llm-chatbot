/* Dashboard client-side JS — vanilla, no frameworks */
(function () {
  "use strict";

  const API = "";
  let csrfToken = "";
  let autoRefreshInterval = null;
  let currentPage = { overview: 1, guilds: 1, dms: 1, audit: 1 };
  let currentSection = "overview";

  /* ---- Navigation ---- */
  function showSection(name) {
    currentSection = name;
    document.querySelectorAll(".section").forEach(function (el) {
      el.classList.remove("active");
    });
    document.getElementById(name + "-section").classList.add("active");
    document.querySelectorAll("header nav button[data-section]").forEach(function (btn) {
      btn.classList.toggle("active", btn.dataset.section === name);
    });
    loadSection(name);
  }

  function loadSection(name) {
    switch (name) {
      case "overview": loadOverview(); break;
      case "guilds": loadGuilds(); break;
      case "dms": loadDMs(); break;
      case "audit": loadAudit(); break;
      case "send": break;
    }
  }

  /* ---- Auth ---- */
  async function login(token) {
    try {
      var res = await fetch(API + "/api/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ auth_token: token }),
      });
      var data = await res.json();
      if (data.success) {
        csrfToken = data.csrf_token;
        document.getElementById("login-page").style.display = "none";
        document.getElementById("dashboard-page").style.display = "block";
        showSection("overview");
      } else {
        var err = document.getElementById("login-error");
        err.textContent = "Invalid token";
        err.style.display = "block";
      }
    } catch (e) {
      document.getElementById("login-error").textContent = "Connection failed: " + e.message;
      document.getElementById("login-error").style.display = "block";
    }
  }

  async function logout() {
    await fetch(API + "/api/logout", { method: "POST" });
    document.getElementById("dashboard-page").style.display = "none";
    document.getElementById("login-page").style.display = "block";
    csrfToken = "";
    if (autoRefreshInterval) { clearInterval(autoRefreshInterval); autoRefreshInterval = null; }
  }

  /* ---- Fetch wrapper ---- */
  async function apiFetch(path) {
    var res = await fetch(API + path, {
      headers: { "X-CSRF-Token": csrfToken },
    });
    if (res.status === 401) { logout(); return null; }
    return res.json();
  }

  /* ---- Overview ---- */
  async function loadOverview() {
    var data = await apiFetch("/api/summary");
    if (!data) return;
    setText("summary-status", data.status);
    setText("summary-bot", (data.bot_username || "?") + " (" + (data.bot_id || "?") + ")");
    setText("summary-uptime", data.uptime_human || "0s");
    setText("summary-guilds", data.guild_count);
    setText("summary-users", data.total_users_estimate || 0);
    setText("summary-cogs", data.cog_count);
    setText("summary-latency", data.latency_ms + " ms");
    setText("summary-audit-count", data.audit_event_count || 0);
  }

  /* ---- Guilds ---- */
  async function loadGuilds(page) {
    page = page || currentPage.guilds;
    var data = await apiFetch("/api/guilds?page=" + page + "&page_size=50");
    if (!data) return;
    currentPage.guilds = data.page;
    var tbody = document.getElementById("guilds-tbody");
    tbody.innerHTML = "";
    data.guilds.forEach(function (g) {
      var tr = document.createElement("tr");
      tr.innerHTML = "<td>" + esc(g.id) + "</td><td>" + esc(g.name) + "</td><td>" + esc(g.member_count || "?") + "</td><td>" + esc(g.text_channel_count || "?") + "</td><td>" + esc(g.permissions) + "</td>";
      tbody.appendChild(tr);
    });
    setPagination("guilds-pagination", data.page, data.total_pages, loadGuilds);
  }

  /* ---- DMs ---- */
  async function loadDMs(page) {
    page = page || currentPage.dms;
    var data = await apiFetch("/api/dms?page=" + page + "&page_size=50");
    if (!data) return;
    currentPage.dms = data.page;
    var tbody = document.getElementById("dms-tbody");
    tbody.innerHTML = "";
    data.threads.forEach(function (t) {
      var tr = document.createElement("tr");
      var name = t.display_name || t.global_name || t.username || t.other_user_id || "?";
      var dir = t.last_direction === "outbound" ? "→" : "←";
      tr.innerHTML = "<td>" + esc(t.channel_id) + "</td><td>" + esc(name) + "</td><td>" + esc(t.message_count) + "</td><td>" + dir + " " + esc(t.last_preview || "") + "</td><td>" + esc(t.last_message_at || "") + "</td><td><button onclick=\"viewDMThread('" + esc(t.channel_id) + "')\">view</button></td>";
      tbody.appendChild(tr);
    });
    setPagination("dms-pagination", data.page, data.total_pages, loadDMs);
  }

  async function viewDMThread(channelId) {
    var data = await apiFetch("/api/dms/" + channelId + "?page=1&page_size=50");
    if (!data) return;
    var tbody = document.getElementById("dm-thread-tbody");
    tbody.innerHTML = "";
    document.getElementById("dm-thread-section").style.display = "block";
    data.messages.forEach(function (m) {
      var tr = document.createElement("tr");
      var dir = m.direction === "outbound" ? "→ (bot)" : "← (user)";
      tr.innerHTML = "<td>" + dir + "</td><td>" + esc(m.content_preview || "") + "</td><td>" + esc(m.created_at) + "</td>";
      tbody.appendChild(tr);
    });
  }

  /* ---- Audit ---- */
  async function loadAudit(page) {
    page = page || currentPage.audit;
    var filters = getAuditFilters();
    var qs = "page=" + page + "&page_size=50";
    if (filters.event_type) qs += "&event_type=" + encodeURIComponent(filters.event_type);
    if (filters.actor_user_id) qs += "&actor_user_id=" + filters.actor_user_id;
    if (filters.result) qs += "&result=" + encodeURIComponent(filters.result);

    var data = await apiFetch("/api/audit?" + qs);
    if (!data) return;
    currentPage.audit = data.page;
    var tbody = document.getElementById("audit-tbody");
    tbody.innerHTML = "";
    data.events.forEach(function (e) {
      var tr = document.createElement("tr");
      var badgeClass = e.result === "success" ? "success" : e.result === "pending" ? "pending" : "failed";
      tr.innerHTML = "<td><span class='badge " + badgeClass + "'>" + esc(e.result) + "</span></td><td>" + esc(e.event_type) + "</td><td>" + esc(e.actor_user_id || "-") + "</td><td>" + esc(e.target_user_id || e.target_guild_id || "-") + "</td><td>" + esc(e.content_preview || "") + "</td><td>" + esc(e.created_at) + "</td>";
      tbody.appendChild(tr);
    });
    setPagination("audit-pagination", data.page, data.total_pages, loadAudit);
  }

  function getAuditFilters() {
    return {
      event_type: document.getElementById("audit-filter-event") ? document.getElementById("audit-filter-event").value : "",
      actor_user_id: document.getElementById("audit-filter-actor") ? document.getElementById("audit-filter-actor").value : "",
      result: document.getElementById("audit-filter-result") ? document.getElementById("audit-filter-result").value : "",
    };
  }

  /* ---- Send DM ---- */
  async function sendDM(userId, content) {
    var resultEl = document.getElementById("send-dm-result");
    resultEl.style.display = "none";
    try {
      var res = await fetch(API + "/api/dms/" + userId + "/send", {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-CSRF-Token": csrfToken },
        body: JSON.stringify({ content: content }),
      });
      var data = await res.json();
      resultEl.style.display = "block";
      if (data.success) {
        resultEl.className = "success";
        resultEl.textContent = "Sent! Message ID: " + data.message_id;
      } else {
        resultEl.className = "error";
        resultEl.textContent = "Failed: " + (data.error || "unknown");
      }
    } catch (e) {
      resultEl.style.display = "block";
      resultEl.className = "error";
      resultEl.textContent = "Request failed: " + e.message;
    }
  }

  /* ---- Send guild message ---- */
  async function sendGuildMessage(guildId, channelId, content) {
    var resultEl = document.getElementById("send-guild-result");
    resultEl.style.display = "none";
    try {
      var res = await fetch(API + "/api/guilds/" + guildId + "/channels/" + channelId + "/send", {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-CSRF-Token": csrfToken },
        body: JSON.stringify({ content: content }),
      });
      var data = await res.json();
      resultEl.style.display = "block";
      if (data.success) {
        resultEl.className = "success";
        resultEl.textContent = "Sent! Message ID: " + data.message_id;
      } else {
        resultEl.className = "error";
        resultEl.textContent = "Failed: " + (data.error || "unknown");
      }
    } catch (e) {
      resultEl.style.display = "block";
      resultEl.className = "error";
      resultEl.textContent = "Request failed: " + e.message;
    }
  }

  /* ---- Helpers ---- */
  function setText(id, value) {
    var el = document.getElementById(id);
    if (el) el.textContent = value;
  }

  function esc(str) {
    if (!str) return "";
    var div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  function setPagination(containerId, currentPage, totalPages, loadFn) {
    var container = document.getElementById(containerId);
    if (!container) return;
    container.innerHTML = "";
    var prev = document.createElement("button");
    prev.textContent = "prev";
    prev.disabled = currentPage <= 1;
    prev.onclick = function () { loadFn(currentPage - 1); };
    container.appendChild(prev);

    var info = document.createElement("span");
    info.textContent = " " + currentPage + " / " + totalPages + " ";
    container.appendChild(info);

    var next = document.createElement("button");
    next.textContent = "next";
    next.disabled = currentPage >= totalPages;
    next.onclick = function () { loadFn(currentPage + 1); };
    container.appendChild(next);
  }

  function toggleAutoRefresh() {
    var checked = document.getElementById("auto-refresh-toggle").checked;
    if (checked) {
      autoRefreshInterval = setInterval(function () { loadSection(currentSection); }, 10000);
    } else {
      if (autoRefreshInterval) { clearInterval(autoRefreshInterval); autoRefreshInterval = null; }
    }
  }

  /* ---- Event listeners ---- */
  document.addEventListener("DOMContentLoaded", function () {
    document.getElementById("login-form").addEventListener("submit", function (e) {
      e.preventDefault();
      var token = document.getElementById("auth-token-input").value;
      if (token) login(token);
    });

    document.querySelectorAll("header nav button[data-section]").forEach(function (btn) {
      btn.addEventListener("click", function () { showSection(btn.dataset.section); });
    });

    document.getElementById("logout-btn").addEventListener("click", logout);
    document.getElementById("auto-refresh-toggle").addEventListener("change", toggleAutoRefresh);

    document.getElementById("audit-filter-btn").addEventListener("click", function () { loadAudit(1); });

    document.getElementById("send-dm-btn").addEventListener("click", function () {
      var userId = document.getElementById("send-dm-user-id").value.trim();
      var content = document.getElementById("send-dm-content").value.trim();
      if (userId && content) sendDM(userId, content);
    });

    document.getElementById("send-guild-btn").addEventListener("click", function () {
      var guildId = document.getElementById("send-guild-id").value.trim();
      var channelId = document.getElementById("send-guild-channel-id").value.trim();
      var content = document.getElementById("send-guild-content").value.trim();
      if (guildId && channelId && content) sendGuildMessage(guildId, channelId, content);
    });
  });

  /* Expose global functions for onclick handlers */
  window.viewDMThread = viewDMThread;
})();
