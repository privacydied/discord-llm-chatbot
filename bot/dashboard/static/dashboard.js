/* Dashboard — Discord-like chat console for bot ops */
(function () {
  "use strict";

  /* ============================================================
     STATE
     ============================================================ */
  const API = "";
  let csrfToken = "";
  let config = {};

  let state = {
    authenticated: false,
    view: "overview",
    guildId: null,
    channelId: null,
    channelName: "",
    channelType: "guild",
    isNewDm: false,       // true when DMing an arbitrary user ID
    messagesPage: 1,
    messagesTotalPages: 1,
    hasMoreOlder: false,
    replyTo: null,
    guilds: [],
    rateLimited: false,
    messages: [],       // cached messages for current channel/dm
    canSend: true,
    sendDisabledReason: "",
    botName: "",
    loadingMessages: false,
  };

  /* ============================================================
     HELPERS
     ============================================================ */
  const $ = (id) => document.getElementById(id);

  function esc(str) {
    if (!str && str !== 0) return "";
    const d = document.createElement("div");
    d.textContent = String(str);
    return d.innerHTML;
  }

  function qs(sel) {
    return document.querySelector(sel);
  }

  function qsa(sel) {
    return document.querySelectorAll(sel);
  }

  function setText(id, val) {
    const el = $(id);
    if (el) el.textContent = val ?? "—";
  }

  function sleep(ms) {
    return new Promise((r) => setTimeout(r, ms));
  }

  function ago(isoStr) {
    if (!isoStr) return "";
    const t = new Date(isoStr).getTime();
    const now = Date.now();
    const diff = now - t;
    if (diff < 60000) return "just now";
    if (diff < 3600000) return Math.floor(diff / 60000) + "m ago";
    if (diff < 86400000) return Math.floor(diff / 3600000) + "h ago";
    const d = new Date(isoStr);
    const opts = { month: "short", day: "numeric", year: d.getFullYear() !== new Date().getFullYear() ? "numeric" : undefined };
    return d.toLocaleDateString("en-US", opts);
  }

  function formatTime(isoStr) {
    if (!isoStr) return "";
    const d = new Date(isoStr);
    return d.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" });
  }

  function formatDate(isoStr) {
    if (!isoStr) return "";
    const d = new Date(isoStr);
    const today = new Date();
    const yesterday = new Date(today);
    yesterday.setDate(yesterday.getDate() - 1);
    if (d.toDateString() === today.toDateString()) return "Today";
    if (d.toDateString() === yesterday.toDateString()) return "Yesterday";
    return d.toLocaleDateString("en-US", { weekday: "long", month: "long", day: "numeric", year: d.getFullYear() !== today.getFullYear() ? "numeric" : undefined });
  }

  function colorFromId(id) {
    const colors = [
      "#5865F2", "#ED4245", "#FEE75C", "#57F287", "#EB459E",
      "#FF73FA", "#00A8FC", "#95E5D7", "#9B59B6", "#1ABC9C",
    ];
    return colors[Number(id) % colors.length];
  }

  function createElement(tag, attrs, children) {
    const el = document.createElement(tag);
    if (attrs) {
      for (const [k, v] of Object.entries(attrs)) {
        if (k === "className") el.className = v;
        else if (k === "textContent") el.textContent = v;
        else el.setAttribute(k, v);
      }
    }
    if (children) {
      if (typeof children === "string") el.innerHTML = children;
      else if (Array.isArray(children)) children.forEach((c) => el.appendChild(c));
    }
    return el;
  }

  function getMessageId(msg) {
    return msg.id || msg.message_id || msg.discord_message_id || "";
  }

  /* ============================================================
     TOAST
     ============================================================ */
  function toast(msg, type) {
    type = type || "info";
    const container = $("toast-container");
    const el = createElement("div", { className: "toast " + type, textContent: msg });
    container.appendChild(el);
    setTimeout(() => { el.classList.add("removing"); }, 3000);
    setTimeout(() => el.remove(), 3200);
  }

  /* ============================================================
     FETCH
     ============================================================ */
  async function apiFetch(path, opts) {
    opts = opts || {};
    const headers = opts.headers || {};
    headers["X-CSRF-Token"] = csrfToken;
    // POST requests (backfill, send) get 120s timeout; GETs get 30s
    const isPost = opts.method === "POST";
    const timeoutMs = isPost ? 120000 : 30000;
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const res = await fetch(API + path, { ...opts, headers, signal: controller.signal });
      clearTimeout(timeout);
      if (res.status === 401) { handleLogout(); return null; }
      if (res.status === 429) {
        state.rateLimited = true;
        toast("Rate limited — slow down", "warning");
        return null;
      }
      try {
        return await res.json();
      } catch {
        return null;
      }
    } catch (e) {
      clearTimeout(timeout);
      if (e.name === "AbortError") {
        toast("Request timed out", "error");
      } else {
        toast("Request failed: " + e.message, "error");
      }
      return null;
    }
  }

  /* ============================================================
     AUTH
     ============================================================ */
  async function checkSession() {
    const data = await apiFetch("/api/session", { headers: {} });
    if (data && data.authenticated) {
      csrfToken = data.csrf_token || "";
      state.authenticated = true;
      $("login-page").classList.add("hidden");
      $("dashboard-page").classList.remove("hidden");
      await loadConfig();
      await loadGuildRail();
      // Restore hash or go home
      if (location.hash) {
        handleHash();
      } else {
        showView("overview");
      }
      return true;
    }
    return false;
  }

  async function handleLogin(token) {
    const data = await apiFetch("/api/login", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-CSRF-Token": "" },
      body: JSON.stringify({ auth_token: token }),
    });
    if (data && data.success) {
      csrfToken = data.csrf_token || "";
      state.authenticated = true;
      $("login-page").classList.add("hidden");
      $("dashboard-page").classList.remove("hidden");
      await loadConfig();
      await loadGuildRail();
      if (location.hash) { handleHash(); }
      else { showView("overview"); }
      toast("Logged in", "success");
    } else {
      $("login-error").textContent = "Invalid token";
      $("login-error").classList.remove("hidden");
    }
  }

  async function handleLogout() {
    await fetch(API + "/api/logout", { method: "POST" });
    state.authenticated = false;
    csrfToken = "";
    $("dashboard-page").classList.add("hidden");
    $("login-page").classList.remove("hidden");
    $("login-error").classList.add("hidden");
  }

  /* ============================================================
     CONFIG
     ============================================================ */
  async function loadConfig() {
    const data = await apiFetch("/api/static-config");
    if (data) {
      config = data;
      state.botName = data.bot_username || "Bot";
    }
  }

  /* ============================================================
     URL HASH ROUTING
     ============================================================ */
  function updateHash(view, guildId, channelId, channelType) {
    let hash = "";
    switch (view) {
      case "guild":
        if (guildId && channelId) hash = `#/guild/${guildId}/channel/${channelId}`;
        else if (guildId) hash = `#/guild/${guildId}`;
        break;
      case "dm":
        if (channelId) hash = `#/dm/${channelId}`;
        break;
      case "audit": hash = "#/audit"; break;
      case "backfill": hash = "#/backfill"; break;
      default: hash = "#/overview"; break;
    }
    if (location.hash !== hash) {
      history.replaceState(null, "", hash);
    }
  }

  function handleHash() {
    const hash = location.hash || "#/overview";
    const matchGuild = hash.match(/^#\/guild\/(\d+)(?:\/channel\/(\d+))?$/);
    const matchDm = hash.match(/^#\/dm\/(\d+)$/);

    if (matchGuild) {
      const guildId = matchGuild[1];
      const channelId = matchGuild[2];
      state.guildId = guildId;
      if (channelId) {
        loadChannelMessages(channelId, "", "guild");
      } else {
        showView("guild", guildId);
      }
    } else if (matchDm) {
      const channelId = matchDm[1];
      loadChannelMessages(channelId, "", "dm");
    } else if (hash === "#/new-dm") {
      showView("overview");
      // Focus the new DM input after sidebar renders
      setTimeout(() => {
        const input = $("new-dm-input");
        if (input) input.focus();
      }, 200);
    } else if (hash === "#/audit") {
      showView("audit");
    } else if (hash === "#/backfill") {
      showView("backfill");
    } else {
      showView("overview");
    }
  }

  /* ============================================================
     VIEW SWITCHING
     ============================================================ */
  function showView(view, guildId) {
    state.view = view;
    state.guildId = guildId || null;
    state.channelId = null;
    state.channelName = "";
    state.messagesPage = 1;
    state.messagesTotalPages = 1;
    state.hasMoreOlder = false;
    state.messages = [];
    state.replyTo = null;
    state.canSend = true;
    state.sendDisabledReason = "";
    hideComposer();
    renderMessageList([]);
    clearDetails();
    updateHash(view, guildId);

    qsa(".guild-item").forEach((el) => el.classList.remove("active"));
    if (view === "overview") qs(".guild-home")?.classList.add("active");
    else if (view === "backfill") $("nav-backfill")?.classList.add("active");
    else if (view === "audit") $("nav-audit")?.classList.add("active");

    $("messages-empty").classList.add("hidden");
    $("messages-end").classList.add("hidden");
    $("load-older-btn").classList.add("hidden");
    $("backfill-prompt").classList.add("hidden");
    $("messages-error").classList.add("hidden");
    $("messages-loading").classList.add("hidden");
    $("composer-status").textContent = "";
    updateComposerPlaceholder();

    switch (view) {
      case "overview":
        $("sidebar-title").textContent = "Direct Messages";
        $("sidebar-search").classList.add("hidden");
        renderHomeSidebar();
        renderHomeContent();
        break;
      case "guild":
        $("sidebar-title").textContent = state.channelName || "Select a channel";
        $("sidebar-search").classList.remove("hidden");
        loadSidebarChannels(guildId);
        break;
      case "dm":
        $("sidebar-title").textContent = "Direct Messages";
        $("sidebar-search").classList.add("hidden");
        renderHomeSidebar();
        break;
      case "backfill":
        $("sidebar-title").textContent = "Backfill";
        $("sidebar-search").classList.add("hidden");
        renderBackfillSidebar();
        renderBackfillContent();
        break;
      case "audit":
        $("sidebar-title").textContent = "Audit Log";
        $("sidebar-search").classList.add("hidden");
        renderAuditSidebar();
        renderAuditContent(1);
        break;
    }
  }

  /* ============================================================
     GUILD RAIL
     ============================================================ */
  async function loadGuildRail() {
    const data = await apiFetch("/api/guilds?page=1&page_size=200");
    if (!data || !data.guilds) return;
    state.guilds = data.guilds;
    const list = $("guild-list");
    list.innerHTML = "";
    data.guilds.forEach((g) => {
      const item = createElement("div", {
        className: "guild-item",
        title: g.name,
        "data-guild-id": g.id,
      });
      if (g.icon_url) {
        item.appendChild(createElement("img", { src: g.icon_url, alt: g.name }));
      } else {
        const acronym = g.name.split(/\s+/).slice(0, 2).map((w) => w[0]).join("").toUpperCase();
        item.appendChild(createElement("div", {
          className: "guild-acronym",
          textContent: acronym || "?",
          style: "background:" + colorFromId(g.id),
        }));
      }
      item.addEventListener("click", () => {
        state.messages = [];
        updateHash("guild", g.id);
        showView("guild", g.id);
      });
      list.appendChild(item);
    });
  }

  /* ============================================================
     SIDEBAR — HOME / DM THREADS
     ============================================================ */
  let dmThreadsCache = [];

  async function renderHomeSidebar() {
    const content = $("sidebar-content");
    content.innerHTML = '<div class="sidebar-loading"><div class="spinner"></div></div>';
    const data = await apiFetch("/api/dms?page=1&page_size=100");
    content.innerHTML = "";

    // "New DM" input at top
    const newDmRow = createElement("div", { className: "sidebar-item new-dm-item", style: "padding:8px;border-bottom:1px solid var(--bg-hover)" });
    newDmRow.innerHTML = `
      <span class="hash" style="font-size:14px">✉️</span>
      <input type="text" id="new-dm-input" placeholder="User ID to DM..." style="flex:1;border:none;background:transparent;color:var(--text-normal);outline:none;font-size:12px">
    `;
    newDmRow.querySelector("#new-dm-input").addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        const uid = e.target.value.trim();
        if (uid && /^\d{17,20}$/.test(uid)) {
          state.messages = [];
          state.isNewDm = true;
          loadChannelMessages(uid, "User " + uid, "dm");
        } else {
          toast("Enter a valid Discord user ID (17-20 digits)", "warning");
        }
      }
    });
    newDmRow.querySelector("#new-dm-input").addEventListener("focus", () => {
      newDmRow.classList.add("active");
    });
    newDmRow.querySelector("#new-dm-input").addEventListener("blur", () => {
      newDmRow.classList.remove("active");
    });
    content.appendChild(newDmRow);

    if (!data || !data.threads || data.threads.length === 0) {
      content.innerHTML += '<div class="empty-state" style="padding:16px"><p class="text-muted small">No DM conversations yet</p></div>';
      dmThreadsCache = [];
      content.prepend(newDmRow);
      return;
    }
    dmThreadsCache = data.threads;
    data.threads.forEach((t) => {
      const el = createElement("div", {
        className: "sidebar-item" + (state.channelId === t.dm_channel_id && state.channelType === "dm" && !state.isNewDm ? " active" : ""),
        "data-channel-id": t.dm_channel_id,
        "data-channel-type": "dm",
      });
      let avatarHtml;
      if (t.avatar_url) {
        avatarHtml = `<img src="${esc(t.avatar_url)}" class="icon" alt="">`;
      } else {
        const initial = (t.display_name || t.username || "?").charAt(0).toUpperCase();
        avatarHtml = `<div class="icon avatar-fallback" style="background:${colorFromId(t.user_id)};display:flex;align-items:center;justify-content:center;color:#fff;font-size:10px;font-weight:700">${esc(initial)}</div>`;
      }
      const lastMsg = t.last_message_at ? ago(t.last_message_at) : "";
      el.innerHTML = `${avatarHtml}<span class="name">${esc(t.display_name || t.username || "Unknown")}</span><span class="last-msg">${esc(lastMsg)}</span>`;
      el.addEventListener("click", () => {
        state.messages = [];
        state.isNewDm = false;
        loadChannelMessages(t.dm_channel_id, t.display_name || t.username || "DM", "dm");
      });
      content.appendChild(el);
    });
  }

  /* ============================================================
     SIDEBAR — CHANNELS
     ============================================================ */
  async function loadSidebarChannels(guildId) {
    const data = await apiFetch("/api/guilds/" + guildId);
    if (!data) return;
    state.channelName = data.name || "Unknown Guild";
    $("sidebar-title").textContent = state.channelName;
    setText("content-name", data.name);
    setText("content-meta", (data.member_count || "?") + " members");

    const avatar = $("content-avatar");
    if (data.icon_url) {
      avatar.innerHTML = `<img src="${esc(data.icon_url)}" alt="">`;
      avatar.classList.remove("hidden");
    } else {
      const acronym = (data.name || "?").split(/\s+/).slice(0, 2).map((w) => w[0]).join("").toUpperCase();
      avatar.innerHTML = `<div class="avatar-fallback" style="background:${colorFromId(Number(guildId))}">${esc(acronym)}</div>`;
      avatar.classList.remove("hidden");
    }

    const content = $("sidebar-content");
    content.innerHTML = "";
    const categories = data.categories || [];
    const uncategorized = data.uncategorized_channels || [];

    const addChannel = (ch) => {
      const isText = ch.type === "0" || ch.type === "text" || ch.type === 0;
      if (!isText) return;
      const el = createElement("div", {
        className: "sidebar-item" + (ch.id == state.channelId ? " active" : ""),
        "data-channel-id": ch.id,
        "data-channel-type": "guild",
      });
      let extra = "";
      const canRead = ch.can_read !== false;
      const canSend = ch.can_send === true;
      if (!canRead) {
        extra = ' <span class="badge" style="background:var(--text-muted)">no view</span>';
      } else if (!canSend) {
        extra = ' <span class="badge" style="background:var(--text-warning);color:#000">no send</span>';
      }
      el.innerHTML = `<span class="hash">#</span><span class="name">${esc(ch.name || ch.id)}</span>${extra}`;
      if (canRead) {
        el.addEventListener("click", () => {
          state.messages = [];
          loadChannelMessages(ch.id, ch.name || ch.id, "guild");
        });
      } else {
        el.classList.add("disabled");
      }
      content.appendChild(el);
    };

    categories.forEach((cat) => {
      if (cat.name) {
        content.appendChild(createElement("div", {
          className: "sidebar-category",
          textContent: cat.name.toUpperCase(),
        }));
      }
      (cat.channels || []).forEach(addChannel);
    });

    if (uncategorized.length > 0) {
      if (categories.length > 0) {
        content.appendChild(createElement("div", {
          className: "sidebar-category",
          textContent: "TEXT CHANNELS",
        }));
      }
      uncategorized.forEach(addChannel);
    }

    if (content.children.length === 0) {
      content.innerHTML = '<div class="empty-state" style="padding:16px"><p class="text-muted small">No accessible text channels</p></div>';
    }
  }

  /* ============================================================
     CHANNEL / DM MESSAGES
     ============================================================ */
  async function loadChannelMessages(channelId, channelName, type) {
    state.loadingMessages = true;
    state.channelId = channelId;
    state.channelName = channelName;
    state.channelType = type || "guild";
    state.messagesPage = 1;
    state.messagesTotalPages = 1;
    state.hasMoreOlder = false;
    state.messages = [];
    state.replyTo = null;
    hideReply();
    hideComposer();

    qsa(".sidebar-item").forEach((el) => el.classList.remove("active"));
    const activeItem = qs(`.sidebar-item[data-channel-id="${channelId}"]`);
    if (activeItem) activeItem.classList.add("active");

    setText("content-name", type === "dm" ? (channelName || "DM") : "#" + (channelName || "Channel"));
    setText("content-meta", type === "dm" ? "Direct Message" : "");
    $("content-avatar").classList.add("hidden");
    $("content-status").classList.add("hidden");

    // Show loading skeleton
    renderMessageList([]);
    $("messages-error").classList.add("hidden");
    $("composer-status").textContent = "";

    const isDm = type === "dm";
    const path = isDm
      ? `/api/dms/${channelId}/messages?page=1&page_size=50&auto_backfill=true`
      : `/api/channels/${channelId}/messages?page=1&page_size=50&auto_backfill=true`;

    const data = await apiFetch(path);
    state.loadingMessages = false;

    if (!data) {
      $("messages-error").classList.remove("hidden");
      $("messages-error").querySelector("p").textContent = "Failed to load messages";
      $("messages-list").innerHTML = "";
      return;
    }

    const msgs = data.messages || data.events || [];
    // API returns newest-first (ORDER BY created_at DESC). Reverse to oldest-first
    // so the DOM renders chronologically and scrollToBottom shows the newest messages.
    state.messages = [...msgs].reverse();
    state.messagesPage = data.page || 1;
    state.messagesTotalPages = data.total_pages || 1;
    state.hasMoreOlder = data.page < data.total_pages;

    renderMessageList(state.messages);
    updateHash(state.view, state.guildId, channelId, type);
    checkSendPermission();
    scheduleRefresh();

    // Update guild rail active
    qsa(".guild-item").forEach((el) => el.classList.remove("active"));
    if (isDm) {
      qs(".guild-home")?.classList.add("active");
    } else {
      const guildItem = qs(`.guild-item[data-guild-id="${state.guildId}"]`);
      if (guildItem) guildItem.classList.add("active");
    }
  }

  /* ============================================================
     CHECK SEND PERMISSION
     ============================================================ */
  async function checkSendPermission() {
    state.canSend = true;
    state.sendDisabledReason = "";

    if (!state.channelId) {
      state.canSend = false;
      state.sendDisabledReason = "No channel selected";
      hideComposer();
      updateComposerPlaceholder();
      return;
    }

    if (state.channelType === "dm") {
      state.canSend = true;
      state.sendDisabledReason = "";
      showComposer();
      updateComposerPlaceholder();
      return;
    }

    // For guild channels, check permissions from the sidebar data we already have
    const channelEl = qs(`.sidebar-item[data-channel-id="${state.channelId}"]`);
    if (channelEl && channelEl.classList.contains("disabled")) {
      // Check why — we need to fetch permission info
      try {
        const permData = await apiFetch(`/api/permissions/${state.channelId}`);
        if (permData && permData.found) {
          const perms = permData.permissions || {};
          if (!perms.send_messages && !perms.administrator) {
            state.canSend = false;
            state.sendDisabledReason = "Bot lacks Send Messages permission in #" + (state.channelName || "this channel");
          } else if (!perms.read_messages) {
            state.canSend = false;
            state.sendDisabledReason = "Bot cannot read messages in #" + (state.channelName || "this channel");
          } else {
            state.canSend = true;
            state.sendDisabledReason = "";
          }
        }
      } catch {
        state.canSend = false;
        state.sendDisabledReason = "Cannot verify permissions";
      }
    }

    updateComposerPlaceholder();
    if (state.canSend) {
      showComposer();
    } else {
      hideComposer();
    }
  }

  /* ============================================================
     COMPOSER PLACEHOLDER
     ============================================================ */
  function updateComposerPlaceholder() {
    const input = $("composer-input");
    const status = $("composer-status");
    if (!state.canSend && state.sendDisabledReason) {
      input.placeholder = "Cannot send";
      status.textContent = state.sendDisabledReason;
      status.className = "text-muted small composer-warning";
    } else if (state.channelType === "dm") {
      input.placeholder = `Message @${state.channelName || "User"} as ${state.botName || "Bot"}`;
      status.textContent = `Messaging @${state.channelName || "User"} as ${state.botName || "Bot"}`;
      status.className = "text-muted small composer-info";
    } else if (state.channelName) {
      input.placeholder = `Message #${state.channelName} as ${state.botName || "Bot"}`;
      status.textContent = `Messaging #${state.channelName} as ${state.botName || "Bot"}`;
      status.className = "text-muted small composer-info";
    } else {
      input.placeholder = "Message...";
      status.textContent = "";
    }
  }

  /* ============================================================
     LIVE REFRESH
     ============================================================ */
  let refreshTimer = null;

  function clearRefresh() {
    if (refreshTimer) {
      clearTimeout(refreshTimer);
      refreshTimer = null;
    }
  }

  function scheduleRefresh() {
    clearRefresh();
    if (
      !state.authenticated ||
      !state.channelId ||
      state.view === "overview" ||
      state.view === "backfill" ||
      state.view === "audit"
    ) {
      return;
    }
    refreshTimer = setTimeout(() => {
      refreshCurrentChannel();
    }, 15000);
  }

  async function refreshCurrentChannel() {
    if (!state.channelId || !Array.isArray(state.messages) || state.messages.length === 0) {
      scheduleRefresh();
      return;
    }

    const isDm = state.channelType === "dm";
    const latest = state.messages[state.messages.length - 1];
    const latestId = getMessageId(latest);
    if (!latestId) {
      scheduleRefresh();
      return;
    }

    const path = isDm
      ? `/api/dms/${state.channelId}/messages?page=1&page_size=50&after_id=${latestId}&auto_backfill=true`
      : `/api/channels/${state.channelId}/messages?page=1&page_size=50&after_id=${latestId}&auto_backfill=true`;

    const container = $("messages-container");
    const wasNearBottom = !container || (container.scrollHeight - container.scrollTop - container.clientHeight < 150);

    const data = await apiFetch(path);
    if (data && Array.isArray(data.messages) && data.messages.length > 0) {
      // API returns newest-first; reverse so they append in chronological order
      state.messages = state.messages.concat([...data.messages].reverse());
      renderMessageList(state.messages, { scrollToLatest: wasNearBottom });
    }
    scheduleRefresh();
  }

  function stopRefreshOnViewChange(view) {
    clearRefresh();
    if (view === "guild" || view === "dm") {
      scheduleRefresh();
    }
  }

  /* ============================================================
     RENDER MESSAGE LIST
     ============================================================ */
  function renderMessageList(messages, { scrollToLatest = true } = {}) {
    const list = $("messages-list");
    list.innerHTML = "";

    if (state.loadingMessages) {
      // Loading skeleton
      for (let i = 0; i < 5; i++) {
        const skeleton = createElement("div", { className: "message-group skeleton" });
        skeleton.innerHTML = `
          <div class="msg-avatar"><div class="skeleton-avatar"></div></div>
          <div class="msg-body">
            <div class="msg-header"><span class="skeleton-line" style="width:120px"></span><span class="skeleton-line" style="width:60px;margin-left:8px"></span></div>
            <div class="skeleton-line" style="width:80%;margin-top:4px"></div>
            <div class="skeleton-line" style="width:50%;margin-top:2px"></div>
          </div>`;
        list.appendChild(skeleton);
      }
      return;
    }

    if (!messages || messages.length === 0) {
      $("messages-empty").classList.remove("hidden");
      $("messages-end").classList.add("hidden");
      $("load-older-btn").classList.add("hidden");
      $("backfill-prompt").classList.add("hidden");
      return;
    }

    $("messages-empty").classList.add("hidden");
    $("messages-end").classList.remove("hidden");
    $("load-older-btn").classList.toggle("hidden", !state.hasMoreOlder);

    // End-of-archive indicator
    if (!state.hasMoreOlder) {
      $("backfill-prompt").classList.remove("hidden");
    } else {
      $("backfill-prompt").classList.add("hidden");
    }

    let lastAuthor = null;
    let lastDate = null;
    let lastTimestamp = null;
    let groupEl = null;

    // We show messages in chronological order (oldest first)
    // But API returns newest first. Already reversed in rendering.
    messages.forEach((msg) => {
      const content = msg.content || msg.content_preview || "";
      const author = msg.author_display_name || msg.author_username || (msg.direction === "outbound" ? "Bot" : "Unknown");
      const authorId = msg.author_id || (msg.direction === "outbound" ? "bot" : "0");
      const timestamp = msg.created_at || msg.timestamp || "";
      const msgId = getMessageId(msg);
      const isBot = msg.author_is_bot === true || msg.is_own_bot === true;
      const authorAvatar = msg.author_avatar_url || null;
      const replyToId = msg.reply_to_message_id || null;
      const rawAttachments = msg.attachments_json || msg.attachments || [];
      const attachments = Array.isArray(rawAttachments) ? rawAttachments : [];
      const rawEmbeds = msg.embeds_json || msg.embeds || [];
      const embeds = Array.isArray(rawEmbeds) ? rawEmbeds : [];

      // Date separator
      const msgDate = formatDate(timestamp);
      if (msgDate && msgDate !== lastDate) {
        lastDate = msgDate;
        const sep = createElement("div", { className: "date-separator" });
        sep.innerHTML = `<span>${esc(msgDate)}</span>`;
        list.appendChild(sep);
        lastAuthor = null;
        lastTimestamp = null;
        groupEl = null;
      }

      const sameAuthor = lastAuthor === authorId;
      const withinWindow = lastTimestamp && timestamp &&
        (new Date(timestamp) - new Date(lastTimestamp)) < 7 * 60 * 1000;
      const isGrouped = sameAuthor && withinWindow && !replyToId;
      lastAuthor = authorId;
      lastTimestamp = timestamp;

      if (!isGrouped || !groupEl) {
        // New message group — first message show avatar + header
        groupEl = createElement("div", { className: "message-group" });

        // Avatar
        const avatarDiv = createElement("div", { className: "msg-avatar" });
        if (authorAvatar) {
          avatarDiv.innerHTML = `<img src="${esc(authorAvatar)}" alt="" loading="lazy">`;
        } else {
          const initial = (author || "?").charAt(0).toUpperCase();
          const bg = colorFromId(authorId === "bot" ? "0" : (Number(authorId) || 1));
          avatarDiv.innerHTML = `<div class="avatar-fallback" style="background:${bg}">${esc(initial)}</div>`;
        }
        groupEl.appendChild(avatarDiv);

        // Body
        const bodyDiv = createElement("div", { className: "msg-body" });

        // Header
        let headerHtml = `<span class="msg-author">${esc(author)}</span>`;
        if (isBot) headerHtml += `<span class="msg-bot-badge">BOT</span>`;
        headerHtml += `<span class="msg-timestamp">${formatTime(timestamp)}</span>`;
        const headerDiv = createElement("div", { className: "msg-header" });
        headerDiv.innerHTML = headerHtml;
        bodyDiv.appendChild(headerDiv);

        // Reply indicator
        if (replyToId) {
          bodyDiv.appendChild(createElement("div", { className: "reply-preview-msg", textContent: "↪ Replying to a message" }));
        }

        // Content
        const contentDiv = createElement("div", { className: "msg-content" });
        contentDiv.textContent = content;
        bodyDiv.appendChild(contentDiv);

        // Edited tag
        if (msg.edited_at) {
          contentDiv.appendChild(createElement("span", { className: "edited", textContent: " (edited)" }));
        }

        // Attachments
        attachments.forEach((a) => {
          const url = a.url || "";
          const name = a.filename || "attachment";
          const size = a.size ? ` (${(a.size / 1024).toFixed(0)} KB)` : "";
          if (url) {
            bodyDiv.appendChild(createElement("a", {
              className: "attachment-card",
              href: url,
              target: "_blank",
              rel: "noopener",
              innerHTML: `<span class="attach-icon">📎</span><span class="attach-name">${esc(name)}${esc(size)}</span>`,
            }));
          }
        });

        // Embeds
        embeds.forEach((e) => {
          const embedEl = createElement("div", { className: "embed-card" });
          let html = "";
          if (e.title) html += `<div class="embed-title">${esc(e.title)}</div>`;
          if (e.description) html += `<div class="embed-desc">${esc(e.description)}</div>`;
          if (e.url) html += `<div class="embed-provider"><a href="${esc(e.url)}" target="_blank" rel="noopener">${esc(e.url)}</a></div>`;
          if (html) embedEl.innerHTML = html;
          bodyDiv.appendChild(embedEl);
        });

        // Actions (reply btn)
        if (msgId) {
          const actionsDiv = createElement("div", { className: "msg-actions", style: "display:none" });
          const replyBtn = createElement("button", { className: "icon-btn", title: "Reply", style: "font-size:11px;padding:2px 6px" });
          replyBtn.textContent = "↩ Reply";
          replyBtn.addEventListener("click", (e) => { e.stopPropagation(); setReply(msgId, author, content); });
          actionsDiv.appendChild(replyBtn);
          bodyDiv.appendChild(actionsDiv);
          groupEl.addEventListener("mouseenter", () => { actionsDiv.style.display = "flex"; });
          groupEl.addEventListener("mouseleave", () => { actionsDiv.style.display = "none"; });
        }

        groupEl.appendChild(bodyDiv);
        list.appendChild(groupEl);
      } else {
        // Compact mode — same author within 7 min, continuation message
        const compactDiv = createElement("div", { className: "message-compact" });
        let replyHtml = replyToId ? `<div class="reply-preview-msg"><span class="reply-author">↪ Replying to a message</span></div>` : "";
        // Timestamp goes in the left gutter (revealed on hover via CSS)
        compactDiv.innerHTML = `${replyHtml}<span class="msg-timestamp">${formatTime(timestamp)}</span><div class="msg-content">${esc(content)}</div>`;

        attachments.forEach((a) => {
          const url = a.url || "";
          const name = a.filename || "attachment";
          const size = a.size ? ` (${(a.size / 1024).toFixed(0)} KB)` : "";
          if (url) {
            compactDiv.appendChild(createElement("a", {
              className: "attachment-card",
              href: url,
              target: "_blank",
              rel: "noopener",
              innerHTML: `<span class="attach-icon">📎</span><span class="attach-name">${esc(name)}${esc(size)}</span>`,
            }));
          }
        });

        if (msgId) {
          const actionsDiv = createElement("div", { className: "msg-actions", style: "display:none" });
          const replyBtn = createElement("button", { className: "icon-btn", title: "Reply", style: "font-size:11px;padding:2px 6px" });
          replyBtn.textContent = "↩ Reply";
          replyBtn.addEventListener("click", (e) => { e.stopPropagation(); setReply(msgId, author, content); });
          actionsDiv.appendChild(replyBtn);
          compactDiv.appendChild(actionsDiv);
          compactDiv.addEventListener("mouseenter", () => { actionsDiv.style.display = "flex"; });
          compactDiv.addEventListener("mouseleave", () => { actionsDiv.style.display = "none"; });
        }

        list.appendChild(compactDiv);
      }
    });

    // Scroll to latest
    if (scrollToLatest) scrollToBottom();
  }

  function scrollToBottom() {
    const list = $("messages-list");
    setTimeout(() => {
      const container = $("messages-container");
      if (container) container.scrollTop = container.scrollHeight;
    }, 50);
  }

  function clearMessages() {
    state.messages = [];
    $("messages-list").innerHTML = "";
    $("messages-empty").classList.add("hidden");
    $("messages-end").classList.add("hidden");
    $("load-older-btn").classList.add("hidden");
    $("backfill-prompt").classList.add("hidden");
    $("messages-error").classList.add("hidden");
  }

  /* ============================================================
     LOAD OLDER MESSAGES
     ============================================================ */
  async function loadOlderMessages() {
    if (!state.hasMoreOlder || !state.channelId) return;
    const nextPage = state.messagesPage + 1;
    const path = state.channelType === "dm"
      ? `/api/dms/${state.channelId}/messages?page=${nextPage}&page_size=50&auto_backfill=false`
      : `/api/channels/${state.channelId}/messages?page=${nextPage}&page_size=50&auto_backfill=false`;
    const data = await apiFetch(path);
    if (!data) return;
    state.messagesPage = data.page || nextPage;
    state.messagesTotalPages = data.total_pages || 1;
    state.hasMoreOlder = data.page < data.total_pages;
    const msgs = data.messages || data.events || [];
    if (msgs.length > 0) {
      // API returns newest-first; reverse so older pages prepend in chronological order
      state.messages = [...msgs].reverse().concat(state.messages);
      renderMessageList(state.messages, { scrollToLatest: false });
    }
    $("load-older-btn").classList.toggle("hidden", !state.hasMoreOlder);
    if (!state.hasMoreOlder) {
      $("backfill-prompt").classList.remove("hidden");
    }
  }

  /* ============================================================
     COMPOSER
     ============================================================ */
  function showComposer() {
    $("composer-container").classList.remove("hidden");
    $("composer-input").disabled = false;
    $("composer-send").disabled = false;
  }

  function hideComposer() {
    $("composer-container").classList.add("hidden");
    $("composer-input").disabled = true;
    $("composer-send").disabled = true;
    hideReply();
  }

  function hideReply() {
    state.replyTo = null;
    $("reply-preview").classList.add("hidden");
    $("reply-author").textContent = "";
    $("reply-preview-text").textContent = "";
  }

  function setReply(msgId, author, preview) {
    state.replyTo = { id: msgId, author, preview, channelId: state.channelId };
    $("reply-author").textContent = esc(author);
    $("reply-preview-text").textContent = esc((preview || "").substring(0, 80));
    $("reply-preview").classList.remove("hidden");
    $("composer-input").focus();
  }

  function updateCharCount() {
    const input = $("composer-input");
    const maxChars = config.max_message_chars || 1800;
    const len = input.value.length;
    const el = $("char-count");
    if (len > 0) {
      el.textContent = `${len}/${maxChars}`;
      el.classList.toggle("char-warning", len > maxChars * 0.9);
      el.classList.toggle("char-over", len > maxChars);
      el.classList.remove("hidden");
    } else {
      el.classList.add("hidden");
    }
  }

  async function sendMessage() {
    const input = $("composer-input");
    const content = input.value.trim();
    if (!content || !state.channelId) return;

    const maxChars = config.max_message_chars || 1800;
    if (content.length > maxChars) {
      toast(`Message exceeds ${maxChars} characters`, "warning");
      return;
    }

    $("composer-send").disabled = true;
    $("composer-send").innerHTML = "<div class='spinner-small'></div>";

    let path, body;
    if (state.replyTo) {
      path = "/api/messages/" + state.replyTo.id + "/reply";
      body = { content, channel_id: String(state.channelId) };
    } else if (state.channelType === "dm") {
      // If this is a new DM (arbitrary user ID), use the legacy endpoint
      // which calls send_dm(target_user_id=...) directly
      if (state.isNewDm) {
        path = "/api/dms/" + state.channelId + "/send";
        body = { content, user_id: String(state.channelId) };
      } else {
        path = "/api/dms/" + state.channelId + "/send";
        body = { content };
      }
    } else {
      path = "/api/channels/" + state.channelId + "/send";
      body = { content };
    }

    const data = await apiFetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-CSRF-Token": csrfToken },
      body: JSON.stringify(body),
    });

    $("composer-send").innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>`;
    $("composer-send").disabled = false;

    if (data && data.success) {
      input.value = "";
      input.style.height = "auto";
      hideReply();
      updateCharCount();
      // Reload to show the new message
      loadChannelMessages(state.channelId, state.channelName, state.channelType);
    } else if (data) {
      toast("Failed: " + (data.error || "unknown"), "error");
    } else {
      toast("Send failed — no response", "error");
    }
  }

  /* ============================================================
     HOME / OVERVIEW VIEW
     ============================================================ */
  async function renderHomeContent() {
    setText("content-name", "Bot Console");
    setText("content-meta", "Select a channel or DM to start");
    $("content-avatar").classList.add("hidden");
    $("content-status").classList.add("hidden");

    const data = await apiFetch("/api/overview");
    const container = createElement("div", { className: "home-cards" });
    const list = $("messages-list");
    const existing = list.querySelector(".home-cards");
    if (existing) existing.remove();

    if (!data) {
      const empty = createElement("div", { className: "empty-state", style: "padding:40px 16px" });
      empty.innerHTML = `<p class="text-muted">Failed to load overview data</p>`;
      container.appendChild(empty);
      list.appendChild(container);
      return;
    }

    // --- Bot Status Card ---
    const statusCard = createElement("div", { className: "home-card" });
    const statusColor = data.status === "ready" ? "green" : data.status === "error" ? "red" : "yellow";
    statusCard.innerHTML = `
      <div class="card-label">BOT STATUS</div>
      <div class="card-value ${statusColor}">${esc(data.status || "unknown")}</div>
      <div class="card-sub">${esc(data.bot_username)}${data.bot_discriminator ? "#" + esc(data.bot_discriminator) : ""}</div>
    `;
    container.appendChild(statusCard);

    // --- Uptime Card ---
    const uptimeCard = createElement("div", { className: "home-card" });
    const uptime = data.bot_uptime_seconds || data.uptime_seconds || 0;
    const uptimeDisplay = uptime >= 86400
      ? Math.floor(uptime / 86400) + "d"
      : uptime >= 3600
        ? Math.floor(uptime / 3600) + "h"
        : uptime >= 60
          ? Math.floor(uptime / 60) + "m"
          : uptime + "s";
    uptimeCard.innerHTML = `
      <div class="card-label">UPTIME</div>
      <div class="card-value">${esc(uptimeDisplay)}</div>
      <div class="card-sub">${uptime.toLocaleString()} seconds</div>
    `;
    container.appendChild(uptimeCard);

    // --- Latency Card ---
    if (data.latency_ms !== undefined) {
      const latencyCard = createElement("div", { className: "home-card" });
      latencyCard.innerHTML = `
        <div class="card-label">LATENCY</div>
        <div class="card-value">${esc(data.latency_ms)}ms</div>
        <div class="card-sub">WebSocket ping</div>
      `;
      container.appendChild(latencyCard);
    }

    // --- Guilds Card ---
    const guildCard = createElement("div", { className: "home-card" });
    guildCard.innerHTML = `
      <div class="card-label">GUILDS</div>
      <div class="card-value">${esc(data.total_guilds)}</div>
      <div class="card-sub">Connected servers</div>
    `;
    container.appendChild(guildCard);

    // --- Channels Card ---
    const chCard = createElement("div", { className: "home-card" });
    chCard.innerHTML = `
      <div class="card-label">CHANNELS</div>
      <div class="card-value">${esc(data.total_channels)}</div>
      <div class="card-sub">Text & voice channels</div>
    `;
    container.appendChild(chCard);

    // --- DMs Card ---
    const dmCard = createElement("div", { className: "home-card" });
    dmCard.innerHTML = `
      <div class="card-label">DM THREADS</div>
      <div class="card-value">${esc(data.total_dms)}</div>
      <div class="card-sub">Conversations</div>
    `;
    container.appendChild(dmCard);

    // --- Archived Messages Card ---
    const msgCard = createElement("div", { className: "home-card" });
    msgCard.innerHTML = `
      <div class="card-label">ARCHIVED MESSAGES</div>
      <div class="card-value">${esc(data.total_archived_messages)}</div>
      <div class="card-sub">In message store</div>
    `;
    container.appendChild(msgCard);

    // --- Audit Events Card ---
    const auditCard = createElement("div", { className: "home-card" });
    auditCard.innerHTML = `
      <div class="card-label">AUDIT EVENTS</div>
      <div class="card-value">${esc(data.total_audit_events)}</div>
      <div class="card-sub">Logged events</div>
    `;
    container.appendChild(auditCard);

    // --- Playwright Status Card ---
    const pwCard = createElement("div", { className: "home-card" });
    let pwStatusColor = "yellow";
    let pwStatusText = "checking...";
    if (data.playwright_available === true) {
      pwStatusColor = "green";
      pwStatusText = "Connected";
    } else if (data.playwright_available === false) {
      if (data.playwright_degraded) {
        pwStatusColor = "red";
        pwStatusText = "Degraded";
      } else {
        pwStatusColor = "red";
        pwStatusText = "Unavailable";
      }
    } else {
      pwStatusColor = "yellow";
      pwStatusText = "Not configured";
    }
    pwCard.innerHTML = `
      <div class="card-label">PLAYWRIGHT</div>
      <div class="card-value ${pwStatusColor}">${esc(pwStatusText)}</div>
      <div class="card-sub">Browser automation</div>
    `;
    container.appendChild(pwCard);

    // --- Cogs Card ---
    if (data.cog_count !== undefined) {
      const cogCard = createElement("div", { className: "home-card" });
      cogCard.innerHTML = `
        <div class="card-label">COGS</div>
        <div class="card-value">${esc(data.cog_count)}</div>
        <div class="card-sub">Loaded modules</div>
      `;
      container.appendChild(cogCard);
    }

    // --- Total Users Card ---
    if (data.total_users_estimate !== undefined) {
      const usersCard = createElement("div", { className: "home-card" });
      usersCard.innerHTML = `
        <div class="card-label">TOTAL USERS</div>
        <div class="card-value">${esc(data.total_users_estimate)}</div>
        <div class="card-sub">Across all guilds</div>
      `;
      container.appendChild(usersCard);
    }

    // --- Guild Cards (one per guild for quick nav) ---
    if (data.guilds && data.guilds.length > 0) {
      const guildHeader = createElement("div", { className: "home-card", style: "grid-column:1/-1" });
      guildHeader.innerHTML = `<div class="card-label">SERVERS (${esc(data.guilds.length)})</div>`;
      container.appendChild(guildHeader);

      data.guilds.forEach((g) => {
        const gCard = createElement("div", {
          className: "home-card",
          style: "cursor:pointer",
          title: "Click to view channels",
        });
        const acronym = (g.name || "?").split(/\s+/).slice(0, 2).map((w) => w[0]).join("").toUpperCase();
        const avatarHtml = g.icon_url
          ? `<img src="${esc(g.icon_url)}" alt="" style="width:24px;height:24px;border-radius:4px;float:left;margin-right:8px">`
          : `<div style="width:24px;height:24px;border-radius:4px;background:${colorFromId(g.id)};float:left;margin-right:8px;display:flex;align-items:center;justify-content:center;color:#fff;font-size:9px;font-weight:700">${esc(acronym)}</div>`;
        gCard.innerHTML = `
          ${avatarHtml}
          <div class="card-label">${esc(g.name)}</div>
          <div class="card-value" style="font-size:14px">${esc(g.member_count)}</div>
          <div class="card-sub">members · ${esc(g.id)}</div>
        `;
        gCard.addEventListener("click", () => {
          state.messages = [];
          updateHash("guild", g.id);
          showView("guild", g.id);
        });
        container.appendChild(gCard);
      });
    }

    list.appendChild(container);
  }

  /* ============================================================
     BACKFILL VIEW
     ============================================================ */
  function renderBackfillSidebar() {
    $("sidebar-content").innerHTML = `<div class="sidebar-item active"><span class="icon">📦</span><span class="name">Jobs</span></div>`;
  }

  async function renderBackfillContent() {
    const main = $("main-content");
    setText("content-name", "Backfill Manager");
    setText("content-meta", "Archive channel & DM history");
    $("content-avatar").classList.add("hidden");
    $("content-status").classList.add("hidden");

    const data = await apiFetch("/api/backfill/jobs");
    const container = createElement("div", { className: "home-cards" });

    const formCard = createElement("div", { className: "home-card", style: "grid-column:1/-1" });
    formCard.innerHTML = `
      <div class="card-label">NEW BACKFILL</div>
      <div style="display:flex;gap:8px;margin-top:8px;flex-wrap:wrap">
        <input type="text" id="backfill-channel-input" placeholder="Channel ID" style="flex:1;min-width:150px;padding:6px;background:var(--bg-input);border:1px solid var(--border);color:var(--text-normal);border-radius:var(--radius)">
        <select id="backfill-type-select" style="padding:6px;background:var(--bg-input);border:1px solid var(--border);color:var(--text-normal);border-radius:var(--radius)">
          <option value="channel">Channel</option>
          <option value="dm">DM</option>
          <option value="guild">Guild (all channels)</option>
        </select>
        <button id="backfill-start-btn" class="btn btn-sm">Start Backfill</button>
      </div>
    `;
    container.appendChild(formCard);

    const jobs = data?.jobs || [];
    if (jobs.length > 0) {
      const jobsCard = createElement("div", { className: "home-card", style: "grid-column:1/-1" });
      jobsCard.innerHTML = `<div class="card-label">ACTIVE / RECENT JOBS</div>`;
      jobs.forEach((j) => {
        const row = createElement("div", { className: "detail-row", style: "padding:6px 0;border-bottom:1px solid var(--bg-hover);font-size:12px" });
        const statusCls = j.status === "completed" ? "green" : j.status === "failed" ? "red" : "yellow";
        row.innerHTML = `
          <span class="detail-label">${esc(j.type || "?")} → ${esc(j.target_id || "")}</span>
          <span><span class="card-value ${statusCls}" style="font-size:12px">${esc(j.status)}</span>
          ${j.status === "running" || j.status === "pending" ? `<button class="btn btn-sm" data-cancel="${j.job_id}" style="margin-left:6px">Cancel</button>` : ""}</span>
        `;
        jobsCard.appendChild(row);
      });
      container.appendChild(jobsCard);
    } else {
      const emptyCard = createElement("div", { className: "home-card", style: "grid-column:1/-1" });
      emptyCard.innerHTML = `<div class="card-label">NO JOBS</div><div class="text-muted" style="margin-top:4px">Start a backfill above</div>`;
      container.appendChild(emptyCard);
    }

    const existing = main.querySelector(".home-cards");
    if (existing) existing.remove();
    main.appendChild(container);
    $("backfill-start-btn")?.addEventListener("click", triggerBackfill);

    // Wire cancel buttons
    setTimeout(() => {
      qsa("[data-cancel]").forEach((btn) => {
        btn.addEventListener("click", async () => {
          const jobId = btn.dataset.cancel;
          await apiFetch(`/api/backfill/jobs/${jobId}/cancel`, { method: "POST" });
          renderBackfillContent();
        });
      });
    }, 50);
  }

  async function triggerBackfill() {
    const channelId = $("backfill-channel-input").value.trim();
    const type = $("backfill-type-select").value;
    if (!channelId) { toast("Enter a channel/DM/guild ID", "warning"); return; }
    let path;
    if (type === "channel") path = "/api/backfill/channel/" + channelId;
    else if (type === "dm") path = "/api/backfill/dm/" + channelId;
    else path = "/api/backfill/guild/" + channelId;
    const data = await apiFetch(path, { method: "POST" });
    if (data && (data.job_id || data.status === "started" || data.status === "completed")) {
      if (data.status === "failed") {
        toast("Backfill failed: " + (data.error || "unknown"), "error");
      } else {
        toast("Backfill " + (data.status || "started"), "success");
        renderBackfillContent();
      }
    } else {
      toast("Backfill: " + ((data && (data.error || data.status)) || "timeout"), "error");
    }
  }

  /* ============================================================
     AUDIT VIEW
     ============================================================ */
  function renderAuditSidebar() {
    $("sidebar-content").innerHTML = `
      <div style="padding:8px">
        <select id="audit-filter-event" style="width:100%;padding:6px;margin-bottom:6px;background:var(--bg-input);border:1px solid var(--border);color:var(--text-normal);border-radius:var(--radius);font-size:12px">
          <option value="">All events</option>
          <option value="dashboard.channel.view">Channel view</option>
          <option value="dashboard.dm.view">DM view</option>
          <option value="dashboard.message.send.success">Send success</option>
          <option value="dashboard.message.send.failure">Send fail</option>
          <option value="dashboard.message.reply.success">Reply success</option>
          <option value="dashboard.backfill.requested">Backfill</option>
          <option value="dashboard.auth.failure">Auth fail</option>
        </select>
        <select id="audit-filter-result" style="width:100%;padding:6px;margin-bottom:6px;background:var(--bg-input);border:1px solid var(--border);color:var(--text-normal);border-radius:var(--radius);font-size:12px">
          <option value="">All results</option>
          <option value="success">Success</option>
          <option value="failed">Failed</option>
          <option value="pending">Pending</option>
          <option value="rate_limited">Rate limited</option>
        </select>
        <button id="audit-filter-btn" class="btn btn-sm" style="width:100%">Filter</button>
      </div>
    `;
  }

  let auditPage = 1;

  async function renderAuditContent(page) {
    auditPage = page || 1;
    const main = $("main-content");
    setText("content-name", "Audit Log");
    setText("content-meta", "Event history");
    $("content-avatar").classList.add("hidden");
    $("content-status").classList.add("hidden");

    const eventFilter = $("audit-filter-event")?.value || "";
    const resultFilter = $("audit-filter-result")?.value || "";
    let qs = "page=" + auditPage + "&page_size=50";
    if (eventFilter) qs += "&event_type=" + encodeURIComponent(eventFilter);
    if (resultFilter) qs += "&result=" + encodeURIComponent(resultFilter);

    const data = await apiFetch("/api/audit?" + qs);
    const container = createElement("div", { className: "home-cards" });

    if (data && data.events && data.events.length > 0) {
      const tableCard = createElement("div", { className: "home-card", style: "grid-column:1/-1" });
      let html = `<div class="card-label">EVENTS (page ${data.page || 1} / ${data.total_pages || 1})</div><div style="margin-top:8px">`;
      data.events.forEach((e) => {
        const badgeCls = e.result === "success" ? "green" : e.result === "pending" ? "yellow" : "red";
        html += `<div class="detail-row" style="padding:6px 0;border-bottom:1px solid var(--bg-hover);font-size:12px">
          <span class="detail-label"><span class="card-value ${badgeCls}" style="font-size:10px;margin-right:4px">${esc(e.result)}</span> ${esc(e.event_type)}</span>
          <span class="detail-value">${esc(e.content_preview || "")} ${ago(e.created_at) ? "- " + ago(e.created_at) : ""}</span>
        </div>`;
      });
      html += `</div>`;
      if (data.total_pages > 1) {
        html += `<div style="display:flex;gap:8px;margin-top:12px;align-items:center">`;
        if (data.page > 1) html += `<button class="btn btn-sm" id="audit-prev-btn">← Prev</button>`;
        html += `<span class="text-muted" style="font-size:12px">${data.page} / ${data.total_pages}</span>`;
        if (data.page < data.total_pages) html += `<button class="btn btn-sm" id="audit-next-btn">Next →</button>`;
        html += `</div>`;
      }
      tableCard.innerHTML = html;
      container.appendChild(tableCard);
    } else {
      container.appendChild(createElement("div", { className: "home-card", style: "grid-column:1/-1", innerHTML: `<div class="card-label">NO EVENTS</div><div class="text-muted" style="margin-top:4px">No audit events match</div>` }));
    }

    const existing = main.querySelector(".home-cards");
    if (existing) existing.remove();
    main.appendChild(container);
    setTimeout(() => {
      $("audit-prev-btn")?.addEventListener("click", () => renderAuditContent(auditPage - 1));
      $("audit-next-btn")?.addEventListener("click", () => renderAuditContent(auditPage + 1));
      $("audit-filter-btn")?.addEventListener("click", () => renderAuditContent(1));
    }, 50);
  }

  /* ============================================================
     DETAILS PANEL
     ============================================================ */
  function clearDetails() {
    $("details-panel").classList.add("hidden");
    $("details-content").innerHTML = "";
  }

  /* ============================================================
     COMPOSER KEYBOARD + EVENT BINDING
     ============================================================ */
  function setupComposerKeyboard() {
    const input = $("composer-input");
    const sendBtn = $("composer-send");

    input.addEventListener("input", () => {
      sendBtn.disabled = !input.value.trim() || !state.canSend;
      input.style.height = "auto";
      input.style.height = Math.min(input.scrollHeight, 200) + "px";
      updateCharCount();
    });

    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    });

    sendBtn.addEventListener("click", sendMessage);
    $("reply-cancel").addEventListener("click", hideReply);
  }

  /* ============================================================
     INIT
     ============================================================ */
  async function init() {
    const authed = await checkSession();

    // Login form
    $("login-form").addEventListener("submit", (e) => {
      e.preventDefault();
      const token = $("auth-token-input").value.trim();
      if (token) handleLogin(token);
    });

    // Logout
    $("logout-btn")?.addEventListener("click", handleLogout);

    // Guild rail
    if (!authed) await loadGuildRail();

    // Navigation
    qs(".guild-home")?.addEventListener("click", () => {
      state.messages = [];
      showView("overview");
    });
    $("nav-backfill")?.addEventListener("click", () => {
      state.messages = [];
      showView("backfill");
    });
    $("nav-audit")?.addEventListener("click", () => {
      state.messages = [];
      showView("audit");
    });

    // Refresh
    $("refresh-btn")?.addEventListener("click", () => {
      switch (state.view) {
        case "overview": renderHomeContent(); renderHomeSidebar(); break;
        case "guild":
        case "dm": if (state.channelId) loadChannelMessages(state.channelId, state.channelName, state.channelType); break;
        case "backfill": renderBackfillContent(); break;
        case "audit": renderAuditContent(auditPage); break;
      }
    });

    // Load older
    $("load-older-btn")?.addEventListener("click", loadOlderMessages);

    // Backfill prompt button
    $("backfill-channel-btn")?.addEventListener("click", async () => {
      if (!state.channelId) return;
      const path = state.channelType === "dm"
        ? "/api/backfill/dm/" + state.channelId
        : "/api/backfill/channel/" + state.channelId;
      const data = await apiFetch(path, { method: "POST" });
      if (data && (data.job_id || data.status === "started" || data.status === "completed")) {
        if (data.status === "failed") {
          toast("Backfill failed: " + (data.error || "unknown"), "error");
        } else {
          toast("Backfill " + (data.status || "started") + " for " + state.channelName, "success");
          await sleep(3000);
          loadChannelMessages(state.channelId, state.channelName, state.channelType);
        }
      } else {
        toast("Backfill: " + ((data && (data.error || data.status)) || "timeout/no response"), "error");
      }
    });

    // Composer
    setupComposerKeyboard();

    // Sidebar search
    $("sidebar-search")?.addEventListener("input", (e) => {
      const q = e.target.value.toLowerCase();
      qsa(".sidebar-item").forEach((el) => {
        const name = el.querySelector(".name")?.textContent?.toLowerCase() || "";
        el.style.display = name.includes(q) ? "" : "none";
      });
    });

    // Jump to latest button
    $("jump-to-latest")?.addEventListener("click", () => {
      const container = $("messages-container");
      if (container) container.scrollTop = container.scrollHeight;
      $("jump-to-latest").classList.add("hidden");
    });

    // Scroll detection for jump-to-latest button
    const msgContainer = $("messages-container");
    if (msgContainer) {
      msgContainer.addEventListener("scroll", () => {
        const threshold = 200;
        const isNearBottom = msgContainer.scrollHeight - msgContainer.scrollTop - msgContainer.clientHeight < threshold;
        $("jump-to-latest")?.classList.toggle("hidden", isNearBottom);
      });
    }

    // Hash change handler
    window.addEventListener("hashchange", () => {
      if (state.authenticated) handleHash();
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
