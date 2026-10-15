/* CareImpact AI help assistant — floating chat widget. Talks to POST /api/chat. */
(() => {
  "use strict";
  const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  // Tiny, safe Markdown subset: escape first, then **bold**, *italic*, `code`, line breaks.
  const md = (text) => esc(text)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>")
    .replace(/\n/g, "<br>");

  const STARTERS = ["How do I compare two policies?", "Summarize my current result", "What does 'verified' mean?", "Why is the AI model offline?"];
  const history = [];

  document.body.insertAdjacentHTML("beforeend", `
    <button class="chat-launcher" id="chatLauncher" aria-label="Open help assistant" aria-expanded="false" aria-controls="chatPanel">
      <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 5h16v11H9l-5 4z"/><path d="M8 10h8M8 13h5"/></svg>
      <span>Ask CareImpact</span>
    </button>
    <section class="chat-panel" id="chatPanel" role="dialog" aria-label="Help assistant" hidden>
      <header class="chat-head">
        <div><strong>CareImpact Assistant</strong><span id="chatMode">Ask about the app or your result</span></div>
        <button class="chat-icon-btn" id="chatReset" title="New conversation" aria-label="Start a new conversation"><svg viewBox="0 0 24 24"><path d="M4 12a8 8 0 1 0 3-6.2M4 4v4h4"/></svg></button>
        <button class="chat-icon-btn" id="chatClose" title="Close" aria-label="Close assistant"><svg viewBox="0 0 24 24"><path d="M6 6l12 12M18 6L6 18"/></svg></button>
      </header>
      <div class="chat-log" id="chatLog" aria-live="polite"></div>
      <div class="chat-suggest" id="chatSuggest"></div>
      <form class="chat-form" id="chatForm">
        <label class="sr-only" for="chatInput">Your question</label>
        <textarea id="chatInput" rows="1" maxlength="2000" placeholder="Type a question…"></textarea>
        <button class="chat-send" type="submit" aria-label="Send"><svg viewBox="0 0 24 24"><path d="M4 12l16-8-6 16-2-7z"/></svg></button>
      </form>
      <p class="chat-foot">Answers can be wrong. Not legal or CMS guidance.</p>
    </section>`);

  const $ = (s) => document.querySelector(s);
  const panel = $("#chatPanel"), log = $("#chatLog"), input = $("#chatInput"), launcher = $("#chatLauncher");

  function bubble(role, html) {
    const el = document.createElement("div");
    el.className = `msg ${role}`;
    el.innerHTML = html;
    log.appendChild(el);
    log.scrollTop = log.scrollHeight;
    return el;
  }
  function suggestions(list) {
    $("#chatSuggest").innerHTML = (list || []).map((q) => `<button type="button">${esc(q)}</button>`).join("");
  }
  function greet() {
    log.innerHTML = "";
    history.length = 0;
    bubble("bot", md("Hi! I'm the CareImpact assistant. I can explain how to use the app, what the badges and categories mean, or summarize the comparison you have open."));
    suggestions(STARTERS);
  }

  function open() {
    panel.hidden = false;
    launcher.setAttribute("aria-expanded", "true");
    launcher.classList.add("hidden");
    if (!log.children.length) greet();
    input.focus();
  }
  function close() {
    panel.hidden = true;
    launcher.setAttribute("aria-expanded", "false");
    launcher.classList.remove("hidden");
    launcher.focus();
  }

  async function send(text) {
    text = text.trim();
    if (!text) return;
    bubble("user", esc(text));
    history.push({ role: "user", content: text });
    suggestions([]);
    input.value = ""; input.style.height = "";
    const typing = bubble("bot typing", "<span></span><span></span><span></span>");
    const form = $("#chatForm");
    form.classList.add("busy");
    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: history.slice(-10), context: window.CareImpact?.lastResult || null }),
      });
      if (res.status === 404) throw new Error("the chat service was not found. Restart the server so it picks up the latest code");
      if (!res.ok) throw new Error(`request failed with status ${res.status}`);
      const data = await res.json();
      typing.remove();
      bubble("bot", md(data.reply));
      history.push({ role: "assistant", content: data.reply });
      $("#chatMode").textContent = data.source === "model" ? `Answering with ${data.model}` : "Answering from the built-in guide";
      suggestions(data.suggestions);
    } catch (e) {
      typing.remove();
      bubble("bot error", esc(e instanceof TypeError ? "Sorry, I couldn't reach the server. Check that the app is still running." : `Sorry, ${e.message}.`));
      history.pop();
    } finally {
      form.classList.remove("busy");
    }
  }

  launcher.addEventListener("click", open);
  $("#chatClose").addEventListener("click", close);
  $("#chatReset").addEventListener("click", greet);
  $("#chatSuggest").addEventListener("click", (e) => { if (e.target.matches("button")) send(e.target.textContent); });
  $("#chatForm").addEventListener("submit", (e) => { e.preventDefault(); if (!e.currentTarget.classList.contains("busy")) send(input.value); });
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#chatForm").requestSubmit(); }
  });
  input.addEventListener("input", () => { input.style.height = ""; input.style.height = `${Math.min(input.scrollHeight, 120)}px`; });
  panel.addEventListener("keydown", (e) => { if (e.key === "Escape") close(); });

  window.CareImpactChat = { open, close, send };
})();
