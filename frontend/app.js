/* CareImpact AI dashboard — vanilla JS, no build step. */
(() => {
  "use strict";
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => [...el.querySelectorAll(s)];
  const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const pct = (v) => (v === null || v === undefined ? "—" : `${(v * 100).toFixed(1)}%`);
  const num = (v) => (v === null || v === undefined ? "—" : Number(v).toLocaleString());
  const LABELS = {
    coverage: "Coverage", clinical_criteria: "Clinical criteria", documentation: "Documentation",
    coding_billing: "Coding & billing", provider_requirements: "Provider requirements",
    retirement_replacement: "Retirement / replacement", administrative: "Administrative",
    no_substantive_change: "No substantive change", insufficient_evidence: "Insufficient evidence",
  };
  const label = (k) => LABELS[k] || String(k ?? "—").replace(/_/g, " ");
  const PAGES = { dashboard: "Home", compare: "Compare Policies", library: "Policy Library", dataset: "Dataset & Training", benchmarks: "Model Benchmarking", about: "Help & Responsible Use" };
  const SUBTITLES = {
    dashboard: "Compare Medicare coverage policy versions and see what changed.",
    compare: "Pick an old and a new version, then press Compare.",
    library: "Sample and real policy pairs, ready to compare.",
    dataset: "Measured counts for the data used to train the model.",
    benchmarks: "How each analyzer scores on held-out evaluation sets.",
    about: "How the app works, its limits, and common questions.",
  };
  const state = { status: null, library: null, docs: {}, benchmarks: null };

  async function api(path, opts = {}) {
    const res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
    let body = null;
    try { body = await res.json(); } catch { /* non-JSON */ }
    if (!res.ok) throw new Error((body && (body.detail?.[0]?.msg || body.detail)) || `Request failed (${res.status})`);
    return body;
  }
  const empty = (title, hint) => `<div class="empty"><strong>${esc(title)}</strong>${hint || ""}</div>`;
  function toast(msg) {
    const t = document.createElement("div");
    t.className = "toast"; t.setAttribute("role", "status"); t.textContent = msg;
    document.body.appendChild(t);
    setTimeout(() => t.remove(), 2600);
  }
  const kpi = (lab, value, sub = "") => `<div class="card kpi"><div class="label">${esc(lab)}</div><div class="value">${value}</div><div class="sub">${sub}</div></div>`;

  /* ---------------------------------------------------------------- routing */
  function route() {
    const page = (location.hash || "#dashboard").slice(1).split("?")[0];
    const name = PAGES[page] ? page : "dashboard";
    $$(".page").forEach((p) => (p.hidden = p.id !== `page-${name}`));
    $$(".nav a").forEach((a) => (a.dataset.page === name ? a.setAttribute("aria-current", "page") : a.removeAttribute("aria-current")));
    $("#pageTitle").textContent = PAGES[name];
    $("#pageSub").textContent = SUBTITLES[name];
    document.title = `${PAGES[name]} · CareImpact AI`;
    closeMenu();
    const render = ({ dashboard: renderDashboard, library: renderLibrary, dataset: renderDataset, benchmarks: renderBenchmarks })[name];
    render?.().catch((e) => {
      const host = $(`#page-${name}`);
      $(".notice.error.load", host)?.remove();
      host.insertAdjacentHTML("afterbegin", `<div class="notice error load"><strong>Could not load this page.</strong> ${esc(e.message)} Is the server running?</div>`);
    });
  }
  function closeMenu() { $("#sidebar").classList.remove("open"); $("#menuBtn").setAttribute("aria-expanded", "false"); }
  $("#scrim").addEventListener("click", closeMenu);
  $("#menuBtn").addEventListener("click", () => {
    const open = $("#sidebar").classList.toggle("open");
    $("#menuBtn").setAttribute("aria-expanded", String(open));
  });

  /* ---------------------------------------------------------------- status */
  
async function loadStatus() {
  const sel = $("#modelSelect");
  const hint = $("#modelHint");

  try {
    state.status = await api("/api/status");
    const s = state.status;
    const models = Array.isArray(s.ollama_models)
      ? s.ollama_models
      : [];

    const normalizeModelName = (name) => {
      const value = String(name || "").trim();
      if (!value) return "";
      return value.includes(":") ? value : `${value}:latest`;
    };

    const configured = normalizeModelName(s.configured_model);
    const installed = models.some(
      (model) => normalizeModelName(model) === configured
    );

    hint.textContent = !s.ollama_reachable
      ? "AI model offline, using the rule-based baseline."
      : installed
        ? ""
        : `${s.configured_model} is not installed, using the rule-based baseline.`;

    const opts = [];

    if (s.ollama_reachable) {
      models.forEach((m) => {
        opts.push([
          m,
          `${m}${normalizeModelName(m) === configured ? " (configured)" : ""}`
        ]);
      });
    }

    opts.push([s.baseline, "Deterministic diff baseline"]);

    sel.innerHTML = opts
      .map(([v, t]) =>
        `<option value="${esc(v)}">${esc(t)}</option>`
      )
      .join("");

    // Prefer the configured model if installed; otherwise use the baseline.
    sel.value = installed
      ? models.find((m) => normalizeModelName(m) === configured)
      : s.baseline;

  } catch (e) {
    console.error("loadStatus failed:", e);
    hint.textContent = "Server unreachable. Is the app running?";
    sel.innerHTML =
      '<option value="">Unable to load models</option>';
  }
}


  /* ------------------------------------------------------------- dashboard */
  async function renderDashboard() {
    const d = state.status?.dataset || (await api("/api/dataset"));
    $("#dashKpis").innerHTML = [
      kpi("Synthetic training examples", num(d.synthetic_train), `${num(d.synthetic_val)} val · ${num(d.synthetic_test)} held-out test`),
      kpi("CMS documents retrieved", num(d.documents_retrieved), d.collection_last_run ? `last run ${esc(d.collection_last_run.slice(0, 10))}` : "collector not run yet"),
      kpi("Verified real pairs", num(d.pairs_verified), `${num(d.pairs_unverified)} unverified · ${num(d.pairs_unavailable)} unavailable`),
      kpi("Real examples awaiting review", num(d.real_pending_review), `${num(d.real_approved)} approved`),
      kpi("Training set size", num(d.training_train), `${num(d.training_val)} validation`),
    ].join("");
    const b = await loadBenchmarks();
    const rows = b.results;
    $("#dashBench").innerHTML = rows.length ? `<div class="table-wrap"><table><thead><tr><th>Eval set</th><th>Model</th><th class="num">n</th><th class="num">Class. accuracy</th><th class="num">Evidence match</th></tr></thead><tbody>${
      rows.map((r) => `<tr><td>${esc(r.eval_set)}</td><td>${esc(r.model)}</td><td class="num">${num(r.metrics.n)}</td><td class="num">${pct(r.metrics.classification_accuracy)}</td><td class="num">${pct(r.metrics.evidence_exact_match_rate)}</td></tr>`).join("")
    }</tbody></table></div>` : empty("No benchmark results yet", `Run <code>python -m careimpact.cli evaluate --model diff-baseline</code>`);
  }

  /* --------------------------------------------------------------- library */
  async function loadLibrary(force = false) {
    if (!state.library || force) state.library = await api("/api/library");
    state.docs = Object.fromEntries(state.library.documents.map((d) => [d.key, d]));
    const groups = { sample: [], real: [] };
    state.library.documents.forEach((d) => (d.data_origin === "synthetic" ? groups.sample : groups.real).push(d));
    const opt = (d) => `<option value="${esc(d.key)}">${esc(d.doc_id)} v${esc(d.version)} — ${esc(d.title)}</option>`;
    const html = `<option value="">Select a document…</option>` +
      (groups.sample.length ? `<optgroup label="Synthetic samples (fictional)">${groups.sample.map(opt).join("")}</optgroup>` : "") +
      (groups.real.length ? `<optgroup label="Collected CMS documents">${groups.real.map(opt).join("")}</optgroup>` : "");
    $$(".doc-select").forEach((s) => { const v = s.value; s.innerHTML = html; s.value = v; });
  }

  function statusBadge(st) {
    const cls = { verified: "ok", synthetic: "info", unverified: "warn", unavailable: "bad" }[st] || "";
    const icon = { verified: "✓", synthetic: "◇", unverified: "!", unavailable: "×" }[st] || "";
    return `<span class="badge ${cls}">${icon} ${esc(st)}</span>`;
  }

  async function renderLibrary() {
    await loadLibrary(true);
    const f = $("#libFilter").value;
    const pairs = state.library.pairs.filter((p) => f === "all" || p.kind === f);
    if (!pairs.length) {
      $("#libTable").innerHTML = empty("No policy pairs yet", f === "real"
        ? `Collect real pairs with <code>python -m careimpact.cli collect --limit 10</code>`
        : `Create demo samples with <code>python -m careimpact.cli samples</code>`);
      return;
    }
    $("#libTable").innerHTML = `<div class="table-wrap"><table><thead><tr><th>Policy</th><th>Title</th><th>Versions</th><th>Status</th><th>Notes</th><th></th></tr></thead><tbody>${
      pairs.map((p) => {
        const can = p.old_key && p.new_key;
        const note = p.kind === "sample" ? `Synthetic scenario: ${esc(label(p.scenario))}` :
          (p.identical_content ? "Policy content identical between versions" : esc((p.reasons || []).slice(0, 2).join("; ")));
        return `<tr><td>${esc(p.doc_id)}</td><td>${esc(p.title)}</td><td>${esc(p.old_version ?? "—")} → ${esc(p.new_version)}</td>
          <td>${statusBadge(p.status)}</td><td class="muted small">${note || "—"}</td>
          <td>${can ? `<button class="btn small" data-old="${esc(p.old_key)}" data-new="${esc(p.new_key)}">Compare</button>` : `<span class="muted small">Not comparable</span>`}</td></tr>`;
      }).join("")
    }</tbody></table></div>`;
    $$("#libTable button[data-old]").forEach((b) => b.addEventListener("click", () => {
      location.hash = "#compare";
      setSaved("old", b.dataset.old);
      setSaved("new", b.dataset.new);
    }));
  }
  $("#libFilter").addEventListener("change", renderLibrary);

  async function firstSample() {
    await loadLibrary(!state.library);
    return state.library.pairs.find((p) => p.kind === "sample" && p.old_key && p.new_key) ||
      state.library.pairs.find((p) => p.old_key && p.new_key);
  }
  async function loadSamplePair(run) {
    try {
      const p = await firstSample();
      if (!p) { toast("No sample pairs yet. Run: python -m careimpact.cli samples"); return; }
      location.hash = "#compare";
      setSaved("old", p.old_key); setSaved("new", p.new_key);
      if (run) $("#compareBtn").click(); else toast(`Loaded ${p.doc_id}: v${p.old_version} → v${p.new_version}`);
    } catch (e) { toast(`Could not load samples: ${e.message}`); }
  }
  $("#trySample").addEventListener("click", () => loadSamplePair(true));
  $("#loadSample").addEventListener("click", () => loadSamplePair(false));
  $("#askQuick").addEventListener("click", () => window.CareImpactChat?.open());

  /* --------------------------------------------------------------- compare */
  function panel(side) { return $(`.panel[data-side="${side}"]`); }
  function mode(side) { return $(".tabs [aria-selected='true']", panel(side)).dataset.mode; }
  function setMode(side, m) {
    const p = panel(side);
    $$(".tabs button", p).forEach((b) => b.setAttribute("aria-selected", String(b.dataset.mode === m)));
    $$(".src", p).forEach((el) => (el.hidden = !el.classList.contains(`src-${m}`)));
    showMeta(side);
  }
  function setSaved(side, key) {
    setMode(side, "saved");
    $(".doc-select", panel(side)).value = key;
    showMeta(side);
  }
  function showMeta(side) {
    const p = panel(side);
    const dl = $(".meta", p);
    if (mode(side) !== "saved") { dl.innerHTML = `<dt>Retrieval</dt><dd>${mode(side) === "url" ? "Fetched from CMS on compare and validated" : "User-supplied text · unverified"}</dd>`; return; }
    const d = state.docs[$(".doc-select", p).value];
    if (!d) { dl.innerHTML = ""; return; }
    dl.innerHTML = [
      ["Policy ID", esc(d.doc_id)], ["Title", esc(d.title)], ["Version", esc(d.version ?? "—")],
      ["Effective", esc(d.revision_effective_date ?? "unknown")], ["Status", esc(d.status ?? "unknown")],
      ["Origin", d.data_origin === "synthetic" ? `<span class="badge info">◇ synthetic · fictional</span>` : esc(d.data_origin)],
      ["Retrieval", statusBadge(d.verification_status)],
      ...(d.source_url ? [["Source", `<a href="${esc(d.source_url)}" target="_blank" rel="noopener">${esc(d.source_url)}</a>`]] : []),
    ].map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
  }
  $$(".panel").forEach((p) => {
    const side = p.dataset.side;
    $$(".tabs button", p).forEach((b) => b.addEventListener("click", () => setMode(side, b.dataset.mode)));
    $(".tabs", p).addEventListener("keydown", (e) => {
      if (!["ArrowLeft", "ArrowRight"].includes(e.key)) return;
      const btns = $$(".tabs button", p);
      const i = btns.findIndex((b) => b.getAttribute("aria-selected") === "true");
      const n = btns[(i + (e.key === "ArrowRight" ? 1 : btns.length - 1)) % btns.length];
      setMode(side, n.dataset.mode); n.focus();
    });
    $(".doc-select", p).addEventListener("change", () => showMeta(side));
  });

  function readSide(side) {
    const p = panel(side);
    return { mode: mode(side), saved: $(".doc-select", p).value, url: $(".url-input", p).value, text: $(".text-input", p).value };
  }
  function writeSide(side, v) {
    const p = panel(side);
    $(".doc-select", p).value = v.saved; $(".url-input", p).value = v.url; $(".text-input", p).value = v.text;
    setMode(side, v.mode);
  }
  $("#swapSides").addEventListener("click", () => {
    const o = readSide("old"), n = readSide("new");
    writeSide("old", n); writeSide("new", o);
    toast("Swapped old and new versions");
  });
  $("#clearAll").addEventListener("click", () => {
    ["old", "new"].forEach((side) => writeSide(side, { mode: "saved", saved: "", url: "", text: "" }));
    $("#results").innerHTML = "";
    window.CareImpact.lastResult = null;
  });

  function sourceOf(side) {
    const p = panel(side), m = mode(side);
    if (m === "saved") { const k = $(".doc-select", p).value; if (!k) throw new Error(`Select a saved ${side} document.`); return { document_key: k }; }
    if (m === "url") { const u = $(".url-input", p).value.trim(); if (!u) throw new Error(`Enter the ${side} MCD URL.`); return { url: u }; }
    const t = $(".text-input", p).value; if (!t.trim()) throw new Error(`Paste the ${side} policy text.`); return { text: t };
  }

  $("#compareBtn").addEventListener("click", async () => {
    const out = $("#results");
    let body;
    try { body = { old: sourceOf("old"), new: sourceOf("new"), model: $("#modelSelect").value || null }; }
    catch (e) { out.innerHTML = `<div class="notice error">${esc(e.message)}</div>`; return; }
    const btn = $("#compareBtn");
    btn.disabled = true;
    const usesModel = body.model && body.model !== state.status?.baseline;
    out.innerHTML = `<div class="card loading"><div class="spinner" aria-hidden="true"></div><div>Retrieving versions and computing the section diff${usesModel ? `, then analysing with <strong>${esc(body.model)}</strong> (CPU inference can take a minute)` : ""}…</div></div>`;
    try { const r = await api("/api/compare", { method: "POST", body: JSON.stringify(body) }); renderResults(r); window.CareImpact.lastResult = summarize(r); }
    catch (e) { out.innerHTML = `<div class="notice error"><strong>Comparison failed.</strong> ${esc(e.message)}</div>`; }
    finally { btn.disabled = false; }
  });

  // Plain-text digest of a comparison, shared with the chatbot as context.
  function summarize(r) {
    const a = r.analysis, d = r.documents;
    const lines = [`Policy ${d.old.doc_id} v${d.old.version} → v${d.new.version}${d.new.title ? ` (${d.new.title})` : ""}`,
      `Classification: ${label(a.change_classification)}${a.is_substantive ? " (substantive)" : ""}`,
      `Summary: ${a.change_summary}`,
      ...a.findings.slice(0, 8).map((f, i) => `Finding ${i + 1} [${label(f.category)}] ${f.section}: ${f.what_changed}`),
      `Evidence quotes matched: ${r.grounding.quotes_matched}/${r.grounding.quotes}`,
      ...r.warnings.map((w) => `Warning: ${w}`)];
    return lines.join("\n");
  }

  function renderResults(r) {
    const a = r.analysis, g = r.grounding, docs = r.documents;
    const issuesBy = {};
    g.issues.forEach((i) => { if (i.finding !== undefined) (issuesBy[i.finding] ||= []).push(i); });
    const subst = a.findings.filter((f) => !["administrative", "no_substantive_change"].includes(f.category));
    const count = (cats) => a.findings.filter((f) => cats.includes(f.category)).length;
    const changed = r.diff.sections.filter((s) => s.status !== "unchanged").length;
    const warnings = r.warnings.length ? `<div class="notice warn"><strong>Check before relying on this result</strong><ul>${r.warnings.map((w) => `<li>${esc(w)}</li>`).join("")}</ul></div>` : "";
    const clsBadge = a.abstain ? `<span class="badge warn">! ${esc(label(a.change_classification))}</span>` :
      `<span class="badge ${a.is_substantive ? "info" : ""}">${esc(label(a.change_classification))}</span>`;

    const findings = a.findings.length ? a.findings.map((f, i) => {
      const iss = issuesBy[i] || [];
      const bad = (side) => iss.some((x) => x.type === "unsupported_quote" && x.side === side);
      const q = (side, text, ver) => text
        ? `<div class="quote ${side}"><div class="q-label"><span>${side === "old" ? "Old" : "New"} · v${esc(ver ?? "?")}</span>${bad(side) ? `<span class="badge bad">⚠ not found in source</span>` : `<span class="badge ok">✓ found in source</span>`}</div><blockquote>${esc(text)}</blockquote></div>`
        : `<div class="quote ${side} none"><div class="q-label"><span>${side === "old" ? "Old" : "New"} · v${esc(ver ?? "?")}</span></div><blockquote>No ${side === "old" ? "previous" : "new"} text (${side === "old" ? "added" : "removed"})</blockquote></div>`;
      return `<article class="finding">
        <div class="finding-head"><h3>${esc(f.section)}</h3>
          <span class="badge">${esc(label(f.category))}</span><span class="badge">${esc(f.change_type)}</span>
          <span class="badge" title="Qualitative label, not a calibrated probability">confidence: ${esc(f.confidence)}</span>
          ${f.requires_verification ? `<span class="badge warn">! verify</span>` : ""}
          ${iss.some((x) => x.type === "section_not_changed") ? `<span class="badge bad">⚠ section not changed in diff</span>` : ""}</div>
        <p class="finding-text">${esc(f.what_changed)}</p>
        <div class="evidence">${q("old", f.evidence_old, docs.old.version)}${q("new", f.evidence_new, docs.new.version)}</div>
        <dl class="kv"><dt>Potential impact <span class="muted">(inferred)</span></dt><dd>${esc(f.potential_operational_impact)}</dd>
          <dt>Recommended action</dt><dd>${esc(f.recommended_action)}</dd>
          <dt>Impact areas</dt><dd>${f.impact_areas.length ? f.impact_areas.map((x) => esc(label(x))).join(", ") : "None identified"}</dd></dl>
      </article>`;
    }).join("") : empty(a.abstain ? "The analyzer abstained" : "No findings", `<span class="small">${esc(a.change_summary)}</span>`);

    $("#results").innerHTML = `
      ${warnings}
      <div class="card">
        <div class="card-head"><h2>Change summary</h2><span class="muted small">Analyzer: ${esc(r.predictor)}${r.latency_s ? ` · ${r.latency_s.toFixed(1)} s` : ""}</span></div>
        <p class="summary-text">${esc(a.change_summary)}</p>
      </div>
      <div class="grid summary-cards">
        ${kpi("Classification", clsBadge, a.is_substantive ? "Substantive change" : "Not substantive")}
        ${kpi("Changed sections", num(changed), `${num(r.diff.sections.length)} sections compared`)}
        ${kpi("Coverage & clinical", num(count(["coverage", "clinical_criteria", "retirement_replacement"])), "findings")}
        ${kpi("Documentation & coding", num(count(["documentation", "coding_billing"])), "findings")}
        ${kpi("Provider & administrative", num(count(["provider_requirements", "administrative"])), "findings")}
        ${kpi("Evidence quotes matched", g.quotes ? `${g.quotes_matched}/${g.quotes}` : "—", "exact match to source text")}
      </div>
      <div class="card"><h2>Findings (${a.findings.length}${subst.length !== a.findings.length ? ` · ${subst.length} substantive` : ""})</h2><div class="findings">${findings}</div></div>
      <div class="card"><h2>Limitations &amp; uncertainty</h2><ul>${a.limitations.map((l) => `<li>${esc(l)}</li>`).join("") || "<li>None stated.</li>"}</ul>
        <p class="muted small">${esc(r.disclaimer)}</p></div>
      <div class="card"><div class="card-head"><h2>Side-by-side diff</h2>
        <div><button class="btn small" id="expandAll">Expand all</button> <button class="btn small" id="collapseAll">Collapse all</button></div></div>
        <div class="legend"><span><b class="l-del">−</b>Removed</span><span><b class="l-add">+</b>Added</span><span><b class="l-mod">~</b>Modified</span></div>
        ${renderDiff(r.diff)}</div>`;
    $("#expandAll").onclick = () => $$("#results details").forEach((d) => (d.open = true));
    $("#collapseAll").onclick = () => $$("#results details").forEach((d) => (d.open = false));
    $("#results").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function renderDiff(diff) {
    const statusB = { modified: `<span class="badge warn">~ modified</span>`, added: `<span class="badge ok">+ added</span>`, removed: `<span class="badge bad">− removed</span>`, formatting_only: `<span class="badge">formatting only</span>`, unchanged: `<span class="badge">unchanged</span>` };
    const sr = (t) => `<span class="sr-only">${t}: </span>`;
    return diff.sections.map((s) => {
      const L = [], R = [];
      s.ops.forEach((op, i) => {
        if (op.op === "equal") {
          const units = op.old;
          const show = units.length > 4 && s.status !== "unchanged" ? [units[0], null, units[units.length - 1]] : units;
          show.forEach((u) => {
            const html = u === null ? `<div class="u gap">… ${units.length - 2} unchanged sentences …</div>` : `<div class="u eq">${esc(u)}</div>`;
            L.push(html); R.push(html);
          });
        } else if (op.op === "delete") op.old.forEach((u) => L.push(`<div class="u del">${sr("Removed")}${esc(u)}</div>`));
        else if (op.op === "insert") op.new.forEach((u) => R.push(`<div class="u ins">${sr("Added")}${esc(u)}</div>`));
        else { op.old.forEach((u) => L.push(`<div class="u mod">${sr("Modified, old")}${esc(u)}</div>`)); op.new.forEach((u) => R.push(`<div class="u mod">${sr("Modified, new")}${esc(u)}</div>`)); }
      });
      const open = s.status !== "unchanged" ? " open" : "";
      const heading = s.heading_old && s.heading_new && s.heading_old !== s.heading_new ? `${esc(s.heading_old)} → ${esc(s.heading_new)}` : esc(s.heading_new || s.heading_old);
      return `<details class="diff-section"${open}><summary><span class="title">${heading}</span>${statusB[s.status] || ""}<span class="muted small">similarity ${(s.similarity * 100).toFixed(0)}%</span></summary>
        <div class="diff-cols"><div class="diff-col"><h4>Old</h4>${L.join("") || `<div class="u gap">Section not present</div>`}</div>
        <div class="diff-col"><h4>New</h4>${R.join("") || `<div class="u gap">Section not present</div>`}</div></div></details>`;
    }).join("");
  }

  /* --------------------------------------------------------------- dataset */
  function bars(obj, fmt = (k) => label(k)) {
    const entries = Object.entries(obj || {}).sort((a, b) => b[1] - a[1]);
    if (!entries.length) return empty("No data yet", `Run <code>python -m careimpact.cli synth-all</code>`);
    const max = Math.max(...entries.map((e) => e[1]));
    return `<div class="bars" role="table" aria-label="Distribution">${entries.map(([k, v]) =>
      `<div class="bar-row" role="row" title="${esc(fmt(k))}: ${v}"><span class="name" role="cell">${esc(fmt(k))}</span><span class="bar-track" aria-hidden="true"><span class="bar-fill" style="width:${(v / max) * 100}%"></span></span><span class="v" role="cell">${num(v)}</span></div>`).join("")}</div>`;
  }

  async function renderDataset() {
    const [d, t] = await Promise.all([api("/api/dataset"), api("/api/training")]);
    $("#dataKpis").innerHTML = [
      kpi("Documents retrieved", num(d.documents_retrieved), "official MCD pages"),
      kpi("Verified pairs", num(d.pairs_verified), `${num(d.pairs_identical_content)} with identical content`),
      kpi("Unverified / unavailable", `${num(d.pairs_unverified)} / ${num(d.pairs_unavailable)}`, "never used as confirmed changes"),
      kpi("Real vs synthetic (train)", `${num(d.training_manifest?.train_real ?? 0)} / ${num(d.training_manifest?.train_synthetic ?? 0)}`, "approved real / auto-validated synthetic"),
      kpi("Awaiting review", num(d.real_pending_review), `${num(d.real_approved)} approved · ${num(d.real_rejected)} rejected`),
      kpi("Failed quality checks", num(d.synthetic_failed_checks), "synthetic, routed to review"),
      kpi("Training set", num(d.training_train), `${num(d.training_val)} val · generator ${esc(d.synthetic_generator_version ?? "—")}`),
    ].join("");
    $("#scenarioBars").innerHTML = bars(d.synthetic_by_scenario);
    $("#classBars").innerHTML = bars(d.synthetic_by_classification);
    $("#runs").innerHTML = t.runs.length ? `<div class="table-wrap"><table><thead><tr><th>Run</th><th>Base model</th><th>Method</th><th class="num">Examples</th><th class="num">Steps</th><th class="num">Final train loss</th><th class="num">Val loss</th><th class="num">s / step</th><th class="num">Peak RAM</th><th>Status</th></tr></thead><tbody>${
      t.runs.map((r) => `<tr><td>${esc(r.run_dir)}</td><td>${esc(r.base_model)}</td><td>${esc(r.method)}</td><td class="num">${num(r.train_examples)}</td><td class="num">${num(r.steps)}</td><td class="num">${r.final_train_loss?.toFixed?.(4) ?? "—"}</td><td class="num">${r.val_loss?.toFixed?.(4) ?? "—"}</td><td class="num">${r.seconds_per_step?.toFixed?.(1) ?? "—"}</td><td class="num">${r.peak_rss_gb ? r.peak_rss_gb.toFixed(1) + " GB" : "—"}</td><td>${esc(r.status)}</td></tr>`).join("")
    }</tbody></table></div>` : empty("No training runs recorded", `Run <code>python -m careimpact.training.train_lora --config configs/training/cpu_smoke.yaml</code>`);
  }

  /* ------------------------------------------------------------ benchmarks */
  async function loadBenchmarks() { state.benchmarks = await api("/api/benchmarks"); return state.benchmarks; }
  const METRICS = [
    ["classification_accuracy", "Classification accuracy", true], ["substantive_f1", "Substantive change F1", true],
    ["finding_recall", "Finding recall", true], ["missed_important_change_rate", "Missed important changes", false],
    ["finding_category_accuracy", "Finding category accuracy", true], ["unsupported_claim_rate", "Unsupported claim rate", false],
    ["evidence_exact_match_rate", "Evidence exact-match", true], ["schema_valid_rate", "Valid JSON (schema)", true],
    ["section_coverage", "Section coverage", true], ["no_change_accuracy", "No-change handling", true],
    ["abstention_recall", "Abstention recall", true], ["false_abstention_rate", "False abstention", false],
    ["latency_mean_s", "Mean latency (s)", false], ["model_memory_bytes", "Model memory", false],
  ];
  async function renderBenchmarks() {
    const { results } = await loadBenchmarks();
    const sets = ["synthetic_test", "real_test"];
    $("#benchTables").innerHTML = sets.map((set) => {
      const rs = results.filter((r) => r.eval_set === set);
      const title = set === "synthetic_test" ? "Synthetic held-out families" : "Real CMS policy pairs (human-approved)";
      if (!rs.length) return `<div class="card"><h2>${title}</h2>${empty("No results for this evaluation set", set === "real_test" ? "Real evaluation requires collected pairs that a reviewer has approved (<code>collect</code> → <code>draft-real</code> → <code>review</code> → <code>build-dataset</code>)." : "Run <code>python -m careimpact.cli evaluate</code>")}</div>`;
      const fmt = (k, v) => v === null || v === undefined ? "—" : k === "latency_mean_s" ? v.toFixed(2) : k === "model_memory_bytes" ? `${(v / 1e9).toFixed(1)} GB` : pct(v);
      const body = METRICS.map(([k, name, higher]) => {
        const vals = rs.map((r) => r.metrics[k]);
        const valid = vals.filter((v) => v !== null && v !== undefined);
        const best = valid.length > 1 ? (higher ? Math.max(...valid) : Math.min(...valid)) : null;
        return `<tr><td>${name}${higher ? "" : ' <span class="muted small">(lower is better)</span>'}</td>${vals.map((v) => `<td class="num">${v === best ? `<span class="best" title="Best in row">★ ${fmt(k, v)}</span>` : fmt(k, v)}</td>`).join("")}</tr>`;
      }).join("");
      return `<div class="card"><div class="card-head"><h2>${title}</h2><span class="muted small">n = ${rs.map((r) => r.metrics.n).join(" / ")}</span></div>
        <div class="table-wrap"><table><thead><tr><th>Metric</th>${rs.map((r) => `<th class="num">${esc(r.model)}<br><span class="muted small">${esc(r.created_at.slice(0, 10))}</span></th>`).join("")}</tr></thead><tbody>${body}</tbody></table></div></div>`;
    }).join("");
    const models = [...new Set(results.filter((r) => r.eval_set === "synthetic_test").map((r) => r.model))];
    const sel = $("#failModel");
    sel.innerHTML = models.map((m) => `<option>${esc(m)}</option>`).join("");
    renderFailures();
  }
  async function renderFailures() {
    const m = $("#failModel").value;
    if (!m) { $("#failures").innerHTML = empty("No results yet", ""); return; }
    const f = await api(`/api/benchmarks/examples?eval_set=synthetic_test&model=${encodeURIComponent(m)}&kind=failures&limit=12`);
    $("#failures").innerHTML = f.rows.length ? `<div class="table-wrap"><table><thead><tr><th>Example</th><th>Scenario</th><th>Expected</th><th>Predicted</th><th>Notes</th></tr></thead><tbody>${
      f.rows.map((r) => `<tr><td class="small">${esc(r.example_id)}</td><td>${esc(label(r.scenario))}</td><td>${esc(label(r.gold_class))}</td><td>${esc(r.pred_class ? label(r.pred_class) : "invalid output")}</td><td class="muted small">${esc([(r.parse_errors || []).join("; "), r.unsupported_findings ? `${r.unsupported_findings} unsupported finding(s)` : "", r.gold_abstain && !r.pred_abstain ? "should have abstained" : ""].filter(Boolean).join(" · ") || "—")}</td></tr>`).join("")
    }</tbody></table></div>` : empty("No failures recorded for this model", "");
  }
  $("#failModel").addEventListener("change", renderFailures);

  /* ------------------------------------------------------------------ init */
  window.CareImpact = { lastResult: null, toast };
  window.addEventListener("hashchange", route);
  // Show the page immediately; the model status check can be slow when Ollama is offline.
  route();
  loadStatus();
  loadLibrary().catch(() => {});
})();
