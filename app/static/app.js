"use strict";
const $ = (s) => document.querySelector(s);
const messagesEl = $("#messages"), inputEl = $("#input"), sendBtn = $("#send");
let history = [];   // [{role, content}] sent back to the server for follow-up questions
let busy = false;

// ---------- helpers ----------
const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

/** Tiny, safe markdown subset: paragraphs, bullets, quotes, **bold**, *italic*, `code`, and [n] citations. */
function renderMarkdown(text, nSources) {
  const inline = (s) => esc(s)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>")
    .replace(/[\[【](\d+)[\]】]/g, (_, n) => (+n >= 1 && +n <= nSources
      ? `<button type="button" class="cite" data-n="${n}" title="Show source ${n}">${n}</button>`
      : `<span class="cite bad" title="Source ${n} was not provided to the model">${n}</span>`));
  let html = "", list = false;
  const closeList = () => { if (list) { html += "</ul>"; list = false; } };
  for (const para of text.split(/\n{2,}/)) {
    const lines = para.split("\n");
    if (lines.every((l) => /^\s*[-*]\s+/.test(l))) {
      if (!list) { html += "<ul>"; list = true; }
      html += lines.map((l) => `<li>${inline(l.replace(/^\s*[-*]\s+/, ""))}</li>`).join("");
      continue;
    }
    closeList();
    html += lines[0].startsWith("> ")
      ? `<blockquote>${inline(para.replace(/^> ?/gm, ""))}</blockquote>`
      : `<p>${inline(para).replace(/\n/g, "<br>")}</p>`;
  }
  closeList();
  return html;
}

function scrollDown() { messagesEl.scrollTop = messagesEl.scrollHeight; }

function addUser(text) {
  $("#empty")?.remove();
  const el = document.createElement("div");
  el.className = "msg user";
  el.textContent = text;
  messagesEl.append(el);
  scrollDown();
}

function addAssistant() {
  const el = document.createElement("div");
  el.className = "msg assistant";
  el.innerHTML = '<div class="body cursor"></div><div class="sources"></div><div class="trace"></div>';
  messagesEl.append(el);
  return el;
}

function renderSources(el, sources) {
  const box = el.querySelector(".sources");
  box.innerHTML = sources.map((s) => {
    const where = [s.doc, s.page ? `p.${s.page}` : null, s.heading].filter(Boolean).join(" · ");
    return `<details class="src" id="${el.dataset.id}-s${s.n}">
      <summary><span class="n">${s.n}</span><span class="where">${esc(where)}</span><span class="score">${s.score.toFixed(2)}</span></summary>
      <div class="passage">${esc(s.text)}</div></details>`;
  }).join("");
}

function renderTrace(el, done) {
  const t = done.timings_ms || {};
  const bits = [];
  if (done.grounded) {
    bits.push(`${((t.total || 0) / 1000).toFixed(1)}s`);
    if (t.ttft) bits.push(`first token ${(t.ttft / 1000).toFixed(1)}s`);
    if (done.usage?.output_tokens) bits.push(`${done.usage.input_tokens}→${done.usage.output_tokens} tokens`);
    if (done.search_query && el.dataset.q !== done.search_query) bits.push(`searched: “${esc(done.search_query)}”`);
  } else {
    bits.push(`no passage above the relevance threshold (best ${done.top_dense ?? 0})`);
  }
  const warn = [];
  if (done.flags?.includes("invalid_citation")) warn.push("⚠ some citations point at sources that weren't provided");
  if (done.flags?.includes("truncated")) warn.push("⚠ answer was cut off");
  if (done.flags?.includes("model_refusal")) warn.push("⚠ the model declined to answer");
  el.querySelector(".trace").innerHTML =
    bits.map((b) => `<span>${b}</span>`).join("") + warn.map((w) => `<span class="warn">${w}</span>`).join("");
}

// Clicking a [n] chip opens + highlights that source.
messagesEl.addEventListener("click", (e) => {
  const chip = e.target.closest("button.cite");
  if (!chip) return;
  const msg = chip.closest(".msg");
  const src = document.getElementById(`${msg.dataset.id}-s${chip.dataset.n}`);
  if (!src) return;
  src.open = true;
  src.classList.add("flash");
  src.scrollIntoView({ block: "nearest", behavior: "smooth" });
  setTimeout(() => src.classList.remove("flash"), 1400);
});

// ---------- chat ----------
async function ask(question) {
  if (busy || !question.trim()) return;
  busy = true; sendBtn.disabled = true;
  addUser(question);
  const el = addAssistant();
  el.dataset.id = "m" + Date.now();
  el.dataset.q = question;
  const body = el.querySelector(".body");
  let text = "", sources = [], done = null;

  try {
    const res = await fetch("/api/chat", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, history: history.slice(-12) }),
    });
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || `HTTP ${res.status}`);

    const reader = res.body.getReader(), dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done: finished } = await reader.read();
      if (finished) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n\n")) >= 0) {
        const block = buf.slice(0, i); buf = buf.slice(i + 2);
        const ev = block.match(/^event: (.*)$/m)?.[1];
        const data = JSON.parse(block.match(/^data: (.*)$/m)?.[1] ?? "null");
        if (ev === "sources") { sources = data; renderSources(el, sources); }
        else if (ev === "token") { text += data; body.innerHTML = renderMarkdown(text, sources.length); scrollDown(); }
        else if (ev === "done") { done = data; }
        else if (ev === "error") { throw new Error(data); }
      }
    }
    if (done && !done.grounded) el.classList.add("refusal");
    if (done) renderTrace(el, done);
    // Mark which sources the answer actually cited.
    (done?.cited || []).forEach((n) => document.getElementById(`${el.dataset.id}-s${n}`)?.classList.add("used"));
    history.push({ role: "user", content: question }, { role: "assistant", content: text });
  } catch (err) {
    body.innerHTML = `<p class="error">${esc(err.message || "Something went wrong.")}</p>`;
  } finally {
    body.classList.remove("cursor");
    busy = false; sendBtn.disabled = false; inputEl.focus(); scrollDown();
  }
}

$("#composer").addEventListener("submit", (e) => {
  e.preventDefault();
  const q = inputEl.value; inputEl.value = ""; inputEl.style.height = "auto";
  ask(q);
});
inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#composer").requestSubmit(); }
});
inputEl.addEventListener("input", () => {
  inputEl.style.height = "auto"; inputEl.style.height = Math.min(inputEl.scrollHeight, 160) + "px";
});

// ---------- documents ----------
async function refreshDocs() {
  const docs = await (await fetch("/api/documents")).json();
  $("#doc-count").textContent = docs.length;
  $("#docs").innerHTML = docs.map((d) => `
    <li><span class="name" title="${esc(d.name)}">${esc(d.name)}</span>
        <span class="meta">${d.n_chunks} chunks</span>
        <button data-id="${d.id}" title="Remove" aria-label="Remove ${esc(d.name)}">×</button></li>`).join("");
  renderSuggestions(docs);
  const s = await (await fetch("/api/stats")).json();
  $("#stats").textContent = s.queries
    ? `${s.queries} queries · p50 ${s.latency_ms.p50 ?? "–"} ms · ${Math.round((s.refusal_rate || 0) * 100)}% declined`
    : "";
}

function renderSuggestions(docs) {
  const box = $("#suggestions");
  if (!box) return;
  const hasSamples = docs.some((d) => d.name === "remote_work_policy.md");
  box.innerHTML = hasSamples
    ? ["How many days a week can I work remotely?", "What should I do if I clicked a phishing link?", "How long does Orbit's battery last?", "What's the vacation allowance?"]
        .map((q) => `<button class="chip" type="button">${esc(q)}</button>`).join("")
    : "";
}

async function upload(files) {
  if (!files.length) return;
  const status = $("#upload-status");
  status.textContent = `Indexing ${files.length} file(s)…`;
  const form = new FormData();
  [...files].forEach((f) => form.append("files", f));
  try {
    const res = await fetch("/api/documents", { method: "POST", body: form });
    const results = await res.json();
    status.innerHTML = results.map((r) =>
      r.status === "error" ? `<div class="err">${esc(r.name)}: ${esc(r.detail)}</div>`
      : r.status === "duplicate" ? `<div>${esc(r.name)}: already indexed</div>`
      : `<div>${esc(r.name)}: ${r.chunks} chunks indexed</div>`).join("");
  } catch { status.innerHTML = '<div class="err">Upload failed.</div>'; }
  refreshDocs();
}

$("#file").addEventListener("change", (e) => { upload(e.target.files); e.target.value = ""; });
const drop = $("#drop");
["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => upload(e.dataTransfer.files));

$("#docs").addEventListener("click", async (e) => {
  const id = e.target.closest("button")?.dataset.id;
  if (id && confirm("Remove this document from the index?")) {
    await fetch(`/api/documents/${id}`, { method: "DELETE" });
    refreshDocs();
  }
});
$("#suggestions").addEventListener("click", (e) => { if (e.target.matches(".chip")) ask(e.target.textContent); });

refreshDocs();
setInterval(() => { if (!busy) refreshDocs().catch(() => {}); }, 30000);
