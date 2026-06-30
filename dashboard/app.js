"use strict";

const $ = (id) => document.getElementById(id);
const api = (path, method = "GET", body = null) =>
  fetch(path, {
    method,
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  }).then((r) => r.json());

const fmtMoney = (v) =>
  v == null ? "—" : "$" + Number(v).toLocaleString(undefined, { maximumFractionDigits: 0 });
const fmtPct = (v) => (v == null ? "—" : (Number(v) * 100).toFixed(2) + "%");
const fmtNum = (v) => (v == null ? "—" : Number(v).toLocaleString());

function metric(label, value, cls = "") {
  return `<div class="metric"><div class="label">${label}</div><div class="value ${cls}">${value}</div></div>`;
}
function signClass(v) {
  if (v == null) return "";
  return v > 0 ? "pos" : v < 0 ? "neg" : "";
}

// ---- controls ----
$("btn-start").onclick = () => api("/api/start", "POST");
$("btn-pause").onclick = () => api("/api/pause", "POST");
$("btn-reset").onclick = () => api("/api/reset", "POST").then(refresh);
$("btn-step").onclick = () => api("/api/step", "POST").then(refresh);
$("btn-speed").onclick = () =>
  api("/api/speed", "POST", { seconds_per_month: parseFloat($("speed").value) || 30 });

// ---- chart (self-contained canvas line chart) ----
function drawChart(series) {
  const canvas = $("chart");
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || 700;
  const h = 260;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);

  const port = series.portfolio_value || [];
  const bench = series.benchmark_value || [];
  const pad = { l: 56, r: 12, t: 12, b: 24 };
  const all = port.concat(bench);
  if (all.length === 0) {
    ctx.fillStyle = "#9aa4b5";
    ctx.font = "13px -apple-system, sans-serif";
    ctx.fillText("No data yet — press Start.", pad.l, h / 2);
    return;
  }
  const min = Math.min(...all);
  const max = Math.max(...all);
  const range = max - min || 1;
  const n = Math.max(port.length, bench.length, 2);
  const x = (i) => pad.l + (i / (n - 1)) * (w - pad.l - pad.r);
  const y = (v) => pad.t + (1 - (v - min) / range) * (h - pad.t - pad.b);

  // gridlines + y labels
  ctx.strokeStyle = "#eef1f6";
  ctx.fillStyle = "#9aa4b5";
  ctx.font = "10px -apple-system, sans-serif";
  ctx.lineWidth = 1;
  for (let g = 0; g <= 4; g++) {
    const val = min + (range * g) / 4;
    const yy = y(val);
    ctx.beginPath();
    ctx.moveTo(pad.l, yy);
    ctx.lineTo(w - pad.r, yy);
    ctx.stroke();
    ctx.fillText("$" + Math.round(val).toLocaleString(), 6, yy + 3);
  }

  const line = (data, color, width) => {
    if (data.length < 1) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.beginPath();
    data.forEach((v, i) => (i === 0 ? ctx.moveTo(x(i), y(v)) : ctx.lineTo(x(i), y(v))));
    ctx.stroke();
  };
  line(bench, "#94a3b8", 1.5);
  line(port, "#2563eb", 2);
}

// ---- renderers ----
function renderPills(s) {
  const c = s.controls || {};
  const dataPill = $("pill-data");
  dataPill.textContent = "data: " + (c.data_ready ? "ready" : "not ready");
  dataPill.className = "pill " + (c.data_ready ? "on" : "off");

  const llm = s.llm || {};
  const llmPill = $("pill-llm");
  llmPill.textContent = "llm: " + (llm.enabled ? llm.model : "fallback");
  llmPill.className = "pill " + (llm.enabled ? "on" : "subtle");

  const ls = s.langsmith || {};
  const lsPill = $("pill-langsmith");
  lsPill.textContent = "langsmith: " + (ls.enabled ? "on" : "off");
  lsPill.className = "pill " + (ls.enabled ? "on" : "subtle");
}

function renderControls(c) {
  const pct = c.total_months ? (c.month_index / c.total_months) * 100 : 0;
  $("progress-fill").style.width = pct + "%";
  $("progress-text").textContent = c.progress || `${c.month_index} / ${c.total_months}`;
  $("current-date").textContent = c.current_date || "—";
  const st = $("run-state");
  if (c.finished) {
    st.textContent = "finished";
    st.className = "state finished";
  } else if (c.running) {
    st.textContent = "running";
    st.className = "state running";
  } else {
    st.textContent = "idle";
    st.className = "state idle";
  }
}

function renderFinancial(f) {
  $("financial-metrics").innerHTML =
    metric("Portfolio Value", fmtMoney(f.portfolio_value)) +
    metric("Benchmark Value", fmtMoney(f.benchmark_value)) +
    metric("Total Return", fmtPct(f.total_return), signClass(f.total_return)) +
    metric("Excess Return", fmtPct(f.excess_return), signClass(f.excess_return)) +
    metric("Benchmark Return", fmtPct(f.benchmark_return), signClass(f.benchmark_return)) +
    metric("Max Drawdown", fmtPct(f.max_drawdown), "neg") +
    metric("Avg Turnover", fmtPct(f.turnover_avg)) +
    metric("Txn Cost Drag", fmtPct(f.transaction_cost_drag), "warn");
}

function renderHoldings(s) {
  const latest = s.latest;
  const f = s.aggregates.financial;
  $("holdings-metrics").innerHTML =
    metric("Txn Count (total)", fmtNum(s.aggregates.agent.transaction_count)) +
    metric("Cash Weight", fmtPct(latest ? latest.cash_weight : null)) +
    metric("Max Position", fmtPct(latest ? latest.max_position_weight : null));

  const tbody = $("holdings-table").querySelector("tbody");
  const holdings = latest ? latest.holdings || {} : {};
  const rows = Object.entries(holdings)
    .filter(([k]) => k !== "CASH")
    .sort((a, b) => b[1] - a[1]);
  tbody.innerHTML = rows.length
    ? rows.map(([t, w]) => `<tr><td>${t}</td><td>${fmtPct(w)}</td></tr>`).join("")
    : `<tr><td colspan="2" class="empty">No holdings yet.</td></tr>`;

  const fs = $("forced-sells");
  const fsArr = latest ? latest.forced_sells || [] : [];
  fs.innerHTML = fsArr.length
    ? fsArr.map((t) => `<li class="removed">${t}</li>`).join("")
    : `<li class="empty">none</li>`;

  const ne = $("new-entrants");
  const neArr = latest ? latest.new_entrants_considered || [] : [];
  ne.innerHTML = neArr.length
    ? neArr.map((t) => `<li class="added">${t}</li>`).join("")
    : `<li class="empty">none</li>`;

  const tt = $("trades-table").querySelector("tbody");
  const trades = latest ? latest.trades || [] : [];
  tt.innerHTML = trades.length
    ? trades
        .map(
          (t) =>
            `<tr><td>${t.ticker}</td><td>${t.side}</td><td>$${Math.round(
              Math.abs(t.dollars)
            ).toLocaleString()}</td><td>${t.forced ? "✓" : ""}</td></tr>`
        )
        .join("")
    : `<tr><td colspan="4" class="empty">No trades this month.</td></tr>`;
}

function renderAgent(s) {
  const a = s.aggregates.agent;
  $("agent-metrics").innerHTML =
    metric("Risk Blocked", fmtNum(a.risk_blocked_count), a.risk_blocked_count ? "warn" : "") +
    metric("Risk Modified", fmtNum(a.risk_modified_count)) +
    metric("Violations", fmtNum(a.constraint_violation_count), a.constraint_violation_count ? "neg" : "") +
    metric("Invalid JSON", fmtNum(a.invalid_json_count), a.invalid_json_count ? "warn" : "") +
    metric("Retries", fmtNum(a.retry_count)) +
    metric("Same-Asset Flips", fmtNum(a.same_asset_flip_count));

  const r = $("rationale");
  const rat = s.latest ? s.latest.rationale || [] : [];
  r.innerHTML = rat.length
    ? rat.map((x) => `<li>${escapeHtml(x)}</li>`).join("")
    : `<li class="empty">No decision yet.</li>`;
  $("agent-source").textContent = "source: " + (s.latest ? s.latest.agent_source : "—");
}

function renderJudge(s) {
  const j = s.aggregates.judge;
  $("judge-metrics").innerHTML =
    metric("Groundedness", j.groundedness_score == null ? "—" : j.groundedness_score) +
    metric("Hallucination", j.hallucination_risk == null ? "—" : j.hallucination_risk,
      (j.hallucination_risk || 0) > 0.3 ? "neg" : "") +
    metric("Consistency", j.decision_consistency_score == null ? "—" : j.decision_consistency_score) +
    metric("Constraint Aware", j.constraint_awareness_score == null ? "—" : j.constraint_awareness_score) +
    metric("Memory Use", j.memory_use_score == null ? "—" : j.memory_use_score) +
    metric("Unsupported Claims", fmtNum(j.unsupported_claim_count), j.unsupported_claim_count ? "warn" : "");

  const issues = (s.latest && s.latest.judge && s.latest.judge.issues) || [];
  const ul = $("judge-issues");
  ul.innerHTML = issues.length
    ? issues
        .map(
          (i) =>
            `<li class="sev-${i.severity}"><span class="sev">${i.severity}</span>${escapeHtml(
              i.message
            )}</li>`
        )
        .join("")
    : `<li class="empty">No issues flagged for the latest decision.</li>`;
}

function renderMemory(s) {
  const m = s.memory || {};
  const j = s.aggregates.judge;
  $("memory-metrics").innerHTML =
    metric("Total Memories", fmtNum(m.total)) +
    metric("Active", fmtNum(m.active)) +
    metric("Stale", fmtNum(m.stale), m.stale ? "warn" : "") +
    metric("Judge Warnings", fmtNum(j.judge_warning_count), j.judge_warning_count ? "warn" : "") +
    metric("Judge Critical", fmtNum(j.judge_critical_count), j.judge_critical_count ? "neg" : "") +
    metric("Mem Use Score", j.memory_use_score == null ? "—" : j.memory_use_score);
}

function renderTimeline(alerts) {
  const ul = $("timeline");
  ul.innerHTML = alerts && alerts.length
    ? alerts
        .map(
          (a) =>
            `<li><span class="when">${a.date}</span><span class="badge ${a.severity}">${a.severity}</span><span>${escapeHtml(
              a.message
            )}</span></li>`
        )
        .join("")
    : `<li class="empty">No incidents recorded.</li>`;
}

function escapeHtml(str) {
  return String(str || "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
  );
}

// ---- main refresh ----
async function refresh() {
  try {
    const s = await api("/api/state");
    renderPills(s);
    renderControls(s.controls || {});
    if (s.error === "data_not_ready") {
      $("financial-metrics").innerHTML =
        `<div class="metric" style="grid-column:1/-1"><div class="label">Setup required</div><div class="value" style="font-size:14px">${s.message}</div></div>`;
      return;
    }
    if (s.aggregates) {
      renderFinancial(s.aggregates.financial);
      renderHoldings(s);
      renderAgent(s);
      renderJudge(s);
      renderMemory(s);
    }
    drawChart(s.series || {});
    renderTimeline(s.alerts || []);
  } catch (e) {
    /* server not up yet */
  }
}

refresh();
setInterval(refresh, 1000);
window.addEventListener("resize", refresh);
