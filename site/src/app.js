const DATA_DIR = "./public/data";

const state = {
  manifest: null,
  methods: [],
  maneuvers: [],
  scenario: "open_loop",
  source: "direct",
};

const fmt = new Intl.NumberFormat("en-US", { maximumSignificantDigits: 3 });

function finiteNumber(value) {
  return typeof value === "number" && Number.isFinite(value);
}

function formatNumber(value, fallback = "--") {
  return finiteNumber(value) ? fmt.format(value) : fallback;
}

function cleanMethodName(method) {
  return String(method || "").replace(" (mocap)", "");
}

async function loadJson(name) {
  const response = await fetch(`${DATA_DIR}/${name}`);
  if (!response.ok) {
    throw new Error(`failed to load ${name}: ${response.status}`);
  }
  return response.json();
}

function selectedRows() {
  return state.methods
    .filter((row) => row.scenario === state.scenario && row.state_source === state.source)
    .filter((row) => finiteNumber(row.validation_score))
    .sort((a, b) => a.validation_score - b.validation_score);
}

function scenarioTitle() {
  return state.manifest.scenarios.find((scenario) => scenario.id === state.scenario)?.title || state.scenario;
}

function matchingManeuver() {
  const title = scenarioTitle();
  return state.maneuvers.find((row) => row.mode === title) || null;
}

function renderControls() {
  const select = document.querySelector("#scenario-select");
  select.innerHTML = "";
  for (const scenario of state.manifest.scenarios) {
    const option = document.createElement("option");
    option.value = scenario.id;
    option.textContent = scenario.title;
    select.append(option);
  }
  select.value = state.scenario;
  select.addEventListener("change", () => {
    state.scenario = select.value;
    render();
  });

  document.querySelector("#source-filter").addEventListener("click", (event) => {
    const button = event.target.closest("button[data-source]");
    if (!button) return;
    state.source = button.dataset.source;
    for (const node of document.querySelectorAll("#source-filter button")) {
      node.classList.toggle("active", node.dataset.source === state.source);
    }
    render();
  });
}

function renderMeta() {
  const sha = state.manifest.git_sha ? state.manifest.git_sha.slice(0, 7) : "unknown";
  const generated = new Date(state.manifest.generated_at).toLocaleString();
  document.querySelector("#run-meta").textContent = `schema ${state.manifest.schema_version} | ${sha} | ${generated}`;
}

function renderSummary(rows) {
  const best = rows[0];
  const maneuver = matchingManeuver();
  document.querySelector("#best-method").textContent = best ? cleanMethodName(best.method) : "--";
  document.querySelector("#best-score").textContent = best ? formatNumber(best.validation_score) : "--";
  document.querySelector("#method-count").textContent = String(rows.length);
  document.querySelector("#maneuver-envelope").textContent = maneuver
    ? `${formatNumber(maneuver.max_abs_alpha_deg)} deg alpha, ${formatNumber(maneuver.max_speed_mps)} m/s`
    : "--";
}

function logExtent(values) {
  const finite = values.filter((value) => finiteNumber(value) && value > 0);
  if (!finite.length) return [0.01, 1];
  const min = Math.min(...finite);
  const max = Math.max(...finite);
  return [min * 0.75, max * 1.4];
}

function logScale(value, min, max, start, end) {
  const logMin = Math.log10(min);
  const logMax = Math.log10(max);
  const t = (Math.log10(Math.max(value, min)) - logMin) / Math.max(logMax - logMin, 1e-9);
  return start + t * (end - start);
}

function powers(min, max) {
  const start = Math.floor(Math.log10(min));
  const end = Math.ceil(Math.log10(max));
  const out = [];
  for (let power = start; power <= end; power += 1) {
    out.push(10 ** power);
  }
  return out;
}

function renderTradeoff(rows) {
  const host = document.querySelector("#tradeoff-plot");
  host.innerHTML = "";
  const width = Math.max(host.clientWidth || 900, 640);
  const height = 430;
  const margin = { top: 20, right: 36, bottom: 58, left: 78 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const xExtent = logExtent(rows.map((row) => row.train_elapsed_s));
  const yExtent = logExtent(rows.map((row) => row.validation_score));
  const nominal = rows.find((row) => cleanMethodName(row.method) === "Nominal");
  const color = state.source === "direct" ? "var(--direct)" : "var(--mocap)";

  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("aria-label", `${scenarioTitle()} ${state.source} tradeoff`);

  const add = (tag, attrs, text) => {
    const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
    if (text !== undefined) node.textContent = text;
    svg.append(node);
    return node;
  };

  for (const tick of powers(...xExtent)) {
    const x = logScale(tick, xExtent[0], xExtent[1], margin.left, margin.left + plotWidth);
    add("line", { x1: x, y1: margin.top, x2: x, y2: margin.top + plotHeight, class: "grid-line" });
    add("text", { x, y: height - 28, "text-anchor": "middle", class: "tick" }, `10^${Math.round(Math.log10(tick))}`);
  }
  for (const tick of powers(...yExtent)) {
    const y = logScale(tick, yExtent[0], yExtent[1], margin.top + plotHeight, margin.top);
    add("line", { x1: margin.left, y1: y, x2: margin.left + plotWidth, y2: y, class: "grid-line" });
    add("text", { x: margin.left - 10, y: y + 4, "text-anchor": "end", class: "tick" }, `10^${Math.round(Math.log10(tick))}`);
  }

  add("rect", {
    x: margin.left,
    y: margin.top,
    width: plotWidth,
    height: plotHeight,
    fill: "none",
    stroke: "var(--line)",
  });
  add("text", { x: margin.left + plotWidth / 2, y: height - 8, "text-anchor": "middle", class: "axis-label" }, "training / solve time [s]");
  add("text", {
    x: 18,
    y: margin.top + plotHeight / 2,
    transform: `rotate(-90 18 ${margin.top + plotHeight / 2})`,
    "text-anchor": "middle",
    class: "axis-label",
  }, "validation score: mean state NRMSE");

  if (nominal && finiteNumber(nominal.validation_score)) {
    const y = logScale(nominal.validation_score, yExtent[0], yExtent[1], margin.top + plotHeight, margin.top);
    add("line", {
      x1: margin.left,
      y1: y,
      x2: margin.left + plotWidth,
      y2: y,
      stroke: "var(--nominal)",
      "stroke-dasharray": "6 4",
    });
    add("text", { x: margin.left + plotWidth - 4, y: y - 6, "text-anchor": "end", fill: "var(--nominal)", class: "tick" }, "Nominal baseline");
  }

  for (const row of rows) {
    if (!finiteNumber(row.train_elapsed_s) || row.train_elapsed_s <= 0) continue;
    const x = logScale(row.train_elapsed_s, xExtent[0], xExtent[1], margin.left, margin.left + plotWidth);
    const y = logScale(row.validation_score, yExtent[0], yExtent[1], margin.top + plotHeight, margin.top);
    const radius = Math.min(13, 5 + Math.sqrt(Math.max(row.rollout_elapsed_s || 0, 0)));
    const isNominal = cleanMethodName(row.method) === "Nominal";
    const circle = add("circle", {
      cx: x,
      cy: y,
      r: radius,
      fill: isNominal ? "white" : color,
      stroke: isNominal ? "var(--nominal)" : "#1d2430",
      "stroke-width": isNominal ? 1.8 : 1,
      opacity: 0.82,
    });
    const title = document.createElementNS("http://www.w3.org/2000/svg", "title");
    title.textContent = `${cleanMethodName(row.method)} | score ${formatNumber(row.validation_score)} | train ${formatNumber(row.train_elapsed_s)} s`;
    circle.append(title);
  }

  const labeled = rows.slice(0, 8).concat(rows.filter((row) => ["Nominal", "Frequency-Welch", "Frequency-Stitching"].includes(cleanMethodName(row.method))));
  const seen = new Set();
  for (const row of labeled) {
    const name = cleanMethodName(row.method);
    if (seen.has(name) || !finiteNumber(row.train_elapsed_s) || row.train_elapsed_s <= 0) continue;
    seen.add(name);
    const x = logScale(row.train_elapsed_s, xExtent[0], xExtent[1], margin.left, margin.left + plotWidth);
    const y = logScale(row.validation_score, yExtent[0], yExtent[1], margin.top + plotHeight, margin.top);
    add("text", { x: x + 8, y: y - 8, class: "point-label" }, name);
  }

  host.append(svg);
}

function renderLeaderboard(rows) {
  const body = document.querySelector("#leaderboard-body");
  body.innerHTML = "";
  for (const row of rows) {
    const tr = document.createElement("tr");
    const values = [
      cleanMethodName(row.method),
      formatNumber(row.validation_score),
      formatNumber(row.train_elapsed_s),
      formatNumber(row.rollout_elapsed_s),
      row.training_scenario || "--",
    ];
    for (const [index, value] of values.entries()) {
      const cell = document.createElement("td");
      cell.textContent = value;
      if (index > 0 && index < 4) cell.className = "numeric";
      tr.append(cell);
    }
    body.append(tr);
  }
}

function renderManeuver() {
  const maneuver = matchingManeuver();
  const list = document.querySelector("#maneuver-list");
  list.innerHTML = "";
  const rows = maneuver
    ? [
        ["Max |alpha|", `${formatNumber(maneuver.max_abs_alpha_deg)} deg`],
        ["Max |theta|", `${formatNumber(maneuver.max_abs_theta_deg)} deg`],
        ["Speed range", `${formatNumber(maneuver.min_speed_mps)}-${formatNumber(maneuver.max_speed_mps)} m/s`],
        ["Vertical extent", `${formatNumber(maneuver.vertical_extent_m)} m`],
      ]
    : [["No maneuver summary", "--"]];
  for (const [term, detail] of rows) {
    const dt = document.createElement("dt");
    const dd = document.createElement("dd");
    dt.textContent = term;
    dd.textContent = detail;
    list.append(dt, dd);
  }
}

function render() {
  const rows = selectedRows();
  renderSummary(rows);
  renderTradeoff(rows);
  renderLeaderboard(rows);
  renderManeuver();
}

async function init() {
  state.manifest = await loadJson("manifest.json");
  state.methods = await loadJson("method_results.json");
  state.maneuvers = await loadJson("maneuver_summary.json");
  state.scenario = state.manifest.scenarios[0]?.id || "open_loop";
  renderControls();
  renderMeta();
  render();
}

init().catch((error) => {
  console.error(error);
  document.querySelector("#run-meta").textContent = error.message;
});
