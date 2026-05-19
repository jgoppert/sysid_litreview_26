import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.165.0/build/three.module.js";

const DATA_DIR = "./public/data";
const AIRCRAFT_MODEL_SCALE = 0.6 / 2.2;

const state = {
  manifest: null,
  rows: [],
  maneuvers: [],
  playback: [],
  methodTraces: [],
  playbackScene: null,
  playbackPlaying: true,
  playbackSpeed: 1,
  playbackTimeS: 0,
  playbackLastMs: null,
  playbackScrubbing: false,
  playbackSegmentIndex: 0,
  playbackView: "animation",
  playbackFollow: true,
  selectedMethods: new Set(),
  tradeoffZoom: null,
  modelFamily: "aircraft6dof",
  scenario: "sportcub_mocap_4_17_26",
  source: "mocap",
};

const ENU_TO_THREE_QUAT = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), -Math.PI / 2);
const MESH_TO_BODY_FRD_QUAT = ENU_TO_THREE_QUAT.clone();
const fmt = new Intl.NumberFormat("en-US", { maximumSignificantDigits: 3 });

function finiteNumber(value) {
  return typeof value === "number" && Number.isFinite(value);
}

function formatNumber(value, fallback = "--") {
  return finiteNumber(value) ? fmt.format(value) : fallback;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function cleanMethodName(method) {
  return String(method || "").replace(" (mocap)", "").replace(" (direct)", "");
}

async function loadJson(name) {
  const response = await fetch(`${DATA_DIR}/${name}`);
  if (!response.ok) throw new Error(`failed to load ${name}: ${response.status}`);
  return response.json();
}

function allRows() {
  return state.rows;
}

function allScenarios() {
  return state.manifest.scenarios;
}

function allDatasets() {
  return state.manifest.dataset_registry;
}

function scenariosForModel() {
  return allScenarios().filter((scenario) => scenario.model_family === state.modelFamily);
}

function datasetsForModel() {
  return allDatasets().filter((dataset) => dataset.model_family === state.modelFamily);
}

function selectedRows() {
  return allRows()
    .filter((row) => row.scenario === state.scenario && row.state_source === state.source)
    .filter((row) => finiteNumber(row.validation_score))
    .sort((a, b) => a.validation_score - b.validation_score);
}

function methodKey(method) {
  return cleanMethodName(method);
}

function scenarioTitle(id = state.scenario) {
  return allScenarios().find((scenario) => scenario.id === id)?.title || id;
}

function matchingManeuver() {
  const title = scenarioTitle();
  return state.maneuvers.find((row) => row.mode === title) || null;
}

function selectedPlayback() {
  return state.playback.find((track) => track.id === state.scenario) || state.playback.find((track) => track.model_family === state.modelFamily) || null;
}

function activeSegment(track = selectedPlayback()) {
  const segments = track?.segments?.length ? track.segments : track ? [track] : [];
  if (!segments.length) return null;
  const index = clamp(state.playbackSegmentIndex, 0, segments.length - 1);
  return segments[index];
}

function traceSegmentForMethod(key) {
  const trace = state.methodTraces.find((item) =>
    item.scenario === state.scenario && item.state_source === state.source && methodKey(item.method) === key
  );
  const segment = trace?.segments?.[state.playbackSegmentIndex];
  return segment ? { ...segment, method: key } : null;
}

function methodHasTrace(key) {
  return Boolean(traceSegmentForMethod(key));
}

function selectedTraceSegments() {
  const keys = state.selectedMethods;
  if (!keys.size) return [];
  return Array.from(keys).map((key) => traceSegmentForMethod(key)).filter(Boolean);
}

function setDefaultScenario() {
  const scenarios = scenariosForModel();
  const hasScenario = scenarios.some((scenario) => scenario.id === state.scenario);
  const sourceRows = (source) => allRows().filter((row) => row.model_family === state.modelFamily && row.state_source === source);
  if (!sourceRows(state.source).length) {
    const fallbackSource = ["mocap", "direct"].find((source) => sourceRows(source).length);
    if (fallbackSource) state.source = fallbackSource;
  }
  if (!hasScenario || !selectedRows().length) {
    const withRows = scenarios.find((scenario) =>
      allRows().some((row) => row.scenario === scenario.id && row.state_source === state.source && finiteNumber(row.validation_score))
    );
    state.scenario = withRows?.id || scenarios[0]?.id || "";
  }
  if (!selectedRows().length) {
    const fallbackSource = ["mocap", "direct"].find((source) =>
      allRows().some((row) => row.scenario === state.scenario && row.state_source === source && finiteNumber(row.validation_score))
    );
    if (fallbackSource) state.source = fallbackSource;
  }
}

function renderModelTabs() {
  const host = document.querySelector("#model-tabs");
  host.innerHTML = "";
  for (const family of state.manifest.model_families) {
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.model = family;
    button.className = family === state.modelFamily ? "active" : "";
    button.textContent = family.replace("aircraft", "").toUpperCase();
    button.addEventListener("click", () => {
      state.modelFamily = family;
      state.selectedMethods.clear();
      resetTradeoffZoom();
      setDefaultScenario();
      render();
    });
    host.append(button);
  }
}

function renderScenarioSelect() {
  const select = document.querySelector("#scenario-select");
  select.innerHTML = "";
  for (const scenario of scenariosForModel()) {
    const option = document.createElement("option");
    option.value = scenario.id;
    option.textContent = scenario.title;
    select.append(option);
  }
  select.value = state.scenario;
}

function bindControls() {
  document.querySelector("#scenario-select").addEventListener("change", (event) => {
    state.scenario = event.target.value;
    state.playbackSegmentIndex = 0;
    state.selectedMethods.clear();
    resetTradeoffZoom();
    render();
  });

  document.querySelector("#source-filter").addEventListener("click", (event) => {
    const button = event.target.closest("button[data-source]");
    if (!button) return;
    state.source = button.dataset.source;
    state.selectedMethods.clear();
    resetTradeoffZoom();
    render();
  });

  document.querySelector("#playback-tabs").addEventListener("click", (event) => {
    const button = event.target.closest("button[data-playback-view]");
    if (!button) return;
    state.playbackView = button.dataset.playbackView;
    renderPlaybackTabs();
    if (state.playbackView === "animation") resizePlayback();
  });

  document.querySelector("#playback-toggle").addEventListener("click", () => {
    state.playbackPlaying = !state.playbackPlaying;
    renderPlaybackControls(selectedPlayback());
  });
  document.querySelector("#playback-rewind").addEventListener("click", () => {
    seekPlayback(state.playbackTimeS - 2.0);
  });
  document.querySelector("#playback-forward").addEventListener("click", () => {
    seekPlayback(state.playbackTimeS + 2.0);
  });
  document.querySelector("#playback-follow").addEventListener("click", () => {
    state.playbackFollow = !state.playbackFollow;
    renderPlaybackControls(selectedPlayback());
  });
  document.querySelector("#playback-speed").addEventListener("change", (event) => {
    state.playbackSpeed = Number.parseFloat(event.target.value) || 1;
  });
  document.querySelector("#playback-segment").addEventListener("change", (event) => {
    state.playbackSegmentIndex = Number.parseInt(event.target.value, 10) || 0;
    state.playbackTimeS = 0;
    if (Array.from(state.selectedMethods).some((key) => !methodHasTrace(key))) {
      state.selectedMethods.clear();
    }
    setPlaybackTrack(selectedPlayback(), true);
    renderTimeseries();
    renderTradeoff(selectedRows());
    renderLeaderboard(selectedRows());
  });
  document.querySelector("#playback-scrub").addEventListener("input", (event) => {
    const track = selectedPlayback();
    const duration = playbackDuration(track);
    state.playbackScrubbing = true;
    seekPlayback(Number.parseFloat(event.target.value) * duration);
  });
  document.querySelector("#playback-scrub").addEventListener("change", () => {
    state.playbackScrubbing = false;
  });
}

function renderMeta() {
  const sha = state.manifest.git_sha ? state.manifest.git_sha.slice(0, 7) : "unknown";
  const generated = new Date(state.manifest.generated_at).toLocaleString();
  document.querySelector("#run-meta").textContent = `schema ${state.manifest.schema_version} | ${sha} | ${generated}`;
}

function renderPlaybackTabs() {
  document.querySelectorAll("#playback-tabs button[data-playback-view]").forEach((button) => {
    button.classList.toggle("active", button.dataset.playbackView === state.playbackView);
  });
  const animation = document.querySelector("#animation-view");
  const history = document.querySelector("#history-view");
  if (animation) animation.hidden = state.playbackView !== "animation";
  if (history) history.hidden = state.playbackView !== "history";
}

function renderSummary(rows) {
  const best = rows[0];
  const datasets = datasetsForModel();
  const methods = new Set(allRows().filter((row) => row.model_family === state.modelFamily).map((row) => cleanMethodName(row.method)));
  document.querySelector("#best-method").textContent = best ? cleanMethodName(best.method) : "--";
  document.querySelector("#best-score").textContent = best ? formatNumber(best.validation_score) : "--";
  document.querySelector("#method-count").textContent = String(methods.size);
  document.querySelector("#dataset-count").textContent = String(datasets.length);
}

function logExtent(values) {
  const finite = values.filter((value) => finiteNumber(value) && value > 0);
  if (!finite.length) return [0.01, 1];
  return [Math.min(...finite) * 0.75, Math.max(...finite) * 1.4];
}

function logScale(value, min, max, start, end) {
  const logMin = Math.log10(min);
  const logMax = Math.log10(max);
  const t = (Math.log10(Math.max(value, min)) - logMin) / Math.max(logMax - logMin, 1e-9);
  return start + t * (end - start);
}

function tradeoffKey() {
  return `${state.modelFamily}:${state.scenario}:${state.source}`;
}

function resetTradeoffZoom() {
  state.tradeoffZoom = null;
}

function constrainedLogView(view, base) {
  const minSpan = 0.12;
  const baseXSpan = base.xMax - base.xMin;
  const baseYSpan = base.yMax - base.yMin;
  let xMin = view.xMin;
  let xMax = view.xMax;
  let yMin = view.yMin;
  let yMax = view.yMax;
  if (xMax - xMin < minSpan) {
    const center = (xMin + xMax) / 2;
    xMin = center - minSpan / 2;
    xMax = center + minSpan / 2;
  }
  if (yMax - yMin < minSpan) {
    const center = (yMin + yMax) / 2;
    yMin = center - minSpan / 2;
    yMax = center + minSpan / 2;
  }
  if (xMax - xMin >= baseXSpan) {
    xMin = base.xMin;
    xMax = base.xMax;
  } else {
    if (xMin < base.xMin) {
      xMax += base.xMin - xMin;
      xMin = base.xMin;
    }
    if (xMax > base.xMax) {
      xMin -= xMax - base.xMax;
      xMax = base.xMax;
    }
  }
  if (yMax - yMin >= baseYSpan) {
    yMin = base.yMin;
    yMax = base.yMax;
  } else {
    if (yMin < base.yMin) {
      yMax += base.yMin - yMin;
      yMin = base.yMin;
    }
    if (yMax > base.yMax) {
      yMin -= yMax - base.yMax;
      yMax = base.yMax;
    }
  }
  return { xMin, xMax, yMin, yMax };
}

function powers(min, max) {
  const out = [];
  for (let power = Math.floor(Math.log10(min)); power <= Math.ceil(Math.log10(max)); power += 1) {
    out.push(10 ** power);
  }
  return out;
}

function renderTradeoff(rows) {
  const host = document.querySelector("#tradeoff-plot");
  host.innerHTML = "";
  const width = Math.max(host.clientWidth || 900, 640);
  const height = 410;
  const margin = { top: 20, right: 34, bottom: 54, left: 76 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const baseXExtent = logExtent(rows.map((row) => row.train_elapsed_s || row.total_elapsed_s || row.rollout_elapsed_s));
  const baseYExtent = logExtent(rows.map((row) => row.validation_score));
  const baseLogView = {
    xMin: Math.log10(baseXExtent[0]),
    xMax: Math.log10(baseXExtent[1]),
    yMin: Math.log10(baseYExtent[0]),
    yMax: Math.log10(baseYExtent[1]),
  };
  if (!state.tradeoffZoom || state.tradeoffZoom.key !== tradeoffKey()) {
    state.tradeoffZoom = { key: tradeoffKey(), ...baseLogView };
  }
  state.tradeoffZoom = { key: tradeoffKey(), ...constrainedLogView(state.tradeoffZoom, baseLogView) };
  const xExtent = [10 ** state.tradeoffZoom.xMin, 10 ** state.tradeoffZoom.xMax];
  const yExtent = [10 ** state.tradeoffZoom.yMin, 10 ** state.tradeoffZoom.yMax];
  const color = state.source === "direct" ? "var(--direct)" : "var(--mocap)";

  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("aria-label", "Cost-error tradeoff. Mouse wheel zooms, drag pans, double-click resets.");
  const eventPoint = (event) => {
    const rect = svg.getBoundingClientRect();
    const x = (event.clientX - rect.left) * width / Math.max(rect.width, 1);
    const y = (event.clientY - rect.top) * height / Math.max(rect.height, 1);
    return { x, y };
  };
  const inPlot = ({ x, y }) => x >= margin.left && x <= margin.left + plotWidth && y >= margin.top && y <= margin.top + plotHeight;
  const logAtPoint = ({ x, y }) => ({
    x: state.tradeoffZoom.xMin + ((x - margin.left) / plotWidth) * (state.tradeoffZoom.xMax - state.tradeoffZoom.xMin),
    y: state.tradeoffZoom.yMax - ((y - margin.top) / plotHeight) * (state.tradeoffZoom.yMax - state.tradeoffZoom.yMin),
  });
  const setLogView = (view) => {
    state.tradeoffZoom = { key: tradeoffKey(), ...constrainedLogView(view, baseLogView) };
    render();
  };
  svg.addEventListener("wheel", (event) => {
    const point = eventPoint(event);
    if (!inPlot(point)) return;
    event.preventDefault();
    const anchor = logAtPoint(point);
    const factor = event.deltaY < 0 ? 0.82 : 1.22;
    const view = state.tradeoffZoom;
    setLogView({
      xMin: anchor.x - (anchor.x - view.xMin) * factor,
      xMax: anchor.x + (view.xMax - anchor.x) * factor,
      yMin: anchor.y - (anchor.y - view.yMin) * factor,
      yMax: anchor.y + (view.yMax - anchor.y) * factor,
    });
  }, { passive: false });
  svg.addEventListener("pointerdown", (event) => {
    if (event.button !== 0 || event.target.classList?.contains("tradeoff-point")) return;
    const point = eventPoint(event);
    if (!inPlot(point)) return;
    const dragStart = { clientX: event.clientX, clientY: event.clientY, view: { ...state.tradeoffZoom } };
    const moveDrag = (moveEvent) => {
      const dx = (moveEvent.clientX - dragStart.clientX) * width / Math.max(svg.getBoundingClientRect().width, 1);
      const dy = (moveEvent.clientY - dragStart.clientY) * height / Math.max(svg.getBoundingClientRect().height, 1);
      const xSpan = dragStart.view.xMax - dragStart.view.xMin;
      const ySpan = dragStart.view.yMax - dragStart.view.yMin;
      state.tradeoffZoom = {
        key: tradeoffKey(),
        ...constrainedLogView({
          xMin: dragStart.view.xMin - (dx / plotWidth) * xSpan,
          xMax: dragStart.view.xMax - (dx / plotWidth) * xSpan,
          yMin: dragStart.view.yMin + (dy / plotHeight) * ySpan,
          yMax: dragStart.view.yMax + (dy / plotHeight) * ySpan,
        }, baseLogView),
      };
      renderTradeoff(rows);
    };
    const stopDrag = () => {
      document.removeEventListener("pointermove", moveDrag);
      document.removeEventListener("pointerup", stopDrag);
      document.removeEventListener("pointercancel", stopDrag);
      svg.classList.remove("dragging");
    };
    document.addEventListener("pointermove", moveDrag);
    document.addEventListener("pointerup", stopDrag);
    document.addEventListener("pointercancel", stopDrag);
    svg.classList.add("dragging");
    svg.setPointerCapture(event.pointerId);
  });
  svg.addEventListener("dblclick", () => {
    resetTradeoffZoom();
    render();
  });

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

  add("rect", { x: margin.left, y: margin.top, width: plotWidth, height: plotHeight, fill: "none", stroke: "var(--line)" });
  add("text", { x: margin.left + plotWidth / 2, y: height - 8, "text-anchor": "middle", class: "axis-label" }, "training time [s]");
  add("text", { x: 18, y: margin.top + plotHeight / 2, transform: `rotate(-90 18 ${margin.top + plotHeight / 2})`, "text-anchor": "middle", class: "axis-label" }, "validation score (lower is better)");
  add("text", { x: margin.left + plotWidth, y: margin.top - 6, "text-anchor": "end", class: "plot-hint" }, "wheel zoom | drag pan | double-click reset");

  const nominal = rows.find((row) => cleanMethodName(row.method).includes("Nominal") && finiteNumber(row.validation_score));
  if (nominal) {
    const y = logScale(nominal.validation_score, yExtent[0], yExtent[1], margin.top + plotHeight, margin.top);
    add("text", {
      x: margin.left + plotWidth * 0.5,
      y: Math.max(margin.top + 34, y - 18),
      "text-anchor": "middle",
      class: "known-label",
    }, "known");
    add("text", {
      x: margin.left + plotWidth * 0.5,
      y: Math.min(margin.top + plotHeight - 18, y + 34),
      "text-anchor": "middle",
      class: "unknown-label",
    }, "unknown");
    add("line", {
      x1: margin.left,
      y1: y,
      x2: margin.left + plotWidth,
      y2: y,
      class: "nominal-line",
    });
  }

  for (const row of rows) {
    const xValue = row.train_elapsed_s || row.total_elapsed_s || row.rollout_elapsed_s || 0.01;
    const x = logScale(xValue, xExtent[0], xExtent[1], margin.left, margin.left + plotWidth);
    const y = logScale(row.validation_score, yExtent[0], yExtent[1], margin.top + plotHeight, margin.top);
    const isNominal = cleanMethodName(row.method).includes("Nominal");
    const key = methodKey(row.method);
    const hasTrace = methodHasTrace(key);
    const selected = state.selectedMethods.has(key);
    const circle = add("circle", {
      cx: x,
      cy: y,
      r: selected ? 8.0 : isNominal ? 6.5 : 5.5,
      fill: isNominal ? "white" : color,
      stroke: selected ? "#111827" : isNominal ? "var(--nominal)" : "#1d2430",
      "stroke-width": selected ? 2.6 : isNominal ? 1.8 : 1,
      opacity: hasTrace ? (state.selectedMethods.size && !selected ? 0.34 : 0.88) : 0.22,
      class: `tradeoff-point${hasTrace ? "" : " no-trace"}`,
      tabindex: hasTrace ? "0" : "-1",
    });
    if (hasTrace) circle.addEventListener("click", () => toggleMethodSelection(key));
    const title = document.createElementNS("http://www.w3.org/2000/svg", "title");
    title.textContent = hasTrace
      ? `${cleanMethodName(row.method)} | ${formatNumber(row.validation_score)}`
      : `${cleanMethodName(row.method)} | ${formatNumber(row.validation_score)} | no exported trajectory`;
    circle.append(title);
  }

  const bestNonNominal = rows.find((row) => !cleanMethodName(row.method).includes("Nominal"));
  if (bestNonNominal) {
    const xValue = bestNonNominal.train_elapsed_s || bestNonNominal.total_elapsed_s || bestNonNominal.rollout_elapsed_s || 0.01;
    const x = logScale(xValue, xExtent[0], xExtent[1], margin.left, margin.left + plotWidth);
    const y = logScale(bestNonNominal.validation_score, yExtent[0], yExtent[1], margin.top + plotHeight, margin.top);
    add("text", {
      x: Math.min(x + 10, margin.left + plotWidth - 130),
      y: Math.max(y - 10, margin.top + 16),
      class: "best-point-label",
    }, cleanMethodName(bestNonNominal.method));
  }

  host.append(svg);
}

function renderLeaderboard(rows) {
  const body = document.querySelector("#leaderboard-body");
  body.innerHTML = "";
  for (const row of rows) {
    const tr = document.createElement("tr");
    const key = methodKey(row.method);
    const hasTrace = methodHasTrace(key);
    tr.className = `${state.selectedMethods.has(key) ? "selected-method-row" : ""}${hasTrace ? "" : " no-trace-row"}`;
    tr.title = hasTrace ? "Click to show this model trajectory." : "No exported model trajectory is available for this flight segment.";
    if (hasTrace) tr.addEventListener("click", () => toggleMethodSelection(key));
    const values = [
      cleanMethodName(row.method),
      formatNumber(row.validation_score),
      formatNumber(row.rmse_position_m ?? row.rmse_mocap_position_m),
      formatNumber(row.train_elapsed_s),
      formatNumber(row.rollout_elapsed_s),
      row.training_scenario || "--",
    ];
    for (const [index, value] of values.entries()) {
      const cell = document.createElement("td");
      cell.textContent = value;
      if (index > 0 && index < 5) cell.className = "numeric";
      tr.append(cell);
    }
    body.append(tr);
  }
}

function toggleMethodSelection(key) {
  if (!methodHasTrace(key)) {
    return;
  }
  if (state.selectedMethods.has(key)) {
    state.selectedMethods.delete(key);
  } else {
    state.selectedMethods.clear();
    state.selectedMethods.add(key);
  }
  render();
}

function renderDatasets() {
  const body = document.querySelector("#dataset-body");
  body.innerHTML = "";
  for (const dataset of datasetsForModel()) {
    const tr = document.createElement("tr");
    const values = [
      dataset.title || dataset.id,
      dataset.status || "--",
      dataset.source_type || "--",
      dataset.local_data_dir || dataset.generator || "--",
    ];
    for (const value of values) {
      const cell = document.createElement("td");
      cell.textContent = value;
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
        ["Speed", `${formatNumber(maneuver.min_speed_mps)}-${formatNumber(maneuver.max_speed_mps)} m/s`],
        ["Vertical", `${formatNumber(maneuver.vertical_extent_m)} m`],
      ]
    : [["Envelope", "--"]];
  for (const [term, detail] of rows) {
    const dt = document.createElement("dt");
    const dd = document.createElement("dd");
    dt.textContent = term;
    dd.textContent = detail;
    list.append(dt, dd);
  }
}

function enuToThree(position) {
  return new THREE.Vector3(position[0], position[2], -position[1]);
}

function attitudeToThree(quaternionWxyz) {
  const bodyToEnu = new THREE.Quaternion(
    quaternionWxyz[1],
    quaternionWxyz[2],
    quaternionWxyz[3],
    quaternionWxyz[0],
  ).normalize();
  return ENU_TO_THREE_QUAT.clone().multiply(bodyToEnu).multiply(MESH_TO_BODY_FRD_QUAT).normalize();
}

function quaternionToEulerDeg(quaternionWxyz) {
  if (!quaternionWxyz?.length) return [0, 0, 0];
  const q = new THREE.Quaternion(
    quaternionWxyz[1],
    quaternionWxyz[2],
    quaternionWxyz[3],
    quaternionWxyz[0],
  ).normalize();
  const matrix = new THREE.Matrix4().makeRotationFromQuaternion(q).elements;
  const forward = new THREE.Vector3(matrix[0], matrix[1], matrix[2]).normalize();
  const right = new THREE.Vector3(matrix[4], matrix[5], matrix[6]).normalize();
  const pitch = Math.asin(clamp(forward.z, -1, 1));
  const yaw = Math.atan2(forward.y, forward.x);
  const rightLevel = new THREE.Vector3(Math.sin(yaw), -Math.cos(yaw), 0).normalize();
  const downLevel = new THREE.Vector3().crossVectors(forward, rightLevel).normalize();
  const roll = Math.atan2(right.dot(downLevel), right.dot(rightLevel));
  const radToDeg = 180 / Math.PI;
  return [roll * radToDeg, pitch * radToDeg, yaw * radToDeg];
}

function playbackDuration(trackOrSegment) {
  const segment = trackOrSegment?.segments ? activeSegment(trackOrSegment) : trackOrSegment;
  return Math.max(segment?.time_s?.at(-1) || 1, 1);
}

function seekPlayback(timeS) {
  const segment = activeSegment();
  state.playbackTimeS = clamp(timeS, 0, playbackDuration(segment));
  state.playbackLastMs = null;
  updatePlaybackScrub(segment);
}

function makeTaperedControlGeometry(chord, span, thickness, spanAxis = "z") {
  const halfSpan = span / 2;
  const halfThickness = thickness / 2;
  const positions = spanAxis === "z"
    ? [
        0, -halfThickness, -halfSpan, 0, -halfThickness, halfSpan, 0, halfThickness, halfSpan, 0, halfThickness, -halfSpan,
        -chord, 0, -halfSpan, -chord, 0, halfSpan,
      ]
    : [
        0, -halfSpan, -halfThickness, 0, halfSpan, -halfThickness, 0, halfSpan, halfThickness, 0, -halfSpan, halfThickness,
        -chord, -halfSpan, 0, -chord, halfSpan, 0,
      ];
  const indices = [
    0, 1, 2, 0, 2, 3,
    0, 4, 5, 0, 5, 1,
    3, 2, 5, 3, 5, 4,
    0, 3, 4,
    1, 5, 2,
  ];
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  return geometry;
}

function makeWingPanelGeometry(rootChord, tipChord, span, thickness, side) {
  const zRoot = side * 0.18;
  const zTip = side * (0.18 + span);
  const leadingRoot = rootChord / 2;
  const trailingRoot = -rootChord / 2;
  const leadingTip = rootChord / 2 - 0.1;
  const trailingTip = leadingTip - tipChord;
  const yTop = thickness / 2;
  const yBottom = -thickness / 2;
  const positions = [
    leadingRoot, yTop, zRoot, trailingRoot, yTop, zRoot, trailingTip, yTop, zTip, leadingTip, yTop, zTip,
    leadingRoot, yBottom, zRoot, trailingRoot, yBottom, zRoot, trailingTip, yBottom, zTip, leadingTip, yBottom, zTip,
  ];
  const indices = [
    0, 1, 2, 0, 2, 3,
    4, 7, 6, 4, 6, 5,
    0, 4, 5, 0, 5, 1,
    1, 5, 6, 1, 6, 2,
    2, 6, 7, 2, 7, 3,
    3, 7, 4, 3, 4, 0,
  ];
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  return geometry;
}

function makeAircraftMesh() {
  const group = new THREE.Group();
  const model = new THREE.Group();
  model.scale.setScalar(AIRCRAFT_MODEL_SCALE);
  group.add(model);
  const bodyMaterial = new THREE.MeshStandardMaterial({ color: 0x2f5f9f, roughness: 0.46, metalness: 0.08 });
  const wingMaterial = new THREE.MeshStandardMaterial({ color: 0xd9e2ef, roughness: 0.55, metalness: 0.04 });
  const accentMaterial = new THREE.MeshStandardMaterial({ color: 0xd97706, roughness: 0.5, metalness: 0.02 });
  const cockpitMaterial = new THREE.MeshStandardMaterial({ color: 0x8aa5bf, roughness: 0.28, metalness: 0.04, transparent: true, opacity: 0.78 });
  const controlMaterial = new THREE.MeshStandardMaterial({ color: 0xff8a1f, roughness: 0.42, metalness: 0.02, side: THREE.DoubleSide });
  const propMaterial = new THREE.MeshStandardMaterial({ color: 0x2b3442, roughness: 0.35, metalness: 0.08 });
  const propDiskMaterial = new THREE.MeshBasicMaterial({ color: 0x7fb4ff, transparent: true, opacity: 0.16, side: THREE.DoubleSide });

  const fuselage = new THREE.Mesh(new THREE.CylinderGeometry(0.115, 0.115, 1.35, 18), bodyMaterial);
  fuselage.rotation.z = Math.PI / 2;
  model.add(fuselage);
  const cockpit = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.15, 0.15), cockpitMaterial);
  cockpit.position.set(0.28, 0.14, 0);
  model.add(cockpit);
  const nose = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.105, 0.28, 24), wingMaterial);
  nose.rotation.z = Math.PI / 2;
  nose.position.x = 0.82;
  model.add(nose);
  const wingCenter = new THREE.Mesh(new THREE.BoxGeometry(0.42, 0.035, 0.42), wingMaterial);
  wingCenter.position.set(0.2, -0.035, 0);
  model.add(wingCenter);
  const leftWing = new THREE.Mesh(makeWingPanelGeometry(0.48, 0.32, 0.78, 0.035, -1), wingMaterial);
  leftWing.position.set(0.2, -0.035, 0);
  model.add(leftWing);
  const rightWing = new THREE.Mesh(makeWingPanelGeometry(0.48, 0.32, 0.78, 0.035, 1), wingMaterial);
  rightWing.position.set(0.2, -0.035, 0);
  model.add(rightWing);

  const leftAileron = new THREE.Group();
  leftAileron.position.set(0.02, -0.05, -0.72);
  const leftAileronPanel = new THREE.Mesh(makeTaperedControlGeometry(0.18, 0.46, 0.032), controlMaterial);
  leftAileron.add(leftAileronPanel);
  model.add(leftAileron);
  const rightAileron = new THREE.Group();
  rightAileron.position.set(0.02, -0.05, 0.72);
  const rightAileronPanel = new THREE.Mesh(makeTaperedControlGeometry(0.18, 0.46, 0.032), controlMaterial);
  rightAileron.add(rightAileronPanel);
  model.add(rightAileron);

  const tail = new THREE.Mesh(new THREE.BoxGeometry(0.28, 0.03, 0.96), wingMaterial);
  tail.position.x = -0.64;
  model.add(tail);
  const elevator = new THREE.Group();
  elevator.position.set(-0.78, -0.01, 0);
  const elevatorPanel = new THREE.Mesh(makeTaperedControlGeometry(0.16, 0.96, 0.038), controlMaterial);
  elevator.add(elevatorPanel);
  model.add(elevator);
  const fin = new THREE.Mesh(new THREE.BoxGeometry(0.26, 0.52, 0.05), wingMaterial);
  fin.position.set(-0.66, 0.31, 0);
  model.add(fin);
  const rudder = new THREE.Group();
  rudder.position.set(-0.79, 0.31, 0);
  const rudderPanel = new THREE.Mesh(makeTaperedControlGeometry(0.15, 0.52, 0.045, "y"), controlMaterial);
  rudder.add(rudderPanel);
  model.add(rudder);

  const prop = new THREE.Group();
  prop.position.x = 1.03;
  const bladeA = new THREE.Mesh(new THREE.BoxGeometry(0.018, 0.72, 0.045), propMaterial);
  const bladeB = new THREE.Mesh(new THREE.BoxGeometry(0.018, 0.045, 0.72), propMaterial);
  const disk = new THREE.Mesh(new THREE.CircleGeometry(0.42, 48), propDiskMaterial);
  disk.rotation.y = Math.PI / 2;
  prop.add(bladeA, bladeB, disk);
  model.add(prop);
  group.userData = { leftAileron, rightAileron, elevator, rudder, prop, propDisk: disk };
  return group;
}

function makeTransparentAircraftMesh(color) {
  const aircraft = makeAircraftMesh();
  aircraft.traverse((child) => {
    if (!child.isMesh || !child.material) return;
    const material = child.material.clone();
    if (material.color) material.color.set(color);
    material.transparent = true;
    material.opacity = child.geometry?.type === "CircleGeometry" ? 0.08 : 0.34;
    material.depthWrite = false;
    material.side = THREE.DoubleSide;
    child.material = material;
    child.renderOrder = 10;
  });
  return aircraft;
}

function updateAircraftControls(aircraft, controls, deltaS) {
  const parts = aircraft.userData || {};
  const thrust = clamp(controls[0] ?? 0.45, 0, 1);
  const aileron = clamp(controls[1] ?? 0, -1, 1);
  const elevator = clamp(controls[2] ?? 0, -1, 1);
  const rudder = clamp(controls[3] ?? 0, -1, 1);
  if (parts.leftAileron) parts.leftAileron.rotation.z = -0.95 * aileron;
  if (parts.rightAileron) parts.rightAileron.rotation.z = 0.95 * aileron;
  if (parts.elevator) parts.elevator.rotation.z = -1.0 * elevator;
  if (parts.rudder) parts.rudder.rotation.y = 0.95 * rudder;
  if (parts.prop) parts.prop.rotation.x += deltaS * (22 + 90 * thrust);
  if (parts.propDisk) parts.propDisk.material.opacity = 0.08 + 0.22 * thrust;
}

function updateControlHud(controls) {
  const ids = ["thrust", "aileron", "elevator", "rudder"];
  const values = [
    clamp(controls[0] ?? 0, 0, 1),
    clamp(controls[1] ?? 0, -1, 1),
    clamp(controls[2] ?? 0, -1, 1),
    clamp(controls[3] ?? 0, -1, 1),
  ];
  ids.forEach((id, index) => {
    const fill = document.querySelector(`#control-${id}`);
    const label = document.querySelector(`#control-${id}-value`);
    if (fill) {
      const value = values[index];
      if (id === "thrust") {
        fill.style.left = "0";
        fill.style.width = `${100 * value}%`;
        fill.classList.remove("negative");
      } else {
        const magnitude = 50 * Math.abs(value);
        fill.style.left = value < 0 ? `${50 - magnitude}%` : "50%";
        fill.style.width = `${magnitude}%`;
        fill.classList.toggle("negative", value < 0);
      }
    }
    if (label) label.textContent = values[index].toFixed(2);
  });
}

function disposeMaterial(material) {
  if (Array.isArray(material)) {
    for (const item of material) item.dispose();
  } else if (material) {
    material.dispose();
  }
}

function disposeLine(line) {
  line.geometry.dispose();
  disposeMaterial(line.material);
}

function disposeObject3D(object) {
  object.traverse((child) => {
    if (child.geometry) child.geometry.dispose();
    if (child.material) {
      const materials = Array.isArray(child.material) ? child.material : [child.material];
      for (const material of materials) {
        if (material.map) material.map.dispose();
        material.dispose();
      }
    }
  });
}

function niceTickSpacing(size) {
  const target = Math.max(size / 5, 1);
  const exponent = Math.floor(Math.log10(target));
  const fraction = target / 10 ** exponent;
  const nice = fraction <= 1 ? 1 : fraction <= 2 ? 2 : fraction <= 5 ? 5 : 10;
  return nice * 10 ** exponent;
}

function makeAxisLabel(text, color = "#334155") {
  const canvas = document.createElement("canvas");
  canvas.width = 256;
  canvas.height = 80;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.font = "600 28px system-ui, -apple-system, BlinkMacSystemFont, sans-serif";
  ctx.fillStyle = color;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, canvas.width / 2, canvas.height / 2);
  const texture = new THREE.CanvasTexture(canvas);
  const material = new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false });
  const sprite = new THREE.Sprite(material);
  sprite.scale.set(1.6, 0.5, 1);
  return sprite;
}

function addAxisLine(group, origin, direction, length, color, label, tickSpacing) {
  const material = new THREE.LineBasicMaterial({ color });
  const end = origin.clone().addScaledVector(direction, length);
  group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([origin, end]), material));

  const tickHalf = Math.max(length * 0.012, 0.08);
  const tickMaterial = new THREE.LineBasicMaterial({ color });
  const tickDirection = Math.abs(direction.y) > 0.5 ? new THREE.Vector3(1, 0, 0) : new THREE.Vector3(0, 1, 0);
  for (let tick = tickSpacing; tick < length + tickSpacing * 0.25; tick += tickSpacing) {
    const center = origin.clone().addScaledVector(direction, Math.min(tick, length));
    group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([
      center.clone().addScaledVector(tickDirection, -tickHalf),
      center.clone().addScaledVector(tickDirection, tickHalf),
    ]), tickMaterial));
    const tickLabel = makeAxisLabel(`${Math.round(tick)} m`);
    tickLabel.position.copy(center).addScaledVector(tickDirection, tickHalf * 3);
    tickLabel.scale.set(1.1, 0.34, 1);
    group.add(tickLabel);
  }

  const axisLabel = makeAxisLabel(label, `#${color.toString(16).padStart(6, "0")}`);
  axisLabel.position.copy(end).addScaledVector(direction, tickHalf * 5);
  group.add(axisLabel);
}

function makePlaybackAxes(center, size, floorY) {
  const group = new THREE.Group();
  const length = Math.max(niceTickSpacing(size) * 2, Math.min(size * 0.45, 40));
  const tickSpacing = niceTickSpacing(length);
  const origin = new THREE.Vector3(center.x - size * 0.48, floorY + 0.04, center.z + size * 0.48);
  addAxisLine(group, origin, new THREE.Vector3(1, 0, 0), length, 0xdc2626, "East", tickSpacing);
  addAxisLine(group, origin, new THREE.Vector3(0, 0, -1), length, 0x16a34a, "North", tickSpacing);
  addAxisLine(group, origin, new THREE.Vector3(0, 1, 0), Math.max(length * 0.45, tickSpacing), 0x2563eb, "Up", tickSpacing);
  return group;
}

function updatePlaybackCamera(playback) {
  const controls = playback.controls;
  const distance = controls.distance;
  const pitch = controls.pitch;
  const yaw = controls.yaw;
  const offset = new THREE.Vector3(
    distance * Math.cos(pitch) * Math.sin(yaw),
    distance * Math.sin(pitch),
    distance * Math.cos(pitch) * Math.cos(yaw),
  );
  playback.camera.position.copy(controls.target).add(offset);
  playback.camera.lookAt(controls.target);
}

function bindOrbitControls(host, playback) {
  const controls = playback.controls;
  host.addEventListener("contextmenu", (event) => event.preventDefault());
  host.addEventListener("pointerdown", (event) => {
    controls.pointer = { x: event.clientX, y: event.clientY, button: event.button, pan: event.shiftKey || event.button === 2 };
    host.setPointerCapture(event.pointerId);
  });
  host.addEventListener("pointermove", (event) => {
    if (!controls.pointer) return;
    const dx = event.clientX - controls.pointer.x;
    const dy = event.clientY - controls.pointer.y;
    controls.pointer.x = event.clientX;
    controls.pointer.y = event.clientY;
    if (controls.pointer.pan) {
      const panScale = controls.distance * 0.0015;
      const right = new THREE.Vector3().subVectors(playback.camera.position, controls.target).cross(playback.camera.up).normalize();
      const up = playback.camera.up.clone().normalize();
      controls.target.addScaledVector(right, -dx * panScale).addScaledVector(up, dy * panScale);
    } else {
      controls.yaw -= dx * 0.006;
      controls.pitch = clamp(controls.pitch - dy * 0.006, -1.42, 1.42);
    }
    updatePlaybackCamera(playback);
  });
  host.addEventListener("pointerup", () => {
    controls.pointer = null;
  });
  host.addEventListener("wheel", (event) => {
    event.preventDefault();
    controls.distance = clamp(controls.distance * Math.exp(event.deltaY * 0.001), controls.minDistance, controls.maxDistance);
    updatePlaybackCamera(playback);
  }, { passive: false });
}

function ensurePlaybackScene() {
  if (state.playbackScene) return state.playbackScene;
  const host = document.querySelector("#aircraft-playback");
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
  } catch (_error) {
    host.textContent = "3D playback requires WebGL. The plots and dataset comparisons are still available.";
    host.classList.add("playback-unavailable");
    return null;
  }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(host.clientWidth || 900, host.clientHeight || 360);
  renderer.setClearColor(0xd9dee7, 1);
  host.append(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 1000);
  scene.add(new THREE.HemisphereLight(0xffffff, 0xc8d1dd, 2.2));
  const sun = new THREE.DirectionalLight(0xffffff, 1.8);
  sun.position.set(6, 10, 8);
  scene.add(sun);
  const grid = new THREE.GridHelper(18, 18, 0x9aa8ba, 0xc2cad6);
  scene.add(grid);
  const aircraft = makeAircraftMesh();
  scene.add(aircraft);

  state.playbackScene = {
    renderer,
    scene,
    camera,
    aircraft,
    grid,
    axes: null,
    trackLine: null,
    methodLines: [],
    methodAircraft: [],
    track: null,
    methodSignature: "",
    controls: {
      target: new THREE.Vector3(),
      yaw: 0.78,
      pitch: 0.45,
      distance: 12,
      minDistance: 1,
      maxDistance: 200,
      pointer: null,
    },
    lastRenderMs: performance.now(),
  };

  updatePlaybackCamera(state.playbackScene);
  bindOrbitControls(host, state.playbackScene);
  window.addEventListener("resize", () => resizePlayback());
  requestAnimationFrame(tickPlayback);
  return state.playbackScene;
}

function resizePlayback() {
  const playback = state.playbackScene;
  if (!playback) return;
  const host = document.querySelector("#aircraft-playback");
  const width = host.clientWidth || 900;
  const height = host.clientHeight || 360;
  playback.renderer.setSize(width, height);
  playback.camera.aspect = width / Math.max(height, 1);
  playback.camera.updateProjectionMatrix();
}

function setPlaybackTrack(track, force = false) {
  const playback = ensurePlaybackScene();
  if (!playback) {
    renderPlaybackControls(track);
    return;
  }
  const segment = activeSegment(track);
  const segmentName = segment?.name || "segment";
  const methodSignature = Array.from(state.selectedMethods).sort().join("|");
  const trackChanged = playback.track?.id !== track?.id || playback.segmentName !== segmentName;
  if (!force && !trackChanged && playback.methodSignature === methodSignature) return;
  playback.track = track;
  playback.segmentName = segmentName;
  playback.methodSignature = methodSignature;
  if (force || trackChanged) {
    state.playbackTimeS = 0;
    state.playbackLastMs = null;
  }
  if (playback.trackLine) {
    playback.scene.remove(playback.trackLine);
    disposeLine(playback.trackLine);
    playback.trackLine = null;
  }
  for (const line of playback.methodLines) {
    playback.scene.remove(line);
    disposeLine(line);
  }
  playback.methodLines = [];
  for (const overlay of playback.methodAircraft) {
    playback.scene.remove(overlay.mesh);
    disposeObject3D(overlay.mesh);
  }
  playback.methodAircraft = [];
  if (!track || !segment) return;
  const points = segment.position_enu_m.map(enuToThree);
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  const material = new THREE.LineBasicMaterial({ color: 0x1f6feb, linewidth: 2 });
  playback.trackLine = new THREE.Line(geometry, material);
  playback.scene.add(playback.trackLine);
  const methodColors = [0x111827, 0x7c3aed, 0x059669, 0xb45309, 0xbe123c];
  selectedTraceSegments().forEach((trace, index) => {
    if (!trace.position_enu_m?.length) return;
    const color = methodColors[index % methodColors.length];
    const tracePoints = trace.position_enu_m.map(enuToThree);
    const traceGeometry = new THREE.BufferGeometry().setFromPoints(tracePoints);
    const traceMaterial = new THREE.LineBasicMaterial({
      color,
      linewidth: 2,
      transparent: true,
      opacity: 0.78,
    });
    const traceLine = new THREE.Line(traceGeometry, traceMaterial);
    playback.methodLines.push(traceLine);
    playback.scene.add(traceLine);
    const traceAircraft = makeTransparentAircraftMesh(color);
    playback.methodAircraft.push({ mesh: traceAircraft, segment: trace });
    playback.scene.add(traceAircraft);
  });
  const box = new THREE.Box3().setFromPoints(points);
  const center = box.getCenter(new THREE.Vector3());
  const extents = box.getSize(new THREE.Vector3());
  const size = Math.max(extents.x, extents.y, extents.z, 1);
  playback.controls.target.copy(center);
  playback.controls.distance = clamp(size * 2.2, 4, 120);
  playback.controls.minDistance = 0.25;
  playback.controls.maxDistance = Math.max(size * 8, 20);
  if (playback.grid) {
    playback.scene.remove(playback.grid);
    playback.grid.geometry.dispose();
    disposeMaterial(playback.grid.material);
  }
  const gridSize = Math.max(10, Math.ceil(size * 1.5));
  playback.grid = new THREE.GridHelper(gridSize, 20, 0x9aa8ba, 0xc2cad6);
  playback.grid.position.copy(center);
  playback.grid.position.y = Math.min(...points.map((point) => point.y));
  playback.scene.add(playback.grid);
  if (playback.axes) {
    playback.scene.remove(playback.axes);
    disposeObject3D(playback.axes);
  }
  playback.axes = makePlaybackAxes(center, gridSize, playback.grid.position.y);
  playback.scene.add(playback.axes);
  updatePlaybackCamera(playback);
  resizePlayback();
  renderPlaybackControls(track);
}

function sampleTrack(trackOrSegment, elapsedS, options = {}) {
  const track = trackOrSegment?.segments ? activeSegment(trackOrSegment) : trackOrSegment;
  if (!track) return null;
  const times = track.time_s;
  const duration = Math.max(times[times.length - 1] || 1, 1);
  if (!options.loop && (elapsedS < times[0] || elapsedS > duration)) return null;
  const t = options.loop ? ((elapsedS % duration) + duration) % duration : clamp(elapsedS, times[0], duration);
  let index = 0;
  while (index < times.length - 2 && times[index + 1] < t) index += 1;
  const t0 = times[index];
  const t1 = times[index + 1] ?? t0 + 1;
  const ratio = Math.max(0, Math.min(1, (t - t0) / Math.max(t1 - t0, 1e-9)));
  const p0 = enuToThree(track.position_enu_m[index]);
  const p1 = enuToThree(track.position_enu_m[index + 1] || track.position_enu_m[index]);
  const q0 = track.quaternion_wxyz[index];
  const q1 = track.quaternion_wxyz[index + 1] || q0;
  const quat0 = attitudeToThree(q0);
  const quat1 = attitudeToThree(q1);
  const c0 = track.control_meas?.[index] || [0.45, 0, 0, 0];
  const c1 = track.control_meas?.[index + 1] || c0;
  const controls = c0.map((value, controlIndex) => value + (c1[controlIndex] - value) * ratio);
  return { position: p0.lerp(p1, ratio), quaternion: quat0.slerp(quat1, ratio), controls };
}

function updatePlaybackScrub(track) {
  const scrub = document.querySelector("#playback-scrub");
  if (!scrub || state.playbackScrubbing) return;
  scrub.value = String(clamp(state.playbackTimeS / playbackDuration(track), 0, 1));
}

function renderPlaybackControls(track) {
  const toggle = document.querySelector("#playback-toggle");
  const follow = document.querySelector("#playback-follow");
  const speed = document.querySelector("#playback-speed");
  const segmentSelect = document.querySelector("#playback-segment");
  if (toggle) {
    toggle.textContent = state.playbackPlaying ? "||" : ">";
    toggle.setAttribute("aria-label", state.playbackPlaying ? "Pause" : "Play");
  }
  if (follow) {
    follow.classList.toggle("active", state.playbackFollow);
    follow.setAttribute("aria-pressed", String(state.playbackFollow));
  }
  if (speed) speed.value = String(state.playbackSpeed);
  if (segmentSelect && track) {
    const segments = track.segments?.length ? track.segments : [track];
    segmentSelect.innerHTML = "";
    segments.forEach((segment, index) => {
      const option = document.createElement("option");
      option.value = String(index);
      option.textContent = segment.name || `flight ${index + 1}`;
      segmentSelect.append(option);
    });
    state.playbackSegmentIndex = clamp(state.playbackSegmentIndex, 0, segments.length - 1);
    segmentSelect.value = String(state.playbackSegmentIndex);
    segmentSelect.disabled = segments.length <= 1;
  }
  updatePlaybackScrub(activeSegment(track));
}

function tickPlayback(nowMs) {
  const playback = state.playbackScene;
  const deltaS = Math.min(((nowMs || performance.now()) - (state.playbackLastMs || nowMs || performance.now())) / 1000, 0.08);
  state.playbackLastMs = nowMs || performance.now();
  if (playback?.track) {
    const segment = activeSegment(playback.track);
    if (state.playbackPlaying && !state.playbackScrubbing) {
      state.playbackTimeS = (state.playbackTimeS + deltaS * state.playbackSpeed) % playbackDuration(segment);
    }
    const sample = sampleTrack(segment, state.playbackTimeS, { loop: true });
    if (sample) {
      playback.aircraft.position.copy(sample.position);
      playback.aircraft.quaternion.copy(sample.quaternion);
      updateAircraftControls(playback.aircraft, sample.controls, deltaS);
      updateControlHud(sample.controls);
      for (const overlay of playback.methodAircraft) {
        const methodSample = sampleTrack(overlay.segment, state.playbackTimeS);
        overlay.mesh.visible = Boolean(methodSample);
        if (!methodSample) continue;
        overlay.mesh.position.copy(methodSample.position);
        overlay.mesh.quaternion.copy(methodSample.quaternion);
        updateAircraftControls(overlay.mesh, methodSample.controls, deltaS);
      }
      if (state.playbackFollow) {
        playback.controls.target.copy(sample.position);
        updatePlaybackCamera(playback);
      }
    }
    updatePlaybackScrub(segment);
    playback.renderer.render(playback.scene, playback.camera);
  }
  requestAnimationFrame(tickPlayback);
}

function renderPlayback() {
  const track = selectedPlayback();
  const status = document.querySelector("#playback-status");
  if (!track) {
    status.textContent = "No trajectory available";
    return;
  }
  const segment = activeSegment(track);
  status.textContent = `${track.title} | ${track.source}${segment?.name ? ` | ${segment.name}` : ""}`;
  setPlaybackTrack(track);
  renderPlaybackControls(track);
}

function linearExtent(values) {
  const finite = values.filter((value) => finiteNumber(value));
  if (!finite.length) return [-1, 1];
  let min = Math.min(...finite);
  let max = Math.max(...finite);
  if (Math.abs(max - min) < 1e-9) {
    min -= 1;
    max += 1;
  }
  const pad = 0.08 * (max - min);
  return [min - pad, max + pad];
}

function renderMiniSeries(title, series, traces) {
  const width = 520;
  const height = 170;
  const margin = { top: 28, right: 18, bottom: 34, left: 58 };
  const values = series.values.concat(...traces.map((trace) => trace.values));
  const xExtent = [series.time[0] || 0, series.time.at(-1) || 1];
  const yExtent = linearExtent(values);
  const xScale = (value) => margin.left + ((value - xExtent[0]) / Math.max(xExtent[1] - xExtent[0], 1e-9)) * (width - margin.left - margin.right);
  const yScale = (value) => height - margin.bottom - ((value - yExtent[0]) / Math.max(yExtent[1] - yExtent[0], 1e-9)) * (height - margin.top - margin.bottom);
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  const add = (tag, attrs, text) => {
    const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
    if (text !== undefined) node.textContent = text;
    svg.append(node);
    return node;
  };
  add("text", { x: margin.left, y: 16, class: "series-title" }, title);
  for (let i = 0; i <= 3; i += 1) {
    const fraction = i / 3;
    const value = yExtent[1] - fraction * (yExtent[1] - yExtent[0]);
    const y = margin.top + fraction * (height - margin.top - margin.bottom);
    add("line", { x1: margin.left, y1: y, x2: width - margin.right, y2: y, class: "grid-line" });
    add("text", {
      x: margin.left - 8,
      y: y + 4,
      "text-anchor": "end",
      class: "tick",
    }, formatNumber(value));
  }
  add("line", { x1: margin.left, y1: height - margin.bottom, x2: width - margin.right, y2: height - margin.bottom, class: "axis-line" });
  add("line", { x1: margin.left, y1: margin.top, x2: margin.left, y2: height - margin.bottom, class: "axis-line" });
  const path = (time, data) => time.map((t, index) => `${index ? "L" : "M"}${xScale(t).toFixed(2)},${yScale(data[index]).toFixed(2)}`).join(" ");
  add("path", { d: path(series.time, series.values), class: "truth-series" });
  for (const trace of traces) {
    const pairs = trace.time
      .map((timeValue, index) => [timeValue, trace.values[index]])
      .filter(([timeValue]) => timeValue >= xExtent[0] && timeValue <= xExtent[1]);
    if (pairs.length > 1) {
      add("path", { d: path(pairs.map((row) => row[0]), pairs.map((row) => row[1])), class: "method-series" });
    }
  }
  add("text", { x: margin.left, y: height - 17, class: "tick" }, formatNumber(xExtent[0]));
  add("text", { x: width - margin.right, y: height - 17, "text-anchor": "end", class: "tick" }, formatNumber(xExtent[1]));
  add("text", { x: (margin.left + width - margin.right) / 2, y: height - 7, "text-anchor": "middle", class: "axis-label" }, "time [s]");
  return svg;
}

function renderTimeseries() {
  const host = document.querySelector("#timeseries-plots");
  const status = document.querySelector("#timeseries-status");
  host.innerHTML = "";
  const track = selectedPlayback();
  const segment = activeSegment(track);
  if (!segment) {
    status.textContent = "No time history data available";
    return;
  }
  const time = segment.time_s;
  const pose = segment.position_enu_m;
  const quat = segment.quaternion_wxyz || [];
  const controls = segment.control_meas || [];
  const eulerDeg = quat.map(quaternionToEulerDeg);
  const traceSegments = selectedTraceSegments();
  const definitions = [
    ["East position [m]", pose.map((row) => row[0]), (trace) => trace.position_enu_m?.map((row) => row[0])],
    ["North position [m]", pose.map((row) => row[1]), (trace) => trace.position_enu_m?.map((row) => row[1])],
    ["Up position [m]", pose.map((row) => row[2]), (trace) => trace.position_enu_m?.map((row) => row[2])],
    ["Roll [deg]", eulerDeg.map((row) => row[0]), (trace) => trace.quaternion_wxyz?.map((row) => quaternionToEulerDeg(row)[0])],
    ["Pitch [deg]", eulerDeg.map((row) => row[1]), (trace) => trace.quaternion_wxyz?.map((row) => quaternionToEulerDeg(row)[1])],
    ["Yaw [deg]", eulerDeg.map((row) => row[2]), (trace) => trace.quaternion_wxyz?.map((row) => quaternionToEulerDeg(row)[2])],
    ["Thrust command [-]", controls.map((row) => row[0] ?? 0), (trace) => trace.control_meas?.map((row) => row[0] ?? 0)],
    ["Aileron command [-]", controls.map((row) => row[1] ?? 0), (trace) => trace.control_meas?.map((row) => row[1] ?? 0)],
    ["Elevator command [-]", controls.map((row) => row[2] ?? 0), (trace) => trace.control_meas?.map((row) => row[2] ?? 0)],
    ["Rudder command [-]", controls.map((row) => row[3] ?? 0), (trace) => trace.control_meas?.map((row) => row[3] ?? 0)],
  ].filter((item) => item[1].length);
  for (const [title, values, traceAccessor] of definitions) {
    const traces = traceSegments
      .map((trace) => ({ method: trace.method, time: trace.time_s, values: traceAccessor(trace) }))
      .filter((trace) => trace.values?.length === trace.time?.length);
    host.append(renderMiniSeries(title, { time, values }, traces));
  }
  const selected = state.selectedMethods.size ? `${Array.from(state.selectedMethods)[0]} selected` : "select one method to overlay its exported model trajectory";
  const available = traceSegments.length ? "model trajectories shown" : "no exported model trajectories for this dataset yet";
  status.textContent = `${segment.name || "flight"} | ${selected} | ${available}`;
}

function render() {
  setDefaultScenario();
  renderPlaybackTabs();
  renderModelTabs();
  renderScenarioSelect();
  for (const node of document.querySelectorAll("#source-filter button")) {
    node.classList.toggle("active", node.dataset.source === state.source);
  }
  const rows = selectedRows();
  renderSummary(rows);
  renderTradeoff(rows);
  renderLeaderboard(rows);
  renderDatasets();
  renderManeuver();
  renderPlayback();
  renderTimeseries();
}

async function init() {
  state.manifest = await loadJson("manifest.json");
  state.rows = await loadJson("method_results.json");
  state.maneuvers = await loadJson("maneuver_summary.json");
  state.playback = await loadJson("playback.json");
  state.methodTraces = await loadJson("method_traces.json");
  if (!state.manifest.model_families.includes(state.modelFamily)) {
    state.modelFamily = state.manifest.model_families[0] || "aircraft3dof";
  }
  setDefaultScenario();
  bindControls();
  renderMeta();
  render();
}

init().catch((error) => {
  console.error(error);
  document.querySelector("#run-meta").textContent = error.message;
});
