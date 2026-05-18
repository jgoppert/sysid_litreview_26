import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.165.0/build/three.module.js";

const DATA_DIR = "./public/data";

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
  selectedMethods: new Set(),
  modelFamily: "aircraft3dof",
  scenario: "",
  source: "direct",
  uploadRows: [],
  uploadScenarios: [],
  uploadDatasets: [],
  lastValidatedDataset: null,
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

function slug(value) {
  return String(value || "uploaded")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, 48) || "uploaded";
}

async function loadJson(name) {
  const response = await fetch(`${DATA_DIR}/${name}`);
  if (!response.ok) throw new Error(`failed to load ${name}: ${response.status}`);
  return response.json();
}

function allRows() {
  return [...state.rows, ...state.uploadRows];
}

function allScenarios() {
  return [...state.manifest.scenarios, ...state.uploadScenarios];
}

function allDatasets() {
  return [...state.manifest.dataset_registry, ...state.uploadDatasets];
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

function selectedTraceSegments() {
  const keys = state.selectedMethods;
  if (!keys.size) return [];
  return state.methodTraces
    .filter((trace) => trace.scenario === state.scenario && trace.state_source === state.source && keys.has(methodKey(trace.method)))
    .map((trace) => {
      const segments = trace.segments || [];
      const segment = segments[state.playbackSegmentIndex] || segments[0];
      return segment ? { ...segment, method: methodKey(trace.method) } : null;
    })
    .filter(Boolean);
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

function renderMethodSelect() {
  const select = document.querySelector("#upload-method");
  const methods = state.manifest.method_registry
    .filter((method) => method.model_families.includes(state.modelFamily))
    .map((method) => method.name)
    .sort();
  select.innerHTML = "";
  for (const method of methods) {
    const option = document.createElement("option");
    option.value = method;
    option.textContent = method;
    select.append(option);
  }
}

function selectedMethodMetadata() {
  const name = document.querySelector("#upload-method").value;
  return state.manifest.method_registry.find((method) => method.name === name) || null;
}

function bindControls() {
  document.querySelector("#scenario-select").addEventListener("change", (event) => {
    state.scenario = event.target.value;
    state.playbackSegmentIndex = 0;
    state.selectedMethods.clear();
    render();
  });

  document.querySelector("#source-filter").addEventListener("click", (event) => {
    const button = event.target.closest("button[data-source]");
    if (!button) return;
    state.source = button.dataset.source;
    state.selectedMethods.clear();
    render();
  });

  document.querySelector("#playback-tabs").addEventListener("click", (event) => {
    const button = event.target.closest("button[data-playback-view]");
    if (!button) return;
    state.playbackView = button.dataset.playbackView;
    renderPlaybackTabs();
    if (state.playbackView === "animation") resizePlayback();
  });

  document.querySelector("#upload-run").addEventListener("click", analyzeUpload);
  document.querySelector("#method-command").addEventListener("click", renderMethodCommand);
  document.querySelector("#simulate-command").addEventListener("click", renderSimCommand);
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
  document.querySelector("#playback-speed").addEventListener("change", (event) => {
    state.playbackSpeed = Number.parseFloat(event.target.value) || 1;
  });
  document.querySelector("#playback-segment").addEventListener("change", (event) => {
    state.playbackSegmentIndex = Number.parseInt(event.target.value, 10) || 0;
    state.playbackTimeS = 0;
    setPlaybackTrack(selectedPlayback(), true);
    renderTimeseries();
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
  document.querySelector("#upload-data-family").addEventListener("change", (event) => {
    state.modelFamily = event.target.value;
    state.selectedMethods.clear();
    setDefaultScenario();
    render();
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

function powers(min, max) {
  const out = [];
  for (let power = Math.floor(Math.log10(min)); power <= Math.ceil(Math.log10(max)); power += 1) {
    out.push(10 ** power);
  }
  return out;
}

function setActionProgress(percent, message) {
  const bar = document.querySelector("#action-progress-bar");
  const status = document.querySelector("#upload-status");
  if (bar) bar.style.width = `${clamp(percent, 0, 100)}%`;
  if (status && message) status.textContent = message;
}

function renderTradeoff(rows) {
  const host = document.querySelector("#tradeoff-plot");
  host.innerHTML = "";
  const width = Math.max(host.clientWidth || 900, 640);
  const height = 410;
  const margin = { top: 20, right: 34, bottom: 54, left: 76 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const xExtent = logExtent(rows.map((row) => row.train_elapsed_s || row.total_elapsed_s || row.rollout_elapsed_s));
  const yExtent = logExtent(rows.map((row) => row.validation_score));
  const color = state.source === "direct" ? "var(--direct)" : "var(--mocap)";

  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);

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
    const selected = state.selectedMethods.has(key);
    const circle = add("circle", {
      cx: x,
      cy: y,
      r: selected ? 8.0 : isNominal ? 6.5 : 5.5,
      fill: isNominal ? "white" : color,
      stroke: selected ? "#111827" : isNominal ? "var(--nominal)" : "#1d2430",
      "stroke-width": selected ? 2.6 : isNominal ? 1.8 : 1,
      opacity: state.selectedMethods.size && !selected ? 0.34 : 0.88,
      class: "tradeoff-point",
      tabindex: "0",
    });
    circle.addEventListener("click", () => toggleMethodSelection(key));
    const title = document.createElementNS("http://www.w3.org/2000/svg", "title");
    title.textContent = `${cleanMethodName(row.method)} | ${formatNumber(row.validation_score)}`;
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
    tr.className = state.selectedMethods.has(key) ? "selected-method-row" : "";
    tr.addEventListener("click", () => toggleMethodSelection(key));
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

function toggleMethodSelection(key) {
  if (state.selectedMethods.has(key)) {
    state.selectedMethods.delete(key);
  } else {
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
  state.playbackTimeS = ((timeS % playbackDuration(segment)) + playbackDuration(segment)) % playbackDuration(segment);
  state.playbackLastMs = null;
  updatePlaybackScrub(segment);
}

function makeAircraftMesh() {
  const group = new THREE.Group();
  const bodyMaterial = new THREE.MeshStandardMaterial({ color: 0x2f5f9f, roughness: 0.46, metalness: 0.08 });
  const wingMaterial = new THREE.MeshStandardMaterial({ color: 0xd9e2ef, roughness: 0.55, metalness: 0.04 });
  const accentMaterial = new THREE.MeshStandardMaterial({ color: 0xd97706, roughness: 0.5, metalness: 0.02 });
  const controlMaterial = new THREE.MeshStandardMaterial({ color: 0xff8a1f, roughness: 0.42, metalness: 0.02 });
  const propMaterial = new THREE.MeshStandardMaterial({ color: 0x2b3442, roughness: 0.35, metalness: 0.08 });
  const propDiskMaterial = new THREE.MeshBasicMaterial({ color: 0x7fb4ff, transparent: true, opacity: 0.16, side: THREE.DoubleSide });

  const fuselage = new THREE.Mesh(new THREE.BoxGeometry(1.35, 0.18, 0.16), bodyMaterial);
  group.add(fuselage);
  const nose = new THREE.Mesh(new THREE.ConeGeometry(0.12, 0.32, 24), accentMaterial);
  nose.rotation.z = -Math.PI / 2;
  nose.position.x = 0.84;
  group.add(nose);
  const wingCenter = new THREE.Mesh(new THREE.BoxGeometry(0.18, 0.035, 0.85), wingMaterial);
  wingCenter.position.x = 0.02;
  group.add(wingCenter);
  const leftWing = new THREE.Mesh(new THREE.BoxGeometry(0.18, 0.035, 0.42), wingMaterial);
  leftWing.position.set(0.02, 0, -0.64);
  group.add(leftWing);
  const rightWing = new THREE.Mesh(new THREE.BoxGeometry(0.18, 0.035, 0.42), wingMaterial);
  rightWing.position.set(0.02, 0, 0.64);
  group.add(rightWing);

  const leftAileron = new THREE.Mesh(new THREE.BoxGeometry(0.28, 0.05, 0.72), controlMaterial);
  leftAileron.position.set(-0.18, -0.01, -0.8);
  group.add(leftAileron);
  const rightAileron = new THREE.Mesh(new THREE.BoxGeometry(0.28, 0.05, 0.72), controlMaterial);
  rightAileron.position.set(-0.18, -0.01, 0.8);
  group.add(rightAileron);

  const tail = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.03, 0.42), wingMaterial);
  tail.position.x = -0.58;
  group.add(tail);
  const elevator = new THREE.Mesh(new THREE.BoxGeometry(0.3, 0.05, 0.95), controlMaterial);
  elevator.position.set(-0.78, -0.008, 0);
  group.add(elevator);
  const fin = new THREE.Mesh(new THREE.BoxGeometry(0.14, 0.3, 0.035), accentMaterial);
  fin.position.set(-0.56, 0.18, 0);
  group.add(fin);
  const rudder = new THREE.Mesh(new THREE.BoxGeometry(0.14, 0.52, 0.06), controlMaterial);
  rudder.position.set(-0.74, 0.25, 0);
  group.add(rudder);

  const prop = new THREE.Group();
  prop.position.x = 1.03;
  const bladeA = new THREE.Mesh(new THREE.BoxGeometry(0.018, 0.72, 0.045), propMaterial);
  const bladeB = new THREE.Mesh(new THREE.BoxGeometry(0.018, 0.045, 0.72), propMaterial);
  const disk = new THREE.Mesh(new THREE.CircleGeometry(0.42, 48), propDiskMaterial);
  disk.rotation.y = Math.PI / 2;
  prop.add(bladeA, bladeB, disk);
  group.add(prop);
  group.userData = { leftAileron, rightAileron, elevator, rudder, prop, propDisk: disk };
  return group;
}

function updateAircraftControls(aircraft, controls, deltaS) {
  const parts = aircraft.userData || {};
  const thrust = clamp(controls[0] ?? 0.45, 0, 1);
  const aileron = clamp(controls[1] ?? 0, -1, 1);
  const elevator = clamp(controls[2] ?? 0, -1, 1);
  const rudder = clamp(controls[3] ?? 0, -1, 1);
  if (parts.leftAileron) parts.leftAileron.rotation.z = 0.95 * aileron;
  if (parts.rightAileron) parts.rightAileron.rotation.z = -0.95 * aileron;
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
  renderer.setClearColor(0xf4f7fb, 1);
  host.append(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 1000);
  scene.add(new THREE.HemisphereLight(0xffffff, 0xc8d1dd, 2.2));
  const sun = new THREE.DirectionalLight(0xffffff, 1.8);
  sun.position.set(6, 10, 8);
  scene.add(sun);
  const grid = new THREE.GridHelper(18, 18, 0xc7d1df, 0xe1e7f0);
  scene.add(grid);
  const aircraft = makeAircraftMesh();
  scene.add(aircraft);

  state.playbackScene = {
    renderer,
    scene,
    camera,
    aircraft,
    grid,
    trackLine: null,
    methodLines: [],
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
  if (!track || !segment) return;
  const points = segment.position_enu_m.map(enuToThree);
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  const material = new THREE.LineBasicMaterial({ color: 0x1f6feb, linewidth: 2 });
  playback.trackLine = new THREE.Line(geometry, material);
  playback.scene.add(playback.trackLine);
  const methodColors = [0x111827, 0x7c3aed, 0x059669, 0xb45309, 0xbe123c];
  selectedTraceSegments().forEach((trace, index) => {
    if (!trace.position_enu_m?.length) return;
    const tracePoints = trace.position_enu_m.map(enuToThree);
    const traceGeometry = new THREE.BufferGeometry().setFromPoints(tracePoints);
    const traceMaterial = new THREE.LineBasicMaterial({
      color: methodColors[index % methodColors.length],
      linewidth: 2,
      transparent: true,
      opacity: 0.78,
    });
    const traceLine = new THREE.Line(traceGeometry, traceMaterial);
    playback.methodLines.push(traceLine);
    playback.scene.add(traceLine);
  });
  const box = new THREE.Box3().setFromPoints(points);
  const center = box.getCenter(new THREE.Vector3());
  const extents = box.getSize(new THREE.Vector3());
  const size = Math.max(extents.x, extents.y, extents.z, 1);
  playback.controls.target.copy(center);
  playback.controls.distance = clamp(size * 2.2, 4, 120);
  playback.controls.minDistance = Math.max(size * 0.18, 0.8);
  playback.controls.maxDistance = Math.max(size * 8, 20);
  if (playback.grid) {
    playback.scene.remove(playback.grid);
    playback.grid.geometry.dispose();
    disposeMaterial(playback.grid.material);
  }
  const gridSize = Math.max(10, Math.ceil(size * 1.5));
  playback.grid = new THREE.GridHelper(gridSize, 20, 0xc7d1df, 0xe1e7f0);
  playback.grid.position.copy(center);
  playback.grid.position.y = Math.min(...points.map((point) => point.y));
  playback.scene.add(playback.grid);
  updatePlaybackCamera(playback);
  resizePlayback();
  renderPlaybackControls(track);
}

function sampleTrack(trackOrSegment, elapsedS) {
  const track = trackOrSegment?.segments ? activeSegment(trackOrSegment) : trackOrSegment;
  if (!track) return null;
  const times = track.time_s;
  const duration = Math.max(times[times.length - 1] || 1, 1);
  const t = ((elapsedS % duration) + duration) % duration;
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
  const speed = document.querySelector("#playback-speed");
  const segmentSelect = document.querySelector("#playback-segment");
  if (toggle) {
    toggle.textContent = state.playbackPlaying ? "||" : ">";
    toggle.setAttribute("aria-label", state.playbackPlaying ? "Pause" : "Play");
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
    const sample = sampleTrack(segment, state.playbackTimeS);
    if (sample) {
      playback.aircraft.position.copy(sample.position);
      playback.aircraft.quaternion.copy(sample.quaternion);
      updateAircraftControls(playback.aircraft, sample.controls, deltaS);
      updateControlHud(sample.controls);
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
  const margin = { top: 24, right: 18, bottom: 26, left: 42 };
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
  add("text", { x: margin.left, y: 15, class: "series-title" }, title);
  for (let i = 0; i <= 3; i += 1) {
    const y = margin.top + (i / 3) * (height - margin.top - margin.bottom);
    add("line", { x1: margin.left, y1: y, x2: width - margin.right, y2: y, class: "grid-line" });
  }
  const path = (time, data) => time.map((t, index) => `${index ? "L" : "M"}${xScale(t).toFixed(2)},${yScale(data[index]).toFixed(2)}`).join(" ");
  add("path", { d: path(series.time, series.values), class: "truth-series" });
  for (const trace of traces) {
    add("path", { d: path(trace.time, trace.values), class: "method-series" });
  }
  add("text", { x: margin.left, y: height - 7, class: "tick" }, `${formatNumber(xExtent[0])} s`);
  add("text", { x: width - margin.right, y: height - 7, "text-anchor": "end", class: "tick" }, `${formatNumber(xExtent[1])} s`);
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
    ["x east", pose.map((row) => row[0]), (trace) => trace.position_enu_m?.map((row) => row[0])],
    ["y north", pose.map((row) => row[1]), (trace) => trace.position_enu_m?.map((row) => row[1])],
    ["z up", pose.map((row) => row[2]), (trace) => trace.position_enu_m?.map((row) => row[2])],
    ["roll deg", eulerDeg.map((row) => row[0]), (trace) => trace.quaternion_wxyz?.map((row) => quaternionToEulerDeg(row)[0])],
    ["pitch deg", eulerDeg.map((row) => row[1]), (trace) => trace.quaternion_wxyz?.map((row) => quaternionToEulerDeg(row)[1])],
    ["yaw deg", eulerDeg.map((row) => row[2]), (trace) => trace.quaternion_wxyz?.map((row) => quaternionToEulerDeg(row)[2])],
    ["thrust", controls.map((row) => row[0] ?? 0), (trace) => trace.control_meas?.map((row) => row[0] ?? 0)],
    ["aileron", controls.map((row) => row[1] ?? 0), (trace) => trace.control_meas?.map((row) => row[1] ?? 0)],
    ["elevator", controls.map((row) => row[2] ?? 0), (trace) => trace.control_meas?.map((row) => row[2] ?? 0)],
    ["rudder", controls.map((row) => row[3] ?? 0), (trace) => trace.control_meas?.map((row) => row[3] ?? 0)],
  ].filter((item) => item[1].length);
  for (const [title, values, traceAccessor] of definitions) {
    const traces = traceSegments
      .map((trace) => ({ method: trace.method, time: trace.time_s, values: traceAccessor(trace) }))
      .filter((trace) => trace.values?.length === trace.time?.length);
    host.append(renderMiniSeries(title, { time, values }, traces));
  }
  const selected = state.selectedMethods.size ? `${state.selectedMethods.size} selected method(s)` : "select methods to overlay exported rollouts";
  const available = traceSegments.length ? "method rollouts shown" : "no exported method rollout traces for this dataset yet";
  status.textContent = `${segment.name || "flight"} | ${selected} | ${available}`;
}

const FORMAT_VERSION = "sysid.timeseries.ragged.v1";
const REQUIRED_SPLIT_KEYS = [
  "time_s",
  "valid_mask",
  "control_meas",
  "pose_meas",
  "segment_names",
  "control_names",
  "pose_names",
  "system_dof",
  "format_version",
  "dataset_id",
  "split_name",
  "sample_period_s",
  "truth_available",
];
const POSE_NAMES_BY_MODEL = {
  aircraft3dof: ["x_e", "z_u", "theta"],
  aircraft6dof: ["x_e", "y_n", "z_u", "q_w", "q_x", "q_y", "q_z"],
};
const DIRECT_STATE_NAMES_BY_MODEL = {
  aircraft3dof: ["V", "alpha", "gamma", "q"],
  aircraft6dof: ["x_n", "y_e", "z_d", "u", "v", "w", "q_w", "q_x", "q_y", "q_z", "p", "q", "r"],
};

function arrayShape(value) {
  const shape = [];
  let current = value;
  while (Array.isArray(current)) {
    shape.push(current.length);
    current = current[0];
  }
  return shape;
}

function product(values) {
  return values.reduce((total, value) => total * value, 1);
}

function parseFixedWidthStrings(bytes, dtype, count) {
  if (dtype.includes("U")) {
    const width = Number.parseInt(dtype.split("U").pop(), 10);
    const out = [];
    for (let item = 0; item < count; item += 1) {
      let text = "";
      for (let char = 0; char < width; char += 1) {
        const offset = (item * width + char) * 4;
        const codepoint = bytes[offset] | (bytes[offset + 1] << 8) | (bytes[offset + 2] << 16) | (bytes[offset + 3] << 24);
        if (codepoint) text += String.fromCodePoint(codepoint);
      }
      out.push(text);
    }
    return out;
  }
  if (dtype.includes("S")) {
    const width = Number.parseInt(dtype.split("S").pop(), 10);
    const decoder = new TextDecoder("latin1");
    return Array.from({ length: count }, (_unused, item) => decoder.decode(bytes.slice(item * width, (item + 1) * width)).replace(/\0+$/, ""));
  }
  return null;
}

function parseSmallArray(bytes, dtype, shape) {
  const count = product(shape.length ? shape : [1]);
  if (count > 64) return undefined;
  const strings = parseFixedWidthStrings(bytes, dtype, count);
  if (strings) return shape.length === 0 ? strings[0] : strings;
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (dtype === "|b1" || dtype === "?") {
    const values = Array.from({ length: count }, (_unused, index) => Boolean(view.getUint8(index)));
    return shape.length === 0 ? values[0] : values;
  }
  const little = dtype.startsWith("<") || dtype.startsWith("|");
  const kind = dtype.slice(-2);
  const readers = {
    i1: (offset) => view.getInt8(offset),
    u1: (offset) => view.getUint8(offset),
    i2: (offset) => view.getInt16(offset, little),
    u2: (offset) => view.getUint16(offset, little),
    i4: (offset) => view.getInt32(offset, little),
    u4: (offset) => view.getUint32(offset, little),
    f4: (offset) => view.getFloat32(offset, little),
    f8: (offset) => view.getFloat64(offset, little),
  };
  const sizes = { i1: 1, u1: 1, i2: 2, u2: 2, i4: 4, u4: 4, f4: 4, f8: 8 };
  const reader = readers[kind];
  if (!reader) return undefined;
  const values = Array.from({ length: count }, (_unused, index) => reader(index * sizes[kind]));
  return shape.length === 0 ? values[0] : values;
}

function parseNpyHeader(buffer, name) {
  const bytes = new Uint8Array(buffer);
  if (bytes[0] !== 0x93 || String.fromCharCode(...bytes.slice(1, 6)) !== "NUMPY") {
    throw new Error(`${name} is not a .npy entry`);
  }
  const view = new DataView(buffer);
  const major = bytes[6];
  const headerLength = major === 1 ? view.getUint16(8, true) : view.getUint32(8, true);
  const headerOffset = major === 1 ? 10 : 12;
  const header = new TextDecoder("latin1").decode(bytes.slice(headerOffset, headerOffset + headerLength));
  const dtype = header.match(/'descr':\s*'([^']+)'/)?.[1] || "";
  const shapeText = header.match(/'shape':\s*\(([^)]*)\)/)?.[1] || "";
  const shape = shapeText
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean)
    .map((item) => Number.parseInt(item, 10));
  const dataOffset = headerOffset + headerLength;
  const value = parseSmallArray(bytes.slice(dataOffset), dtype, shape);
  return { name: name.replace(/\.npy$/, ""), dtype, shape, value };
}

async function inflateRaw(bytes) {
  if (!("DecompressionStream" in window)) {
    throw new Error("NPZ validation needs a browser with DecompressionStream support; upload JSON instead.");
  }
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("deflate-raw"));
  return new Response(stream).arrayBuffer();
}

async function readNpzManifest(file) {
  const buffer = await file.arrayBuffer();
  const bytes = new Uint8Array(buffer);
  const view = new DataView(buffer);
  let eocd = -1;
  for (let index = bytes.length - 22; index >= Math.max(0, bytes.length - 65557); index -= 1) {
    if (view.getUint32(index, true) === 0x06054b50) {
      eocd = index;
      break;
    }
  }
  if (eocd < 0) throw new Error("NPZ zip footer not found");
  const entries = view.getUint16(eocd + 10, true);
  let cursor = view.getUint32(eocd + 16, true);
  const manifest = {};
  for (let entry = 0; entry < entries; entry += 1) {
    if (view.getUint32(cursor, true) !== 0x02014b50) throw new Error("NPZ central directory is invalid");
    const method = view.getUint16(cursor + 10, true);
    const compressedSize = view.getUint32(cursor + 20, true);
    const fileNameLength = view.getUint16(cursor + 28, true);
    const extraLength = view.getUint16(cursor + 30, true);
    const commentLength = view.getUint16(cursor + 32, true);
    const localOffset = view.getUint32(cursor + 42, true);
    const filename = new TextDecoder().decode(bytes.slice(cursor + 46, cursor + 46 + fileNameLength));
    cursor += 46 + fileNameLength + extraLength + commentLength;
    if (!filename.endsWith(".npy")) continue;
    if (view.getUint32(localOffset, true) !== 0x04034b50) throw new Error(`bad local header for ${filename}`);
    const localNameLength = view.getUint16(localOffset + 26, true);
    const localExtraLength = view.getUint16(localOffset + 28, true);
    const dataStart = localOffset + 30 + localNameLength + localExtraLength;
    const compressed = bytes.slice(dataStart, dataStart + compressedSize);
    let arrayBuffer;
    if (method === 0) {
      arrayBuffer = compressed.buffer.slice(compressed.byteOffset, compressed.byteOffset + compressed.byteLength);
    } else if (method === 8) {
      arrayBuffer = await inflateRaw(compressed);
    } else {
      throw new Error(`unsupported NPZ compression method ${method} for ${filename}`);
    }
    const parsed = parseNpyHeader(arrayBuffer, filename);
    manifest[parsed.name] = parsed;
  }
  return manifest;
}

async function readDatasetManifest(file) {
  if (file.name.toLowerCase().endsWith(".npz")) return readNpzManifest(file);
  const parsed = JSON.parse(await file.text());
  const manifest = {};
  for (const [key, value] of Object.entries(parsed)) {
    manifest[key] = { name: key, dtype: Array.isArray(value) ? "json-array" : typeof value, shape: arrayShape(value), value };
  }
  return manifest;
}

function validateUploadedDataset(manifest, modelFamily) {
  const errors = [];
  const expectedPoseNames = POSE_NAMES_BY_MODEL[modelFamily];
  const expectedDirectNames = DIRECT_STATE_NAMES_BY_MODEL[modelFamily];
  const expectedDof = modelFamily === "aircraft6dof" ? 6 : 3;
  for (const key of REQUIRED_SPLIT_KEYS) {
    if (!manifest[key]) errors.push(`missing ${key}`);
  }
  const timeShape = manifest.time_s?.shape || [];
  const validShape = manifest.valid_mask?.shape || [];
  const controlShape = manifest.control_meas?.shape || [];
  const poseShape = manifest.pose_meas?.shape || [];
  if (timeShape.length !== 2) errors.push("time_s must be [segment, sample]");
  if (validShape.join(",") !== timeShape.join(",")) errors.push("valid_mask shape must match time_s");
  if (controlShape.length !== 3 || controlShape[0] !== timeShape[0] || controlShape[1] !== timeShape[1] || controlShape[2] !== 4) {
    errors.push("control_meas must be [segment, sample, 4]");
  }
  if (poseShape.length !== 3 || poseShape[0] !== timeShape[0] || poseShape[1] !== timeShape[1] || poseShape[2] !== expectedPoseNames.length) {
    errors.push(`pose_meas must be [segment, sample, ${expectedPoseNames.length}] for ${modelFamily}`);
  }
  if (manifest.direct_state_meas) {
    const shape = manifest.direct_state_meas.shape || [];
    if (!manifest.direct_state_names) errors.push("direct_state_meas requires direct_state_names");
    if (shape.length !== 3 || shape[0] !== timeShape[0] || shape[1] !== timeShape[1] || shape[2] !== expectedDirectNames.length) {
      errors.push(`direct_state_meas must be [segment, sample, ${expectedDirectNames.length}] for ${modelFamily}`);
    }
  }
  if (manifest.format_version?.shape?.length > 1) errors.push(`format_version must be scalar ${FORMAT_VERSION}`);
  if (manifest.format_version?.value && manifest.format_version.value !== FORMAT_VERSION) {
    errors.push(`format_version must be ${FORMAT_VERSION}`);
  }
  if (manifest.split_name?.value && !["train", "validation"].includes(manifest.split_name.value)) {
    errors.push("split_name must be train or validation");
  }
  if (manifest.system_dof?.value && Number(manifest.system_dof.value) !== expectedDof) {
    errors.push(`system_dof must be ${expectedDof}`);
  }
  if (manifest.pose_names?.value && manifest.pose_names.value.join(",") !== expectedPoseNames.join(",")) {
    errors.push(`pose_names must be ${expectedPoseNames.join(", ")}`);
  }
  if (manifest.control_names?.value && manifest.control_names.value.join(",") !== ["thrust", "aileron", "elevator", "rudder"].join(",")) {
    errors.push("control_names must be thrust, aileron, elevator, rudder");
  }
  if (manifest.direct_state_names?.value && manifest.direct_state_names.value.join(",") !== expectedDirectNames.join(",")) {
    errors.push(`direct_state_names must be ${expectedDirectNames.join(", ")}`);
  }
  return errors;
}

async function analyzeUpload() {
  const input = document.querySelector("#upload-file");
  const status = document.querySelector("#upload-status");
  const modelFamily = document.querySelector("#upload-data-family").value;
  const file = input.files?.[0];
  setActionProgress(0, "");
  if (!file) {
    status.textContent = "No file selected.";
    return;
  }
  try {
    setActionProgress(25, "Reading compact dataset...");
    const manifest = await readDatasetManifest(file);
    setActionProgress(70, "Validating canonical schema...");
    const errors = validateUploadedDataset(manifest, modelFamily);
    if (errors.length) {
      state.lastValidatedDataset = null;
      setActionProgress(0, "");
      status.textContent = `Invalid ${modelFamily} dataset: ${errors.slice(0, 4).join("; ")}`;
      return;
    }
    const scenario = `upload_${slug(file.name)}`;
    const title = file.name.replace(/\.[^.]+$/, "");
    state.uploadScenarios = state.uploadScenarios.filter((item) => item.id !== scenario).concat({
      id: scenario,
      title,
      model_family: modelFamily,
      method_result_count: 0,
    });
    state.uploadDatasets = state.uploadDatasets.filter((item) => item.id !== scenario).concat({
      id: scenario,
      title,
      status: "uploaded_validated",
      model_family: modelFamily,
      source_type: "browser_upload",
      local_data_dir: file.name,
    });
    state.lastValidatedDataset = { id: scenario, title, fileName: file.name, modelFamily };
    state.modelFamily = modelFamily;
    state.scenario = scenario;
    setActionProgress(100, `Valid ${modelFamily} compact dataset: ${file.name}`);
    render();
  } catch (error) {
    state.lastValidatedDataset = null;
    setActionProgress(0, "");
    status.textContent = error.message;
  }
}

async function copyCommand(text) {
  if (!navigator.clipboard) return false;
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    return false;
  }
}

async function renderMethodCommand() {
  const output = document.querySelector("#command-output");
  const method = selectedMethodMetadata();
  const dataset = state.lastValidatedDataset || { id: state.scenario, fileName: "<dataset-id-or-flat-npz>", modelFamily: state.modelFamily };
  setActionProgress(15, "Preparing reproducible command...");
  if (!method) {
    setActionProgress(0, "");
    output.textContent = "Select a method first.";
    return;
  }
  const datasetArg = dataset.id.startsWith("upload_") ? `<validated-flat-npz-for-${dataset.fileName}>` : dataset.id;
  const warnings = [];
  if (method.requires_gpu) {
    warnings.push("This method is GPU-oriented; expect a slow run or dependency failures on CPU-only machines.");
  }
  if (method.heavy) {
    warnings.push("This method is marked as a heavier benchmark path; use the committed website snapshot unless you need fresh results.");
  }
  let command;
  if (dataset.modelFamily === "aircraft6dof" || state.modelFamily === "aircraft6dof") {
    command = `python3 -m models.aircraft6dof.comparison_suite --dataset ${datasetArg} --state-source ${state.source}`;
  } else if (dataset.id.startsWith("upload_")) {
    command = `python3 comparison_suite.py --dataset ${datasetArg} --include-methods "${method.name}" --state-source ${state.source}`;
  } else {
    command = `./results.py suite --dataset-modes ${datasetArg} --include-methods "${method.name}" --state-source ${state.source}`;
  }
  setActionProgress(75, "Command prepared; browser execution is not available on the static site.");
  const copied = await copyCommand(command);
  output.textContent = [
    "The website shows the committed benchmark snapshot by default.",
    "Static GitHub Pages cannot execute Python benchmark methods directly.",
    ...warnings,
    copied ? "Command copied to clipboard:" : "Run this command locally or in CI:",
    command,
  ].join("\n");
  setActionProgress(100, copied ? "Prepared and copied command." : "Prepared command.");
}

async function renderSimCommand() {
  const output = document.querySelector("#command-output");
  setActionProgress(40, "Preparing simulation command...");
  let command;
  if (state.modelFamily === "aircraft6dof") {
    command = "./results.py simulate-6dof --dataset-mode aggressive --train-trials 32 --validation-trials 8";
  } else {
    command = "./results.py compact-3dof --dataset-mode open_loop --train-trials 64 --validation-trials 16";
  }
  const copied = await copyCommand(command);
  output.textContent = [
    "Simulation datasets are generated on demand; committed results are the default snapshot.",
    copied ? "Command copied to clipboard:" : "Run this command locally or in CI:",
    command,
  ].join("\n");
  setActionProgress(100, copied ? "Prepared and copied simulation command." : "Prepared simulation command.");
}

function render() {
  setDefaultScenario();
  renderPlaybackTabs();
  renderModelTabs();
  renderScenarioSelect();
  document.querySelector("#upload-data-family").value = state.modelFamily;
  renderMethodSelect();
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
  state.modelFamily = state.manifest.model_families[0] || "aircraft3dof";
  setDefaultScenario();
  bindControls();
  renderMeta();
  render();
}

init().catch((error) => {
  console.error(error);
  document.querySelector("#run-meta").textContent = error.message;
});
