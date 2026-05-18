import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.165.0/build/three.module.js";

const DATA_DIR = "./public/data";

const state = {
  manifest: null,
  rows: [],
  maneuvers: [],
  playback: [],
  playbackScene: null,
  modelFamily: "aircraft3dof",
  scenario: "",
  source: "direct",
  uploadRows: [],
  uploadScenarios: [],
  uploadDatasets: [],
  lastValidatedDataset: null,
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

function setDefaultScenario() {
  const scenarios = scenariosForModel();
  if (!scenarios.some((scenario) => scenario.id === state.scenario)) {
    state.scenario = scenarios[0]?.id || "";
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
    render();
  });

  document.querySelector("#source-filter").addEventListener("click", (event) => {
    const button = event.target.closest("button[data-source]");
    if (!button) return;
    state.source = button.dataset.source;
    render();
  });

  document.querySelector("#upload-run").addEventListener("click", analyzeUpload);
  document.querySelector("#method-command").addEventListener("click", renderMethodCommand);
  document.querySelector("#simulate-command").addEventListener("click", renderSimCommand);
  document.querySelector("#upload-data-family").addEventListener("change", (event) => {
    state.modelFamily = event.target.value;
    setDefaultScenario();
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
  add("text", { x: 18, y: margin.top + plotHeight / 2, transform: `rotate(-90 18 ${margin.top + plotHeight / 2})`, "text-anchor": "middle", class: "axis-label" }, "validation score");

  for (const row of rows) {
    const xValue = row.train_elapsed_s || row.total_elapsed_s || row.rollout_elapsed_s || 0.01;
    const x = logScale(xValue, xExtent[0], xExtent[1], margin.left, margin.left + plotWidth);
    const y = logScale(row.validation_score, yExtent[0], yExtent[1], margin.top + plotHeight, margin.top);
    const isNominal = cleanMethodName(row.method).includes("Nominal");
    const circle = add("circle", {
      cx: x,
      cy: y,
      r: isNominal ? 6.5 : 5.5,
      fill: isNominal ? "white" : color,
      stroke: isNominal ? "var(--nominal)" : "#1d2430",
      "stroke-width": isNominal ? 1.8 : 1,
      opacity: 0.86,
    });
    const title = document.createElementNS("http://www.w3.org/2000/svg", "title");
    title.textContent = `${cleanMethodName(row.method)} | ${formatNumber(row.validation_score)}`;
    circle.append(title);
  }

  for (const row of rows.slice(0, 8)) {
    const xValue = row.train_elapsed_s || row.total_elapsed_s || row.rollout_elapsed_s || 0.01;
    const x = logScale(xValue, xExtent[0], xExtent[1], margin.left, margin.left + plotWidth);
    const y = logScale(row.validation_score, yExtent[0], yExtent[1], margin.top + plotHeight, margin.top);
    add("text", { x: x + 8, y: y - 8, class: "point-label" }, cleanMethodName(row.method));
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

function makeAircraftMesh() {
  const group = new THREE.Group();
  const bodyMaterial = new THREE.MeshStandardMaterial({ color: 0x2f5f9f, roughness: 0.46, metalness: 0.08 });
  const wingMaterial = new THREE.MeshStandardMaterial({ color: 0xd9e2ef, roughness: 0.55, metalness: 0.04 });
  const accentMaterial = new THREE.MeshStandardMaterial({ color: 0xd97706, roughness: 0.5, metalness: 0.02 });

  const fuselage = new THREE.Mesh(new THREE.BoxGeometry(1.35, 0.18, 0.16), bodyMaterial);
  group.add(fuselage);
  const nose = new THREE.Mesh(new THREE.ConeGeometry(0.12, 0.32, 24), accentMaterial);
  nose.rotation.z = -Math.PI / 2;
  nose.position.x = 0.84;
  group.add(nose);
  const wing = new THREE.Mesh(new THREE.BoxGeometry(0.18, 0.035, 1.95), wingMaterial);
  wing.position.x = 0.02;
  group.add(wing);
  const tail = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.03, 0.72), wingMaterial);
  tail.position.x = -0.58;
  group.add(tail);
  const fin = new THREE.Mesh(new THREE.BoxGeometry(0.14, 0.36, 0.035), accentMaterial);
  fin.position.set(-0.56, 0.18, 0);
  group.add(fin);
  return group;
}

function ensurePlaybackScene() {
  if (state.playbackScene) return state.playbackScene;
  const host = document.querySelector("#aircraft-playback");
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(host.clientWidth || 900, host.clientHeight || 360);
  renderer.setClearColor(0xf4f7fb, 1);
  host.append(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 1000);
  camera.position.set(8, 5, 9);
  camera.lookAt(0, 0, 0);
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
    trackLine: null,
    track: null,
    startMs: performance.now(),
  };

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

function setPlaybackTrack(track) {
  const playback = ensurePlaybackScene();
  if (playback.track?.id === track?.id) return;
  playback.track = track;
  playback.startMs = performance.now();
  if (playback.trackLine) {
    playback.scene.remove(playback.trackLine);
    playback.trackLine.geometry.dispose();
    playback.trackLine.material.dispose();
    playback.trackLine = null;
  }
  if (!track) return;
  const points = track.position_enu_m.map(enuToThree);
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  const material = new THREE.LineBasicMaterial({ color: 0x1f6feb, linewidth: 2 });
  playback.trackLine = new THREE.Line(geometry, material);
  playback.scene.add(playback.trackLine);
  const box = new THREE.Box3().setFromPoints(points);
  const center = box.getCenter(new THREE.Vector3());
  const size = Math.max(box.getSize(new THREE.Vector3()).length(), 1);
  playback.camera.position.copy(center).add(new THREE.Vector3(size * 0.55, size * 0.38, size * 0.7));
  playback.camera.lookAt(center);
  resizePlayback();
}

function sampleTrack(track, elapsedS) {
  const times = track.time_s;
  const duration = Math.max(times[times.length - 1] || 1, 1);
  const t = elapsedS % duration;
  let index = 0;
  while (index < times.length - 2 && times[index + 1] < t) index += 1;
  const t0 = times[index];
  const t1 = times[index + 1] ?? t0 + 1;
  const ratio = Math.max(0, Math.min(1, (t - t0) / Math.max(t1 - t0, 1e-9)));
  const p0 = enuToThree(track.position_enu_m[index]);
  const p1 = enuToThree(track.position_enu_m[index + 1] || track.position_enu_m[index]);
  const q0 = track.quaternion_wxyz[index];
  const q1 = track.quaternion_wxyz[index + 1] || q0;
  const quat0 = new THREE.Quaternion(q0[1], q0[2], q0[3], q0[0]).normalize();
  const quat1 = new THREE.Quaternion(q1[1], q1[2], q1[3], q1[0]).normalize();
  return { position: p0.lerp(p1, ratio), quaternion: quat0.slerp(quat1, ratio) };
}

function tickPlayback(nowMs) {
  const playback = state.playbackScene;
  if (playback?.track) {
    const sample = sampleTrack(playback.track, (nowMs - playback.startMs) / 1000);
    playback.aircraft.position.copy(sample.position);
    playback.aircraft.quaternion.copy(sample.quaternion);
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
  status.textContent = `${track.title} | ${track.source}`;
  setPlaybackTrack(track);
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
  if (!file) {
    status.textContent = "No file selected.";
    return;
  }
  try {
    const manifest = await readDatasetManifest(file);
    const errors = validateUploadedDataset(manifest, modelFamily);
    if (errors.length) {
      state.lastValidatedDataset = null;
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
    status.textContent = `Valid ${modelFamily} compact dataset: ${file.name}`;
    render();
  } catch (error) {
    state.lastValidatedDataset = null;
    status.textContent = error.message;
  }
}

function renderMethodCommand() {
  const output = document.querySelector("#command-output");
  const method = selectedMethodMetadata();
  const dataset = state.lastValidatedDataset || { id: state.scenario, fileName: "<dataset-id-or-flat-npz>", modelFamily: state.modelFamily };
  if (!method) {
    output.textContent = "Select a method first.";
    return;
  }
  const datasetArg = dataset.id.startsWith("upload_") ? `<validated-flat-npz-for-${dataset.fileName}>` : dataset.id;
  if (dataset.modelFamily === "aircraft6dof" || state.modelFamily === "aircraft6dof") {
    output.textContent = `python3 -m models.aircraft6dof.comparison_suite --dataset ${datasetArg} --state-source ${state.source}`;
  } else if (dataset.id.startsWith("upload_")) {
    output.textContent = `python3 comparison_suite.py --dataset ${datasetArg} --include-methods "${method.name}" --state-source ${state.source}`;
  } else {
    output.textContent = `./results.py suite --dataset-modes ${datasetArg} --include-methods "${method.name}" --state-source ${state.source}`;
  }
}

function renderSimCommand() {
  const output = document.querySelector("#command-output");
  if (state.modelFamily === "aircraft6dof") {
    output.textContent = "./results.py simulate-6dof --dataset-mode aggressive --train-trials 32 --validation-trials 8";
  } else {
    output.textContent = "./results.py compact-3dof --dataset-mode open_loop --train-trials 64 --validation-trials 16";
  }
}

function render() {
  setDefaultScenario();
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
}

async function init() {
  state.manifest = await loadJson("manifest.json");
  state.rows = await loadJson("method_results.json");
  state.maneuvers = await loadJson("maneuver_summary.json");
  state.playback = await loadJson("playback.json");
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
