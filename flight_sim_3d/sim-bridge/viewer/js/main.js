// GA flight-sim trajectory viewer (three.js, no build step).
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { Line2 } from 'three/addons/lines/Line2.js';
import { LineGeometry } from 'three/addons/lines/LineGeometry.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';
import { fetchJSON, decodeBuffer, isIndex, parseIndex, parseTrajectory, locate, normSense, lerpCh, lerpAngleDeg, quatAt } from 'fv/traj.js';
import { resolvePreset, buildProcedural, buildGltf, applyControls, attachStructure, applyStructure } from 'fv/aircraft.js';

const $ = (id) => document.getElementById(id);
const FT = 0.3048, KT = 0.514444, R_EARTH = 6371008.8, D2R = Math.PI / 180;
const PALETTE = ['#ff6b35', '#2ec4b6', '#ffd23f', '#e71d36', '#9b5de5', '#00bbf9', '#f15bb5', '#8ac926', '#ff924c', '#d0d0d0'];
const SINGLE_COLOR = '#ff6b35';
const params = new URLSearchParams(location.search);
// embedders (standalone/Colab inline build) can preset URL params via window.FV_PARAMS
if (window.FV_PARAMS) for (const [k, v] of Object.entries(window.FV_PARAMS)) if (!params.has(k)) params.set(k, String(v));
const DEFAULT_RUNS_URL = params.get('runs') || '../data/runs/runs.json';

// ---------------------------------------------------------------- state
const S = {
  runs: [], indexUrl: null, indexMeta: {}, entries: [], cache: new Map(), localFiles: new Map(),
  mode: 'single', singleIdx: 0, compareSet: new Set(), shown: [], focus: 0,
  t: 0, tEnd: 1, playing: false, speed: 1, cam: 'chase', exag: 1, mscale: 1, spacing: 25, dofs: null,
  defl: 1,         // soft-body: structural deflection exaggeration (x)
  layout: 'true',  // 'true' = real positions | 'formation' = along-track distance equalised so all keep station
  vref: 'auto',    // 3D vertical reference: 'abs' (MSL) | 'rel' (relative to own trim altitude) | 'norm' (rel, steps scaled equal) | 'auto'
  cy: 'auto',      // chart: 'alt' | 'rel' | 'err' (vs step target) | 'rerr' (vs ramp reference) | 'pct' | 'nz' | 'kcas' | 'dkcas' | 'auto'
  presetRun: null, // multi-run indexes: run used by the last/first/pair presets
  ref: null, // reference origin {lat, lon, alt}
  ready: false, errors: [], buildToken: 0,
};

// ---------------------------------------------------------------- three setup
const canvas = $('scene');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
const scene = new THREE.Scene();
const SKY = new THREE.Color('#9fc8ee');
scene.background = SKY;
scene.fog = new THREE.Fog(SKY, 3000, 26000);
const camera = new THREE.PerspectiveCamera(55, 1, 0.5, 80000);
camera.position.set(-40, 15, 40);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.enabled = false;
scene.add(new THREE.HemisphereLight(0xdfefff, 0x3a4a2a, 1.4));
const sun = new THREE.DirectionalLight(0xffffff, 2.0);
sun.position.set(0.4, 1, 0.3);
scene.add(sun);

const world = new THREE.Group();
scene.add(world);
// ground + grid (placed once trajectories are known)
const ground = new THREE.Mesh(new THREE.PlaneGeometry(1, 1).rotateX(-Math.PI / 2),
  new THREE.MeshStandardMaterial({ color: 0x5f7f4a, roughness: 1 }));
world.add(ground);
let groundGrid = null, refGrid = null;
const targetPlane = new THREE.Mesh(new THREE.PlaneGeometry(1, 1).rotateX(-Math.PI / 2),
  new THREE.MeshBasicMaterial({ color: 0xffa040, transparent: true, opacity: 0.13, side: THREE.DoubleSide, depthWrite: false }));
const targetEdge = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.PlaneGeometry(1, 1).rotateX(-Math.PI / 2)),
  new THREE.LineBasicMaterial({ color: 0xffa040, transparent: true, opacity: 0.6 }));
targetPlane.add(targetEdge);
targetPlane.renderOrder = 2;
world.add(targetPlane);
const lineMats = new Set();

function resize() {
  const w = canvas.clientWidth || window.innerWidth, h = canvas.clientHeight || window.innerHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  for (const m of lineMats) m.resolution.set(w, h);
}
window.addEventListener('resize', resize);

// ---------------------------------------------------------------- messages
function msg(text, err = false) {
  const el = $('msg');
  el.textContent = text || '';
  el.className = 'small' + (err ? ' err' : '');
  if (err) { console.error(text); S.errors.push(text); }
}

// ---------------------------------------------------------------- loading
async function loadRunsList() {
  try {
    const runs = await fetchJSON(DEFAULT_RUNS_URL);
    const list = Array.isArray(runs) ? runs : (runs.runs || []);
    const base = new URL(DEFAULT_RUNS_URL, location.href);
    S.runs = list.map((r) => ({ ...r, url: new URL(r.index, base).href }));
  } catch (e) { S.runs = []; }
  const sel = $('run-select');
  sel.innerHTML = '<option value="">(choose run)</option>' + S.runs.map((r, i) =>
    `<option value="${i}">${r.run_id} · ${r.aircraft} · ${r.generations ?? '?'} gens</option>`).join('');
}

async function getTraj(entry) {
  const key = entry.key;
  if (!S.cache.has(key)) {
    const p = (async () => {
      let obj;
      if (S.localFiles.has(entry.file)) obj = await decodeBuffer(await S.localFiles.get(entry.file).arrayBuffer());
      else obj = await fetchJSON(entry.url);
      const tr = parseTrajectory(obj, entry.file);
      if (entry.fitness == null && tr.fitness != null) entry.fitness = tr.fitness;
      if (entry.generation == null && tr.generation != null) entry.generation = tr.generation;
      return tr;
    })();
    S.cache.set(key, p);
    p.catch(() => S.cache.delete(key));
  }
  return S.cache.get(key);
}

function setEntries(entries, meta, label) {
  S.entries = entries;
  S.indexMeta = meta || {};
  S.cache.clear();
  S.ref = null;
  const m = S.indexMeta;
  $('run-info').textContent = `${label}\n${entries.length} trajectories` +
    (m.aircraft ? ` · ${m.aircraft}` : '') + (m.seed != null ? ` · seed ${m.seed}` : '') +
    (m.jsbsim_version ? ` · JSBSim ${m.jsbsim_version}` : '');
  $('run-info').style.whiteSpace = 'pre-wrap';
  populateGenUI();
}

async function loadIndexUrl(url) {
  msg('loading ' + url + ' ...');
  const abs = new URL(url, location.href).href;
  const obj = await fetchJSON(abs);
  if (isIndex(obj)) {
    const { meta, entries } = parseIndex(obj, abs);
    S.indexUrl = abs;
    setEntries(entries, meta, meta.run_id ? `run ${meta.run_id}` : url.split('/').slice(-3).join('/'));
  } else { // a single trajectory URL
    const e = { generation: obj.generation ?? null, fitness: obj.cost ?? obj.fitness ?? null, sense: obj.fitness_sense ?? (obj.cost != null ? "min" : null), aircraft: obj.aircraft, file: abs.split('/').pop(), url: abs, key: abs };
    setEntries([e], { aircraft: obj.aircraft, run_id: obj.run_id, seed: obj.seed, jsbsim_version: obj.jsbsim_version }, abs.split('/').pop());
    S.cache.set(abs, Promise.resolve(parseTrajectory(obj, e.file)));
  }
  msg('');
}

async function loadTrajUrls(urls) {
  const entries = [];
  for (const u of urls) {
    const abs = new URL(u, location.href).href;
    const obj = await fetchJSON(abs);
    entries.push({ generation: obj.generation ?? null, fitness: obj.cost ?? obj.fitness ?? null, sense: obj.fitness_sense ?? (obj.cost != null ? "min" : null), aircraft: obj.aircraft, file: abs.split('/').pop(), url: abs, key: abs, _obj: obj });
  }
  setEntries(entries, { aircraft: entries[0]?.aircraft }, `${entries.length} file(s)`);
  for (const e of entries) { S.cache.set(e.key, Promise.resolve(parseTrajectory(e._obj, e.file))); delete e._obj; }
}

async function loadLocalFiles(files) {
  S.localFiles = new Map([...files].map((f) => [f.name, f]));
  let indexObj = null;
  const trajEntries = [];
  for (const f of files) {
    let obj;
    try { obj = await decodeBuffer(await f.arrayBuffer()); } catch (e) { msg(`${f.name}: ${e.message}`, true); continue; }
    if (isIndex(obj)) indexObj = obj;
    else trajEntries.push({ generation: obj.generation ?? null, fitness: obj.cost ?? obj.fitness ?? null, sense: obj.fitness_sense ?? (obj.cost != null ? "min" : null), aircraft: obj.aircraft, file: f.name, url: f.name, key: 'local:' + f.name, _obj: obj });
  }
  if (indexObj) {
    const { meta, entries } = parseIndex(indexObj, null);
    for (const e of entries) {
      const base = e.file.split('/').pop();
      const local = trajEntries.find((t) => t.file === base);
      e.key = 'local:' + base; e.file = base;
      if (local) e._obj = local._obj;
    }
    const usable = entries.filter((e) => e._obj);
    if (!usable.length) { msg('index.json picked, but none of its trajectory files were selected; select them together (multi-select) or load by URL.', true); return; }
    setEntries(usable, meta, 'local index');
    for (const e of usable) { S.cache.set(e.key, Promise.resolve(parseTrajectory(e._obj, e.file))); delete e._obj; }
  } else if (trajEntries.length) {
    trajEntries.sort((a, b) => (a.generation ?? 0) - (b.generation ?? 0));
    setEntries(trajEntries, { aircraft: trajEntries[0].aircraft }, 'local files');
    for (const e of trajEntries) { S.cache.set(e.key, Promise.resolve(parseTrajectory(e._obj, e.file))); delete e._obj; }
  }
  await rebuild(true);
}

// ---------------------------------------------------------------- generation UI
const multiAircraft = () => new Set(S.entries.map((e) => e.aircraft)).size > 1;
// multi-run (combined) indexes: runs in index order, short labels ("phase1-s2" -> "s2" when unique), families
const runList = () => [...new Set(S.entries.map((e) => e.run))];
const multiRun = () => runList().length > 1;
function runShort(run) {
  const runs = runList();
  const tail = (r) => String(r ?? '').split('-').pop();
  const tails = runs.map(tail);
  return new Set(tails).size === runs.length ? tail(run) : String(run ?? '');
}
const runFamily = (run) => { const r = String(run ?? ''); const k = r.lastIndexOf('-'); return k > 0 ? r.slice(0, k) : r; };
const runTag = (run) => (multiRun() ? runShort(run) + ' ' : '');
// optimisation sense: "min" (cost, ER default and every file so far) unless an index/file says "max"
const senseOf = (x) => normSense((x && x.sense) || S.indexMeta?.fitness_sense || 'min');
const metricName = (x) => (senseOf(x) === 'max' ? 'fitness' : 'cost');
const isBetter = (a, b, sense) => (sense === 'max' ? a > b + 1e-12 : a < b - 1e-12);
// replay indexes: several scenarios / individuals per generation -> tag them (sc<k>, r<rank>)
// ids are opaque (never parsed): a non-best individual is tagged r<rank> (or its full id), a scenario by its position
const scenKey = (e) => (e.scenarioIndex ?? e.scenario ?? null);
const multiScenario = () => new Set(S.entries.map(scenKey).filter((v) => v != null)).size > 1;
const indTag = (e) => {
  let t = '';
  if (e.isBest === false) t += ' ' + (e.rank != null ? 'r' + e.rank : String(e.individual ?? '?'));
  if (scenKey(e) != null && multiScenario()) t += ' sc' + scenKey(e);
  return t;
};
const genLabel = (e, i) => `${multiAircraft() && e.aircraft ? e.aircraft + ' ' : ''}${runTag(e.run)}${e.generation != null ? 'g' + String(e.generation).padStart(3, '0') : '#' + i}${indTag(e)}  ${e.fitness != null ? Number(e.fitness).toFixed(6) : '?'}${e.status && e.status !== 'ok' ? '  ✗ ' + e.status : ''}${e.verdict && e.verdict !== 'match' ? '  [' + e.verdict + ']' : ''}`;

function improvementPicks(max = 6) {
  const out = [];
  const best = {}; // per aircraft+run (respecting fitness_sense)
  S.entries.forEach((e, i) => {
    if (multiScenario() && scenKey(e) != null && String(scenKey(e)) !== String(scenKey(S.entries.find((x) => scenKey(x) != null)))) return;
    if (e.isBest === false) return;
    const k = (e.aircraft || '') + '|' + (e.run ?? '');
    if (e.fitness != null && (best[k] == null || isBetter(e.fitness, best[k], senseOf(e)))) { best[k] = e.fitness; out.push(i); }
  });
  if (!out.length) return spreadPicks(5);
  if (out.length <= max) return out;
  const picks = new Set([out[0], out[out.length - 1]]);
  for (let k = 1; picks.size < max && k < max; k++) picks.add(out[Math.round(k * (out.length - 1) / (max - 1))]);
  return [...picks].sort((a, b) => a - b);
}
function spreadPicks(k) {
  const n = S.entries.length;
  if (n <= k) return [...Array(n).keys()];
  return [...new Set([...Array(k).keys()].map((j) => Math.round(j * (n - 1) / (k - 1))))];
}

// token: "12" (generation), "t6texan2:12" (aircraft:generation), optional "@run" (full or short run id),
// optional "#sc<k>" (scenario position or id) / "#r<rank>" / "#best" / "#<individual_id>" (replay indexes)
function findEntry(tok) {
  let run = null, tag = null;
  if (tok.includes('@')) [tok, run] = tok.split('@');
  if (tok.includes('#')) [tok, tag] = tok.split('#'); // replay indexes: '#sc1', '#r2', '#r2.sc1'
  if (tag) {
    const want = tag.split('.');
    const [ac0, g0] = tok.includes(':') ? tok.split(':') : [null, tok];
    return S.entries.findIndex((e) => String(e.generation) === g0 && (!ac0 || e.aircraft === ac0) &&
      (!run || e.run === run || runShort(e.run) === run) &&
      want.every((w) => (w.startsWith('sc') ? (String(scenKey(e)) === w.slice(2) || String(e.scenario) === w.slice(2))
        : w === 'best' ? e.isBest !== false
          : /^r\d+$/.test(w) && e.rank != null ? 'r' + e.rank === w
            : w === 'r0' ? e.isBest !== false : String(e.individual) === w)));
  }
  const [ac, g] = tok.includes(':') ? tok.split(':') : [null, tok];
  const runOk = (e) => !run || e.run === run || runShort(e.run) === run || (!multiRun() && true);
  const pref = (e) => run || !multiRun() || e.run === S.presetRun;
  let j = S.entries.findIndex((e) => String(e.generation) === g && (!ac || e.aircraft === ac) && runOk(e) && pref(e));
  if (j < 0) j = S.entries.findIndex((e) => String(e.generation) === g && (!ac || e.aircraft === ac) && runOk(e));
  return j;
}

function populateGenUI() {
  const sel = $('gen-select');
  sel.innerHTML = S.entries.map((e, i) => `<option value="${i}">${genLabel(e, i)}</option>`).join('');
  const urlGen = params.get('gen');
  let si = S.entries.length - 1;
  if (urlGen != null) { const j = findEntry(urlGen); if (j >= 0) si = j; }
  S.singleIdx = Math.max(0, si);
  sel.value = String(S.singleIdx);
  const urlGens = params.get('gens');
  if (urlGens) S.compareSet = new Set(urlGens.split(',').map((g) => findEntry(g.trim())).filter((j) => j >= 0));
  if (urlGens && !S.compareSet.size) msg(`gens=${urlGens}: no matching entry; showing the default selection`, true);
  if (!urlGens || !S.compareSet.size) S.compareSet = new Set(improvementPicks());
  if (params.get('run')) { const r = params.get('run'); S.presetRun = runList().find((x) => x === r || runShort(x) === r) ?? null; }
  if (!S.presetRun) S.presetRun = runList()[0];
  if (params.get('preset') && !urlGens) applyPreset(params.get('preset')); // explicit gens= wins over a preset
  renderPresets();
  renderGenList();
}

// compare presets for multi-aircraft indexes: latest / first gen of each aircraft, and first-vs-latest per aircraft
// Multi-run (e.g. 3 seeds + an older run): a run selector drives last/first/pair; "seeds:<ac>" overlays the
// latest gen of <ac> from every run of the same family (phase1-s1/s2/s3); "cmp:<ac>" puts the latest gen of
// <ac> from each family side by side (e.g. bench_jets-j1 vs phase1-s1: before/after a fitness change).
function renderPresets() {
  const el = $('presets');
  const acs = [...new Set(S.entries.map((e) => e.aircraft))];
  const runs = runList();
  if (acs.length < 2 && runs.length < 2) { el.innerHTML = ''; return; }
  if (!runs.includes(S.presetRun)) S.presetRun = runs[0];
  const fam = runFamily(S.presetRun);
  const sameFam = runs.filter((r) => runFamily(r) === fam);
  const fams = [...new Set(runs.map(runFamily))];
  const btn = (p, label, title = '') => `<button data-p="${p}" title="${title}">${label}</button>`;
  const row = (label, html) => `<div class="prow"><span class="plabel">${label}</span>${html}</div>`;
  let h = '';
  if (runs.length > 1) h += row('run', `<select id="preset-run">${runs.map((r) => `<option value="${r}" ${r === S.presetRun ? 'selected' : ''}>${runShort(r)} = ${r}</option>`).join('')}</select>`);
  if (acs.length > 1) h += row('all', btn('last', 'latest gen', 'latest generation of every aircraft (selected run)') + btn('mid', 'mid gen', 'middle logged generation of every aircraft (selected run)') + btn('first', 'gen 0', 'first generation of every aircraft (selected run)'));
  h += row('gen 0 vs last', acs.map((a) => btn(`pair:${a}`, a, `first vs latest generation of ${a} (selected run)`)).join(''));
  if (params.get('planpresets') === '1' || S.entries.some((e) => e.planform)) h += row('planform top: gen 0 vs last', acs.map((a) => btn(`plan:${a}`, a, `top view, first vs latest generation of ${a} side by side (wing shape change)`)).join(''));
  if (S.entries.some((e) => S.entries.filter((x) => x.aircraft === e.aircraft && x.run === e.run).length > 2))
    h += row('gen 0/mid/last', acs.map((a) => btn(`evo:${a}`, a, `every logged generation of ${a} (selected run)`)).join(''));
  if (sameFam.length > 1) h += row(`seeds ${sameFam.map(runShort).join('/')}`, acs.map((a) => btn(`seeds:${a}`, a, `latest gen of ${a} from ${sameFam.join(', ')}`)).join(''));
  if (fams.length > 1) h += row(fams.join(' vs '), acs.map((a) => btn(`cmp:${a}`, a, `latest gen of ${a}: one run per family (${fams.join(' / ')})`)).join('') + btn('cmp:*', 'all', 'latest gen of every aircraft from each family'));
  el.innerHTML = h;
  el.querySelectorAll('button').forEach((b) => b.addEventListener('click', () => {
    applyPreset(b.dataset.p); presetCam(b.dataset.p); renderGenList(); rebuild(true);
    if (b.dataset.p.startsWith('plan:') && S.shown.length) {  // side by side with ~30 % span gap (step 5 m, the slider's)
      const sp = Math.max(5, Math.ceil(Math.max(...S.shown.map((d) => d.model.span)) * 1.3 / 5) * 5);
      if (sp !== S.spacing) { S.spacing = sp; $('spacing').value = sp; $('spacing-val').textContent = `${sp} m`; rebuild(); }
    }
    snapCam();
  }));
  const rs = $('preset-run');
  if (rs) rs.addEventListener('change', () => { S.presetRun = rs.value; renderPresets(); applyPreset(S.lastPreset || 'last'); renderGenList(); rebuild(true); });
}
function presetSet(p) {
  const gen = (i) => S.entries[i].generation ?? i;
  const last = (l) => l.reduce((a, b) => (gen(b) > gen(a) ? b : a));
  const first = (l) => l.reduce((a, b) => (gen(b) < gen(a) ? b : a));
  const runs = runList();
  const run = runs.includes(S.presetRun) ? S.presetRun : runs[0];
  const idxs = (pred) => S.entries.map((e, i) => (pred(e) ? i : -1)).filter((i) => i >= 0);
  const groupBy = (list, key) => { const m = new Map(); list.forEach((i) => { const k = key(S.entries[i]); if (!m.has(k)) m.set(k, []); m.get(k).push(i); }); return m; };
  const inRun = idxs((e) => e.run === run);
  const byAc = groupBy(inRun, (e) => e.aircraft);
  if (p === 'last') return [...byAc.values()].map(last);
  if (p === 'first') return [...byAc.values()].map(first);
  if (p === 'mid') {  // logged generation closest to the middle of [first, last] (not first / last when there are >2)
    return [...byAc.values()].map((l) => {
      const a = gen(first(l)), b = gen(last(l)), m = (a + b) / 2;
      const inner = l.length > 2 ? l.filter((i) => gen(i) !== a && gen(i) !== b) : l;
      return inner.reduce((x, y) => (Math.abs(gen(y) - m) < Math.abs(gen(x) - m) ? y : x));
    });
  }
  if (p.startsWith('evo:')) { const l = byAc.get(p.slice(4)); return l ? [...l].sort((x, y) => gen(x) - gen(y)) : []; }
  if (p.startsWith('pair:') || p.startsWith('plan:')) { const l = byAc.get(p.slice(5)); return l ? [first(l), last(l)] : []; }
  if (p.startsWith('seeds:')) {
    const ac = p.slice(6), fam = runFamily(run);
    return [...groupBy(idxs((e) => e.aircraft === ac && runFamily(e.run) === fam), (e) => e.run).values()].map(last);
  }
  if (p.startsWith('cmp:')) {
    const ac = p.slice(4);
    // one run per family: the selected run for its own family, else that family's first run
    const pickRun = new Map();
    for (const r of runs) { const f = runFamily(r); if (!pickRun.has(f) || r === run) pickRun.set(f, r); }
    const chosen = new Set(pickRun.values());
    const sel = idxs((e) => chosen.has(e.run) && (ac === '*' || e.aircraft === ac));
    const g = groupBy(sel, (e) => `${e.aircraft}|${runFamily(e.run)}`);
    return [...g.values()].map(last);
  }
  return null;
}
// plan:<ac> = pair:<ac> seen from above (camera 'top'); leaving it restores the chase camera
function presetCam(p) {
  const q = p.split('@')[0];
  if (q.startsWith('plan:')) S.cam = 'top';
  else if (S.cam === 'top') S.cam = 'chase';
  $('cam-mode').value = S.cam;
}
function applyPreset(p) {
  if (p.includes('@')) { const [q, r] = p.split('@'); const hit = runList().find((x) => x === r || runShort(x) === r); if (hit) S.presetRun = hit; p = q; }
  S.lastPreset = p;
  const set = presetSet(p);
  if (set && set.length) { S.compareSet = new Set(set); S.focus = -1; }
}

function renderGenList() {
  const sorted = [...S.compareSet].sort((a, b) => a - b);
  $('gen-list').innerHTML = S.entries.map((e, i) => {
    const slot = sorted.indexOf(i);
    const col = slot >= 0 ? PALETTE[slot % PALETTE.length] : 'transparent';
    return `<label><input type="checkbox" data-i="${i}" ${S.compareSet.has(i) ? 'checked' : ''}><span class="sw" style="background:${col}"></span>${genLabel(e, i)}</label>`;
  }).join('');
}

// ---------------------------------------------------------------- building displays
function makeLine(points, color, width, opacity = 1, dashed = false) {
  const g = new LineGeometry();
  g.setPositions(points);
  const m = new LineMaterial({ color, linewidth: width, transparent: opacity < 1, opacity, dashed, dashSize: 40, gapSize: 25, worldUnits: false, depthWrite: opacity >= 1 });
  m.resolution.set(canvas.clientWidth || 800, canvas.clientHeight || 600);
  lineMats.add(m);
  const l = new Line2(g, m);
  if (dashed) l.computeLineDistances();
  return l;
}

function disposeDisp(d) {
  for (const o of [d.ghost, d.prog, d.tline, d.drop, d.root]) {
    if (!o) continue;
    world.remove(o);
    o.traverse?.((c) => { c.geometry?.dispose?.(); if (c.material) { lineMats.delete(c.material); c.material.dispose?.(); } });
  }
}

function originOffsetENU(tr) {
  const f = tr.frame || {};
  if (!S.ref) S.ref = { lat: f.origin_lat_deg ?? 0, lon: f.origin_lon_deg ?? 0, alt: tr.originAlt ?? 0 };
  const lat = f.origin_lat_deg ?? S.ref.lat, lon = f.origin_lon_deg ?? S.ref.lon;
  return [(lon - S.ref.lon) * D2R * R_EARTH * Math.cos(S.ref.lat * D2R), (lat - S.ref.lat) * D2R * R_EARTH, (tr.originAlt ?? 0) - S.ref.alt];
}

// scene coords: three x = east, y = up (exaggerated), z = -north
// along-track / cross-track of a trajectory w.r.t. its initial heading
function trackInfo(tr) {
  const psi0 = (tr.hud.psi[0] || 0) * D2R;
  const h = [Math.sin(psi0), Math.cos(psi0)], r = [Math.cos(psi0), -Math.sin(psi0)];
  const n = tr.n, s = new Float64Array(n), c = new Float64Array(n);
  const arc = new Float64Array(n), chi = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    s[i] = tr.x[i] * h[0] + tr.y[i] * h[1]; c[i] = tr.x[i] * r[0] + tr.y[i] * r[1];
    if (i) arc[i] = arc[i - 1] + Math.hypot(tr.x[i] - tr.x[i - 1], tr.y[i] - tr.y[i - 1]);
  }
  // horizontal track angle (rad, clockwise from north), unwrapped; from velocity if present, else positions
  const vx = tr.ch.vx, vy = tr.ch.vy;
  for (let i = 0; i < n; i++) {
    let a = vx && vy ? Math.atan2(vx[i], vy[i]) : Math.atan2(tr.x[Math.min(i + 1, n - 1)] - tr.x[Math.max(i - 1, 0)], tr.y[Math.min(i + 1, n - 1)] - tr.y[Math.max(i - 1, 0)]);
    if (i) { while (a - chi[i - 1] > Math.PI) a -= 2 * Math.PI; while (a - chi[i - 1] < -Math.PI) a += 2 * Math.PI; }
    chi[i] = a;
  }
  const dur = Math.max(1e-6, tr.t[n - 1] - tr.t[0]);
  return { psi0, h, r, s, c, arc, chi, speed: (s[n - 1] - s[0]) / dur, arcSpeed: arc[n - 1] / dur };
}
const vrefEff = () => (S.vref === 'auto' ? (autoRel() ? 'rel' : 'abs') : S.vref);
const cyEff = () => (S.cy === 'auto' ? (autoRel() ? ((S.shownTrs || []).some((t) => t.hasRamp) ? 'rerr' : 'err') : 'alt') : S.cy);
// relative display makes sense when the shown aircraft fly different trim altitudes
function autoRel() {
  const alts = S.shownTrs || [];
  if (alts.length < 2) return false;
  const a = alts.map((t) => t.originAlt);
  return Math.max(...a) - Math.min(...a) > 30;
}

// Layout 'formation' ("lock to own track"): every aircraft flies a straight lane along the lead's
// initial heading at its own arc length scaled by refSpeed/ownSpeed (so different trim speeds keep
// station), with its heading change relative to its own track removed from the displayed attitude.
// Vertical: 'abs' = MSL, 'rel' = relative to own trim altitude, 'norm' = relative and scaled so every
// aircraft's altitude step has the same displayed height (refStep).
function buildDisp(tr, entry, color, slot, nSlots, refSpeed, lead, refStep) {
  let [oe, on, ou] = originOffsetENU(tr);
  const vmode = vrefEff();
  if (vmode !== 'abs') ou = 0; // everyone's trim altitude at scene y = 0
  const st = stepSize(tr);
  const vscale = vmode === 'norm' && Number.isFinite(st) && st > 0 && refStep > 0 ? refStep / st : 1;
  const ti = trackInfo(tr);
  const lat = (slot - (nSlots - 1) / 2) * (nSlots > 1 ? S.spacing : 0);
  const n = tr.n, pts = new Float32Array(3 * n), tpts = [];
  const formation = S.layout === 'formation' && nSlots > 1;
  const k = formation && ti.arcSpeed > 1 ? refSpeed / ti.arcSpeed : 1;
  const H = lead.h, Rt = lead.r; // common axes (first aircraft's initial heading)
  let yaw = null;
  if (formation) { yaw = new Float32Array(n); for (let i = 0; i < n; i++) yaw[i] = ti.chi[i] - lead.psi0; }
  const EN = (i) => {
    if (formation) { const sa = ti.arc[i] * k; return [sa * H[0] + lat * Rt[0] + oe, sa * H[1] + lat * Rt[1] + on]; }
    return [tr.x[i] + oe + ti.r[0] * lat, tr.y[i] + on + ti.r[1] * lat];
  };
  for (let i = 0; i < n; i++) {
    const [E, N] = EN(i), U = (tr.z[i] + ou) * vscale;
    pts[3 * i] = E; pts[3 * i + 1] = U * S.exag; pts[3 * i + 2] = -N;
    if (tr.hasTarget && i % 3 === 0) tpts.push(E, (tr.targetAt(tr.t[i], i) - tr.originAlt + ou) * vscale * S.exag, -N);
  }
  const d = { tr, entry, color, pts, slot, ou, vscale, yaw, alongScale: k };
  d.ghost = makeLine(pts, color, 1.5, 0.35);
  d.prog = makeLine(pts, color, 3.5, 1.0);
  world.add(d.ghost, d.prog);
  if (tpts.length > 6) { d.tline = makeLine(new Float32Array(tpts), '#ffffff', 1.5, 0.55, true); world.add(d.tline); }
  const dg = new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute([0, 0, 0, 0, -1, 0], 3));
  d.drop = new THREE.Line(dg, new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.5 }));
  world.add(d.drop);
  const preset = resolvePreset(tr.aircraft, tr.meta.aircraft_model || S.indexMeta.aircraft_model);
  d.preset = preset;
  d.model = buildProcedural(preset, color, tr.planform);
  d.root = d.model.root;
  world.add(d.root);
  d.deformer = tr.structure ? attachStructure(d.model, tr.structure) : null;
  if (preset.gltf) buildGltf(preset).then((gm) => { d.root.remove(d.model.root.children[0]); d.root.add(gm.root); }).catch((e) => msg('glTF: ' + e.message, true));
  return d;
}

function sceneBounds() {
  const b = new THREE.Box3();
  for (const d of S.shown) for (let i = 0; i < d.pts.length; i += 3) b.expandByPoint(new THREE.Vector3(d.pts[i], d.pts[i + 1], d.pts[i + 2]));
  return b;
}

function layoutEnvironment() {
  const b = sceneBounds();
  if (b.isEmpty()) return;
  const c = b.getCenter(new THREE.Vector3()), sz = b.getSize(new THREE.Vector3());
  const span = Math.max(sz.x, sz.z, 2000) + 6000;
  // ground below the start of each shown aircraft (alt_agl_m if present, else sea level); use the lowest
  let groundY = Infinity;
  for (const d of S.shown) {
    const f = d.tr, agl0 = f.ch.alt_agl_m ? f.ch.alt_agl_m[0] : f.alt[0];
    groundY = Math.min(groundY, (d.pts[1] / S.exag) - (Number.isFinite(agl0) ? agl0 : f.alt[0]) * d.vscale);
  }
  S.groundY = groundY;
  ground.scale.set(span * 3, 1, span * 3);
  ground.position.set(c.x, groundY - 0.5, c.z);
  if (groundGrid) { world.remove(groundGrid); groundGrid.geometry.dispose(); }
  const cell = 200, divs = Math.ceil(span * 2 / cell);
  groundGrid = new THREE.GridHelper(divs * cell, divs, 0x2f4a24, 0x4d6b3a);
  groundGrid.position.set(Math.round(c.x / cell) * cell, groundY, Math.round(c.z / cell) * cell);
  world.add(groundGrid);
  if (refGrid) { world.remove(refGrid); refGrid.geometry.dispose(); }
  const rc = 100, rd = Math.ceil(Math.max(sz.x, sz.z, 1000) * 1.2 / rc);
  refGrid = new THREE.GridHelper(rd * rc, rd, 0x557799, 0x88aacc);
  refGrid.material.transparent = true; refGrid.material.opacity = 0.35;
  refGrid.position.set(c.x, 0, c.z);
  refGrid.visible = $('show-refgrid').checked;
  world.add(refGrid);
  targetPlane.scale.set(Math.max(sz.x, 600) + 800, 1, Math.max(sz.z, 600) + 800);
  targetPlane.position.x = c.x; targetPlane.position.z = c.z;
  scene.fog.near = Math.max(3000, span * 0.25); scene.fog.far = span * 2.5;
  S.bounds = b;
}

async function rebuild(snapCamera = false) {
  const token = ++S.buildToken;
  const idxs = S.mode === 'single' ? [S.singleIdx] : [...S.compareSet].sort((a, b) => a - b);
  if (!S.entries.length || !idxs.length) { for (const d of S.shown) disposeDisp(d); S.shown = []; updateLegend(); drawChartStatic(); return; }
  msg(`loading ${idxs.length} trajector${idxs.length > 1 ? 'ies' : 'y'} ...`);
  let trs;
  try { trs = await Promise.all(idxs.map((i) => getTraj(S.entries[i]))); } catch (e) { msg(e.message, true); return; }
  if (token !== S.buildToken) return;
  for (const d of S.shown) disposeDisp(d);
  S.shownTrs = trs;
  const infos = trs.map(trackInfo);
  const refSpeed = infos.reduce((a, b) => a + b.arcSpeed, 0) / infos.length;
  const steps = trs.map(stepSize).filter((v) => Number.isFinite(v) && v > 0);
  const refStep = steps.length ? steps.reduce((a, b) => a + b, 0) / steps.length : 0;
  S.shown = trs.map((tr, k) => buildDisp(tr, S.entries[idxs[k]], S.mode === 'single' ? SINGLE_COLOR : PALETTE[k % PALETTE.length], k, trs.length, refSpeed, infos[0], refStep));
  syncAltUI();
  S.focus = S.mode === 'compare' && S.shown.length > 1 ? Math.min(S.focus, S.shown.length - 1) : Math.max(0, Math.min(S.focus, S.shown.length - 1));
  S.tEnd = Math.max(...trs.map((t) => t.t1));
  S.t = Math.min(S.t, S.tEnd);
  layoutEnvironment();
  applyModelScale();
  populateLabels();
  updateLegend();
  $('defl-row').hidden = !S.shown.some((d) => d.deformer);
  drawChartStatic();
  msg('');
  update(0);
  if (snapCamera || S.cam === 'free') snapCam();
  S.ready = true;
}

function populateLabels() {
  const sel = $('gen-select');
  [...sel.options].forEach((o, i) => { o.textContent = genLabel(S.entries[i], i); });
}

function syncAltUI() {
  $('vref-eff').textContent = S.vref === 'auto' ? `(${vrefEff()})` : '';
  $('cy-eff').textContent = S.cy === 'auto' ? `(${cyEff()})` : '';
}

function applyModelScale() { for (const d of S.shown) d.root.scale.setScalar(S.mscale); }

// ---------------------------------------------------------------- per-frame update
const _q = new THREE.Quaternion(), _v = new THREE.Vector3(), _qYaw = new THREE.Quaternion(), _Y = new THREE.Vector3(0, 1, 0);
function sampleDisp(d, time) {
  const tr = d.tr;
  const [i, f] = locate(tr, time);
  const j = Math.min(i + 1, tr.n - 1), p = d.pts;
  const pos = new THREE.Vector3(p[3 * i] + (p[3 * j] - p[3 * i]) * f, p[3 * i + 1] + (p[3 * j + 1] - p[3 * i + 1]) * f, p[3 * i + 2] + (p[3 * j + 2] - p[3 * i + 2]) * f);
  const q = quatAt(tr, i, f, new THREE.Quaternion());
  if (d.yaw) { // formation: remove own track-heading change (rotate about scene up; +angle = counter-clockwise)
    const a = d.yaw[i] + (d.yaw[j] - d.yaw[i]) * f;
    q.premultiply(_qYaw.setFromAxisAngle(_Y, a));
  }
  return { i, f, pos, q, ended: time > tr.t1 + 1e-6 };
}

function update(dtSec) {
  if (S.playing && S.shown.length) {
    S.t += dtSec * S.speed;
    if (S.t > S.tEnd) S.t = 0;
  }
  for (const d of S.shown) {
    const s = sampleDisp(d, S.t);
    d.state = s;
    d.root.position.copy(s.pos);
    if (S.exag !== 1) { // keep attitude visually consistent-ish with an exaggerated vertical axis: no change (true attitude)
    }
    d.root.quaternion.copy(s.q);
    const c = (n) => lerpCh(d.tr.ch[n], s.i, s.f);
    applyControls(d.model, { elevator: c('elevator'), aileron: c('aileron'), rudder: c('rudder'), throttle: c('throttle') }, s.ended ? 0 : dtSec);
    if (d.deformer) applyStructure(d.model, d.deformer, (arr) => lerpCh(arr, s.i, s.f), S.defl, S.dofs);
    d.prog.geometry.instanceCount = Math.max(1, s.i + (s.f > 0 ? 1 : 0));
    d.ghost.visible = d.prog.visible = $('show-trail').checked;
    if (d.tline) d.tline.visible = $('show-target').checked;
    const a = d.drop.geometry.attributes.position;
    a.setXYZ(0, s.pos.x, s.pos.y, s.pos.z); a.setXYZ(1, s.pos.x, S.groundY ?? s.pos.y, s.pos.z); a.needsUpdate = true;
    d.drop.geometry.computeBoundingSphere();
  }
  const fd = hudDisp();
  if (fd && fd.state && fd.tr.hasTarget && $('show-target').checked) {
    targetPlane.visible = true;
    targetPlane.position.y = (fd.tr.targetAt(S.t, fd.state.i) - fd.tr.originAlt + fd.ou) * fd.vscale * S.exag;
  } else targetPlane.visible = false;
  updateCamera(dtSec);
  updateHUD();
  updateTimeUI();
}

// ---------------------------------------------------------------- camera
let lastFocusPos = null, chaseInit = false;
function hudDisp() { return S.shown[Math.max(0, S.focus)]; }
// Camera target: one aircraft, or (S.focus === -1, compare mode) the centroid of the formation.
function focusDisp() {
  if (S.focus >= 0 || S.shown.length < 2) return S.shown[Math.max(0, S.focus)];
  const ds = S.shown.filter((d) => d.state);
  if (!ds.length) return null;
  const pos = new THREE.Vector3();
  ds.forEach((d) => pos.add(d.state.pos));
  pos.multiplyScalar(1 / ds.length);
  const lead = ds[0];
  const hf = new THREE.Vector3(1, 0, 0).applyQuaternion(lead.state.q).setY(0);
  if (hf.lengthSq() < 1e-6) hf.set(0, 0, -1);
  hf.normalize();
  let wl = 0, wv = 0, rear = 0, spanMax = 0;
  ds.forEach((d) => {
    const dp = d.state.pos.clone().sub(pos), half = d.model.span * S.mscale / 2;
    const along = dp.x * hf.x + dp.z * hf.z, lat = Math.abs(dp.x * hf.z - dp.z * hf.x);
    rear = Math.max(rear, -along + half);           // how far the rearmost aircraft trails the centroid
    wl = Math.max(wl, lat + half);
    wv = Math.max(wv, Math.abs(dp.y) + half * 0.3);
    spanMax = Math.max(spanMax, d.model.span);
  });
  // distance so the whole formation fits (~80 deg horizontal, ~55 deg vertical FOV, with margin),
  // measured from the rearmost aircraft so nobody ends up beside/behind the camera
  const chaseDist = Math.max(30 * S.mscale, spanMax * 5.0 * S.mscale, rear + Math.max(wl * 1.9, wv * 2.6, 15 * S.mscale));
  return { state: { pos, q: lead.state.q }, model: { span: spanMax }, chaseDist, halfWidth: wl, halfLen: rear };
}
function snapCam() {
  chaseInit = false;
  const d = focusDisp();
  if (!d) return;
  if (S.cam === 'free') {
    const b = S.bounds || sceneBounds();
    const c = b.getCenter(new THREE.Vector3()), sz = b.getSize(new THREE.Vector3());
    const r = Math.max(sz.x, sz.z, 300) * 0.8;
    controls.target.copy(c);
    camera.position.set(c.x - r * 0.75, c.y + r * 0.45, c.z + r * 0.35);
  } else if (S.cam === 'orbit') {
    const p = d.state.pos, L = d.chaseDist ?? Math.max(25, d.model.span * 2.6) * S.mscale;
    controls.target.copy(p);
    camera.position.set(p.x + L * 0.9, p.y + L * 0.35, p.z + L * 0.6);
  }
  lastFocusPos = d.state.pos.clone();
  controls.update();
}

function updateCamera(dtSec) {
  const d = focusDisp();
  if (!d || !d.state) return;
  const p = d.state.pos;
  controls.enabled = S.cam !== 'chase' && S.cam !== 'top';
  if (S.cam === 'top') {  // plan view: straight down onto the focus (formation centroid), nose toward the screen top
    const fwd = new THREE.Vector3(1, 0, 0).applyQuaternion(d.state.q);
    const hf = new THREE.Vector3(fwd.x, 0, fwd.z);
    if (hf.lengthSq() < 1e-6) hf.set(0, 0, -1);
    hf.normalize();
    // fit the formation's width into the free middle of the screen (side panel left, HUD / legend right):
    // ~40 % of the view width, centred at ~44 % from the left
    const tanV = Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2), asp = camera.aspect || 1.6;
    const halfW = d.halfWidth ?? d.model.span * S.mscale / 2;
    const halfL = Math.max(d.halfLen ?? 0, d.model.span * S.mscale / 2);
    const H = Math.max(halfW * 1.15 / (0.40 * tanV * asp), halfL * 1.3 / tanV, 8 * S.mscale);
    const right = new THREE.Vector3().crossVectors(hf, new THREE.Vector3(0, 1, 0)).normalize();
    const c = p.clone().addScaledVector(right, (0.5 - 0.44) * 2 * H * tanV * asp);
    camera.position.copy(c).add(new THREE.Vector3(0, H, 0)).addScaledVector(hf, -0.01 * H);
    camera.up.copy(hf);
    camera.lookAt(c);
    controls.target.copy(c);
    chaseInit = false;
  } else if (S.cam === 'chase') {
    const fwd = new THREE.Vector3(1, 0, 0).applyQuaternion(d.state.q);
    const hf = new THREE.Vector3(fwd.x, 0, fwd.z);
    if (hf.lengthSq() < 1e-6) hf.set(0, 0, -1);
    hf.normalize();
    const L = d.chaseDist ?? Math.max(22, d.model.span * 2.4) * S.mscale;
    const want = p.clone().addScaledVector(hf, -L).add(new THREE.Vector3(0, L * 0.3, 0));
    if (!chaseInit || dtSec === 0) { camera.position.copy(want); chaseInit = true; }
    else camera.position.lerp(want, 1 - Math.exp(-dtSec * 5));
    camera.up.set(0, 1, 0);
    camera.lookAt(p.clone().addScaledVector(hf, L * 0.4).add(new THREE.Vector3(0, L * 0.05, 0)));
  } else {
    if (S.cam === 'orbit' && lastFocusPos) {
      const delta = p.clone().sub(lastFocusPos);
      camera.position.add(delta);
      controls.target.add(delta);
    }
    lastFocusPos = p.clone();
    controls.update();
  }
}

// ---------------------------------------------------------------- HUD / legend / time
const fmtS = (v, d = 1, w = 7) => (Number.isFinite(v) ? (v >= 0 ? '+' : '') + v.toFixed(d) : '—').padStart(w);
const fmtU = (v, d = 1, w = 7) => (Number.isFinite(v) ? v.toFixed(d) : '—').padStart(w);
function bar(v, signed = true) {
  if (!Number.isFinite(v)) return '<span class="bar"><b></b></span>';
  const c = Math.max(-1, Math.min(1, v));
  if (signed) { const l = c < 0 ? 50 + c * 50 : 50, w = Math.abs(c) * 50; return `<span class="bar"><i style="left:${l}%;width:${w}%"></i><b></b></span>`; }
  return `<span class="bar"><i style="left:0;width:${Math.max(0, c) * 100}%"></i></span>`;
}
function updateHUD() {
  const el = $('hud');
  el.style.display = $('show-hud').checked ? '' : 'none';
  const d = hudDisp();
  if (!d || !d.state) { el.innerHTML = 'no trajectory loaded'; return; }
  const tr = d.tr, { i, f } = d.state, c = (n) => lerpCh(tr.ch[n], i, f);
  const alt = lerpCh(tr.alt, i, f), tgt = tr.targetAt(S.t, i);
  const pick = (...names) => { for (const n of names) if (tr.ch[n]) return c(n); return NaN; };
  const ias = pick('kcas', 'vc_kts', 'ias_kts'), tas = pick('ktas', 'vtrue_kts', 'tas_kts');
  const iasLbl = tr.ch.kcas ? 'KCAS' : 'IAS ';
  const nz = pick('nz', 'n_z', 'load_factor');
  const cmd = tr.hasRamp ? tr.stepAt(S.t, i) : NaN;
  const gs = Math.hypot(c('vx'), c('vy')) / KT;
  const vs = c('vz') / FT * 60;
  const phi = lerpCh(tr.hud.phi, i, f), th = lerpCh(tr.hud.theta, i, f), psi = (lerpAngleDeg(tr.hud.psi, i, f) + 360) % 360;
  const ev = [...tr.events].reverse().find((e) => e.t <= S.t + 1e-6 && S.t - e.t < 4);
  const status = d.state.ended ? `<span class="warn">ENDED at ${tr.t1.toFixed(2)} s</span>` : 'flying';
  const endEv = tr.events.find((e) => e.type === 'envelope_violation' || e.type === 'terminated');
  el.innerHTML =
    `<span class="big">${tr.aircraft} · ${tr.generation != null ? 'gen ' + tr.generation : tr.source}</span>${S.mode === 'compare' ? `  <span style="color:${d.color}">■</span>` : ''}\n` +
    `${metricName(tr).padEnd(5)} ${tr.fitness != null ? Number(tr.fitness).toFixed(6) : '—'} <span class="small">(${senseOf(tr) === 'max' ? 'higher' : 'lower'}=better${tr.scenarioIndex != null && tr.replay ? `; scenario ${tr.scenarioIndex}: ${Number(tr.meta.scenario_cost).toFixed(6)}` : ''})</span>\n` +
    (tr.replay ? `<span class="small">replay ${tr.replay.replay_id} · ${tr.replay.fidelity} · ${tr.replay.model_version ?? ''}</span>\n` : '') +
    (trimLabel(tr) ? `trim  ${trimLabel(tr).trim()}\n` : '') +
    `t     ${fmtU(S.t, 2, 7)} / ${tr.t1.toFixed(2)} s  ${status}\n` +
    `ALT   ${fmtU(alt / FT, 1, 7)} ft  ${fmtU(alt, 1, 7)} m MSL\n` +
    (Number.isFinite(cmd) ? `CMD   ${fmtU(cmd / FT, 1, 7)} ft  err ${fmtS((alt - cmd) / FT, 1, 6)} ft <span class="small">step</span>\n` : '') +
    (Number.isFinite(tgt) ? `${tr.hasRamp ? 'REF ' : 'TGT '}  ${fmtU(tgt / FT, 1, 7)} ft  err ${fmtS((alt - tgt) / FT, 1, 6)} ft${tr.hasRamp ? ' <span class="small">ramp</span>' : ''}\n` : '') +
    (Number.isFinite(c('alt_agl_m')) ? `AGL   ${fmtU(c('alt_agl_m') / FT, 0, 7)} ft\n` : '') +
    (Number.isFinite(ias) ? `${iasLbl}  ${fmtU(ias, 1, 7)} kt` + (Number.isFinite(tas) ? `  TAS ${fmtU(tas, 1, 6)} kt` : '') + '\n' : `GS    ${fmtU(gs, 1, 7)} kt\n`) +
    (Number.isFinite(nz) ? `NZ    ${fmtU(nz, 2, 7)} g\n` : '') +
    `VS    ${fmtS(vs, 0, 7)} ft/min  GS ${fmtU(gs, 1, 6)} kt\n` +
    `ROLL  φ ${fmtS(phi, 1, 6)}°  PITCH θ ${fmtS(th, 1, 6)}°\n` +
    `HDG   ψ  ${(psi.toFixed(1) === '360.0' ? '0.0' : psi.toFixed(1)).padStart(5, '0')}°\n` +
    `ELEV  ${bar(c('elevator'))} ${fmtS(c('elevator'), 3, 7)}\n` +
    `AIL   ${bar(c('aileron'))} ${fmtS(c('aileron'), 3, 7)}\n` +
    `RUD   ${bar(c('rudder'))} ${fmtS(c('rudder'), 3, 7)}\n` +
    `THR   ${bar(c('throttle'), false)} ${fmtU(c('throttle'), 3, 7)}` +
    planformHud(tr) +
    flexHud(tr, i, f) +
    (endEv ? `\n<span class="warn">envelope: ${endEv.detail} @ ${endEv.t.toFixed(2)} s</span>` : '') +
    (ev && ev.type !== 'start' ? `\n<span class="warn">» ${ev.type}: ${ev.detail}</span>` : '');
  if (S.mode === 'compare') updateLegendValues();
  const lg = $('legend');
  if (!lg.hidden) lg.style.top = (el.style.display === 'none' ? 8 : el.offsetTop + el.offsetHeight + 8) + 'px';
}

// P3-B1 planform line (only when the trajectory carries a planform block): quarter-chord sweep, tip/root taper,
// tip built-in twist; synthetic fixtures say so.
export function planformText(pf) {
  if (!pf) return '';
  const deg = (r) => (Number.isFinite(r) ? (r / D2R).toFixed(1) + '°' : '—');
  return `PLANFORM ${pf.source || 'B1'} Λqc ${deg(pf.sweep_qc_rad)} taper ${Number.isFinite(pf.taper) ? pf.taper.toFixed(2) : '—'}` +
    (pf.twist_tip_rad != null ? ` twist tip ${deg(pf.twist_tip_rad)}` : '') + (pf.symmetric ? '' : ' (L≠R)') +
    (pf.geom === 'nodes' ? ' · FD r1 node geometry' : '');
}
function pfTag(tr) {
  const pf = tr.planform;
  if (!pf) return '';
  const sw = Number.isFinite(pf.sweep_qc_rad) ? `Λ${(pf.sweep_qc_rad / D2R).toFixed(0)}° ` : '';
  return ` <span class="small">${sw}λ${Number.isFinite(pf.taper) ? pf.taper.toFixed(2) : '—'}${pf.synthetic ? ' syn' : ''}</span>`;
}
function planformHud(tr) {
  const pf = tr.planform;
  if (!pf) return '';
  return `\n${planformText(pf)}` + (pf.synthetic ? ' <span class="warn">SYNTHETIC planform (test fixture)</span>' : '');
}

// soft-body readout: tip (last node) values of each component, true scale (not exaggerated)
function flexHud(tr, i, f) {
  const st = tr.structure;
  if (!st) return '';
  let h = st.synthetic ? '\n<span class="warn">SYNTHETIC structure data (test pattern, not a simulation)</span>' : '';
  const fid = tr.meta.fidelity || (tr.meta.replay && tr.meta.replay.fidelity);
  h += `\n<span class="small">flex ×${S.defl} display · tip values true scale${fid ? ' · ' + fid : ''}</span>`;
  if (S.dofs) h += `\n<span class="warn">display shows only: ${S.dofs.join(', ')} (rest hidden)</span>`;
  if (st.estimated && st.estimated.length) h += `\n<span class="warn">ESTIMATED (tip-only, no FD nodes): ${st.estimated.join(', ')}</span>`;
  else if (!st.synthetic) h += '\n<span class="small">FD nodal data (all components)</span>';
  for (const c of st.components) {
    if ((c.name || '').endsWith('_modal')) continue;  // FE tips shown; modal kept in channels for comparison
    const last = c.axis_nodes.length - 1;
    // htail runs left tip -> right tip: show both tips (L/R differ under roll / asymmetric loads)
    const at = (d, k) => (c.ch[d] && c.ch[d][k] ? lerpCh(c.ch[d][k], i, f) : NaN);
    const fmt = (k) => {
      const dz = at('dz', k), dy = at('dy', k), dx = at('dx', k), tw = at('twist', k);
      const parts = [];
      if (Number.isFinite(dz)) parts.push(`dz ${fmtS(dz, 3, 6)}`);
      if (Number.isFinite(dy)) parts.push(`dy ${fmtS(dy, 3, 6)}`);
      if (Number.isFinite(dx)) parts.push(`dx ${fmtS(dx, 4, 7)}`);
      if (Number.isFinite(tw)) parts.push(`tw ${fmtS(tw / D2R, 2, 6)}°`);
      return parts.join(' ');
    };
    const est = c.estimated ? ' <span class="warn">est.</span>' : '';
    if (c.name === 'htail') {
      const l = fmt(0), r = fmt(last);
      if (l) h += `\nTIP htail L ${l}${est}`;
      if (r) h += `\nTIP htail R ${r}${est}`;
    } else {
      const p = fmt(last);
      if (p) h += `\nTIP ${c.name.padEnd(8)} ${p}${est}`;
    }
  }
  if (st.components.length) h += '\n<span class="small">m / deg; dz+ down dy+ right dx+ fwd</span>';
  return h;
}

function trimLabel(tr) {
  const kt = tr.meta.target && (tr.meta.target.speed_kcas ?? tr.meta.target.speed_kts);
  const sched = tr.schedule;
  const h0 = sched && sched.length ? sched[0].alt_m : tr.originAlt;
  return (kt ? `  ${String(Math.round(kt)).padStart(3)}kt/${String(Math.round(h0 / FT)).padStart(5)}ft` : '');
}
function updateLegend() {
  const el = $('legend');
  el.hidden = S.mode !== 'compare' || !S.shown.length;
  if (el.hidden) return;
  el.innerHTML = '<b>compare</b> (click = camera/HUD focus)\n' +
    `<div data-k="-1" class="${S.focus === -1 ? 'focus' : ''}">◎ formation (camera on all)</div>` + S.shown.map((d, k) =>
    `<div data-k="${k}" class="${k === S.focus ? 'focus' : ''}"><span style="color:${d.color}">■</span> ${multiAircraft() ? d.tr.aircraft.padEnd(6) + ' ' : ''}${multiRun() ? runShort(d.entry.run).padEnd(3) + ' ' : ''}g${String(d.tr.generation ?? k).padStart(3, '0')}${indTag(d.entry)}${pfTag(d.tr)}  ${senseOf(d.tr) === 'max' ? 'fit' : 'cost'} ${d.tr.fitness != null ? Number(d.tr.fitness).toFixed(4) : '—'}${trimLabel(d.tr)}  <span class="lv"></span></div>`).join('') +
    ((n) => (n ? `<details${S.notesOpen ? ' open' : ''}><summary class="small">ⓘ layout notes</summary>${n.replace(/^\n/, '')}</details>` : ''))('' +
    (S.layout === 'formation' && S.shown.length > 1 ? `\n<span class="small">formation (locked to own track): straight lanes, distance flown\nscaled per aircraft (${S.shown.map((d) => d.tr.aircraft + ' ×' + d.alongScale.toFixed(2)).join(', ')});\nheading drift removed, climb angles not to scale</span>` : '') +
    (vrefEff() === 'rel' ? `\n<span class="small">3D altitude: relative to each aircraft's own trim altitude</span>` : '') +
    (vrefEff() === 'norm' ? `\n<span class="small">3D altitude: relative to own trim, scaled so each step shows the\nsame height (${S.shown.map((d) => d.tr.aircraft + ' ×' + d.vscale.toFixed(2)).join(', ')}); HUD/chart are true</span>` : ''));
  const det = el.querySelector('details');
  if (det) det.addEventListener('toggle', () => { S.notesOpen = det.open; });
  [...el.querySelectorAll('div[data-k]')].forEach((div) => div.addEventListener('click', () => { S.focus = +div.dataset.k; updateLegend(); snapCam(); }));
}
function updateLegendValues() {
  const divs = $('legend').querySelectorAll('div[data-k] .lv');
  S.shown.forEach((d, k) => {
    if (!divs[k] || !d.state) return;
    const a = lerpCh(d.tr.alt, d.state.i, d.state.f), tg = d.tr.targetAt(S.t, d.state.i);
    const ias = d.tr.ch.kcas || d.tr.ch.vc_kts;
    divs[k].textContent = `alt ${(a / FT).toFixed(0).padStart(5)} err ${Number.isFinite(tg) ? (((a - tg) / FT >= 0 ? '+' : '') + ((a - tg) / FT).toFixed(0)).padStart(5) : '    —'}` +
      (ias ? ` IAS ${lerpCh(ias, d.state.i, d.state.f).toFixed(0).padStart(3)}` : '') +
      (d.tr.ch.nz ? ` nz ${lerpCh(d.tr.ch.nz, d.state.i, d.state.f).toFixed(2)}` : '') + (d.state.ended ? ' (ended)' : '');
  });
}

let scrubbing = false;
function updateTimeUI() {
  $('time-label').textContent = `${S.t.toFixed(2)} / ${S.tEnd.toFixed(2)} s`;
  if (!scrubbing) $('timeline').value = String(Math.round(S.t / S.tEnd * 10000));
  $('play').innerHTML = S.playing ? '&#x275A;&#x275A;' : '&#x25B6;';
  drawChartCursor();
}

// ---------------------------------------------------------------- chart
const chart = $('chart');
const chartBg = document.createElement('canvas');
let chartMap = null;
const CHART_LABEL = {
  alt: 'altitude [ft MSL]', rel: 'altitude vs own trim alt [ft]', err: 'altitude error vs own step target [ft]',
  rerr: 'altitude error vs own ramp reference [ft]', pct: 'altitude error vs step [% of own step]',
  nz: 'load factor nz [g]', kcas: 'airspeed [KCAS]', dkcas: 'airspeed vs own trim [kt]',
};
const CHART_UNIT_DEC = { nz: 2 };
function stepSize(tr) {
  if (tr.schedule && tr.schedule.length > 1) { const a = tr.schedule.map((s) => s.alt_m); return Math.max(...a) - Math.min(...a); }
  return NaN;
}
const iasCh = (tr) => tr.ch.kcas || tr.ch.vc_kts || tr.ch.ias_kts || null;
const nzCh = (tr) => tr.ch.nz || tr.ch.n_z || tr.ch.load_factor || null;
function trimKts(tr) { const t = tr.meta.target; const v = t && (t.speed_kcas ?? t.speed_kts); const a = iasCh(tr); return Number.isFinite(v) ? v : a ? a[0] : NaN; }
function chartVal(tr, i, mode) {
  const a = tr.alt[i];
  if (mode === 'rel') return (a - tr.originAlt) / FT;
  if (mode === 'err') return (a - tr.stepAt(tr.t[i], i)) / FT;
  if (mode === 'rerr') return (a - tr.targetAt(tr.t[i], i)) / FT;
  if (mode === 'pct') { const st = stepSize(tr), e = a - tr.stepAt(tr.t[i], i); return Number.isFinite(st) && st > 0 ? 100 * e / st : e / FT; }
  if (mode === 'nz') { const c = nzCh(tr); return c ? c[i] : NaN; }
  if (mode === 'kcas') { const c = iasCh(tr); return c ? c[i] : NaN; }
  if (mode === 'dkcas') { const c = iasCh(tr); return c ? c[i] - trimKts(tr) : NaN; }
  return a / FT;
}
// reference curves per mode: [{ key, val(i) }] (dashed). In alt/rel modes the ramp reference is primary and
// the step command is drawn dotted; in 'err' mode the ramp-minus-step curve shows what perfect ramp tracking scores.
function chartRefs(tr, mode) {
  const ramp = tr.hasRamp;
  const off = mode === 'rel' ? tr.originAlt : 0;
  if (mode === 'alt' || mode === 'rel') {
    const r = [{ key: 'ref', dash: [5, 4], val: (i) => (tr.targetAt(tr.t[i], i) - off) / FT }];
    if (ramp) r.push({ key: 'cmd', dash: [1, 3], val: (i) => (tr.stepAt(tr.t[i], i) - off) / FT });
    return tr.hasTarget ? r : [];
  }
  if (mode === 'err') return [{ key: 'zero', dash: [5, 4], val: () => 0 }].concat(ramp ? [{ key: 'ramp-cmd', dash: [1, 3], val: (i) => (tr.targetAt(tr.t[i], i) - tr.stepAt(tr.t[i], i)) / FT }] : []);
  if (mode === 'rerr' || mode === 'pct' || mode === 'dkcas') return [{ key: 'zero', dash: [5, 4], val: () => 0 }];
  if (mode === 'nz') return [{ key: 'one', dash: [5, 4], val: () => 1 }];
  if (mode === 'kcas') { const v = trimKts(tr); return Number.isFinite(v) ? [{ key: 'trim', dash: [5, 4], val: () => v }] : []; }
  return [];
}
function drawChartStatic() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const W = chart.clientWidth || 460, H = chart.clientHeight || 170;
  chart.width = chartBg.width = W * dpr; chart.height = chartBg.height = H * dpr;
  const g = chartBg.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, W, H);
  if (!S.shown.length) { chartMap = null; drawChartCursor(); return; }
  const mode = cyEff();
  const padL = 46, padR = 8, padT = 16, padB = 20;
  let lo = Infinity, hi = -Infinity;
  for (const d of S.shown) {
    for (let i = 0; i < d.tr.n; i++) { const a = chartVal(d.tr, i, mode); if (Number.isFinite(a)) { if (a < lo) lo = a; if (a > hi) hi = a; } }
    for (const r of chartRefs(d.tr, mode)) for (let i = 0; i < d.tr.n; i += 3) { const v = r.val(i); if (Number.isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); } }
  }
  if (!Number.isFinite(lo)) { lo = 0; hi = 1; }
  const minPad = mode === 'nz' ? 0.05 : mode === 'pct' ? 2 : mode === 'kcas' || mode === 'dkcas' ? 2 : 10;
  const pad = Math.max(minPad, (hi - lo) * 0.08); lo -= pad; hi += pad;
  const X = (t) => padL + (t / S.tEnd) * (W - padL - padR), Y = (a) => padT + (1 - (a - lo) / (hi - lo)) * (H - padT - padB);
  chartMap = { X, Y, padL, padR, padT, padB, W, H, mode };
  g.font = '10px ui-monospace, Menlo, monospace'; g.fillStyle = '#b7c6d6'; g.strokeStyle = 'rgba(255,255,255,0.12)'; g.lineWidth = 1;
  const step = niceStep((hi - lo) / 4), dec = step < 1 ? (CHART_UNIT_DEC[mode] ?? 1) : 0;
  for (let a = Math.ceil(lo / step) * step; a <= hi; a += step) { g.beginPath(); g.moveTo(padL, Y(a)); g.lineTo(W - padR, Y(a)); g.stroke(); g.fillText(a.toFixed(dec), 4, Y(a) + 3); }
  const tstep = niceStep(S.tEnd / 6);
  for (let t = 0; t <= S.tEnd + 1e-9; t += tstep) { g.beginPath(); g.moveTo(X(t), padT); g.lineTo(X(t), H - padB); g.stroke(); g.fillText(t.toFixed(0), X(t) - 6, H - 6); }
  const missing = S.shown.filter((d) => !Number.isFinite(chartVal(d.tr, 0, mode))).length;
  const anyRamp = S.shown.some((d) => d.tr.hasRamp);
  g.fillText(`${CHART_LABEL[mode]} vs time [s]` + ((mode === 'alt' || mode === 'rel') && anyRamp ? '  (dash = ramp ref, dots = step cmd)' : '') +
    (mode === 'err' && anyRamp ? '  (dots = ramp ref − step)' : '') + (missing ? `  [${missing} without this channel]` : ''), padL + 4, 11);
  // reference curves: one per distinct profile (in that aircraft's colour if they differ)
  const sig = (d, r) => { const v = []; for (let i = 0; i < d.tr.n; i += Math.max(1, Math.floor(d.tr.n / 40))) v.push(r.val(i).toFixed(1)); return r.key + ':' + v.join(','); };
  const allRefs = S.shown.flatMap((d) => chartRefs(d.tr, mode).map((r) => ({ d, r, k: sig(d, r) })));
  const multiTgt = new Set(allRefs.filter((x) => x.r.key === allRefs[0]?.r.key).map((x) => x.k)).size > 1;
  const seen = new Set();
  for (const { d, r, k } of allRefs) {
    if (seen.has(k)) continue;
    seen.add(k);
    g.setLineDash(r.dash); g.lineWidth = 1;
    const constant = r.key === 'zero' || r.key === 'one';
    g.strokeStyle = multiTgt && !constant ? d.color : 'rgba(255,255,255,0.8)';
    g.beginPath();
    for (let i = 0; i < d.tr.n; i += 2) { const x = X(d.tr.t[i]), y = Y(r.val(i)); i ? g.lineTo(x, y) : g.moveTo(x, y); }
    g.stroke(); g.setLineDash([]);
  }
  for (const d of S.shown) {
    g.strokeStyle = d.color; g.lineWidth = 1.6; g.beginPath();
    let pen = false;
    for (let i = 0; i < d.tr.n; i += 2) {
      const v = chartVal(d.tr, i, mode);
      if (!Number.isFinite(v)) { pen = false; continue; }
      const x = X(d.tr.t[i]), y = Y(v); pen ? g.lineTo(x, y) : g.moveTo(x, y); pen = true;
    }
    g.stroke();
  }
  drawChartCursor();
}
function niceStep(raw) { const p = Math.pow(10, Math.floor(Math.log10(raw || 1))); for (const m of [1, 2, 2.5, 5, 10]) if (m * p >= raw) return m * p; return 10 * p; }
function drawChartCursor() {
  chart.style.display = $('show-chart').checked ? '' : 'none';
  const g = chart.getContext('2d');
  g.setTransform(1, 0, 0, 1, 0, 0);
  g.clearRect(0, 0, chart.width, chart.height);
  g.drawImage(chartBg, 0, 0);
  if (!chartMap) return;
  const dpr = chart.width / chartMap.W;
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  const x = chartMap.X(S.t);
  g.strokeStyle = '#ffffff'; g.lineWidth = 1; g.beginPath(); g.moveTo(x, chartMap.padT); g.lineTo(x, chartMap.H - chartMap.padB); g.stroke();
  for (const d of S.shown) if (d.state) { const v = chartVal(d.tr, d.state.i, chartMap.mode); if (!Number.isFinite(v)) continue; g.fillStyle = d.color; g.beginPath(); g.arc(chartMap.X(Math.min(S.t, d.tr.t1)), chartMap.Y(v), 3, 0, 7); g.fill(); }
}
function chartSeek(ev) {
  if (!chartMap) return;
  const r = chart.getBoundingClientRect();
  const x = (ev.clientX - r.left) * (chartMap.W / r.width);
  S.t = Math.max(0, Math.min(S.tEnd, (x - chartMap.padL) / (chartMap.W - chartMap.padL - chartMap.padR) * S.tEnd));
}
let chartDrag = false;
chart.addEventListener('pointerdown', (e) => { chartDrag = true; chartSeek(e); });
window.addEventListener('pointerup', () => { chartDrag = false; });
chart.addEventListener('pointermove', (e) => { if (chartDrag) chartSeek(e); });

// ---------------------------------------------------------------- UI wiring
function setMode(m) {
  S.mode = m;
  document.querySelector(`input[name=mode][value=${m}]`).checked = true;
  $('single-ui').hidden = m !== 'single';
  $('compare-ui').hidden = m !== 'compare';
  S.focus = m === 'compare' ? -1 : 0;
  return rebuild(true);
}
document.querySelectorAll('input[name=mode]').forEach((r) => r.addEventListener('change', () => setMode(r.value)));
$('gen-select').addEventListener('change', (e) => { S.singleIdx = +e.target.value; rebuild(); });
$('gen-prev').addEventListener('click', () => { if (S.singleIdx > 0) { S.singleIdx--; $('gen-select').value = S.singleIdx; rebuild(); } });
$('gen-next').addEventListener('click', () => { if (S.singleIdx < S.entries.length - 1) { S.singleIdx++; $('gen-select').value = S.singleIdx; rebuild(); } });
$('gen-list').addEventListener('change', (e) => {
  const i = +e.target.dataset.i;
  if (e.target.checked) S.compareSet.add(i); else S.compareSet.delete(i);
  renderGenList(); rebuild();
});
$('cmp-improve').addEventListener('click', () => { S.compareSet = new Set(improvementPicks()); renderGenList(); rebuild(); });
$('cmp-spread').addEventListener('click', () => { S.compareSet = new Set(spreadPicks(5)); renderGenList(); rebuild(); });
$('cmp-none').addEventListener('click', () => { S.compareSet = new Set(); renderGenList(); rebuild(); });
$('spacing').addEventListener('input', (e) => { S.spacing = +e.target.value; $('spacing-val').textContent = `${S.spacing} m`; rebuild(); });
$('cam-mode').addEventListener('change', (e) => { S.cam = e.target.value; snapCam(); });
$('exag').addEventListener('change', (e) => { S.exag = +e.target.value; rebuild(true); });
$('layout').addEventListener('change', (e) => { S.layout = e.target.value; rebuild(true); });
$('vref').addEventListener('change', (e) => { S.vref = e.target.value; rebuild(true); });
$('cy').addEventListener('change', (e) => { S.cy = e.target.value; drawChartStatic(); });
$('defl').addEventListener('input', (e) => { S.defl = +e.target.value; $('defl-val').textContent = `${S.defl}x`; });
$('mscale').addEventListener('input', (e) => { S.mscale = +e.target.value; $('mscale-val').textContent = `${S.mscale}x`; applyModelScale(); });
$('show-refgrid').addEventListener('change', (e) => { if (refGrid) refGrid.visible = e.target.checked; });
$('show-chart').addEventListener('change', () => drawChartCursor());
$('play').addEventListener('click', () => { S.playing = !S.playing; if (S.playing && S.t >= S.tEnd) S.t = 0; });
$('speed').addEventListener('change', (e) => { S.speed = +e.target.value; });
$('timeline').addEventListener('input', (e) => { scrubbing = true; S.t = (+e.target.value / 10000) * S.tEnd; });
$('timeline').addEventListener('change', () => { scrubbing = false; });
$('collapse').addEventListener('click', () => { $('panel').classList.toggle('collapsed'); });
$('run-select').addEventListener('change', async (e) => {
  if (e.target.value === '') return;
  try { await loadIndexUrl(S.runs[+e.target.value].url); $('index-url').value = S.runs[+e.target.value].url; await rebuild(true); } catch (err) { msg(err.message, true); }
});
$('load-url').addEventListener('click', async () => {
  const u = $('index-url').value.trim();
  if (!u) return;
  try { await loadIndexUrl(u); await rebuild(true); } catch (err) { msg(err.message, true); }
});
$('file-input').addEventListener('change', (e) => loadLocalFiles(e.target.files).catch((err) => msg(err.message, true)));
window.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' && e.target.type === 'text') return;
  if (e.code === 'Space') { S.playing = !S.playing; e.preventDefault(); }
  else if (e.code === 'ArrowRight') S.t = Math.min(S.tEnd, S.t + 1);
  else if (e.code === 'ArrowLeft') S.t = Math.max(0, S.t - 1);
  else if (e.key === 'c') { const m = ['chase', 'orbit', 'free', 'top']; S.cam = m[(m.indexOf(S.cam) + 1) % 3]; $('cam-mode').value = S.cam; snapCam(); }
});

// ---------------------------------------------------------------- main loop
const clock = new THREE.Clock();
function animate() {
  const dt = Math.min(clock.getDelta(), 0.1);
  update(dt);
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}

// ---------------------------------------------------------------- debug / test API
window.fv = {
  S, THREE, camera, renderer, scene, controls,
  get ready() { return S.ready; },
  setTime(t) { S.t = t; update(0); renderer.render(scene, camera); },
  setMode, rebuild,
  setCam(m) { S.cam = m; $('cam-mode').value = m; snapCam(); update(0); },
  // body axes of shown aircraft k expressed in scene coords + velocity direction from the trail
  vectors(k = 0) {
    const d = S.shown[k]; const s = d.state;
    const nose = new THREE.Vector3(1, 0, 0).applyQuaternion(s.q), right = new THREE.Vector3(0, 1, 0).applyQuaternion(s.q);
    const tr = d.tr, i = s.i, j = Math.min(i + 1, tr.n - 1), i0 = j === i ? i - 1 : i;
    const vel = new THREE.Vector3(d.pts[3 * j] - d.pts[3 * i0], (d.pts[3 * j + 1] - d.pts[3 * i0 + 1]) / S.exag, d.pts[3 * j + 2] - d.pts[3 * i0 + 2]).normalize();
    return { nose: nose.toArray(), right: right.toArray(), vel: vel.toArray(), pos: s.pos.toArray(), modelY: d.root.position.y / d.vscale,
      altLogged: lerpCh(tr.alt, s.i, s.f), originAlt: tr.originAlt - d.ou, phiDeg: lerpCh(tr.hud.phi, s.i, s.f), t: S.t };
  },
};

// ---------------------------------------------------------------- boot
(async function boot() {
  resize();
  animate();
  S.cam = params.get('cam') || ((params.get('preset') || '').startsWith('plan:') ? 'top' : 'chase'); $('cam-mode').value = S.cam;
  if (params.get('speed')) { S.speed = +params.get('speed'); $('speed').value = params.get('speed'); }
  if (params.get('exag')) { S.exag = +params.get('exag'); $('exag').value = params.get('exag'); }
  if (params.get('scale')) { S.mscale = +params.get('scale'); $('mscale').value = S.mscale; $('mscale-val').textContent = `${S.mscale}x`; }
  if (params.get('notes') === '1') S.notesOpen = true;
  if (params.get('note')) {  // embedder caveat banner (e.g. build_b1_page: FD structure nodes vs planform strips)
    const nb = document.createElement('div');
    nb.id = 'page-note'; nb.textContent = params.get('note');
    nb.style.cssText = 'position:fixed;bottom:44px;left:45%;transform:translateX(-50%);max-width:40%;z-index:20;' +
      'background:rgba(60,40,0,.82);color:#ffd27a;border:1px solid #c90;border-radius:4px;padding:3px 8px;font:12px/1.35 sans-serif;text-align:center;pointer-events:none';
    document.body.appendChild(nb);
  }
  if (params.get('dofs')) S.dofs = params.get('dofs').split(',').filter(Boolean);
  if (params.get('defl')) { S.defl = +params.get('defl'); $('defl').value = S.defl; $('defl-val').textContent = `${S.defl}x`; }
  for (const [k, id] of [['layout', 'layout'], ['vref', 'vref'], ['cy', 'cy']]) if (params.get(k)) { S[k] = params.get(k); $(id).value = S[k]; }
  if (params.get('spacing')) { S.spacing = +params.get('spacing'); $('spacing').value = S.spacing; $('spacing-val').textContent = `${S.spacing} m`; }
  if (params.get('panel') === '0') $('panel').classList.add('collapsed');
  if (params.get('chart') === '0') $('show-chart').checked = false;
  if (params.get('refgrid') === '1') $('show-refgrid').checked = true;
  S.mode = params.get('mode') === 'compare' ? 'compare' : 'single';
  S.focus = S.mode === 'compare' ? (params.get('focus') != null ? +params.get('focus') : -1) : 0;
  document.querySelector(`input[name=mode][value=${S.mode}]`).checked = true;
  $('single-ui').hidden = S.mode !== 'single'; $('compare-ui').hidden = S.mode !== 'compare';
  try {
    const inl = window.FV_INLINE;
    if (!inl) await loadRunsList();
    if (inl && inl.index) {
      const { meta, entries } = parseIndex(inl.index, null);
      entries.forEach((e) => { e.url = e.file.split('/').pop(); e.key = e.url; });
      setEntries(entries, meta, meta.run_id ? `run ${meta.run_id} (inline)` : 'inline');
    } else if (params.get('traj')) {
      await loadTrajUrls(params.get('traj').split(','));
    } else if (params.get('index')) {
      $('index-url').value = params.get('index');
      await loadIndexUrl(params.get('index'));
    } else if (S.runs.length) {
      $('run-select').value = '0'; $('index-url').value = S.runs[0].url;
      await loadIndexUrl(S.runs[0].url);
    } else {
      msg('No run found. Use ?index=<url to index.json>, ?traj=<url>, or the file picker.');
    }
    if (S.entries.length) {
      if (params.get('t')) S.t = +params.get('t');
      await rebuild(true);
      if (params.get('play') === '1') S.playing = true;
    }
  } catch (e) { msg(String(e.message || e), true); }
})();
