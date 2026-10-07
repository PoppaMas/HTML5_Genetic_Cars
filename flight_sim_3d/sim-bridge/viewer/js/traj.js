// Loading + parsing of ga-flightsim-traj/1|/2 trajectories and their index.json.
// Channels are always looked up by NAME from `channels`; unknown channels/fields are ignored.
import * as THREE from 'three';

export const SUPPORTED_SCHEMA = /^ga-flightsim-traj\/[12](\.|$)/;

// ---------- fetching (plain JSON or gzip, detected by magic bytes) ----------
export async function decodeBuffer(buf) {
  const u8 = new Uint8Array(buf);
  let text;
  if (u8.length > 2 && u8[0] === 0x1f && u8[1] === 0x8b) {
    if (typeof DecompressionStream === 'undefined') throw new Error('gzip file but this browser has no DecompressionStream');
    const stream = new Blob([u8]).stream().pipeThrough(new DecompressionStream('gzip'));
    text = await new Response(stream).text();
  } else {
    text = new TextDecoder().decode(u8);
  }
  return JSON.parse(text);
}

export async function fetchJSON(url) {
  // inline mode (standalone HTML / Colab): window.FV_INLINE.files[name]
  const inl = (typeof window !== 'undefined') && window.FV_INLINE;
  if (inl && inl.files) {
    const key = url.split('/').pop().split('?')[0];
    if (inl.files[url]) return inl.files[url];
    if (inl.files[key]) return inl.files[key];
  }
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return decodeBuffer(await r.arrayBuffer());
}

// ---------- index ----------
export function isIndex(obj) {
  if (Array.isArray(obj)) return true;
  return obj && !obj.data && !obj.channels &&
    ['trajectories', 'files', 'entries', 'generations', 'items'].some((k) => Array.isArray(obj[k]));
}

export function parseIndex(obj, baseUrl) {
  const list = Array.isArray(obj) ? obj :
    (obj.trajectories || obj.files || obj.entries || obj.generations || obj.items);
  const meta = Array.isArray(obj) ? {} : obj;
  const entries = list.map((e, i) => {
    if (typeof e === 'string') e = { file: e };
    const file = e.file || e.path || e.url || e.filename;
    let url = file;
    try { url = baseUrl ? new URL(file, baseUrl).href : file; } catch (_) { /* keep */ }
    const gen = e.generation ?? e.gen ?? null;
    // run id: entry.run (combined multi-run indexes), else the index's run_id, else from traj_<ac>_<run>_g<N>
    const m = /^traj_(.+?)_(.+)_g(\d+)\.json(\.gz)?$/.exec(String(file).split('/').pop());
    const run = e.run ?? e.run_id ?? meta.run_id ?? (m ? m[2] : null);
    // cost (lower=better) is the agreed metric; older files carry "fitness" (ER's was always a cost / min)
    const sense = normSense(e.fitness_sense ?? meta.fitness_sense ?? (e.cost != null ? 'min' : null));
    // ids are opaque strings (ER: individual_id "T38:g19:r0", scenario id "T38:s1"); scenario_index = position
    const scenarioIndex = Number.isInteger(e.scenario_index) ? e.scenario_index : (Number.isInteger(e.scenario) ? e.scenario : null);
    return { generation: gen, fitness: e.cost ?? e.fitness ?? null, sense, aircraft: e.aircraft || meta.aircraft || null,
      run, file, url, status: e.status, key: `${url}`, order: i,
      scenario: e.scenario_id ?? e.scenario ?? null, scenarioIndex, individual: e.individual_id ?? null,
      rank: Number.isInteger(e.rank) ? e.rank : null, isBest: typeof e.is_best === 'boolean' ? e.is_best : null,
      verdict: e.verdict ?? null };
  }).filter((e) => e.file);
  const runOrder = new Map();
  entries.forEach((e) => { if (!runOrder.has(e.run)) runOrder.set(e.run, runOrder.size); });
  entries.sort((a, b) => String(a.aircraft ?? '').localeCompare(String(b.aircraft ?? '')) ||
    (runOrder.get(a.run) - runOrder.get(b.run)) || ((a.generation ?? a.order) - (b.generation ?? b.order)) ||
    ((a.isBest === false) - (b.isBest === false)) || ((a.rank ?? 0) - (b.rank ?? 0)) ||
    String(a.individual ?? '').localeCompare(String(b.individual ?? ''), undefined, { numeric: true }) ||
    ((a.scenarioIndex ?? 0) - (b.scenarioIndex ?? 0)) ||
    String(a.scenario ?? '').localeCompare(String(b.scenario ?? ''), undefined, { numeric: true }));
  return { meta, entries };
}

// ---------- trajectory ----------
const D2R = Math.PI / 180;
// ENU -> three.js (x=east, y=up, z=-north): rotation of -90 deg about x.
export const Q_THREE_FROM_ENU = new THREE.Quaternion(-Math.SQRT1_2, 0, 0, Math.SQRT1_2);
const Q_FRD_FROM_FLU = new THREE.Quaternion(1, 0, 0, 0); // 180 deg about x
const Q_ENU_FROM_NED = new THREE.Quaternion(Math.SQRT1_2, Math.SQRT1_2, 0, 0); // (x,y,z,w)

// Aero ZYX Euler (wrt NED) -> quaternion rotating body-FRD vectors into ENU (three.Quaternion).
export function eulerToQuatEnu(phi, theta, psi, out = new THREE.Quaternion()) {
  const qz = new THREE.Quaternion(0, 0, Math.sin(psi / 2), Math.cos(psi / 2));
  const qy = new THREE.Quaternion(0, Math.sin(theta / 2), 0, Math.cos(theta / 2));
  const qx = new THREE.Quaternion(Math.sin(phi / 2), 0, 0, Math.cos(phi / 2));
  return out.copy(Q_ENU_FROM_NED).multiply(qz).multiply(qy).multiply(qx);
}

function angleScale(units, name) {
  const u = (units && (units[name] || (units.channels && units.channels[name]))) || 'rad';
  return /deg/i.test(String(u)) ? D2R : 1;
}

export function parseTrajectory(obj, source = '') {
  if (!obj || !Array.isArray(obj.channels) || !Array.isArray(obj.data)) throw new Error(`${source}: not a trajectory (no channels/data)`);
  if (obj.schema && !SUPPORTED_SCHEMA.test(obj.schema)) console.warn(`${source}: schema ${obj.schema} not ga-flightsim-traj/1|/2; trying anyway`);
  const idx = {};
  obj.channels.forEach((c, i) => { idx[c] = i; });
  const n = obj.data.length;
  const col = (name) => {
    if (!(name in idx)) return null;
    const j = idx[name], a = new Float64Array(n);
    for (let i = 0; i < n; i++) { const v = obj.data[i][j]; a[i] = v === null || v === undefined ? NaN : v; }
    return a;
  };
  const ch = {};
  for (const c of obj.channels) ch[c] = col(c);
  const t = ch.t || ch.time || ch.time_s;
  if (!t) throw new Error(`${source}: no time channel 't'`);
  const frame = obj.frame || {};
  const originAlt = frame.origin_alt_m ?? 0;
  // altitude MSL
  let alt = ch.alt_msl_m || ch.alt_m || null;
  if (!alt && ch.z) alt = ch.z.map((z) => z + originAlt);
  if (!alt && ch.alt_ft) alt = ch.alt_ft.map((v) => v * 0.3048);
  if (!ch.x || !ch.y) throw new Error(`${source}: needs x/y (ENU) channels`);
  const z = ch.z || alt.map((a) => a - originAlt);

  // attitude: quaternion body->ENU preferred; Euler fallback
  const qENU = new Float64Array(4 * n);
  const bodyFLU = /FLU/i.test(frame.body_axes || '');
  const sPhi = angleScale(obj.units, 'phi'), sTh = angleScale(obj.units, 'theta'), sPsi = angleScale(obj.units, 'psi');
  const q = new THREE.Quaternion();
  const hasQ = ch.qw && ch.qx && ch.qy && ch.qz;
  for (let i = 0; i < n; i++) {
    if (hasQ) {
      q.set(ch.qx[i], ch.qy[i], ch.qz[i], ch.qw[i]).normalize();
      if (bodyFLU) q.multiply(Q_FRD_FROM_FLU);
    } else if (ch.phi && ch.theta && ch.psi) {
      eulerToQuatEnu(ch.phi[i] * sPhi, ch.theta[i] * sTh, ch.psi[i] * sPsi, q);
    } else q.copy(Q_ENU_FROM_NED);
    if (i > 0) { // keep hemisphere continuous for interpolation
      const d = q.x * qENU[4 * i - 4] + q.y * qENU[4 * i - 3] + q.z * qENU[4 * i - 2] + q.w * qENU[4 * i - 1];
      if (d < 0) q.set(-q.x, -q.y, -q.z, -q.w);
    }
    qENU.set([q.x, q.y, q.z, q.w], 4 * i);
  }
  // Euler for the HUD (deg). Use channels if present, else derive from the quaternion.
  const hud = { phi: new Float64Array(n), theta: new Float64Array(n), psi: new Float64Array(n) };
  for (let i = 0; i < n; i++) {
    if (ch.phi && ch.theta && ch.psi) {
      hud.phi[i] = ch.phi[i] * sPhi / D2R; hud.theta[i] = ch.theta[i] * sTh / D2R; hud.psi[i] = ch.psi[i] * sPsi / D2R;
    } else {
      // nose & right-wing vectors in ENU
      q.set(qENU[4 * i], qENU[4 * i + 1], qENU[4 * i + 2], qENU[4 * i + 3]);
      const nose = new THREE.Vector3(1, 0, 0).applyQuaternion(q), right = new THREE.Vector3(0, 1, 0).applyQuaternion(q);
      hud.theta[i] = Math.asin(Math.max(-1, Math.min(1, nose.z))) / D2R;
      hud.psi[i] = (Math.atan2(nose.x, nose.y) / D2R + 360) % 360;
      hud.phi[i] = Math.asin(Math.max(-1, Math.min(1, -right.z / Math.cos(hud.theta[i] * D2R)))) / D2R;
    }
  }
  // target altitude. Two optional channel kinds, told apart by shape (not by name, since producers differ):
  //   step command  - piecewise constant, jumps at target changes (e.g. bench target_alt_m)
  //   ramp reference - continuous profile the controller is asked to follow (e.g. phase1 600 fpm ramp)
  // Evolution Runner phase 1 logs target_alt_m = ramp reference and target_cmd_alt_m = step command.
  const tgt = obj.target || null;
  let schedule = null;
  const steps = tgt && (tgt.steps || tgt.schedule);
  if (Array.isArray(steps) && steps.length) schedule = steps.map((s) => Array.isArray(s) ? { t: s[0], alt_m: s[1] } : s);
  let rampCh = null, stepCh = null;
  for (const name of ['target_alt_m', 'alt_target_m', 'target_cmd_alt_m', 'alt_cmd_m', 'target_ref_alt_m']) {
    const a = ch[name];
    if (!a) continue;
    let jump = 0, lo = Infinity, hi = -Infinity;
    for (let i = 0; i < n; i++) {
      const v = a[i]; if (!Number.isFinite(v)) continue;
      if (v < lo) lo = v; if (v > hi) hi = v;
      if (i && Number.isFinite(a[i - 1])) jump = Math.max(jump, Math.abs(v - a[i - 1]));
    }
    const isRamp = hi - lo > 1 && jump < Math.max(5, 0.25 * (hi - lo));
    if (isRamp) { if (!rampCh) rampCh = name; } else if (!stepCh) stepCh = name;
  }
  const stepAt = (time, i) => {
    const a = stepCh && ch[stepCh];
    if (a && i !== undefined && Number.isFinite(a[i])) return a[i];
    if (schedule) { let v = schedule[0].alt_m; for (const s of schedule) if (time >= s.t) v = s.alt_m; return v; }
    if (tgt && Number.isFinite(tgt.alt_m)) return tgt.alt_m;
    if (rampCh && i !== undefined) return ch[rampCh][i];
    return NaN;
  };
  // the reference drawn in 3D / chart: ramp if present, else the step target
  const targetAt = rampCh ? (time, i) => (i !== undefined && Number.isFinite(ch[rampCh][i]) ? ch[rampCh][i] : stepAt(time, i)) : stepAt;
  const { data, ...meta } = obj;
  const structure = parseStructure(obj, ch);
  const planform = parsePlanform(obj);
  return {
    structure, planform,
    meta, source, n, t, x: ch.x, y: ch.y, z, alt, qENU, hud, ch, frame, originAlt, targetAt, stepAt, schedule,
    hasTarget: !!(rampCh || stepCh || schedule || (tgt && Number.isFinite(tgt.alt_m))),
    hasRamp: !!rampCh, rampChannel: rampCh, stepChannel: stepCh,
    aircraft: obj.aircraft || 'unknown', generation: obj.generation, fitness: obj.cost ?? obj.fitness,
    sense: normSense(obj.fitness_sense ?? (obj.cost != null ? 'min' : null)),
    scenarioIndex: obj.scenario_index ?? null, individualId: obj.individual_id ?? null, replay: obj.replay || null,
    runId: obj.run_id, events: Array.isArray(obj.events) ? obj.events : [],
    t0: t[0], t1: t[n - 1],
  };
}

// fitness_sense as ER writes it: "min" in run.json, free text in trajectory files ("minimize (GA cost, ...)")
export function normSense(v) {
  if (v == null) return null;
  return String(v).trim().toLowerCase().startsWith('max') ? 'max' : 'min';
}

// P3-B1 planform (FD INTERFACE_v2 section 14), optional header field `planform` (also accepted: `planform_b1`, or
// either inside `structure`): {schema: "fd-planform/1", source: "P3-B1", genes: {name: value}, synthetic?,
//   wingR / wingL (or symmetric: true with one of wingR / wingL / wing): {span_frac[], y_m[], chord_m[], le_x_m[],
//   twist_rad[]}, sweep_qc_rad}. SI, body FRD (x fwd, y right), twist + = LE up. Absent / malformed -> null (ignored).
const _num = (a) => Array.isArray(a) && a.length >= 2 && a.every((v) => Number.isFinite(v));
function _pfSide(w) {
  if (!w || typeof w !== 'object' || !_num(w.chord_m) || w.chord_m.some((c) => !(c > 0))) return null;
  const n = w.chord_m.length;
  const ok = (a) => _num(a) && a.length === n;
  let sf = ok(w.span_frac) ? w.span_frac.slice() : null;
  const y = ok(w.y_m) ? w.y_m.map(Math.abs) : null;
  if (!sf && y) { const y0 = y[0], y1 = y[n - 1]; sf = y.map((v) => (y1 > y0 ? (v - y0) / (y1 - y0) : 0)); }
  if (!sf) sf = w.chord_m.map((_, i) => i / (n - 1));
  for (let i = 1; i < n; i++) if (!(sf[i] > sf[i - 1]) || (y && !(y[i] > y[i - 1]))) return null;  // must increase outboard
  return { span_frac: sf, y_m: y, chord_m: w.chord_m.slice(), le_x_m: ok(w.le_x_m) ? w.le_x_m.slice() : null,
    twist_rad: ok(w.twist_rad) ? w.twist_rad.slice() : null, n };
}
// B1 r1: FD node_layout_b1 per-node wing geometry on structure.components[wingR|wingL] (SI, body FRD, CG origin):
// le_nodes_body_m / te_nodes_body_m [[x,y,z]], chord_m, geometric_twist_rad (+ = LE up), axis_nodes_body_m = elastic
// axis. -> a planform side with ABSOLUTE body x (absolute: true; ea_x = elastic-axis x, the twist pivot).
function _nodeSide(c) {
  if (!c || !Array.isArray(c.le_nodes_body_m) || !Array.isArray(c.te_nodes_body_m)) return null;
  const le = c.le_nodes_body_m, te = c.te_nodes_body_m, ax = c.axis_nodes_body_m, n = le.length;
  const pt = (p) => Array.isArray(p) && p.length >= 3 && p.every(Number.isFinite);
  if (n < 2 || te.length !== n || !le.every(pt) || !te.every(pt)) return null;
  const chord = _num(c.chord_m) && c.chord_m.length === n ? c.chord_m.slice()
    : le.map((p, i) => Math.hypot(p[0] - te[i][0], p[2] - te[i][2]));
  const y = le.map((p) => Math.abs(p[1]));
  for (let i = 1; i < n; i++) if (!(y[i] > y[i - 1])) return null;
  if (chord.some((v) => !(v > 0))) return null;
  const sf = _num(c.node_span_frac) && c.node_span_frac.length === n ? c.node_span_frac.slice()
    : y.map((v) => (v - y[0]) / (y[n - 1] - y[0]));
  const tw = _num(c.geometric_twist_rad) && c.geometric_twist_rad.length === n ? c.geometric_twist_rad.slice() : null;
  const eax = Array.isArray(ax) && ax.length === n && ax.every(pt) ? ax.map((p) => p[0]) : le.map((p, i) => p[0] - 0.25 * chord[i]);
  return { span_frac: sf, y_m: y, chord_m: chord, le_x_m: le.map((p) => p[0]), twist_rad: tw, ea_x_m: eax, n, absolute: true };
}
export function nodePlanform(obj) {
  const comps = (obj && obj.structure && Array.isArray(obj.structure.components)) ? obj.structure.components : [];
  const get = (nm) => _nodeSide(comps.find((c) => c && c.name === nm));
  const R = get('wingR'), L = get('wingL');
  return R || L ? { wingR: R || L, wingL: L || R } : null;
}
const _sameArr = (a, b) => (a === b) || (!!a && !!b && a.length === b.length && a.every((v, i) => Math.abs(v - b[i]) < 1e-9));
const _sameSide = (a, b) => _sameArr(a.chord_m, b.chord_m) && _sameArr(a.le_x_m, b.le_x_m) && _sameArr(a.twist_rad, b.twist_rad) &&
  _sameArr(a.span_frac, b.span_frac);
export function parsePlanform(obj) {
  if (!obj || typeof obj !== 'object') return null;
  const st = obj.structure || {};
  let field = null, raw = null;
  for (const [k, o] of [['planform', obj], ['planform_b1', obj], ['structure.planform', st], ['structure.planform_b1', st]]) {
    const v = o[k.replace('structure.', '')];
    if (v && typeof v === 'object') { field = k; raw = v; break; }
  }
  const nodes = nodePlanform(obj);
  if (!raw && nodes) raw = { source: 'structure nodes' };   // per-node geometry without a planform header
  if (!raw) return null;
  if (!field) field = 'structure.nodes';
  let R = _pfSide(raw.wingR), L = _pfSide(raw.wingL);
  const W = _pfSide(raw.wing);
  if (raw.symmetric === true || !R || !L) { R = R || L || W; L = L || R; }
  // geometry source: FD per-node LE / TE (r1 node_layout_b1) when present, else the planform strips
  const geom = nodes ? 'nodes' : 'strips';
  const strips = R ? { wingR: R, wingL: L } : null;
  if (nodes) { R = nodes.wingR; L = nodes.wingL; }
  if (!R) return null;
  const sweep = Number.isFinite(raw.sweep_qc_rad) ? raw.sweep_qc_rad
    : (Number.isFinite(raw.sweep_qc_deg) ? raw.sweep_qc_deg * Math.PI / 180 : null);
  const tw = R.twist_rad;
  return { field, schema: raw.schema || null, source: raw.source || null, genes: raw.genes || null,
    synthetic: raw.synthetic === true, symmetric: raw.symmetric === true || R === L || _sameSide(R, L), wingR: R, wingL: L,
    sweep_qc_rad: sweep, taper: (strips ? strips.wingR : R).chord_m[(strips ? strips.wingR : R).n - 1] / (strips ? strips.wingR : R).chord_m[0],
    twist_tip_rad: tw ? tw[R.n - 1] : null, geom, strips,
    node_layout: (obj.structure && obj.structure.node_layout) || null };
}

// soft-body v2: optional `structure` block + channels '<component>.<dof>.<node_idx>' (metres / rad, body FRD),
// dof dz / dy / dx / twist; a component flagged `estimated: true` (tip-only estimate, no FD nodal data) is labelled.
// Unknown component names are kept (the viewer only deforms the ones its procedural model has).
export const STRUCT_COMPONENTS = ['wingL', 'wingR', 'htail', 'vtail', 'fuselage'];
export function parseStructure(obj, ch) {
  const st = obj && obj.structure;
  if (!st || !Array.isArray(st.components)) return null;
  const comps = [];
  for (const c of st.components) {
    if (!c || typeof c.name !== 'string' || !Array.isArray(c.axis_nodes_body_m) || c.axis_nodes_body_m.length < 2) continue;
    const n = c.axis_nodes_body_m.length;
    const dof = Array.isArray(c.dof) && c.dof.length ? c.dof : ['dz', 'dy', 'twist'];
    const chm = {};
    let found = 0;
    for (const d of dof) {
      const arrs = [];
      for (let i = 0; i < n; i++) { const a = ch[`${c.name}.${d}.${i}`] || null; arrs.push(a); if (a) found++; }
      if (arrs.some(Boolean)) chm[d] = arrs;
    }
    comps.push({ name: c.name, axis_nodes: c.axis_nodes_body_m, dof, ch: chm, nChannels: found, estimated: c.estimated === true });
  }
  if (!comps.length) return null;
  return { synthetic: !!(st.synthetic || obj.synthetic), source: st.source || null, components: comps,
    estimated: comps.filter((c) => c.estimated).map((c) => c.name) };
}

// index i and fraction f such that time = t[i] + f*(t[i+1]-t[i]) (clamped)
export function locate(tr, time) {
  const t = tr.t;
  if (time <= t[0]) return [0, 0];
  if (time >= t[tr.n - 1]) return [tr.n - 1, 0];
  let lo = 0, hi = tr.n - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (t[m] <= time) lo = m; else hi = m; }
  return [lo, (time - t[lo]) / (t[hi] - t[lo])];
}

export function lerpCh(arr, i, f) {
  if (!arr) return NaN;
  if (f === 0 || i + 1 >= arr.length) return arr[i];
  return arr[i] + (arr[i + 1] - arr[i]) * f;
}

export function lerpAngleDeg(arr, i, f) {
  if (f === 0 || i + 1 >= arr.length) return arr[i];
  let d = arr[i + 1] - arr[i];
  d = ((d + 540) % 360) - 180;
  return arr[i] + d * f;
}

const _qa = new THREE.Quaternion(), _qb = new THREE.Quaternion();
export function quatAt(tr, i, f, out) {
  const q = tr.qENU;
  _qa.set(q[4 * i], q[4 * i + 1], q[4 * i + 2], q[4 * i + 3]);
  if (f > 0 && i + 1 < tr.n) { _qb.set(q[4 * i + 4], q[4 * i + 5], q[4 * i + 6], q[4 * i + 7]); _qa.slerp(_qb, f); }
  return out.copy(Q_THREE_FROM_ENU).multiply(_qa);
}
