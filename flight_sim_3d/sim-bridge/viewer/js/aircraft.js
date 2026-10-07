// Procedural low-poly aircraft models built from a small config, plus optional glTF.
//
// Model space is the aircraft BODY frame, FRD: +x = nose, +y = right wing, +z = down.
// (The viewer applies the body->ENU quaternion from the trajectory, then ENU->three.)
import * as THREE from 'three';

// Dimensions in metres. Add a new aircraft by adding an entry here (or by putting an
// `aircraft_model` object with the same keys into the trajectory / index JSON).
export const PRESETS = {
  'generic-prop': {
    label: 'Generic light single', span_m: 10.5, length_m: 8.0, fuselage_width_m: 1.1, fuselage_height_m: 1.3,
    wing_root_chord_m: 1.55, wing_tip_chord_m: 1.2, wing_pos: 'high', wing_x_frac: 0.30, sweep_deg: 0,
    dihedral_deg: 1.7, htail_span_m: 3.4, htail_chord_m: 1.0, vtail_height_m: 1.5, vtail_chord_m: 1.2,
    vtail_sweep_deg: 35, propeller: true, nozzle: false,
  },
  c172x: {
    label: 'Cessna 172 (JSBSim c172x)', span_m: 11.0, length_m: 8.28, fuselage_width_m: 1.1, fuselage_height_m: 1.4,
    wing_root_chord_m: 1.63, wing_tip_chord_m: 1.13, wing_pos: 'high', wing_x_frac: 0.30, sweep_deg: 0,
    dihedral_deg: 1.7, htail_span_m: 3.45, htail_chord_m: 1.1, vtail_height_m: 1.55, vtail_chord_m: 1.25,
    vtail_sweep_deg: 35, propeller: true, nozzle: false,
  },
  t6texan2: {
    label: 'Beechcraft T-6 Texan II (JSBSim t6texan2)', span_m: 10.2, length_m: 10.2, fuselage_width_m: 1.1, fuselage_height_m: 1.5,
    wing_root_chord_m: 2.3, wing_tip_chord_m: 1.1, wing_pos: 'low', wing_x_frac: 0.36, sweep_deg: 2,
    dihedral_deg: 6, htail_span_m: 4.6, htail_chord_m: 1.3, vtail_height_m: 1.9, vtail_chord_m: 1.7,
    vtail_sweep_deg: 30, propeller: true, nozzle: false,
  },
  // Northrop T-38 Talon (JSBSim T38): small, thin, low trapezoidal wing, twin engines, long pointed nose.
  t38: {
    label: 'Northrop T-38 Talon (JSBSim T38)', span_m: 7.70, length_m: 14.13, fuselage_width_m: 1.25, fuselage_height_m: 1.35,
    wing_root_chord_m: 3.4, wing_tip_chord_m: 0.75, wing_pos: 'low', wing_x_frac: 0.52, sweep_deg: 32,
    dihedral_deg: 0, htail_span_m: 4.3, htail_chord_m: 1.7, htail_sweep_deg: 35, vtail_height_m: 2.4,
    vtail_chord_m: 2.7, vtail_sweep_deg: 42, propeller: false, nozzle: true, nozzles: 2,
    nose_frac: 0.30, cabin_frac: 0.30, canopy: 'bubble', canopy_len_frac: 0.22,
  },
  // General Dynamics F-16 (JSBSim f16): cropped-delta wing, single engine, ventral-ish intake omitted.
  f16: {
    label: 'F-16 Fighting Falcon (JSBSim f16)', span_m: 9.96, length_m: 15.06, fuselage_width_m: 1.5, fuselage_height_m: 1.6,
    wing_root_chord_m: 5.0, wing_tip_chord_m: 1.0, wing_pos: 'mid', wing_x_frac: 0.45, sweep_deg: 40,
    dihedral_deg: 0, htail_span_m: 5.6, htail_chord_m: 2.2, htail_sweep_deg: 40, vtail_height_m: 3.0,
    vtail_chord_m: 3.0, vtail_sweep_deg: 45, propeller: false, nozzle: true, nozzles: 1,
    nose_frac: 0.22, cabin_frac: 0.33, canopy: 'bubble', canopy_len_frac: 0.2,
  },
  // Boeing 737-300 class (JSBSim 737): low swept wing with dihedral, two underwing turbofans,
  // conventional (not T-) tail with the stabiliser on the fuselage.
  '737': {
    label: 'Boeing 737 (JSBSim 737)', span_m: 28.9, length_m: 33.4, fuselage_width_m: 3.76, fuselage_height_m: 4.0,
    wing_root_chord_m: 7.3, wing_tip_chord_m: 1.6, wing_pos: 'low', wing_x_frac: 0.40, sweep_deg: 28,
    dihedral_deg: 6, htail_span_m: 12.7, htail_chord_m: 3.6, htail_sweep_deg: 32, htail_dihedral_deg: 7,
    vtail_height_m: 6.0, vtail_chord_m: 5.6, vtail_sweep_deg: 38, propeller: false, nozzle: false,
    engines_underwing: true, engine_span_frac: 0.34, engine_len_m: 3.6, engine_dia_m: 1.6,
    nose_frac: 0.09, cabin_frac: 0.66, canopy: 'airliner',
  },
  'generic-jet': {
    label: 'Generic fighter jet', span_m: 9.96, length_m: 15.06, fuselage_width_m: 1.4, fuselage_height_m: 1.6,
    wing_root_chord_m: 5.0, wing_tip_chord_m: 1.0, wing_pos: 'mid', wing_x_frac: 0.42, sweep_deg: 40,
    dihedral_deg: 0, htail_span_m: 5.6, htail_chord_m: 2.2, htail_sweep_deg: 40, vtail_height_m: 3.0,
    vtail_chord_m: 3.0, vtail_sweep_deg: 45, propeller: false, nozzle: true,
  },
  'generic-airliner': {
    label: 'Generic airliner', span_m: 34.1, length_m: 37.6, fuselage_width_m: 3.9, fuselage_height_m: 4.1,
    wing_root_chord_m: 7.0, wing_tip_chord_m: 1.6, wing_pos: 'low', wing_x_frac: 0.42, sweep_deg: 25,
    dihedral_deg: 5, htail_span_m: 12.5, htail_chord_m: 3.5, htail_sweep_deg: 30, vtail_height_m: 7.0,
    vtail_chord_m: 5.5, vtail_sweep_deg: 35, propeller: false, nozzle: false, engines_underwing: true,
    nose_frac: 0.09, cabin_frac: 0.66, canopy: 'airliner',
  },
};
// JSBSim model names -> preset (exact match first, then regex).
export const ALIASES = [
  [/^c172/i, 'c172x'], [/^t6/i, 't6texan2'], [/^t-?38/i, 't38'], [/^f-?16/i, 'f16'], [/^(b-?)?737/i, '737'],
  [/^(c182|c310|pa28|j3|cub|ball|dhc|ov10|pc)/i, 'generic-prop'],
  [/^(f[-_]?\d+|f15|f22|f35|x15|a4|mig|su\d|jet|j246)/i, 'generic-jet'],
  [/^(747|757|767|777|787|a3\d\d|b7\d\d|md11|dc|crj|e1\d\d|global5000)/i, 'generic-airliner'],
];

export function resolvePreset(aircraft, override) {
  let key = aircraft && PRESETS[aircraft] ? aircraft : (aircraft && PRESETS[String(aircraft).toLowerCase()] ? String(aircraft).toLowerCase() : null);
  if (!key && aircraft) for (const [re, k] of ALIASES) if (re.test(aircraft)) { key = k; break; }
  key = key || 'generic-prop';
  return { key, ...PRESETS['generic-prop'], ...PRESETS[key], ...(override || {}) };
}

function planformGeom(points, thickness) {
  // points: [[x, y], ...] in FRD x/y; extruded along +z (down) and centred on z.
  const shape = new THREE.Shape(points.map(([x, y]) => new THREE.Vector2(x, y)));
  const g = new THREE.ExtrudeGeometry(shape, { depth: thickness, bevelEnabled: false });
  g.translate(0, 0, -thickness / 2);
  return g;
}

// A lifting surface half (right side when side=+1) with an optional hinged trailing-edge control
// surface on the outer part. Returns {mesh, hinge} where hinge rotates about its local y axis
// (positive angle = trailing edge DOWN, i.e. +z in FRD).
// Planform (P3-B1) -> station functions of f = |y| / semiSpan for halfSurface. Chord and built-in twist are taken
// as given (metres / rad); the LE line keeps the procedural root LE as its anchor and follows the planform's LE x
// offsets relative to its first station (le_x_m, else derived from sweep_qc_rad: x_qc = -tan(sweep) * dy,
// LE = x_qc + c/4). Inboard of the first station (inside the fuselage) the first station is held.
export function planformStations(pf, semiSpan, sweepQc) {
  const n = pf.n;
  const y = pf.y_m ? pf.y_m.map((v) => Math.min(Math.abs(v), semiSpan)) : pf.span_frac.map((f) => f * semiSpan);
  let le = pf.le_x_m;
  if (!le) {
    const tq = Math.tan(Number.isFinite(sweepQc) ? sweepQc : 0);
    le = y.map((yy, i) => -tq * (yy - y[0]) + 0.25 * pf.chord_m[i]);
  }
  const eaOf = (i) => (pf.ea_x_m ? pf.ea_x_m[i] - le[i] : -0.25 * pf.chord_m[i]);   // elastic axis x - LE x
  const zOf = (i) => (pf.z_rel_m ? pf.z_rel_m[i] : 0);   // B2a: FD node z relative to the root node (dihedral delta)
  const f = [0], c = [pf.chord_m[0]], dle = [0], tw = [pf.twist_rad ? pf.twist_rad[0] : 0], ea = [eaOf(0)], zr = [zOf(0)];
  for (let i = 0; i < n; i++) {
    const fi = y[i] / semiSpan;
    if (fi <= f[f.length - 1] + 1e-9) { c[c.length - 1] = pf.chord_m[i]; ea[ea.length - 1] = eaOf(i); zr[zr.length - 1] = zOf(i); continue; }
    f.push(fi); c.push(pf.chord_m[i]); dle.push(le[i] - le[0]); tw.push(pf.twist_rad ? pf.twist_rad[i] : 0); ea.push(eaOf(i)); zr.push(zOf(i));
  }
  if (f[f.length - 1] < 1 - 1e-9) {  // strips end at the last strip centre: extrapolate the last segment to the tip
    const k = f.length - 1, g = (1 - f[k]) / Math.max(1e-9, f[k] - f[k - 1]);
    f.push(1); c.push(Math.max(0.05 * c[0], c[k] + (c[k] - c[k - 1]) * g)); dle.push(dle[k] + (dle[k] - dle[k - 1]) * g);
    tw.push(tw[k] + (tw[k] - tw[k - 1]) * g); ea.push(ea[k] + (ea[k] - ea[k - 1]) * g); zr.push(zr[k] + (zr[k] - zr[k - 1]) * g);
  }
  const lerp = (arr) => (q) => {
    if (q <= f[0]) return arr[0];
    for (let j = 1; j < f.length; j++) if (q <= f[j]) return arr[j - 1] + (arr[j] - arr[j - 1]) * (q - f[j - 1]) / (f[j] - f[j - 1]);
    return arr[arr.length - 1];
  };
  const out = { f, chordAt: lerp(c), dleAt: lerp(dle), twistAt: lerp(tw), hasTwist: tw.some((v) => v !== 0) };
  if (pf.absolute) {  // r1 per-node geometry: absolute body x of the LE; built-in twist pivots on FD's elastic axis
    const eaAt = lerp(ea);
    out.absolute = true;
    out.le0 = le[0];
    out.pivotAt = (q) => le[0] + out.dleAt(q) + eaAt(q);
  }
  // B2a: node z (dihedral delta, baked in by FD) relative to the root, applied on top of the procedural wing height
  if (pf.z_rel_m) out.zAt = lerp(zr);
  return out;
}

// A lifting surface half (right side when side=+1) with an optional hinged trailing-edge control
// surface on the outer part. Returns {mesh, hinge} where hinge rotates about its local y axis
// (positive angle = trailing edge DOWN, i.e. +z in FRD). `planform` (optional, parsePlanform side): chord / LE /
// built-in twist per strip replace the straight-tapered trapezoid.
function halfSurface({ side, xLE, rootChord, tipChord, semiSpan, sweepDeg, thickness, mat, ctrlMat,
  ctrlFrac = 0.25, ctrlFrom = 0.0, ctrlTo = 1.0, dihedralDeg = 0, planform = null, sweepQc = null }) {
  const grp = new THREE.Group();
  const sweep = Math.tan(THREE.MathUtils.degToRad(sweepDeg)) * semiSpan;
  const pst = planform ? planformStations(planform, semiSpan, sweepQc) : null;
  const chordAt = pst ? pst.chordAt : (f) => rootChord + (tipChord - rootChord) * f;
  // r1 per-node geometry (pst.absolute): LE at FD's absolute body x; else anchored at the procedural root LE
  const leAt = pst ? (pst.absolute ? (f) => pst.le0 + pst.dleAt(f) : (f) => xLE + pst.dleAt(f)) : (f) => xLE - sweep * f;
  // main surface (aft boundary = hinge line over the control-surface span)
  const pts = [];
  const N = 6;
  const fs = pst ? pst.f.slice() : [0, 1];
  if (pst) for (const q of [ctrlFrom, ctrlTo]) if (!fs.some((v) => Math.abs(v - q) < 1e-6)) fs.push(q);
  fs.sort((a, b) => a - b);
  for (const f of fs) pts.push([leAt(f), f * semiSpan]);
  if (pst) {
    // trailing edge tip -> root; across the control span only its two ends (the hinge line is straight)
    for (let i = fs.length - 1; i >= 0; i--) {
      const f = fs[i];
      const inCtrl = f >= ctrlFrom - 1e-6 && f <= ctrlTo + 1e-6;
      const edge = Math.abs(f - ctrlFrom) < 1e-6 || Math.abs(f - ctrlTo) < 1e-6;
      if (inCtrl && !edge) continue;
      pts.push([leAt(f) - chordAt(f) * (inCtrl ? (1 - ctrlFrac) : 1), f * semiSpan]);
    }
  } else {
    for (let i = N; i >= 0; i--) {
      const f = i / N;
      const inCtrl = f >= ctrlFrom - 1e-6 && f <= ctrlTo + 1e-6;
      const c = chordAt(f) * (inCtrl ? (1 - ctrlFrac) : 1);
      pts.push([leAt(f) - c, f * semiSpan]);
    }
  }
  const g = planformGeom(pts.map(([x, y]) => [x, y * side]), thickness);
  if (side < 0) g.computeVertexNormals();
  const mesh = new THREE.Mesh(g, mat);
  grp.add(mesh);
  let hinge = null;
  if (ctrlTo > ctrlFrom) {
    // control surface, built relative to a hinge pivot at the inner hinge point
    const fi = ctrlFrom, fo = ctrlTo;
    const hx0 = leAt(fi) - chordAt(fi) * (1 - ctrlFrac), hy0 = fi * semiSpan;
    const hx1 = leAt(fo) - chordAt(fo) * (1 - ctrlFrac), hy1 = fo * semiSpan;
    const cpts = [[0, 0], [hx1 - hx0, hy1 - hy0], [hx1 - hx0 - chordAt(fo) * ctrlFrac, hy1 - hy0],
      [-chordAt(fi) * ctrlFrac, 0]];
    const cg = planformGeom(cpts.map(([x, y]) => [x, y * side]), thickness * 0.8);
    hinge = new THREE.Group();
    hinge.position.set(hx0, hy0 * side, 0);
    // align the hinge group's y axis with the (possibly swept) hinge line
    const ang = Math.atan2(hx1 - hx0, hy1 - hy0); // rotation about z
    hinge.rotation.z = -ang * side;
    const inner = new THREE.Mesh(cg, ctrlMat);
    inner.rotation.z = ang * side;
    hinge.add(inner);
    grp.add(hinge);
  }
  if (pst && (pst.hasTwist || pst.zAt)) {
    // built-in (geometric) twist: rotate each station's section about its quarter-chord point (r1 per-node geometry:
    // about FD's elastic axis), + = LE up (-z)
    grp.updateMatrixWorld(true);
    const M = new THREE.Matrix4(), Mi = new THREE.Matrix4(), v = new THREE.Vector3();
    grp.traverse((m) => {
      if (!m.isMesh) return;
      M.copy(m.matrixWorld); Mi.copy(M).invert();
      const pa = m.geometry.attributes.position;
      for (let k = 0; k < pa.count; k++) {
        v.fromBufferAttribute(pa, k).applyMatrix4(M);
        const f = Math.min(1, Math.abs(v.y) / semiSpan), th = pst.hasTwist ? pst.twistAt(f) : 0;
        if (th) {
          const xq = pst.pivotAt ? pst.pivotAt(f) : leAt(f) - 0.25 * chordAt(f), dx = v.x - xq;
          v.x = xq + dx * Math.cos(th); v.z -= dx * Math.sin(th);
        }
        if (pst.zAt) v.z += pst.zAt(f);   // FD node z (dihedral delta): absolute node shape, not re-derived from dGamma
        v.applyMatrix4(Mi);
        pa.setXYZ(k, v.x, v.y, v.z);
      }
      pa.needsUpdate = true;
      m.geometry.computeVertexNormals();
    });
  }
  grp.rotation.x = -side * THREE.MathUtils.degToRad(dihedralDeg); // tips up (-z)
  return { group: grp, hinge };
}

export function buildProcedural(cfg, color = '#ff6b35', planform = null) {
  const accent = new THREE.Color(color);
  const body = new THREE.MeshStandardMaterial({ color: 0xf2f2f2, roughness: 0.55, metalness: 0.1, flatShading: true });
  const wingMat = new THREE.MeshStandardMaterial({ color: accent, roughness: 0.5, metalness: 0.1, flatShading: true, side: THREE.DoubleSide });
  const ctrlMat = new THREE.MeshStandardMaterial({ color: accent.clone().offsetHSL(0, 0, -0.18), roughness: 0.5, flatShading: true, side: THREE.DoubleSide });
  const dark = new THREE.MeshStandardMaterial({ color: 0x222222, roughness: 0.8, flatShading: true });
  const L = cfg.length_m, W = cfg.fuselage_width_m, H = cfg.fuselage_height_m;
  const R = Math.max(W, H) / 2;
  const root = new THREE.Group();
  const model = new THREE.Group();
  root.add(model);
  // soft-body v2: objects per structural component (deformed by applyStructure when a trajectory has `structure`)
  const parts = { wingL: [], wingR: [], htail: [], vtail: [], fuselage: [] };

  // fuselage: nose cone, cabin cylinder, tail cone (axis along +x)
  const noseL = L * (cfg.nose_frac ?? 0.12), cabL = L * (cfg.cabin_frac ?? 0.38), tailL = L - noseL - cabL;
  const xNose = L * 0.45; // nose tip x (origin ~ CG)
  const addCyl = (rTop, rBot, len, xCenter, mat, seg = 10) => {
    const g = new THREE.CylinderGeometry(rTop, rBot, len, seg, 1);
    g.rotateZ(-Math.PI / 2); // +y (top) -> +x
    const m = new THREE.Mesh(g, mat);
    m.position.x = xCenter;
    m.scale.set(1, W / (2 * R), H / (2 * R));
    model.add(m);
    parts.fuselage.push(m);
    return m;
  };
  const jetNose = cfg.nozzle && !cfg.engines_underwing;
  if (cfg.canopy === 'airliner') { // blunt rounded nose: short cone + cap
    addCyl(R * 0.62, R, noseL, xNose - noseL / 2, body, 14);
  } else {
    addCyl(R * (jetNose ? 0.06 : 0.55), R, noseL, xNose - noseL / 2, body, jetNose ? 10 : 10);
  }
  addCyl(R, R, cabL, xNose - noseL - cabL / 2, body, cfg.canopy === 'airliner' ? 16 : 10);
  const tailTip = cfg.nozzle ? (cfg.nozzles === 2 ? 0.8 : 0.55) : (cfg.canopy === 'airliner' ? 0.18 : 0.22);
  const tail = addCyl(R, R * tailTip, tailL, xNose - noseL - cabL - tailL / 2, body, cfg.canopy === 'airliner' ? 16 : 10);
  tail.position.z = cfg.nozzle ? 0 : -R * (cfg.canopy === 'airliner' ? 0.35 : 0.25); // tail cone rises a bit
  // canopy / windows
  if (cfg.canopy === 'airliner') {
    // windshield band near the nose + a row of cabin windows (dark strips)
    const ws = new THREE.Mesh(new THREE.BoxGeometry(L * 0.025, W * 0.7, H * 0.12), dark);
    ws.position.set(xNose - noseL * 0.85, 0, -H * 0.28);
    model.add(ws);
    parts.fuselage.push(ws);
    for (const side of [1, -1]) {
      const win = new THREE.Mesh(new THREE.BoxGeometry(cabL * 0.85, 0.05, H * 0.08), dark);
      win.position.set(xNose - noseL - cabL * 0.5, side * W * 0.5, -H * 0.12);
      model.add(win);
      parts.fuselage.push(win);
    }
  } else {
    const bubble = cfg.canopy === 'bubble';
    const canopy = new THREE.Mesh(new THREE.SphereGeometry(R * 0.75, 12, 6, 0, Math.PI * 2, 0, Math.PI / 2), dark);
    canopy.rotation.x = -Math.PI / 2; // dome toward -z (up)
    const clen = bubble ? (cfg.canopy_len_frac ?? 0.2) * L : R * 1.2;
    canopy.scale.set(bubble ? clen / (2 * R * 0.75) : 1.6, W / (2 * R) * (bubble ? 0.7 : 0.95), bubble ? 0.9 : 0.7);
    canopy.position.set(bubble ? xNose - noseL * 0.95 - clen * 0.3 : xNose - noseL - cabL * 0.3, 0, -H * (bubble ? 0.38 : 0.32));
    model.add(canopy);
    parts.fuselage.push(canopy);
  }

  // wings
  const zWing = cfg.wing_pos === 'high' ? -H / 2 : cfg.wing_pos === 'low' ? H * 0.38 : 0;
  const xWingLE = xNose - L * cfg.wing_x_frac;
  const wingT = Math.max(0.08, cfg.wing_root_chord_m * 0.09);
  const surfaces = {};
  // B2b (area_scale / aspect_scale, span-scaled stations): the wing span follows FD's node tip; B1 r1 / B2a keep the
  // procedural span (identical rendering)
  const b2b = planform && planform.b2 && (planform.b2.area_scale != null || planform.b2.aspect_scale != null);
  const tipY = b2b && planform.wingR && planform.wingR.y_m ? planform.wingR.y_m[planform.wingR.n - 1] : null;
  const wingSemi = Number.isFinite(tipY) && tipY > 0 ? tipY : cfg.span_m / 2;
  for (const side of [1, -1]) {
    const s = halfSurface({ side, xLE: xWingLE, rootChord: cfg.wing_root_chord_m, tipChord: cfg.wing_tip_chord_m,
      semiSpan: wingSemi, sweepDeg: cfg.sweep_deg, thickness: wingT, mat: wingMat, ctrlMat,
      ctrlFrac: 0.25, ctrlFrom: 0.55, ctrlTo: 0.95, dihedralDeg: cfg.dihedral_deg,
      planform: planform ? planform[side > 0 ? 'wingR' : 'wingL'] : null, sweepQc: planform ? planform.sweep_qc_rad : null });
    s.group.position.z = zWing;
    model.add(s.group);
    surfaces[side > 0 ? 'aileronR' : 'aileronL'] = s.hinge;
    const wingPart = parts[side > 0 ? 'wingR' : 'wingL'];
    wingPart.push(s.group);
    if (cfg.wing_pos === 'high') { // struts
      const sg = new THREE.CylinderGeometry(0.035, 0.035, 1, 5);
      const strut = new THREE.Mesh(sg, dark);
      // B2a: the strut top follows FD's node z (dihedral delta) at its span station
      const pw = planform && planform.b2 ? planform[side > 0 ? 'wingR' : 'wingL'] : null;
      let zs = 0;
      if (pw && pw.z_rel_m && pw.y_m) { const ys = cfg.span_m * 0.28, yy = pw.y_m; let j = 1; while (j < yy.length - 1 && yy[j] < ys) j++;
        const u = Math.max(0, Math.min(1, (ys - yy[j - 1]) / ((yy[j] - yy[j - 1]) || 1))); zs = pw.z_rel_m[j - 1] + (pw.z_rel_m[j] - pw.z_rel_m[j - 1]) * u; }
      const a = new THREE.Vector3(xWingLE - cfg.wing_root_chord_m * 0.4, side * cfg.span_m * 0.28, zWing + zs);
      const b = new THREE.Vector3(xWingLE - cfg.wing_root_chord_m * 0.4, side * W * 0.45, H * 0.4);
      strut.position.copy(a).add(b).multiplyScalar(0.5);
      strut.scale.y = a.distanceTo(b);
      strut.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), b.clone().sub(a).normalize());
      model.add(strut);
      wingPart.push(strut);
    }
    if (cfg.engines_underwing) {
      const eL = cfg.engine_len_m || cfg.wing_root_chord_m * 0.6, eD = cfg.engine_dia_m || R * 0.7;
      const ys = cfg.span_m / 2 * (cfg.engine_span_frac ?? 0.34);
      const xLEat = xWingLE - Math.tan(THREE.MathUtils.degToRad(cfg.sweep_deg)) * ys;
      const eg = new THREE.CylinderGeometry(eD * 0.5, eD * 0.42, eL, 14, 1);
      eg.rotateZ(-Math.PI / 2);
      const e = new THREE.Mesh(eg, body);
      const zw = zWing - ys * Math.tan(THREE.MathUtils.degToRad(cfg.dihedral_deg)); // wing is higher there (dihedral)
      e.position.set(xLEat + eL * 0.3, side * ys, zw + eD * 0.62);
      model.add(e);
      wingPart.push(e);
      const intake = new THREE.Mesh(new THREE.CircleGeometry(eD * 0.42, 14).rotateY(Math.PI / 2), dark);
      intake.position.set(xLEat + eL * 0.8 + 0.01, side * ys, zw + eD * 0.62);
      model.add(intake);
      wingPart.push(intake);
      const pylon = new THREE.Mesh(new THREE.BoxGeometry(eL * 0.6, 0.15, eD * 0.5), wingMat);
      pylon.position.set(xLEat + eL * 0.1, side * ys, zw + eD * 0.2);
      model.add(pylon);
      wingPart.push(pylon);
    }
  }
  // horizontal tail with elevator (full span)
  const xTail = xNose - L + cfg.htail_chord_m * 1.05;
  for (const side of [1, -1]) {
    const s = halfSurface({ side, xLE: xTail, rootChord: cfg.htail_chord_m, tipChord: cfg.htail_chord_m * 0.7,
      semiSpan: cfg.htail_span_m / 2, sweepDeg: cfg.htail_sweep_deg || 5, thickness: wingT * 0.7, mat: wingMat,
      ctrlMat, ctrlFrac: 0.4, ctrlFrom: 0.08, ctrlTo: 1.0, dihedralDeg: cfg.htail_dihedral_deg || 0 });
    s.group.position.z = cfg.nozzle ? -R * 0.1 : -R * 0.3;
    model.add(s.group);
    surfaces[side > 0 ? 'elevatorR' : 'elevatorL'] = s.hinge;
    parts.htail.push(s.group);
  }
  // vertical tail with rudder: build as a "right half surface" in x-y, then rotate y -> -z (up)
  {
    const s = halfSurface({ side: 1, xLE: xTail + cfg.vtail_chord_m * 0.25, rootChord: cfg.vtail_chord_m,
      tipChord: cfg.vtail_chord_m * 0.55, semiSpan: cfg.vtail_height_m, sweepDeg: cfg.vtail_sweep_deg,
      thickness: wingT * 0.7, mat: wingMat, ctrlMat, ctrlFrac: 0.35, ctrlFrom: 0.05, ctrlTo: 0.95 });
    const fin = new THREE.Group();
    fin.add(s.group);
    fin.rotation.x = -Math.PI / 2; // local +y -> -z (up); local +z (hinge 'down') -> +y (right)
    fin.position.z = -R * 0.45;
    model.add(fin);
    surfaces.rudder = s.hinge;
    parts.vtail.push(fin);
  }
  // propeller / nozzle
  let prop = null;
  if (cfg.propeller) {
    prop = new THREE.Group();
    const blade = new THREE.Mesh(new THREE.BoxGeometry(0.06, 0.16, cfg.span_m * 0.17), dark);
    prop.add(blade);
    const spinner = new THREE.Mesh(new THREE.ConeGeometry(R * 0.3, R * 0.5, 8).rotateZ(-Math.PI / 2), body);
    spinner.position.x = 0.15;
    prop.add(spinner);
    prop.position.x = xNose + 0.05;
    model.add(prop);
  }
  if (cfg.nozzle) {
    const nn = cfg.nozzles || 1;
    for (let k = 0; k < nn; k++) {
      const off = nn === 1 ? 0 : (k === 0 ? -1 : 1) * R * 0.42;
      const rr = nn === 1 ? R * 0.4 : R * 0.36;
      const ng = new THREE.CylinderGeometry(rr * 0.85, rr, L * 0.05, 12, 1, true).rotateZ(-Math.PI / 2);
      const n = new THREE.Mesh(ng, dark);
      n.position.set(xNose - L - L * 0.02, off, 0);
      model.add(n);
      parts.fuselage.push(n);
    }
  }
  if (prop) parts.fuselage.push(prop);
  return { root, model, parts, surfaces, prop, length: L, span: cfg.span_m, planform: planform || null };
}

// Optional glTF model. cfg.gltf = url; cfg.gltf_rotation_deg = [x, y, z] Euler (XYZ) that maps the
// glTF's own axes into FRD body axes; cfg.gltf_scale = number.
export async function buildGltf(cfg) {
  const { GLTFLoader } = await import('three/addons/loaders/GLTFLoader.js');
  const gltf = await new GLTFLoader().loadAsync(cfg.gltf);
  const root = new THREE.Group();
  const m = gltf.scene;
  const r = (cfg.gltf_rotation_deg || [0, 0, 0]).map(THREE.MathUtils.degToRad);
  m.rotation.set(r[0], r[1], r[2]);
  m.scale.setScalar(cfg.gltf_scale || 1);
  root.add(m);
  return { root, surfaces: {}, prop: null, length: cfg.length_m, span: cfg.span_m };
}

const MAXDEF = THREE.MathUtils.degToRad(25);
// Animate control surfaces from normalized commands (JSBSim signs):
// elevator>0 = trailing edge down; aileron>0 = roll right (left TE down, right TE up);
// rudder>0 = trailing edge left (nose-left yaw); throttle 0..1 spins the prop.
export function applyControls(model, c, dtSec) {
  const s = model.surfaces;
  const val = (v) => (Number.isFinite(v) ? Math.max(-1, Math.min(1, v)) : 0);
  const e = val(c.elevator) * MAXDEF, a = val(c.aileron) * MAXDEF, r = val(c.rudder) * MAXDEF;
  // hinge groups were rotated about z to follow the hinge line; rotate their child about local y
  const set = (h, ang) => { if (h) h.children[0].rotation.y = ang; };
  set(s.elevatorL, e); set(s.elevatorR, e);
  set(s.aileronL, a); set(s.aileronR, -a);
  // fin local frame: +z_local -> +y_body (right); "TE down" in local = TE to the right. TE left => negative.
  set(s.rudder, -r);
  if (model.prop) model.prop.rotation.x += (8 + 60 * (Number.isFinite(c.throttle) ? c.throttle : 0.5)) * dtSec;
}

// ---------------------------------------------------------------- soft-body v2 (structure block)
// structure = {components: [{name: 'wingL'|'wingR'|'htail'|'vtail'|'fuselage', axis_nodes_body_m: [[x,y,z],...],
//   dof: ['dz','dy','dx','twist']}]}, channels '<component>.<dof>.<node_idx>' (metres / rad, body FRD).
// Every vertex of the component's procedural meshes is projected (at rest, controls neutral) onto the node polyline;
// its displacement is the linear interpolation of the node values at that station:
//   dz, dy, dx: translation along body z (down) / y (right) / x (forward; wing in-plane bending);
//   twist:  rotation about the local axis tangent (node i -> i+1 direction, right-hand rule) through the axis point.
// `exag` multiplies all deflections (twist included). `dofs` (optional list of DOF names, component names or
// '<component>.<dof>', e.g. ['dx'] or ['htail','vtail']) shows only those (URL ?dofs=dx isolates wing in-plane bending
// at a large exaggeration; ?dofs=htail,vtail,fuselage the tail). Meshes not in a component are untouched.
const _m4 = new THREE.Matrix4(), _m4b = new THREE.Matrix4(), _m3 = new THREE.Matrix3(), _v = new THREE.Vector3();

export function attachStructure(model, structure) {
  if (!model.parts || !structure || !Array.isArray(structure.components)) return null;
  model.root.updateMatrixWorld(true);
  const modelInv = _m4.copy(model.model.matrixWorld).invert().clone();
  const comps = [];
  for (const c of structure.components) {
    const objs = model.parts[c.name];
    const nodes = (c.axis_nodes || []).map((p) => new THREE.Vector3(p[0], p[1], p[2]));
    if (!objs || !objs.length || nodes.length < 2) continue;
    const segDir = [], segLen = [];
    for (let j = 0; j < nodes.length - 1; j++) {
      const d = nodes[j + 1].clone().sub(nodes[j]);
      segLen.push(d.length() || 1e-9);
      segDir.push(d.normalize());
    }
    // wings: elastic twist rotates each section about the elastic axis IN the wing plane (the procedural wing's
    // height differs from FD's node z, e.g. 737 ~3 m; a vertical lever arm would turn twist into fake fore/aft motion)
    const inPlane = c.name === 'wingR' || c.name === 'wingL';
    const meshes = [];
    const seen = new Set();
    for (const o of objs) o.traverse((m) => {
      if (!m.isMesh || seen.has(m)) return;
      seen.add(m);
      m.geometry = m.geometry.clone(); // never share a deformed buffer
      m.frustumCulled = false;
      const pos = m.geometry.attributes.position;
      const rest = Float32Array.from(pos.array);
      const rel = _m4b.multiplyMatrices(modelInv, m.matrixWorld);
      const n = pos.count, seg = new Uint16Array(n), u = new Float32Array(n), r = new Float32Array(n * 3);
      for (let k = 0; k < n; k++) {
        _v.set(rest[3 * k], rest[3 * k + 1], rest[3 * k + 2]).applyMatrix4(rel);
        // nearest point on the node polyline
        let best = Infinity, bj = 0, bu = 0;
        for (let j = 0; j < segDir.length; j++) {
          const t = THREE.MathUtils.clamp(_v.clone().sub(nodes[j]).dot(segDir[j]) / segLen[j], 0, 1);
          const p = nodes[j].clone().addScaledVector(segDir[j], t * segLen[j]);
          const dd = p.distanceToSquared(_v);
          if (dd < best - 1e-12) { best = dd; bj = j; bu = t; }
        }
        seg[k] = bj; u[k] = bu;
        const a = nodes[bj].clone().addScaledVector(segDir[bj], bu * segLen[bj]);
        r[3 * k] = _v.x - a.x; r[3 * k + 1] = _v.y - a.y; r[3 * k + 2] = inPlane ? 0 : _v.z - a.z;
      }
      meshes.push({ mesh: m, rest, seg, u, r });
    });
    comps.push({ name: c.name, dof: c.dof || [], ch: c.ch || {}, nNodes: nodes.length, segDir, meshes });
  }
  return comps.length ? { comps, modelInv } : null;
}

// value(arr) -> interpolated channel value at the current time (arr may be null)
export function applyStructure(model, deformer, value, exag = 1, dofs = null) {
  if (!deformer) return;
  model.root.updateMatrixWorld(true);
  const modelInv = _m4.copy(model.model.matrixWorld).invert();
  const q = new THREE.Quaternion(), rv = new THREE.Vector3();
  for (const c of deformer.comps) {
    const node = (dof) => {
      const shown = !dofs || dofs.includes(dof) || dofs.includes(c.name) || dofs.includes(`${c.name}.${dof}`);
      const arrs = shown ? c.ch[dof] : null;
      const out = new Float32Array(c.nNodes);
      if (arrs) for (let i = 0; i < c.nNodes; i++) { const v = arrs[i] ? value(arrs[i]) : 0; out[i] = Number.isFinite(v) ? v * exag : 0; }
      return out;
    };
    const dz = node('dz'), dy = node('dy'), dx = node('dx'), tw = node('twist');
    for (const mm of c.meshes) {
      const m = mm.mesh, pos = m.geometry.attributes.position, a = pos.array;
      _m3.setFromMatrix4(_m4b.multiplyMatrices(modelInv, m.matrixWorld)).invert(); // body -> mesh-local (linear part)
      for (let k = 0; k < pos.count; k++) {
        const j = mm.seg[k], t = mm.u[k];
        const lz = dz[j] + (dz[j + 1] - dz[j]) * t, ly = dy[j] + (dy[j + 1] - dy[j]) * t, th = tw[j] + (tw[j + 1] - tw[j]) * t;
        const lx = dx[j] + (dx[j + 1] - dx[j]) * t;
        _v.set(lx, ly, lz);
        if (th) {
          rv.set(mm.r[3 * k], mm.r[3 * k + 1], mm.r[3 * k + 2]);
          const r0 = rv.clone();
          rv.applyQuaternion(q.setFromAxisAngle(c.segDir[j], th));
          _v.add(rv.sub(r0));
        }
        _v.applyMatrix3(_m3);
        a[3 * k] = mm.rest[3 * k] + _v.x; a[3 * k + 1] = mm.rest[3 * k + 1] + _v.y; a[3 * k + 2] = mm.rest[3 * k + 2] + _v.z;
      }
      pos.needsUpdate = true;
    }
  }
}
