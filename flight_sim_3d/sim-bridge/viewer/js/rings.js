// Phase 4 ring course rendering (spec flight_sim_3d/PHASE4_RINGS_SPEC.md section 10).
// Pure state logic mirrors sim_bridge/rings.py ring_state_at (tested in tests/test_rings_viewer.py via tools/build/ringcheck.mjs).
import * as THREE from 'three';

export const RING_WINDOW = 5;
export const RING_COLORS = { target: 0xffd23f, active: 0xd9a400, passed: 0x3fdc5a, missed: 0xff3b3b, rim: 0xff8c1a, hidden: 0x888888 };

export function courseOf(tr) {
  const c = tr && tr.meta && tr.meta.course;
  return c && c.schema === 'ga-flightsim-course/1' && Array.isArray(c.rings) ? c : null;
}

export function ringStateAt(gates, n, t, win = RING_WINDOW) {   // n = M (scored rings)
  const st = new Array(n).fill('hidden');
  const done = (gates || []).filter((g) => g.t <= t);
  for (const g of done) st[g.k] = g.result === 'pass' ? 'passed' : g.result === 'rim' ? 'rim' : 'missed';   // miss | missed_order | missed_time
  const nxt = done.length;
  const crashed = done.some((g) => g.result === 'rim');
  if (!crashed) for (let k = nxt; k < Math.min(n, nxt + win); k++) st[k] = k === nxt ? 'target' : 'active';
  return st;
}

export function ringsHudText(course, gates, t) {
  const n = course.M ?? course.rings.length;   // scored rings (run-out rings are not counted)
  const done = (gates || []).filter((g) => g.t <= t);
  const cnt = (r) => done.filter((g) => (Array.isArray(r) ? r.includes(g.result) : g.result === r)).length;
  const lastMiss = [...done].reverse().find((g) => g.result !== 'pass');
  const acc = done.filter((g) => g.result === 'pass').map((g) => 1 - (g.rho_norm ?? 0));
  return `RINGS ${done.length}/${n} · pass ${cnt('pass')} miss ${cnt(['miss', 'missed_order', 'missed_time'])} rim ${cnt('rim')}` +
    (acc.length ? ` · acc ${(acc.reduce((a, b) => a + b, 0) / acc.length).toFixed(2)}` : '') +
    (lastMiss ? ` · last ${lastMiss.result} k${lastMiss.k}${lastMiss.miss_m != null ? ' ' + lastMiss.miss_m.toFixed(1) + ' m' : ''}` : '') +
    `\ncourse ${course.model} ${course.stage} seed ${course.seed} · M ${n}${course.synthetic ? ' · SYNTHETIC' : ''} · T_nom ${Number(course.nominal_time_s).toFixed(1)} s · limit ${Number(course.time_limit_s).toFixed(1)} s`;
}

// toScene(e, n, u) -> THREE.Vector3 in scene coords (same transform as the trail).
export function buildRings(course, toScene, showAll = false) {
  const group = new THREE.Group();
  const meshes = course.rings.map((r) => {
    const R = r.radius_m, tube = Math.max(r.tube_m, R * 0.05);
    const m = new THREE.Mesh(new THREE.TorusGeometry(R, tube, 10, 48),
      new THREE.MeshBasicMaterial({ color: RING_COLORS.active, transparent: true, opacity: 0.9 }));
    const ce = r.centre_enu_m || r.center_m, ne = r.normal_enu || r.normal;   // course_block() ENU copy (course frame is NED)
    const c = toScene(ce[0], ce[1], ce[2]);
    const tip = toScene(ce[0] + ne[0], ce[1] + ne[1], ce[2] + ne[2]);
    m.position.copy(c);
    m.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), tip.sub(c).normalize());
    group.add(m);
    return m;
  });
  return { group, meshes, showAll };
}

export function updateRings(rs, course, gates, t) {
  const st = ringStateAt(gates, course.M ?? course.rings.length, t);
  rs.meshes.forEach((m, k) => {
    const s = st[k];
    m.visible = s !== 'hidden' || rs.showAll;
    m.material.color.setHex(RING_COLORS[s]);
    m.material.opacity = s === 'hidden' ? 0.15 : s === 'active' ? 0.55 : 0.95;
    m.scale.setScalar(s === 'target' ? 1.0 : 0.97);
  });
  return st;
}
