// Node check of the viewer's planform loader (traj.parsePlanform + aircraft.planformStations) on trajectory files:
//   node tools/build/jscheck.mjs <traj.json> [...]     -> one JSON line per file (used by tests/test_planform.py)
import * as esbuild from 'esbuild';
import path from 'node:path';
import fs from 'node:fs';
import os from 'node:os';
import { fileURLToPath, pathToFileURL } from 'node:url';
const here = path.dirname(fileURLToPath(import.meta.url));
const viewer = path.resolve(here, '../../viewer');
const map = (p) => {
  if (p === 'three') return path.join(viewer, 'vendor/three.module.min.js');
  if (p.startsWith('three/addons/')) return path.join(viewer, 'vendor/addons', p.slice('three/addons/'.length));
  if (p.startsWith('fv/')) return path.join(viewer, 'js', p.slice(3));
  return null;
};
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'fvjs-'));
const entry = path.join(tmp, 'entry.mjs');
fs.writeFileSync(entry, `export { parsePlanform, b2Info, b2HudText, sectionShape, sectionCamberPos, sectionSvg, B2A_NOTE, cgDisplayOffset, centreOnFuselage, setCgCentre } from ${JSON.stringify(path.join(viewer, 'js/traj.js'))};
export { planformStations, resolvePreset } from ${JSON.stringify(path.join(viewer, 'js/aircraft.js'))};`);
const out = path.join(tmp, 'bundle.mjs');
await esbuild.build({ entryPoints: [entry], bundle: true, format: 'esm', outfile: out, logLevel: 'error',
  plugins: [{ name: 'importmap', setup(b) { b.onResolve({ filter: /^(three|three\/addons\/.*|fv\/.*)$/ }, (a) => ({ path: map(a.path) })); } }] });
const m = await import(pathToFileURL(out).href);
for (const f of process.argv.slice(2)) {
  const doc = JSON.parse(fs.readFileSync(f, 'utf8'));
  const pf = m.parsePlanform(doc);
  let st = null;
  if (pf) {
    const cfg = m.resolvePreset(doc.aircraft);
    const S = cfg.span_m / 2, p = m.planformStations(pf.wingR, S, pf.sweep_qc_rad);
    const w = pf.wingR, n = w.n;
    st = { semiSpan: S, f0: p.f[0], fEnd: p.f[p.f.length - 1], nStations: p.f.length,
      chordRoot: p.chordAt(0), chordTip: p.chordAt(1), dleTip: p.dleAt(1), twistTip: p.twistAt(1),
      chordAtFirstStrip: p.chordAt(Math.abs(w.y_m ? w.y_m[0] : w.span_frac[0] * S) / S), firstStripChord: w.chord_m[0],
      chordAtLastStrip: p.chordAt(Math.abs(w.y_m ? w.y_m[n - 1] : w.span_frac[n - 1] * S) / S), lastStripChord: w.chord_m[n - 1],
      absolute: !!p.absolute, le0: p.le0 ?? null,
      leAtFirst: p.absolute ? p.le0 + p.dleAt(Math.abs(w.y_m[0]) / S) : null, firstLe: w.le_x_m ? w.le_x_m[0] : null,
      leAtMid: p.absolute ? p.le0 + p.dleAt(Math.abs(w.y_m[n >> 1]) / S) : null, midLe: w.le_x_m ? w.le_x_m[n >> 1] : null,
      pivotAtMid: p.pivotAt ? p.pivotAt(Math.abs(w.y_m[n >> 1]) / S) : null, midEa: w.ea_x_m ? w.ea_x_m[n >> 1] : null,
      twistAtMid: p.twistAt(Math.abs((w.y_m ? w.y_m[n >> 1] : w.span_frac[n >> 1] * S)) / S), midTwist: w.twist_rad ? w.twist_rad[n >> 1] : null,
      hasZ: !!p.zAt, zRelNodes: w.z_rel_m || null,
      zAtNodes: p.zAt && w.y_m ? w.y_m.map((yy) => p.zAt(Math.min(1, Math.abs(yy) / S))) : null };
  }
  let b2 = null;
  if (pf && pf.b2) {
    const B = pf.b2, k = B.tc ? B.tc.length - 1 : 0, cp = m.sectionCamberPos(B.section_baseline);
    const sec = (i) => { const sh = m.sectionShape(B.tc[i], B.camber_pct ? B.camber_pct[i] : 0, cp); return { tmax: sh.tmax, cmax: sh.cmax, tc: B.tc[i], camber: B.camber_pct ? B.camber_pct[i] : null }; };
    b2 = { stage: B.stage, isDefault: B.isDefault, dihedral_delta_deg: B.dihedral_delta_deg, dihedral_baseline_deg: B.dihedral_baseline_deg,
      section_baseline: B.section_baseline, area_scale: B.area_scale, aspect_scale: B.aspect_scale, camberPos: cp,
      root: B.tc ? sec(0) : null, tip: B.tc ? sec(k) : null, hud: m.b2HudText(B),
      svg: m.sectionSvg(B, pf.wingR.chord_m[0], pf.wingR.chord_m[pf.wingR.n - 1]).length, note: m.B2A_NOTE };
  }
  const raw = m.b2Info(doc);
  // display-only CG centring (default on): offset, wing root y both sides after centring, data untouched
  const dy = m.cgDisplayOffset(doc);
  const cdoc = m.centreOnFuselage(doc, dy);
  const rootY = (o, nm) => { const c = ((o.structure || {}).components || []).find((x) => x.name === nm); return c && c.axis_nodes_body_m ? c.axis_nodes_body_m[0][1] : null; };
  const cg = { dy, rootR: rootY(cdoc, 'wingR'), rootL: rootY(cdoc, 'wingL'), rawRootR: rootY(doc, 'wingR'), rawRootL: rootY(doc, 'wingL'),
    sameData: cdoc.data === doc.data, rawUnchanged: rootY(doc, 'wingR') === (((doc.structure || {}).components || []).find((x) => x.name === 'wingR') || {axis_nodes_body_m: [[0, null]]}).axis_nodes_body_m[0][1] };
  console.log(JSON.stringify({ file: path.basename(f), planform: pf && { field: pf.field, schema: pf.schema, source: pf.source,
    synthetic: pf.synthetic, symmetric: pf.symmetric, taper: pf.taper, sweep_qc_rad: pf.sweep_qc_rad, twist_tip_rad: pf.twist_tip_rad,
    n: pf.wingR.n, sameLR: pf.wingL === pf.wingR, geom: pf.geom, node_layout: pf.node_layout }, stations: st, b2,
    b2raw: raw && { isDefault: raw.isDefault, stage: raw.stage }, cg }));
}
fs.rmSync(tmp, { recursive: true, force: true });
