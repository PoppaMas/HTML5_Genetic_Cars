// Node check of viewer/js/rings.js: node tools/build/ringcheck.mjs <traj.json> t1,t2,... -> JSON {t: states[]} + hud
import * as esbuild from 'esbuild';
import path from 'node:path';
import fs from 'node:fs';
import os from 'node:os';
import { fileURLToPath, pathToFileURL } from 'node:url';
const here = path.dirname(fileURLToPath(import.meta.url));
const viewer = path.resolve(here, '../../viewer');
const map = (p) => (p === 'three' ? path.join(viewer, 'vendor/three.module.min.js') : path.join(viewer, 'js', p.slice(3)));
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'fvring-'));
const entry = path.join(tmp, 'entry.mjs');
fs.writeFileSync(entry, `export { ringStateAt, ringsHudText, courseOf } from ${JSON.stringify(path.join(viewer, 'js/rings.js'))};`);
const out = path.join(tmp, 'bundle.mjs');
await esbuild.build({ entryPoints: [entry], bundle: true, format: 'esm', outfile: out, logLevel: 'error',
  plugins: [{ name: 'importmap', setup(b) { b.onResolve({ filter: /^(three|fv\/.*)$/ }, (a) => ({ path: map(a.path) })); } }] });
const m = await import(pathToFileURL(out).href);
const doc = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const course = m.courseOf({ meta: doc });
const res = {};
for (const t of process.argv[3].split(',').map(Number)) res[t] = { st: m.ringStateAt(doc.gates, course.rings.length, t), hud: m.ringsHudText(course, doc.gates, t) };
console.log(JSON.stringify(res));
