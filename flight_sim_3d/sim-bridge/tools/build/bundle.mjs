// Bundles viewer/js/main.js + vendored three.js into one ES module (viewer/dist/fv.bundle.js)
// used by the single-file/inline (Colab) build. The normal viewer does NOT need this.
//   cd tools/build && npm install && node bundle.mjs
import * as esbuild from 'esbuild';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const here = path.dirname(fileURLToPath(import.meta.url));
const viewer = path.resolve(here, '../../viewer');
const map = (p) => {
  if (p === 'three') return path.join(viewer, 'vendor/three.module.min.js');
  if (p.startsWith('three/addons/')) return path.join(viewer, 'vendor/addons', p.slice('three/addons/'.length));
  if (p.startsWith('fv/')) return path.join(viewer, 'js', p.slice(3));
  return null;
};
await esbuild.build({
  entryPoints: [path.join(viewer, 'js/main.js')], bundle: true, format: 'esm', minify: true,
  target: 'es2020', outfile: path.join(viewer, 'dist/fv.bundle.js'), legalComments: 'eof',
  plugins: [{ name: 'importmap', setup(b) { b.onResolve({ filter: /^(three|three\/addons\/.*|fv\/.*)$/ }, (a) => ({ path: map(a.path) })); } }],
});
console.log('wrote', path.join(viewer, 'dist/fv.bundle.js'));
