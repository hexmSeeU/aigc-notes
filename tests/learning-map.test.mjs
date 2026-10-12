import {test} from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';

test('homepage learning map has five real destinations in reading order', () => {
  const file = 'layouts/_partials/learning_map.html';
  assert.ok(fs.existsSync(file), 'the learning map partial is missing');
  const template = fs.readFileSync(file, 'utf8');
  const destinations = [...template.matchAll(/href="\{\{ "([^"]+)" \| relURL \}\}"/g)].map(match => match[1]);
  assert.deepEqual(destinations, [
    'posts/ae-to-vae/', 'collections/elbo/', 'collections/ddpm/',
    'posts/ddim-derivation/', 'collections/guidance/',
  ]);
  for (const destination of destinations) assert.ok(fs.existsSync(`content/${destination}index.md`));
  assert.match(template, /aria-labelledby="learning-map-title"/);
  assert.match(template, /<ol\b/);
  assert.doesNotMatch(template, /<script|home-collections/);
});

test('handwriting is a small self-hosted Latin font with its license', () => {
  const font = 'static/fonts/caveat-latin-600.woff';
  assert.ok(fs.existsSync(font), 'the local handwriting font is missing');
  assert.ok(fs.statSync(font).size < 30000, 'keep the Latin accent font lightweight');
  assert.match(fs.readFileSync('static/fonts/OFL-Caveat.txt', 'utf8'), /SIL OPEN FONT LICENSE/);
  const css = fs.readFileSync('assets/css/extended/learning-map.css', 'utf8');
  assert.match(css, /font-display:\s*swap/);
  assert.match(css, /unicode-range:\s*U\+0020-007E/i);
  assert.doesNotMatch(css, /url\(["']?https?:\/\/|@import|animation:/);
});
