import {test} from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {assertHomeOrder, checkHomeOrder, expectedHomeSlugs} from '../scripts/check-home-order.mjs';

test('home order regression rejects the original DDPM and ELBO ordering', () => {
  assert.throws(() => assertHomeOrder([
    'classifier-free-guidance', 'classifier-guidance', 'ddim-derivation',
    'ddpm-2', 'ddpm-3', 'ddpm-4', 'ddpm-1', 'ae-to-vae', 'elbo-2', 'elbo-3', 'elbo-1',
  ]));
});

test('home order regression rejects missing, duplicate and old-day-first releases', () => {
  assert.throws(() => assertHomeOrder(expectedHomeSlugs.slice(0, 10)));
  assert.throws(() => assertHomeOrder([...expectedHomeSlugs, 'elbo-3']));
  assert.throws(() => assertHomeOrder([...expectedHomeSlugs].reverse()));
});

for (const base of ['https://example.org/', 'https://example.org/aigc-notes/']) {
  test(`home output checker reads every numeric page and supports minified links at ${base}`, t => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'home-order-'));
    t.after(() => fs.rmSync(dir, {recursive: true, force: true}));
    // One release per page exercises numeric page order beyond page 9.
    expectedHomeSlugs.forEach((slug, index) => {
      const folder = index === 0 ? dir : path.join(dir, 'page', String(index + 1));
      fs.mkdirSync(folder, {recursive: true});
      fs.writeFileSync(path.join(folder, 'index.html'), index % 2
        ? `<a class=entry-link aria-label="post link" href=${base}posts/${slug}/></a>`
        : `<a class="entry-link" href="${base}posts/${slug}/"></a>`);
    });
    fs.mkdirSync(path.join(dir, 'page', '1'), {recursive: true});
    fs.writeFileSync(path.join(dir, 'page', '1', 'index.html'), 'Redirect to home');
    assert.doesNotThrow(() => checkHomeOrder(dir));
    fs.writeFileSync(path.join(dir, 'page', '11', 'index.html'), 'Missing final release');
    assert.throws(() => checkHomeOrder(dir));
  });
}
