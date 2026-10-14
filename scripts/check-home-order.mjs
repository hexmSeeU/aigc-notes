import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';

export const expectedHomeSlugs = [
  'ddim-1', 'ddim-2', 'ddim-3', 'ddim-4', 'ddim-5',
  'classifier-free-guidance', 'classifier-guidance',
  'ddpm-1', 'ddpm-2', 'ddpm-3', 'ddpm-4', 'ae-to-vae',
  'elbo-1', 'elbo-2', 'elbo-3',
];

export function assertHomeOrder(slugs) {
  assert.deepEqual(slugs, expectedHomeSlugs,
    'All home pages must preserve newest days, ascending same-day episodes, and every release exactly once');
}

export function checkHomeOrder(dir) {
  const files = [path.join(dir, 'index.html')];
  const pagination = path.join(dir, 'page');
  if (fs.existsSync(pagination)) {
    for (const number of fs.readdirSync(pagination).filter(n => /^\d+$/.test(n) && Number(n) > 1).sort((a, b) => Number(a) - Number(b))) {
      files.push(path.join(pagination, number, 'index.html'));
    }
  }
  const slugs = files.flatMap(file => {
    const html = fs.readFileSync(file, 'utf8');
    return [...html.matchAll(/<a\b[^>]*\bclass=(?:"entry-link"|'entry-link'|entry-link(?=[\s>]))[^>]*\bhref=(?:"([^"]+)"|'([^']+)'|([^\s>]+))/g)]
      .map(match => new URL(match[1] ?? match[2] ?? match[3], 'https://example.org').pathname.split('/').filter(Boolean).at(-1));
  });
  assertHomeOrder(slugs);
}

