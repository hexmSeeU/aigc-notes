import fs from 'node:fs';
// Only these reviewed navigation changes are reversed for historical body snapshots.
export const historicalLinks = source => source
  .replaceAll('../../collections/ddim/', '../ddim-derivation/')
  .replaceAll('"collections/ddpm"', '"posts/ddpm-derivation"')
  .replaceAll('[DDPM 学习合集]', '[DDPM：从似然目标到噪声预测的完整推导]');
export const preservedSource = slug => fs.readFileSync(slug.endsWith('-derivation')
  ? `archive/legacy/${slug}.md` : `content/posts/${slug}/index.md`, 'utf8');
