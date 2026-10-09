import {test} from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createHash} from 'node:crypto';
import {validateArticle} from '../scripts/check-content.mjs';

const articlePath = 'content/posts/ddim-derivation/index.md';
const readArticle = () => {
  assert.ok(fs.existsSync(articlePath), 'The approved DDIM article must be published');
  const text = fs.readFileSync(articlePath, 'utf8');
  const end = text.indexOf('\n}\n') + 2;
  return {metadata: JSON.parse(text.slice(0, end)), body: text.slice(end).trim()};
};

test('DDIM is a dated math-enabled third article', () => {
  const {metadata} = readArticle();
  assert.deepEqual(validateArticle(metadata), []);
  assert.equal(metadata.title, 'DDIM：从边缘分布到跳步采样');
  assert.equal(metadata.slug, 'ddim-derivation');
  assert.equal(metadata.date, '2026-10-09T18:17:00+08:00');
  assert.equal(metadata.draft, false);
  assert.equal(metadata.math, true);
  assert.deepEqual(metadata.categories, ['基础']);
  assert.deepEqual(metadata.tags, ['DDIM', '扩散模型']);
  assert.ok(metadata.description && metadata.summary);
});

test('DDIM preserves every approved formula and all 26 equation numbers', () => {
  const {body} = readArticle();
  const blocks = [...body.matchAll(/\$\$([\s\S]*?)\$\$/g)].map(match => match[1]);
  const prose = body.replace(/\$\$[\s\S]*?\$\$/g, '');
  const inline = [...prose.matchAll(/\\\(([\s\S]*?)\\\)/g)].map(match => match[1]);
  assert.equal(blocks.length, 28);
  assert.equal(inline.length, 134);
  assert.doesNotMatch(prose, /\$/);
  // Preserve every approved expression in document order. The long time schedule
  // becomes a display equation on the web, and one literal '<' needs HTML escaping.
  const math = [...body.matchAll(/\$\$([\s\S]*?)\$\$|\\\(([\s\S]*?)\\\)/g)]
    .map(match => (match[1] ?? match[2]).trim().replaceAll('&lt;', '<'));
  const hash = createHash('sha256').update(math.join('\n')).digest('hex');
  assert.equal(hash, 'fc5012de200a94e6364c294d739bbbed415aa699a3b3b3021d83d6575bb9aded');
  assert.deepEqual([...body.matchAll(/\\qquad \((\d+)\)/g)].map(match => Number(match[1])),
    Array.from({length: 26}, (_, index) => index + 1));
});

test('DDIM keeps the concise eight-section version and its approved sampling code', () => {
  const {body} = readArticle();
  const prose = body.replace(/```[\s\S]*?```/g, '');
  assert.equal([...prose.matchAll(/^## \d+ /gm)].length, 8);
  assert.equal([...prose.matchAll(/^#{2,3} /gm)].length, 24);
  assert.doesNotMatch(prose, /^# |PAGEBREAK|下一页|上一页|TODO|TBD|待补充|占位符/gm);
  assert.doesNotMatch(prose, /^## (?:9|10) |马尔可夫/gm);
  for (const heading of [
    '1 为什么可以沿用 DDPM 的训练', '3 为什么每个时刻的分布都能保留',
    '6 eta 等于 1 时怎样对应 DDPM', '7 怎样跳过中间时间步',
    '8 确定性采样的简短伪代码',
  ]) assert.ok(body.includes(`## ${heading}\n`), heading);
  const code = body.match(/```python\n([\s\S]*?)\n```/)?.[1];
  assert.ok(code);
  assert.equal(createHash('sha256').update(code).digest('hex'),
    '93239f2c7d7e8e0060716df11eaf2556fea312f30cfdebfecacd18ab625ed3a1');
  assert.match(body, /\[DDPM 笔记\]\(https:\/\/hexmseeu.github.io\/aigc-notes\/posts\/ddpm-derivation\/\)/);
});
