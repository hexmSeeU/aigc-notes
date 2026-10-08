import {test} from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';

test('main menu starts with an explicit home link and retains existing sections', () => {
  const config = fs.readFileSync('hugo.toml', 'utf8');
  const entries = [...config.matchAll(/\[\[menu\.main\]\]([^[]*)/g)].map(([, block]) => ({
    name: block.match(/name\s*=\s*'([^']*)'/)?.[1],
    url: block.match(/url\s*=\s*'([^']*)'/)?.[1],
    weight: Number(block.match(/weight\s*=\s*(-?\d+)/)?.[1]),
  })).sort((a, b) => a.weight - b.weight);
  assert.deepEqual(entries.map(({name, url}) => ({name, url})), [
    {name: '首页', url: '/'},
    {name: '文章', url: '/posts/'},
    {name: '基础', url: '/categories/基础/'},
    {name: '前沿', url: '/categories/前沿/'},
    {name: '归档', url: '/archives/'},
  ]);
});
