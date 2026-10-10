import {test} from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createHash} from 'node:crypto';
import {withoutReferenceAnswers} from './helpers/reference-answers.mjs';
const baseline=JSON.parse(fs.readFileSync('tests/fixtures/continuous-reading-baseline.json','utf8'));
test('all seven articles are continuous reading without optional proof disclosures',()=>{
 assert.equal(baseline.length,7);
 for(const {slug} of baseline){
  const source=fs.readFileSync(`content/posts/${slug}/index.md`,'utf8').replace(/\s*\{#[^}]+\}/g,'');
  assert.doesNotMatch(source,/proof-(?:open|close)|<details\b|<summary\b|选读|按需|可选验算|第一次可以不展开|正文到这里结束|只想先看使用流程|方便需要时逐项查阅/,slug);
 }
 const collection=fs.readFileSync('content/collections/ddpm/index.md','utf8');
 assert.doesNotMatch(collection,/选读|按需|跳过/);
});
test('continuous articles preserve the exact original formula sequence and reference targets',()=>{
 for(const item of baseline){
  const source=fs.readFileSync(`content/posts/${item.slug}/index.md`,'utf8');
  const body=withoutReferenceAnswers(source.slice(source.indexOf('\n}\n')+2).trim());
  const math=[...body.matchAll(/\$\$([\s\S]*?)\$\$|\\\(([\s\S]*?)\\\)/g)].map(m=>m[1]??m[2]);
  assert.equal(math.length,item.math_count,item.slug);
  assert.equal(createHash('sha256').update(JSON.stringify(math)).digest('hex'),item.math_sha256,item.slug);
  assert.deepEqual([...body.matchAll(/\]\(([^\n]+?)\)/g)].map(m=>m[1]),item.links,item.slug);
 }
});

test('homepage no longer has the learning-collection shortcut row',()=>{
 assert.doesNotMatch(fs.readFileSync('layouts/_partials/home_info.html','utf8'),/home-collections|学习合集/);
});
