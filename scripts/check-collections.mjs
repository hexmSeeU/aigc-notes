import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
const dir=process.argv[2]||'public';
const manifest=JSON.parse(fs.readFileSync('tests/fixtures/collection-manifest.json','utf8'));
for(const item of manifest){
 const slug=item.target.split('/')[2];
 const html=fs.readFileSync(path.join(dir,'posts',slug,'index.html'),'utf8');
 assert.equal((html.match(/<details\b[^>]*\bclass=(?:"proof"|'proof'|proof(?=[\s>]))/g)||[]).length,item.proofs,slug+' proof count');
 assert.ok(!html.includes('raw HTML omitted'),slug+' omitted content');
 assert.equal((html.match(/aria-current=(?:"page"|'page'|page(?=[\s>]))/g)||[]).length,1,slug+' current episode');
 assert.ok(html.includes('/collections/'+slug.split('-')[0]+'/'),slug+' collection');
}
for(const id of ['ddpm','elbo'])assert.ok(fs.existsSync(path.join(dir,'collections',id,'index.html')));
const rss=fs.readFileSync(path.join(dir,'index.xml'),'utf8');
for(const item of manifest)assert.ok(rss.includes('/posts/'+item.target.split('/')[2]+'/'));
assert.ok(!rss.includes('/posts/ddpm-derivation/'));
assert.ok(!rss.includes('/collections/'));
console.log('Collection output checks passed');
const archive=fs.readFileSync(path.join(dir,'archives','index.html'),'utf8');
const expectedTitles=['从 AE 到 VAE：让潜变量成为一个分布','ELBO：从似然下界到训练损失','理解 DDPM：原理与推导','DDIM：从边缘分布到跳步采样'];
assert.equal((archive.match(/<article\b[^>]*\bclass=(?:"archive-entry"|'archive-entry'|archive-entry(?=[\s>]))/g)||[]).length,4,'Four uniformly styled archive rows');
let prior=-1;
for(const title of expectedTitles){const position=archive.indexOf(title);assert.ok(position>prior,'Archive chronology: '+title);prior=position;}
assert.ok(!archive.includes('学习合集')&&!archive.includes('独立笔记'),'Archive has no type-separated sections');
assert.ok(!archive.includes('/posts/ddpm-derivation/'),'Legacy article is not a duplicate archive entry');
console.log('Unified archive output checks passed');
