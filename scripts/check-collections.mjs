import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import {checkHomeOrder} from './check-home-order.mjs';
const dir=process.argv[2]||'public';
checkHomeOrder(dir);
console.log('Paginated home order checks passed');
const readingBaseline=JSON.parse(fs.readFileSync('tests/fixtures/continuous-reading-baseline.json','utf8'));
const manifest=JSON.parse(fs.readFileSync('tests/fixtures/collection-manifest.json','utf8'));
for(const item of manifest){
 const slug=item.target.split('/')[2];
 const html=fs.readFileSync(path.join(dir,'posts',slug,'index.html'),'utf8');
 assert.equal((html.match(/<details\b[^>]*\bclass=(?:"proof"|'proof'|proof(?=[\s>]))/g)||[]).length,item.proofs,slug+' proof count');
 const ids=[...html.matchAll(/<h[23]\b[^>]*\bid=(?:"([^"]+)"|'([^']+)'|([^\s>]+))/g)].map(m=>m[1]??m[2]??m[3]);
 for(const id of readingBaseline.find(x=>x.slug===slug).heading_ids)assert.ok(ids.includes(id),slug+' preserved heading '+id);
 assert.equal(new Set(ids).size,ids.length,slug+' unique heading IDs');
 assert.ok(!html.includes('raw HTML omitted'),slug+' omitted content');
 assert.equal((html.match(/aria-current=(?:"page"|'page'|page(?=[\s>]))/g)||[]).length,1,slug+' current episode');
 assert.ok(html.includes('/collections/'+slug.split('-')[0]+'/'),slug+' collection');
}
for(const id of ['ddpm','elbo','ddim'])assert.ok(fs.existsSync(path.join(dir,'collections',id,'index.html')));
const rss=fs.readFileSync(path.join(dir,'index.xml'),'utf8');
for(const item of manifest)assert.ok(rss.includes('/posts/'+item.target.split('/')[2]+'/'));
assert.ok(!rss.includes('/posts/ddpm-derivation/'));
assert.ok(!rss.includes('/collections/'));
console.log('Collection output checks passed');
const archive=fs.readFileSync(path.join(dir,'archives','index.html'),'utf8');
const expectedTitles=['从 AE 到 VAE：让潜变量成为一个分布','ELBO：从似然下界到训练损失','理解 DDPM：原理与推导','DDIM：从边缘分布到跳步采样','扩散模型引导：CG 与 CFG','Score：噪声预测网络到底学到了什么'];
assert.equal((archive.match(/<article\b[^>]*\bclass=(?:"archive-entry"|'archive-entry'|archive-entry(?=[\s>]))/g)||[]).length,6,'Six uniformly styled archive rows');
let prior=-1;
for(const title of expectedTitles){const position=archive.indexOf(title);assert.ok(position>prior,'Archive chronology: '+title);prior=position;}
assert.ok(!archive.includes('学习合集')&&!archive.includes('独立笔记'),'Archive has no type-separated sections');
assert.ok(!archive.includes('/posts/ddpm-derivation/'),'Legacy article is not a duplicate archive entry');
assert.ok(!archive.includes('/posts/ddim-derivation/'),'Legacy DDIM is not a duplicate archive entry');
assert.ok(archive.includes('/collections/ddim/'),'DDIM archive opens its collection');
console.log('Unified archive output checks passed');
const guidance=fs.readFileSync(path.join(dir,'collections/guidance/index.html'),'utf8');
const cg=fs.readFileSync(path.join(dir,'posts/classifier-guidance/index.html'),'utf8');
assert.ok(guidance.includes('共 2 篇'),'Guidance contains two published episodes');
assert.ok(guidance.includes('/posts/classifier-guidance/'),'CG episode has a real article link');
assert.ok(guidance.includes('/collections/ddim/'),'Guidance links its DDIM prerequisite');
assert.ok(cg.includes('/collections/guidance/'),'CG links back to its collection');
assert.equal((cg.match(/aria-current=(?:"page"|'page'|page(?=[\s>]))/g)||[]).length,1,'CG marks one current episode');
assert.ok(!cg.includes('raw HTML omitted'),'CG does not lose math or prose to HTML parsing');
assert.ok(rss.includes('/posts/classifier-guidance/'),'CG appears in RSS');
const cfg=fs.readFileSync(path.join(dir,'posts/classifier-free-guidance/index.html'),'utf8');
assert.ok(guidance.includes('/posts/classifier-free-guidance/'),'CFG episode has a real article link');
assert.ok(guidance.indexOf('/posts/classifier-guidance/')<guidance.indexOf('/posts/classifier-free-guidance/'),'Guidance episodes retain learning order');
assert.ok(!guidance.includes('后续补充'),'Guidance description reflects both published episodes');
assert.ok(cfg.includes('/collections/guidance/'),'CFG links back to its collection');
assert.equal((cfg.match(/aria-current=(?:"page"|'page'|page(?=[\s>]))/g)||[]).length,1,'CFG marks one current episode');
assert.ok(!cfg.includes('raw HTML omitted'),'CFG does not lose math or prose to HTML parsing');
assert.ok(rss.includes('/posts/classifier-free-guidance/'),'CFG appears in RSS');
assert.ok(cg.includes('/posts/classifier-free-guidance/'),'CG links to its next episode');
assert.ok(cfg.includes('/posts/classifier-guidance/'),'CFG links to its previous episode');
console.log('Guidance output checks passed');


// Aliases must stay redirects, never duplicate articles or index entries.
for(const id of ['ddim','ddpm']) {
 const alias=fs.readFileSync(path.join(dir,'posts',`${id}-derivation`,'index.html'),'utf8');
 assert.match(alias,/http-equiv=["']?refresh/);
 assert.match(alias,new RegExp(`url=[^>]*?/collections/${id}/`));
 assert.match(alias,/noindex/);
 assert.ok(!alias.includes('post-content')&&!alias.includes('MathJax'));
 assert.ok(alias.length<2000,'Alias contains only redirect and fallback');
 const walk=d=>fs.readdirSync(d,{withFileTypes:true}).flatMap(e=>e.isDirectory()?walk(path.join(d,e.name)):[path.join(d,e.name)]);
 for(const file of walk(dir).filter(f=>/\.(html|xml|json)$/.test(f))){
  if(file===path.join(dir,'posts',`${id}-derivation`,'index.html'))continue;
  assert.ok(!fs.readFileSync(file,'utf8').includes(`/posts/${id}-derivation/`),`Legacy link/index entry in ${file}`);
 }
}
assert.ok(!fs.existsSync(path.join(dir,'archive','legacy')),'Archival sources are never published');
console.log('Legacy redirect and index exclusion checks passed');
