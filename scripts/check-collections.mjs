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
