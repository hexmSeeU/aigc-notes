import {test} from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createHash} from 'node:crypto';
import {withoutReferenceAnswers} from './helpers/reference-answers.mjs';
const read=path=>{const s=fs.readFileSync(path,'utf8'),e=s.indexOf('\n}\n')+2;return {meta:JSON.parse(s.slice(0,e)),body:s.slice(e).trim()};};
test('collection and episode dates use the requested October learning dates',()=>{
 for(const [id,n,date] of [['ddpm',4,'2026-10-09'],['elbo',3,'2026-10-08']]){
  assert.equal(read(`content/collections/${id}/index.md`).meta.date?.slice(0,10),date);
  for(let i=1;i<=n;i++)assert.equal(read(`content/posts/${id}-${i}/index.md`).meta.date?.slice(0,10),date);
 }
});
test('DDPM collection has the requested revised title',()=>{
 assert.equal(read('content/collections/ddpm/index.md').meta.title,'理解 DDPM：原理与推导');
});
test('same-day archive ties follow AE, ELBO, DDPM, DDIM without changing original dates',()=>{
 const paths=['posts/ae-to-vae','collections/elbo','collections/ddpm','posts/ddim-derivation'];
 assert.deepEqual(paths.map(p=>read(`content/${p}/index.md`).meta.archiveOrder),[1,2,3,4]);
 assert.equal(read('content/posts/ae-to-vae/index.md').meta.date,'2026-10-08T16:00:00+08:00');
 assert.equal(read('content/posts/ddim-derivation/index.md').meta.date,'2026-10-09T18:17:00+08:00');
});
test('published article bodies match the reviewed presentation-only revision',()=>{
 const hashes=JSON.parse(fs.readFileSync('tests/fixtures/published-body-hashes.json','utf8'));
 assert.equal(Object.keys(hashes).length,10);
 for(const [p,hash] of Object.entries(hashes))assert.equal(createHash('sha256').update(withoutReferenceAnswers(read(p).body)).digest('hex'),hash,p);
});
