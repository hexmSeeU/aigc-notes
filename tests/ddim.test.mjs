import {test} from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {validateArticle} from '../scripts/check-content.mjs';
const read = slug => {
 const text=fs.readFileSync(`content/posts/${slug}/index.md`,'utf8');
 const end=text.indexOf('\n}\n')+2;
 return {meta:JSON.parse(text.slice(0,end)),body:text.slice(end).trim()};
};
test('DDIM exposes five ordered episodes and preserves its original URL',()=>{
 const all=[];
 for(let i=1;i<=5;i++){
  const {meta,body}=read(`ddim-${i}`);
  assert.deepEqual(validateArticle(meta),[]);
  assert.equal(meta.collection,'ddim');assert.equal(meta.episode,i);
  assert.equal(meta.math,true);assert.equal(meta.draft,false);
  assert.match(meta.title,new RegExp(`^DDIM（${'一二三四五'[i-1]}）`));
  assert.doesNotMatch(body.replace(/```[\s\S]*?```/g,''),/TODO|TBD|待补充|占位符|^# /m);
  assert.equal((body.match(/\$\$/g)||[]).length%2,0);
  all.push(body);
 }
 const legacy=read('ddim-derivation');
 assert.equal(legacy.meta.legacyCollection,'ddim');
 assert.equal(legacy.meta.hiddenInHomeList,true);
 assert.equal(legacy.meta.hiddenInRss,true);
 for(const episode of all) {
  const core=episode.split('\n\n[前一篇]')[0].split('\n\n[继续阅读下一篇]')[0];
  const math=s=>[...s.matchAll(/\$\$([\s\S]*?)\$\$|\\\(([\s\S]*?)\\\)/g)].map(m=>m[1]??m[2]);
  for(const expression of math(core))assert.ok(math(legacy.body).includes(expression));
 }
 const index=fs.readFileSync('content/collections/ddim/index.md','utf8');
 assert.match(index,/"collectionId": "ddim"/);
});
test('cross-step posterior variance reduces to DDPM and obeys the noise budget',()=>{
 const prev=.75,current=.6,beta=1-current/prev;
 const variance=(1-prev)/(1-current)*(1-current/prev);
 assert.ok(Math.abs(variance-((1-prev)/(1-current)*beta))<1e-12);
 for(const [s,t] of [[.99,.001],[.7,.2],[.4,.39],[1,.1]])for(const eta of [0,.5,1]){
  const v=(1-s)/(1-t)*(1-t/s);
  assert.ok(v>=0&&eta*eta*v<=1-s+1e-12);
  const rhoSquared=s===1?1:1-eta*eta*v/(1-s);
  assert.ok(rhoSquared>=-1e-12&&rhoSquared<=1);
 }
});
test('eta one matches DDPM mean while deleting DDPM noise differs from eta zero',()=>{
 const prev=.8,alpha=.9,current=prev*alpha,beta=1-alpha;
 const v=(1-prev)/(1-current)*beta,x=.37,eps=-.2;
 const x0=(x-Math.sqrt(1-current)*eps)/Math.sqrt(current);
 const ddpm=(x-beta/Math.sqrt(1-current)*eps)/Math.sqrt(alpha);
 const etaOne=Math.sqrt(prev)*x0+Math.sqrt(1-prev-v)*eps;
 const etaZero=Math.sqrt(prev)*x0+Math.sqrt(1-prev)*eps;
 assert.ok(Math.abs(ddpm-etaOne)<1e-12);
 assert.ok(Math.abs(etaZero-ddpm)>1e-4);
});
