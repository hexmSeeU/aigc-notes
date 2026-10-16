import {test} from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createHash} from 'node:crypto';
const read=p=>fs.readFileSync(p,'utf8');
for(const [id,count] of [['ddim',5],['ddpm',4]]) {
 test(`${id} old URL is a collection alias, never a content page`,()=>{
  const source=read(`content/collections/${id}/index.md`);
  const meta=JSON.parse(source.slice(0,source.indexOf('\n}\n')+2));
  assert.deepEqual(meta.aliases,[`/posts/${id}-derivation/`]);
  assert.ok(!fs.existsSync(`content/posts/${id}-derivation`));
  assert.ok(fs.existsSync(`archive/legacy/${id}-derivation.md`));
 });
}
test('every substantive DDIM legacy line survives in the five published episodes',()=>{
 const body=read('archive/legacy/ddim-derivation.md').split('\n}\n')[1];
 const all=Array.from({length:5},(_,i)=>read(`content/posts/ddim-${i+1}/index.md`)).join('\n');
 const lines=body.split('\n').filter(l=>l.trim()&&!l.startsWith('#')&&!l.startsWith('本页保留原链接，'));
 for(const line of lines)assert.ok(all.includes(line),`Missing DDIM content: ${line}`);
});
test('published content and navigation never offer legacy full readings',()=>{
 const walk=d=>fs.readdirSync(d,{withFileTypes:true}).flatMap(e=>e.isDirectory()?walk(`${d}/${e.name}`):[`${d}/${e.name}`]);
 for(const path of [...walk('content'),...walk('layouts')].filter(p=>/\.(md|html)$/.test(p))){
  const source=read(path).replace(/^\s*"aliases":.*$/gm,'');
  assert.doesNotMatch(source,/posts\/(ddim|ddpm)-derivation|\.\.\/(ddim|ddpm)-derivation|查看原完整笔记|查看完整推导|完整推导版/,path);
 }
});

test('nonpublished sources remain byte-identical to the audited main revision',()=>{
 const hashes={ddim:'e08d7977eb56774343b6f7cf3bac71df68fa69c751df78a0823b8285a74c2f5e',ddpm:'d033109b9d6550dca11323c635f89667c5aa35c42734307c5a060a8c71536641'};
 for(const [id,hash] of Object.entries(hashes))assert.equal(createHash('sha256').update(read(`archive/legacy/${id}-derivation.md`)).digest('hex'),hash);
});
