import {test} from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {validateArticle} from '../scripts/check-content.mjs';
const read = path => {
  assert.ok(fs.existsSync(path), `Expected ${path}`);
  const text=fs.readFileSync(path,'utf8'), end=text.indexOf('\n}\n')+2;
  return {meta:JSON.parse(text.slice(0,end)),body:text.slice(end).trim()};
};
for(const [collection,count] of [['ddpm',4],['elbo',3]]){
  test(`${collection} has ${count} independent numbered articles and one collection`,()=>{
    const index=read(`content/collections/${collection}/index.md`);
    assert.equal(index.meta.collectionId,collection);
    assert.equal(index.meta.layout,'collection');
    assert.equal(index.meta.hiddenInRss,true);
    for(let i=1;i<=count;i++){
      const {meta,body}=read(`content/posts/${collection}-${i}/index.md`);
      assert.deepEqual(validateArticle(meta),[]);
      assert.equal(meta.collection,collection);assert.equal(meta.episode,i);
      assert.equal(meta.slug,`${collection}-${i}`);assert.equal(meta.draft,false);
      assert.equal(meta.math,true);
      assert.match(meta.title,new RegExp(`^${collection.toUpperCase()}（${'一二三四'[i-1]}）`));
      assert.doesNotMatch(body,/^# |\]\([^)]*\.md\)/m);
      assert.doesNotMatch(body,/<details>|<summary>/);
    }
  });
}
test('legacy DDPM keeps its URL and content while avoiding duplicate releases',()=>{
  const {meta,body}=read('content/posts/ddpm-derivation/index.md');
  assert.equal(meta.slug,'ddpm-derivation');assert.equal(meta.hiddenInHomeList,true);
  assert.equal(meta.hiddenInRss,true);assert.equal(meta.legacyCollection,'ddpm');
  assert.equal((body.match(/\$\$/g)||[]).length/2,82);
});
test('DDPM published articles preserve every audited original body block',()=>{
  const audit=JSON.parse(fs.readFileSync('tests/fixtures/ddpm-preservation.json','utf8'));
  const all=[1,2,3,4].map(i=>read(`content/posts/ddpm-${i}/index.md`).body).join('\n');
  const compact=s=>s.replace(/\s/g,'');
  assert.equal(audit.paragraph_mapping.length,226);
  for(const unit of audit.paragraph_mapping){
    let expected=unit.original_text;
    for(const change of unit.allowed_changes)expected=expected.replaceAll(change.old,change.new);
    assert.ok(compact(all).includes(compact(expected)),`Missing source unit ${unit.id}`);
  }
});
