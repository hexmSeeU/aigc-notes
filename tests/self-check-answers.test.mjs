import {test} from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createHash} from 'node:crypto';
import {withoutReferenceAnswers,referenceAnswers} from './helpers/reference-answers.mjs';
const fixture=JSON.parse(fs.readFileSync('tests/fixtures/self-check-answers.json','utf8'));
test('all 13 existing self-check questions have directly readable numbered answers',()=>{
 assert.equal(fixture.reduce((n,x)=>n+x.question_count,0),13);
 for(const item of fixture){
  const source=fs.readFileSync(`content/posts/${item.slug}/index.md`,'utf8');
  for(const question of item.questions)assert.ok(source.includes(question),`${item.slug}: original question preserved`);
  assert.equal((source.match(/^### 参考答案$/gm)||[]).length,1,item.slug);
  const answers=referenceAnswers(source);
  assert.deepEqual([...answers.matchAll(/^(\d)\. \*\*([^\n]+)/gm)].map(m=>Number(m[1])),Array.from({length:item.question_count},(_,i)=>i+1),item.slug);
  assert.doesNotMatch(answers,/选读|proof-open|<details|<summary/);
 }
});
test('adding reference answers leaves every original source byte and all publication metadata intact',()=>{
 for(const item of fixture){
  const source=fs.readFileSync(`content/posts/${item.slug}/index.md`,'utf8');
  assert.equal(createHash('sha256').update(withoutReferenceAnswers(source)).digest('hex'),item.original_source_sha256,item.slug);
  assert.deepEqual(JSON.parse(source.slice(0,source.indexOf('\n}\n')+2)),item.original_meta,item.slug);
 }
});
