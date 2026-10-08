import {test} from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
const path='scripts/check-content.mjs';
test('content validator exists',()=>assert.ok(fs.existsSync(path),'missing validator'));
if(fs.existsSync(path)) {
const {validateArticle}=await import('../scripts/check-content.mjs');
const valid={title:'文章',description:'描述',date:'2026-10-08T16:00:00+08:00',slug:'example',categories:['基础'],tags:['VAE'],draft:true};
test('validArticle',()=>assert.deepEqual(validateArticle(valid),[]));
test('missingRequiredField',()=>assert.ok(validateArticle({...valid,title:undefined}).length));
test('invalidCategory',()=>assert.ok(validateArticle({...valid,categories:['其他']}).length));
test('dateWithoutTimezone',()=>assert.ok(validateArticle({...valid,date:'2026-10-08'}).length));
test('optionalMathOmitted',()=>assert.deepEqual(validateArticle(valid),[]));
}
