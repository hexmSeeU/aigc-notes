import {test} from 'node:test';
import assert from 'node:assert/strict';
import {referenceAnswers,withoutReferenceAnswers} from './helpers/reference-answers.mjs';
for(const suffix of ['', '\n## 参考资料\nA link', '\n下一篇会把公式接起来']) {
 test(`answer extraction handles ${suffix ? 'a following block' : 'the final chapter'}`,()=>{
  const source=`## 小结与自检\nQuestion\n### 参考答案\n1. Answer\n${suffix}`;
  assert.equal(referenceAnswers(source),'1. Answer\n');
  assert.equal(withoutReferenceAnswers(source),`## 小结与自检\nQuestion${suffix}`);
 });
}
