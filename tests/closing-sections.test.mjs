import {test} from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createHash} from 'node:crypto';
import {validateClosingSections} from '../scripts/check-content.mjs';
for(const slug of fs.readdirSync('content/posts')) {
 const path=`content/posts/${slug}/index.md`;
 if(!fs.existsSync(path)) continue;
 const source=fs.readFileSync(path,'utf8');
 if(JSON.parse(source.slice(0,source.indexOf('\n}\n')+2)).draft) continue;
 test(`${slug}: summary and self-check close their article or legacy part`,()=>{
  assert.deepEqual(validateClosingSections(source),[],slug);
 });
}
test('moving closing chapters preserves every non-heading content line and its multiplicity',()=>{
 const baseline=JSON.parse(fs.readFileSync('tests/fixtures/closing-sections-preservation.json','utf8'));
 for(const item of baseline.articles){
  const source=fs.readFileSync(`content/posts/${item.slug}/index.md`,'utf8');
  const body=source.slice(source.indexOf('\n}\n')+2);
  const lines=body.split('\n').filter(x=>x.trim()&&!/^#{2,6} /.test(x)).sort();
  const math=[...body.matchAll(/\$\$([\s\S]*?)\$\$|\\\(([\s\S]*?)\\\)/g)].map(m=>m[1]??m[2]).sort();
  assert.equal(math.length,item.math_count,item.slug);
  assert.equal(createHash('sha256').update(JSON.stringify(math)).digest('hex'),item.math_multiset_sha256,item.slug);
  assert.equal(lines.length,item.body_line_count,item.slug);
  assert.equal(createHash('sha256').update(lines.join('\n')).digest('hex'),item.body_lines_sha256,item.slug);
 }
});

test('content validation rejects new chapters after a closing section',()=>{
 assert.deepEqual(validateClosingSections('## 小结与自检\n### 参考答案\nAnswer'),[]);
 assert.equal(validateClosingSections('## 小结\nSummary\n## 参考资料\nLink').length,1);
 assert.equal(validateClosingSections('## 自检\n### 参考答案\nAnswer\n## 补充推导').length,1);
 assert.equal(validateClosingSections('## Part\n### 小结\nSummary\n### More').length,1);
 assert.deepEqual(validateClosingSections('## Part 1\n### 小结\nSummary\n## Part 2\n### Main\n### 自检'),[]);
 assert.deepEqual(validateClosingSections('## 小结与自检\n### 小结\nSummary\n### 自检\nQuestion\n### 参考答案\nAnswer'),[]);
 assert.deepEqual(validateClosingSections('## 小结\n```md\n## Example\n```'),[]);
 assert.deepEqual(validateClosingSections('## 小结\n~~~md\n## Example\n~~~'),[]);
});

test('new articles start with references before the final summary/self-check',()=>{
 const template=fs.readFileSync('archetypes/posts/index.md','utf8');
 assert.deepEqual([...template.matchAll(/^## (.+)$/gm)].map(m=>m[1]).slice(-2),['参考资料','小结与自检']);
 assert.deepEqual(validateClosingSections(template),[]);
});
