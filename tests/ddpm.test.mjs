import {test} from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createHash} from 'node:crypto';
import {validateArticle} from '../scripts/check-content.mjs';

const articlePath = 'content/posts/ddpm-derivation/index.md';
const readArticle = () => {
  assert.ok(fs.existsSync(articlePath), 'The approved DDPM article must exist');
  const text = fs.readFileSync(articlePath, 'utf8');
  const end = text.indexOf('\n}\n') + 2;
  return {metadata: JSON.parse(text.slice(0, end)), body: text.slice(end).trim()};
};

test('DDPM is the published second article with a real dated math-enabled bundle', () => {
  const {metadata} = readArticle();
  assert.deepEqual(validateArticle(metadata), []);
  assert.equal(metadata.title, 'DDPM：从似然目标到噪声预测的完整推导');
  assert.equal(metadata.slug, 'ddpm-derivation');
  assert.equal(metadata.date, '2026-10-08T22:57:50+08:00');
  assert.equal(metadata.draft, false);
  assert.equal(metadata.math, true);
  assert.deepEqual(metadata.categories, ['基础']);
  assert.deepEqual(metadata.tags, ['DDPM', '扩散模型']);
  assert.ok(metadata.description && metadata.summary);
});

test('DDPM retains all approved mathematical expressions and equation numbers', () => {
  const {body} = readArticle();
  const blocks = [...body.matchAll(/\$\$([\s\S]*?)\$\$/g)].map(match => match[1]);
  const prose = body.replace(/\$\$[\s\S]*?\$\$/g, '');
  const inline = [...prose.matchAll(/\\\(([\s\S]*?)\\\)/g)].map(match => match[1]);
  assert.equal(blocks.length, 75);
  assert.equal(inline.length, 189);
  assert.doesNotMatch(prose, /\$/);
  // Snapshot of all formulas in the audited source, before delimiter conversion.
  // This protects full derivations and expectation subscripts, not just keywords.
  const mathHash = createHash('sha256').update([...blocks, ...inline].join('\n')).digest('hex');
  assert.equal(mathHash, '3dcb5cc106efe933b9cbb87bf35c54025a71ae2eac003eb64a56c31dc2540226');
  assert.deepEqual([...body.matchAll(/\\qquad \((\d+)\)/g)].map(match => Number(match[1])),
    Array.from({length: 22}, (_, index) => index + 1));
});

test('DDPM preserves the constant-equivalence proof and web heading hierarchy', () => {
  const {body} = readArticle();
  assert.equal([...body.matchAll(/^## \d+ /gm)].length, 12);
  assert.doesNotMatch(body, /^# |PAGEBREAK|上一页|TODO|TBD|待补充|占位符/gm);
  for (const heading of [
    '阅读路线', '符号约定', '9 训练知道原图 为什么模型仍只需要带噪输入',
    '10 这个常数等价结论究竟说明什么', '11 简化噪声损失与原始上界差在哪里',
    '12 训练与采样分别怎样执行', '最后把整条推导接起来', '参考资料',
  ]) assert.ok(body.includes(`## ${heading}\n`), heading);
  for (const heading of [
    '先积分原图 再识别真实反向 KL', '用参考后验训练与拟合真实反向分布等价',
    '从带权目标转向简化噪声损失', '训练一次参数更新', '生成一张图片',
  ]) assert.ok(body.includes(`### ${heading}\n`), heading);
  for (const fragment of [
    String.raw`C_{1,t}=\iiint`, String.raw`C_{2,t}&=\iint`,
    String.raw`J_t(\theta)=K_t(\theta)+C_t,\qquad C_t=C_{1,t}-C_{2,t}`,
    String.raw`C_t=\mathbb E_{x_0\sim q_{\mathrm{data}}}`,
    '不是两个概率密度', '必须先进行数据平均', '不能直接当作原始负对数似然的上界',
  ]) assert.ok(body.includes(fragment), fragment);
});
