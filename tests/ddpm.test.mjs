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
  const {body: fullBody} = readArticle();
  const body = fullBody
    .replace(/^\*\*图片维数 d。\*\*[^\n]*\n\n/m, '')
    .replace(/^## 参数速查：beta、alpha 与 sigma 怎样赋值\n[\s\S]*?(?=^## 12 )/m, '');
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

test('DDPM documents schedule choices, standard deviations, and the endpoint separately', () => {
  const {body} = readArticle();
  const supplement = body.split('## 参数速查：beta、alpha 与 sigma 怎样赋值\n')[1]?.split('## 12 ')[0];
  assert.ok(supplement, 'The parameter assignment section must precede training and sampling');
  for (const fragment of [
    String.raw`\beta_t=\beta_{\min}+\frac{t-1}{T-1}(\beta_{\max}-\beta_{\min})`,
    String.raw`\alpha_t=1-\beta_t`, String.raw`\overline{\alpha}_0=1`,
    String.raw`\overline{\alpha}^{*}_t=\frac{f(t)}{f(0)}`, 's=0.008', '0.999',
    String.raw`\sigma_t^2=\beta_t,\qquad \sigma_t=\sqrt{\beta_t}`,
    String.raw`\sigma_t&=\sqrt{\tilde\beta_t}`, '0.07143', '0.4472', '0.2673',
    '截断后', '累计乘积重算', '第 7 节的 KL 公式',
    '不能把零方差直接代入普通高斯密度', '纯噪声均方误差也不能单独训练方差输出',
  ]) assert.ok(supplement.includes(fragment), fragment);
  assert.match(body, /图片维数 d[\s\S]*?3072/);
  const alphaBarT = Array.from({length: 1000}, (_, index) =>
    1 - (1e-4 + index / 999 * (0.02 - 1e-4))).reduce((product, alpha) => product * alpha, 1);
  assert.ok(Math.abs(alphaBarT - 4.04e-5) < 5e-8);
  assert.ok(supplement.includes(String.raw`4.04\times10^{-5}`));
});
