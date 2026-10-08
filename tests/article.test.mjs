import {test} from 'node:test';import assert from 'node:assert/strict';import fs from 'node:fs';
test('VAE article and original diagrams exist',()=>{for(const f of ['index.md','training.svg','generation.svg'])assert.ok(fs.existsSync('content/posts/ae-to-vae/'+f),f+' missing');});
test('Gaussian KL examples',()=>{const kl=(mu,lv)=>.5*(mu*mu+Math.exp(lv)-1-lv);assert.equal(kl(0,0),0);assert.equal(kl(1,0),.5);});
test('VAE article stays within the introductory lesson',()=>{
  const article=fs.readFileSync('content/posts/ae-to-vae/index.md','utf8');
  assert.doesNotMatch(article,/ELBO|Jensen|Bernoulli|BCE|MSE|logits|softplus|logvar|log 方差|负对数似然|观测分布|\\beta|\\partial/i);
  const training=fs.readFileSync('content/posts/ae-to-vae/training.svg','utf8');
  assert.doesNotMatch(training,/log σ|负对数似然/);
});
