import {test} from 'node:test';import assert from 'node:assert/strict';import fs from 'node:fs';
test('VAE article and original diagrams exist',()=>{for(const f of ['index.md','training.svg','generation.svg'])assert.ok(fs.existsSync('content/posts/ae-to-vae/'+f),f+' missing');});
test('Gaussian KL examples',()=>{const kl=(mu,lv)=>.5*(mu*mu+Math.exp(lv)-1-lv);assert.equal(kl(0,0),0);assert.equal(kl(1,0),.5);});
