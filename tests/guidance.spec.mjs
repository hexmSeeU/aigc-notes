import {test,expect} from '@playwright/test';
import fs from 'node:fs';
const route=(base,path)=>new URL(path,base).href;
for(const width of [320,375,768,1440]){
 test(`CG article preserves full math and visible proofs at ${width}px`,async({page,baseURL})=>{
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.setViewportSize({width,height:1000});
  await page.goto(route(baseURL,'posts/classifier-guidance/'));
  await expect(page.locator('h1')).toHaveText('扩散模型引导（一）：Classifier Guidance');
  await expect(page.locator('.post-meta')).toContainText('2026年10月10日');
  await expect(page.locator('.collection-nav__header')).toContainText('第 1 / 1 篇');
  await expect(page.locator('.collection-nav [aria-current="page"]')).toHaveCount(1);
  await expect(page.locator('.collection-prev-next [rel="prev"],.collection-prev-next [rel="next"]')).toHaveCount(0);
  const content=page.locator('.post-content');
  await expect(content.locator('details,summary')).toHaveCount(0);
  const source=fs.readFileSync('content/posts/classifier-guidance/index.md','utf8');
  const expected=[...source.matchAll(/\$\$([\s\S]*?)\$\$|\\\(([\s\S]*?)\\\)/g)].map(m=>(m[1]??m[2]).trim());
  await expect(content.locator('mjx-container')).toHaveCount(expected.length);
  expect(await page.evaluate(()=>[...MathJax.startup.document.math].map(x=>x.math.trim()))).toEqual(expected);
  await expect(content.locator('mjx-merror,[data-mjx-error]')).toHaveCount(0);
  await expect(content).not.toContainText('$$');
  await expect(content).not.toContainText('\\(');
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
  expect(await content.locator('mjx-container').evaluateAll(xs=>xs.every(x=>x.getClientRects().length>0&&!x.closest('details')))).toBe(true);
  const boxes=await content.locator('mjx-container[display="true"]').evaluateAll(xs=>xs.map(x=>({left:x.getBoundingClientRect().left,right:x.getBoundingClientRect().right,overflow:getComputedStyle(x).overflowX})));
  for(const box of boxes){expect(box.left).toBeGreaterThanOrEqual(0);expect(box.right).toBeLessThanOrEqual(width);expect(box.overflow).toBe('auto');}
  if(width<768){
   const scrolled=await content.locator('mjx-container[display="true"]').evaluateAll(xs=>{
    const wide=xs.find(x=>x.scrollWidth>x.clientWidth+1);
    if(!wide)return 0;
    wide.scrollLeft=wide.scrollWidth;return wide.scrollLeft;
   });
   expect(scrolled).toBeGreaterThan(0);
  }
  const ids=await content.locator('h2,h3').evaluateAll(xs=>xs.map(x=>x.id));
  expect(new Set(ids).size).toBe(ids.length);
  expect(await page.locator('.reading-toc a').evaluateAll(xs=>xs.map(x=>decodeURIComponent(x.hash.slice(1))))).toEqual(ids);
  const toc=page.locator('.reading-toc details');await toc.evaluate(el=>el.open=true);
  await toc.locator('a').nth(2).click();
  const anchor=new URL(page.url()).hash;
  await expect(page.locator(`[id="${decodeURIComponent(anchor.slice(1))}"]`)).toBeInViewport();
  await page.goto('about:blank');await page.goto(route(baseURL,`posts/classifier-guidance/${anchor}`));
  await page.evaluate(()=>MathJax.startup.promise);
  await expect(page.locator(`[id="${decodeURIComponent(anchor.slice(1))}"]`)).toBeInViewport();
  if(width===375||width===1440){
   fs.mkdirSync('test-results/visual-review',{recursive:true});
   await page.goto(route(baseURL,'posts/classifier-guidance/'));
   await page.evaluate(()=>MathJax.startup.promise);
   await page.screenshot({path:`test-results/visual-review/classifier-guidance-${width}.png`});
   const display=[...source.matchAll(/\$\$([\s\S]*?)\$\$/g)].map(m=>m[1]);
   for(const tag of [20,24,38]){
    const index=display.findIndex(math=>math.includes(`\\tag{${tag}}`));
    await content.locator('mjx-container[display="true"]').nth(index).scrollIntoViewIfNeeded();
    await page.screenshot({path:`test-results/visual-review/classifier-guidance-proof-${tag}-${width}.png`});
   }
  }
  expect(errors).toEqual([]);
 });
 test(`guidance collection has one real episode and correct navigation at ${width}px`,async({page,request,baseURL})=>{
  await page.setViewportSize({width,height:1000});
  await page.goto(route(baseURL,'archives/'));
  await page.locator('.archive-entry a[href$="/collections/guidance/"]').click();
  await expect(page.locator('h1')).toHaveText('扩散模型引导：CG 与 CFG');
  await expect(page.locator('.collection-count')).toContainText('共 1 篇');
  await expect(page.locator('.collection-episodes .episode-card')).toHaveCount(1);
  await expect(page.locator('.collection-footer a[href$="/posts/ddim-derivation/"]')).toHaveText('前置：DDIM 跳步采样');
  const links=await page.locator('main a').evaluateAll(xs=>xs.map(x=>x.href));
  for(const href of links)expect((await request.get(href)).status()).toBe(200);
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
  if(width===375||width===1440){
   fs.mkdirSync('test-results/visual-review',{recursive:true});
   await page.screenshot({path:`test-results/visual-review/guidance-collection-${width}.png`,fullPage:true});
  }
  await page.locator('.episode-card').click();
  await expect(page).toHaveURL(route(baseURL,'posts/classifier-guidance/'));
  await page.locator('.collection-nav__header a').click();
  await expect(page).toHaveURL(route(baseURL,'collections/guidance/'));
  await page.goBack();await expect(page).toHaveURL(route(baseURL,'posts/classifier-guidance/'));
  await page.goForward();await expect(page).toHaveURL(route(baseURL,'collections/guidance/'));
 });
}
test('CG has one RSS entry while the guidance collection and CFG have none',async({request,baseURL})=>{
 const rss=await(await request.get(route(baseURL,'index.xml'))).text();
 const entries=[...rss.matchAll(/<item>([\s\S]*?)<\/item>/g)].map(m=>m[1]);
 const cg=entries.filter(e=>e.includes('/posts/classifier-guidance/'));
 expect(cg).toHaveLength(1);
 expect(cg[0]).toContain('Classifier Guidance');
 expect(new Date(cg[0].match(/<pubDate>(.*?)<\/pubDate>/)[1]).toISOString().slice(0,10)).toBe('2026-10-10');
 expect(rss).not.toContain('/collections/guidance/');expect(rss).not.toContain('/posts/classifier-free-guidance/');
 expect((await request.get(route(baseURL,'posts/classifier-free-guidance/'))).status()).toBe(404);
});
