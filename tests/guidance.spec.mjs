import {test,expect} from '@playwright/test';
import fs from 'node:fs';
const route=(base,path)=>new URL(path,base).href;
const articles=[
 {slug:'classifier-guidance',title:'扩散模型引导（一）：Classifier Guidance',date:'2026-10-10',day:10,episode:1,proofTags:[20,24,38]},
 {slug:'classifier-free-guidance',title:'扩散模型引导（二）：Classifier-Free Guidance',date:'2026-10-12',day:12,episode:2,proofTags:[24,25,27,33,35,51,53]}
];
for(const width of [320,375,768,1440]){
 for(const article of articles){
 test(`${article.slug} article preserves full math and visible proofs at ${width}px`,async({page,baseURL})=>{
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.setViewportSize({width,height:1000});
  await page.goto(route(baseURL,`posts/${article.slug}/`));
  await expect(page.locator('h1')).toHaveText(article.title);
  await expect(page.locator('.post-meta')).toContainText(`2026年10月${article.day}日`);
  await expect(page.locator('.collection-nav__header')).toContainText(`第 ${article.episode} / 2 篇`);
  await expect(page.locator('.collection-nav [aria-current="page"]')).toHaveCount(1);
  await expect(page.locator('.collection-prev-next [rel="prev"],.collection-prev-next [rel="next"]')).toHaveCount(1);
  const sibling=articles.find(x=>x.episode!==article.episode);
  await expect(page.locator(`.collection-prev-next [rel="${article.episode===1?'next':'prev'}"]`)).toHaveAttribute('href',new URL(route(baseURL,`posts/${sibling.slug}/`)).pathname);
  const content=page.locator('.post-content');
  await expect(content.locator('details,summary')).toHaveCount(0);
  const source=fs.readFileSync(`content/posts/${article.slug}/index.md`,'utf8');
  const expected=[...source.matchAll(/\$\$([\s\S]*?)\$\$|\\\(([\s\S]*?)\\\)/g)].map(m=>(m[1]??m[2]).trim());
  await expect(content.locator('mjx-container')).toHaveCount(expected.length);
  expect(await page.evaluate(()=>[...MathJax.startup.document.math].filter(x=>x.typesetRoot?.closest('.post-content')).map(x=>x.math.trim()))).toEqual(expected);
  await expect(page.locator('mjx-merror,[data-mjx-error]')).toHaveCount(0);
  const headingMath=[...source.matchAll(/^#{2,3} (.*)$/gm)].flatMap(m=>[...m[1].matchAll(/\\\(([\s\S]*?)\\\)/g)].map(x=>x[1].trim()));
  await expect(page.locator('.reading-toc mjx-container')).toHaveCount(headingMath.length);
  expect(await page.evaluate(()=>[...MathJax.startup.document.math].filter(x=>x.typesetRoot?.closest('.reading-toc')).map(x=>x.math.trim()))).toEqual(headingMath);
  await expect(content).not.toContainText('$$');
  await expect(content).not.toContainText('\\(');
  const geometry=await page.evaluate(()=>({
   width:innerWidth,pageWidth:document.documentElement.scrollWidth,
   overflowing:[...document.querySelectorAll('.post-content>*,.post-content mjx-container,.collection-nav,.post-title')].map(el=>{
    const rect=el.getBoundingClientRect();return {tag:el.tagName,display:el.getAttribute('display'),text:el.textContent.slice(0,100),left:rect.left,right:rect.right,width:rect.width,minWidth:getComputedStyle(el).minWidth,overflow:getComputedStyle(el).overflowX,style:el.getAttribute('style')};
   }).filter(x=>x.left<0||x.right>innerWidth+1)
  }));
  expect(geometry.pageWidth,JSON.stringify(geometry)).toBeLessThanOrEqual(width);
  expect(await content.locator('mjx-container').evaluateAll(xs=>xs.every(x=>x.getClientRects().length>0&&!x.closest('details')))).toBe(true);
  const boxes=await content.locator('mjx-container[display="true"]').evaluateAll(xs=>xs.map(x=>({left:x.getBoundingClientRect().left,right:x.getBoundingClientRect().right,minWidth:getComputedStyle(x).minWidth,overflow:getComputedStyle(x).overflowX})));
  for(const box of boxes){expect(box.left).toBeGreaterThanOrEqual(0);expect(box.right).toBeLessThanOrEqual(width);expect(box.overflow).toBe('auto');expect(box.minWidth).toBe('0px');}
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
  await page.goto('about:blank');await page.goto(route(baseURL,`posts/${article.slug}/${anchor}`));
  await page.evaluate(()=>MathJax.startup.promise);
  await expect(page.locator(`[id="${decodeURIComponent(anchor.slice(1))}"]`)).toBeInViewport();
  if(width===375||width===1440){
   fs.mkdirSync('test-results/visual-review',{recursive:true});
   await page.goto(route(baseURL,`posts/${article.slug}/`));
   await page.evaluate(()=>MathJax.startup.promise);
   await page.screenshot({path:`test-results/visual-review/${article.slug}-${width}.png`});
   const display=[...source.matchAll(/\$\$([\s\S]*?)\$\$/g)].map(m=>m[1]);
   for(const tag of article.proofTags){
    const index=display.findIndex(math=>math.includes(`\\tag{${tag}}`));
    await content.locator('mjx-container[display="true"]').nth(index).scrollIntoViewIfNeeded();
    await page.screenshot({path:`test-results/visual-review/${article.slug}-proof-${tag}-${width}.png`});
   }
  }
  expect(errors).toEqual([]);
 });
 }
 test(`guidance collection has two real episodes and correct navigation at ${width}px`,async({page,request,baseURL})=>{
  await page.setViewportSize({width,height:1000});
  await page.goto(route(baseURL,'archives/'));
  await page.locator('.archive-entry a[href$="/collections/guidance/"]').click();
  await expect(page.locator('h1')).toHaveText('扩散模型引导：CG 与 CFG');
  await expect(page.locator('.collection-count')).toContainText('共 2 篇');
  await expect(page.locator('.collection-episodes .episode-card')).toHaveCount(2);
  await expect(page.locator('.collection-episodes h2')).toHaveText(articles.map(x=>x.title));
  await expect(page.locator('.collection-count time')).toHaveAttribute('datetime','2026-10-12');
  await expect(page.locator('main')).not.toContainText('后续补充');
  await expect(page.locator('.collection-footer a[href$="/posts/ddim-derivation/"]')).toHaveText('前置：DDIM 跳步采样');
  const links=await page.locator('main a').evaluateAll(xs=>xs.map(x=>x.href));
  for(const href of links)expect((await request.get(href)).status()).toBe(200);
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
  if(width===375||width===1440){
   fs.mkdirSync('test-results/visual-review',{recursive:true});
   await page.screenshot({path:`test-results/visual-review/guidance-collection-${width}.png`,fullPage:true});
  }
  await page.locator('.episode-card').first().click();
  await expect(page).toHaveURL(route(baseURL,'posts/classifier-guidance/'));
  await page.locator('.collection-prev-next [rel="next"]').click();
  await expect(page).toHaveURL(route(baseURL,'posts/classifier-free-guidance/'));
  await page.locator('.collection-prev-next [rel="prev"]').click();
  await expect(page).toHaveURL(route(baseURL,'posts/classifier-guidance/'));
  await page.locator('.collection-nav__header a').click();
  await expect(page).toHaveURL(route(baseURL,'collections/guidance/'));
  await page.goBack();await expect(page).toHaveURL(route(baseURL,'posts/classifier-guidance/'));
  await page.goForward();await expect(page).toHaveURL(route(baseURL,'collections/guidance/'));
 });
}
test('CG and CFG each have one RSS entry and the collection has none',async({request,baseURL})=>{
 const rss=await(await request.get(route(baseURL,'index.xml'))).text();
 const entries=[...rss.matchAll(/<item>([\s\S]*?)<\/item>/g)].map(m=>m[1]);
 for(const article of articles){
  const found=entries.filter(e=>e.includes(`/posts/${article.slug}/`));
  expect(found).toHaveLength(1);
  expect(found[0]).toContain(article.title);
  expect(new Date(found[0].match(/<pubDate>(.*?)<\/pubDate>/)[1]).toISOString().slice(0,10)).toBe(article.date);
  expect((await request.get(route(baseURL,`posts/${article.slug}/`))).status()).toBe(200);
 }
 expect(rss).not.toContain('/collections/guidance/');
});
