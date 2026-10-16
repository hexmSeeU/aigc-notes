import {test,expect} from '@playwright/test';
import fs from 'node:fs';
const baseline=JSON.parse(fs.readFileSync('tests/fixtures/closing-sections-preservation.json','utf8'));
const closing=/小结|自检|回头检查这条推导|这一篇先停在哪里|把一次生成从头到尾收起来|把完整推导接起来|最后把整条推导接起来/;
for(const width of [320,1440])for(const item of baseline.articles.filter(x=>!x.slug.endsWith("-derivation"))){
 test(`${item.slug} keeps final summaries and old anchors at ${width}px`,async({page,baseURL})=>{
  await page.setViewportSize({width,height:900});
  await page.goto(new URL(`posts/${item.slug}/`,baseURL).href);
  const content=page.locator('.post-content');
  const headings=await content.locator('h1,h2,h3,h4,h5,h6').evaluateAll(xs=>xs.map(x=>({id:x.id,level:Number(x.tagName[1]),title:x.textContent})));
  for(const id of item.heading_ids)expect(headings.some(h=>h.id===id),`${item.slug} retains #${id}`).toBe(true);
  const ids=headings.map(h=>h.id);expect(new Set(ids).size).toBe(ids.length);
  for(let i=0;i<headings.length;i++)if(closing.test(headings[i].title)){
   const next=headings.slice(i+1).find(h=>h.level<=headings[i].level);
   expect(next?.level).not.toBe(headings[i].level);
  }
  const ending=headings.find(h=>h.level===2&&closing.test(h.title))??headings.find(h=>closing.test(h.title));
  if(ending){
   const toc=page.locator('.reading-toc details');await toc.evaluate(el=>el.open=true);
   await toc.locator(`a[href="#${ending.id}"]`).click();
   await expect(content.locator(`[id="${ending.id}"]`)).toBeInViewport();
  }
  await expect(content.locator('mjx-container')).toHaveCount(item.math_count);
  await expect(content.locator('mjx-merror,[data-mjx-error],details,summary')).toHaveCount(0);
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
 });
}
