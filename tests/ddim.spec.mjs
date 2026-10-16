import {test, expect} from '@playwright/test';
import fs from 'node:fs';

for (const width of [375, 1440]) for (let i=1; i<=5; i++) {
 test(`DDIM episode ${i} renders math and navigation at ${width}px`,async({page,baseURL})=>{
  await page.setViewportSize({width,height:900});
  await page.goto(new URL(`posts/ddim-${i}/`,baseURL).href);
  const source=fs.readFileSync(`content/posts/ddim-${i}/index.md`,'utf8');
  const expected=[...source.matchAll(/\$\$([\s\S]*?)\$\$|\\\(([\s\S]*?)\\\)/g)].map(m=>(m[1]??m[2]).trim());
  await expect(page.locator('.post-content mjx-container')).toHaveCount(expected.length);
  expect(await page.evaluate(()=>[...MathJax.startup.document.math].map(x=>x.math.trim()))).toEqual(expected);
  await expect(page.locator('mjx-merror,[data-mjx-error]')).toHaveCount(0);
  await expect(page.locator('.collection-nav [aria-current="page"]')).toHaveCount(1);
  await expect(page.locator('.collection-prev-next [rel="next"]')).toHaveCount(i<5?1:0);
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
 });
}

for(const width of [320,375,768,1440]) {
 test(`DDIM split sampling article keeps scrollable code and schedule at ${width}px`,async({page,baseURL})=>{
  await page.setViewportSize({width,height:900});
  await page.goto(new URL('posts/ddim-5/',baseURL).href);
  const content=page.locator('.post-content');
  const source=fs.readFileSync('content/posts/ddim-5/index.md','utf8');
  await expect(content.locator('mjx-container')).toHaveCount([...source.matchAll(/\$\$([\s\S]*?)\$\$|\\\(([\s\S]*?)\\\)/g)].length);
  const code=content.locator('pre');
  await expect(code).toHaveCount(1);
  await expect(code).toContainText('for t, s in adjacent_pairs(times):');
  await expect(code).toContainText('x0_hat = (x - sqrt(1 - alpha_bar[t]) * eps) / sqrt(alpha_bar[t])');
  await expect(code).toHaveCSS('overflow-x','auto');
  if(width===320){
   const schedule=content.locator('mjx-container[display="true"]').filter({hasText:'1000'}).first();
   await expect(schedule).toHaveCSS('overflow-x','auto');
   expect(await schedule.evaluate(el=>{el.scrollLeft=el.scrollWidth;return el.scrollLeft;})).toBeGreaterThan(0);
  }
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
 });
}
