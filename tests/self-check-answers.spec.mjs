import {test,expect} from '@playwright/test';
import fs from 'node:fs';
const fixture=JSON.parse(fs.readFileSync('tests/fixtures/self-check-answers.json','utf8'));
for(const width of [320,375,768,1440]){
 for(const item of fixture){
  test(`${item.slug} self-check answers are visible and render correctly at ${width}px`,async({page,baseURL})=>{
   const errors=[];page.on('pageerror',error=>errors.push(error.message));
   await page.setViewportSize({width,height:1000});
   await page.goto(new URL(`posts/${item.slug}/#${encodeURIComponent('参考答案')}`,baseURL).href);
   const heading=page.locator('[id="参考答案"]');
   const answers=page.locator('[id="参考答案"] + ol');
   await expect(heading).toBeVisible();
   await expect(answers.locator(':scope > li')).toHaveCount(item.question_count);
   await expect(page.locator('.post-content details,.post-content summary')).toHaveCount(0);
   for(const answer of await answers.locator(':scope > li').all())await expect(answer).toBeVisible();
   const source=fs.readFileSync(`content/posts/${item.slug}/index.md`,'utf8');
   const expected=[...source.matchAll(/\$\$([\s\S]*?)\$\$|\\\(([\s\S]*?)\\\)/g)].map(m=>(m[1]??m[2]).trim());
   await expect(page.locator('.post-content mjx-container')).toHaveCount(expected.length);
   expect(await page.evaluate(()=>[...MathJax.startup.document.math].map(x=>x.math.trim()))).toEqual(expected);
   await expect(page.locator('mjx-merror,[data-mjx-error]')).toHaveCount(0);
   await expect(answers).not.toContainText('\\(');
   expect(await answers.locator('mjx-container').evaluateAll(xs=>xs.every(x=>x.getClientRects().length>0))).toBe(true);
   expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
   // The new TOC entry remains usable after ordinary repeated navigation.
   await page.goto(new URL(`posts/${item.slug}/`,baseURL).href);
   const toc=page.locator('.reading-toc details');await toc.evaluate(el=>el.open=true);
   await toc.locator('a[href="#参考答案"]').click();
   await expect(heading).toBeInViewport();
   if(width===375||width===1440){
    fs.mkdirSync('test-results/visual-review',{recursive:true});
    await answers.screenshot({path:`test-results/visual-review/answers-${item.slug}-${width}.png`});
   }
   expect(errors).toEqual([]);
  });
 }
}
