import {test, expect} from '@playwright/test';
import fs from 'node:fs';
const manifest=JSON.parse(fs.readFileSync('tests/fixtures/collection-manifest.json','utf8'));
const url=(base,path)=>new URL(path,base).href;
for(const width of [320,375,768,1440]){
  for(const item of manifest){
    const slug=item.target.split('/')[2];
    test(`${slug} preserves all math and folded proofs at ${width}px`,async({page,baseURL})=>{
      const errors=[];page.on('pageerror',e=>errors.push(e.message));
      await page.setViewportSize({width,height:900});
      await page.goto(url(baseURL,`posts/${slug}/`));
      await expect(page.locator('.collection-nav [aria-current="page"]')).toHaveCount(1);
      await expect(page.locator('.post-content details.proof')).toHaveCount(item.proofs);
      const content=page.locator('.post-content');
      await expect(content).toHaveCSS('font-size','18px');
      if(item.display_math+item.inline_math){
        await expect(content.locator('mjx-container[display="true"]')).toHaveCount(item.display_math);
        await expect(content.locator('mjx-container:not([display="true"])')).toHaveCount(item.inline_math);
        const source=fs.readFileSync(item.target,'utf8');
        const expected=[...source.matchAll(/\$\$([\s\S]*?)\$\$|\\\(([\s\S]*?)\\\)/g)].map(m=>(m[1]??m[2]).trim());
        expect(await page.evaluate(()=>[...MathJax.startup.document.math].map(x=>x.math.trim()))).toEqual(expected);
      }
      await expect(content.locator('mjx-merror,[data-mjx-error]')).toHaveCount(0);
      await expect(content).not.toContainText('$$');
      expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
      // Opening every proof exercises hidden math, tables and long equations as well.
      const proofs=content.locator('details.proof');
      for(let i=0;i<await proofs.count();i++){
        await proofs.nth(i).locator('summary').click();
        await expect(proofs.nth(i)).toHaveAttribute('open','');
      }
      expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
      const boxes=await content.locator('mjx-container[display="true"]').evaluateAll(xs=>xs.map(x=>({left:x.getBoundingClientRect().left,right:x.getBoundingClientRect().right,overflow:getComputedStyle(x).overflowX})));
      for(const box of boxes){expect(box.left).toBeGreaterThanOrEqual(0);expect(box.right).toBeLessThanOrEqual(width);expect(box.overflow).toBe('auto');}
      if(width<768 && slug==='ddpm-3'){
        const scrolled=await content.locator('mjx-container[display="true"]').evaluateAll(xs=>{
          const wide=xs.find(x=>x.scrollWidth>x.clientWidth);
          if(!wide)return 0;
          wide.scrollLeft=wide.scrollWidth;return wide.scrollLeft;
        });
        expect(scrolled).toBeGreaterThan(0);
      }
      const ids=await content.locator('h2,h3').evaluateAll(xs=>xs.map(x=>x.id));
      expect(new Set(ids).size).toBe(ids.length);
      expect(await page.locator('.reading-toc a').evaluateAll(xs=>xs.map(x=>decodeURIComponent(x.hash.slice(1))))).toEqual(ids);
      expect(errors).toEqual([]);
    });
  }
  test(`Archive and collection order at ${width}px`,async({page,baseURL})=>{
    await page.setViewportSize({width,height:900});
    await page.goto(url(baseURL,'archives/'));
    await expect(page.locator('.archive-collection')).toHaveCount(2);
    for(const [id,count] of [['ddpm',4],['elbo',3]]){
      const folder=page.locator('.archive-collection').filter({has:page.locator(`a[href$="/collections/${id}/"]`)});
      await expect(folder.locator('.episode-list a')).toHaveCount(count);
      await folder.locator('summary').click();await expect(folder).not.toHaveAttribute('open','');
      await folder.locator('summary').click();await expect(folder).toHaveAttribute('open','');
      await folder.locator('.collection-index-link').click();
      await expect(page).toHaveURL(url(baseURL,`collections/${id}/`));
      const links=page.locator('.collection-episodes .episode-card');
      expect(await links.evaluateAll(xs=>xs.map(x=>new URL(x.href).pathname.split('/').filter(Boolean).at(-1)))).toEqual(Array.from({length:count},(_,i)=>`${id}-${i+1}`));
      expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
      await links.first().click();
      for(let i=1;i<=count;i++){
        await expect(page).toHaveURL(url(baseURL,`posts/${id}-${i}/`));
        await expect(page.locator('.collection-nav [aria-current="page"]')).toHaveText(new RegExp(`${i}`));
        await expect(page.locator('.collection-prev-next [rel="prev"]')).toHaveCount(i>1?1:0);
        await expect(page.locator('.collection-prev-next [rel="next"]')).toHaveCount(i<count?1:0);
        if(i<count)await page.locator('.collection-prev-next [rel="next"]').click();
      }
      await page.locator('.collection-nav__header a').click();
      await expect(page).toHaveURL(url(baseURL,`collections/${id}/`));
      await page.goBack();await expect(page).toHaveURL(url(baseURL,`posts/${id}-${count}/`));
      await page.goForward();await expect(page).toHaveURL(url(baseURL,`collections/${id}/`));
      await page.goto(url(baseURL,'archives/'));
    }
    await expect(page.locator('.archive-standalone a[href$="/posts/ae-to-vae/"]')).toHaveCount(1);
    await expect(page.locator('.archive-standalone a[href$="/posts/ddim-derivation/"]')).toHaveCount(1);
    expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
  });
}
test('direct proof anchors reveal content and remain usable after repeated navigation',async({page,baseURL})=>{
  await page.goto(url(baseURL,'posts/ddpm-3/'));
  const heading=page.locator('.proof h3').filter({hasText:'先积分原图 再识别真实反向 KL'});
  const id=await heading.getAttribute('id');
  await page.goto('about:blank');
  await page.goto(url(baseURL,`posts/ddpm-3/#${encodeURIComponent(id)}`));
  await page.evaluate(()=>MathJax.startup.promise);
  await expect(heading).toBeVisible();await expect(heading).toBeInViewport();
  const proof=heading.locator('xpath=ancestor::details');
  await proof.locator('summary').click();await expect(proof).not.toHaveAttribute('open','');
  const toc=page.locator('.reading-toc details');await toc.evaluate(el=>el.open=true);
  await toc.locator(`a[href="#${id}"]`).click();await expect(heading).toBeVisible();
});
test('home and RSS expose seven releases without duplicate legacy or collection entries',async({page,request,baseURL})=>{
  await page.goto(baseURL);
  await expect(page.locator('.home-collections a')).toHaveCount(2);
  await expect(page.locator('.post-entry')).toHaveCount(9);
  const rss=await (await request.get(url(baseURL,'index.xml'))).text();
  for(const item of manifest)expect(rss).toContain(`/posts/${item.target.split('/')[2]}/`);
  expect(rss).not.toContain('/posts/ddpm-derivation/');
  expect(rss).not.toContain('/collections/ddpm/');
  expect(rss).not.toContain('/collections/elbo/');
});
test('capture desktop and phone collection reading layouts',async({page,baseURL})=>{
  fs.mkdirSync('test-results/visual-review',{recursive:true});
  for(const width of [375,1440]){
    await page.setViewportSize({width,height:1000});
    for(const route of ['archives/','collections/ddpm/','posts/ddpm-2/','posts/elbo-3/']){
      await page.goto(url(baseURL,route));
      if(route.startsWith('posts/'))await expect(page.locator('.post-content mjx-container').first()).toBeVisible();
      await page.screenshot({path:`test-results/visual-review/${route.replaceAll('/','_')}${width}.png`,fullPage:route==='archives/'||route.startsWith('collections/')});
    }
    const proof=page.locator('.post-content details.proof').first();
    await proof.locator('summary').click();await proof.scrollIntoViewIfNeeded();
    await page.screenshot({path:`test-results/visual-review/expanded-proof-${width}.png`});
  }
});
