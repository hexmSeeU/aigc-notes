import {test,expect} from '@playwright/test';
for(const [id,count] of [['ddim',5],['ddpm',4]])for(const js of [true,false]) {
 test(`${id} redirects with JavaScript ${js ? 'enabled' : 'disabled'}`,async({browser,baseURL,request})=>{
  const old=new URL(`posts/${id}-derivation/`,baseURL).href;
  const target=new URL(`collections/${id}/`,baseURL).href;
  const response=await request.get(old);
  const html=await response.text();
  expect(html).toContain('noindex, follow');
  expect(html).toContain(`content="0; url=${target}"`);
  expect(html).toContain(`rel="canonical" href="${target}"`);
  expect(html).not.toContain('post-content');
  const context=await browser.newContext({javaScriptEnabled:js});
  const page=await context.newPage();
  await page.goto(old+'#old-equation');
  await expect(page).toHaveURL(target);
  await expect(page.locator('.episode-card')).toHaveCount(count);
  await expect(page.locator('a[href*="-derivation"]')).toHaveCount(0);
  await context.close();
 });
}
