import {test, expect} from '@playwright/test';

const routes = ['/posts/ae-to-vae/', '/posts/', '/archives/', '/categories/基础/'];
const menuLabels = ['首页', '文章', '基础', '前沿', '归档'];

for (const width of [320, 375, 1440]) {
  for (const route of routes) {
    test(`首页 returns from ${route} at ${width}px`, async ({page, baseURL}) => {
      const siteURL = new URL('./', baseURL);
      const sourceURL = new URL(`.${route}`, siteURL).href;
      await page.setViewportSize({width, height: 900});
      await page.goto(sourceURL);
      // Hugo's development server canonical host can differ from the test host.
      const canonical = await page.locator('link[rel="canonical"]').getAttribute('href');
      const homeURL = new URL(siteURL.pathname, canonical).href;
      const menu = page.locator('#menu');
      await expect(menu.locator('a')).toHaveText(menuLabels);
      const home = menu.getByRole('link', {name: '首页', exact: true});
      await expect(home).toHaveAttribute('href', homeURL);
      // The existing brand link remains an alternative way home.
      await expect(page.locator('.logo > a')).toHaveAttribute('href', homeURL);
      const bounds = await menu.locator('a').evaluateAll(links => links.map(link => {
        const rect = link.getBoundingClientRect();
        return {left: rect.left, right: rect.right, top: rect.top, bottom: rect.bottom};
      }));
      for (const link of bounds) {
        expect(link.left).toBeGreaterThanOrEqual(0);
        expect(link.right).toBeLessThanOrEqual(width);
        expect(link.bottom - link.top).toBeGreaterThanOrEqual(44);
      }
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
      await home.click();
      await expect(page).toHaveURL(homeURL);
      await expect(page.locator('[data-home-cover]')).toBeVisible();
      await expect(page.locator('#home-cover-title')).toHaveText('理解原理，记录探索。');
      const image = page.locator('.home-cover__image');
      await expect(image).toBeVisible();
      expect(await image.evaluate(el => el.complete && el.naturalWidth > 0)).toBe(true);
      await expect(home.locator('span')).toHaveClass('active');
      // Back/forward and repeat navigation must return to real pages, not anchors.
      await page.goBack();
      await expect(page).toHaveURL(sourceURL);
      if (route === '/posts/ae-to-vae/') {
        await expect(page.locator('.post-content mjx-container').first()).toBeVisible();
        await expect(page.locator('.post-content mjx-merror')).toHaveCount(0);
      }
      await page.goForward();
      await expect(page.locator('[data-home-cover]')).toBeVisible();
      await home.click();
      await expect(page).toHaveURL(homeURL);
      await expect(page.locator('[data-home-cover]')).toBeVisible();
    });
  }
}
