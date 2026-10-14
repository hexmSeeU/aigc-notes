import {test, expect} from '@playwright/test';
import {expectedHomeSlugs as expected} from '../scripts/check-home-order.mjs';

for (const width of [375, 1440]) {
  test(`home preserves newest days and ascending episodes across every page at ${width}px`, async ({page, baseURL}) => {
    await page.setViewportSize({width, height: 900});
    await page.goto(baseURL);
    const releases = [];
    const visited = new Set();
    while (true) {
      expect(visited.has(page.url()), 'pagination must not loop').toBe(false);
      visited.add(page.url());
      await expect(page.locator('.home-collections')).toHaveCount(0);
      releases.push(...await page.locator('.post-entry .entry-link').evaluateAll(links =>
        links.map(link => new URL(link.href).pathname.split('/').filter(Boolean).at(-1))));
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
      const next = page.locator('.pagination .next');
      if (!await next.count()) break;
      await next.click();
    }
    expect(visited.size).toBe(2);
    expect(releases).toEqual(expected);
    expect(new Set(releases).size).toBe(expected.length);
    await page.locator('.pagination .prev').click();
    await expect(page).toHaveURL(baseURL);
    await expect(page.locator('[data-home-cover]')).toBeVisible();
    await page.goBack();
    await expect(page).toHaveURL(new URL('page/2/', baseURL).href);
    await expect(page.locator('.post-entry h2').filter({hasText:'ELBO（三）'})).toHaveCount(1);
  });
}
