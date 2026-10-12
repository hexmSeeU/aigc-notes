import {test, expect} from '@playwright/test';

for (const width of [320, 375, 768, 1440]) {
  test(`handwritten hero preserves the current homepage at ${width}px`, async ({page, baseURL}) => {
    await page.setViewportSize({width, height: 1000});
    await page.goto(baseURL);
    await page.evaluate(() => document.fonts.ready);
    await expect(page.locator('[data-home-cover]')).toBeVisible();
    await expect(page.locator('#home-cover-title')).toHaveText('从噪声，到可能');
    await expect(page.locator('.learning-map,.home-journey,.home-collections')).toHaveCount(0);
    expect(await page.locator('#home-cover-title').evaluate(node => getComputedStyle(node).fontFamily)).toMatch(/long cang hero/i);
    expect(await page.locator('.home-cover__eyebrow').evaluate(node => getComputedStyle(node).fontFamily)).toMatch(/caveat hero/i);
    expect(await page.locator('.post-entry h2').first().evaluate(node => getComputedStyle(node).fontFamily)).not.toMatch(/hero/i);
    expect(await page.locator('.logo a').evaluate(node => getComputedStyle(node).fontFamily)).not.toMatch(/hero/i);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    for (const file of ['caveat-hero.woff', 'long-cang-hero.woff']) {
      const url = new URL(`fonts/${file}`, baseURL).href;
      expect(await page.evaluate(expected => performance.getEntriesByType('resource').some(entry => entry.name === expected), url)).toBe(true);
    }
  });
}

test('hero lettering remains usable when both custom fonts are unavailable', async ({page, baseURL}) => {
  await page.route('**/fonts/*.woff', route => route.abort());
  for (const width of [320, 375, 700, 768, 1440]) {
    await page.setViewportSize({width, height: 1000});
    await page.goto(baseURL);
    await expect(page.locator('#home-cover-title')).toBeVisible();
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await page.getByRole('button', {name: '暂停封面动画'}).click();
    await expect(page.getByRole('button', {name: '播放封面动画'})).toBeVisible();
  }
});
