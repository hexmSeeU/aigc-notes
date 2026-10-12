import {test, expect} from '@playwright/test';

const destinations = [
  'posts/ae-to-vae/', 'collections/elbo/', 'collections/ddpm/',
  'posts/ddim-derivation/', 'collections/guidance/',
];

for (const width of [320, 375, 768, 1024, 1440]) {
  test(`learning route is legible and fits the homepage at ${width}px`, async ({page, baseURL}) => {
    await page.setViewportSize({width, height: 1000});
    await page.goto(baseURL);
    const map = page.getByRole('navigation', {name: 'Learning map'});
    await expect(map).toBeVisible();
    await expect(map.getByRole('listitem')).toHaveCount(5);
    const links = map.getByRole('link');
    await expect(links).toHaveCount(5);
    for (const [index, destination] of destinations.entries()) {
      await expect(links.nth(index)).toHaveAttribute('href', new URL(destination, baseURL).pathname);
      const box = await links.nth(index).boundingBox();
      expect(box.height).toBeGreaterThanOrEqual(44);
      expect(box.x).toBeGreaterThanOrEqual(0);
      expect(box.x + box.width).toBeLessThanOrEqual(width);
    }
    const heroBox = await page.locator('[data-home-cover]').boundingBox();
    const mapBox = await map.boundingBox();
    const postBox = await page.locator('.post-entry').first().boundingBox();
    expect(mapBox.y).toBeGreaterThanOrEqual(heroBox.y + heroBox.height);
    expect(postBox.y).toBeGreaterThanOrEqual(mapBox.y + mapBox.height);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    if (width < 700) {
      const boxes = await links.evaluateAll(nodes => nodes.map(node => ({x: node.getBoundingClientRect().x, y: node.getBoundingClientRect().y})));
      expect(new Set(boxes.map(box => box.x)).size).toBe(1);
      expect(boxes.map(box => box.y)).toEqual(boxes.map(box => box.y).sort((a, b) => a - b));
    }
  });
}

test('learning map supports keyboard navigation and real article round trips', async ({page, baseURL}) => {
  await page.goto(baseURL);
  const links = page.getByRole('navigation', {name: 'Learning map'}).getByRole('link');
  await links.first().focus();
  for (let index = 0; index < destinations.length; index++) {
    await expect(links.nth(index)).toBeFocused();
    expect(await links.nth(index).evaluate(node => getComputedStyle(node).outlineStyle)).not.toBe('none');
    if (index < destinations.length - 1) await page.keyboard.press('Tab');
  }
  for (const [index, destination] of destinations.entries()) {
    await links.nth(index).click();
    await expect(page).toHaveURL(new URL(destination, baseURL).href);
    await expect(page.locator('h1')).toBeVisible();
    await page.goBack();
    await expect(page).toHaveURL(baseURL);
  }
});

test('handwriting loads from this site and all map text retains readable contrast', async ({page, baseURL}) => {
  await page.goto(baseURL);
  await page.evaluate(() => document.fonts.ready);
  const fontURL = new URL('fonts/caveat-latin-600.woff', baseURL).href;
  expect(await page.evaluate(url => performance.getEntriesByType('resource').some(entry => entry.name === url), fontURL)).toBe(true);
  const contrast = await page.locator('.learning-map').evaluate(map => {
    const luminance = rgb => {
      const values = rgb.match(/[\d.]+/g).slice(0, 3).map(value => {
        const channel = Number(value) / 255;
        return channel <= 0.04045 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4;
      });
      return values[0] * 0.2126 + values[1] * 0.7152 + values[2] * 0.0722;
    };
    const background = luminance(getComputedStyle(document.body).backgroundColor);
    return [...map.querySelectorAll('p, h2, h3, .learning-map__annotation, .learning-map__number')].map(node => {
      const foreground = luminance(getComputedStyle(node).color);
      return (Math.max(background, foreground) + 0.05) / (Math.min(background, foreground) + 0.05);
    });
  });
  for (const ratio of contrast) expect(ratio).toBeGreaterThanOrEqual(4.5);
});

test('learning map is page-one-only and works without JavaScript or animation', async ({browser, baseURL}) => {
  const context = await browser.newContext({javaScriptEnabled: false, reducedMotion: 'reduce'});
  const page = await context.newPage();
  await page.goto(baseURL);
  await expect(page.getByRole('navigation', {name: 'Learning map'})).toBeVisible();
  await page.locator('.pagination .next').click();
  await expect(page.getByRole('navigation', {name: 'Learning map'})).toHaveCount(0);
  await page.goto(new URL('posts/', baseURL).href);
  await expect(page.getByRole('navigation', {name: 'Learning map'})).toHaveCount(0);
  await context.close();
});

test('narrow-screen annotations stay separate when the handwriting font is unavailable', async ({page, baseURL}) => {
  await page.route('**/caveat-latin-600.woff', route => route.abort());
  for (const width of [320, 375]) {
    await page.setViewportSize({width, height: 1000});
    await page.goto(baseURL);
    const overlaps = await page.locator('.learning-map__stop').evaluateAll(stops => stops.map(stop => {
      const label = stop.querySelector('.learning-map__annotation').getBoundingClientRect();
      const topic = stop.querySelector('.learning-map__topic').getBoundingClientRect();
      return label.left < topic.right && label.right > topic.left && label.top < topic.bottom && label.bottom > topic.top;
    }));
    expect(overlaps).toEqual([false, false, false, false, false]);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  }
});
