import {test, expect} from '@playwright/test';
import fs from 'node:fs';

const title = 'DDPM：从似然目标到噪声预测的完整推导';
const route = '/posts/ddpm-derivation/';

for (const width of [320, 375, 768, 1440]) {
  test(`DDPM renders every equation without page overflow at ${width}px`, async ({page}) => {
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.setViewportSize({width, height: 900});
    await page.goto(route);
    await expect(page.locator('h1')).toHaveText(title);
    const content = page.locator('.post-content');
    await expect(content).toHaveCSS('font-size', '18px');
    await expect(content.locator('mjx-container[display="true"]')).toHaveCount(82);
    await expect(content.locator('mjx-container:not([display="true"])')).toHaveCount(233);
    await expect(content.locator('mjx-merror, [data-mjx-error]')).toHaveCount(0);
    await expect(content.locator('p mjx-container:not([display="true"])').first()).toBeVisible();
    await expect(content.locator('td mjx-container:not([display="true"])').first()).toBeVisible();
    await expect(content).not.toContainText('\\(');
    await expect(content).not.toContainText('$$');
    // Compare MathJax's parsed input, in document order, with every source formula.
    const source = fs.readFileSync('content/posts/ddpm-derivation/index.md', 'utf8');
    const expectedMath = [...source.matchAll(/\$\$([\s\S]*?)\$\$|\\\(([\s\S]*?)\\\)/g)]
      .map(match => (match[1] ?? match[2]).trim());
    const renderedMath = await page.evaluate(() =>
      [...MathJax.startup.document.math].map(item => item.math.trim()));
    expect(renderedMath).toEqual(expectedMath);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    // Long equations must scroll within their own box, never be clipped or widen the page.
    const equations = await content.locator('mjx-container[display="true"]').evaluateAll(elements =>
      elements.map(el => ({
        overflow: getComputedStyle(el).overflowX,
        clientWidth: el.clientWidth,
        scrollWidth: el.scrollWidth,
        left: el.getBoundingClientRect().left,
        right: el.getBoundingClientRect().right,
      })));
    for (const equation of equations) {
      expect(equation.overflow).toBe('auto');
      expect(equation.left).toBeGreaterThanOrEqual(0);
      expect(equation.right).toBeLessThanOrEqual(width);
    }
    if (width < 768) {
      expect(equations.some(equation => equation.scrollWidth > equation.clientWidth)).toBe(true);
      const scrolled = await content.locator('mjx-container[display="true"]').evaluateAll(elements => {
        const equation = elements.find(el => el.scrollWidth > el.clientWidth);
        equation.scrollLeft = equation.scrollWidth;
        return equation.scrollLeft;
      });
      expect(scrolled).toBeGreaterThan(0);
    }
    expect(errors).toEqual([]);
  });
}

for (const width of [375, 1440]) {
  test(`DDPM table of contents reaches the full constant proof at ${width}px`, async ({page}) => {
    await page.setViewportSize({width, height: 900});
    await page.goto(route);
    const toc = page.locator('.reading-toc details');
    if (width < 1200) await toc.locator('summary').click();
    await expect(toc).toHaveAttribute('open', '');
    const ids = await page.locator('.post-content h2, .post-content h3').evaluateAll(elements =>
      elements.map(el => el.id));
    expect(new Set(ids).size).toBe(ids.length);
    expect(ids.length).toBe(26);
    const links = toc.locator('a');
    await expect(links).toHaveCount(ids.length);
    expect(await links.evaluateAll(elements => elements.map(el => decodeURIComponent(el.hash.slice(1)))))
      .toEqual(ids);
    const target = toc.getByRole('link', {name: '先积分原图 再识别真实反向 KL', exact: true});
    const href = await target.getAttribute('href');
    await target.click();
    await expect(page).toHaveURL(new RegExp(`${encodeURI(href)}$`));
    await expect(page.locator('.post-content h3').filter({hasText: '先积分原图 再识别真实反向 KL'}))
      .toBeInViewport();
  });
}

test('DDPM remains on the homepage and in RSS before VAE', async ({page, request}) => {
  await page.goto('/');
  const entry = page.locator('.post-entry').filter({hasText: title});
  await expect(entry).toHaveCount(1);
  await expect(entry.locator('.entry-link')).toHaveAttribute('href', /\/posts\/ddpm-derivation\/$/);
  await expect(page.locator('.post-entry').filter({hasText: '从 AE 到 VAE：让潜变量成为一个分布'})).toHaveCount(1);
  const response = await request.get('/index.xml');
  expect(response.ok()).toBe(true);
  const rss = await response.text();
  expect(rss).toContain(`<title>${title}</title>`);
  expect(rss).toContain('/posts/ddpm-derivation/');
  expect(rss).toContain('/posts/ae-to-vae/');
  expect(rss.indexOf('/posts/ddpm-derivation/')).toBeLessThan(rss.indexOf('/posts/ae-to-vae/'));
});
