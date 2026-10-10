import {test, expect} from '@playwright/test';
import fs from 'node:fs';

const title = 'DDIM：从边缘分布到跳步采样';
const route = '/posts/ddim-derivation/';

for (const width of [320, 375, 768, 1440]) {
  test(`DDIM renders every equation without page overflow at ${width}px`, async ({page}) => {
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.setViewportSize({width, height: 900});
    await page.goto(route);
    await expect(page.locator('h1')).toHaveText(title);
    const content = page.locator('.post-content');
    await expect(content).toHaveCSS('font-size', '18px');
    await expect(content.locator('mjx-container[display="true"]')).toHaveCount(28);
    await expect(content.locator('mjx-container:not([display="true"])')).toHaveCount(134);
    await expect(content.locator('mjx-merror, [data-mjx-error]')).toHaveCount(0);
    await expect(content.locator('p mjx-container:not([display="true"])').first()).toBeVisible();
    await expect(content).not.toContainText('\\(');
    await expect(content).not.toContainText('$$');
    // Compare MathJax's parsed input, in document order, with every source formula.
    const source = fs.readFileSync('content/posts/ddim-derivation/index.md', 'utf8');
    const expectedMath = [...source.matchAll(/\$\$([\s\S]*?)\$\$|\\\(([\s\S]*?)\\\)/g)]
      .map(match => (match[1] ?? match[2]).trim().replaceAll('&lt;', '<'));
    const renderedMath = await page.evaluate(() =>
      [...MathJax.startup.document.math].map(item => item.math.trim()));
    expect(renderedMath).toEqual(expectedMath);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    if (width === 320) {
      const schedule = content.locator('mjx-container[display="true"]').filter({hasText: '1000'}).first();
      await expect(schedule).toHaveCSS('overflow-x', 'auto');
      const scrolled = await schedule.evaluate(el => {
        el.scrollLeft = el.scrollWidth;
        return el.scrollLeft;
      });
      expect(scrolled).toBeGreaterThan(0);
    }
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
    const code = content.locator('pre');
    await expect(code).toHaveCount(1);
    await expect(code).toContainText('for t, s in adjacent_pairs(times):');
    await expect(code).toContainText('x0_hat = (x - sqrt(1 - alpha_bar[t]) * eps) / sqrt(alpha_bar[t])');
    await expect(code).toHaveCSS('overflow-x', 'auto');
    const codeBounds = await code.evaluate(el => ({
      left: el.getBoundingClientRect().left, right: el.getBoundingClientRect().right,
      fontSize: parseFloat(getComputedStyle(el.querySelector('code')).fontSize),
      whiteSpace: getComputedStyle(el.querySelector('code')).whiteSpace,
    }));
    expect(codeBounds.left).toBeGreaterThanOrEqual(0);
    expect(codeBounds.right).toBeLessThanOrEqual(width);
    expect(codeBounds.fontSize).toBeGreaterThanOrEqual(13);
    expect(codeBounds.whiteSpace).toBe('pre');
  });
}

test('DDIM and VAE remain available alongside the new releases', async ({page, request}) => {
  const titles = [title, '从 AE 到 VAE：让潜变量成为一个分布'];
  for (const path of ['/', '/posts/', '/categories/基础/']) {
    await page.goto(path);
    await expect(page.locator('.post-entry')).toHaveCount(9);
    for(const name of titles) await expect(page.locator('.post-entry h2').filter({hasText:name})).toHaveCount(1);
    await expect(page.locator('.post-entry .entry-link[href$="/posts/ddpm-derivation/"]')).toHaveCount(0);
  }
  await page.goto('/archives/');
  await expect(page.locator('.archive-entry')).toHaveCount(4);
  await expect(page.locator('.archive-entry').last()).toContainText(title);
  const response = await request.get('/index.xml');
  expect(response.ok()).toBe(true);
  const rss = await response.text();
  for (const name of titles) expect(rss).toContain(`<title>${name}</title>`);
  expect(rss).not.toContain('/posts/ddpm-derivation/');
  expect(rss.indexOf('/posts/ddim-derivation/')).toBeLessThan(rss.indexOf('/posts/ae-to-vae/'));
});

for (const width of [375, 1440]) {
  test(`DDIM connects to DDPM and home with working back/forward at ${width}px`, async ({page}) => {
    await page.setViewportSize({width, height: 900});
    await page.goto(route);
    const previous = page.locator('.post-content').getByRole('link', {name: 'DDPM 笔记', exact: true});
    await expect(previous).toHaveAttribute('href', /\/posts\/ddpm-derivation\/$/);
    await previous.click();
    await expect(page).toHaveURL(/\/posts\/ddpm-derivation\/$/);
    await expect(page.locator('h1')).toHaveText('DDPM：从似然目标到噪声预测的完整推导');
    await page.goBack();
    await expect(page.locator('h1')).toHaveText(title);
    await page.locator('#menu').getByRole('link', {name: '首页', exact: true}).click();
    await expect(page.locator('[data-home-cover]')).toBeVisible();
    await expect(page.locator('.post-entry').filter({hasText:title})).toHaveCount(1);
    await page.goBack();
    await expect(page.locator('h1')).toHaveText(title);
    await page.goForward();
    await expect(page.locator('[data-home-cover]')).toBeVisible();
  });
}

for (const width of [375, 1440]) {
  test(`DDIM table of contents reaches the sampling section at ${width}px`, async ({page}) => {
    await page.setViewportSize({width, height: 900});
    await page.goto(route);
    const toc = page.locator('.reading-toc details');
    if (width < 1200) await toc.locator('summary').click();
    await expect(toc).toHaveAttribute('open', '');
    const ids = await page.locator('.post-content h2, .post-content h3').evaluateAll(elements =>
      elements.map(el => el.id));
    expect(new Set(ids).size).toBe(ids.length);
    expect(ids.length).toBe(24);
    const links = toc.locator('a');
    await expect(links).toHaveCount(ids.length);
    expect(await links.evaluateAll(elements => elements.map(el => decodeURIComponent(el.hash.slice(1)))))
      .toEqual(ids);
    const target = toc.getByRole('link', {name: '8 确定性采样的简短伪代码', exact: true});
    const href = await target.getAttribute('href');
    await target.click();
    await expect(page).toHaveURL(new RegExp(`${encodeURI(href)}$`));
    await expect(page.locator('.post-content h2').filter({hasText: '8 确定性采样的简短伪代码'}))
      .toBeInViewport();
  });
}

