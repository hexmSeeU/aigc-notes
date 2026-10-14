import {test, expect} from '@playwright/test';
import fs from 'node:fs';

const title = 'DDIM：从边缘分布到跳步采样';
const route = '/posts/ddim-derivation/';

for (const width of [320, 375, 768, 1440]) {
  test(`DDIM renders every equation without page overflow at ${width}px`, async ({page, baseURL}) => {
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.setViewportSize({width, height: 900});
    await page.goto(new URL(route.replace(/^\//, ''), baseURL).href);
    await expect(page.locator('h1')).toHaveText(title);
    const content = page.locator('.post-content');
    await expect(content).toHaveCSS('font-size', '18px');
    const sourceText = fs.readFileSync('content/posts/ddim-derivation/index.md', 'utf8');
    await expect(content.locator('mjx-container[display="true"]')).toHaveCount([...sourceText.matchAll(/\$\$([\s\S]*?)\$\$/g)].length);
    await expect(content.locator('mjx-container:not([display="true"])')).toHaveCount([...sourceText.matchAll(/\\\(([\s\S]*?)\\\)/g)].length);
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
