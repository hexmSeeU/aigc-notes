import {test, expect} from '@playwright/test';
import fs from 'node:fs';

const articles = fs.readdirSync('content/posts').flatMap(slug => {
  const source = fs.readFileSync(`content/posts/${slug}/index.md`, 'utf8');
  const meta = JSON.parse(source.slice(0, source.indexOf('\n}\n') + 2));
  return meta.collection && !meta.draft ? [{slug, ...meta}] : [];
});

for (const width of [375, 1440]) for (const article of articles) {
  test(`${article.slug} has only unified previous/next navigation at ${width}px`, async ({page, baseURL}) => {
    await page.setViewportSize({width, height: 900});
    await page.goto(new URL(`posts/${article.slug}/`, baseURL).href);
    await expect(page.locator('.post-content a').filter({hasText: /^(?:前一篇|上一篇|下一篇|继续阅读下一篇)$/})).toHaveCount(0);
    const footer = page.locator('.collection-prev-next');
    await expect(footer).toHaveCount(1);
    for (const [rel, offset] of [['prev', -1], ['next', 1]]) {
      const neighbor = articles.find(item => item.collection === article.collection && item.episode === article.episode + offset);
      const link = footer.locator(`[rel="${rel}"]`);
      await expect(link).toHaveCount(neighbor ? 1 : 0);
      if (neighbor) {
        await expect(link).toHaveAttribute('href', new URL(`posts/${neighbor.slug}/`, baseURL).pathname);
        await expect(link).toContainText(neighbor.title);
      }
    }
  });
}
