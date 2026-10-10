import {test, expect} from '@playwright/test';
import fs from 'node:fs/promises';

const cover = page => page.locator('[data-home-cover]');
const animationStates = (page, waitUntilReady = false) =>
  page.locator('.home-cover__visual').evaluate(async (el, settle) => {
    const animations = el.getAnimations({subtree: true});
    // CSS changes before the pending Web Animations pause task has settled.
    if (settle) await Promise.all(animations.map(animation => animation.ready));
    return animations.map(animation => ({
      currentTime: animation.currentTime,
      playState: animation.playState,
      pending: animation.pending,
    }));
  }, waitUntilReady);
const animationTimes = async page => (await animationStates(page)).map(state => state.currentTime);
const expectFrozen = async page => {
  await expect(page.locator('.home-cover__image')).toHaveCSS('animation-play-state', 'paused');
  const before = await animationStates(page, true);
  expect(before).toHaveLength(3);
  for (const state of before) {
    expect(state).toMatchObject({playState: 'paused', pending: false});
    expect(Number.isFinite(state.currentTime)).toBe(true);
  }
  await page.waitForTimeout(100);
  const after = await animationStates(page);
  expect(after).toHaveLength(before.length);
  for (const [index, state] of after.entries()) {
    expect(state).toMatchObject({playState: 'paused', pending: false});
    // Ignore sub-millisecond clock rounding, not a frame of actual motion.
    expect(Math.abs(state.currentTime - before[index].currentTime)).toBeLessThanOrEqual(1);
  }
  return before.map(state => state.currentTime);
};

for (const width of [320, 375, 768, 1440]) {
  test(`homepage cover fits ${width}px`, async ({page}) => {
    await page.setViewportSize({width, height: 900});
    await page.goto('/');
    await expect(cover(page)).toBeVisible();
    await expect(page.locator('#home-cover-title')).toHaveText('从噪声，到可能');
    await expect(cover(page)).toContainText('AIGC 学习笔记 · 从基础概念到研究前沿');
    const image = cover(page).locator('img');
    await expect(image).toBeVisible();
    expect(await image.evaluate(el => el.complete && el.naturalWidth > 0)).toBe(true);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    if (width < 700) {
      const text = await page.locator('.home-cover__copy').boundingBox();
      const visual = await page.locator('.home-cover__visual').boundingBox();
      expect(visual.y).toBeGreaterThanOrEqual(text.y + text.height);
    }
  });
}

test('keyboard pause freezes the 12-second loop and resumes it', async ({page}) => {
  await page.goto('/');
  await expect(cover(page)).toHaveAttribute('data-animation', 'running');
  const button = page.locator('[data-home-cover-toggle]');
  await expect(button).toHaveAccessibleName('暂停封面动画');
  await button.focus();
  await page.keyboard.press('Enter');
  await expect(cover(page)).toHaveAttribute('data-animation', 'paused');
  await expect(button).toHaveAttribute('aria-pressed', 'true');
  await expect(button).toHaveAccessibleName('播放封面动画');
  const times = await expectFrozen(page);
  const durations = await page.locator('.home-cover__visual').evaluate(el =>
    el.getAnimations({subtree: true}).map(animation => animation.effect.getTiming().duration));
  expect(durations.every(duration => duration === 12000)).toBe(true);
  await page.keyboard.press('Space');
  await expect(cover(page)).toHaveAttribute('data-animation', 'running');
  await expect(button).toHaveAttribute('aria-pressed', 'false');
  await page.waitForTimeout(100);
  expect((await animationTimes(page))[0]).toBeGreaterThan(times[0] + 1);
});

test('reduced motion displays a still image without motion controls', async ({page}) => {
  await page.emulateMedia({reducedMotion: 'reduce'});
  await page.goto('/');
  await expect(cover(page)).toBeVisible();
  await expect(cover(page).locator('img')).toBeVisible();
  await expect(cover(page)).toHaveAttribute('data-animation', 'reduced');
  await expect(page.locator('[data-home-cover-toggle]')).toBeHidden();
  expect(await animationTimes(page)).toHaveLength(0);
  await expect(page.locator('.home-cover__grain')).toHaveCSS('opacity', '0');
  await expect(page.locator('.home-cover__veil')).toHaveCSS('opacity', '0');
});

test('live motion preference changes and offscreen suspension preserve manual pause', async ({page}) => {
  await page.goto('/');
  await expect(cover(page)).toHaveAttribute('data-animation', 'running');
  await page.emulateMedia({reducedMotion: 'reduce'});
  await expect(cover(page)).toHaveAttribute('data-animation', 'reduced');
  expect(await animationTimes(page)).toHaveLength(0);
  await page.emulateMedia({reducedMotion: 'no-preference'});
  await expect(cover(page)).toHaveAttribute('data-animation', 'running');
  await page.evaluate(() => {
    const spacer = document.createElement('div');
    spacer.style.height = '1600px';
    document.body.append(spacer);
    window.scrollTo(0, 1300);
  });
  await expect(cover(page)).toHaveAttribute('data-animation', 'suspended');
  await expectFrozen(page);
  await page.evaluate(() => window.scrollTo(0, 0));
  await expect(cover(page)).toHaveAttribute('data-animation', 'running');
  await page.getByRole('button', {name: '暂停封面动画'}).click();
  await page.evaluate(() => window.scrollTo(0, 1300));
  await expect(cover(page)).toHaveAttribute('data-animation', 'paused');
  await page.evaluate(() => window.scrollTo(0, 0));
  await expect(cover(page)).toHaveAttribute('data-animation', 'paused');
});

test('visibility events suspend motion without losing the pause choice', async ({page}) => {
  await page.goto('/');
  await expect(cover(page)).toHaveAttribute('data-animation', 'running');
  const setHidden = hidden => page.evaluate(value => {
    Object.defineProperty(document, 'hidden', {configurable: true, value});
    document.dispatchEvent(new Event('visibilitychange'));
  }, hidden);
  await setHidden(true);
  await expect(cover(page)).toHaveAttribute('data-animation', 'suspended');
  await expectFrozen(page);
  await setHidden(false);
  await expect(cover(page)).toHaveAttribute('data-animation', 'running');
  await page.getByRole('button', {name: '暂停封面动画'}).click();
  await setHidden(true);
  await setHidden(false);
  await expect(cover(page)).toHaveAttribute('data-animation', 'paused');
});

test('without JavaScript the artwork and copy remain visible and still', async ({browser}) => {
  const context = await browser.newContext({javaScriptEnabled: false});
  const page = await context.newPage();
  await page.goto('/');
  await expect(cover(page)).toBeVisible();
  await expect(cover(page).locator('img')).toBeVisible();
  await expect(page.locator('[data-home-cover-toggle]')).toBeHidden();
  await expect(page.locator('.home-cover__image')).toHaveCSS('animation-name', 'none');
  await expect(page.locator('.home-cover__grain')).toHaveCSS('opacity', '0');
  await expect(page.locator('.home-cover__veil')).toHaveCSS('opacity', '0');
  await context.close();
});

for (const url of ['/posts/ae-to-vae/', '/posts/', '/archives/', '/categories/基础/']) {
  test(`cover and script are absent from ${url}`, async ({page}) => {
    await page.goto(url);
    await expect(cover(page)).toHaveCount(0);
    await expect(page.locator('script[src*="home-cover.js"]')).toHaveCount(0);
  });
}

for (const width of [375, 1440]) {
  test(`noise resolves into the original artwork at ${width}px`, async ({page}) => {
    await page.setViewportSize({width, height: 900});
    await page.goto('/');
    await expect(cover(page)).toHaveAttribute('data-animation', 'running');
    await page.getByRole('button', {name: '暂停封面动画'}).click();
    await expectFrozen(page);
    await fs.mkdir('test-results/visual-review', {recursive: true});

    const frames = [
      {name: 'start', time: 0, opacity: 1},
      {name: 'early', time: 1200, opacity: 1},
      {name: 'middle', time: 3600},
      {name: 'clear', time: 7200, opacity: 0},
      {name: 'last-clear', time: 9600, opacity: 0},
    ];
    for (const frame of frames) {
      await page.locator('.home-cover__visual').evaluate((el, time) => {
        for (const animation of el.getAnimations({subtree: true})) {
          animation.currentTime = time;
        }
      }, frame.time);
      const opacity = await page.locator('.home-cover__visual').evaluate(el => ({
        veil: Number(getComputedStyle(el.querySelector('.home-cover__veil')).opacity),
        grain: Number(getComputedStyle(el.querySelector('.home-cover__grain')).opacity),
      }));
      if (frame.opacity !== undefined) {
        expect(opacity.veil).toBeCloseTo(frame.opacity, 3);
        expect(opacity.grain).toBeCloseTo(frame.opacity, 3);
      } else {
        expect(opacity.veil).toBeGreaterThan(0.1);
        expect(opacity.veil).toBeLessThan(0.9);
        expect(opacity.grain).toBeGreaterThan(0.1);
        expect(opacity.grain).toBeLessThan(0.9);
      }
      await cover(page).screenshot({
        path: `test-results/visual-review/home-cover-${width}-${frame.name}.png`,
        animations: 'allow',
      });
    }
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  });
}
