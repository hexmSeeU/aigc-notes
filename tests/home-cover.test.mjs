import {test} from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';

test('home cover markup uses a local landscape and project-aware asset URLs', () => {
  const file = 'layouts/_partials/home_info.html';
  assert.ok(fs.existsSync(file), 'homepage cover partial is missing');
  const template = fs.readFileSync(file, 'utf8');
  assert.match(template, /data-home-cover/);
  assert.match(template, /"images\/home-cover\.webp" \| relURL/);
  assert.match(template, /"js\/home-cover\.js" \| relURL/);
  assert.match(template, /data-home-cover-toggle[^>]*hidden/s);
  assert.ok(fs.existsSync('static/images/home-cover.webp'));
});

test('homepage title follows the noise-to-possibility artwork', () => {
  const config = fs.readFileSync('hugo.toml', 'utf8');
  assert.match(config, /Title = '从噪声，到可能'/);
  assert.doesNotMatch(config, /理解原理，记录探索/);
});

const keyframeOpacity = (css, name, percentage) => {
  const rule = css.match(new RegExp(String.raw`@keyframes ${name} \{([\s\S]*?)\n\}`));
  assert.ok(rule, `${name} is missing`);
  for (const frame of rule[1].matchAll(/([^{}]+)\{([^}]+)\}/g)) {
    if (frame[1].split(',').some(value => Number.parseFloat(value) === percentage)) {
      return Number(frame[2].match(/opacity:\s*([\d.]+)/)?.[1]);
    }
  }
  assert.fail(`${name} needs a frame at ${percentage}%`);
};

test('animated cover hides the mountain through its opening and clears the noise completely', () => {
  const css = fs.readFileSync('assets/css/extended/home-cover.css', 'utf8');
  for (const percentage of [0, 10, 100]) {
    assert.equal(keyframeOpacity(css, 'home-cover-reveal', percentage), 1,
      'the opening veil must completely hide the mountain silhouette');
    assert.equal(keyframeOpacity(css, 'home-cover-grain', percentage), 1,
      'the opening noise must remain at full strength');
  }
  for (const percentage of [52, 80]) {
    assert.equal(keyframeOpacity(css, 'home-cover-reveal', percentage), 0);
    assert.equal(keyframeOpacity(css, 'home-cover-grain', percentage), 0);
  }
});
