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
