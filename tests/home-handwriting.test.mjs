import {test} from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createHash} from 'node:crypto';

test('homepage handwriting is limited to the existing hero and uses local licensed fonts', () => {
  const file = 'assets/css/extended/home-handwriting.css';
  assert.ok(fs.existsSync(file), 'hero handwriting styles are missing');
  const css = fs.readFileSync(file, 'utf8');
  assert.match(css, /\.home-cover/);
  assert.match(css, /font-family:/);
  assert.doesNotMatch(css, /font-size:|padding:|margin:|background:|animation:|grid-template|learning-map|home-journey/);
  for (const font of ['caveat-hero.woff', 'long-cang-hero.woff']) {
    assert.ok(fs.statSync(`static/fonts/${font}`).size < 50000);
  }
  for (const license of ['OFL-Caveat.txt', 'OFL-LongCang.txt']) {
    assert.match(fs.readFileSync(`static/fonts/${license}`, 'utf8'), /SIL OPEN FONT LICENSE/);
  }
  const coverage = JSON.parse(fs.readFileSync('static/fonts/hero-font-coverage.json', 'utf8'));
  const config = fs.readFileSync('hugo.toml', 'utf8').match(/\[params\.homeInfoParams\]([\s\S]*?)(?=\n\[|$)/)[1];
  const text = config + fs.readFileSync('layouts/_partials/home_info.html', 'utf8') + fs.readFileSync('static/js/home-cover.js', 'utf8');
  for (const character of text.match(/[\u2e80-\u9fff\uff00-\uffef]/gu)) {
    assert.ok(coverage['long-cang-hero.woff'].includes(character), `missing hero glyph: ${character}`);
  }
});

test('the original hero markup, geometry, artwork and animation code remain byte-identical', () => {
  const baseline = JSON.parse(fs.readFileSync('tests/fixtures/hero-handwriting-baseline.json', 'utf8'));
  for (const [file, hash] of Object.entries(baseline)) {
    assert.equal(createHash('sha256').update(fs.readFileSync(file)).digest('hex'), hash, file);
  }
  assert.ok(!fs.existsSync('layouts/_partials/learning_map.html'));
  assert.ok(!fs.existsSync('assets/css/extended/learning-map.css'));
});
