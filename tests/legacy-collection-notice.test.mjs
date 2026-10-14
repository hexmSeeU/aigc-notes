import {test} from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';

test('legacy collection notices derive their title and count from the collection', () => {
  const template = fs.readFileSync('layouts/_partials/collection_nav.html', 'utf8');
  const notice = template.slice(template.indexOf('{{- with .Params.legacyCollection }}'));
  assert.match(notice, /\$id := \./);
  assert.match(notice, /where site\.RegularPages "Params\.collection" \$id/);
  assert.match(notice, /\{\{ len \$episodes \}\} 篇独立文章/);
  assert.match(notice, /进入 \{\{ upper \$id \}\} 合集/);
  assert.match(notice, /href="\{\{ \$collection\.RelPermalink \}\}"/);
  assert.doesNotMatch(notice, /四篇|进入 DDPM 合集|原有章节链接/);
});
