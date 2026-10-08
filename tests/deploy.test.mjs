import {test} from 'node:test';import assert from 'node:assert/strict';import fs from 'node:fs';
test('production build clears preview draft artifacts',()=>assert.match(fs.readFileSync('.github/workflows/pages.yml','utf8'),/hugo --minify --cleanDestinationDir/));
