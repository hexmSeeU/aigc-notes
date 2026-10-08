import {test} from 'node:test';import assert from 'node:assert/strict';import fs from 'node:fs';
test('production build clears preview draft artifacts',()=>assert.match(fs.readFileSync('.github/workflows/pages.yml','utf8'),/hugo --minify --cleanDestinationDir/));
test('Pages artifact uploads the Hugo public output directory',()=>{
const workflow=fs.readFileSync('.github/workflows/pages.yml','utf8');
const upload=workflow.split(/(?=^      - )/m).find(step=>step.includes('uses: actions/upload-pages-artifact@'));
assert.ok(upload,'Pages artifact upload step is required');
assert.match(upload,/^          path: public$/m);
});
