import fs from 'node:fs';fs.mkdirSync('content/math-regression',{recursive:true});fs.copyFileSync('tests/fixtures-math.md','content/math-regression/index.md');
