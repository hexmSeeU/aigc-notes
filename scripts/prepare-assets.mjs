import fs from 'node:fs';fs.mkdirSync('static/vendor',{recursive:true});fs.cpSync('node_modules/mathjax/es5','static/vendor/mathjax',{recursive:true});
