import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
export function validateArticle(a){
const errors=[];
for(const key of ['title','description','date','slug','categories','tags','draft']) if(a[key]===undefined||a[key]==='') errors.push(`Missing ${key}`);
if(!/T.*(?:Z|[+-]\d{2}:\d{2})$/.test(a.date??'')) errors.push('Date needs timezone');
if(!Array.isArray(a.categories)||a.categories.some(c=>!['基础','前沿'].includes(c))) errors.push('Invalid category');
if(typeof a.draft!=='boolean') errors.push('Draft must be boolean');
return errors;
}
export function checkContent(root){const errors=[];const dir=path.join(root,'content/posts');if(!fs.existsSync(dir))return errors;for(const entry of fs.readdirSync(dir)){const file=path.join(dir,entry,'index.md');if(!fs.existsSync(file))continue;try{const text=fs.readFileSync(file,'utf8');const end=text.indexOf('\n}\n');const a=JSON.parse(text.slice(0,end+2));errors.push(...validateArticle(a).map(e=>`${entry}: ${e}`));}catch(e){errors.push(`${entry}: ${e.message}`);}}return errors;}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href){const errors=checkContent(process.cwd());console.log(errors.length?errors.join('\n'):'Content checks passed');process.exitCode=errors.length?1:0;}
