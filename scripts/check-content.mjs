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
// Closing chapters may contain answer subsections, but no later sibling chapter.
export function validateClosingSections(source){
 const closing=/小结|自检|回头检查这条推导|这一篇先停在哪里|把一次生成从头到尾收起来|把完整推导接起来|最后把整条推导接起来/;
 const headings=[];
 let fence=null;
 for(const line of source.split('\n')){
  const marker=line.match(/^ {0,3}(`{3,}|~{3,})/);
  if(marker){
   if(!fence)fence=marker[1];
   else if(marker[1][0]===fence[0]&&marker[1].length>=fence.length&&line.slice(marker[0].length).trim()==='')fence=null;
   continue;
  }
  if(fence)continue;
  const heading=line.match(/^(#{1,6}) (.+)$/);
  if(heading)headings.push({level:heading[1].length,title:heading[2]});
 }
 const errors=[],ancestors=[];
 for(let i=0;i<headings.length;i++){
  const current=headings[i];
  while(ancestors.length&&ancestors.at(-1).level>=current.level)ancestors.pop();
  if(closing.test(current.title)&&!ancestors.some(h=>closing.test(h.title))){
   const next=headings.slice(i+1).find(h=>h.level<=current.level);
   if(next?.level===current.level)errors.push(`Closing section "${current.title}" must be last; followed by "${next.title}"`);
  }
  ancestors.push(current);
 }
 return errors;
}
export function checkContent(root){const errors=[];const dir=path.join(root,'content/posts');if(!fs.existsSync(dir))return errors;for(const entry of fs.readdirSync(dir)){const file=path.join(dir,entry,'index.md');if(!fs.existsSync(file))continue;try{const text=fs.readFileSync(file,'utf8');const end=text.indexOf('\n}\n');const a=JSON.parse(text.slice(0,end+2));errors.push(...validateArticle(a).map(e=>`${entry}: ${e}`));if(!a.draft)errors.push(...validateClosingSections(text.slice(end+2)).map(e=>`${entry}: ${e}`));}catch(e){errors.push(`${entry}: ${e.message}`);}}return errors;}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href){const errors=checkContent(process.cwd());console.log(errors.length?errors.join('\n'):'Content checks passed');process.exitCode=errors.length?1:0;}
