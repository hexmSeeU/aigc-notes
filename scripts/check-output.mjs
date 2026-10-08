import fs from 'node:fs';import path from 'node:path';import {pathToFileURL} from 'node:url';
export function checkOutput(publicDir,baseURL){
const root=path.resolve(publicDir),base=new URL(baseURL),errors=[];
const walk=d=>fs.readdirSync(d,{withFileTypes:true}).flatMap(e=>e.isDirectory()?walk(path.join(d,e.name)):[path.join(d,e.name)]);
for(const p of ['index.html','404.html','index.xml','sitemap.xml'])if(!fs.existsSync(path.join(root,p)))errors.push('Missing '+p);
if(!fs.existsSync(root))return errors;
for(const file of walk(root).filter(f=>/\.(html|xml)$/.test(f))){const html=fs.readFileSync(file,'utf8');const relative=path.relative(root,file).replaceAll('\\','/');const current=new URL(relative.replace(/index\.html$/,''),base);
if(html.includes('DRAFT-PRIVATE-FIXTURE'))errors.push('Draft leak '+relative);
if(file.endsWith('.html')){const canonical=html.match(/<link[^>]*rel=["']canonical["'][^>]*href=["']([^"']+)/);if(canonical&&(!new URL(canonical[1]).href.startsWith(base.href)))errors.push('Wrong canonical '+relative);}
for(const m of html.matchAll(/(?:href|src)=["']([^"']+)["']|<loc>([^<]+)<\/loc>|<link>([^<]+)<\/link>/g)){const raw=(m[1]||m[2]||m[3]).replaceAll('&amp;','&');if(/^(mailto:|data:|javascript:)/.test(raw))continue;let url;try{url=new URL(raw,current);}catch{errors.push('Bad URL '+raw);continue;}if(url.origin!==base.origin)continue;if(!url.pathname.startsWith(base.pathname)){errors.push('Outside base '+raw);continue;}const rel=decodeURIComponent(url.pathname.slice(base.pathname.length));let target=path.join(root,rel);if(fs.existsSync(target)&&fs.statSync(target).isDirectory())target=path.join(target,'index.html');if(!fs.existsSync(target)){errors.push('Broken '+raw+' in '+relative);continue;}if(url.hash&&target.endsWith('.html')){const id=decodeURIComponent(url.hash.slice(1));const text=fs.readFileSync(target,'utf8');if(!text.includes(`id="${id}"`)&&!text.includes(`id='${id}'`))errors.push('Missing anchor '+raw);}}
}return [...new Set(errors)];}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href){const errors=checkOutput(process.argv[2]||'public',process.argv[3]||'https://hexmSeeU.github.io/aigc-notes/');console.log(errors.length?errors.join('\n'):'Output checks passed');process.exitCode=errors.length?1:0;}
