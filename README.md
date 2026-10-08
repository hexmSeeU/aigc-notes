# mingNotes

中文生成式 AI 学习博客。Hugo + PaperMod；文章在 `content/posts/<slug>/index.md`。

## 本地开发
安装 Hugo **0.150.1 extended**、Node 22+，然后：

```sh
bash scripts/prepare-theme.sh
npm ci --ignore-scripts
node scripts/prepare-assets.mjs
hugo server --buildDrafts
```

## 新增文章
复制 `archetypes/posts/index.md` 到新 page bundle；图片和正文放在同一个目录。保留 JSON front matter，填写标题、描述、含时区的日期、slug、基础/前沿分类、标签。数学文章设 `math: true`。日期用真实发布日期，`lastmod` 只在实质更新时修改。

草稿设 `draft: true`。**公开仓库中的草稿与 Git 历史也是公开的：私人内容不能存入此仓库。**

## 检查与发布
```sh
npm test
npm run check
hugo --minify --baseURL https://hexmSeeU.github.io/aigc-notes/
node scripts/check-output.mjs public
npx playwright install chromium
npm run test:browser
```

发布前校对正文，将文章 `draft` 改为 `false`；main 的更新触发 Pages 工作流。首次在 Settings → Pages 选择 GitHub Actions。部署后检查文章、目录、公式、手机页面、RSS 与资源路径。回退使用 `git revert` 恢复上次正常版本并重新部署。

无需后台、数据库、第三方字体、统计追踪或评论。博客暂无个人署名。
