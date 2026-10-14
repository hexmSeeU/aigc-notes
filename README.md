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

## 文章结尾规范

- 每篇学习笔记，以及完整推导页中的每一部分，已有的“小结与自检”必须放在该篇／该部分最后一个章节；参考答案紧跟对应问题，直接展开。只有小结或自检时也遵循同样顺序，不要求为没有自检的文章补造题目。
- 补充推导、参考资料和后续阅读先写，再以小结／自检收尾。不要在结尾后追加同级正文或参考资料章节；站点自动生成的上一篇／下一篇导航可以保留在页脚。
- 调整顺序时保留全部正文、公式、问题、答案和链接。章节编号随新顺序调整；改标题或层级时，用显式 `{#原锚点}` 保留已有链接。
- `npm test` 会检查已发表文章的结尾顺序，`npm run check` 也会拒绝提前出现的小结／自检。改动历史顺序快照前，先用 `closing-sections-preservation.json` 验证原有正文未删减。

## 检查与发布
```sh
npm test
npm run check
hugo --minify --cleanDestinationDir --baseURL https://hexmSeeU.github.io/aigc-notes/
node scripts/check-output.mjs public
npx playwright install chromium
npm run test:browser
```

发布前校对正文，将文章 `draft` 改为 `false`；main 的更新触发 Pages 工作流。首次在 Settings → Pages 选择 GitHub Actions。部署后检查文章、目录、公式、手机页面、RSS 与资源路径。回退使用 `git revert` 恢复上次正常版本并重新部署。

无需后台或数据库；字体本地托管，不使用第三方字体服务、统计追踪或评论。博客暂无个人署名。
