{
  "aliases": ["/posts/ddim-derivation/"],
  "title": "DDIM：从边缘分布到跳步采样",
  "description": "五篇独立文章，按动机、参考过程、实际生成、DDPM 对应、跳步推导依次学习。",
  "layout": "collection",
  "collectionId": "ddim",
  "weight": 2,
  "hiddenInRss": true,
  "ShowToc": false,
  "date": "2026-10-09T18:17:00+08:00",
  "lastmod": "2026-10-14T16:23:16+08:00",
  "archiveOrder": 4,
  "prerequisite": "/collections/ddpm",
  "prerequisiteLabel": "前置：理解 DDPM"
}


从[DDPM 系列]({{< relref "collections/ddpm" >}})出发，先弄清为什么训练不变却可以换路径，再逐步构造正反向过程、代入网络估计、核对 DDPM 特例，最后推导跨步方差。
