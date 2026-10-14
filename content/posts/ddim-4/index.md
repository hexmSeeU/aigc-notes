{
  "title": "DDIM（四）：sigma、eta 与 DDPM 的关系",
  "description": "逐项验证 eta 等于 1 恢复后验方差 DDPM，并解释直接删随机项的区别。",
  "summary": "逐项验证 eta 等于 1 恢复后验方差 DDPM，并解释直接删随机项的区别。",
  "date": "2026-10-14T16:23:16+08:00",
  "slug": "ddim-4",
  "categories": [
    "基础"
  ],
  "tags": [
    "DDIM",
    "扩散模型"
  ],
  "draft": false,
  "math": true,
  "collection": "ddim",
  "episode": 4
}

我们已有通用生成公式。本篇选择新噪声大小，逐步验证与 DDPM 的关系，并解释为什么直接删除 DDPM 随机项不等于确定性 DDIM。

## 1 用后验标准差作为基准

\(\sigma_t\) 是新噪声标准差。改变它时，新噪声项与复用噪声系数必须一起变化：

$$x_{t-1}=\sqrt{\bar\alpha_{t-1}}\hat x_0+
\sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\hat\epsilon_t+\sigma_tz.$$

DDPM 已知原图的后验方差是

$$\tilde\beta_t=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t.$$

因此常用选择为

$$\sigma_t=\eta\sqrt{\tilde\beta_t},\qquad0\leq\eta\leq1.$$

\(\eta\) 缩放标准差，方差是 \(\eta^2\tilde\beta_t\)。例如 \(\eta=0.5\) 时是四分之一方差。\(\eta=0\) 不加新噪声，\(\eta=1\) 与 DDPM 后验方差一致，中间值加入部分新噪声。这个范围满足根号约束，但不是参考构造唯一允许的范围。

## 2 eta 等于 1 先化简复用噪声系数

方差相同还不够，需要检查均值。代入 \(\sigma_t^2=\tilde\beta_t\)：

$$\begin{aligned}
1-\bar\alpha_{t-1}-\tilde\beta_t
&=(1-\bar\alpha_{t-1})
\frac{1-\bar\alpha_t-\beta_t}{1-\bar\alpha_t}\\
&=\frac{\alpha_t(1-\bar\alpha_{t-1})^2}{1-\bar\alpha_t}.
\end{aligned}$$

最后一步使用

$$1-\bar\alpha_t-\beta_t
=\alpha_t-\alpha_t\bar\alpha_{t-1}
=\alpha_t(1-\bar\alpha_{t-1}).$$

因此

$$\sqrt{1-\bar\alpha_{t-1}-\tilde\beta_t}
=\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{\sqrt{1-\bar\alpha_t}}.$$

## 3 展开原图项 再合并两个噪声系数

因为 \(\bar\alpha_t=\alpha_t\bar\alpha_{t-1}\)，

$$\begin{aligned}
\sqrt{\bar\alpha_{t-1}}\hat x_0
&=\sqrt{\frac{\bar\alpha_{t-1}}{\bar\alpha_t}}
\left(x_t-\sqrt{1-\bar\alpha_t}\hat\epsilon_t\right)\\
&=\frac{x_t}{\sqrt{\alpha_t}}-
\frac{\sqrt{1-\bar\alpha_t}}{\sqrt{\alpha_t}}\hat\epsilon_t.
\end{aligned}$$

加上第二项后，预测噪声的总系数为

$$\begin{aligned}
&-\frac{\sqrt{1-\bar\alpha_t}}{\sqrt{\alpha_t}}
+\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{\sqrt{1-\bar\alpha_t}}\\
&=\frac{-(1-\bar\alpha_t)+\alpha_t(1-\bar\alpha_{t-1})}
{\sqrt{\alpha_t}\sqrt{1-\bar\alpha_t}}\\
&=-\frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1-\bar\alpha_t}}.
\end{aligned}$$

分子展开后，\(\bar\alpha_t\) 与 \(\alpha_t\bar\alpha_{t-1}\) 抵消，剩下 \(\alpha_t-1=-\beta_t\)。所以完整更新是

$$\boxed{x_{t-1}=\frac1{\sqrt{\alpha_t}}
\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\right)
+\sqrt{\tilde\beta_t}z.}$$

这恢复了使用后验方差的 DDPM 更新。对应关系要求同一套相邻时间步、同一网络预测和相同额外处理；它不代表等同于选择 \(\beta_t\) 或学习方差的所有 DDPM 实现。

## 4 为什么直接删随机项不等于 DDIM

将 DDPM 写回原图估计形式，只删除最后一项，会得到

$$x_{t-1}^{\mathrm{delete}}=
\sqrt{\bar\alpha_{t-1}}\hat x_0+
\sqrt{1-\bar\alpha_{t-1}-\tilde\beta_t}\hat\epsilon_t.$$

而令通用公式的 \(\sigma_t=0\)，得到

$$x_{t-1}^{\mathrm{DDIM}}=
\sqrt{\bar\alpha_{t-1}}\hat x_0+
\sqrt{1-\bar\alpha_{t-1}}\hat\epsilon_t.$$

原图项相同，复用噪声系数不同。参考构造把目标噪声方差分为旧噪声和新噪声。只删除新噪声会少一份方差；确定性 DDIM 则把目标方差全部分给复用噪声。

例如目标方差 \(0.4\)，原本旧噪声 \(0.3\)、新噪声 \(0.1\)：

$$\sqrt{0.3}\epsilon+\sqrt{0.1}z
\quad\longrightarrow\quad
\begin{cases}
\sqrt{0.3}\epsilon&\text{直接删除},\\
\sqrt{0.4}\epsilon&\text{确定性参考构造}.
\end{cases}$$

方差解释针对真实标准噪声的参考过程；实际网络预测不必标准高斯，但两条实际更新公式的系数差异仍是精确的。

参考：[DDIM 原论文第 4.1 节及附录 D.3](https://arxiv.org/html/2010.02502v4)。本文按原论文与官方代码使用标准差参数 \(\eta\)，不要与用 \(\eta\) 直接缩放方差的约定混用。


[前一篇]({{< relref "posts/ddim-3" >}})

[继续阅读下一篇]({{< relref "posts/ddim-5" >}})
