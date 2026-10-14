{
  "title": "DDIM（二）：怎样构造参考正向与反向过程",
  "description": "从两份噪声的方差分配出发，验证边缘分布，并用贝叶斯配方推导正向条件。",
  "summary": "从两份噪声的方差分配出发，验证边缘分布，并用贝叶斯配方推导正向条件。",
  "date": "2026-10-14T16:23:16+08:00",
  "slug": "ddim-2",
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
  "episode": 2
}

这一篇暂时假设真实原图 \(x_0\) 已知。我们先设计反向参考条件，再验证它保留单时刻分布，最后用贝叶斯公式推导对应的正向条件。这里还不是实际生成算法。

## 1 从当前状态提取标准化噪声

保留 DDPM 的单时刻分布后，定义

$$e_t=\frac{x_t-\sqrt{\bar\alpha_t}x_0}{\sqrt{1-\bar\alpha_t}}.$$

固定原图、让当前状态按参考分布变化，有 \(e_t\sim\mathcal N(0,I)\)。但同时固定 \(x_0,x_t\) 时，\(e_t\) 已经是一个确定的向量。

## 2 把目标噪声方差分成两份

目标状态需要原图项 \(\sqrt{\bar\alpha_{t-1}}x_0\)，以及总方差 \((1-\bar\alpha_{t-1})I\) 的噪声。考虑 \(t\geq2\)，选择

$$x_{t-1}=\sqrt{\bar\alpha_{t-1}}x_0+c_te_t+\sigma_tz,
\qquad z\sim\mathcal N(0,I).$$

新噪声 \(z\) 独立于已有状态。复用 \(e_t\) 让两时刻建立联系，加入 \(z\) 让连接有可调随机性。两份标准高斯独立，方差随系数平方缩放，因此要求

$$c_t^2+\sigma_t^2=1-\bar\alpha_{t-1}.$$

选择非负系数：

$$c_t=\sqrt{1-\bar\alpha_{t-1}-\sigma_t^2},
\qquad0\leq\sigma_t^2\leq1-\bar\alpha_{t-1}.$$

这是一种设计，不是“保持边缘”唯一能推出的答案。

## 3 验证只固定原图时的目标分布

只固定 \(x_0\)，两份噪声都随机且均值为零：

$$\mathbb E[x_{t-1}\mid x_0]=\sqrt{\bar\alpha_{t-1}}x_0,$$

$$\operatorname{Cov}(x_{t-1}\mid x_0)
=(c_t^2+\sigma_t^2)I=(1-\bar\alpha_{t-1})I.$$

独立高斯的线性组合仍是高斯，所以目标边缘分布得到保留。再同时固定 \(x_t\)，旧噪声已确定，条件方差则只有 \(\sigma_t^2I\)。两种方差没有矛盾，因为固定的条件不同。

把 \(e_t\) 的定义代入，得到参考反向分布：

$$\begin{aligned}
q_\sigma(x_{t-1}\mid x_t,x_0)
=\mathcal N\!\Bigg(&\sqrt{\bar\alpha_{t-1}}x_0\\
&+c_t\frac{x_t-\sqrt{\bar\alpha_t}x_0}{\sqrt{1-\bar\alpha_t}},
\sigma_t^2I\Bigg).
\end{aligned}$$

先规定正确的末端边缘，再倒序应用这些条件，得到整条路径：

$$q_\sigma(x_{1:T}\mid x_0)
=q_\sigma(x_T\mid x_0)\prod_{t=2}^{T}q_\sigma(x_{t-1}\mid x_t,x_0).$$

刚才的证明保证“当前边缘正确，就得到前一时刻的正确边缘”，因此可以逐步递推到所有时刻。

## 4 为什么引入 rho 以及怎样换回图片

定义标准化目标噪声：

$$e_{t-1}=\frac{x_{t-1}-\sqrt{\bar\alpha_{t-1}}x_0}
{\sqrt{1-\bar\alpha_{t-1}}}.$$

将图片构造除以目标标准差，得到

$$e_{t-1}=\rho_te_t+\sqrt{1-\rho_t^2}z,
\qquad\rho_t=\sqrt{1-\frac{\sigma_t^2}{1-\bar\alpha_{t-1}}}.$$

\(\rho_t\) 不是独立参数，只是标准化后的旧噪声系数。反过来乘回目标标准差：

$$\begin{aligned}
x_{t-1}={}&\sqrt{\bar\alpha_{t-1}}x_0\\
&+\sqrt{1-\bar\alpha_{t-1}}\rho_te_t
+\sqrt{1-\bar\alpha_{t-1}}\sqrt{1-\rho_t^2}z.
\end{aligned}$$

两个系数分别化简为

$$\sqrt{1-\bar\alpha_{t-1}}\rho_t=c_t,\qquad
\sqrt{1-\bar\alpha_{t-1}}\sqrt{1-\rho_t^2}=\sigma_t.$$

这就回到前面的图片构造。

## 5 反方向的标准噪声条件怎样推出来

先设 \(\rho=\rho_t<1\)。根据构造，\(e_t\) 是标准高斯，给定它后，\(e_{t-1}\) 是均值 \(\rho e_t\)、方差 \((1-\rho^2)I\) 的高斯。用贝叶斯公式：

$$p(e_t\mid e_{t-1})\propto
\exp\!\left[-\frac12\left(
\frac{\|e_{t-1}-\rho e_t\|^2}{1-\rho^2}+\|e_t\|^2
\right)\right].$$

固定 \(e_{t-1}\)，展开并配方：

$$\begin{aligned}
&\|e_{t-1}-\rho e_t\|^2+(1-\rho^2)\|e_t\|^2\\
&=\|e_t\|^2-2\rho e_{t-1}^{\mathsf T}e_t+\|e_{t-1}\|^2\\
&=\|e_t-\rho e_{t-1}\|^2+(1-\rho^2)\|e_{t-1}\|^2.
\end{aligned}$$

除以 \(1-\rho^2\) 后，最后一项不依赖正在讨论的变量 \(e_t\)，并入归一化常数。于是

$$e_t\mid e_{t-1}\sim\mathcal N(\rho e_{t-1},(1-\rho^2)I).$$

抽样写法是

$$e_t=\rho_te_{t-1}+\sqrt{1-\rho_t^2}u,
\qquad u\sim\mathcal N(0,I).$$

\(u\) 独立于条件输入，不是原反向构造里的同一份 \(z\)。我们推导的是条件分布，不是在倒用那次随机抽样。

## 6 对应的正向条件仍然依赖原图

将标准噪声换回图片，定义

$$\begin{aligned}
m_t^+(x_{t-1},x_0)={}&\sqrt{\bar\alpha_t}x_0\\
&+\rho_t\sqrt{\frac{1-\bar\alpha_t}{1-\bar\alpha_{t-1}}}
\left(x_{t-1}-\sqrt{\bar\alpha_{t-1}}x_0\right).
\end{aligned}$$

因此

$$q_\sigma(x_t\mid x_{t-1},x_0)=\mathcal N\!\left(
m_t^+(x_{t-1},x_0),
\frac{1-\bar\alpha_t}{1-\bar\alpha_{t-1}}\sigma_t^2I\right).$$

也可以直接写贝叶斯关系：

$$q_\sigma(x_t\mid x_{t-1},x_0)=
\frac{q_\sigma(x_{t-1}\mid x_t,x_0)q_\sigma(x_t\mid x_0)}
{q_\sigma(x_{t-1}\mid x_0)}.$$

一般不能去掉条件 \(x_0\)。原图本来属于整个随机过程的起点；新过程的转移还显式依赖这个起点，所以通常不再是 DDPM 那种只依赖上一状态的正向马尔可夫过程。

## 7 零新噪声与起点单独处理

\(\sigma_t=0\) 时，\(\rho_t=1\)，有 \(e_t=e_{t-1}\)。正反向连接在已知原图后都确定。零方差应理解为点质量；不能代入含倒方差或对数方差的普通高斯密度、KL 公式。

上面的标准化目标公式要求 \(t\geq2\)。正向起点仍为 \(q_\sigma(x_1\mid x_0)\)，而不是对零分母强行使用同一表达式。

参考：[DDIM 原论文第 3.1 节及附录 B](https://arxiv.org/html/2010.02502v4)。


[前一篇]({{< relref "posts/ddim-1" >}})

[继续阅读下一篇]({{< relref "posts/ddim-3" >}})
