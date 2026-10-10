{
  "title": "ELBO（二）：VAE 的编码器究竟在学什么？",
  "description": "把 ELBO 放回编码器和解码器，分清先验 KL、后验 KL 与变分推断。",
  "summary": "把 ELBO 放回编码器和解码器，分清先验 KL、后验 KL 与变分推断。",
  "date": "2026-10-08T14:52:00+08:00",
  "slug": "elbo-2",
  "categories": [
    "基础"
  ],
  "tags": [
    "ELBO",
    "VAE",
    "变分推断"
  ],
  "draft": false,
  "math": true,
  "collection": "elbo",
  "episode": 2
}

VAE 的常见损失是“重建项 + KL”。只看这句话，很容易以为解码器负责重建，编码器负责把潜变量变成标准正态。要理解完整训练目标，需要把两个网络、三个关于潜变量的分布，以及两种不同的 KL 分开。

这一篇先把第一篇的 ELBO 放回 VAE 的计算路径，再解释为什么用一个可计算的损失，能训练编码器去近似难算的后验。最后再回答“变分”究竟是在变什么。

## 1. 同样写成“给定 x 后的 z”，来源却不同

先统一符号：\(x\) 是一张图片，\(z\) 是潜变量；\(\phi\) 是编码器参数，\(\theta\) 是解码器参数。这里采用固定的标准正态先验 \(p(z)=\mathcal N(0,I)\)，\(I\) 是相应维数的单位矩阵。

### 生成模型先确定联合分布

生成时先抽 \(z\)，再生成 \(x\)：

$$z\sim p(z),\qquad x\sim p_\theta(x\mid z).$$

两步合起来确定联合分布

$$p_\theta(x,z)=p(z)p_\theta(x\mid z).$$

观察到图片之后，贝叶斯公式确定模型的精确后验：

$$
\boxed{
p_\theta(z\mid x)
=\frac{p(z)p_\theta(x\mid z)}
{\int p(z')p_\theta(x\mid z')\,dz'}.
}
$$

\(z'\) 只是积分的变量名，遍历的是同一个潜空间。分子同时考虑 \(z\) 在先验中是否常见、它能否解释当前图片；分母把所有贡献加起来，让后验成为归一化的分布。

只要先验和解码器确定，后验就在数学上确定了。它不是解码器直接输出的分布：解码器定义的是 \(p_\theta(x\mid z)\)，后验 \(p_\theta(z\mid x)\) 要结合先验再由贝叶斯公式得到。

难点是归一化所需的积分。所谓“真实后验”，只是当前模型的精确后验，不等于模型已经掌握现实中唯一正确的潜变量来源。

### 编码器提供一个方便处理的近似

VAE 希望根据图片找到合适的潜变量，但直接计算上面的后验通常困难。因此增加编码器，直接输出一个近似分布的参数：

$$x\xrightarrow{\text{编码器 }\phi}\mu_\phi(x),\ \sigma_\phi(x).$$

在常见的对角高斯设定下，它们定义

$$
q_\phi(z\mid x)
=\mathcal N\!\left(z;\mu_\phi(x),
\operatorname{diag}(\sigma_\phi^2(x))\right).
$$

\(\mu_\phi(x)\) 是各维均值，\(\sigma_\phi(x)\) 是各维标准差；\(\operatorname{diag}\) 把各维方差放到协方差矩阵的对角线上。这个分布方便计算密度，也方便抽样。

三个关于 \(z\) 的分布因此有不同角色：

| 分布 | 从哪里来 | 回答什么 |
|---|---|---|
| \(p(z)\) | 预先选定的先验 | 还没有指定图片时，从哪里抽潜变量？ |
| \(p_\theta(z\mid x)\) | 先验、解码器与贝叶斯公式 | 按当前模型，看到图片后哪些潜变量更合理？ |
| \(q_\phi(z\mid x)\) | 编码器输出的分布参数 | 怎样快速给出一个便于处理的后验近似？ |

\(q_\phi\) 在这里被选择为高斯，模型的精确后验却不一定是高斯。分布族受限时，编码器未必能完全匹配它。[VAE 原论文 §2.1、§3](https://arxiv.org/html/1312.6114v11#S3)

## 2. ELBO 里已经包含了两个网络

第一篇得到的下界，代入编码器分布后是

$$
\operatorname{ELBO}(x)
=\mathbb E_{z\sim q_\phi(z\mid x)}
\left[\log\frac{p(z)p_\theta(x\mid z)}{q_\phi(z\mid x)}\right].
$$

这里 \(q_\phi\) 由编码器确定，\(p_\theta(x\mid z)\) 由解码器确定，\(p(z)\) 则固定不训练。

用对数规则展开：

$$
\begin{aligned}
\operatorname{ELBO}(x)
={}&\mathbb E_{z\sim q_\phi(z\mid x)}[\log p_\theta(x\mid z)]\\
&+\mathbb E_{z\sim q_\phi(z\mid x)}[\log p(z)-\log q_\phi(z\mid x)].
\end{aligned}
$$

第二行按 KL 定义改写：

$$
\begin{aligned}
\mathbb E_{z\sim q_\phi(z\mid x)}[\log p(z)-\log q_\phi(z\mid x)]
&=-\mathbb E_{z\sim q_\phi(z\mid x)}
\left[\log\frac{q_\phi(z\mid x)}{p(z)}\right]\\
&=-D_{\mathrm{KL}}\!\left(q_\phi(z\mid x)\|p(z)\right).
\end{aligned}
$$

因此

$$
\operatorname{ELBO}(x)
=\mathbb E_{z\sim q_\phi(z\mid x)}[\log p_\theta(x\mid z)]
-D_{\mathrm{KL}}\!\left(q_\phi(z\mid x)\|p(z)\right).
$$

训练程序通常做最小化，于是定义负 ELBO：

$$
\boxed{
\begin{aligned}
L(x)=-\operatorname{ELBO}(x)
={}&\underbrace{-\mathbb E_{z\sim q_\phi(z\mid x)}
[\log p_\theta(x\mid z)]}_{\text{重建项}}\\
&+\underbrace{D_{\mathrm{KL}}\!\left(q_\phi(z\mid x)\|p(z)\right)}_{\text{先验 KL 项}}.
\end{aligned}
}
$$

到这里都是精确改写。两项不是为了凑出一个好用的损失随意相加，而是来自同一个 ELBO。[VAE 原论文 §2.2，式（3）](https://arxiv.org/html/1312.6114v11#S2.SS2)

## 3. 重建项为什么能同时训练两个网络？

先看一次计算的顺序：

$$
x\xrightarrow{\text{编码器 }\phi}q_\phi(z\mid x)
\xrightarrow{\text{抽样}}z
\xrightarrow{\text{解码器 }\theta}p_\theta(x\mid z).
$$

最后一步是在评价：给解码器这份 \(z\)，它对原来的真实图片 \(x\) 赋予了多高的概率密度？最小化负对数密度，就是鼓励它更好地解释原图。

为了和熟悉的重建误差接起来，假设图片展开后有 \(d\) 维，解码器输出均值 \(f_\theta(z)\)，并选择固定方差的高斯分布：

$$p_\theta(x\mid z)=\mathcal N\!\left(x;f_\theta(z),\tau^2I\right).$$

\(\tau^2>0\) 是固定的解码方差，和编码器输出的方差 \(\sigma_\phi^2(x)\) 是不同的量。这时

$$
-\log p_\theta(x\mid z)
=\frac{1}{2\tau^2}\|x-f_\theta(z)\|^2
+\frac d2\log(2\pi\tau^2).
$$

\(\|\cdot\|^2\) 表示各分量平方后求和。最后一项不依赖网络参数，所以在这个设定下，重建项对应带固定系数的平方误差。

解码器影响 \(f_\theta(z)\)，因此会收到梯度。编码器看似没有直接出现在 \(f_\theta\) 的下标里，但 \(z\) 是它提供的。

用重参数化把这个依赖写出来：

$$
\epsilon\sim\mathcal N(0,I),\qquad
z=\mu_\phi(x)+\sigma_\phi(x)\odot\epsilon.
$$

\(\odot\) 表示对应位置相乘。于是重建平方误差其实是

$$
\left\|x-f_\theta\!\left(
\mu_\phi(x)+\sigma_\phi(x)\odot\epsilon
\right)\right\|^2.
$$

现在可以直接看到：同一个误差中同时出现 \(\phi\) 和 \(\theta\)。固定这次抽到的 \(\epsilon\)，反向传播时，梯度经过解码器、潜变量，再传回编码器。

先验 KL 项则是 \(D_{\mathrm{KL}}(q_\phi(z\mid x)\|p(z))\)。在先验固定的当前设定下，它含 \(\phi\)，不含 \(\theta\)。因此直接的梯度关系为

| 损失项 | 编码器 \(\phi\) | 解码器 \(\theta\) |
|---|---|---|
| 重建项 | 影响潜变量分布，收到梯度 | 影响重建分布，收到梯度 |
| 先验 KL | 约束编码分布，收到梯度 | 当前设定中没有直接梯度 |

两个网络共同优化同一个总损失，并不是各自只负责一项。

## 4. 编码器的目标只是让 z “变成高斯”吗？

在上面的设定中，编码器输出的分布从一开始就被选为高斯族。训练改变的是每张图片对应的均值和标准差，而不是训练以后它才变成高斯。

先验 KL 希望这个高斯靠近特定的标准正态 \(\mathcal N(0,I)\)，也就是鼓励均值靠近零、各维标准差靠近一。

先用一维看这一项。若

$$q_\phi(z\mid x)=\mathcal N(\mu,\sigma^2),\qquad p(z)=\mathcal N(0,1),$$

那么

$$D_{\mathrm{KL}}(q_\phi\|p)=\frac12(\mu^2+\sigma^2-1-\log\sigma^2).$$

- \(\mu=0,\sigma=1\) 时，两个分布相同，KL 为零
- \(\mu=2,\sigma=1\) 时，KL 为 \(2\)，偏离先验需要付出损失

可是编码器还有重建目标。假设只训练 KL，给数字 \(3\) 和数字 \(8\) 都输出

$$q_\phi(z\mid x_{3})=q_\phi(z\mid x_{8})=\mathcal N(0,I),$$

KL 就都为零。每次抽出的具体 \(z\) 仍可能不同，但这些随机差异与输入是 \(3\) 还是 \(8\) 无关。解码器只看到 \(z\)，就不能靠它区分该重建哪张原图。

重建项因此鼓励潜变量保留输入信息。作为一维示意，可以想象

$$q_\phi(z\mid x_3)=\mathcal N(-1,0.2^2),\qquad
q_\phi(z\mid x_8)=\mathcal N(1,0.2^2).$$

从前者抽出的数通常靠近 \(-1\)，从后者抽出的数通常靠近 \(1\)，解码器就更容易区分。这两个分布都偏离标准正态，因此也要付出先验 KL 代价。数值只用于说明两项的作用，不是训练必然得到的结果。

所以完整目标是在先验约束下鼓励保留图片信息，而不是要求所有图片都对应同一个标准正态。这里描述的是损失的推动方向，并不保证每个实际训练结果都达到理想平衡。

## 5. 两种 KL 怎样同时成立？

前面出现两种说法：

1. 先验 KL 让编码器分布靠近 \(p(z)\)
2. 编码器通过 ELBO 去近似模型后验 \(p_\theta(z\mid x)\)

它们对应不同层次。第一篇的差距恒等式是

$$
\operatorname{ELBO}(x)
=\log p_\theta(x)
-D_{\mathrm{KL}}\!\left(q_\phi(z\mid x)\|p_\theta(z\mid x)\right).
$$

本篇把同一个 ELBO 展开为

$$
\operatorname{ELBO}(x)
=\mathbb E_{z\sim q_\phi(z\mid x)}[\log p_\theta(x\mid z)]
-D_{\mathrm{KL}}\!\left(q_\phi(z\mid x)\|p(z)\right).
$$

让这两种写法相等，再取负号：

$$
\boxed{
\begin{aligned}
&-\mathbb E_{z\sim q_\phi(z\mid x)}[\log p_\theta(x\mid z)]
+D_{\mathrm{KL}}\!\left(q_\phi(z\mid x)\|p(z)\right)\\
&=D_{\mathrm{KL}}\!\left(q_\phi(z\mid x)\|p_\theta(z\mid x)\right)
-\log p_\theta(x).
\end{aligned}
}
$$

左边是能计算的“重建项 + 先验 KL”；右边是“后验 KL − 对数似然”。它们是同一个目标的精确改写。

先固定解码器和先验，只更新编码器。\(\log p_\theta(x)\) 不依赖 \(\phi\)，所以最小化左边的完整损失，等价于最小化右边与模型后验的 KL。

这就是关键连接：不必先算出模型后验，也能用重建项和先验 KL 共同训练编码器去近似它。

从贝叶斯公式也能看出为什么需要两项：

$$p_\theta(z\mid x)\propto p(z)p_\theta(x\mid z).$$

正比是针对 \(z\) 而言，省略的是固定图片的归一化因子。后验既考虑潜变量在先验中是否常见，也考虑它对当前图片的解释能力。只保留先验 KL，会丢掉后一部分。

实际训练也会更新 \(\theta\)，于是似然和目标后验一起改变；编码器近似的是当前模型的后验。对 \(\phi\) 求偏导时，\(-\log p_\theta(x)\) 始终不含 \(\phi\)。但跨训练步骤 \(\theta\) 也在更新，因此不能把似然和后验当作始终不变的目标。

## 6. “变分”是在优化一个什么对象？

现在固定 \(x\) 和 \(\theta\)。目标后验已经确定，近似推断变成一个选择问题：在能处理的分布中，选哪个最合适？

普通优化可以写成

$$\min_a(a-3)^2.$$

我们调整一个数 \(a\)，最优解是 \(a=3\)。变分推断则可以写成

$$
\min_{q\in\mathcal Q}
D_{\mathrm{KL}}\!\left(q(z\mid x)\|p_\theta(z\mid x)\right).
$$

\(\mathcal Q\) 是允许选择的分布集合，称为变分分布族；\(q\in\mathcal Q\) 表示从中选择一个分布。我们把“算出一个难算的后验”，转成“在可处理的分布里优化一个近似”。[变分推断综述 §2](https://arxiv.org/abs/1601.00670)

固定 \(x\) 后，密度 \(q(z\mid x)\) 是关于 \(z\) 的函数。KL 接收整条密度函数，最后给出一个数。这种“输入是函数、输出是数”的对象叫泛函；变分法研究如何改变函数，使泛函达到极值。

这里改变的是概率密度，还必须满足

$$q(z\mid x)\geq0,\qquad \int q(z\mid x)\,dz=1.$$

也就是无论怎样调整，都要保持为合法分布。

## 7. 为什么最后仍然用普通反向传播？

实际计算不会任意修改整条密度曲线，而是先限制分布形式。例如一维高斯族：

$$
\mathcal Q=\{\mathcal N(\mu,\sigma^2):\mu\in\mathbb R,\ \sigma>0\}.
$$

选择分布，就变成选择均值和标准差：

$$
\min_{\mu\in\mathbb R,\,\sigma>0}
D_{\mathrm{KL}}\!\left(\mathcal N(\mu,\sigma^2)\|p_\theta(z\mid x)\right).
$$

调整 \(\mu\) 会移动密度曲线，调整 \(\sigma\) 会改变宽窄。于是通过有限个参数，就能调整整个分布。

VAE 再把这些参数交给一个编码器产生：

$$x\xrightarrow{\phi}\mu_\phi(x),\sigma_\phi(x)
\quad\Longrightarrow\quad q_\phi(z\mid x).$$

更新网络参数 \(\phi\)，就在改变编码器提供的近似分布。因此普通反向传播与变分推断并不矛盾：前者用来计算梯度，后者描述这里在寻找哪种分布近似。

ELBO 的作用是让这件事可计算。固定生成模型后，

$$
\operatorname{ELBO}(x)
=\underbrace{\log p_\theta(x)}_{\text{对所选 }q\text{ 为常数}}
-D_{\mathrm{KL}}\!\left(q(z\mid x)\|p_\theta(z\mid x)\right).
$$

因此寻找使 ELBO 最大的 \(q\)，与寻找后验 KL 最小的 \(q\) 等价。允许任意合适分布时，理想解是精确后验；限制为对角高斯等分布族时，理想目标是族内的最好近似。

实际 VAE 使用同一个编码器服务整个数据集，并通过有限训练优化平均目标，不能据此保证每张图片都达到单独优化时的最好近似。这不影响上面的目标恒等式，只是要区分理论目标和实际达到的结果。

## 小结与自检

- 编码器输出的是近似后验 \(q_\phi(z\mid x)\)，解码器定义的是 \(p_\theta(x\mid z)\)
- 模型后验 \(p_\theta(z\mid x)\) 由先验和解码器确定，通常难算
- 先验 KL 是可计算损失的一项；后验 KL 是整个 ELBO 与似然的差距
- 重建项同时训练两个网络；固定先验的 KL 项直接训练编码器
- 变分推断是在分布族里优化近似，VAE 通过编码器参数间接调整这些分布

自检：如果只最小化先验 KL，会不会自动得到有用的编码？为什么解释“完整编码器目标等价于后验拟合”时，必须先固定生成模型？

### 参考答案

1. **不会自动得到有用的编码。** 只最小化先验 KL，理想解是让每张图片都满足 \(q_\phi(z\mid x)=p(z)\)。在本文的标准正态先验下，就是各维 \(\mu_j=0\)、\(\sigma_j=1\)；不同图片不再通过潜变量得到区分。重建项要求潜变量解释当前图片，因此不能省去。
2. **固定生成模型，才能把似然当作常数、把后验当作固定目标。** 对固定图片，完整编码器损失等于 \(D_{\mathrm{KL}}(q_\phi(z\mid x)\|p_\theta(z\mid x))-\log p_\theta(x)\)。固定先验和 \(\theta\) 后，第二项不随编码器参数 \(\phi\) 改变，因此最小化完整损失等价于减小后验 KL。若同时更新 \(\theta\)，恒等式仍成立，但似然和目标后验也会移动，不能把联合训练解释成只拟合一个固定后验。

下一篇把期望、KL 和网络输出逐个变成数，得到可以交给反向传播的 loss，再用同一个框架接回 DDPM。

## 参考资料

- Kingma, D. P. & Welling, M. [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)：§2.2–2.4、§3
- Blei, D. M., Kucukelbir, A. & McAuliffe, J. D. [Variational Inference: A Review for Statisticians](https://arxiv.org/abs/1601.00670)：变分分布族与 ELBO
