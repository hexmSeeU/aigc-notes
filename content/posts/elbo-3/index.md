{
  "title": "ELBO（三）：怎样把训练目标算成一个 loss？",
  "description": "把网络输出、重参数化、重建项和 KL 变成实际 loss，再接回 DDPM。",
  "summary": "把网络输出、重参数化、重建项和 KL 变成实际 loss，再接回 DDPM。",
  "date": "2026-10-10T14:52:00+08:00",
  "slug": "elbo-3",
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
  "episode": 3
}

前两篇解释了为什么使用 ELBO，以及它怎样同时训练 VAE 的编码器和解码器。到了实际训练，还差一个问题：公式里有期望、有两个分布的 KL，程序最后怎样得到一个能反向传播的数？

这一篇按一次训练的顺序，把每个量真正算出来，再把同一个变分框架对应到 DDPM。先分清三个层次：理论损失、抽样估计，以及为了优化而省略的固定常数。

## 1. 先选定这一次讨论的模型

使用与前两篇一致的设定：

- 一批有 \(B\) 张图片，每张展平后是 \(d\) 维
- 每张图片对应一个 \(k\) 维潜变量
- 编码器参数为 \(\phi\)，解码器参数为 \(\theta\)
- 先验固定为 \(p(z)=\mathcal N(0,I_k)\)
- 编码器给出对角高斯 \(q_\phi(z\mid x)\)
- 解码器采用固定方差的高斯分布，输出它的均值

\(I_k\) 是 \(k\times k\) 的单位矩阵；后文 \(I_d\) 同理。注意这里 \(d\) 是图片维度，\(k\) 是潜变量维度，\(B\) 是样本数，三者不能混用。

例如，\(B=32\)、\(d=784\)、\(k=16\)，表示一次输入 32 张展平的 \(28\times28\) 灰度图，每张图使用 16 维潜变量。

## 2. 最大化 ELBO，先改成最小化负 ELBO

单张图片的理论损失为

$$
\begin{aligned}
L(x)&=-\operatorname{ELBO}(x)\\
&=\underbrace{-\mathbb E_{z\sim q_\phi(z\mid x)}
[\log p_\theta(x\mid z)]}_{L_{\mathrm{rec}}(x)}
+\underbrace{D_{\mathrm{KL}}\!\left(q_\phi(z\mid x)\|p(z)\right)}_{L_{\mathrm{KL}}(x)}.
\end{aligned}
$$

取负号只是配合程序中常见的最小化操作，没有引入近似。接下来才逐个处理重建期望和 KL。

## 3. 编码器输出 logvar，怎样得到实际的 z？

实现中，编码器常输出均值和对数方差：

$$\mu_\phi(x),\qquad \ell_\phi(x)=\log\sigma_\phi^2(x).$$

\(\ell\) 就是常见的 logvar。它是对数方差，不是对数标准差。因此

$$\sigma_\phi^2(x)=\exp(\ell_\phi(x)),\qquad
\sigma_\phi(x)=\exp\!\left(\tfrac12\ell_\phi(x)\right).$$

这些操作逐维进行。网络可以输出任意实数的 \(\ell\)，经过指数变换后，方差为正。

抽取标准正态噪声并重参数化：

$$
\epsilon\sim\mathcal N(0,I_k),\qquad
z=\mu_\phi(x)+\exp\!\left(\tfrac12\ell_\phi(x)\right)\odot\epsilon.
$$

\(\odot\) 表示逐元素相乘。把这个具体的 \(z\) 送入解码器，得到重建均值

$$\hat x=f_\theta(z).$$

这时计算损失需要的原图 \(x\)、重建均值 \(\hat x\)、潜变量均值 \(\mu\)、对数方差 \(\ell\) 都有了。

| 量 | 含义 | 一批数据的形状 |
|---|---|---|
| \(x\) | 输入图片 | \([B,d]\) |
| \(\mu,\ell\) | 每张图片的潜变量分布参数 | \([B,k]\) |
| \(\epsilon,z\) | 标准噪声与潜变量样本 | \([B,k]\) |
| \(\hat x\) | 解码器给出的重建均值 | \([B,d]\) |

固定网络和输入时，\(\mu\)、\(\ell\) 不变；重新抽取 \(\epsilon\)，\(z\) 仍可以改变。抽样没有消失，只是被放进了一个便于求梯度的表达式。[VAE 原论文 §2.3–2.4](https://arxiv.org/html/1312.6114v11#S2.SS3)

## 4. 解码分布决定重建项怎样计算

这里选择

$$p_\theta(x\mid z)=\mathcal N\!\left(x;f_\theta(z),\tau^2I_d\right).$$

\(\tau^2>0\) 是固定的解码方差，用来描述给定 \(z\) 后图片的条件波动。它和编码器输出的 \(\sigma_\phi^2(x)\) 是不同的量：一个描述图片分布，一个描述潜变量分布。

这个高斯密度为

$$
p_\theta(x\mid z)
=(2\pi\tau^2)^{-d/2}
\exp\!\left[-\frac{1}{2\tau^2}
\sum_{r=1}^{d}(x_r-\hat x_r)^2\right].
$$

对它取负对数，乘法变成加法，指数中的平方项被留下：

$$
-\log p_\theta(x\mid z)
=\frac{1}{2\tau^2}\sum_{r=1}^{d}(x_r-\hat x_r)^2
+\frac d2\log(2\pi\tau^2).
$$

\(r\) 遍历图片各分量。最后一项对两个网络参数都不变，可以在优化时省略。它在数值上仍然存在，所以省略后的数值不再是完整负对数密度。

重建平方误差之所以出现，依赖于固定方差高斯这个建模选择。不能不说明解码分布，就把所有 VAE 的重建项都直接写成 MSE。

还有一个系数不能随意丢掉：\(1/(2\tau^2)\)。它决定重建项相对于 KL 的权重。只把重建项的系数改掉而保持 KL 不变，通常会改变优化目标；这与省略加法常数不同。

## 5. 理论上要对 z 平均，每次训练怎么做？

重建项的理论定义是

$$L_{\mathrm{rec}}(x)
=\mathbb E_{z\sim q_\phi(z\mid x)}[-\log p_\theta(x\mid z)].$$

它平均的是同一张图片对应的不同 \(z\)，不是平均图片像素，也不是平均不同图片。

实际训练常常每张图片只抽一个 \(z\)，用这一次的负对数密度估计期望。省略固定加法常数后，得到

$$
\widehat L_{\mathrm{rec}}(x)
=\frac{1}{2\tau^2}\sum_{r=1}^{d}(x_r-\hat x_r)^2.
$$

帽子表示抽样估计。这里没有精确算完所有 \(z\) 的平均，而是使用一次随机结果；重新抽样时，这个数可以变化。

如果每张图片抽取多个独立的 \(z\)，也可以把这些结果再平均。无论抽一次还是多次，都要区分理论期望和这次取得的估计值。[VAE 原论文 §2.3，式（5）至（8）](https://arxiv.org/html/1312.6114v11#S2.SS3)

## 6. KL 为什么可以直接用 μ 和 logvar 算？

当前编码分布是对角高斯，先验是标准正态。这两个分布之间的 KL 有解析公式：

$$
L_{\mathrm{KL}}(x)
=\frac12\sum_{j=1}^{k}
\left(\mu_j^2+\sigma_j^2-1-\log\sigma_j^2\right).
$$

\(j\) 遍历潜变量维度。这里的 \(\mu_j,\sigma_j\) 都属于当前图片的编码器输出。

将 \(\ell_j=\log\sigma_j^2\) 代入，得到实际计算的形式：

$$
\boxed{
L_{\mathrm{KL}}(x)
=\frac12\sum_{j=1}^{k}
\left(\mu_j^2+\exp(\ell_j)-1-\ell_j\right).
}
$$

不需要再抽一个 \(z\) 来估计这项 KL，直接代入分布参数就能精确计算。重建项使用抽样、KL 使用解析式，是两种不同的计算方式。[VAE 原论文附录 B](https://arxiv.org/html/1312.6114v11#A2)

{{< proof-open summary="可选验算：一维高斯 KL 的公式从哪里来？" >}}

固定图片，简写 \(q=\mathcal N(\mu,\sigma^2)\)、\(p=\mathcal N(0,1)\)。把两个高斯的对数密度相减：

$$
\log\frac{q(z)}{p(z)}
=-\log\sigma-\frac{(z-\mu)^2}{2\sigma^2}+\frac{z^2}{2}.
$$

按 \(q\) 求平均，使用 \(\mathbb E_q[(z-\mu)^2]=\sigma^2\) 和 \(\mathbb E_q[z^2]=\mu^2+\sigma^2\)：

$$
\begin{aligned}
D_{\mathrm{KL}}(q\|p)
&=-\log\sigma-\frac12+\frac12(\mu^2+\sigma^2)\\
&=\frac12(\mu^2+\sigma^2-1-\log\sigma^2).
\end{aligned}
$$

对角高斯的各维对数密度相加，所以再对 \(j=1,\ldots,k\) 求和，就是上面的多维公式。

{{< proof-close >}}

## 7. 把每张图片的两项相加，再平均 batch

用上标 \((i)\) 表示第 \(i\) 张图片。每张图片有自己的 \(\mu^{(i)}\)、\(\ell^{(i)}\)、\(z^{(i)}\) 和 \(\hat x^{(i)}\)。

单张图片的抽样损失为

$$\widehat L(x^{(i)})
=\widehat L_{\mathrm{rec}}(x^{(i)})+L_{\mathrm{KL}}(x^{(i)}).$$

对一批图片平均，最终交给反向传播的标量是

$$
\boxed{
\begin{aligned}
\widehat L_{\mathrm{batch}}
=\frac1B\sum_{i=1}^{B}\Bigg[
&\frac{1}{2\tau^2}\sum_{r=1}^{d}
\left(x_r^{(i)}-\hat x_r^{(i)}\right)^2\\
&+\frac12\sum_{j=1}^{k}
\left((\mu_j^{(i)})^2+\exp(\ell_j^{(i)})-1-\ell_j^{(i)}\right)
\Bigg].
\end{aligned}
}
$$

求和、平均的层次要分清：

1. 每张图片内部，对 \(d\) 个图片分量的重建误差求和
2. 每张图片内部，对 \(k\) 个潜变量维度的 KL 贡献求和
3. 把两项相加，得到每张图片的损失
4. 最后对 \(B\) 张图片求平均

如果重建项单独改成对所有像素取平均，却不相应调整 KL，它就额外缩小了 \(d\) 倍，改变了两项的相对权重。不能只看最后都叫 mean，就认为目标一样。

一次训练的路径现在完整了：

图片 → 编码器给出均值与对数方差 → 重参数化抽样 → 解码器给出重建均值 → 计算重建项与 KL → 相加并做 batch 平均 → 反向传播、更新两个网络。

## 8. 这次得到的数字，还一定是一个上界吗？

理论关系是

$$-\log p_\theta(x)\leq-\operatorname{ELBO}(x)=L(x).$$

这里的 \(L(x)\) 包含精确期望与完整常数。训练中计算的 \(\widehat L_{\mathrm{batch}}\) 是省略固定常数后、batch 平均负 ELBO 的抽样估计。

因此，不能要求每次抽样得到的这个数字都大于相应的负对数似然。下界或上界的保证属于理论期望；有限样本会波动，而省略常数也改变了数值。优化梯度不受固定加法常数影响，并不表示数值上的界也原样保留。

整个转换中有三种不同操作：

| 操作 | 性质 |
|---|---|
| 最大化 ELBO 改成最小化负 ELBO | 完全等价，只取负号 |
| 选定固定方差高斯解码分布 | 建模选择；在此设定下负对数密度精确化为平方误差加常数 |
| 省略固定加法常数、抽样估计期望 | 前者保持参数梯度，后者给出随机估计，不能混称精确 loss 数值 |

而最开始从最大化真实似然转向最大化 ELBO，仍然是在选择可计算的替代目标，不是无条件的等价替换。

## 9. 接回 DDPM：把潜变量换成整条路径

ELBO 并不要求潜变量一定是一个短向量，也不要求辅助分布由可训练编码器提供。

在通用形式中，观测是 \(x\)，潜变量是 \(z\)：

$$
\operatorname{ELBO}(x)
=\mathbb E_{z\sim q(z\mid x)}
\left[\log\frac{p_\theta(x,z)}{q(z\mid x)}\right].
$$

DDPM 的观测是原图 \(x_0\)。令 \(T\) 表示总加噪步数，\(x_1,\ldots,x_T\) 是中间状态；训练数据只提供原图，所以这些中间状态共同充当潜变量：

$$x\longleftrightarrow x_0,\qquad z\longleftrightarrow x_{1:T}=(x_1,\ldots,x_T).$$

这里的潜变量是整条路径，不只是最后的 \(x_T\)。

生成模型先抽噪声端的 \(x_T\)，再逐步生成到 \(x_0\)：

$$p_\theta(x_{0:T})
=p(x_T)\prod_{t=1}^{T}p_\theta(x_{t-1}\mid x_t).$$

\(x_{0:T}\) 表示从 \(x_0\) 到 \(x_T\) 的整组变量，\(p(x_T)\) 是生成先验，\(p_\theta(x_{t-1}\mid x_t)\) 是模型的一步反向分布。乘积沿同一条路径的各个生成步骤展开。

辅助分布选择固定的正向加噪过程：

$$q(x_{1:T}\mid x_0)=\prod_{t=1}^{T}q(x_t\mid x_{t-1}).$$

给定原图后，它告诉我们怎样抽取一整条加噪路径。代入同一个通用公式：

$$
\boxed{
\operatorname{ELBO}(x_0)
=\mathbb E_{x_{1:T}\sim q(x_{1:T}\mid x_0)}
\left[
\log\frac{p(x_T)\prod_{t=1}^{T}p_\theta(x_{t-1}\mid x_t)}
{\prod_{t=1}^{T}q(x_t\mid x_{t-1})}
\right].
}
$$

这里固定原图 \(x_0\)，期望平均的是这张图片可能产生的不同加噪路径。到这一步只是分布代入，还没有用噪声预测 MSE 替代任何目标。[DDPM 原论文 §2](https://arxiv.org/abs/2006.11239)

## 10. 辅助分布固定，为什么下界差距仍会变？

VAE 与当前讨论的 DDPM，可以这样对照：

| 对象 | VAE | 固定加噪日程的 DDPM |
|---|---|---|
| 观测 | 图片 \(x\) | 原图 \(x_0\) |
| 潜变量 | \(z\) | 整条路径 \(x_{1:T}\) |
| 辅助分布 | 编码器的 \(q_\phi(z\mid x)\) | 正向路径 \(q(x_{1:T}\mid x_0)\) |
| 辅助分布是否训练 | 是 | 否 |
| 学习的生成过程 | 从潜变量生成图片 | 从噪声逐步生成图片 |

所以 ELBO 需要的是合适的辅助分布，并不强制要求一个可学习编码器。

不过辅助分布固定，不代表下界差距固定。对应的差距恒等式仍然是

$$
\begin{aligned}
\log p_\theta(x_0)-\operatorname{ELBO}(x_0)
=D_{\mathrm{KL}}\!\left(
q(x_{1:T}\mid x_0)\|p_\theta(x_{1:T}\mid x_0)
\right).
\end{aligned}
$$

右边的第二个分布是生成模型给定 \(x_0\) 后的整条路径后验。更新反向生成模型参数 \(\theta\)，它也会改变，因此差距 KL 仍可能改变。

负 ELBO 如何继续分解成噪声端 KL、反向单步 KL 和重建项，以及怎样进一步得到噪声预测目标，已经在 [DDPM 笔记]({{< relref "posts/ddpm-derivation" >}}) 中完整推过，这里沿用那部分。

需要保留边界：路径分解是负 ELBO 的精确改写；在固定方差等设定下可继续化成带权误差；去掉时间步权重得到常见的简化噪声 MSE，是训练目标的调整，不能直接把它称作原始负 ELBO。

## 小结：三个连接现在闭合了

这组笔记解决的是三个问题：

1. 为什么需要 ELBO：潜变量的积分让数据似然难算，于是构造可处理的下界
2. 为什么能训练网络：同一个下界包含生成分布与辅助分布；固定生成模型时，优化辅助分布等价于减小后验 KL
3. 怎样得到实际 loss：取负号、选定解码分布、用抽样估计重建期望、解析计算 KL，最后平均 batch

VAE 与 DDPM 共享这个框架。一个用可学习编码器给出近似分布，一个使用固定的正向加噪路径；共同点是把潜变量模型的似然问题转成一个可训练的变分目标。

自检：logvar 怎样变成标准差？编码器方差与解码方差分别描述谁？为什么单次 loss 不必保持理论上界？DDPM 中作为潜变量的到底是最终噪声，还是整条路径？

## 参考资料

- Kingma, D. P. & Welling, M. [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)：§2.3–2.4、§3、附录 B
- Ho, J., Jain, A. & Abbeel, P. [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)：§2 的变分界，§3 的参数化与简化训练目标
- 已有文章：[从 AE 到 VAE]({{< relref "posts/ae-to-vae" >}})、[DDPM：从似然目标到噪声预测的完整推导]({{< relref "posts/ddpm-derivation" >}})
