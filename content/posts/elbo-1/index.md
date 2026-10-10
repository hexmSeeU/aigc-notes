{
  "title": "ELBO（一）：为什么生成模型需要一个似然下界？",
  "description": "从潜变量积分的困难出发，推到 Jensen 下界与模型后验差距。",
  "summary": "从潜变量积分的困难出发，推到 Jensen 下界与模型后验差距。",
  "date": "2026-10-08T14:52:00+08:00",
  "slug": "elbo-1",
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
  "episode": 1
}

一个生成模型已经能把潜变量变成图片，训练时为什么还需要 ELBO？问题出在另一个方向：给定一张真实图片，我们想知道模型认为它有多合理，却往往算不出这个分数。

这一篇从这个困难出发，走完“潜变量积分 → 引入辅助分布 → Jensen 不等式 → ELBO → 后验差距”的完整过程。先弄清每一步为什么出现，再把它接回 VAE 的编码器和解码器。

先记住名称：ELBO 是对数似然的下界；取负号后，才得到负对数似然的上界。

## 1. 训练生成模型，究竟想提高什么？

从 VAE 的生成过程开始。先抽取潜变量，再根据潜变量生成图片：

$$
z\sim p(z),\qquad x\sim p_\theta(x\mid z).
$$

这里每个符号各有一个角色：

- \(x\)：图片，后面固定为一张真实训练图片
- \(z\)：潜变量，可以是一个向量
- \(p(z)\)：潜变量的先验分布，例如标准正态分布
- \(\theta\)：生成模型的参数，在这里主要是解码器的网络权重
- \(p_\theta(x\mid z)\)：给定 \(z\) 后，模型给图片 \(x\) 的条件分布

解码器在概率模型里描述的是一个分布。网络可以输出这个分布的均值等参数，而不只是输出一个没有概率含义的向量。

现在拿来一张真实图片 \(x\)。我们希望调整 \(\theta\)，让模型更倾向于生成这样的图片，因此关心

$$p_\theta(x).$$

如果图片被建模为离散变量，这是概率；如果图片被建模为连续变量，这是概率密度，不是“恰好生成这个点的概率”。下面沿用密度的写法，离散时把积分换成求和即可。

## 2. 为什么一定要把各种 z 都考虑进去？

模型先抽 \(z\)，再生成 \(x\)。同一张图片可能对应很多潜变量，因此计算图片的总概率时，要把这些可能的来源合在一起。

先用一个只为理解而设的离散例子。假设潜变量只有两个取值：

$$p(z=1)=0.6,\qquad p(z=2)=0.4.$$

对一张固定的离散图片 \(x\)，模型给出

$$p_\theta(x\mid z=1)=0.1,\qquad p_\theta(x\mid z=2)=0.3.$$

于是

$$
\begin{aligned}
p_\theta(x)
&=p(z=1)p_\theta(x\mid z=1)
 +p(z=2)p_\theta(x\mid z=2)\\
&=0.6\times0.1+0.4\times0.3\\
&=0.18.
\end{aligned}
$$

每个乘积表示“先抽到这个 \(z\)，再生成这张 \(x\)”的联合概率。把两个乘积相加，才得到这张图片的总概率。

一般的离散情形是

$$p_\theta(x)=\sum_z p(z)p_\theta(x\mid z).$$

当 \(z\) 连续时，求和变成积分：

$$\boxed{p_\theta(x)=\int p(z)p_\theta(x\mid z)\,dz.}$$

这个操作叫把潜变量 \(z\) 边缘化：汇总各种潜变量的贡献，留下只关于图片的密度。

要看清积分时谁在变化：图片 \(x\) 固定，模型参数 \(\theta\) 暂时固定，遍历的是 \(z\)。这里没有在对不同训练图片取平均，也不能随便挑一个 \(z\) 的条件密度来代替整体。

## 3. 难点不是运行一次解码器，而是算完整个积分

给定一个具体的 \(z\)，运行解码器并计算 \(p_\theta(x\mid z)\) 往往可行。可是要把高维潜空间里所有 \(z\) 的贡献积起来，经过非线性神经网络后，通常没有容易计算的解析结果。

训练通常使用对数似然：

$$\log p_\theta(x)=\log\int p(z)p_\theta(x\mid z)\,dz.$$

\(\log\) 表示自然对数。它严格递增，所以对一张固定图片，最大化 \(p_\theta(x)\) 与最大化 \(\log p_\theta(x)\) 的方向相同。整个数据集的目标则汇总各张图片的对数似然。

取对数没有消除积分的困难。我们要找一个容易估计、容易优化的替代目标。这也是 VAE 原论文讨论的出发点。[VAE 原论文 §2.1](https://arxiv.org/html/1312.6114v11#S2.SS1)

## 4. 引入 q，只是先换一种写法

固定这张图片 \(x\)，选择一个方便抽样和计算密度的分布

$$q(z\mid x).$$

它告诉我们：看到这张图片后，准备怎样在潜空间里抽取 \(z\)。现在先把它看作我们选择的辅助分布；下一篇再让 VAE 的编码器提供它。

第一步，在积分里乘除同一个量：

$$
\begin{aligned}
p_\theta(x)
&=\int p(z)p_\theta(x\mid z)\,dz\\
&=\int q(z\mid x)
\frac{p(z)p_\theta(x\mid z)}{q(z\mid x)}\,dz.
\end{aligned}
$$

等式依赖一个条件：原被积函数有贡献的地方，\(q(z\mid x)\) 不能为零，否则会漏掉贡献。这篇假设 \(0<p_\theta(x)<\infty\)，且下文涉及的对数期望有限。后面采用正方差高斯编码分布与高斯解码分布时，两者均有全空间的正密度，满足所需的支持条件。

第二步，辨认期望。期望的定义是

$$
\mathbb E_{z\sim q(z\mid x)}[f(z)]
=\int q(z\mid x)f(z)\,dz.
$$

左边读作：按照 \(q(z\mid x)\) 抽取 \(z\)，对 \(f(z)\) 求平均。对照刚才的积分，括号里的函数就是那个比值：

$$
\boxed{
p_\theta(x)
=\mathbb E_{z\sim q(z\mid x)}
\left[\frac{p(z)p_\theta(x\mid z)}{q(z\mid x)}\right].
}
$$

为什么要除以 \(q\)？因为改用 \(q\) 决定哪些 \(z\) 更常被抽到之后，需要修正抽样频率。直接计算 \(\mathbb E_q[p(z)p_\theta(x\mid z)]\)，得到的就是另一个量。

到这里仍然没有近似，也没有改变 \(p_\theta(x)\)。原来难以直接计算的积分，被写成了另一种抽样分布下的平均，可以用抽样估计。

但训练目标还带一个对数：

$$
\log p_\theta(x)
=\log\mathbb E_{z\sim q(z\mid x)}
\left[\frac{p(z)p_\theta(x\mid z)}{q(z\mid x)}\right].
$$

此刻的顺序是先平均，再取对数。接下来的不等式正是从这个顺序产生的。

## 5. Jensen：先取对数再平均，会得到什么？

先看两个数字。一个正数随机量，一半概率取 \(1\)，一半概率取 \(9\)。

先平均，再取对数：

$$\log\frac{1+9}{2}=\log5\approx1.609.$$

先取对数，再平均：

$$\frac{\log1+\log9}{2}=\log3\approx1.099.$$

第二个结果更小。这不是巧合：\(\log\) 是凹函数，两点间的连线位于曲线下方。对应到加权平均，就是 Jensen 不等式

$$\boxed{\mathbb E[\log A]\leq\log\mathbb E[A],}$$

其中 \(A\) 是正的随机量，并假设这些期望有定义。

在我们的问题里，让 \(A\) 等于随 \(z\) 变化的密度比，直接代入：

$$
\begin{aligned}
\log p_\theta(x)
&=\log\mathbb E_{z\sim q(z\mid x)}
\left[\frac{p(z)p_\theta(x\mid z)}{q(z\mid x)}\right]\\
&\geq\mathbb E_{z\sim q(z\mid x)}
\left[\log\frac{p(z)p_\theta(x\mid z)}{q(z\mid x)}\right].
\end{aligned}
$$

右边定义为 ELBO：

$$
\boxed{
\operatorname{ELBO}(x)
:=\mathbb E_{z\sim q(z\mid x)}
\left[\log\frac{p(z)p_\theta(x\mid z)}{q(z\mid x)}\right].
}
$$

\(:=\) 表示“定义为”。为了简洁，名字里没有写出 \(\theta\) 和所选 \(q\)，但 ELBO 仍然依赖它们。

ELBO 是 Evidence Lower Bound 的缩写。Evidence 指观测数据的边缘似然 \(p_\theta(x)\)，这里得到的是它的对数下界：

$$\operatorname{ELBO}(x)\leq\log p_\theta(x).$$

把不等式两边取负号，方向反过来：

$$-\log p_\theta(x)\leq-\operatorname{ELBO}(x).$$

所以“对数似然的下界”和“负对数似然的上界”说的是同一关系的两种符号方向。

## 6. 下界为什么比原目标好处理？

把对数展开：

$$
\operatorname{ELBO}(x)
=\mathbb E_{z\sim q(z\mid x)}
\left[\log p_\theta(x\mid z)+\log p(z)-\log q(z\mid x)\right].
$$

给定一个抽到的 \(z\)，括号内三项通常都能计算。因此可以先计算这些对数，再通过抽样平均估计 ELBO，不需要在取对数前算完那个困难积分。

这给出了一个可以最大化的替代目标。可是下界和原目标仍有差距；提高下界，并不保证每一次都提高真实对数似然，因为差距本身也可能改变。接下来把它算清楚。

## 7. 先定义模型的精确后验

生成时是先有 \(z\)，再有 \(x\)。现在已经看到了图片 \(x\)，反过来问：按照当前生成模型，哪些 \(z\) 更可能是它的来源？

这个分布是模型后验 \(p_\theta(z\mid x)\)。贝叶斯公式给出

$$
p_\theta(z\mid x)
=\frac{p(z)p_\theta(x\mid z)}{p_\theta(x)}.
$$

分子同时考虑这个 \(z\) 原本有多常见，以及它对当前图片的解释能力；分母把所有 \(z\) 的贡献汇总，保证后验归一化。

这里“精确后验”或“真实后验”都是指当前模型确定的后验，不意味着模型已正确描述现实数据。

先验和解码分布一旦确定，后验就在数学上确定了，不需要额外一个网络来定义它；难点仍是分母 \(p_\theta(x)\)。我们选的 \(q(z\mid x)\) 则是另一个方便处理的分布，两者目前不一定相同。

## 8. ELBO 与对数似然到底差多少？

仍然固定 \(x\) 和 \(\theta\)，只对 \(z\sim q(z\mid x)\) 求平均。从两者的差开始：

$$
\log p_\theta(x)-\operatorname{ELBO}(x)
=\log p_\theta(x)
-\mathbb E_{z\sim q(z\mid x)}
\left[\log\frac{p(z)p_\theta(x\mid z)}{q(z\mid x)}\right].
$$

\(\log p_\theta(x)\) 不随 \(z\) 变化，取平均后还是它自己，因此可以放进同一个期望：

$$
\begin{aligned}
\log p_\theta(x)-\operatorname{ELBO}(x)
&=\mathbb E_{z\sim q(z\mid x)}
\left[\log p_\theta(x)-\log\frac{p(z)p_\theta(x\mid z)}{q(z\mid x)}\right]\\
&=\mathbb E_{z\sim q(z\mid x)}
\left[\log\frac{q(z\mid x)p_\theta(x)}{p(z)p_\theta(x\mid z)}\right]\\
&=\mathbb E_{z\sim q(z\mid x)}
\left[\log\frac{q(z\mid x)}{p_\theta(z\mid x)}\right].
\end{aligned}
$$

第二行使用 \(\log a-\log b=\log(a/b)\)，第三行代入贝叶斯公式。没有省略新的近似。

对于两个关于同一变量的分布，KL 散度定义为

$$
D_{\mathrm{KL}}(q\|p)
=\int q(z)\log\frac{q(z)}{p(z)}\,dz
=\mathbb E_{z\sim q}\left[\log\frac{q(z)}{p(z)}\right].
$$

顺序很重要：按照前面的 \(q\) 求平均，比较的比值也是 \(q/p\)，一般不能交换 \(q\) 和 \(p\)。KL 整体非负；一次抽样得到的对数比不必非负。

于是我们得到精确恒等式：

$$
\boxed{
\log p_\theta(x)
=\operatorname{ELBO}(x)
+D_{\mathrm{KL}}\!\left(q(z\mid x)\,\|\,p_\theta(z\mid x)\right).
}
$$

这个 KL 就是下界与目标之间的差距。当两个分布相同（忽略零测集）时，它为零，下界贴住对数似然。[VAE 原论文 §2.2，式（1）至（3）](https://arxiv.org/html/1312.6114v11#S2.SS2)

## 9. 固定模型，和同时训练模型，是两种情况

先固定 \(\theta\)。此时 \(p_\theta(x)\) 和模型后验都不变，只有我们选择的 \(q\) 能变。所以

$$
\text{提高 ELBO}
\quad\Longleftrightarrow\quad
\text{减小 }D_{\mathrm{KL}}(q(z\mid x)\|p_\theta(z\mid x)).
$$

这说明：优化一个能计算的下界，可以间接让辅助分布近似难算的模型后验。

如果同时更新 \(\theta\)，似然和后验也会变化。上面的恒等式仍然成立，但不能再把似然当作优化过程中的固定常数，也不能保证每一步的 ELBO 改善都对应真实似然改善。

还有一个后面会反复用到的区别：这里的差距 KL 比较的是辅助分布与模型后验。VAE 常见损失里的 KL 比较的却是编码器分布与先验。它们不是同一项，下一篇会把两者接起来。

## 小结与自检

这篇只有一个主线：潜变量使似然包含难算积分；辅助分布把积分改写成期望；Jensen 给出下界；后验 KL 则精确描述下界的差距。

1. 积分时固定的是谁，遍历的是谁？为什么一个 \(z\) 的条件密度不等于图片似然？
2. 引入 \(q\) 时为什么必须再除以它？这一阶段有没有改变原来的似然？
3. 为什么“先取 log 再平均”得到的是下界？取负号后方向怎样变化？
4. 固定生成模型时，调整 \(q\) 为什么只能改变下界差距，而不会改变图片似然？

### 参考答案

1. **固定图片与模型，遍历潜变量。** 计算 \(p_\theta(x)\) 时，\(x\)、\(\theta\) 和先验固定，积分变量是 \(z\)。一个 \(p_\theta(x\mid z)\) 只描述给定这一种潜变量时的图片密度；图片似然需要按先验权重汇总所有 \(z\) 的贡献，不能用某一个条件密度代替。
2. **乘除同一个 q，才能只改写而不改值。** 在本文的支持条件下，\(q(z\mid x)/q(z\mid x)=1\)，所以原积分和似然都不变，只是改写成按 \(q\) 求期望。若只乘不除，就改变了各处的权重；有贡献的地方也不能让 \(q\) 为零。
3. **log 是凹函数，所以先取 log 再平均不大于先平均再取 log。** 对正的随机变量

   $$W=p(z)p_\theta(x\mid z)/q(z\mid x)$$

   Jensen 不等式给出

   $$\mathbb E_q[\log W]\leq\log\mathbb E_q[W]=\log p_\theta(x)$$

   左边就是 ELBO；乘以负号后不等号反向，得到

   $$-\operatorname{ELBO}(x)\geq-\log p_\theta(x)$$

   这就是负对数似然的上界。
4. **q 不属于生成模型的似然定义。** 固定先验、解码器参数与图片后，\(p_\theta(x)\) 和模型后验 \(p_\theta(z\mid x)\) 已经确定。由恒等式

   $$\log p_\theta(x)=\operatorname{ELBO}(x)+D_{\mathrm{KL}}(q(z\mid x)\|p_\theta(z\mid x))$$

   可知，调整 \(q\) 只会改变 ELBO 与后验 KL 之间的分配；减小差距会抬高下界，不会改变这张图片的真实似然。

下一篇会把这些分布放回 VAE 的网络里，回答“ELBO 到底怎样训练编码器和解码器”。

## 参考资料

- Kingma, D. P. & Welling, M. [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)：§2.1 的问题设定，§2.2 的变分下界
- 前置笔记：[从 AE 到 VAE：让潜变量成为一个分布]({{< relref "posts/ae-to-vae" >}})
