{
  "title": "DDPM（三）：从似然到噪声预测",
  "description": "从图片似然接到可训练上界，再逐步推导噪声预测目标。",
  "summary": "从图片似然接到可训练上界，再逐步推导噪声预测目标。",
  "date": "2026-10-09T14:52:00+08:00",
  "slug": "ddpm-3",
  "categories": [
    "基础"
  ],
  "tags": [
    "DDPM",
    "扩散模型"
  ],
  "draft": false,
  "math": true,
  "collection": "ddpm",
  "episode": 3
}

[第二篇]({{< relref "posts/ddpm-2" >}})已经给出了一个能执行的训练任务：根据带噪图预测噪声。这一篇回答它背后的问题：我们明明想生成图片，为什么最后在做噪声平方误差？

先把关键逻辑接起来，再沿同一条路线逐行完成推导。

## 先说明什么叫学到了图片规律

生成时换一组随机数，结果可能不同。模型所有可能的结果共同定义了一个图片分布，记为 \(p_\theta(x_0)\)。训练希望这个分布接近真实图片的来源 \(q_{\mathrm{data}}\)。

一个自然做法是，让真实图片在模型下有较高的平均对数似然。固定一张观察到的图片，把模型给它的密度当作评价参数的函数，就叫似然。这里暂把像素当连续数值，所以“密度”不等于生成某一个精确数值点的概率。

\(\log\) 是自然对数，数值随正数增大而增大，还能把连乘变成连加。加上负号，就把“最大化平均对数似然”变成最小化

$$\mathbb E_{x_0\sim q_{\mathrm{data}}}[-\log p_\theta(x_0)].$$

## 难点在于一张图片可以对应很多生成路径

DDPM 能写出从噪声到图片的一整条路径的密度：起始噪声的密度，乘以每一步的条件密度。

但我们只观察到最终图片。要计算这张图片的密度，就必须汇总所有可能的中间路径，相当于对 \(x_1,\ldots,x_T\) 做一个很高维的积分。这个积分通常难以直接计算。

于是借用容易抽样的正向加噪过程，构造一个替代目标 \(L(x_0)\)。通过 Jensen 不等式，可以证明

$$-\log p_\theta(x_0)\leq L(x_0).$$

这里仍固定一张原图；\(L(x_0)\) 内部平均的是它产生的不同加噪路径。\(L\) 是负对数似然的上界，\(-L\) 才是对数似然的下界，也叫 ELBO。它有似然依据，但不与原目标数值相等。

## 整体目标可以拆成每一步的匹配任务

把路径密度用贝叶斯公式重写，并连续约分，\(L(x_0)\) 会精确拆成三部分：

- 噪声端：加噪最终状态与生成起点的分布是否匹配
- 中间步骤：每一步反向模型与训练参考分布是否匹配
- 图片端：最后一步对真实图片的重建

中间项使用 KL 来比较分布。KL 是对两个密度的对数比取平均，非负；为零时两个分布一致，但前后顺序通常不能互换。

这里 \(q\) 表示固定加噪过程产生的分布，\(p_\theta\) 表示学习的生成模型；条件分布中，竖线右边是已经知道的信息。某一个时间步的中间项是

$$
\mathbb E_{x_t\sim q(x_t\mid x_0)}
\left[
D_{\mathrm{KL}}\!\left(
q(x_{t-1}\mid x_t,x_0)
\,\|\,p_\theta(x_{t-1}\mid x_t)
\right)
\right].
$$

长式子只在说：给这张原图制造不同的 \(x_t\)，分别比较“训练参考的上一步分布”和“模型给出的上一步分布”，再平均。KL 内部还会平均上一步 \(x_{t-1}\)，所以它不能替代外面对 \(x_t\) 的平均。

## 训练知道原图，所以有一个能算的参考答案

只知道带噪状态 \(x_t\) 时，真实的上一步分布通常难算。训练时额外知道 \(x_0\)，就可以结合两段已知的加噪规则，精确算出

$$q(x_{t-1}\mid x_t,x_0)=
\mathcal N\!\left(x_{t-1};\tilde\mu_t(x_t,x_0),\tilde\beta_t I\right),
\qquad t\geq2.$$

分号后的两项分别是参考均值和协方差；这里 \(\tilde\beta_t I\) 表示各位置方差都是 \(\tilde\beta_t\)，彼此独立。模型也采用高斯反向一步，并先选定不随参数变化的方差，只让网络决定均值。

在这个前提下，两个高斯之间的 KL 可以化为：均值差的平方，乘一个已知系数，再加上与参数无关的项。于是“匹配分布”变成了“把均值预测好”。

给定原图的参考后验是高斯，不表示只给 \(x_t\) 的真实反向分布必然是高斯；后者可能是多个后验的混合。

## 用噪声来表达均值，训练就接上了

参考均值里含有原图。利用第二篇的直接加噪公式，可以把原图改写成 \(x_t\) 与噪声 \(\epsilon\) 的组合，再代回参考均值。

让网络预测这份噪声，就等于给反向均值选了一种新的表达方式。最终，对 \(t\geq2\)，单步高斯 KL 的含参数部分恰好是

$$w_t\|\epsilon-\epsilon_\theta(x_t,t)\|^2,
\qquad
w_t=\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar\alpha_t)}.$$

\(\sigma_t^2\) 是选定的模型反向方差，其余系数都来自正向日程。预测噪声因此直接决定生成时需要的反向均值，并不是一个与生成无关的附加任务。

还剩一个疑问：参考答案知道原图，模型不知道，这样训练有效吗？对所有原图和加噪结果平均后，参考后验的 KL 目标，与直接拟合真实反向分布的 KL 目标，只差一个不依赖参数的常数。因此，在相同模型族中，两者的优化是等价的。完整证明见下文 E 节。

## 第二篇的简单损失仍然多做了一次选择

上面得到的系数 \(w_t\) 随时间步变化。常用的 \(L_{\mathrm{simple}}\) 去掉了这些不同权重，让噪声平方误差直接参与训练。

这改变了各个噪声程度的相对训练重点，属于重新加权，不是仅仅删除常数。因此，简单损失不能直接声称是原始负对数似然的上界，也不与原始 ELBO 严格等价。

噪声端在日程和先验固定时不含模型参数；最后的 \(t=1\) 重建端则要单独处理，不能套入零方差的高斯 KL。原论文使用离散化高斯解码器，具体计算见下文 F 节。

**先把前面的逻辑串起来：似然目标给出训练依据，上界拆出单步分布匹配，固定方差把匹配变成均值误差，再用噪声表示均值，得到噪声预测。**

[第四篇]({{< relref "posts/ddpm-4" >}})把它重新放回实际生成流程，说明每一步怎么走、参数怎样赋值。

参考：[DDPM 原论文，第 2—3 节、式（1）至（14）及附录 A](https://arxiv.org/pdf/2006.11239)。

## 完整推导与边界 {#选读完整推导与边界}

接下来逐步完成推导。先固定正向日程与数据分布，模型反向方差采用固定方案；涉及非退化后验 KL 的推导均限定 \(t\geq2\)。证明保留所有期望下标，区分固定原图与数据平均。

为便于逐行核对，下文沿用完整推导的公式编号。前面已使用的三个起点集中列在这里：

$$q(x_t\mid x_{t-1})=\mathcal N(x_t;\sqrt{\alpha_t}x_{t-1},\beta_t I),
\qquad \alpha_t=1-\beta_t. \qquad (1)$$

$$q(x_t\mid x_0)=\mathcal N(x_t;\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I),
\qquad \bar\alpha_t=\prod_{s=1}^{t}\alpha_s. \qquad (2)$$

$$x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,
\qquad \epsilon\sim\mathcal N(0,I). \qquad (3)$$

下文 \(d\) 表示图片展开后的维度。\(\overline\alpha_t\) 与 \(\bar\alpha_t\) 是同一个累计系数。


## 完整推导的阅读路线与符号约定

### 阅读路线

DDPM 想解决的是：怎样让一个从高斯噪声出发的模型，生成符合真实数据分布的图片？推导的起点是图片的负对数似然，终点才是我们熟悉的噪声预测平方误差。

这份整理把已学内容接成一条完整的链。每一步都回答两个问题：**为什么要做这一步，以及它与前一步是什么关系。**

1. **规定正反向过程。** 正向逐步加噪，反向逐步生成；先弄清每个分布描述什么
2. **从似然到上界。** 图片密度涉及所有中间路径的积分，借助正向路径和 Jensen 不等式得到可训练的上界
3. **从整体到局部。** 用贝叶斯公式和连续约分，把上界精确拆成噪声端、反向单步 KL 和重建项
4. **算出监督答案。** 给定原图与当前状态，反向一步的后验是可解析计算的高斯分布
5. **从分布到预测误差。** 固定模型方差后，高斯 KL 化为均值平方误差；再重参数化为带权噪声平方误差
6. **解释训练与生成。** 区分数据平均与单图损失，证明原图只提供监督；最后整理简化目标、训练与采样

**最重要的边界。** Jensen 给出的是不等式；路径分解、后验计算和固定方差下的 KL 改写是精确等式；选择高斯反向模型是建模选择；去掉 timestep 权重得到简化训练目标，是目标的调整。

### 符号约定

| 符号 | 含义 |
|---|---|
| \(x_0\)，\(x_t\)，\(T\) | 原图、第 \(t\) 步的带噪状态、总步数；图片视为 \(d\) 维向量 |
| \(q_{\mathrm{data}}(x_0)\) | 真实数据分布；训练时由数据集抽样近似 |
| \(q\)，\(p_\theta\) | 固定的正向加噪过程，带参数 \(\theta\) 的生成模型 |
| \(\beta_t\)，\(\alpha_t\)，\(\overline{\alpha}_t\) | 单步噪声方差，\(\alpha_t=1-\beta_t\)，\(\overline{\alpha}_t=\prod_{s=1}^{t}\alpha_s\) |
| \(I\)，\(\epsilon\)，\(z\) | 单位矩阵、直接加噪时的标准噪声、反向采样时新抽的标准噪声 |
| \(\tilde\mu_t\)，\(\tilde\beta_t\) | 已知 \(x_0,x_t\) 时的真实后验均值与方差 |
| \(\mu_\theta\)，\(\sigma_t^2\) | 模型反向分布的均值与方差；\(\sigma_t\) 是标准差 |

**图片维数 d。** \(d\) 是一张图片展开后的标量总数：高为 \(H\)、宽为 \(W\)、通道数为 \(C\) 时，\(d=H\times W\times C\)。例如 \(32\times32\) 的 RGB 图片有 \(d=32\times32\times3=3072\) 个分量。它不是训练集中的图片数；\(I\) 是 \(d\times d\) 的单位矩阵。




## A 从图片似然到可计算上界

### 为什么从负对数似然走向路径上界

生成模型从简单分布 \(p(x_T)=\mathcal N(0,I)\) 出发，依次产生 \(x_{T-1},\ldots,x_0\)。它采用反向马尔可夫链：当前一步只依赖当前状态与步数。因此，整条生成路径的联合密度为

$$p_\theta(x_{0:T})=p(x_T)\prod_{t=1}^{T}p_\theta(x_{t-1}\mid x_t). \qquad (4)$$

这个乘积沿着**同一条路径的时间步**展开，不是在乘不同训练图片的密度。\(x_{0:T}\) 是 \(x_0,\ldots,x_T\) 的简写。

**原始目标。** 先固定一张真实图片 \(x_0\)，希望它在模型下具有较高密度，也就是最小化 \(-\log p_\theta(x_0)\)。但模型容易计算的是给定所有状态后的路径密度；要得到图片密度，必须把所有可能的中间状态积分掉：

$$p_\theta(x_0)=\int p_\theta(x_{0:T})\,dx_{1:T}. \qquad (5)$$

\(dx_{1:T}\) 表示依次对 \(x_1,\ldots,x_T\) 积分。这个高维路径积分通常无法直接算出，因此要寻找可计算的训练目标。

**引入会抽样的分布。** 给定 \(x_0\)，正向路径分布 \(q(x_{1:T}\mid x_0)\) 已知。乘除同一个密度，把积分改写为期望：

$$\begin{aligned}
-\log p_\theta(x_0)
&=-\log\int q(x_{1:T}\mid x_0)\frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}\,dx_{1:T}\\
&=-\log\mathbb E_{x_{1:T}\sim q(x_{1:T}\mid x_0)}
\left[\frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}\right].
\end{aligned}$$

**应用 Jensen。** 因为 \(-\log\) 是凸函数，“先平均再取负对数”不大于“先取负对数再平均”：

$$-\log p_\theta(x_0)\leq
\mathbb E_{x_{1:T}\sim q(x_{1:T}\mid x_0)}
\left[\log\frac{q(x_{1:T}\mid x_0)}{p_\theta(x_{0:T})}\right]
=:L(x_0). \qquad (6)$$

此刻只在平均**这一张原图的加噪路径**，并没有平均原图。\(L(x_0)\) 是负对数似然的上界；\(-L(x_0)\) 才是对数似然的下界，即 ELBO。

界的差距也可以精确写成

$$L(x_0)+\log p_\theta(x_0)
=D_{\mathrm{KL}}\!\left(q(x_{1:T}\mid x_0)\,\|\,p_\theta(x_{1:T}\mid x_0)\right)\geq0.$$

这里的整条路径 KL 描述上界松紧；下一节出现的单步 KL 则是训练目标中的局部项，两者用途不同。[原论文第 2 节及式（3）](https://arxiv.org/pdf/2006.11239)




## B 路径分解与每层期望

### 如何把正向路径改写成反向条件乘积

我们要把式（6）拆成可以逐步比较的项。模型已经沿反向时间分解，因此也把同一个正向路径密度改写成反向条件形式。注意：这只是概率密度的重写，没有把正向抽样过程改掉。

正向路径按定义为

$$q(x_{1:T}\mid x_0)=q(x_1\mid x_0)\prod_{t=2}^{T}q(x_t\mid x_{t-1}).$$

对 \(t\geq2\)，贝叶斯公式与正向马尔可夫性质给出

$$\begin{aligned}
q(x_{t-1}\mid x_t,x_0)
&=\frac{q(x_t\mid x_{t-1},x_0)q(x_{t-1}\mid x_0)}{q(x_t\mid x_0)}\\
&=\frac{q(x_t\mid x_{t-1})q(x_{t-1}\mid x_0)}{q(x_t\mid x_0)}.
\end{aligned} \qquad (7)$$

第二行使用了：已知 \(x_{t-1}\) 后，下一步加噪不再需要额外知道 \(x_0\)。移项得

$$q(x_t\mid x_{t-1})
=q(x_{t-1}\mid x_t,x_0)\frac{q(x_t\mid x_0)}{q(x_{t-1}\mid x_0)}.$$

代回整条正向路径：

$$\begin{aligned}
q(x_{1:T}\mid x_0)
={}&q(x_1\mid x_0)
\left[\prod_{t=2}^{T}q(x_{t-1}\mid x_t,x_0)\right]\\
&\cdot\left[\prod_{t=2}^{T}\frac{q(x_t\mid x_0)}{q(x_{t-1}\mid x_0)}\right].
\end{aligned}$$

后一串分式连续约分：

$$\frac{q(x_2\mid x_0)}{q(x_1\mid x_0)}
\frac{q(x_3\mid x_0)}{q(x_2\mid x_0)}\cdots
\frac{q(x_T\mid x_0)}{q(x_{T-1}\mid x_0)}
=\frac{q(x_T\mid x_0)}{q(x_1\mid x_0)}.$$

再与最前面的 \(q(x_1\mid x_0)\) 抵消，得到

$$q(x_{1:T}\mid x_0)=q(x_T\mid x_0)\prod_{t=2}^{T}q(x_{t-1}\mid x_t,x_0). \qquad (8)$$

用式（8）除以式（4），[再取对数](https://arxiv.org/pdf/2006.11239)，三类项精确出现：

$$\begin{aligned}
\log\frac{q(x_{1:T}\mid x_0)}{p_\theta(x_{0:T})}
={}&\log\frac{q(x_T\mid x_0)}{p(x_T)}\\
&+\sum_{t=2}^{T}\log\frac{q(x_{t-1}\mid x_t,x_0)}{p_\theta(x_{t-1}\mid x_t)}\\
&-\log p_\theta(x_0\mid x_1).
\end{aligned} \qquad (9)$$

### 为什么单步 KL 外面留下这个期望

仍然固定 \(x_0\)。式（9）的每一项原本都在整条路径期望里面。如果某个表达式不依赖其他状态，就把那些状态积分掉，只保留它实际依赖的变量。

**噪声端项。** 它只依赖 \(x_T\)，所以

$$\mathbb E_{x_{1:T}\sim q(x_{1:T}\mid x_0)}\left[\log\frac{q(x_T\mid x_0)}{p(x_T)}\right]
=D_{\mathrm{KL}}\!\left(q(x_T\mid x_0)\,\|\,p(x_T)\right).$$

对 \(x_T\) 的平均已包含在 KL 的定义中，外面不需要再加一次。

**中间项。** 固定某个 \(t\geq2\)，它同时涉及 \(x_t\) 和 \(x_{t-1}\)。把其他状态积分掉后，留下的联合分布是

$$q(x_{t-1},x_t\mid x_0)=q(x_t\mid x_0)q(x_{t-1}\mid x_t,x_0).$$

所以原来的路径平均恰好变成两层：

$$\begin{aligned}
&\mathbb E_{x_{1:T}\sim q(x_{1:T}\mid x_0)}
\left[\log\frac{q(x_{t-1}\mid x_t,x_0)}{p_\theta(x_{t-1}\mid x_t)}\right]\\
&=\mathbb E_{x_t\sim q(x_t\mid x_0)}
\left[\mathbb E_{x_{t-1}\sim q(x_{t-1}\mid x_t,x_0)}
\left[\log\frac{q(x_{t-1}\mid x_t,x_0)}{p_\theta(x_{t-1}\mid x_t)}\right]\right]\\
&=\mathbb E_{x_t\sim q(x_t\mid x_0)}
\left[D_{\mathrm{KL}}\!\left(q(x_{t-1}\mid x_t,x_0)\,\|\,p_\theta(x_{t-1}\mid x_t)\right)\right].
\end{aligned}$$

**内层对 \(x_{t-1}\) 的平均写进了 KL；外层仍然平均这张原图产生的不同 \(x_t\)。** 因而外层下标必须是 \(q(x_t\mid x_0)\)。

**重建项。** 固定原图后，只剩 \(x_1\) 随机，故为 \(\mathbb E_{x_1\sim q(x_1\mid x_0)}[-\log p_\theta(x_0\mid x_1)]\)。合并后得到

$$\begin{aligned}
L(x_0)={}&D_{\mathrm{KL}}\!\left(q(x_T\mid x_0)\,\|\,p(x_T)\right)\\
&+\sum_{t=2}^{T}\mathbb E_{x_t\sim q(x_t\mid x_0)}
\left[D_{\mathrm{KL}}\!\left(q(x_{t-1}\mid x_t,x_0)\,\|\,p_\theta(x_{t-1}\mid x_t)\right)\right]\\
&+\mathbb E_{x_1\sim q(x_1\mid x_0)}[-\log p_\theta(x_0\mid x_1)].
\end{aligned} \qquad (10)$$

到这里，从式（6）往后的分解全部是等式。真正训练整个数据集时，才额外在外面平均原图：

$$\mathbb E_{x_0\sim q_{\mathrm{data}}}[-\log p_\theta(x_0)]
\leq\mathbb E_{x_0\sim q_{\mathrm{data}}}[L(x_0)]
=:L_{\mathrm{VLB}}(\theta). \qquad (11)$$

[原论文式（3）（5）](https://arxiv.org/pdf/2006.11239)




## C 参考后验与完整配方

### 为什么给定原图的反向后验能够精确求出

式（10）的中间项要求比较两个反向分布。先分清它们：

- \(q(x_{t-1}\mid x_t)\) 是我们希望学到的真实反向条件分布，通常难以直接计算，也不保证严格是高斯
- \(q(x_{t-1}\mid x_t,x_0)\) 额外知道原图，是训练时能精确计算的参考后验
- \(p_\theta(x_{t-1}\mid x_t)\) 是模型拟合的反向分布，模型并不以原图为输入

给定不同原图得到的高斯后验混合后，可以成为多峰分布。因此，“给定 \(x_0\) 后是高斯”不能推出“不知道 \(x_0\) 时仍是高斯”。下面的解析推导始终包含条件 \(x_0\)，且只处理 \(t\geq2\)。

从式（7）的贝叶斯公式开始。对 \(x_{t-1}\) 而言，分母 \(q(x_t\mid x_0)\) 是常数，于是

$$q(x_{t-1}\mid x_t,x_0)\propto q(x_t\mid x_{t-1})q(x_{t-1}\mid x_0).$$

右边两个分布已知：

$$\begin{aligned}
q(x_t\mid x_{t-1})&=\mathcal N\!\left(x_t;\sqrt{\alpha_t}x_{t-1},\beta_t I\right),\\
q(x_{t-1}\mid x_0)&=\mathcal N\!\left(x_{t-1};\sqrt{\overline{\alpha}_{t-1}}x_0,(1-\overline{\alpha}_{t-1})I\right).
\end{aligned}$$

保留含 \(x_{t-1}\) 的指数部分：

$$q(x_{t-1}\mid x_t,x_0)\propto
\exp\!\left[-\frac12\left(
\frac{\|x_t-\sqrt{\alpha_t}x_{t-1}\|^2}{\beta_t}
+\frac{\|x_{t-1}-\sqrt{\overline{\alpha}_{t-1}}x_0\|^2}{1-\overline{\alpha}_{t-1}}
\right)\right].$$

把平方展开，按 \(x_{t-1}\) 的二次项和一次项整理：

$$\begin{aligned}
&\frac{\|x_t-\sqrt{\alpha_t}x_{t-1}\|^2}{\beta_t}
+\frac{\|x_{t-1}-\sqrt{\overline{\alpha}_{t-1}}x_0\|^2}{1-\overline{\alpha}_{t-1}}\\
={}&\left(\frac{\alpha_t}{\beta_t}+\frac{1}{1-\overline{\alpha}_{t-1}}\right)\|x_{t-1}\|^2\\
&-2\left(\frac{\sqrt{\alpha_t}}{\beta_t}x_t+
\frac{\sqrt{\overline{\alpha}_{t-1}}}{1-\overline{\alpha}_{t-1}}x_0\right)^{\!\top}x_{t-1}
+\text{与 }x_{t-1}\text{ 无关的项}.
\end{aligned}$$

这里没有把变量换成其他字母。目标是把上式配成 \(\|x_{t-1}-\tilde\mu_t\|^2/\tilde\beta_t\) 加常数，再直接读出后验均值和方差。

“\(\propto\)”只省略了**对正在讨论的变量 \(x_{t-1}\) 为常数**的因子，它们仍可依赖已经给定的 \(x_t,x_0\)。完成配方后还要恢复归一化，不能把正比直接当成相等。[原论文式（6）（7）](https://arxiv.org/pdf/2006.11239)

### 配方读出后验均值和方差

提取指数中的公共因子 \(-1/2\) 后，括号内 \(\|x_{t-1}\|^2\) 的系数是方差的倒数。沿用上一节的展开式，先化简这个系数：

$$\begin{aligned}
\frac1{\tilde\beta_t}
&=\frac{\alpha_t}{\beta_t}+\frac1{1-\overline{\alpha}_{t-1}}\\
&=\frac{\alpha_t(1-\overline{\alpha}_{t-1})+\beta_t}
{\beta_t(1-\overline{\alpha}_{t-1})}\\
&=\frac{1-\overline{\alpha}_t}{\beta_t(1-\overline{\alpha}_{t-1})}.
\end{aligned}$$

最后一步用了 \(\alpha_t+\beta_t=1\) 及 \(\alpha_t\overline{\alpha}_{t-1}=\overline{\alpha}_t\)，所以

$$\tilde\beta_t=\frac{1-\overline{\alpha}_{t-1}}{1-\overline{\alpha}_t}\beta_t. \qquad (12)$$

再比较一次项。因为

$$\frac{\|x_{t-1}-\tilde\mu_t\|^2}{\tilde\beta_t}
=\frac{\|x_{t-1}\|^2}{\tilde\beta_t}
-2\left(\frac{\tilde\mu_t}{\tilde\beta_t}\right)^{\!\top}x_{t-1}
+\frac{\|\tilde\mu_t\|^2}{\tilde\beta_t},$$

应有

$$\frac{\tilde\mu_t}{\tilde\beta_t}
=\frac{\sqrt{\alpha_t}}{\beta_t}x_t+
\frac{\sqrt{\overline{\alpha}_{t-1}}}{1-\overline{\alpha}_{t-1}}x_0.$$

乘上式（12）后，分别约掉两项中的公共因子：

$$\tilde\mu_t(x_t,x_0)
=\frac{\sqrt{\overline{\alpha}_{t-1}}\beta_t}{1-\overline{\alpha}_t}x_0
+\frac{\sqrt{\alpha_t}(1-\overline{\alpha}_{t-1})}{1-\overline{\alpha}_t}x_t. \qquad (13)$$

因此后验的形状已经确定。恢复高斯归一化常数，完整密度为

$$q(x_{t-1}\mid x_t,x_0)
=\frac1{(2\pi\tilde\beta_t)^{d/2}}
\exp\!\left[-\frac{\|x_{t-1}-\tilde\mu_t(x_t,x_0)\|^2}{2\tilde\beta_t}\right].$$

也就是

$$q(x_{t-1}\mid x_t,x_0)=\mathcal N\!\left(x_{t-1};\tilde\mu_t(x_t,x_0),\tilde\beta_t I\right). \qquad (14)$$

**系数的含义。** \(\tilde\mu_t\) 同时利用原图 \(x_0\) 与当前状态 \(x_t\)，给出上一步状态的条件均值；\(\tilde\beta_t\) 表示已知这两者后仍剩下的不确定性。它是后验方差，通常不同于正向一步的噪声方差 \(\beta_t\)。

到这里没有学习任何参数，式（12）至（14）全由已固定的加噪规则解析推导而来。它们的用途是为反向模型提供可计算的训练监督。[原论文式（6）（7）](https://arxiv.org/pdf/2006.11239)与[博客 DDPM 推导](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)




## D 高斯 KL 与噪声参数化

### 高斯 KL 怎样变成均值平方误差

现在规定模型的反向一步为

$$p_\theta(x_{t-1}\mid x_t)=\mathcal N\!\left(x_{t-1};\mu_\theta(x_t,t),\sigma_t^2I\right). \qquad (15)$$

这是**模型族的选择**，并不说明真实的 \(q(x_{t-1}\mid x_t)\) 必然高斯。接下来采用固定的、与 \(\theta\) 无关的 \(\sigma_t^2>0\)，只学习均值；原论文讨论了 \(t\geq2\) 时取 \(\beta_t\) 或 \(\tilde\beta_t\) 的设置。

固定 \(x_0,x_t,t\)，按 KL 定义展开式（14）与式（15）：

$$\begin{aligned}
D_{\mathrm{KL}}(q\|p_\theta)
={}&\frac d2\log\frac{\sigma_t^2}{\tilde\beta_t}
-\frac1{2\tilde\beta_t}\mathbb E_{x_{t-1}\sim q(x_{t-1}\mid x_t,x_0)}\|x_{t-1}-\tilde\mu_t\|^2\\
&+\frac1{2\sigma_t^2}\mathbb E_{x_{t-1}\sim q(x_{t-1}\mid x_t,x_0)}\|x_{t-1}-\mu_\theta\|^2.
\end{aligned}$$

上面的期望始终固定 \(x_t,x_0\)，只平均后验中的 \(x_{t-1}\)。这个高斯每一维的方差都是 \(\tilde\beta_t\)，所以

$$\mathbb E_{x_{t-1}\sim q(x_{t-1}\mid x_t,x_0)}\|x_{t-1}-\tilde\mu_t\|^2=d\tilde\beta_t.$$

对最后一项，在平方里加减真实均值：

$$\begin{aligned}
\mathbb E_{x_{t-1}\sim q(x_{t-1}\mid x_t,x_0)}\|x_{t-1}-\mu_\theta\|^2
={}&\mathbb E_{x_{t-1}\sim q(x_{t-1}\mid x_t,x_0)}\|(x_{t-1}-\tilde\mu_t)+(\tilde\mu_t-\mu_\theta)\|^2\\
={}&d\tilde\beta_t+\|\tilde\mu_t-\mu_\theta\|^2.
\end{aligned}$$

交叉项为零，因为 \(\mathbb E_{x_{t-1}\sim q(x_{t-1}\mid x_t,x_0)}[x_{t-1}-\tilde\mu_t]=0\)。代回得到完整结果：

$$\begin{aligned}
D_{\mathrm{KL}}(q\|p_\theta)
={}&\frac1{2\sigma_t^2}\|\tilde\mu_t-\mu_\theta\|^2\\
&+\frac d2\left(\frac{\tilde\beta_t}{\sigma_t^2}-1+
\log\frac{\sigma_t^2}{\tilde\beta_t}\right).
\end{aligned} \qquad (16)$$

第二行只由固定方差决定，不依赖 \(\theta\)。因此，优化这一项 KL 等价于优化第一行的**带权均值平方误差**。若 \(\sigma_t^2=\tilde\beta_t\)，第二行恰好为零；若取别的固定方差，它仍然不能凭空从数值等式里消失，只能在优化时忽略。

这个等价关系依赖“方差固定”的前提。若方差也由网络学习，就不能把所有方差项当成与参数无关的常数。[原论文第 3.2 节及式（8）](https://arxiv.org/pdf/2006.11239)

### 为什么预测噪声就能确定反向均值

真实后验均值式（13）含有 \(x_0\)。训练时知道 \(x_0\)，但生成时不知道。我们把它改写为能由网络预测的量。

从直接加噪式（3）解出原图：

$$x_0=\frac{x_t-\sqrt{1-\overline{\alpha}_t}\epsilon}{\sqrt{\overline{\alpha}_t}}.$$

代入式（13）：

$$\begin{aligned}
\tilde\mu_t
={}&\frac{\sqrt{\overline{\alpha}_{t-1}}\beta_t}{1-\overline{\alpha}_t}
\frac{x_t-\sqrt{1-\overline{\alpha}_t}\epsilon}{\sqrt{\overline{\alpha}_t}}\\
&+\frac{\sqrt{\alpha_t}(1-\overline{\alpha}_{t-1})}{1-\overline{\alpha}_t}x_t.
\end{aligned}$$

利用 \(\overline{\alpha}_t=\alpha_t\overline{\alpha}_{t-1}\)，第一项中的平方根比值变为 \(1/\sqrt{\alpha_t}\)。两个 \(x_t\) 系数相加：

$$\begin{aligned}
&\frac{\beta_t}{\sqrt{\alpha_t}(1-\overline{\alpha}_t)}
+\frac{\sqrt{\alpha_t}(1-\overline{\alpha}_{t-1})}{1-\overline{\alpha}_t}\\
&=\frac{\beta_t+\alpha_t(1-\overline{\alpha}_{t-1})}{\sqrt{\alpha_t}(1-\overline{\alpha}_t)}
=\frac1{\sqrt{\alpha_t}}.
\end{aligned}$$

噪声系数则化为 \(-\beta_t/[\sqrt{\alpha_t}\sqrt{1-\overline{\alpha}_t}]\)，因此

$$\tilde\mu_t(x_t,x_0)
=\frac1{\sqrt{\alpha_t}}
\left(x_t-\frac{\beta_t}{\sqrt{1-\overline{\alpha}_t}}\epsilon\right). \qquad (17)$$

**定义网络接口。** 让同一个网络输入 \(x_t,t\)，输出与图片同形状的噪声估计 \(\epsilon_\theta(x_t,t)\)；再把模型均值定义为

$$\mu_\theta(x_t,t)
=\frac1{\sqrt{\alpha_t}}
\left(x_t-\frac{\beta_t}{\sqrt{1-\overline{\alpha}_t}}\epsilon_\theta(x_t,t)\right). \qquad (18)$$

这是均值的重参数化。并不是生成时能拿到真实 \(\epsilon\)，而是用网络预测值构造反向均值。

相减并平方：

$$\|\tilde\mu_t-\mu_\theta\|^2
=\frac{\beta_t^2}{\alpha_t(1-\overline{\alpha}_t)}
\|\epsilon-\epsilon_\theta(x_t,t)\|^2.$$

代回高斯 KL，得到

$$D_{\mathrm{KL}}(q\|p_\theta)
=\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\overline{\alpha}_t)}
\|\epsilon-\epsilon_\theta(x_t,t)\|^2+\text{与 }\theta\text{ 无关的项}. \qquad (19)$$

因此，噪声预测误差来自整体似然上界中的局部 KL，并不是另加的无关任务。训练标签 \(\epsilon\) 是我们自己抽取的，天然已知。平方误差使网络学习给定当前输入后的平均噪声估计，而不是识别某次唯一的历史噪声。[原论文式（11）（12）](https://arxiv.org/pdf/2006.11239)




## E 两个训练目标的完整常数等价证明

### 训练知道原图 为什么模型仍只需要带噪输入

要证明的本质是：**对原图与加噪结果平均后的参考后验 KL，与拟合真实反向条件分布的 KL，只差一个不依赖模型参数的量。** 差常数的是两个损失目标，不是两个概率密度。

固定某个 \(t\geq2\)，并固定数据分布与正向加噪规则。以下统一用密度和积分书写；对于离散的数据集，外层关于原图的积分对应求和。先给单图损失加上数据平均，定义

$$\begin{aligned}
J_t(\theta)
=\mathbb E_{x_0\sim q_{\mathrm{data}}}
\mathbb E_{x_t\sim q(x_t\mid x_0)}
\Big[D_{\mathrm{KL}}\!\left(q(x_{t-1}\mid x_t,x_0)\,\|\,p_\theta(x_{t-1}\mid x_t)\right)\Big].
\end{aligned}$$

内层平均同一原图的加噪结果，外层平均不同原图。按照 KL 的定义，把它内部对 \(x_{t-1}\) 的期望也明确写出来：

$$\begin{aligned}
J_t(\theta)
={}&\mathbb E_{x_0\sim q_{\mathrm{data}}}
\mathbb E_{x_t\sim q(x_t\mid x_0)}
\Bigg[\mathbb E_{x_{t-1}\sim q(x_{t-1}\mid x_t,x_0)}\\
&\qquad\qquad\left[\log q(x_{t-1}\mid x_t,x_0)
-\log p_\theta(x_{t-1}\mid x_t)\right]\Bigg].
\end{aligned}$$

这三层平均展开为积分，就是

$$\begin{aligned}
J_t(\theta)=\iiint &q_{\mathrm{data}}(x_0)q(x_t\mid x_0)q(x_{t-1}\mid x_t,x_0)\\
&\cdot\left[\log q(x_{t-1}\mid x_t,x_0)
-\log p_\theta(x_{t-1}\mid x_t)\right]
\,dx_{t-1}\,dx_t\,dx_0.
\end{aligned}$$

这里的三个密度相乘，正好形成同一个联合分布：

$$q_{\mathrm{data}}(x_0)q(x_t\mid x_0)q(x_{t-1}\mid x_t,x_0)
=q(x_0,x_t,x_{t-1}).$$

先把不含参数的部分完整定义出来：

$$C_{1,t}=\iiint q(x_0,x_t,x_{t-1})
\log q(x_{t-1}\mid x_t,x_0)\,dx_{t-1}\,dx_t\,dx_0.$$

\(C_{1,t}\) 是上面**整个三重积分**，不是其中单独一个 \(\log q\)。它不依赖 \(\theta\)，是因为 \(q_{\mathrm{data}}\) 与正向加噪规则已经固定。把它从式子中分出后，所有与模型参数有关的内容集中在

$$J_t(\theta)=C_{1,t}-\iiint q(x_0,x_t,x_{t-1})
\log p_\theta(x_{t-1}\mid x_t)\,dx_{t-1}\,dx_t\,dx_0.$$

接下来只需把这个模型项中的 \(x_0\) 边缘化，整个转换的关键就发生在这一步。

### 先积分原图 再识别真实反向 KL

模型的 \(\log p_\theta(x_{t-1}\mid x_t)\) 不以 \(x_0\) 为自变量，因此可以交换积分次序，把关于 \(x_0\) 的积分放进括号：

$$\begin{aligned}
J_t(\theta)=C_{1,t}-\iint
\left[\int q(x_0,x_t,x_{t-1})\,dx_0\right]
\log p_\theta(x_{t-1}\mid x_t)\,dx_{t-1}\,dx_t.
\end{aligned}$$

括号里就是边缘化的定义，随后再做一次联合分布分解：

$$\int q(x_0,x_t,x_{t-1})\,dx_0
=q(x_t,x_{t-1})
=q(x_t)q(x_{t-1}\mid x_t).$$

因此，原目标变成

$$\begin{aligned}
J_t(\theta)=C_{1,t}
-\iint q(x_t)q(x_{t-1}\mid x_t)
\log p_\theta(x_{t-1}\mid x_t)\,dx_{t-1}\,dx_t.
\end{aligned}$$

也可以把剩下的两层平均完整写回期望：

$$J_t(\theta)=C_{1,t}
-\mathbb E_{x_t\sim q(x_t)}
\left[\mathbb E_{x_{t-1}\sim q(x_{t-1}\mid x_t)}
[\log p_\theta(x_{t-1}\mid x_t)]\right].$$

现在另写一个目标：直接让模型拟合真实反向条件分布，记为

$$K_t(\theta)=\mathbb E_{x_t\sim q(x_t)}
\Big[D_{\mathrm{KL}}\!\left(q(x_{t-1}\mid x_t)\,\|\,p_\theta(x_{t-1}\mid x_t)\right)\Big].$$

把这个 KL 展开，并定义它的不含参数部分：

$$\begin{aligned}
C_{2,t}&=\iint q(x_t)q(x_{t-1}\mid x_t)
\log q(x_{t-1}\mid x_t)\,dx_{t-1}\,dx_t,\\
K_t(\theta)&=C_{2,t}-\iint q(x_t)q(x_{t-1}\mid x_t)
\log p_\theta(x_{t-1}\mid x_t)\,dx_{t-1}\,dx_t.
\end{aligned}$$

现在可以逐项比较：\(J_t\) 与 \(K_t\) 的模型项完全相同，唯一差别是两个不含参数的项。因此

$$J_t(\theta)=K_t(\theta)+C_t,\qquad C_t=C_{1,t}-C_{2,t}. \qquad (20)$$

这里没有假设模型已经训练成功，也没有假设真实反向条件是高斯。恒等式来自对同一联合分布的展开、边缘化与重新分解。

### 这个常数等价结论究竟说明什么

把式（20）中的两个目标完整展开，得到

$$\begin{aligned}
&\mathbb E_{x_0\sim q_{\mathrm{data}}}\mathbb E_{x_t\sim q(x_t\mid x_0)}
\Big[D_{\mathrm{KL}}\!\left(q(x_{t-1}\mid x_t,x_0)\,\|\,p_\theta(x_{t-1}\mid x_t)\right)\Big]\\
&=\mathbb E_{x_t\sim q(x_t)}
\Big[D_{\mathrm{KL}}\!\left(q(x_{t-1}\mid x_t)\,\|\,p_\theta(x_{t-1}\mid x_t)\right)\Big]+C_t.
\end{aligned}$$

为了看清 \(C_t\)，先把 \(C_{2,t}\) 也写回同一个三重积分。利用上一节的边缘化等式，

$$\begin{aligned}
C_{2,t}
&=\iint\left[\int q(x_0,x_t,x_{t-1})\,dx_0\right]
\log q(x_{t-1}\mid x_t)\,dx_{t-1}\,dx_t\\
&=\iiint q(x_0,x_t,x_{t-1})\log q(x_{t-1}\mid x_t)
\,dx_{t-1}\,dx_t\,dx_0.
\end{aligned}$$

所以相减时只需合并两个对数，得到

$$C_t=\iiint q(x_0,x_t,x_{t-1})
\log\frac{q(x_{t-1}\mid x_t,x_0)}{q(x_{t-1}\mid x_t)}
\,dx_{t-1}\,dx_t\,dx_0.$$

再把联合密度重新分成三个条件因子，把内层对 \(x_{t-1}\) 的积分辨认为 KL：

$$\begin{aligned}
C_t=\iint &q_{\mathrm{data}}(x_0)q(x_t\mid x_0)\\
&\cdot\left[\int q(x_{t-1}\mid x_t,x_0)
\log\frac{q(x_{t-1}\mid x_t,x_0)}{q(x_{t-1}\mid x_t)}\,dx_{t-1}\right]\,dx_t\,dx_0.
\end{aligned}$$

因此

$$\begin{aligned}
C_t=\mathbb E_{x_0\sim q_{\mathrm{data}}}
\mathbb E_{x_t\sim q(x_t\mid x_0)}
\Big[D_{\mathrm{KL}}\!\left(q(x_{t-1}\mid x_t,x_0)\,\|\,q(x_{t-1}\mid x_t)\right)\Big]\geq0.
\end{aligned}$$

### 用参考后验训练与拟合真实反向分布等价

这些表达式都只含 \(q\)。在数据与加噪规则固定时，\(C_t\) 与 \(\theta\) 无关，所以**在相同模型族中，优化 \(J_t\) 与优化 \(K_t\) 等价**。这证明训练时利用原图构造参考后验是合理的。它不保证一个受限的高斯模型能精确表示真实反向分布，也不保证从当前带噪输入唯一还原某一张历史原图或那次噪声。

**边缘化掉的是 \(x_0\)，仍然保留条件 \(x_t\)。** 最终拟合的是 \(q(x_{t-1}\mid x_t)\)，不是无条件的 \(q(x_{t-1})\)。真实反向分布可写成

$$q(x_{t-1}\mid x_t)
=\int q(x_{t-1}\mid x_t,x_0)q(x_0\mid x_t)\,dx_0.$$

它可能是许多高斯后验的混合。训练无需显式求出这个混合密度；使用已知原图提供的参考后验来平均训练，就已在优化同一项含参数的部分。

不能在单图 \(L(x_0)\) 里凭空加入对 \(q(x_0\mid x_t)\) 的平均。必须先进行数据平均，再利用

$$q_{\mathrm{data}}(x_0)q(x_t\mid x_0)=q(x_t)q(x_0\mid x_t)$$

交换平均次序。否则就把“原图固定”与“原图随机”混成了一个问题。

**模型接口始终不变。** 原图 \(x_0\) 用来构造 \(x_t\) 与监督标签；网络始终只看到 \(x_t,t\)。生成时沿用相同接口。因此并不存在训练时把原图喂给网络、生成时再撤掉原图的步骤。本节证明限于 \(t\geq2\)，连续变量下的退化端点 \(t=1\) 要单独处理。




## F 简化损失与两端项的完整处理

### 简化噪声损失与原始上界差在哪里

式（19）对 \(t\geq2\) 给出了 timestep 权重

$$w_t=\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\overline{\alpha}_t)}.$$

因此，原始上界中间部分的含参数项是

$$\sum_{t=2}^{T}\mathbb E_{x_0\sim q_{\mathrm{data}}}
\mathbb E_{\epsilon\sim\mathcal N(0,I)}
[w_t\|\epsilon-\epsilon_\theta(x_t,t)\|^2].$$

其中 \(x_t\) 由式（3）构造。

**噪声端。** 当正向日程和先验固定时，\(D_{\mathrm{KL}}(q(x_T\mid x_0)\|p(x_T))\) 不含 \(\theta\)，可以在优化参数时忽略。它仍是似然上界的一部分，不等于数值上消失。

**重建端。** 由约定 \(\overline{\alpha}_0=1\)，式（12）形式上给出 \(\tilde\beta_1=0\)。知道原图后，\(x_0\) 本来就是已固定的值；不能套用要求正方差的高斯 KL。正确项一直是

$$L_0(x_0)=\mathbb E_{x_1\sim q(x_1\mid x_0)}[-\log p_\theta(x_0\mid x_1)].$$

若在连续高斯解码器下取固定正方差 \(\sigma_1^2\)，则负对数密度是 \(\|x_0-\mu_\theta(x_1,1)\|^2/(2\sigma_1^2)\) 加常数。又因为 \(1-\overline{\alpha}_1=\beta_1\)，

$$\begin{aligned}
\mu_\theta(x_1,1)&=\frac1{\sqrt{\alpha_1}}
\left(x_1-\sqrt{\beta_1}\epsilon_\theta(x_1,1)\right),\\
\|x_0-\mu_\theta(x_1,1)\|^2
&=\frac{\beta_1}{\alpha_1}\|\epsilon-\epsilon_\theta(x_1,1)\|^2.
\end{aligned}$$

固定 \(x_0\)，把加噪平均也写回去，就得到

$$\begin{aligned}
L_0(x_0)
={}&\mathbb E_{\epsilon\sim\mathcal N(0,I)}\Bigg[
\frac{\beta_1}{2\sigma_1^2\alpha_1}\\
&\quad\cdot\left\|\epsilon-\epsilon_\theta\!\left(
\sqrt{\alpha_1}x_0+\sqrt{\beta_1}\epsilon,1\right)\right\|^2\Bigg]
+\frac d2\log(2\pi\sigma_1^2).
\end{aligned}$$

因此，在这个连续设定下，单图重建损失精确等于带权噪声平方误差的平均加常数；若讨论整个数据集，还要对 \(x_0\sim q_{\mathrm{data}}\) 再平均。原论文实际用像素区间上的离散化高斯解码器；把它统一写为 \(t=1\) 的噪声平方误差时还涉及近似，不能宣称是原始离散重建似然的精确等式。

### 从带权目标转向简化噪声损失

原论文再去掉不同 timestep 的权重，得到

$$L_{\mathrm{simple}}(\theta)=
\mathbb E_{\substack{x_0\sim q_{\mathrm{data}},\ t\sim\mathrm{Uniform}\{1,\ldots,T\}\\
\epsilon\sim\mathcal N(0,I)}}
\left[\left\|\epsilon-\epsilon_\theta\!\left(
\sqrt{\overline{\alpha}_t}x_0+\sqrt{1-\overline{\alpha}_t}\epsilon,t\right)\right\|^2\right]. \qquad (21)$$

去权重改变了各噪声程度的相对训练重点，是**重新加权后的目标**，不是简单删去与 \(\theta\) 无关的常数。\(L_{\mathrm{simple}}\) 不能直接当作原始负对数似然的上界，也不与原始 ELBO 严格等价。[原论文第 3.3–3.4 节及式（13）（14）](https://arxiv.org/pdf/2006.11239)




## 把完整推导接起来

**图片似然难直接计算 → 用 Jensen 构造路径上界 → 精确拆成反向单步匹配 → 解析求出给定原图的后验 → 固定方差的高斯 KL 变成均值误差 → 均值重参数化为噪声预测 → 选择简化目标训练 → 从噪声逐步采样。**

其中，原图一直只用于构造监督；数据平均使这些监督对应到真实反向条件的拟合。DDIM、score、SDE 和引导生成不属于这条核心推导，此处不展开。



## 完整参考资料

[1] Ho, Jain, Abbeel. [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239), 2020. 重点对照式（1）至（14）、算法 1–2、附录 A。

[2] Lilian Weng. [What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/), 2021. 本文仅使用 DDPM 基础相关部分。

[3] Nichol, Dhariwal. [Improved Denoising Diffusion Probabilistic Models](https://proceedings.mlr.press/v139/nichol21a.html), 2021. 第 3.1–3.2 节：学习反向方差与余弦噪声日程。

[4] OpenAI. [improved-diffusion 官方实现](https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py). 对照日程构造和方差选项。
