{
  "title": "扩散模型引导（一）：Classifier Guidance",
  "description": "怎样用分类器引导去噪：从带噪分类器训练、后验平均与噪声 MSE 推导 Score，再接入 DDIM，说明引导尺度和采样近似。",
  "slug": "classifier-guidance",
  "collection": "guidance",
  "episode": 1,
  "tags": [
    "Classifier Guidance",
    "扩散模型"
  ],
  "summary": "怎样用分类器引导去噪：从带噪分类器训练、后验平均与噪声 MSE 推导 Score，再接入 DDIM，说明引导尺度和采样近似。",
  "date": "2026-10-10T17:50:00+08:00",
  "categories": [
    "基础"
  ],
  "draft": false,
  "math": true
}

## 1 已经能生成图片了 为什么还需要引导

[DDPM](../ddpm-4/) 解决了怎样从噪声逐步生成图片，[DDIM](../ddim-derivation/) 则让我们可以改变采样路径、减少网络调用。接下来的问题是：**怎样告诉模型，这一次想生成什么？**

以手写数字为例。一个无条件去噪网络接收当前带噪图片和时间：

$$
\epsilon_\theta(x_t,t).
$$

输入中没有“这次想要数字 3”的信息。模型可能生成 0、3、7 等不同数字，这符合它学习整个数据分布的任务。

如果希望生成各种不同写法的 3，可以把类别交给去噪网络：

$$
\epsilon_\theta(x_t,t,y),\qquad y=3.
$$

这里指定的是类别，类别内部仍然可以有粗细、倾斜和笔画形状的变化。训练这种条件模型时，也要提供类别信息。

Classifier Guidance，简称 **CG**，提供了另一条连接条件的方式：在采样时，让一个分类器判断当前带噪状态应该怎样变化，才能提高目标类别的概率，再将这个方向放进去噪更新中。它既可以用于无条件扩散模型，也可以叠加在条件扩散模型上。[CG 原论文第 4 节](https://arxiv.org/html/2105.05233v4#S4)

本文先从无条件模型出发，把下面这条链完整接起来：

**带噪分类器怎样训练 → 怎样得到输入梯度 → 贝叶斯公式为什么允许加入这个方向 → Score 怎样变成噪声预测 → 怎样执行引导采样。**

### 沿用前篇的加噪符号

记图片为 \(d\) 维向量，并沿用 DDPM 的约定：

$$
\begin{aligned}
\alpha_t&=1-\beta_t,\\
\bar\alpha_t&=\prod_{r=1}^{t}\alpha_r,\\
\bar\alpha_0&=1.
\end{aligned}
\tag{1}
$$

取 \(0\lt\beta_t\lt1\)。对有噪声的时间 \(t\in\{1,\ldots,T\}\)，有 \(0\lt\bar\alpha_t\lt1\)。正向加噪为

$$
x_t=\sqrt{\bar\alpha_t}x_0
+\sqrt{1-\bar\alpha_t}\epsilon,
\qquad \epsilon\sim\mathcal N(0,I).
\tag{2}
$$

\(x_0\) 是原图，\(I\) 是单位矩阵；\(\sqrt{1-\bar\alpha_t}\) 是累计噪声的标准差。它与 DDPM 反向采样时另外选择的噪声标准差不是同一个量。

后文用 \(q\) 表示由数据和正向加噪确定的真实分布，用 \(p_\phi\) 表示分类器模型，用 \(\epsilon_\theta\) 表示去噪模型。推导分布关系时使用等号，替换成实际网络估计时保留近似号。

## 2 分类器怎样给出调整图片的方向

假设已经有一个分类器，它接收 \(x_t,t\)，输出各个类别的概率。对于目标类别 3，输出中的一个分量是

$$
p_\phi(y=3\mid x_t,t).
$$

\(\phi\) 是分类器参数。这里输入的是带噪图片，因为生成途中尚未得到干净图片；时间 \(t\) 告诉分类器当前的噪声等级。

假如分类器认为当前图片属于 3 的概率是 20%，这个标量只给出了当前判断。要知道**怎样修改图片才能提高这个概率**，还要对输入求梯度：

$$
g_\phi(x_t,t,y)
:=\nabla_{x_t}\log p_\phi(y\mid x_t,t).
\tag{3}
$$

计算这个梯度时，目标类别 \(y\)、时间 \(t\) 和参数 \(\phi\) 固定，允许变化的是图片 \(x_t\)。梯度与图片同维度，每个分量说明相应像素怎样微调会局部提高目标类别的对数概率。

例如，对一张只有两个分量的假想图片，若

$$
x_t=(0.2,0.5),\qquad g_\phi=(0.3,-0.1),
$$

那么稍微增大第一个分量、减小第二个分量，是局部有利的方向。更具体地，对足够小的正数 \(h\)，一阶展开给出

$$
\begin{aligned}
&\log p_\phi(y\mid x_t+h g_\phi,t)\\
&\quad=\log p_\phi(y\mid x_t,t)
+h\|g_\phi\|^2+o(h).
\end{aligned}
$$

对数单调递增，所以提高对数概率也提高概率。这只是说明梯度的局部意义，还没有规定扩散采样应该怎样走一步；直接反复做像素梯度上升，并不等于 CG 的采样算法。

还要区分两种求导：**训练分类器时，对参数 \(\phi\) 求导并更新参数；生成时，参数固定，对当前输入 \(x_t\) 求导。**

## 3 这个带噪分类器怎样训练

CG 使用的分类器需要见过不同噪声等级的输入，并使用与扩散模型相匹配的加噪日程。只在干净图片上训练的普通分类器，不能直接被假定能提供可靠的带噪梯度。[CG 原论文第 4.3 节](https://arxiv.org/html/2105.05233v4#S4.SS3)

### 一次训练样本的构造

第一步，从带标签数据中抽取一对 \((x_0,y)\)。例如，\(x_0\) 是一张数字 3，\(y=3\) 是其标签。

第二步，独立抽取时间和高斯噪声：

$$
t\sim\mathrm{Unif}\{1,\ldots,T\},
\qquad \epsilon\sim\mathcal N(0,I),
$$

然后按式（2）构造 \(x_t\)。图片虽然变得模糊，训练标签仍是原图的类别 \(y\)。

第三步，把 \((x_t,t)\) 交给分类器，得到各类别概率：

$$
p_\phi(k\mid x_t,t),\qquad k=0,\ldots,9.
$$

**正确标签 \(y\) 用来计算损失，不作为这个分类器的输入；干净原图 \(x_0\) 也不输入分类器，它只用于制造 \(x_t\)。**

第四步，用交叉熵训练。单个样本的损失为

$$
\ell(\phi)=-\log p_\phi(y\mid x_t,t).
\tag{4}
$$

目标类别的预测概率越小，这项损失越大。对参数 \(\phi\) 求梯度、执行优化器更新，就完成一次训练。

### 把三个抽样来源写进目标

大写 \(X_0,Y\) 表示随机变量，小写 \(x_0,y\) 表示它们的一次取值。完整目标可以写为

$$
\begin{aligned}
L_{\mathrm{cls}}(\phi)
={}&\mathbb E_{(X_0,Y)\sim q_{\mathrm{data}}}
\mathbb E_{t\sim\mathrm{Unif}\{1,\ldots,T\}}\\
&\mathbb E_{\epsilon\sim\mathcal N(0,I)}
\left[-\log p_\phi(Y\mid X_t,t)\right],\\
X_t={}&\sqrt{\bar\alpha_t}X_0
+\sqrt{1-\bar\alpha_t}\epsilon.
\end{aligned}
\tag{5}
$$

这里依次平均数据对、时间和噪声。三次抽样互相独立，但数据对内部的图片 \(X_0\) 与标签 \(Y\) 有关联。

交叉熵让分类器学习给定带噪图片后的类别概率。噪声很大时，类别信息可能已经很少，不应要求每个样本都能被准确辨认。

与 DDPM 训练对照：

| 模型 | 输入 | 监督答案 | 损失 |
|---|---|---|---|
| 无条件去噪网络 | \(x_t,t\) | 抽到的噪声 \(\epsilon\) | 噪声预测 MSE |
| CG 分类器 | \(x_t,t\) | 原图类别 \(y\) | 类别交叉熵 |

官方实现依次进行加噪、带时间的分类预测和交叉熵计算，与上面的构造一致。[分类器训练代码](https://github.com/openai/guided-diffusion/blob/main/scripts/classifier_train.py)

## 4 为什么分类器方向可以加到生成方向上

现在已经知道分类器怎样得到，也知道怎样对它的输入求梯度。下一步的问题是：**这个方向为什么能接进扩散模型？**

先固定时间 \(t\)。我们要区分三个分布：

- \(q_t(x_t)\)：所有类别的图片加噪后形成的带噪分布
- \(q_t(x_t\mid y)\)：给定类别 \(y\) 时的带噪分布
- \(q_t(y\mid x_t)\)：看到当前带噪图片后，它来自类别 \(y\) 的概率

前两个是对图片的密度，第三个是对类别的概率。下标 \(t\) 表明它们对应当前噪声等级。我们想用前者已有的生成能力，得到符合指定类别的样本。

### Score 描述什么

对数密度对输入的梯度称为 Score：

$$
s_t(x_t):=\nabla_{x_t}\log q_t(x_t).
\tag{6}
$$

它是与图片同维度的向量，表示在当前位置怎样微小改变输入可以提高对数密度。这里的 Score 不是分类器输出的置信度，也不是一个评价整张图片质量的分数。

例如，一维高斯密度为 \(r(x)=\mathcal N(x;\mu,\sigma^2)\) 时，

$$
\frac{d}{dx}\log r(x)
=-\frac{x-\mu}{\sigma^2}.
$$

它在中心右侧为负、左侧为正，局部指向高斯中心。一般的数据分布可能有很多峰，不能把它简单想象成“所有图片都朝同一个中心移动”。

### 从贝叶斯公式开始

对满足 \(q(y)\gt0\) 的目标类别，贝叶斯公式给出

$$
q_t(x_t\mid y)
=\frac{q_t(x_t)q_t(y\mid x_t)}{q(y)}.
\tag{7}
$$

取对数，把乘法变成加法：

$$
\begin{aligned}
\log q_t(x_t\mid y)
={}&\log q_t(x_t)
+\log q_t(y\mid x_t)\\
&-\log q(y).
\end{aligned}
$$

固定 \(y,t\)，对 \(x_t\) 求梯度。类别先验 \(q(y)\) 不随当前图片变化，所以最后一项的梯度为零：

$$
\begin{aligned}
\nabla_{x_t}\log q_t(x_t\mid y)
={}&\nabla_{x_t}\log q_t(x_t)\\
&+\nabla_{x_t}\log q_t(y\mid x_t).
\end{aligned}
\tag{8}
$$

也就是

$$
\text{条件 Score}
=\text{无条件 Score}+\text{分类器方向}.
$$

这解释了为什么使用对数概率：对数将贝叶斯关系中的乘法变成加法，求导后便得到两个方向相加。式（8）对真实分布是精确关系；实际算法中，分类器 \(p_\phi\) 近似 \(q_t(y\mid x_t)\)，它的输入梯度近似最后一项。[CG 原论文第 4.2 节](https://arxiv.org/html/2105.05233v4#S4.SS2)

但我们手里的 DDPM 网络输出的是噪声预测，还不是式（6）的 Score。接下来要把这座桥补上。

## 5 已知原图时 噪声怎样对应 Score

先固定原图 \(x_0\) 和时间 \(t\)，只让式（2）的高斯噪声变化。此时带噪图片的条件密度是

$$
\begin{aligned}
&q(x_t\mid x_0,t)\\
&\quad=\mathcal N\!\left(
x_t;\sqrt{\bar\alpha_t}x_0,
(1-\bar\alpha_t)I\right).
\end{aligned}
\tag{9}
$$

前篇常把时间省略，写成 \(q(x_t\mid x_0)\)。这里显式保留 \(t\)，方便区分之后的条件和平均变量。

写出它的对数密度：

$$
\begin{aligned}
\log q(x_t\mid x_0,t)
&=-\frac{\|x_t-\sqrt{\bar\alpha_t}x_0\|^2}
{2(1-\bar\alpha_t)}+C_t,\\
C_t&=-\frac d2\log\!\left[2\pi(1-\bar\alpha_t)\right].
\end{aligned}
$$

固定 \(t\) 后，\(C_t\) 不依赖 \(x_t\)。又因为

$$
\nabla_x\|x-a\|^2=2(x-a),
$$

所以对带噪输入求导得到

$$
\nabla_{x_t}\log q(x_t\mid x_0,t)
=-\frac{x_t-\sqrt{\bar\alpha_t}x_0}
{1-\bar\alpha_t}.
\tag{10}
$$

由加噪式（2），分子等于 \(\sqrt{1-\bar\alpha_t}\epsilon\)。约掉一个平方根：

$$
\nabla_{x_t}\log q(x_t\mid x_0,t)
=-\frac{\epsilon}{\sqrt{1-\bar\alpha_t}}.
\tag{11}
$$

这里的 \(\epsilon\) 是**给定这张原图与当前带噪图片后，能够唯一反解出的那一份噪声**。负号也可以从高斯直觉理解：噪声把样本带离中心，条件 Score 局部指回中心。

到这里还不能直接用噪声网络替换 \(\epsilon\)。式（11）固定了真实原图，而生成时不知道原图；仅给定 \(x_t\)，通常还有多种原图和噪声解释。我们真正需要的是带噪边缘分布 \(q_t(x_t)\) 的 Score。

## 6 不知道原图时 为什么必须按后验平均

### 先说清楚概率分布从哪里来

固定一个有非零噪声的时间 \(t\)，规定抽样过程

$$
X_0\sim q_{\mathrm{data}},\qquad
\epsilon\sim\mathcal N(0,I),\qquad X_0\perp\epsilon,
$$

$$
X_t=\sqrt{\bar\alpha_t}X_0
+\sqrt{1-\bar\alpha_t}\epsilon.
\tag{12}
$$

这里 \(q_{\mathrm{data}}\) 指图片的边缘分布；它可以由第 3 节的带标签数据分布将标签求和得到。式（12）确定了 \((X_0,\epsilon,X_t)\) 的联合概率分布，记为 \(Q_t\)。

\(Q_t\) 中的 \(t\) 已固定，不是另一个需要平均的随机变量。\(X_t\) 是随机变量，\(x_t\) 是观察到的具体值。后面写 \(\mathbb E_{Q_t}[\cdot\mid X_t=x_t]\)，就是在这个明确的联合分布下取条件期望。

下面先按数据分布有密度的情形书写积分；如果用离散的经验数据分布，对原图的积分改成求和，推导结构不变。

### 第一步 对原图边缘化

一张带噪图片可能由不同原图产生，因此

$$
q_t(x_t)
=\int q_{\mathrm{data}}(x_0)
q(x_t\mid x_0,t)\,dx_0.
\tag{13}
$$

此时用的是数据先验 \(q_{\mathrm{data}}(x_0)\)。这只是对联合密度做边缘化，还没有出现后验平均。

### 第二步 对边缘密度取对数梯度

使用 \(\nabla\log f=(\nabla f)/f\)，并在可以交换求导与积分的条件下，得到

$$
\begin{aligned}
\nabla_{x_t}\log q_t(x_t)
&=\frac{\nabla_{x_t}q_t(x_t)}{q_t(x_t)}\\
&=\frac{\int q_{\mathrm{data}}(x_0)
\nabla_{x_t}q(x_t\mid x_0,t)\,dx_0}{q_t(x_t)}.
\end{aligned}
\tag{14}
$$

求导针对 \(x_t\)，数据先验只依赖积分变量 \(x_0\)，所以它保留在积分中。

### 第三步 把密度梯度写成密度乘 Score

把同一个对数求导公式换个排列：

$$
\begin{aligned}
\nabla_{x_t}q(x_t\mid x_0,t)
={}&q(x_t\mid x_0,t)\\
&\cdot\nabla_{x_t}\log q(x_t\mid x_0,t).
\end{aligned}
$$

将它代入式（14）。分母不随 \(x_0\) 变化，可以放进积分：

$$
\begin{aligned}
\nabla_{x_t}\log q_t(x_t)
=\int&\frac{q_{\mathrm{data}}(x_0)q(x_t\mid x_0,t)}
{q_t(x_t)}\\
&\cdot\nabla_{x_t}\log q(x_t\mid x_0,t)\,dx_0.
\end{aligned}
\tag{15}
$$

### 第四步 用贝叶斯公式认出后验权重

积分里的系数恰好是

$$
q(x_0\mid x_t,t)
=\frac{q_{\mathrm{data}}(x_0)q(x_t\mid x_0,t)}
{q_t(x_t)}.
\tag{16}
$$

于是

$$
\begin{aligned}
\nabla_{x_t}\log q_t(x_t)
=\int&q(x_0\mid x_t,t)\\
&\cdot\nabla_{x_t}\log q(x_t\mid x_0,t)\,dx_0.
\end{aligned}
\tag{17}
$$

用期望记号写，就是

$$
\begin{aligned}
\nabla_{x_t}\log q_t(x_t)
={}&\mathbb E_{X_0\sim q(\cdot\mid x_t,t)}\\
&\left[\nabla_{x_t}\log q(x_t\mid X_0,t)\right].
\end{aligned}
\tag{18}
$$

**后验权重是对边缘密度求对数梯度之后自然出现的结果。** 不能把式（18）的权重换成数据先验；观察到当前 \(x_t\) 后，不同原图解释它的可能性已经不同了。

### 期望下标到底在平均谁

一般地，

$$
\mathbb E_{X\sim p}[f(X)]
=\int p(x)f(x)\,dx.
$$

下标说明让哪个随机变量按什么分布变化，方括号说明被平均的量。例如标准正态分布有

$$
\begin{aligned}
\mathbb E_{X\sim\mathcal N(0,1)}[X]&=0,\\
\mathbb E_{X\sim\mathcal N(0,1)}[X^2]&=1.
\end{aligned}
$$

第一个平均 \(X\) 本身，第二个平均它的平方；用的权重分布相同。第一个式子展开为

$$
\int_{-\infty}^{+\infty}
x\frac{e^{-x^2/2}}{\sqrt{2\pi}}\,dx=0.
$$

如果下标写成

$$
X_0\sim q(\cdot\mid x_t,t),
$$

则变化的是 \(X_0\)，固定的是 \(x_t,t\)。点号 \(\cdot\) 为原图取值留出位置，填入 \(x_0\) 就得到权重 \(q(x_0\mid x_t,t)\)。下标里出现条件，并不表示在对这些条件平均。

式（18）还包含两个不同操作：

1. 里面的 \(\nabla_{x_t}\)：固定一张候选原图，对带噪输入求导
2. 外面的期望：固定当前 \(x_t,t\)，遍历候选原图，平均刚才那些导数在当前点的值

被平均的变量也完全可以出现在内部表达式的条件位置。比如固定 \(x\)，让 \(Y\) 按分布 \(r\) 变化：

$$
\mathbb E_{Y\sim r}[\log q(x\mid Y)]
=\int r(y)\log q(x\mid y)\,dy.
$$

每取一个 \(y\)，先计算对应的条件对数密度，再将这个数加权平均。类似地，固定 \(y\) 时也可以让 \(X\) 变化：

$$
\begin{aligned}
&\mathbb E_{X\sim q(\cdot\mid y)}[\log q(X\mid y)]\\
&\quad=\int q(x\mid y)\log q(x\mid y)\,dx.
\end{aligned}
$$

所以，内部的条件符号与外层的平均是两层操作。判断谁在变化，要看期望的分布或积分末尾的 \(dx_0\)、\(dy\)，不能只看变量位于竖线哪一侧。

### 从原图后验平均 走到噪声条件均值

将式（10）的高斯 Score 代入式（17）：

$$
\nabla_{x_t}\log q_t(x_t)
=-\int q(x_0\mid x_t,t)
\frac{x_t-\sqrt{\bar\alpha_t}x_0}
{1-\bar\alpha_t}\,dx_0.
$$

分母可以拆成两个相同的平方根。固定 \(t\) 后，其中一个提出积分：

$$
\begin{aligned}
\nabla_{x_t}\log q_t(x_t)
={}&-\frac1{\sqrt{1-\bar\alpha_t}}\\
&\cdot\int q(x_0\mid x_t,t)
\frac{x_t-\sqrt{\bar\alpha_t}x_0}
{\sqrt{1-\bar\alpha_t}}\,dx_0.
\end{aligned}
\tag{19}
$$

为什么后面的积分可以叫作噪声的条件期望？因为式（12）给出随机变量之间的精确关系

$$
\epsilon
=\frac{X_t-\sqrt{\bar\alpha_t}X_0}
{\sqrt{1-\bar\alpha_t}}.
$$

因此，在 \(Q_t\) 下可以直接将这个表达式替换进条件期望：

$$
\begin{aligned}
&\mathbb E_{Q_t}[\epsilon\mid X_t=x_t]\\
&=\mathbb E_{Q_t}\!\left[
\left.\frac{X_t-\sqrt{\bar\alpha_t}X_0}
{\sqrt{1-\bar\alpha_t}}\right|X_t=x_t\right]\\
&=\mathbb E_{X_0\sim q(\cdot\mid x_t,t)}
\left[\frac{x_t-\sqrt{\bar\alpha_t}X_0}
{\sqrt{1-\bar\alpha_t}}\right]\\
&=\int q(x_0\mid x_t,t)
\frac{x_t-\sqrt{\bar\alpha_t}x_0}
{\sqrt{1-\bar\alpha_t}}\,dx_0.
\end{aligned}
\tag{20}
$$

第二个等号使用了两件事：给定 \(X_t=x_t\) 后，表达式中的 \(X_t\) 固定为 \(x_t\)；剩余随机变量 \(X_0\) 的分布是后验 \(q(x_0\mid x_t,t)\)。

积分虽然遍历原图，却平均每张候选原图对应的噪声。它与左边的噪声条件期望是同一个量，没有换一种平均，也没有引入模型近似。

将式（20）代回式（19），得到这座桥的第一部分：

$$
\nabla_{x_t}\log q_t(x_t)
=-\frac{\mathbb E_{Q_t}[\epsilon\mid X_t=x_t]}
{\sqrt{1-\bar\alpha_t}}.
\tag{21}
$$

虽然最初抽样时 \(\epsilon\sim\mathcal N(0,I)\)，但给定 \(X_t=x_t\) 后，噪声的条件分布已经改变，因此式（21）里的条件均值通常不能写成零。

### 为什么不能直接等于给定原图的 Score

式（11）使用某一张确定原图对应的具体噪声，式（21）使用所有可能噪声解释的条件均值。两者之间隔着式（18）的后验平均。

因此应当写

$$
\begin{aligned}
\nabla_{x_t}\log q_t(x_t)
&=\mathbb E_{X_0\sim q(\cdot\mid x_t,t)}
\left[\nabla_{x_t}\log q(x_t\mid X_0,t)\right]\\
&=-\frac{\mathbb E_{Q_t}[\epsilon\mid X_t=x_t]}
{\sqrt{1-\bar\alpha_t}},
\end{aligned}
$$

不能省掉平均，再与某一张原图的条件 Score 直接连等号。这里“给定原图 \(x_0\)”也不同于第 4 节的“给定类别 \(y\)”：一个类别仍包含许多可能原图。

## 7 为什么噪声 MSE 学到的正是这个条件均值

式（21）说的是数据分布，而采样中使用的是网络输出。还需要说明，为什么 DDPM 的噪声预测会逼近其中的条件均值。

简化噪声 MSE 目标是

$$
\begin{aligned}
L_{\mathrm{simple}}(\theta)
={}&\mathbb E_{t\sim\mathrm{Unif}\{1,\ldots,T\}}\\
&\mathbb E_{X_0\sim q_{\mathrm{data}}}
\mathbb E_{\epsilon\sim\mathcal N(0,I)}
\left[\|\epsilon-\epsilon_\theta(X_t,t)\|^2\right],
\end{aligned}
\tag{22}
$$

其中 \(X_t\) 仍按式（12）构造。[DDPM 原论文第 3.4 节与算法 1](https://arxiv.org/html/2006.11239v2#S3.SS4)

### 固定输入 网络只能给出一个预测

固定当前 \(x_t,t\)，暂时把网络输出记为确定向量 \(a\)。对应的条件风险为

$$
R(a;x_t,t)
:=\mathbb E_{Q_t}
\left[\|\epsilon-a\|^2\mid X_t=x_t\right].
$$

虽然输入已固定，产生它的噪声仍可能有多种解释。记这些噪声的条件均值为

$$
m(x_t,t):=\mathbb E_{Q_t}[\epsilon\mid X_t=x_t].
$$

下面为简洁写作 \(m\)。固定输入后，\(m\) 和 \(a\) 都是确定的向量。

### 在误差里加减条件均值

先写

$$
\epsilon-a=(\epsilon-m)+(m-a).
$$

展开平方：

$$
\begin{aligned}
\|\epsilon-a\|^2
={}&\|\epsilon-m\|^2\\
&+2(\epsilon-m)^\top(m-a)
+\|m-a\|^2.
\end{aligned}
$$

对每一项按同一个条件分布取期望：

$$
\begin{aligned}
R(a;x_t,t)
={}&\mathbb E_{Q_t}
\left[\|\epsilon-m\|^2\mid X_t=x_t\right]\\
&+2\mathbb E_{Q_t}
[\epsilon-m\mid X_t=x_t]^\top(m-a)\\
&+\|m-a\|^2.
\end{aligned}
\tag{23}
$$

根据 \(m\) 的定义，

$$
\mathbb E_{Q_t}[\epsilon-m\mid X_t=x_t]
=m-m=0,
$$

所以交叉项消失：

$$
\begin{aligned}
R(a;x_t,t)
={}&\mathbb E_{Q_t}
\left[\|\epsilon-m\|^2\mid X_t=x_t\right]\\
&+\|m-a\|^2.
\end{aligned}
\tag{24}
$$

第一项不随预测 \(a\) 变化，第二项在 \(a=m\) 时最小。因此理想的逐点最优预测是

$$
a^*=\mathbb E_{Q_t}[\epsilon\mid X_t=x_t].
\tag{25}
$$

完整训练目标还会对不同输入和时间平均。若预测函数不受限制，可以在几乎处处的输入上取这个最优值；实际网络受数据、容量和优化效果限制，所以只能写

$$
\epsilon_\theta(x_t,t)
\approx\mathbb E_{Q_t}[\epsilon\mid X_t=x_t].
\tag{26}
$$

**网络只看到带噪图片和时间。MSE 让它学习的是噪声的条件平均，不能保证它找回每次训练实际抽中的那份噪声。**

### 终于把噪声预测变成 Score 估计

定义由噪声网络产生的向量场

$$
s_\theta(x_t,t)
:=-\frac{\epsilon_\theta(x_t,t)}
{\sqrt{1-\bar\alpha_t}}.
\tag{27}
$$

结合式（21）和式（26）：

$$
\begin{aligned}
\nabla_{x_t}\log q_t(x_t)
&=-\frac{\mathbb E_{Q_t}[\epsilon\mid X_t=x_t]}
{\sqrt{1-\bar\alpha_t}}\\
&\approx-\frac{\epsilon_\theta(x_t,t)}
{\sqrt{1-\bar\alpha_t}}\\
&=s_\theta(x_t,t).
\end{aligned}
\tag{28}
$$

最后一行是定义，中间的近似来自学习误差。实际网络给出的向量场，也不能无条件宣称就是某个精确密度的对数梯度。

现在可以回到 CG 的目标：分类器方向应加到 Score 上，而式（27）让我们能把这个加法换回噪声预测。[CG 原论文式（11）至（14）](https://arxiv.org/html/2105.05233v4#S4.SS2)

## 8 把分类器方向翻译成噪声修正

先从无条件基模型出发，不额外加强引导。根据贝叶斯关系，构造

$$
s_{\mathrm{guided}}(x_t,t,y)
:=s_\theta(x_t,t)+g_\phi(x_t,t,y).
\tag{29}
$$

为了继续使用熟悉的采样公式，我们希望有一份新的噪声预测 \(\hat\epsilon\)，满足

$$
\begin{aligned}
-\frac{\hat\epsilon}{\sqrt{1-\bar\alpha_t}}
={}&-\frac{\epsilon_\theta(x_t,t)}
{\sqrt{1-\bar\alpha_t}}\\
&+g_\phi(x_t,t,y).
\end{aligned}
$$

两边同时乘以 \(-\sqrt{1-\bar\alpha_t}\)，得到

$$
\hat\epsilon
=\epsilon_\theta(x_t,t)
-\sqrt{1-\bar\alpha_t}\,g_\phi(x_t,t,y).
\tag{30}
$$

**Score 上加分类器方向，噪声预测上就减去相应项。** 这个负号来自 Score 与噪声的换算，不是另外约定出来的。

若希望调节分类器的影响强度，引入 \(\gamma\geq0\)：

$$
s_{\mathrm{guided}}
:=s_\theta+\gamma g_\phi,
$$

对应

$$
\begin{aligned}
\hat\epsilon_{\mathrm{CG}}
={}&\epsilon_\theta(x_t,t)\\
&-\gamma\sqrt{1-\bar\alpha_t}
\nabla_{x_t}\log p_\phi(y\mid x_t,t).
\end{aligned}
\tag{31}
$$

这一步是在已定义的估计量之间做精确代数变换。当 \(\gamma=1\) 时，它对真实条件 Score 的估计仍取决于两个模型；\(\gamma\ne1\) 则主动改变了引导强度，通常不再对应普通条件 Score。

### 接到确定性 DDIM

沿用前篇的确定性 DDIM。先用修正后的噪声计算当前原图估计：

$$
\hat x_0
=\frac{x_t-\sqrt{1-\bar\alpha_t}\hat\epsilon_{\mathrm{CG}}}
{\sqrt{\bar\alpha_t}}.
\tag{32}
$$

再更新到前一个时刻：

$$
x_{t-1}
=\sqrt{\bar\alpha_{t-1}}\hat x_0
+\sqrt{1-\bar\alpha_{t-1}}\hat\epsilon_{\mathrm{CG}}.
\tag{33}
$$

两处都使用同一份修正后的噪声。\(\hat x_0\) 是当前估计，不是已知的真实原图。到达新的 \(x_{t-1}\) 后，要重新计算噪声预测和分类器梯度。[CG 原论文算法 2](https://arxiv.org/html/2105.05233v4#alg2)

跳步时，对任意较早目标 \(0\leq s\lt t\)，相应地使用目标时间的累计系数：

$$
x_s=\sqrt{\bar\alpha_s}\hat x_0
+\sqrt{1-\bar\alpha_s}\hat\epsilon_{\mathrm{CG}}.
\tag{34}
$$

本次梯度仍在当前 \(x_t,t\) 上计算。网络和分类器接收的是训练时对应的时间或噪声等级，不能把原时间编号替换成“这是第几次循环”。最后目标为 \(s=0\) 时，\(\bar\alpha_0=1\)，直接输出当前 \(\hat x_0\)，不必在干净端点再调用分类器。[DDIM 原论文第 4.2 节](https://arxiv.org/html/2010.02502v4#S4.SS2)

### 一次采样怎样执行

下面用单样本伪代码表达逻辑，\(\theta,\phi\) 已训练好并固定，\(y\) 是目标标签，`times` 存放递减的原始时间编号：

```python
# alpha_bar[0] = 1
# times 例如 [1000, 800, 600, 400, 200, 0]
# model 与 classifier 均使用评估模式，参数保持固定
x = randn(image_shape)

for t, s in adjacent_pairs(times):
    eps = model(x, t)

    # 单独保留分类器对当前输入的梯度
    x_in = detach(x).requires_grad_(True)
    log_prob = log_softmax(classifier(x_in, t))[y]
    g = grad(log_prob, x_in)

    eps_guided = eps - gamma * sqrt(1 - alpha_bar[t]) * g
    x0_hat = (x - sqrt(1 - alpha_bar[t]) * eps_guided) / sqrt(alpha_bar[t])
    x = sqrt(alpha_bar[s]) * x0_hat + sqrt(1 - alpha_bar[s]) * eps_guided
    x = detach(x)

return x
```

采样中的动作是：**预测噪声 → 对分类器输入求梯度 → 修正噪声 → 执行 DDIM 更新。**

这里没有分类器优化器更新。去噪预测和状态更新可以不记录梯度，但分类器求 \(g\) 的局部计算必须允许输入梯度；不能把整个采样过程包在完全禁用梯度的环境里又不重新开启它。评估模式与关闭自动求导也是两回事。[官方采样中的输入梯度实现](https://github.com/openai/guided-diffusion/blob/main/scripts/classifier_sample.py)

固定初始噪声、条件、日程和确定的推理计算后，确定性 DDIM 的轨迹才确定。初始 \(x_T\) 仍随机；换一份初始噪声仍可得到不同结果。这里按前篇取 \(x_T\sim\mathcal N(0,I)\)，使用的是末端足够接近标准高斯的近似。

式（32）至（34）没有加入原图裁剪等额外处理。若修改这些步骤，需要重新核对实际使用的更新，而不能假定全部代数关系仍原样成立。

## 9 引导强度的含义 取决于基模型的起点

### 从无条件模型出发

本文式（31）采用

$$
s_{\mathrm{guided}}=s_\theta+\gamma g_\phi.
$$

在这个约定中：

| 强度 | 含义 |
|---|---|
| \(\gamma=0\) | 不使用分类器，保留无条件预测 |
| \(\gamma=1\) | 按贝叶斯关系加入一次分类器方向 |
| \(\gamma\gt1\) | 在此基础上进一步增强分类器方向 |

当无条件 Score 和分类器梯度都精确时，\(\gamma=1\) 给出真实条件 Score。这个结论是每个时间的分布恒等式，**不意味着有限步 DDIM、近似末端分布和实际网络已经保证精确采到真实条件分布**。

增强引导可以让采样更偏向分类器认可的目标类别特征，但也可能牺牲类别内部的多样性。CG 原论文实际观察到保真度与分布覆盖之间的权衡，因此没有一个对所有模型都最好的强度，也不能认为越大越好。[CG 原论文第 4.3 节](https://arxiv.org/html/2105.05233v4#S4.SS3)

### 基模型已经接收类别时

如果基模型本身就是条件模型，起点已经是

$$
s_\theta(x_t,t,y)
:=-\frac{\epsilon_\theta(x_t,t,y)}
{\sqrt{1-\bar\alpha_t}}.
$$

在它上面再加分类器方向，可以写成

$$
s_{\mathrm{guided}}
=s_\theta(x_t,t,y)+w g_\phi(x_t,t,y).
\tag{35}
$$

这里 \(w=0\) 表示保留普通条件预测。它和前面无条件基模型的 \(\gamma=0\) 起点不同。

为了看清两个尺度的关系，暂时回到真实分布，定义

$$
\begin{aligned}
s_{\mathrm u}&:=\nabla_{x_t}\log q_t(x_t),\\
s_{\mathrm c}&:=\nabla_{x_t}\log q_t(x_t\mid y),\\
g&:=\nabla_{x_t}\log q_t(y\mid x_t).
\end{aligned}
$$

式（8）说明 \(s_{\mathrm c}=s_{\mathrm u}+g\)，所以

$$
\begin{aligned}
s_{\mathrm c}+wg
&=(s_{\mathrm u}+g)+wg\\
&=s_{\mathrm u}+(1+w)g.
\end{aligned}
\tag{36}
$$

因此，在精确 Score 层面，两个起点的总引导强度关系为

$$
\gamma=1+w.
$$

这只是本文用来区分起点的记号，不要求所有代码都用同样的参数名。看到“guidance 为 0”时，应先检查基模型是否已经接收条件、参数究竟乘在哪一项。实际训练的有条件和无条件网络各有估计误差，不能仅凭式（36）就断言它们的采样结果严格相同。

## 10 DDPM 的均值修正与本文的噪声修正

本文通过 Score 推导 CG，并完整接到了 DDIM。阅读原论文和代码时，还会遇到另一种写法：在随机反向高斯转移中直接修正均值。

设无条件反向模型给出的均值和协方差为 \(\mu_\theta(x_t,t)\)、\(\Sigma_\theta(x_t,t)\)。CG 原论文算法 1 使用的均值修正形式为

$$
\mu_{\mathrm{guided}}
=\mu_\theta+\gamma\Sigma_\theta g_\phi(x_t,t,y).
\tag{37}
$$

该路线来自对分类器对数概率做局部一阶近似，再与反向高斯配方。原论文第 4.1 节的局部推导在候选反向状态的均值处展开；算法 1 则使用当前状态的分类器梯度。它包含近似，不能把两种取值点混写成精确恒等式。[CG 原论文第 4.1 节与算法 1](https://arxiv.org/html/2105.05233v4#S4.SS1)

这里要保留的关键区别是：**将式（31）的噪声修正直接代入 DDPM 均值，不会在一般有限步设置下自动变成式（37）。**

前篇的噪声参数化均值为

$$
\mu_\theta
=\frac1{\sqrt{\alpha_t}}
\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}
\epsilon_\theta(x_t,t)\right).
$$

把 \(\epsilon_\theta\) 换成式（31）的 \(\hat\epsilon_{\mathrm{CG}}\)，先计算两种均值之差：

$$
\begin{aligned}
\mu_{\mathrm{score}}-\mu_\theta
&=-\frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1-\bar\alpha_t}}
(\hat\epsilon_{\mathrm{CG}}-\epsilon_\theta)\\
&=-\frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1-\bar\alpha_t}}
\left(-\gamma\sqrt{1-\bar\alpha_t}g_\phi\right)\\
&=\frac{\gamma\beta_t}{\sqrt{\alpha_t}}g_\phi.
\end{aligned}
$$

因此代入后的均值为

$$
\mu_{\mathrm{score}}
=\mu_\theta+\frac{\gamma\beta_t}{\sqrt{\alpha_t}}g_\phi.
\tag{38}
$$

乘在梯度上的矩阵一般不同：

$$
\frac{\gamma\beta_t}{\sqrt{\alpha_t}}I
\ne\gamma\Sigma_\theta.
$$

即使选 \(\Sigma_\theta=\beta_t I\)，当 \(\gamma\gt0\) 时仍有 \(1/\sqrt{\alpha_t}\) 的有限步系数差异。

这说明两条路线各有自己的离散更新规则。本文的完整推导走到 Score 与 DDIM；式（37）用于辨认另一种常见实现，不能代替其局部近似的证明。官方代码也分别实现了 `condition_mean` 与 `condition_score`。[两种引导方式的官方实现](https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/gaussian_diffusion.py)

## 11 回头检查这条推导

**为什么分类器要接收时间 \(t\)？**

因为不同时间对应不同噪声等级，分类器要估计的是当前带噪分布上的类别概率。它的训练加噪日程需要与扩散模型匹配。

**分类器训练和引导采样的梯度有什么不同？**

训练对参数 \(\phi\) 求导并更新参数；引导采样固定参数，对当前带噪输入 \(x_t\) 求导，得到与图片同形状的方向。

**为什么边缘 Score 要按后验平均？**

先对边缘密度求对数梯度，再把密度梯度写成“密度乘 Score”，自然出现“先验乘似然除以边缘密度”。这个权重正是看到 \(x_t\) 后的原图后验。

**期望下标里的变量能出现在内部条件位置吗？**

可以。外层期望决定遍历谁、用什么权重；内层条件密度按每个候选值计算。式（18）对 \(X_0\) 平均，里面却对 \(x_t\) 求导，两者各自有明确的操作对象。

**噪声来自标准高斯 为什么条件均值不一定是零？**

标准高斯描述抽样时的无条件噪声。看到 \(X_t=x_t\) 后，各种噪声解释的权重已经改变，应使用条件分布。噪声 MSE 的最优预测正是这个条件均值。

**CG 的噪声修正为什么是减号？**

因为 Score 估计等于负的噪声预测除以累计噪声标准差。把分类器方向加到 Score 上，再乘回负的标准差，就得到式（31）的减号。

**引导强度为零 是无条件生成吗？**

要看基模型。如果从无条件模型出发，\(\gamma=0\) 保留无条件预测；如果从条件模型出发，\(w=0\) 保留普通条件预测。

至此，CG 的训练与使用接在了一起：用带噪图片和标签训练分类器，生成时固定模型参数，从目标类别的输入梯度构造引导后的 Score，再换回噪声预测交给采样器。这个连接让已有的去噪能力能够利用条件信息，同时保留了模型近似与具体采样规则的边界。

## 参考

- Prafulla Dhariwal、Alex Nichol，**Diffusion Models Beat GANs on Image Synthesis**。本文 CG 的主要来源：第 4 节、算法 1–2；无条件与条件基模型及强度讨论见第 4.3 节。[论文](https://arxiv.org/html/2105.05233v4)
- Jonathan Ho、Ajay Jain、Pieter Abbeel，**Denoising Diffusion Probabilistic Models**。用于核对正向加噪、噪声预测训练和 DDPM 均值参数化。[论文](https://arxiv.org/html/2006.11239v2)
- Jiaming Song、Chenlin Meng、Stefano Ermon，**Denoising Diffusion Implicit Models**。用于核对确定性更新与跳步；论文中的累计信号系数 \(\alpha_t\) 对应本文 \(\bar\alpha_t\)。[论文](https://arxiv.org/html/2010.02502v4)
- OpenAI，**guided-diffusion**。分类器的带噪训练、对输入求梯度，以及均值修正和 Score 修正的实现对照。[官方代码](https://github.com/openai/guided-diffusion)
- Lilian Weng，**What are Diffusion Models?**。用于整理整体阅读路线；本文的数学关系从前向加噪、贝叶斯公式与 MSE 展开推导，并对照上面的一手论文与代码。[博客](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
