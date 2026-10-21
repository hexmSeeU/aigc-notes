{
  "title": "Score：噪声预测网络到底学到了什么",
  "description": "从单次噪声标签的不确定性出发，回顾 MSE 条件均值，完整推导后验平均与 Score，并厘清噪声预测、原图估计和生成采样的区别。",
  "summary": "同一个带噪输入为什么有多个噪声答案？从条件均值到混合密度的后验权重，理解噪声预测如何估计整体分布的 Score。",
  "date": "2026-10-21T00:00:00+00:00",
  "slug": "score",
  "categories": ["基础"],
  "tags": ["Score", "DDPM", "扩散模型"],
  "draft": false,
  "math": true
}

[DDPM 的训练篇]({{< relref "posts/ddpm-2" >}})介绍过一个很方便的任务：自己抽噪声、自己制造带噪图片，再让网络预测这次加入的噪声。答案就在训练程序手里，似乎只要学得足够好，网络就能把它原样找回来。

但网络只看见混合后的图片，并不知道这次使用了哪张原图。同一个带噪输入，可能有不同的来源。那么，噪声预测究竟在学习什么？为什么 [CG]({{< relref "posts/classifier-guidance" >}}) 和 [CFG]({{< relref "posts/classifier-free-guidance" >}}) 又把它叫作带噪数据分布的 Score 估计？

本文从这个疑问出发，先回顾平方误差的最优预测是条件均值，再推导它为什么对应整体分布的对数密度梯度。不要求先读引导文章，也暂不展开 Score Matching、SDE 或 ODE 的训练与采样理论。

## 1. 训练程序知道答案，网络为什么仍然不能唯一确定？

### 1.1 先区分随机变量和一次取值

沿用 DDPM 的记号，并缩写两个系数：

$$
\alpha_t=1-\beta_t,\qquad
\bar\alpha_t=\prod_{r=1}^t\alpha_r,\qquad
 a_t=\sqrt{\bar\alpha_t},\qquad b_t=\sqrt{1-\bar\alpha_t}.
$$

全文推导时**固定同一个时间 \(t\)**，因此 \(a_t,b_t\) 都是常数。Score 与噪声的换算要求 \(b_t>0\)；涉及除以 \(a_t\) 的原图估计时，再要求 \(a_t>0\)。通常 DDPM 的有限带噪步满足这两个条件。干净端点 \(t=0\) 的 \(b_0=0\)，不能直接代入下面的除法公式。

把图片看作 \(d\) 维向量，抽样过程写成：

$$
X_0\sim q_{\mathrm{data}},\qquad
\epsilon\sim\mathcal N(0,I),\qquad X_0\perp\epsilon,
$$

$$
X_t=a_tX_0+b_t\epsilon. \tag{1}
$$

\(X_0\) 是随机抽取的原图，\(X_t\) 是由它与噪声共同构造的带噪随机变量。小写 \(x_0,x_t\) 表示一次具体取值，例如观察到 \(X_t=x_t=0.3\)。**\(\epsilon\) 虽然沿用小写希腊字母，也表示随机变量**。本文沿用 DDPM 的习惯：讨论抽样与条件期望时，\(\epsilon\) 是随机噪声；讨论一次具体训练样本时，同一个符号表示本次抽到的噪声取值，由上下文区分，不再另换字母。

独立符号 \(\perp\) 表示抽原图与抽噪声互不影响。\(I\) 是单位矩阵，\(b_t\) 是累计噪声的标准差，不是反向采样中另外加入的随机项系数。这套正向构造与噪声训练目标见 [DDPM 原论文式（4）、式（14）和算法 1](https://arxiv.org/html/2006.11239v2)。

### 1.2 两组来源，可以给出同一个输入

用一个贯穿全文的一维假想数据集：原图只取 \(+1\) 和 \(-1\)，各占一半。某个时间的加噪公式为

$$
X_t=0.8X_0+0.6\epsilon.
$$

观察到 \(x_t=0\) 时，下面两组数都满足加噪公式：

| 原图取值 \(x_0\) | 对应噪声取值 \(\epsilon\) | 带噪结果 |
|---|---|---|
| \(1\) | \(-4/3\) | \(0.8-0.6\times4/3=0\) |
| \(-1\) | \(4/3\) | \(-0.8+0.6\times4/3=0\) |

训练程序知道本次抽到了哪一组，网络却只接收 \((x_t,t)\)。输入相同，确定性的网络就只能输出同一个预测，无法同时还原两个不同的历史噪声。

这里不是说连续高斯以正概率抽到某个精确数值。对连续变量，\(P(X_t=0)=0\)；条件于一个观察值，要通过条件密度或相应的条件分布理解，也可以先看零附近的小区间再取极限。重点是：**这个观察值有不止一种相容的来源，不能只凭输入唯一反推出标签。**

## 2. 简要回顾：MSE 学习的是噪声条件均值

[DDPM（二）的结尾]({{< relref "posts/ddpm-2#有答案也不表示每道题都能猜中" >}})已经说明：平方误差理想上让网络学习给定输入后的平均噪声，而不是保证还原某次唯一的历史噪声。这里回顾这个结论，接下来重点解释它为何对应 Score；完整的条件风险证明可见 [CG 第 7 节]({{< relref "posts/classifier-guidance#7-为什么噪声-mse-学到的正是这个条件均值" >}})。

固定当前输入与时间，记

$$
m:=\mathbb E_{\epsilon\mid X_t=x_t}[\epsilon],
\qquad
R(u):=\mathbb E_{\epsilon\mid X_t=x_t}[\|\epsilon-u\|^2].
$$

下标说明：平均的是噪声，使用的是观察到 \(X_t=x_t\) 后的噪声条件分布。它一般不再是标准高斯；\(x_t,t\) 固定以后，条件均值 \(m\) 与预测 \(u\) 都是确定向量。

在误差中加减 \(m\)，便得到核心分解：

$$
\boxed{
R(u)=\mathbb E_{\epsilon\mid X_t=x_t}[\|\epsilon-m\|^2]
+\|m-u\|^2.
} \tag{2}
$$

交叉项消失，是因为 \(\mathbb E_{\epsilon\mid X_t=x_t}[\epsilon-m]=m-m=0\)，不是另加了独立性假设。第一项不随预测变化，第二项在 \(u=m\) 时最小，因此

$$
\boxed{\epsilon^*(x_t,t)=\mathbb E_{\epsilon\mid X_t=x_t}[\epsilon].} \tag{3}
$$

这里要求相应二阶矩有限。总体训练目标还会平均不同输入与时间；不受函数容量限制的理想解在几乎处处的输入上取这个条件均值，实际网络只是近似它。

这也不同于 DDPM 中的反向高斯均值：\(\tilde\mu_t(x_t,x_0)\) 平均的是已知原图和当前状态后的上一步 \(X_{t-1}\)，而这里平均的是仅给定当前输入后的噪声 \(\epsilon\)。两者通过噪声参数化相连，不能因为都叫“均值”就当成同一个量。

无条件噪声均值为零，不意味着每个输入处的条件均值都为零。本文接下来要解释的，正是这个随输入变化的条件均值与带噪分布形状的关系。

## 3. Score：对数密度对哪个变量求导？

### 3.1 密度值与密度变化方向是两件事

一维密度 \(q(x)\) 描述样本在 \(x\) 附近有多集中；区间内的概率由密度积分得到，某个点的密度值不等于取到该点的概率。Score 定义为

$$
s(x)=\frac{d}{dx}\log q(x).
$$

对图片这样的多维输入，定义变成

$$
s(x)=\nabla_x\log q(x),\qquad
[s(x)]_i=\frac{\partial\log q(x)}{\partial x_i}. \tag{4}
$$

**求导对象是输入 \(x\)，不是网络参数 \(\theta\)**。它与图片同维度，每个分量说明相应像素的小幅变化怎样影响对数密度，也不是评价图片质量的单个分数。

当密度为正且可微、\(s(x)\ne0\) 时，对足够小的正数 \(h\)，一阶展开给出

$$
\log q(x+h s(x))
=\log q(x)+h\|s(x)\|^2+o(h).
$$

所以 Score 给出了局部提高密度的方向。这里 \(o(h)\) 是比 \(h\) 更小阶的误差，并没有承诺任意大步长都能提高密度。

### 3.2 为什么取对数，而不直接求密度的梯度？

直接计算 \(\nabla_xq(x)\) 当然可以。它也能指出局部提高密度的方向。取对数不是为了创造一个不同的上升方向，而是改变我们衡量变化的尺度。

在 \(q(x)>0\) 的位置，链式法则给出

$$
\nabla_x\log q(x)=\frac{\nabla_xq(x)}{q(x)},
\qquad
\nabla_xq(x)=q(x)\,\nabla_x\log q(x).
$$

由于 \(q(x)\) 是正数，两个非零梯度的方向相同，幅度却相差当前位置的密度值。若作一个很小的位移 \(\delta\)，一阶近似为

$$
q(x+\delta)-q(x)\approx\nabla_xq(x)^\top\delta,
$$

$$
\frac{q(x+\delta)-q(x)}{q(x)}
\approx\nabla_x\log q(x)^\top\delta.
$$

前者描述**绝对密度变化**，后者描述**相对于当前位置密度的变化**。因此，仅说明“取对数后仍然单调”还不够：虽然局部上升方向相同，使用同一个步长更新时，移动距离一般不同。

标准高斯可以直接看出这个区别：

$$
q(x)=\frac1{\sqrt{2\pi}}\exp(-x^2/2),
\qquad q'(x)=-xq(x),
\qquad (\log q)'(x)=-x.
$$

例如在 \(x=3\)，\(q(3)\approx0.00443\)，所以 \(q'(3)\approx-0.0133\)，而 Score 为 \(-3\)。两者都指向左边，但前者乘上了尾部很小的密度。在这个高斯例子中，当 \(x\) 继续增大，密度梯度趋近于零，Score 的绝对值却继续增大。这个例子用于解释两种尺度，并不表示所有分布的尾部都具有同样行为，也不表示直接使用密度梯度不可行。

对数梯度还有一个与这个尺度选择相连的性质。若

$$
q(x)=\frac{\widetilde q(x)}{Z},\qquad
Z=\int\widetilde q(z)\,dz,
$$

那么

$$
\nabla_xq(x)=\frac1Z\nabla_x\widetilde q(x),
\qquad
\nabla_x\log q(x)=\nabla_x\log\widetilde q(x).
$$

归一化常数会影响密度梯度的幅度，却在对数梯度中消失。因此即使只知道一个差常数倍的未归一化密度，仍可得到相同的 Score。这要求 \(Z\) 不依赖求导变量 \(x\)；若对模型参数求导，不能不加区分地删掉它。[Yang Song 的作者教程](https://yang-song.net/blog/2021/score/)讨论了这一性质。

对本篇而言，更直接的理由是：后面的高斯加噪与贝叶斯推导会给出

$$
\nabla_{x_t}\log q_t(x_t)
=-\frac{\mathbb E_{\epsilon\mid X_t=x_t}[\epsilon]}{b_t}.
$$

右边正是噪声网络理想上学到的量经过缩放。若换成密度梯度，则要再乘上密度本身：

$$
\nabla_{x_t}q_t(x_t)
=-\frac{q_t(x_t)}{b_t}
\mathbb E_{\epsilon\mid X_t=x_t}[\epsilon].
$$

噪声预测并没有同时给出 \(q_t(x_t)\)。因此，**与噪声条件均值直接对应的是对数密度梯度**，而不是密度的绝对变化率。下面会证明这条恒等式，而不把它当作预先假定的结论。

### 3.3 高斯例子：方向、尺度与零点

一维高斯的方差为 \(v>0\)：

$$
q(x)=\frac1{\sqrt{2\pi v}}
\exp\left(-\frac{(x-\mu)^2}{2v}\right).
$$

取对数再求导：

$$
\log q(x)=-\frac12\log(2\pi v)-\frac{(x-\mu)^2}{2v},
\qquad s(x)=-\frac{x-\mu}{v}. \tag{5}
$$

在中心左侧，Score 为正；在右侧为负；中心处为零。相同偏移下，方差越小，Score 的绝对值越大。标准高斯取 \(\mu=0,v=1\)，就得到 \(s(x)=-x\)。

但“Score 为零”一般只说明一阶变化率为零，**不一定是密度峰顶**。两个分离的对称高斯混合时，中间的低谷也可以有零 Score。不要把单高斯的特殊形状当成所有分布的性质。

## 4. 已知原图时，条件 Score 为什么等于负噪声除以标准差？

固定原图 \(x_0\) 和时间 \(t\)，随机变化的只有噪声。式（1）给出条件密度

$$
q_t(x_t\mid x_0)
=\mathcal N(x_t;a_tx_0,b_t^2I).
$$

这里的 \(q_t(x_t\mid x_0)\) 与 DDPM 中的 \(q(x_t\mid x_0)\) 是同一个加噪核，下标只是强调时间。将它完整写开，再取对数：

$$
\begin{aligned}
q_t(x_t\mid x_0)
&=(2\pi b_t^2)^{-d/2}
\exp\left(-\frac{\|x_t-a_tx_0\|^2}{2b_t^2}\right),\\
\log q_t(x_t\mid x_0)
&=-\frac d2\log(2\pi b_t^2)
-\frac{\|x_t-a_tx_0\|^2}{2b_t^2}.
\end{aligned}
$$

对 \(x_t\) 求梯度，固定 \(x_0,t\)。常数项为零，而平方项的梯度为 \(2(x_t-a_tx_0)\)，所以

$$
\nabla_{x_t}\log q_t(x_t\mid x_0)
=-\frac{x_t-a_tx_0}{b_t^2}.
$$

这里令 \(\epsilon\) 表示本次抽到的噪声取值。对于一组相容的取值 \(x_t=a_tx_0+b_t\epsilon\)，残差等于 \(b_t\epsilon\)，从而

$$
\boxed{
\nabla_{x_t}\log q_t(x_t\mid x_0)=-\frac{\epsilon}{b_t}.
} \tag{6}
$$

分母由方差 \(b_t^2\) 变成标准差 \(b_t\)，是因为残差本身带着一个 \(b_t\)。负号意味着方向与离开高斯中心的残差相反。

这个中心是**固定原图后的条件分布中心 \(a_tx_0\)**，一般不是干净原图 \(x_0\)。我们算出的也只是这张原图对应的条件 Score，尚未得到所有原图混合后的边缘 Score。单次训练标签 \(\epsilon\) 经过缩放对应式（6），不能直接当作整体 Score。

## 5. 从一般原图分布出发，为什么 Score 要按后验平均？

### 5.1 从全概率公式走到密度积分

固定时间 \(t\)，先假设原图分布具有密度 \(q_{\mathrm{data}}(x_0)\)。令 \(A\) 是带噪图片空间中的任意可测区域。一张带噪图片落入 \(A\)，可以由不同原图产生；对所有原图来源应用全概率公式：

$$
P(X_t\in A)
=\int P(X_t\in A\mid X_0=x_0)
q_{\mathrm{data}}(x_0)\,dx_0.
$$

这里先固定一张原图，计算从它加噪后落入 \(A\) 的概率，再按原图的先验密度平均。将区域概率写成密度积分，用 \(z\) 表示区域中的带噪坐标，以免与原图坐标 \(x_0\) 混淆：

$$
\begin{aligned}
\int_A q_t(z)\,dz
&=\int q_{\mathrm{data}}(x_0)
\left[\int_A q_t(z\mid x_0)\,dz\right]dx_0\\
&=\int_A\left[\int q_{\mathrm{data}}(x_0)
q_t(z\mid x_0)\,dx_0\right]dz.
\end{aligned}
$$

被积函数非负，因此可交换这两个积分的次序。等式对任意区域 \(A\) 都成立，所以两边的密度几乎处处相等；本文的正噪声高斯混合密度连续，可以在每个位置写成

$$
\boxed{
q_t(x_t)=\int q_{\mathrm{data}}(x_0)
q_t(x_t\mid x_0)\,dx_0.
} \tag{7}
$$

在这个式子中，当前求值位置 \(x_t\) 固定，积分遍历候选原图 \(x_0\)。同一件事也可写为

$$
q_t(x_t)=\mathbb E_{X_0\sim q_{\mathrm{data}}}
[q_t(x_t\mid X_0)].
$$

这不是平均观察值，而是按**原图先验**平均各张原图对应的**条件密度值**。离散原图用求和代替积分；实际数据集上就是对各张图片加噪形成的密度按采样概率加权。即使原始数据没有平滑密度，正噪声高斯核仍会形成正的、平滑的带噪密度。

我们现在知道密度怎样平均。下一步要算的是这份边缘密度的对数梯度，不能把对数直接移入上述积分，也不能预先假定条件 Score 仍使用同样的先验权重。

### 5.2 求导与积分交换需要什么条件？

下面要求能把对 \(x_t\) 的梯度放进积分。一个充分条件是：条件密度对输入可微，且在所考察位置的某个邻域内，其输入梯度的模由一个对原图分布可积的函数统一控制。这样可用支配收敛定理交换求导与积分。

对本文固定 \(b_t>0\) 的高斯核，梯度是“残差乘高斯密度”，作为残差的函数有界，因此可以给出这样的控制。有限数据集的求和也没有交换无限积分的问题。讨论后验均值时另需相应矩存在；有界像素数据满足这些要求。

### 5.3 从密度求导，逐步认出贝叶斯后验

先用对数求导，再交换求导与积分：

$$
\begin{aligned}
s_t(x_t)
&:=\nabla_{x_t}\log q_t(x_t)\\
&=\frac1{q_t(x_t)}\nabla_{x_t}
\int q_{\mathrm{data}}(x_0)q_t(x_t\mid x_0)\,dx_0\\
&=\frac1{q_t(x_t)}
\int q_{\mathrm{data}}(x_0)\nabla_{x_t}q_t(x_t\mid x_0)\,dx_0.
\end{aligned}
$$

为了把密度梯度写成条件 Score，在积分内乘除同一个正的条件密度，即使用

$$
\nabla_{x_t}q_t(x_t\mid x_0)
=q_t(x_t\mid x_0)\nabla_{x_t}\log q_t(x_t\mid x_0),
$$

于是

$$
\begin{aligned}
s_t(x_t)
=\int &\frac{q_{\mathrm{data}}(x_0)q_t(x_t\mid x_0)}{q_t(x_t)}\\
&\cdot\nabla_{x_t}\log q_t(x_t\mid x_0)\,dx_0.
\end{aligned}
$$

由贝叶斯公式，第一行的因子就是原图的后验密度：

$$
q(x_0\mid X_t=x_t)
=\frac{q_{\mathrm{data}}(x_0)q_t(x_t\mid x_0)}{q_t(x_t)}.
$$

因此

$$
\boxed{
s_t(x_t)
=\mathbb E_{X_0\sim q(\cdot\mid X_t=x_t)}
\left[\nabla_{x_t}\log q_t(x_t\mid X_0)\right].
} \tag{8}
$$

这里先固定一张候选原图，对输入求梯度，再把这个梯度在当前观察值处的值按原图后验平均。**求梯度时变化的是输入坐标，取期望时变化的是候选原图**，是两层不同的操作。写在条件位置的 \(X_0\) 仍然可以是外层期望遍历的随机变量。

式（8）不要求条件核一定是高斯；满足相应正性、可微和交换条件时，它就是一般的混合分布恒等式。高斯假设的作用，是让内部的条件 Score 能够直接计算。

## 6. 为什么平均原图对应的残差，等于平均噪声？

把条件高斯 Score 代入式（8）：

$$
\begin{aligned}
s_t(x_t)
&=\mathbb E_{X_0\sim q(\cdot\mid X_t=x_t)}
\left[-\frac{x_t-a_tX_0}{b_t^2}\right]\\
&=-\frac1{b_t}
\mathbb E_{X_0\sim q(\cdot\mid X_t=x_t)}
\left[\frac{x_t-a_tX_0}{b_t}\right].
\end{aligned} \tag{9}
$$

外面一个 \(b_t\) 被提出去了，里面剩下的量恰好是候选原图对应的噪声。为什么可以把它换成 \(\epsilon\) 的条件均值？需要回到随机变量关系，而不是只对符号做替换。

### 6.1 两个量都是随机变量，也可以满足同一个等式

在原始联合抽样中，式（1）对每次抽样都成立，因此随机变量之间有精确等式

$$
\epsilon=\frac{X_t-a_tX_0}{b_t}.
$$

已知 \(X_t=x_t\) 后，这个关系在相应条件分布下变成

$$
\epsilon=\frac{x_t-a_tX_0}{b_t}.
$$

**\(x_t\) 已经固定，但 \(X_0\) 和 \(\epsilon\) 仍然未知、仍然随机。** 它们不能各自任意变化：每个候选原图都确定了对应噪声。原图为 \(+1\) 时必须配 \(-4/3\)，原图为 \(-1\) 时必须配 \(4/3\)，才能保持刚才的观察值为零。

原始抽样的 \(X_0\perp\epsilon\) 不意味着给定 \(X_t\) 后还独立。观察它们的组合结果，相当于增加了一条约束；在这个例子中，知道其中一个，就知道另一个。**先验独立不能直接搬到条件分布里。**

### 6.2 同一个条件分布，换一种遍历方式

为了写清期望，令 \(Q_t\) 表示式（1）诱导的 \((X_0,\epsilon,X_t)\) 联合分布。则

$$
\begin{aligned}
\mathbb E_{Q_t}[\epsilon\mid X_t=x_t]
&=\mathbb E_{Q_t}\left[
\left.\frac{X_t-a_tX_0}{b_t}\right|X_t=x_t\right]\\
&=\mathbb E_{X_0\sim q(\cdot\mid X_t=x_t)}
\left[\frac{x_t-a_tX_0}{b_t}\right]\\
&=\int\frac{x_t-a_tx_0}{b_t}
q(x_0\mid X_t=x_t)\,dx_0.
\end{aligned} \tag{10}
$$

第一步替换两个相等的随机变量；第二步把条件中已知的 \(X_t\) 换成 \(x_t\)，剩下只需对未知原图的后验平均；第三步把这个期望展开为积分。离散原图用求和，含义相同。

也可以说，噪声条件分布是原图后验经过映射 \(x_0\mapsto(x_t-a_tx_0)/b_t\) 后得到的分布。于是，按原图后验遍历、算出对应噪声再平均，与直接按噪声条件分布平均，当然得到同一个值。这没有重新独立抽噪声，更没有把条件噪声分布当回标准高斯。

将式（10）放回式（9），得到

$$
\boxed{
s_t(x_t)
=-\frac1{b_t}\mathbb E_{\epsilon\mid X_t=x_t}[\epsilon].
} \tag{11}
$$

为避免把不同的期望混在一起，可以对照：

| 表达式 | 遍历谁 | 权重来自哪里 | 平均什么 |
|---|---|---|---|
| \(\mathbb E_{X_0\sim q_{\mathrm{data}}}[q_t(x_t\mid X_0)]\) | 原图 | 原图先验 | 条件密度，得到边缘密度 |
| \(\mathbb E_{X_0\sim q(\cdot\mid X_t=x_t)}[\nabla_{x_t}\log q_t(x_t\mid X_0)]\) | 原图 | 观察后的原图后验 | 条件 Score，得到边缘 Score |
| \(\mathbb E_{\epsilon\mid X_t=x_t}[\epsilon]\) | 噪声 | 观察后的噪声条件分布 | 噪声，得到最优噪声预测 |

每一行都固定当前 \(x_t,t\)。看懂期望时，要同时检查“平均谁、按什么分布平均、括号里算什么”，不能只看到一个 \(\mathbb E\) 就把它当作同一种平均。

## 7. 噪声预测、Score 与原图估计：精确转换和理想含义

### 7.1 噪声 MSE 与边缘 Score，终于出现同一个条件均值

式（3）与式（11）分别来自 MSE 最优化和边缘密度求导：

$$
\epsilon^*(x_t,t)=\mathbb E_{\epsilon\mid X_t=x_t}[\epsilon],
\qquad s_t(x_t)=-\frac{\epsilon^*(x_t,t)}{b_t}.
$$

实际网络 \(\epsilon_\theta\) 近似 \(\epsilon^*\)。因此可直接定义

$$
\boxed{
s_\theta(x_t,t):=-\frac{\epsilon_\theta(x_t,t)}{b_t}
\approx s_t(x_t).
} \tag{12}
$$

等号是同一份网络输出的代数转换，近似号才是学习误差。网络不需要显式输出密度，也不需要对噪声网络再求一次输入梯度；它输出的向量经过缩放，就作为 Score 估计。但任意实际向量网络不保证严格等于某个归一化密度的梯度。

这里要分清三样东西：单次噪声标签 \(\epsilon\)、理想噪声条件均值 \(\epsilon^*\)、实际噪声预测 \(\epsilon_\theta\)。**单次标签经缩放是条件 Score 的监督信号；经过条件平均，才得到边缘 Score。**

### 7.2 把理想预测代入原图估计，到底做了什么？

要求 \(a_t>0\)。实际的原图估计定义为

$$
\hat x_{0,\theta}(x_t,t)
:=\frac{x_t-b_t\epsilon_\theta(x_t,t)}{a_t}. \tag{13}
$$

先不涉及网络，从真实随机变量关系出发，在给定观察值后移项：

$$
X_0=\frac{x_t-b_t\epsilon}{a_t}
\qquad\text{（在给定 }X_t=x_t\text{ 的条件下）}.
$$

两边在同一个条件下取期望，固定的 \(x_t,a_t,b_t\) 可以提出：

$$
\mathbb E_{X_0\mid X_t=x_t}[X_0]
=\frac{x_t-b_t\mathbb E_{\epsilon\mid X_t=x_t}[\epsilon]}{a_t}. \tag{14}
$$

现在才把式（13）里的网络预测换成理想预测：

$$
\begin{aligned}
\hat x_0^*(x_t,t)
&:=\frac{x_t-b_t\epsilon^*(x_t,t)}{a_t}\\
&=\frac{x_t-b_t\mathbb E_{\epsilon\mid X_t=x_t}[\epsilon]}{a_t}\\
&=\mathbb E_{X_0\mid X_t=x_t}[X_0].
\end{aligned} \tag{15}
$$

最后一步成立，是因为它与式（14）的右边完全相同。因此理想的一次去噪估计是**原图后验均值**，不保证等于本次实际原图。

在 \(x_t=0\) 的对称例子中，这个均值为 \((1+(-1))/2=0\)，而数据里的原图只有 \(+1,-1\)，根本没有零。后验均值可以处在多个可能答案之间，不能把它自动当作真实数据样本，更不能当作完整采样最终必然输出的图片。

### 7.3 三种表示的全部转换

由式（12）得到 \(\epsilon_\theta=-b_ts_\theta\)，代入式（13）：

$$
\boxed{
\begin{aligned}
s_\theta&=-\frac{\epsilon_\theta}{b_t},\\
\hat x_{0,\theta}
&=\frac{x_t-b_t\epsilon_\theta}{a_t}
=\frac{x_t+b_t^2s_\theta}{a_t}.
\end{aligned}
} \tag{16}
$$

反过来，给定同一份原图估计也可计算

$$
\epsilon_\theta=\frac{x_t-a_t\hat x_{0,\theta}}{b_t},
\qquad
s_\theta=\frac{a_t\hat x_{0,\theta}-x_t}{b_t^2}.
$$

这些是固定同一个输入与时间、\(a_t>0,b_t>0\) 时的精确代数恒等式。三个网络分别训练，实际输出不保证彼此满足它们；对原图估计做裁剪等额外处理后，也要重新检查一致性。

理想情况下则有分布层面的等式

$$
\boxed{
a_t\mathbb E_{X_0\mid X_t=x_t}[X_0]
=x_t+b_t^2s_t(x_t).
} \tag{17}
$$

右边是当前带噪状态加上由 Score 决定的修正，得到的是**缩放后的原图后验均值**。再除以 \(a_t\)，才是原图后验均值。这是去噪估计公式，还没有规定怎样把一群随机初始状态变成目标分布的样本。

## 8. 沿 Score 一直走，为什么不等于按分布采样？

标准高斯的 Score 是 \(s(x)=-x\)。若每次沿它做一次梯度上升，取 \(0<h<1\)：

$$
x^{(k+1)}=x^{(k)}+h s(x^{(k)})=(1-h)x^{(k)}.
$$

反复更新后

$$
x^{(k)}=(1-h)^k x^{(0)}\longrightarrow0.
$$

不仅单个点会靠近零，一群点也会收缩。即使初始状态已经服从目标分布 \(X^{(0)}\sim\mathcal N(0,1)\)，更新后也变成

$$
X^{(k)}\sim\mathcal N\left(0,(1-h)^{2k}\right).
$$

方差逐渐趋于零，分布最终集中到零点，反而破坏了原本正确的标准高斯分布。从标准高斯采样，需要结果按相应概率散布在不同位置，不是让所有结果都去峰顶。

所以，Score 描述固定噪声等级下的局部密度变化，采样算法还要规定整群样本怎样演化。**这个反例否定的是“直接沿固定密度的 Score 梯度上升就等于采样”，不是否定所有确定性采样方法。**

扩散模型学到的是不同噪声等级下的一族 \(s_t\)。如何配合时间变化、漂移和必要的随机项，把高噪声分布变成低噪声分布，需要额外的采样理论。[Score-SDE 论文](https://arxiv.org/abs/2011.13456)讨论了相应的反向 SDE 与概率流 ODE。本文只把问题留在这里；Score Matching 如何组织训练目标，以及 SDE、ODE 怎样使用这些 Score，可以在后续逐步建立。

## 参考资料

- Jonathan Ho、Ajay Jain、Pieter Abbeel，**Denoising Diffusion Probabilistic Models**。正向高斯核见式（4），噪声 MSE 见式（14）和算法 1。[原论文](https://arxiv.org/html/2006.11239v2) · [作者项目页](https://hojonathanho.github.io/diffusion/)
- Yang Song，**Generative Modeling by Estimating Gradients of the Data Distribution**。Score 的输入梯度定义、归一化常数消去和采样的概念背景。[作者教程](https://yang-song.net/blog/2021/score/)
- Yang Song 等，**Score-Based Generative Modeling through Stochastic Differential Equations**。不同噪声等级的 Score 与连续时间生成过程的后续阅读；本文不展开其中的 SDE、ODE 推导。[原论文](https://arxiv.org/abs/2011.13456)
- 站内衔接：[DDPM：样本、噪声预测与损失]({{< relref "posts/ddpm-2" >}}) · [Classifier Guidance]({{< relref "posts/classifier-guidance" >}}) · [Classifier-Free Guidance]({{< relref "posts/classifier-free-guidance" >}})。

## 9. 小结

本文的连接可以分成两步：噪声 MSE 的理想预测是给定带噪输入后的噪声条件均值；高斯加噪下，这个条件均值乘以 \(-1/b_t\)，得到带噪边缘分布的 Score。由同一份预测构造的噪声、Score 与原图估计可以精确换算，但它们对真实统计量的估计仍可能有误差。一次后验均值估计也不能替代完整采样。
