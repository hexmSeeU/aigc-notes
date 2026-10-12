{
  "title": "扩散模型引导（二）：Classifier-Free Guidance",
  "description": "从条件与无条件 Score 之差推导 CFG，证明随机丢条件训练为何成立，再接入 DDIM，解释引导尺度、多样性与采样边界。",
  "slug": "classifier-free-guidance",
  "collection": "guidance",
  "episode": 2,
  "tags": [
    "Classifier-Free Guidance",
    "扩散模型"
  ],
  "summary": "从条件与无条件 Score 之差推导 CFG，证明随机丢条件训练为何成立，再接入 DDIM，解释引导尺度、多样性与采样边界。",
  "date": "2026-10-12T15:20:00+08:00",
  "categories": [
    "基础"
  ],
  "draft": false,
  "math": true
}

普通条件扩散模型已经能按标签生成图片。Classifier-Free Guidance（CFG）进一步提供一个采样时可调的尺度，让结果更偏向目标类别特征鲜明的样本。它用同一个去噪网络的条件预测与无条件预测构造引导，不需要另外训练带噪分类器。

这篇是扩散模型引导合集的第 2 篇，接着 [Classifier Guidance（CG）](../classifier-guidance/) 往下讲。主线是：为什么还需要引导，两个预测的差从哪里来，随机丢弃条件为什么能学到所需预测，以及怎样把 CFG 接入 [DDIM](../ddim-derivation/) 采样。

全文围绕一个实际操作展开：对当前同一张带噪图片，分别问网络“知道类别时预测什么噪声”和“不知道类别时预测什么噪声”，再组合成一份预测，交给采样器更新图片。

## 1. 已经有条件生成，为什么还需要 guidance？

假设目标是生成手写数字 \(y=3\)。普通条件去噪网络接收：

$$
\epsilon_\theta(x_t,t,y=3).
\tag{1}
$$

其中 \(x_t\) 是当前带噪图片，\(t\) 表示噪声等级，标签 \(y=3\) 告诉网络目标类别。普通条件模型本来就具备生成指定类别的能力。

真实数据里的“3”有很多写法：有的笔画清楚，有的比较潦草，还有的容易与“8”混淆。只要它们确实属于数字 3，理想的条件模型就应该按照数据中的比例覆盖这些写法。

但在使用时，我们可能希望生成一眼就能认出的“3”，并愿意少生成一些模糊或少见的写法。这需要一个在训练完成后仍能调整的旋钮。

因此要区分两个目标：

- **条件生成**：根据标签，学习并生成这个类别的数据
- **加强引导**：在已有条件生成能力上，更偏向目标类别特征鲜明的样本

CG 在条件模型上加入带噪分类器的输入梯度：

$$
s_{\mathrm{CG}}(x_t,t,y)
=s_\theta(x_t,t,y)
+w\nabla_{x_t}\log p_\phi(y\mid x_t,t).
\tag{2}
$$

这里 \(s_\theta\) 是去噪模型的 Score 估计；\(p_\phi\) 是另外训练的分类器；梯度对当前图片 \(x_t\) 求导，类别、时间和模型参数保持固定；\(w\) 控制额外引导强度。因为基模型已经接收条件，所以 \(w=0\) 就是普通条件生成。

CG 需要让分类器认识不同噪声等级的图片，采样时还要计算分类器对输入的梯度。CFG 要解决的问题是：**能否保留额外引导，同时省掉这个分类器？** [CFG 原论文 §1、§3](https://arxiv.org/html/2207.12598v1#S3)

## 2. 先统一符号与讨论对象

沿用 DDPM 和上一篇 CG 的离散时间记号：

$$
\alpha_t=1-\beta_t,
\qquad
\bar\alpha_t=\prod_{i=1}^{t}\alpha_i,
\qquad
\sigma_t=\sqrt{1-\bar\alpha_t}.
\tag{3}
$$

前向加噪写成：

$$
X_t=\sqrt{\bar\alpha_t}X_0+\sigma_t\epsilon,
\qquad \epsilon\sim\mathcal N(0,I).
\tag{4}
$$

大写 \(X_0,X_t,Y\) 表示随机变量，小写 \(x_0,x_t,y\) 表示它们的具体取值。\(\sigma_t\) 是前向边缘分布的噪声标准差，不是 DDIM 中另行注入的随机噪声系数。

后面会用到这些分布：

- \(q_{\mathrm{data}}(x_0,y)\)：原始图片与标签的联合数据分布
- \(q_t(x_t\mid y)\)：指定类别经过前向加噪后的条件密度
- \(q_t(x_t)\)：所有类别按数据比例混合、加噪后的边缘密度
- \(q_t(y\mid x_t)\)：观察到带噪图片后，它属于类别 \(y\) 的后验概率

真实的条件与无条件 Score 分别是：

$$
s_{\mathrm c}^*(x_t,t,y)=\nabla_{x_t}\log q_t(x_t\mid y),
\qquad
s_{\mathrm u}^*(x_t,t)=\nabla_{x_t}\log q_t(x_t).
\tag{5}
$$

星号表示真实分布对应的理想量；下标 \(\theta\) 表示网络估计。**“无条件”只是不把类别 \(Y\) 当作输入，预测仍然依赖 \(x_t\) 和 \(t\)。** 以下 Score 与噪声之间的除法关系讨论 \(\sigma_t\gt 0\) 的带噪时刻。

原论文采用连续的对数信噪比 \(\lambda\)，用 \(\mathbf z_\lambda,\mathbf c\) 表示带噪输入与条件。这里统一换成 \(x_t,t,y\)，保留方法本身。

## 3. 去掉分类器后，引导方向从哪里来？

这一步先只讨论真实分布。固定时间 \(t\) 和目标类别 \(y\)，由贝叶斯公式：

$$
q_t(x_t\mid y)
=\frac{q_t(x_t)q_t(y\mid x_t)}{q(y)}.
\tag{6}
$$

取对数：

$$
\log q_t(x_t\mid y)
=\log q_t(x_t)+\log q_t(y\mid x_t)-\log q(y).
\tag{7}
$$

对 \(x_t\) 求梯度。类别先验 \(q(y)\) 与当前求导变量无关，因此最后一项消失：

$$
\nabla_{x_t}\log q_t(x_t\mid y)
=\nabla_{x_t}\log q_t(x_t)
+\nabla_{x_t}\log q_t(y\mid x_t).
\tag{8}
$$

移项得到：

$$
\boxed{
\nabla_{x_t}\log q_t(y\mid x_t)
=s_{\mathrm c}^*(x_t,t,y)-s_{\mathrm u}^*(x_t,t)
}.
\tag{9}
$$

也就是说，**条件 Score 减去无条件 Score，就是局部提高目标类别对数概率的方向**。只要知道两个 Score，就能相减得到这个方向，无须先计算分类概率再对它求导。这是原论文“隐式分类器”解释的出发点。[原论文 §3.2](https://arxiv.org/html/2207.12598v1#S3.SS2)

例如，对同一个 \(x_t,t\)，假设无条件 Score 为 \((2,1)\)，条件 Score 为 \((2,4)\)，相减就是 \((0,3)\)。两个预测共有的第一个分量抵消，第二个分量的差异保留下来。这只是二维讲解例子，数值没有实验含义。

需要从现在开始区分两层：

1. 对真实分布的 Score，贝叶斯关系是精确恒等式
2. 对网络输出，相减得到的是引导方向的估计；一般不能保证它恰好是某个分类器对数概率的梯度

网络直接输出向量，并没有被强制约束成某个标量函数的梯度。原论文也明确保留了这一差别。

## 4. 从 Score 组合到 CFG 噪声公式

### 4.1 先确定要组合的 Score

我们希望在普通条件 Score 上，再加入 \(w\) 倍的类别方向。用两个网络 Score 估计的差代替它：

$$
\begin{aligned}
s_{\mathrm{CFG}}
&:=s_\theta(x_t,t,y)
+w\left[s_\theta(x_t,t,y)-s_\theta(x_t,t,\varnothing)\right]\\
&=(1+w)s_\theta(x_t,t,y)-w s_\theta(x_t,t,\varnothing).
\end{aligned}
\tag{10}
$$

\(\varnothing\) 表示不提供类别的空条件；\(w\geq0\) 表示在条件预测上额外增强多少。

### 4.2 换成网络实际输出的噪声预测

噪声预测与 Score 估计的关系是：

$$
s_\theta(x_t,t,y)
:=-\frac{\epsilon_\theta(x_t,t,y)}{\sigma_t},
\qquad
s_\theta(x_t,t,\varnothing)
:=-\frac{\epsilon_\theta(x_t,t,\varnothing)}{\sigma_t}.
\tag{11}
$$

两个分母相同，因为比较的是同一个 \(x_t,t\)。第 6 节会补全“为什么 MSE 训练的噪声预测能估计 Score”的连接。现在直接代入：

$$
\begin{aligned}
s_{\mathrm{CFG}}
&=(1+w)\left[-\frac{\epsilon_\theta(x_t,t,y)}{\sigma_t}\right]
-w\left[-\frac{\epsilon_\theta(x_t,t,\varnothing)}{\sigma_t}\right]\\
&=-\frac{(1+w)\epsilon_\theta(x_t,t,y)
-w\epsilon_\theta(x_t,t,\varnothing)}{\sigma_t}.
\end{aligned}
\tag{12}
$$

把引导后的噪声预测定义为 \(-\sigma_t s_{\mathrm{CFG}}\)，得到：

$$
\boxed{
\hat\epsilon_{\mathrm{CFG}}
=(1+w)\epsilon_\theta(x_t,t,y)
-w\epsilon_\theta(x_t,t,\varnothing)
}.
\tag{13}
$$

这对应 [原论文式（6）](https://arxiv.org/html/2207.12598v1#S3.SS2)。一旦选定 Score 的组合规则，从 Score 换成噪声预测就是精确代数变换；网络对真实 Score 的近似误差依然存在。

### 4.3 为什么会有负的无条件项？

把公式重新写成：

$$
\hat\epsilon_{\mathrm{CFG}}
=\underbrace{\epsilon_\theta(x_t,t,y)}_{\text{普通条件预测}}
+w\underbrace{\left[\epsilon_\theta(x_t,t,y)
-\epsilon_\theta(x_t,t,\varnothing)\right]}_{\text{条件预测与无条件预测的差}}.
\tag{14}
$$

当 \(w\gt 0\) 时，它从条件预测出发，沿着“无条件预测指向条件预测”的方向继续向外走。这是**外推**，结果可以超过两种预测的取值范围。

假设某个噪声预测分量的无条件值是 \(2\)，条件值是 \(3\)：

$$
w=0:\ 3;\qquad
w=1:\ 3+(3-2)=4;\qquad
w=2:\ 3+2(3-2)=5.
\tag{15}
$$

负的无条件项来自这次外推，不代表生成一张“负图片”。这些数也是噪声预测分量，不是图片质量分数。还要注意，噪声与 Score 之间有负号，不能直接把噪声差当成图片空间中的类别概率上升方向。

## 5. 一个网络怎样学会两种预测？

原论文用同一个去噪网络完成条件与无条件预测：输入真实标签时做条件预测，输入 \(\varnothing\) 时做无条件预测。两种输入共用参数 \(\theta\)；“两个分支”仅指两种输入方式。[原论文算法 1](https://arxiv.org/html/2207.12598v1#S3.SS2)

训练时先取出一张图片及其标签，例如 \((x_0,y=3)\)，抽取时间 \(t\) 和高斯噪声 \(\epsilon\)，构造：

$$
x_t=\sqrt{\bar\alpha_t}x_0+\sigma_t\epsilon.
\tag{16}
$$

再以概率 \(p_{\mathrm{uncond}}\) 丢弃条件。把实际送入网络的条件写为 \(\tilde y\)：

$$
\tilde y=
\begin{cases}
y,&\text{概率为 }1-p_{\mathrm{uncond}},\\
\varnothing,&\text{概率为 }p_{\mathrm{uncond}}.
\end{cases}
\tag{17}
$$

丢弃决定独立于图片、标签、时间和噪声。无论是否保留标签，这个样本的损失都是：

$$
\ell_\theta
=\left\|\epsilon_\theta(x_t,t,\tilde y)-\epsilon\right\|_2^2.
\tag{18}
$$

保留条件时，网络看见 \((x_t,t,3)\)；丢弃条件时，网络看见 \((x_t,t,\varnothing)\)。**两种情况的监督答案都是构造 \(x_t\) 时抽到的噪声 \(\epsilon\)。**

因此：

- 丢掉的是类别输入，原图仍然用于加噪，噪声仍然用于计算损失
- 无条件训练仍然有明确的监督答案
- \(\varnothing\) 必须表示“不提供类别”，不能拿真实的数字类别 \(0\) 来代替
- 按原论文的训练方式，一个样本只需按抽到的条件状态计算一次预测；采样时才需要两种预测

这里 \(p_{\mathrm{uncond}}\) 是训练参数，而 \(w\) 是采样参数。训练完成后可以改变 \(w\)，不必为每个引导尺度重新训练。

## 6. 为什么随机丢弃条件能学到无条件 Score？

训练操作本身还不够。我们需要证明：空条件输入下的网络，确实在学习无条件分布对应的噪声预测。

### 6.1 独立丢弃标签，不会改变原有数据分布

写清全部随机来源：

$$
(X_0,Y)\sim q_{\mathrm{data}}(x_0,y),\qquad
t\sim\pi(t),\qquad
\epsilon\sim\mathcal N(0,I),
\tag{19}
$$

$$
D\sim\operatorname{Bernoulli}(p_{\mathrm{uncond}}),
\qquad
D\perp(X_0,Y,t,\epsilon).
\tag{20}
$$

\(\pi(t)\) 是训练时抽取噪声等级的分布；\(D=1\) 表示丢弃标签，\(D=0\) 表示保留标签。时间、噪声也独立于数据抽取，并且彼此独立。定义：

$$
X_t=\sqrt{\bar\alpha_t}X_0+\sigma_t\epsilon,
\qquad
\tilde Y=
\begin{cases}
Y,&D=0,\\
\varnothing,&D=1.
\end{cases}
\tag{21}
$$

因为 \(D\) 独立于数据，被选中丢弃标签的样本仍然具有原来的联合分布：

$$
q(x_0,y\mid D=1)=q_{\mathrm{data}}(x_0,y).
\tag{22}
$$

网络此时看不到 \(Y\)，把类别求和消去：

$$
\begin{aligned}
q(x_0\mid D=1)
&=\sum_y q(x_0,y\mid D=1)\\
&=\sum_y q_{\mathrm{data}}(x_0,y)\\
&=q_{\mathrm{data}}(x_0).
\end{aligned}
\tag{23}
$$

这就是完整的无条件数据分布。若数据中数字 3 占 \(20\%\)，独立丢弃标签后，空条件训练样本中数字 3 的总体占比仍是 \(20\%\)，不会自动变成各类别均匀。

这里使用离散类别；连续条件的对应操作是对条件积分。独立性很重要：如果某些类别更容易被丢弃，空条件样本的类别比例就可能被改变。

### 6.2 整体损失拆成条件任务与无条件任务

完整训练目标是：

$$
\begin{aligned}
L(\theta)
={}&\mathbb E_{(X_0,Y)\sim q_{\mathrm{data}}}
\mathbb E_{t\sim\pi}\mathbb E_{\epsilon\sim\mathcal N(0,I)}\mathbb E_D\\
&\left[\left\|\epsilon_\theta(X_t,t,\tilde Y)-\epsilon\right\|_2^2\right].
\end{aligned}
\tag{24}
$$

先对丢弃开关 \(D\) 分情况取期望，再利用它的独立性：

$$
\begin{aligned}
L(\theta)
={}&(1-p_{\mathrm{uncond}})
\mathbb E_{(X_0,Y)\sim q_{\mathrm{data}}}
\mathbb E_{t\sim\pi}\mathbb E_{\epsilon\sim\mathcal N(0,I)}\\
&\quad
\left[\left\|\epsilon_\theta(X_t,t,Y)-\epsilon\right\|_2^2\right]\\
&+p_{\mathrm{uncond}}
\mathbb E_{X_0\sim q_{\mathrm{data}}(x_0)}
\mathbb E_{t\sim\pi}\mathbb E_{\epsilon\sim\mathcal N(0,I)}\\
&\quad
\left[\left\|\epsilon_\theta(X_t,t,\varnothing)-\epsilon\right\|_2^2\right].
\end{aligned}
\tag{25}
$$

第一项平均的是带标签数据、时间和噪声。第二项不再使用标签，所以可以从联合数据分布中把 \(Y\) 求和掉，留下 \(X_0\sim q_{\mathrm{data}}(x_0)\)。

**第二项正是普通无条件扩散模型的噪声 MSE，前面乘了权重 \(p_{\mathrm{uncond}}\)。** 在理想函数空间中，只要 \(0\lt p_{\mathrm{uncond}}\lt 1\)，两个任务都有训练权重；有限共享网络中，这个比例也会影响两种任务分到的训练机会。

### 6.3 MSE 的最优预测是噪声条件均值

固定空条件输入 \(X_t=x_t,t\)。网络需要输出一个向量 \(a\)，最小化：

$$
R(a)=\mathbb E_{\epsilon}\left[\|\epsilon-a\|_2^2\mid X_t=x_t,t,D=1\right].
\tag{26}
$$

这里的期望对噪声的后验分布 \(q(\epsilon\mid X_t=x_t,t,D=1)\) 取平均；已经观察到 \(x_t\) 后，不能重新把这个后验当作未加条件的标准高斯。下标 \(\epsilon\) 标明被平均的随机变量，条件竖线后面标明已知信息。

把当前已知信息简写为 \(H=\{X_t=x_t,t,D=1\}\)，令 \(m=\mathbb E_{\epsilon}[\epsilon\mid H]\)。展开平方：

$$
\begin{aligned}
R(a)
&=\mathbb E_{\epsilon}\left[\|(\epsilon-m)+(m-a)\|_2^2\mid H\right]\\
&=\mathbb E_{\epsilon}\left[\|\epsilon-m\|_2^2\mid H\right]
+\|m-a\|_2^2
+2\mathbb E_{\epsilon}[\epsilon-m\mid H]^\top(m-a)\\
&=\mathbb E_{\epsilon}\left[\|\epsilon-m\|_2^2\mid H\right]+\|m-a\|_2^2.
\end{aligned}
\tag{27}
$$

交叉项为零，因为 \(\mathbb E_{\epsilon}[\epsilon-m\mid H]=0\)。第一项不依赖 \(a\)，第二项在 \(a=m\) 时最小，因此：

$$
a^*=\mathbb E_{\epsilon}[\epsilon\mid X_t=x_t,t,D=1].
\tag{28}
$$

\(D\) 独立于整个数据和加噪过程，而 \(X_t\) 也是这个过程的函数，所以即使已知 \(x_t,t\)，\(D=1\) 也不会再提供关于噪声的信息：

$$
\boxed{
\epsilon_{\mathrm u}^*(x_t,t)
=\mathbb E_{\epsilon}[\epsilon\mid X_t=x_t,t]
}.
\tag{29}
$$

同理，保留标签时的理想最优预测为：

$$
\boxed{
\epsilon_{\mathrm c}^*(x_t,t,y)
=\mathbb E_{\epsilon}[\epsilon\mid X_t=x_t,t,Y=y]
}.
\tag{30}
$$

这也解释了为什么训练标签是一次抽到的具体噪声，而理想网络学到的是条件均值：不同的原图与噪声组合可能产生同样的带噪输入，MSE 在这些可能性上取平均。

### 6.4 从噪声条件均值连接到 Score

前向核是已知的高斯分布：

$$
q_t(x_t\mid x_0)
=\mathcal N\left(x_t;\sqrt{\bar\alpha_t}x_0,\sigma_t^2 I\right).
\tag{31}
$$

所以它对带噪输入的对数梯度为：

$$
\nabla_{x_t}\log q_t(x_t\mid x_0)
=-\frac{x_t-\sqrt{\bar\alpha_t}x_0}{\sigma_t^2}.
\tag{32}
$$

但我们需要的是混合所有原图之后的边缘 Score。先写边缘密度，再把梯度放进积分；在这里的高斯加噪及通常的可微、可积条件下：

$$
\begin{aligned}
\nabla_{x_t}\log q_t(x_t)
&=\frac{\nabla_{x_t}\int q_{\mathrm{data}}(x_0)q_t(x_t\mid x_0)\,dx_0}{q_t(x_t)}\\
&=\int\frac{q_{\mathrm{data}}(x_0)q_t(x_t\mid x_0)}{q_t(x_t)}
\nabla_{x_t}\log q_t(x_t\mid x_0)\,dx_0\\
&=\mathbb E_{X_0}\left[-\frac{x_t-\sqrt{\bar\alpha_t}X_0}{\sigma_t^2}
\,\middle|\,X_t=x_t,t\right]\\
&=-\frac{\mathbb E_{\epsilon}[\epsilon\mid X_t=x_t,t]}{\sigma_t}.
\end{aligned}
\tag{33}
$$

第三行用到了贝叶斯后验 \(q(x_0\mid x_t,t)\)，第四行用到了加噪关系 \(\epsilon=(X_t-\sqrt{\bar\alpha_t}X_0)/\sigma_t\)。把第 6.3 节的最优预测代入：

$$
s_{\mathrm u}^*(x_t,t)
=-\frac{\epsilon_{\mathrm u}^*(x_t,t)}{\sigma_t}
\approx-\frac{\epsilon_\theta(x_t,t,\varnothing)}{\sigma_t}.
\tag{34}
$$

固定类别 \(y\)，将上面边缘化中的 \(q_{\mathrm{data}}(x_0)\) 换成 \(q_{\mathrm{data}}(x_0\mid y)\)。类别不改变给定原图后的加噪核，因此可以逐步写成：

$$
\begin{aligned}
\nabla_{x_t}\log q_t(x_t\mid y)
&=\frac{\nabla_{x_t}\int q_{\mathrm{data}}(x_0\mid y)q_t(x_t\mid x_0)\,dx_0}{q_t(x_t\mid y)}\\
&=\int\frac{q_{\mathrm{data}}(x_0\mid y)q_t(x_t\mid x_0)}{q_t(x_t\mid y)}
\nabla_{x_t}\log q_t(x_t\mid x_0)\,dx_0\\
&=\mathbb E_{X_0}\left[-\frac{x_t-\sqrt{\bar\alpha_t}X_0}{\sigma_t^2}
\,\middle|\,X_t=x_t,t,Y=y\right]\\
&=-\frac{\mathbb E_{\epsilon}[\epsilon\mid X_t=x_t,t,Y=y]}{\sigma_t}.
\end{aligned}
\tag{35}
$$

这次对原图取期望使用的是后验 \(q(x_0\mid x_t,t,y)\)，随后等价地换成对噪声后验的期望。于是：

$$
s_{\mathrm c}^*(x_t,t,y)
=-\frac{\mathbb E_{\epsilon}[\epsilon\mid X_t=x_t,t,Y=y]}{\sigma_t}
=-\frac{\epsilon_{\mathrm c}^*(x_t,t,y)}{\sigma_t}
\approx-\frac{\epsilon_\theta(x_t,t,y)}{\sigma_t}.
\tag{36}
$$

至此，训练和引导所需的数学关系接起来了：

**独立丢弃标签 → 保留完整数据分布 → 无条件噪声 MSE → 噪声条件均值 → 无条件 Score。**

以上最优解针对理想预测函数。实际共享网络是否充分逼近它，仍取决于容量、数据与优化，不能把理论最优值直接当作训练后网络的精确输出。

## 7. 采样时怎样接入确定性 DDIM？

训练结束后固定参数 \(\theta\)。假设当前图片是 \(x_t\)，目标类别为 \(y=3\)，要更新到更早的时间 \(s\lt t\)。一次采样步骤包含三个动作。

### 7.1 对同一个输入计算两种预测

$$
\epsilon_{\mathrm c}=\epsilon_\theta(x_t,t,3),
\qquad
\epsilon_{\mathrm u}=\epsilon_\theta(x_t,t,\varnothing).
\tag{37}
$$

图片和时间相同，只改变条件输入。得到的是两份噪声预测，并没有分别生成两张图片。

### 7.2 组合成一份引导预测

$$
\hat\epsilon=(1+w)\epsilon_{\mathrm c}-w\epsilon_{\mathrm u}.
\tag{38}
$$

CFG 在这里决定本步使用哪份噪声预测。采样公式可以继续使用原有的 DDPM 或 DDIM；原论文算法 2 也明确说明可以换用 DDIM。

### 7.3 用同一份引导预测完成 DDIM 更新

先还原本步的原图估计：

$$
\hat x_0^{(t)}
=\frac{x_t-\sqrt{1-\bar\alpha_t}\,\hat\epsilon}{\sqrt{\bar\alpha_t}}.
\tag{39}
$$

再使用确定性 DDIM，也就是 \(\eta=0\) 的更新：

$$
x_s
=\sqrt{\bar\alpha_s}\,\hat x_0^{(t)}
+\sqrt{1-\bar\alpha_s}\,\hat\epsilon.
\tag{40}
$$

**两处都使用同一份 \(\hat\epsilon\)。** 到达新的 \(x_s\) 后，需要重新计算该时刻的条件与无条件预测，再组合、再更新。[DDIM 原论文](https://arxiv.org/abs/2010.02502)

完整流程可以写成：

1. 从标准高斯抽取 \(x_T\)，这依赖末端前向分布近似标准高斯的设定
2. 按递减的采样时间序列，重复“两种预测 → CFG 组合 → DDIM 更新”
3. 到达干净端点，得到最终图片；采用 \(\bar\alpha_0=1\) 的约定时，最后一步给出本步的 \(\hat x_0\)

整个过程中只有一条图片轨迹，网络参数不更新，也不需要分类器对输入的梯度。

所以两者的职责可以分清：**CFG 构造本步预测，DDIM 使用这份预测推进采样。** CFG 本身并不规定必须用哪一种采样器。

标准 CFG 每步需要两种去噪预测。实现时可以把条件与空条件输入拼成一个 batch，在一次网络调用里一起算；这并没有消除两种预测的计算量。不同实现的实际耗时还受显存、并行和硬件影响，不能简单把“去掉分类器”理解成采样一定更快。[Diffusers 的批处理实现](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py)

## 8. 两种 guidance scale 写法怎样对应？

原论文从条件预测出发：

$$
\hat\epsilon_{\mathrm{CFG}}
=(1+w)\epsilon_{\mathrm c}-w\epsilon_{\mathrm u}.
\tag{41}
$$

把它改写成从无条件预测出发：

$$
\begin{aligned}
\hat\epsilon_{\mathrm{CFG}}
&=\epsilon_{\mathrm u}+(1+w)(\epsilon_{\mathrm c}-\epsilon_{\mathrm u})\\
&=\epsilon_{\mathrm u}+\gamma(\epsilon_{\mathrm c}-\epsilon_{\mathrm u}),
\end{aligned}
\tag{42}
$$

于是：

$$
\boxed{\gamma=1+w}.
\tag{43}
$$

沿用上一篇的 \(\gamma\)，三个基准位置是：

| 预测含义 | \(\gamma\) 写法 | 原论文的 \(w\) 写法 |
|---|---:|---:|
| 无条件预测 | \(0\) | 代数上对应 \(-1\) |
| 普通条件预测 | \(1\) | \(0\) |
| 比普通条件更强的引导 | \(\gt 1\) | \(\gt 0\) |

如果 \(0\lt \gamma\lt 1\)，公式是在无条件与条件预测之间插值；\(\gamma\gt 1\) 才是沿同一方向外推。原论文主要讨论 \(w\geq0\) 的增强范围，所以 \(w=-1\) 只是换算中的代数位置。

因此，看到“scale 为 0”或“scale 为 1”，要先看它采用哪一个公式。某些库把 \(\gamma\) 叫作 \(s\) 或直接叫 guidance scale，字母本身没有统一含义。上表描述的是公式语义；实际 API 还可能在某些尺度下直接关闭双预测路径，不能仅凭公式推断接口行为。

另一个容易混淆的量是 \(p_{\mathrm{uncond}}\)：它决定训练时多大比例的样本隐藏条件，与采样时的 \(w\) 或 \(\gamma\) 不是同一件事。

## 9. 为什么引导尺度越大不一定越好？

### 9.1 即使 Score 完全准确，引导也会改变偏好

暂时假设条件与无条件 Score 都精确。由第 3 节的贝叶斯关系，在固定时间 \(t\) 上：

$$
\begin{aligned}
s_{\mathrm{guided}}^*
&=s_{\mathrm c}^*+w(s_{\mathrm c}^*-s_{\mathrm u}^*)\\
&=\nabla_{x_t}\log q_t(x_t\mid y)
+w\nabla_{x_t}\log q_t(y\mid x_t)\\
&=\nabla_{x_t}\log\left[q_t(x_t\mid y)q_t(y\mid x_t)^w\right].
\end{aligned}
\tag{44}
$$

因此，这个时刻的理想引导 Score 对应重新加权的密度：

$$
r_{t,w}(x_t\mid y)
=\frac{q_t(x_t\mid y)q_t(y\mid x_t)^w}{Z_{t,w}(y)},
\tag{45}
$$

$$
Z_{t,w}(y)
=\int q_t(u\mid y)q_t(y\mid u)^w\,du.
\tag{46}
$$

\(u\) 是积分用的变量。\(Z_{t,w}(y)\) 依赖时间、类别和尺度，但不依赖当前求导变量 \(x_t\)，因此 \(\nabla_{x_t}\log Z_{t,w}(y)=0\)。对于这里离散标签且 \(q(y)\gt 0\)、\(w\geq0\) 的情形，这个权重可以正常归一化。

再用一次贝叶斯关系，可以得到常见的另一种写法：

$$
\begin{aligned}
r_{t,w}(x_t\mid y)
&\propto q_t(x_t\mid y)
\left[\frac{q_t(x_t\mid y)q(y)}{q_t(x_t)}\right]^w\\
&\propto\frac{q_t(x_t\mid y)^{1+w}}{q_t(x_t)^w}\\
&=q_t(x_t\mid y)^\gamma q_t(x_t)^{1-\gamma}.
\end{aligned}
\tag{47}
$$

这个表达式说明，在原有条件密度上，又乘了一个偏向“目标类别容易辨认”的权重。它不是在更精确地恢复原始条件分布，而是在改变偏好。

### 9.2 用密度比看多样性的代价

假设固定噪声等级下，位置 \(a,b\) 的条件密度相同，但：

$$
q_t(y=3\mid a)=0.9,\qquad q_t(y=3\mid b)=0.6.
\tag{48}
$$

由于归一化因子在比值中抵消，引导后的密度比为：

$$
\frac{r_{t,w}(a\mid 3)}{r_{t,w}(b\mid 3)}
=\frac{q_t(a\mid 3)}{q_t(b\mid 3)}
\left[\frac{q_t(3\mid a)}{q_t(3\mid b)}\right]^w
=1.5^w.
\tag{49}
$$

| \(w\) | \(a\) 与 \(b\) 的密度比 |
|---:|---:|
| \(0\) | \(1\) |
| \(1\) | \(1.5\) |
| \(4\) | 约 \(5.1\) |
| \(8\) | 约 \(25.6\) |

即使 \(b\) 也是合理的数字 3，只是更容易与其他类别混淆，强引导也会越来越偏向 \(a\)。这个假想例子展示了“更容易辨认”与“覆盖各种合理写法”之间的冲突，并不是一次实际生成实验。

### 9.3 网络误差的差异也可能被放大

记两种网络预测相对于理想预测的误差为：

$$
\epsilon_{\mathrm c}=\epsilon_{\mathrm c}^*+\delta_{\mathrm c},
\qquad
\epsilon_{\mathrm u}=\epsilon_{\mathrm u}^*+\delta_{\mathrm u}.
\tag{50}
$$

在同一个 \(x_t,t\) 上，实际 CFG 预测与理想组合之差为：

$$
\begin{aligned}
\delta_{\mathrm{guided}}
&=\left[(1+w)\epsilon_{\mathrm c}-w\epsilon_{\mathrm u}\right]
-\left[(1+w)\epsilon_{\mathrm c}^*-w\epsilon_{\mathrm u}^*\right]\\
&=(1+w)\delta_{\mathrm c}-w\delta_{\mathrm u}\\
&=\delta_{\mathrm c}+w(\delta_{\mathrm c}-\delta_{\mathrm u}).
\end{aligned}
\tag{51}
$$

所以尺度越大，越可能放大两种预测误差之间的差异。这并不说明总误差必然随 \(w\) 单调增加，因为误差也可能抵消。例如 \(\delta_{\mathrm c}=\delta_{\mathrm u}\) 时，差异项为零。

这里是对固定输入的误差分析。实际采样中，不同尺度还会让后续轨迹访问不同的 \(x_t\)，不能把这个局部公式当作完整的最终图片误差定理。

### 9.4 固定时间的密度解释，不等于最终采样分布保证

前面的 \(r_{t,w}\) 证明了一个固定时间上的结论：**精确 Score 的线性组合，是这个时刻重新加权密度的 Score。**

但把每个时刻的这种 Score 放进 DDPM 或 DDIM，并不能直接推出最终图片精确服从：

$$
r_{0,w}(x_0\mid y)\propto q_{\mathrm{data}}(x_0\mid y)q(y\mid x_0)^w.
\tag{52}
$$

原因在于，对每个噪声等级分别重新加权得到的整族 \(r_{t,w}\)，一般不等于“先对干净分布重新加权，再用原前向过程加噪”产生的边缘分布族。逐时刻的 Score 恒等式不自动建立一条一致的前向、反向扩散过程。

因此，这个限制并非只来自网络误差或有限采样步数。后续理论工作专门说明，CFG 与 DDPM、DDIM 的组合一般不会精确采出常被直觉上赋予它的幂次重加权分布。[Classifier-Free Guidance is a Predictor-Corrector](https://arxiv.org/abs/2408.09000)

保留这条边界后，重加权依然是理解引导方向和偏好的有用工具，只是不应把它写成最终输出分布的精确保证。

## 10. 原论文的实验支持了什么？

原论文在类别条件 ImageNet 上研究了三个变量：采样引导强度 \(w\)、训练时丢弃条件的概率 \(p_{\mathrm{uncond}}\)、采样步数。它们分别控制不同环节，不能混成同一个“强度”。

实验中，较小的额外引导取得较好的 FID，更强的引导还能提高 IS；图像也呈现出多样性下降、强引导时颜色过饱和的现象。两个指标考察的侧面不同，不能因为 IS 上升就断言整体质量无限改善。[原论文 §4.1、图 3](https://arxiv.org/html/2207.12598v1#S4)

条件丢弃概率影响联合训练的分配，采样步数影响质量与计算成本。原论文也特别提醒：CFG 每步要计算两种去噪预测，比较速度时必须计入这部分成本；分类器可能比去噪网络小，因此 CFG 的采样速度未必优于 CG。[原论文 §4.2–5](https://arxiv.org/html/2207.12598v1#S4.SS2)

从这些结果应带走的结论是：**CFG 提供了一个实用的保真度与多样性调节方式，但不存在“尺度越大就一定越好”的保证。**

如果自己做对照实验，应固定模型、目标类别、采样器、采样时间序列以及同一份初始噪声，只改变 \(\gamma\)，例如对照无条件、普通条件和更强引导。本文的数值例子用于解释公式，没有把未运行的实验当成结果。

## 11. 自检与答案

### 1. 普通条件模型已经输入了类别，CFG 还在增加什么？

增加的是采样时对类别特征的额外偏好。普通条件生成已经能按标签生成；引导进一步偏向更容易辨认的样本，并可能牺牲部分多样性。

### 2. 为什么两个 Score 可以代替分类器方向？

固定 \(t,y\) 后，贝叶斯公式给出 \(\log q_t(x_t\mid y)=\log q_t(x_t)+\log q_t(y\mid x_t)-\log q(y)\)。对 \(x_t\) 求梯度，先验项消失，移项得到 \(\nabla_{x_t}\log q_t(y\mid x_t)=s_{\mathrm c}^*-s_{\mathrm u}^*\)。真实 Score 满足精确恒等式，网络只能估计它。

### 3. 丢弃标签后，训练答案是什么？“无条件”是否意味着没有训练目标？

答案仍是本次加噪抽到的 \(\epsilon\)。被隐藏的只是类别输入，带噪图片与噪声监督仍然保留。这里“无条件”描述输入不含类别，不表示训练没有目标。

### 4. 为什么丢弃开关的独立性很重要？

独立丢弃使 \(q(x_0,y\mid D=1)=q_{\mathrm{data}}(x_0,y)\)，空条件分支仍看到完整数据分布。若类别不同，丢弃概率也不同，空条件训练数据可能发生类别重加权。它便不再自动对应原来的无条件分布。

### 5. MSE 为什么得到条件均值，而不是为每个输入恢复唯一一份噪声？

相同的带噪输入可能来自不同原图与噪声组合。平方误差分解为“条件方差项 + 预测与条件均值的距离平方”，第一项不受预测控制，第二项在预测等于条件均值时最小。

### 6. 每步做两种预测，是同时维护两张图片吗？

只有一条图片轨迹。两种预测使用相同的 \(x_t,t\)，只改变条件输入，再合成一份 \(\hat\epsilon\) 更新这张图片。到了下一步才重新预测。

### 7. \(w=0\) 和 \(\gamma=0\) 是同一种情况吗？

不是。\(w=0\) 是普通条件预测，\(\gamma=0\) 是无条件预测；二者通过 \(\gamma=1+w\) 换算。若 \(\gamma=3\)，则 \(w=2\)，组合为 \(3\epsilon_{\mathrm c}-2\epsilon_{\mathrm u}\)。

### 8. 确定性 DDIM 的两个公式应该各用哪份噪声？

估计 \(\hat x_0^{(t)}\) 和构造 \(x_s\) 时，都使用本步同一份 CFG 组合后的 \(\hat\epsilon\)，不在第二处换回普通条件预测。

### 9. 为什么大尺度不一定更好？

即使 Score 精确，强引导也会更偏向容易辨认的样本，减少对其他合理样本的覆盖；实际网络还存在误差，外推可能放大 \(\delta_{\mathrm c}-\delta_{\mathrm u}\)。这两层原因需要分别理解。

### 10. 已经得到 \(r_{t,w}\) 的 Score，能断言最终图片精确来自 \(r_{0,w}\) 吗？

不能。固定时间上的密度恒等式，不保证这些重新加权的密度组成与原前向加噪过程一致的边缘分布族。网络近似和有限步数之外，采样过程本身也有需要区分的理论边界。

## 12. 把方法串起来

训练时，独立随机隐藏一部分样本的条件，使同一个网络既学习 \(\mathbb E_{\epsilon}[\epsilon\mid x_t,t,y]\)，也学习 \(\mathbb E_{\epsilon}[\epsilon\mid x_t,t]\)。这两种噪声预测分别对应条件与无条件 Score 的估计。

采样时，对当前输入计算两种预测，用它们的差加强条件方向：

$$
\boxed{
\hat\epsilon_{\mathrm{CFG}}
=\epsilon_{\mathrm u}+\gamma(\epsilon_{\mathrm c}-\epsilon_{\mathrm u})
=(1+w)\epsilon_{\mathrm c}-w\epsilon_{\mathrm u},
\qquad \gamma=1+w
}.
\tag{53}
$$

再将这份预测交给 DDIM 等采样器。CFG 不需要单独的分类器，但仍依赖训练得到的条件与无条件预测，并付出每步两种预测的成本。尺度控制生成偏好，调高它要同时考虑条件特征、覆盖范围和图像质量。

## 参考资料

1. Jonathan Ho, Tim Salimans. [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598). 重点见 §3、算法 1–2、§4–5
2. Prafulla Dhariwal, Alex Nichol. [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233). Classifier Guidance 的原始工作
3. Jiaming Song, Chenlin Meng, Stefano Ermon. [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502). 本文确定性采样更新的依据
4. Arwen Bradley, Preetum Nakkiran. [Classifier-Free Guidance is a Predictor-Corrector](https://arxiv.org/abs/2408.09000). 用于校准固定时间重加权与最终采样分布的区别
5. Hugging Face Diffusers. [Stable Diffusion pipeline](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py). 条件与空条件预测的 batch 实现示例
