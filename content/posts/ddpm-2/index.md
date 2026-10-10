{
  "title": "DDPM（二）：样本、噪声预测与损失",
  "description": "沿一次训练走完样本构造、噪声预测、损失与参数更新。",
  "summary": "沿一次训练走完样本构造、噪声预测、损失与参数更新。",
  "date": "2026-10-09T14:52:00+08:00",
  "slug": "ddpm-2",
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
  "episode": 2
}

[第一篇]({{< relref "posts/ddpm-1" >}})建立了整体图景：用真实图片学习，再从新噪声出发生成图片。现在只跟着一次训练走，看看题目从哪里来、网络看见什么，以及它到底要答什么。

## 先做一道自己知道答案的题

拿一张手写数字“7”。图片在计算机里是一组像素数值，我们再产生一组同样大小的随机数，把两者按选定的强度混合。

这样同时得到三份东西：原图、带噪图，以及刚才加入的噪声。我们把带噪图交给网络，把自己抽取的噪声留作答案。没有人需要逐像素标注去噪标签。

网络的输出也是与图片同样大小的一组数，表示每个位置的噪声估计。它还会知道当前噪声程度，因为噪声很少和噪声很多时，需要作出的判断不同。

**原图和噪声答案由训练程序保管，不作为网络输入。**

## 怎样制造指定噪声程度的输入

把原图记为 \(x_0\)，加噪第 \(t\) 步后的状态记为 \(x_t\)，预设总步数叫 \(T\)。这里的“时间”只是步骤编号。

每一步都先缩小旧状态，再加入一点新噪声：

$$x_t=\sqrt{1-\beta_t}\,x_{t-1}+\sqrt{\beta_t}\,z_t.$$

\(\beta_t\) 是事先选好的噪声方差，满足 \(0<\beta_t<1\)。\(z_t\) 是新抽的标准高斯噪声，靠近 0 的数常见，正负对称，离 0 很远的数少见。每个位置均值为 0、方差为 1；不同位置、不同步骤独立抽取。均值表示平均中心，方差表示偏离中心的距离平方的平均，独立表示一次抽数不会改变另一次的抽数规则。

噪声乘以系数后，方差会乘以该系数的平方，所以这里乘 \(\sqrt{\beta_t}\)，实际加入的方差才是 \(\beta_t\)。

不过，训练时想要第 500 步的题目，不必真的计算 500 次。定义

$$\alpha_t=1-\beta_t,\qquad
\bar\alpha_t=\prod_{s=1}^{t}\alpha_s,\qquad \bar\alpha_0=1,$$

就能直接抽取这个时刻的状态：

$$x_t=\sqrt{\bar\alpha_t}\,x_0+
\sqrt{1-\bar\alpha_t}\,\epsilon,
\qquad \epsilon\sim\mathcal N(0,I).$$

\(\prod\) 表示把各步的系数连乘；\(\sim\mathcal N(0,I)\) 表示抽取一份标准高斯噪声，其中 \(I\) 表示各位置方差为 1、彼此独立。\(\bar\alpha_t\) 越小，原图信号越弱。

这份 \(\epsilon\) 表示到当前时刻的累计噪声经过标准化后的量，不是最后一步单独加入的 \(z_t\)。两种构造得到同样的 \(x_t\) 分布，但不要求复现同一条随机路径。计算依据放在篇末，先继续训练流程。

## 让网络作答，再比较误差

网络输入是 \(x_t,t\)，输出记作 \(\epsilon_\theta(x_t,t)\)。其中 \(\theta\) 只是网络内部所有可训练数值的统称。

用预测值和出题时留下的噪声比较：

$$\ell=\|\epsilon-\epsilon_\theta(x_t,t)\|^2.$$

这个平方误差把每个位置的误差平方后相加。例如，噪声答案是 0.3，预测为 0.1，那么该位置贡献 \((0.3-0.1)^2=0.04\)。代码也常对像素数和批次大小求平均。

随后通过反向传播计算梯度，判断各个参数该朝什么方向调整，再更新网络参数，让预测逐渐改善。“反向传播”是训练时计算梯度的技术，与“反向生成图片”是两个不同的过程。

## 一次参数更新，就是这四步

1. 抽一张原图 \(x_0\)，均匀抽一个时间步 \(t\)，再抽一份噪声 \(\epsilon\)
2. 用直接加噪公式构造 \(x_t\)
3. 输入 \(x_t,t\)，得到噪声预测并计算平方误差
4. 反向传播，更新 \(\theta\)；实际通常一次处理一批样本

不断更换原图、时间步和噪声，同一个网络就能练习不同难度。整个简单训练目标写成

$$
L_{\mathrm{simple}}(\theta)=
\mathbb E_{x_0\sim q_{\mathrm{data}}}
\mathbb E_{t\sim\mathrm{Uniform}\{1,\ldots,T\}}
\mathbb E_{\epsilon\sim\mathcal N(0,I)}
\left[\|\epsilon-\epsilon_\theta(x_t,t)\|^2\right],
$$

其中 \(x_t\) 按上面的直接加噪公式构造。\(\mathbb E\) 表示取平均，三个下标依次说明：平均不同原图、不同时间步、不同噪声。\(q_{\mathrm{data}}\) 是真实图片的来源，\(\mathrm{Uniform}\) 表示每个时间步被抽到的机会相同。

## 有答案，也不表示每道题都能猜中

一张带噪图可能由不止一组“原图加噪声”产生，噪声重时尤其如此。训练程序知道这次的答案，网络却未必能从输入中唯一确定它。

平方误差理想上让网络学习的是给定输入后的平均噪声估计。它需要从大量图片中学到规律，才能作出有用的预测；并不是练习足够多，就必然能恢复某次唯一的历史噪声。

到这里，训练流程已经完整了：**输入带噪图与时间步，输出噪声估计，用自己制造的答案评分。** [第三篇]({{< relref "posts/ddpm-3" >}})解释这个任务为什么与图片生成有关；随后[第四篇]({{< relref "posts/ddpm-4" >}})说明实际生成流程。

参考：[DDPM 原论文，式（2）（4）（14）与算法 1](https://arxiv.org/pdf/2006.11239)。

## 两条方便训练的计算依据 {#选读两条方便训练的计算依据}

## 正向加噪的完整推导与说明

先选择一组固定的噪声方差 \(0<\beta_t<1\)。正向一步定义为

$$q(x_t\mid x_{t-1})=\mathcal N\!\left(x_t;\sqrt{\alpha_t}x_{t-1},\beta_t I\right),\qquad \alpha_t=1-\beta_t. \qquad (1)$$

这里 \(\mathcal N(x;\mu,\Sigma)\) 表示在 \(x\) 处的高斯密度。条件中固定了 \(x_{t-1}\)，剩下的随机量是 \(x_t\)。等价的抽样写法是

$$x_t=\sqrt{\alpha_t}x_{t-1}+\sqrt{\beta_t}\epsilon_t,\qquad \epsilon_t\sim\mathcal N(0,I).$$

\(\beta_t I\) 是协方差，所以乘在标准噪声前的是 \(\sqrt{\beta_t}\)。各步的 \(\epsilon_t\) 独立。平方根缩放有意让原有信号逐渐衰减，同时注入噪声。

**先展开两步。** 这能直接看到为什么出现累计乘积：

$$\begin{aligned}
x_2&=\sqrt{\alpha_2}\left(\sqrt{\alpha_1}x_0+\sqrt{\beta_1}\epsilon_1\right)+\sqrt{\beta_2}\epsilon_2\\
&=\sqrt{\alpha_2\alpha_1}x_0+\sqrt{\alpha_2\beta_1}\epsilon_1+\sqrt{\beta_2}\epsilon_2.
\end{aligned}$$

后两项是独立高斯的线性组合，均值为零，总方差为

$$\alpha_2\beta_1+\beta_2=\alpha_2(1-\alpha_1)+(1-\alpha_2)=1-\alpha_1\alpha_2.$$

一般地，设第 \(t-1\) 步条件方差为 \((1-\overline{\alpha}_{t-1})I\)，则第 \(t\) 步方差为

$$\alpha_t(1-\overline{\alpha}_{t-1})+\beta_t=1-\alpha_t\overline{\alpha}_{t-1}=1-\overline{\alpha}_t.$$

条件均值也递推为 \(\sqrt{\overline{\alpha}_t}x_0\)，因此

$$q(x_t\mid x_0)=\mathcal N\!\left(x_t;\sqrt{\overline{\alpha}_t}x_0,(1-\overline{\alpha}_t)I\right). \qquad (2)$$

于是可以直接抽样：

$$x_t=\sqrt{\overline{\alpha}_t}x_0+\sqrt{1-\overline{\alpha}_t}\epsilon,\qquad \epsilon\sim\mathcal N(0,I). \qquad (3)$$

这里的 \(\epsilon\) 是前面多步噪声的合成标准噪声，通常不是最后一步的 \(\epsilon_t\)。式（3）与逐步加噪得到的 \(x_t\) **具有相同的条件分布**，并不表示两种抽样会复现同一条随机路径。它的价值是：训练第 \(t\) 步时，不必真的先加噪 \(t\) 次。

当 \(\overline{\alpha}_T\) 足够小时，原图信号被大幅压低，\(q(x_T\mid x_0)\) 接近标准高斯；有限步下通常只是接近，并非严格相等。[原论文式（2）（4）](https://arxiv.org/pdf/2006.11239)





## 训练算法的完整表述与随机时间步解释

下文式（3）指本篇上面给出的直接加噪公式，编号与第三篇完整推导一致。

### 训练一次参数更新

1. 从数据集抽取原图 \(x_0\)，均匀随机抽取 \(t\in\{1,\ldots,T\}\)，再抽取 \(\epsilon\sim\mathcal N(0,I)\)
2. 直接构造 \(x_t=\sqrt{\overline{\alpha}_t}x_0+\sqrt{1-\overline{\alpha}_t}\epsilon\)
3. 调用网络 \(\epsilon_\theta(x_t,t)\)，计算 \(\|\epsilon-\epsilon_\theta(x_t,t)\|^2\)
4. 对网络参数 \(\theta\) 反向传播并更新；实际通常对一批样本求平均

为什么随机一步就够？若 \(\ell_t(\theta)\) 是固定 timestep、已平均原图和噪声后的损失，则

$$\mathbb E_{t\sim\mathrm{Uniform}\{1,\ldots,T\}}[\ell_t(\theta)]
=\frac1T\sum_{t=1}^{T}\ell_t(\theta).$$

因此，随机 timestep 是对**已经选定的平均目标**进行无偏抽样估计。它和第三篇 F 节中的“去掉权重”是两个不同操作。训练只需直接构造选中时刻的 \(x_t\)，不必生成完整路径，也不必真的抽出 \(x_{t-1}\)。[原论文算法 1](https://arxiv.org/pdf/2006.11239)
