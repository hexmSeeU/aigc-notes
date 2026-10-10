{
  "title": "DDIM：从边缘分布到跳步采样",
  "description": "保留 DDPM 的单时刻加噪分布，从相邻条件构造与边缘证明推导 DDIM 更新，说明确定性采样、eta 与 DDPM 的对应关系及跳步实现。",
  "date": "2026-10-09T18:17:00+08:00",
  "slug": "ddim-derivation",
  "categories": [
    "基础"
  ],
  "tags": [
    "DDIM",
    "扩散模型"
  ],
  "draft": false,
  "math": true,
  "summary": "沿着条件构造、边缘验证、原图估计与重新组合的主线，串起确定性 DDIM、eta 等于 1 的 DDPM 对应关系和跳步采样，附简短伪代码与实现检查。"
}

## 1 为什么可以沿用 DDPM 的训练

DDPM 训练网络预测某个噪声等级下的噪声。DDIM 希望保留这项能力，却用更少的网络调用生成图片。关键是：**保持每个时刻的加噪分布，同时改变不同时刻之间的连接方式。**

### 先统一符号

沿用前篇 DDPM 笔记：\(x_0\) 是真实原图，\(x_t\) 是当前带噪状态，\(x_{t-1}\) 是相邻的较早状态，都是 \(d\) 维向量。记

$$\alpha_t=1-\beta_t,\qquad
\overline{\alpha}_t=\prod_{r=1}^{t}\alpha_r,\qquad
\overline{\alpha}_0=1. \qquad (1)$$

取 \(0<\beta_t<1\)，因此累计系数从 \(1\) 逐步减小，\(\overline{\alpha}_T\) 接近零但仍为正。**DDIM 论文的 \(\alpha_t\) 对应本文的 \(\overline{\alpha}_t\)**，不是本文的单步 \(\alpha_t\)。

固定原图后，我们保留的分布和抽样方式是

$$\begin{aligned}
q_\sigma(x_t\mid x_0)
&=\mathcal N\!\left(x_t;\sqrt{\overline{\alpha}_t}x_0,
(1-\overline{\alpha}_t)I\right),\\
x_t&=\sqrt{\overline{\alpha}_t}x_0+
\sqrt{1-\overline{\alpha}_t}\epsilon,\qquad\epsilon\sim\mathcal N(0,I).
\end{aligned} \qquad (2)$$

下标 \(\sigma\) 暂时标记待构造的过程。\(I\) 是单位矩阵；\(\epsilon\) 每个分量的方差为 \(1\)，所以噪声前乘的是目标方差的平方根。

### 为什么训练损失不变

简化噪声预测目标为

$$\begin{aligned}
L_{\mathrm{simple}}(\theta)
={}&\mathbb E_{t\sim\mathrm{Unif}\{1,\ldots,T\}}
\mathbb E_{x_0\sim q_{\mathrm{data}}}
\mathbb E_{\epsilon\sim\mathcal N(0,I)}\\
&\left[\left\|\epsilon-\epsilon_\theta\!\left(
\sqrt{\overline{\alpha}_t}x_0+\sqrt{1-\overline{\alpha}_t}\epsilon,t
\right)\right\|^2\right].
\end{aligned} \qquad (3)$$

\(t,x_0,\epsilon\) 独立抽取，\(q_{\mathrm{data}}\) 是数据分布。网络只看 \(x_t,t\)，不看整条路径。给定 \(x_0,x_t,t\)，监督噪声又能唯一反解为

$$\epsilon=\frac{x_t-\sqrt{\overline{\alpha}_t}x_0}
{\sqrt{1-\overline{\alpha}_t}}. \qquad (4)$$

因此，保持数据抽样、时间抽样和式（2），就保持了训练输入与监督答案的联合分布，式（3）也精确不变。但各个时刻单独分布相同，不代表它们一起出现的方式相同。这为重设采样路径留下了自由度。

这里仅说明**简化噪声损失不变**，不把它扩大为不同变分目标无条件完全相同。[原论文第 3 节](https://arxiv.org/html/2010.02502v4#S3)

## 2 怎样构造新的相邻条件分布

先假设真实原图 \(x_0\) 已知，考虑 \(t\geq2\)。我们有 \(x_t\)，希望构造 \(x_{t-1}\)，同时保留

$$q_\sigma(x_{t-1}\mid x_0)
=\mathcal N\!\left(x_{t-1};\sqrt{\overline{\alpha}_{t-1}}x_0,
(1-\overline{\alpha}_{t-1})I\right). \qquad (5)$$

目标是：固定 \(x_0\) 后，原图部分为 \(\sqrt{\overline{\alpha}_{t-1}}x_0\)，噪声的总方差为 \(1-\overline{\alpha}_{t-1}\)。

### 将目标噪声分成两份

第一份沿用式（4）从当前状态提取的噪声，使两个时刻建立联系；第二份是独立新抽的 \(z\sim\mathcal N(0,I)\)。

设新噪声标准差为 \(\sigma_t\)，它占用的方差就是 \(\sigma_t^2\)。剩余方差 \(1-\overline{\alpha}_{t-1}-\sigma_t^2\) 分配给沿用噪声，因此选择

$$\begin{aligned}
x_{t-1}={}&\sqrt{\overline{\alpha}_{t-1}}x_0\\
&+\sqrt{1-\overline{\alpha}_{t-1}-\sigma_t^2}
\frac{x_t-\sqrt{\overline{\alpha}_t}x_0}{\sqrt{1-\overline{\alpha}_t}}
+\sigma_t z.
\end{aligned} \qquad (6)$$

\(z\) 在给定 \(x_0\) 时与 \(x_t\) 独立。为了能开平方，必须满足

$$0\leq\sigma_t^2\leq1-\overline{\alpha}_{t-1}. \qquad (7)$$

**为什么不是直接把两个噪声系数相加？** 方差随系数的平方缩放，且独立噪声的方差相加。因此分配的是两份方差，乘在标准噪声前面的则是各自的标准差。

### 同时固定原图和当前状态

现在把 \(x_0,x_t\) 都固定住。式（6）的前两项确定，只有新抽的 \(z\) 随机。所以在 \(\sigma_t>0\) 时，

$$\begin{aligned}
q_\sigma(x_{t-1}\mid x_t,x_0)
=\mathcal N\!\Bigg(x_{t-1};&\ \sqrt{\overline{\alpha}_{t-1}}x_0\\
&+\sqrt{1-\overline{\alpha}_{t-1}-\sigma_t^2}
\frac{x_t-\sqrt{\overline{\alpha}_t}x_0}{\sqrt{1-\overline{\alpha}_t}},\ \sigma_t^2I\Bigg).
\end{aligned} \qquad (8)$$

这里的条件方差只有 \(\sigma_t^2\)。下一节只固定 \(x_0\)、让 \(x_t\) 也变化时，方差就会包括沿用噪声的贡献。

式（8）是作者选择的一种连接方式，不是从“边缘相同”唯一推出来的答案。接下来要验证它确实保留了式（5）。[原论文式（7）](https://arxiv.org/html/2010.02502v4#S3.SS1)

## 3 为什么每个时刻的分布都能保留

现在**只固定 \(x_0\)**，让 \(x_t\) 按式（2）变化，独立抽取 \(z\sim\mathcal N(0,I)\)。从 \(x_t\) 反解出的 \(\epsilon\) 是标准高斯，因此式（6）等价于

$$x_{t-1}=\sqrt{\overline{\alpha}_{t-1}}x_0
+\sqrt{1-\overline{\alpha}_{t-1}-\sigma_t^2}\epsilon
+\sigma_t z. \qquad (9)$$

两份噪声独立，均值都是零。固定原图后的均值为

$$\mathbb E_{\epsilon\sim\mathcal N(0,I)}
\mathbb E_{z\sim\mathcal N(0,I)}[x_{t-1}]
=\sqrt{\overline{\alpha}_{t-1}}x_0. \qquad (10)$$

原图项不贡献随机波动，噪声交叉协方差为零，所以

$$\begin{aligned}
\operatorname{Cov}(x_{t-1}\mid x_0)
&=(1-\overline{\alpha}_{t-1}-\sigma_t^2)I+\sigma_t^2I\\
&=(1-\overline{\alpha}_{t-1})I.
\end{aligned} \qquad (11)$$

独立高斯的线性组合仍然是高斯，且均值和协方差都与式（5）一致。因此，目标单时刻分布确实得到保留。

**两种条件方差并不矛盾。** 只固定 \(x_0\)，旧噪声和新噪声都在变化，总方差为 \(1-\overline{\alpha}_{t-1}\)；同时固定 \(x_0,x_t\)，旧噪声已确定，剩余方差才是 \(\sigma_t^2\)。

### 从末端开始定义整条路径

先规定 \(x_T\mid x_0\) 服从式（2）的末端分布，再按式（8）倒序构造较早状态：

$$q_\sigma(x_{1:T}\mid x_0)
=q_\sigma(x_T\mid x_0)
\prod_{t=2}^{T}q_\sigma(x_{t-1}\mid x_t,x_0). \qquad (12)$$

\(x_{1:T}\) 表示 \(x_1,\ldots,x_T\)。末端边缘已规定正确，而式（9）至（11）证明“当前边缘正确，就能得到前一时刻的正确边缘”。这样一路递推，所有时刻都满足式（2）。

这仍是**给定真实原图的参考构造**，还不是实际生成算法。真正生成时不知道 \(x_0\)，需要下一节的网络估计。

### 零新噪声的边界

当 \(\sigma_t=0\)，式（6）仍然有效：给定 \(x_0,x_t\) 后，下一步完全确定。式（8）此时应理解为点质量，式（12）理解为条件概率核的组合，而不是普通正密度的乘积。不能把零方差直接代入含 \(1/\sigma_t^2\) 或 \(\log\sigma_t^2\) 的高斯 KL 公式。

但只固定 \(x_0\) 时，\(x_t\) 仍随机，\(x_{t-1}\) 也仍有式（11）的边缘方差。[原论文式（6）](https://arxiv.org/html/2010.02502v4#S3.SS1)；[附录 B](https://arxiv.org/html/2010.02502v4#A2)

## 4 怎样得到实际生成公式

实际生成没有真实 \(x_0\) 或 \(\epsilon\)。先把加噪式（2）移项：

$$x_0=\frac{x_t-\sqrt{1-\overline{\alpha}_t}\epsilon}
{\sqrt{\overline{\alpha}_t}}.$$

再用训练好的网络输出 \(\epsilon_\theta(x_t,t)\) 代替真实噪声，定义当前原图估计

$$\hat x_0(x_t,t)
=\frac{x_t-\sqrt{1-\overline{\alpha}_t}\epsilon_\theta(x_t,t)}
{\sqrt{\overline{\alpha}_t}}. \qquad (13)$$

帽子表示估计。后文简写 \(\hat x_0\)，但它始终由**当前** \(x_t,t\) 计算，不保证等于某张真实原图。

### 把估计代回条件构造

在式（6）中，用 \(\hat x_0\) 替换真实 \(x_0\)。沿用噪声部分的残差精确化简为

$$\begin{aligned}
\frac{x_t-\sqrt{\overline{\alpha}_t}\hat x_0}{\sqrt{1-\overline{\alpha}_t}}
&=\frac{x_t-\left(x_t-\sqrt{1-\overline{\alpha}_t}\epsilon_\theta(x_t,t)\right)}
{\sqrt{1-\overline{\alpha}_t}}\\
&=\epsilon_\theta(x_t,t).
\end{aligned} \qquad (14)$$

因此，中间时刻的生成更新为

$$\begin{aligned}
x_{t-1}={}&\sqrt{\overline{\alpha}_{t-1}}\hat x_0\\
&+\sqrt{1-\overline{\alpha}_{t-1}-\sigma_t^2}\epsilon_\theta(x_t,t)
+\sigma_t z,\qquad z\sim\mathcal N(0,I).
\end{aligned} \qquad (15)$$

新抽的 \(z\) 与当前状态及已有随机性独立。三个部分分别是：

1. \(\sqrt{\overline{\alpha}_{t-1}}\hat x_0\)：按目标系数缩放的预测原图
2. \(\sqrt{1-\overline{\alpha}_{t-1}-\sigma_t^2}\epsilon_\theta(x_t,t)\)：沿用当前预测的噪声方向
3. \(\sigma_t z\)：新增随机噪声

**替换是模型近似，化简是精确等式。** 用估计原图替换真实原图，定义了我们要执行的生成模型；式（14）则是这个定义下的代数恒等式。上一节的参考边缘证明，不会自动保证网络生成的分布完全正确，也不保证预测噪声本身严格为标准高斯。

生成从 \(x_T\sim\mathcal N(0,I)\) 开始：因为 \(\overline{\alpha}_T\) 很小，式（2）的末端接近标准高斯。这里是近似，不能为图方便把正的 \(\overline{\alpha}_T\) 设成零，因为式（13）还要除以它的平方根。

[原论文式（9）（10）](https://arxiv.org/html/2010.02502v4#S3.SS2)；[式（12）](https://arxiv.org/html/2010.02502v4#S4.SS1)

## 5 确定性 DDIM 的一次更新

令 \(\sigma_t=0\)，新噪声项消失，式（15）变成

$$x_{t-1}=\sqrt{\overline{\alpha}_{t-1}}\hat x_0
+\sqrt{1-\overline{\alpha}_{t-1}}\epsilon_\theta(x_t,t). \qquad (16)$$

而根据 \(\hat x_0\) 的定义，当前状态必然满足

$$x_t=\sqrt{\overline{\alpha}_t}\hat x_0
+\sqrt{1-\overline{\alpha}_t}\epsilon_\theta(x_t,t). \qquad (17)$$

两行使用同一份预测原图和预测噪声，变化的只有系数。由于 \(\overline{\alpha}_{t-1}>\overline{\alpha}_t\)，目标状态的原图系数更大，噪声系数更小。

因此，一次确定性更新就是：**先把当前状态分解成预测原图和预测噪声，再按较低噪声等级重新组合。**

### 同一份预测只在这一跳中复用

到达 \(x_{t-1}\) 后，要重新调用网络 \(\epsilon_\theta(x_{t-1},t-1)\)，再计算新的原图估计。不能把第一次得到的噪声预测一直用到最后。

这也是多步采样的作用：更新状态后，网络有机会重新判断。减少步数虽然省下调用，也减少了中间重新估计的机会。

### 确定性不等于没有随机输入

固定初始噪声 \(x_T\)、模型和时间日程，并使用确定的推理计算后，后续轨迹就由式（16）决定。最开始仍然随机抽取 \(x_T\)，所以换一份初始噪声，仍能得到另一张图片。

“确定性”也没有保证改变采样时间日程后结果不变。不同路径会在不同中间状态重新调用网络，最终结果通常可能改变。[原论文第 4.1 节](https://arxiv.org/html/2010.02502v4#S4.SS1)

### 用 eta 控制新噪声

为了连接确定性 DDIM 与对应的 DDPM，取

$$\begin{aligned}
\tilde\beta_t&=\frac{1-\overline{\alpha}_{t-1}}
{1-\overline{\alpha}_t}\beta_t,\\
\sigma_t&=\eta\sqrt{\tilde\beta_t},\qquad0\leq\eta\leq1.
\end{aligned} \qquad (18)$$

这里 \(\tilde\beta_t\) 是 DDPM 给定 \(x_t,x_0\) 的后验方差。\(\eta=0\) 就是本节的确定性更新；\(0<\eta<1\) 加入部分新噪声。改变 \(\eta\) 时，式（15）的新噪声项与沿用噪声系数要一起改变。

在这个范围内，\(\sigma_t^2\leq\tilde\beta_t\leq1-\overline{\alpha}_{t-1}\)，所以根号约束成立。接下来只需核对 \(\eta=1\) 的均值，便能明确它与 DDPM 的关系。

## 6 eta 等于 1 时怎样对应 DDPM

取 \(\sigma_t^2=\tilde\beta_t\)。利用 \(\beta_t=1-\alpha_t\) 和 \(\overline{\alpha}_t=\alpha_t\overline{\alpha}_{t-1}\)，先化简沿用噪声的系数：

$$\begin{aligned}
1-\overline{\alpha}_{t-1}-\tilde\beta_t
&=(1-\overline{\alpha}_{t-1})
\frac{1-\overline{\alpha}_t-\beta_t}{1-\overline{\alpha}_t}\\
&=\frac{\alpha_t(1-\overline{\alpha}_{t-1})^2}{1-\overline{\alpha}_t}.
\end{aligned} \qquad (19)$$

因此

$$\sqrt{1-\overline{\alpha}_{t-1}-\tilde\beta_t}
=\frac{\sqrt{\alpha_t}(1-\overline{\alpha}_{t-1})}
{\sqrt{1-\overline{\alpha}_t}}. \qquad (20)$$

### 方差之外 还要核对均值

将式（13）和式（20）代回生成公式的前两项，并合并噪声系数：

$$\begin{aligned}
&\sqrt{\overline{\alpha}_{t-1}}\hat x_0
+\sqrt{1-\overline{\alpha}_{t-1}-\tilde\beta_t}\epsilon_\theta(x_t,t)\\
&=\frac{x_t}{\sqrt{\alpha_t}}
+\frac{-(1-\overline{\alpha}_t)+\alpha_t(1-\overline{\alpha}_{t-1})}
{\sqrt{\alpha_t}\sqrt{1-\overline{\alpha}_t}}\epsilon_\theta(x_t,t)\\
&=\frac1{\sqrt{\alpha_t}}
\left(x_t-\frac{\beta_t}{\sqrt{1-\overline{\alpha}_t}}
\epsilon_\theta(x_t,t)\right).
\end{aligned} \qquad (21)$$

最后一步因为分子中的 \(\overline{\alpha}_t\) 与 \(\alpha_t\overline{\alpha}_{t-1}\) 抵消，剩下 \(\alpha_t-1=-\beta_t\)。加回新噪声后得到

$$x_{t-1}=\frac1{\sqrt{\alpha_t}}
\left(x_t-\frac{\beta_t}{\sqrt{1-\overline{\alpha}_t}}
\epsilon_\theta(x_t,t)\right)
+\sqrt{\tilde\beta_t}\,z. \qquad (22)$$

这就是熟悉的 DDPM 噪声参数化更新，且使用后验方差 \(\tilde\beta_t\)。

### 对应关系的三个条件

- **同一套相邻时间步。** 跳步的 \(\eta=1\) 不是执行原 DDPM 全部小步
- **使用后验方差。** 不等同于选 \(\beta_t\) 或其他方差的所有 DDPM 实现；模型预测与额外处理也应一致
- **删掉随机项不够。** 确定性 DDIM 还把沿用噪声系数改为 \(\sqrt{1-\overline{\alpha}_{t-1}}\)，通常不同于仅删除式（22）的随机项

**参数对照。** 本文按论文及官方代码使用 \(\sigma_t^2=\eta^2\tilde\beta_t\)。Lilian Weng 博客写的是 \(\sigma_t^2=\eta\tilde\beta_t\)，两者 \(0\)、\(1\) 端点一致，中间值需按平方关系换算。[论文附录 D.3](https://arxiv.org/html/2010.02502v4#A4.SS3)；[博客相关段落](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#fewer-sampling-steps--distillation)

## 7 怎样跳过中间时间步

式（16）只是用目标噪声等级的系数重新组合，并不要求目标一定是 \(t-1\)。对任意 \(0\leq s&lt;t\)，确定性更新为

$$x_s=\sqrt{\overline{\alpha}_s}\hat x_0
+\sqrt{1-\overline{\alpha}_s}\epsilon_\theta(x_t,t), \qquad (23)$$

\(\hat x_0\) 仍由当前 \(x_t,t\) 计算。这一跳只需要一次网络调用。

### 为什么目标可以是任意较早时刻

先假设真实 \(x_0\) 已知，用同一个 \(\epsilon\sim\mathcal N(0,I)\) 定义

$$\begin{aligned}
x_t&=\sqrt{\overline{\alpha}_t}x_0+\sqrt{1-\overline{\alpha}_t}\epsilon,\\
x_s&=\sqrt{\overline{\alpha}_s}x_0+\sqrt{1-\overline{\alpha}_s}\epsilon.
\end{aligned} \qquad (24)$$

固定原图、让 \(\epsilon\) 变化，\(x_s\) 自然具有目标均值和方差。这里没有使用 \(s=t-1\)，所以可以只在选定时间点上构造路径，再用网络预测替代未知量。

随机版本则把这一跳的新噪声标准差改为

$$\sigma_{t\to s}=\eta\sqrt{\frac{1-\overline{\alpha}_s}{1-\overline{\alpha}_t}
\left(1-\frac{\overline{\alpha}_t}{\overline{\alpha}_s}\right)}, \qquad (25)$$

并更新为

$$\begin{aligned}
x_s={}&\sqrt{\overline{\alpha}_s}\hat x_0
+\sqrt{1-\overline{\alpha}_s-\sigma_{t\to s}^2}\epsilon_\theta(x_t,t)\\
&+\sigma_{t\to s}z,\qquad z\sim\mathcal N(0,I).
\end{aligned} \qquad (26)$$

\(z\) 独立新抽，仍取 \(0\leq\eta\leq1\)。跳步方差必须使用实际目标 \(s\)，不能沿用 \(t\to t-1\) 的数值。\(\eta=1\) 对应所选子序列上的后验方差随机更新。

### 时间编号与最终输出

选 \(0=\tau_0<\tau_1<\cdots<\tau_S=T\)，倒序执行 \(t=\tau_i\to s=\tau_{i-1}\)。例如：

$$1000\to800\to600\to400\to200\to0$$

共五次更新。到 \(800\) 时输入网络的仍是 \(800\)，使用原日程的 \(\overline{\alpha}_{800}\)，不能改成“第二站”的 \(2\)。

最后 \(s=0\) 时，\(\overline{\alpha}_0=1\) 且 \(\sigma_{t\to0}=0\)，所以直接输出 \(x_0^{\mathrm{out}}=\hat x_0\)，不再调用时间为零的网络。

**能跳步不保证大步无损。** 直接跳跃省去了中间重新预测，通常不等于逐个小步执行；\(T\to0\) 虽可计算，却未被普通噪声训练保证能一次产生高质量结果。[论文第 4.2 节](https://arxiv.org/html/2010.02502v4#S4.SS2)；[附录 C.1](https://arxiv.org/html/2010.02502v4#A3.SS1)

## 8 确定性采样的简短伪代码

下面取 \(\eta=0\)。alpha_bar[t] 对应 \(\overline{\alpha}_t\)，times 保存递减的原始时间编号。

```python
# model: 已训练的噪声预测网络；alpha_bar[0] = 1
# times: 例如 [1000, 800, 600, 400, 200, 0]
x = randn(image_shape)                    # 初始 x_T

for t, s in adjacent_pairs(times):
    eps = model(x, t)                     # 每到一站重新预测
    x0_hat = (x - sqrt(1 - alpha_bar[t]) * eps) / sqrt(alpha_bar[t])
    x = sqrt(alpha_bar[s]) * x0_hat + sqrt(1 - alpha_bar[s]) * eps

return x                                 # 最后 s = 0 时输出 x0_hat
```

循环中只有三个动作：**预测噪声 → 估计原图 → 按目标噪声等级重新组合。** 每一轮重新计算 eps 和 x0_hat；它们不是整条轨迹共用的常量。

### 实现时保留四个检查

1. **时间索引一致。** 数学时刻、数组下标和网络的时间编码要对应；跳步仍使用训练时的噪声等级
2. **当前累计系数为正。** 调用网络时 \(\overline{\alpha}_t>0\)，避免式（13）除零；干净端点 \(0\) 只作为目标
3. **使用推理模式。** 关闭梯度，并处理 dropout 等随机操作，再讨论固定初始噪声后的确定性
4. **额外处理会改变原公式。** 本文没有裁剪 \(\hat x_0\)；若加入裁剪，式（14）的残差等式一般不再自动成立

官方 generalized_steps 使用当前站与目标站的累计系数，计算原图估计和式（25）（26）的两份噪声系数。其代码用 \(-1\) 哨兵对应累计系数 \(1\)，数学上就是本文的干净端点 \(0\)。[官方采样代码](https://github.com/ermongroup/ddim/blob/main/functions/denoising.py)

### 参考

- Jiaming Song、Chenlin Meng、Stefano Ermon，Denoising Diffusion Implicit Models。核心公式对照第 3.1、3.2、4.1、4.2 节及附录 B、C.1、D.3。[论文网页](https://arxiv.org/html/2010.02502v4) / [PDF](https://arxiv.org/pdf/2010.02502)
- Lilian Weng，What are Diffusion Models?。用于与 DDPM 符号及采样直觉衔接；\(\eta\) 约定差异见第 6 节。[博客](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
- 前篇《DDPM 从似然到噪声预测》。用于复习后验方差与噪声参数化均值。[DDPM 笔记]({{< relref "posts/ddpm-derivation" >}})
