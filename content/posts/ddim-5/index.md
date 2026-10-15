{
  "title": "DDIM（五）：跳步采样与跨步方差的推导",
  "description": "证明跨步正向分布，配方推导 sigma，整理时间序列、网络调用与实现。",
  "summary": "证明跨步正向分布，配方推导 sigma，整理时间序列、网络调用与实现。",
  "date": "2026-10-14T16:23:16+08:00",
  "slug": "ddim-5",
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
  "episode": 5
}

本篇从相邻步推广到任意 \(t\to s\)，再推导跳步的新噪声系数。确定性控制的是每跳随机性；步数控制的是调用网络的次数，两者是不同选择。

## 1 为什么可以直接选择更早的目标

固定真实原图，用同一份标准噪声定义

$$\begin{aligned}
x_t&=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,\\
x_s&=\sqrt{\bar\alpha_s}x_0+\sqrt{1-\bar\alpha_s}\epsilon.
\end{aligned}$$

无论两时刻是否相邻，\(x_s\mid x_0\) 都有正确均值和方差。因此可以只在选定时刻构造参考路径，而不经过所有中间状态。

用网络估计替换未知量后，确定性更新为

$$\hat\epsilon_t=\epsilon_\theta(x_t,t),\qquad
\hat x_0^{(t)}=\frac{x_t-\sqrt{1-\bar\alpha_t}\hat\epsilon_t}{\sqrt{\bar\alpha_t}},$$

$$x_s=\sqrt{\bar\alpha_s}\hat x_0^{(t)}+
\sqrt{1-\bar\alpha_s}\hat\epsilon_t.$$

允许构造这一跳，不表示它等于逐个小步执行。小步过程中会反复更新状态与网络预测；直接跳跃省去了这些计算。

## 2 跨步 sigma 为什么要以 DDPM 后验方差为基准

和相邻步一样，我们选择跨步后验标准差的 \(\eta\) 倍。但首先需要计算基准方差。

下面临时使用的是**原 DDPM 的正向过程**，不是任意 \(\eta\) 下 DDIM 的正向连接。令

$$r=\frac{\bar\alpha_t}{\bar\alpha_s}.$$

我们将证明

$$q_{\mathrm{DDPM}}(x_t\mid x_s)=\mathcal N(\sqrt r\,x_s,(1-r)I).$$

## 3 多个小步怎样合成一个大步

DDPM 单步为 \(x_k=\sqrt{\alpha_k}x_{k-1}+\sqrt{\beta_k}z_k\)，各步新噪声独立。从 \(s\) 出发展开两步：

$$\begin{aligned}
x_{s+2}={}&\sqrt{\alpha_{s+2}\alpha_{s+1}}x_s\\
&+\sqrt{\alpha_{s+2}\beta_{s+1}}z_{s+1}
+\sqrt{\beta_{s+2}}z_{s+2}.
\end{aligned}$$

噪声均值为零，总方差为

$$\alpha_{s+2}\beta_{s+1}+\beta_{s+2}
=\alpha_{s+2}(1-\alpha_{s+1})+1-\alpha_{s+2}
=1-\alpha_{s+2}\alpha_{s+1}.$$

令 \(R_k=\prod_{j=s+1}^{k}\alpha_j\)。假设当前条件均值是 \(\sqrt{R_k}x_s\)，方差是 \((1-R_k)I\)。再走一步，方差成为

$$\alpha_{k+1}(1-R_k)+\beta_{k+1}
=1-\alpha_{k+1}R_k=1-R_{k+1}.$$

均值也乘上 \(\sqrt{\alpha_{k+1}}\)。高斯线性组合仍是高斯，所以可一路递推。最后

$$R_t=\prod_{j=s+1}^{t}\alpha_j
=\frac{\prod_{j=1}^{t}\alpha_j}{\prod_{j=1}^{s}\alpha_j}
=\frac{\bar\alpha_t}{\bar\alpha_s}=r.$$

这就得到上一节的大步正向分布。

## 4 用贝叶斯与配方求跨步后验方差

先设 \(0<s<t\)。已知原图时，\(x_s\) 的先验是

$$q(x_s\mid x_0)=\mathcal N(\sqrt{\bar\alpha_s}x_0,(1-\bar\alpha_s)I).$$

大步后验满足

$$q_{\mathrm{DDPM}}(x_s\mid x_t,x_0)
\propto q_{\mathrm{DDPM}}(x_t\mid x_s)q(x_s\mid x_0).$$

代入密度，保留关于变量 \(x_s\) 的指数：

$$\exp\!\left[-\frac12\left(
\frac{\|x_t-\sqrt r\,x_s\|^2}{1-r}
+\frac{\|x_s-\sqrt{\bar\alpha_s}x_0\|^2}{1-\bar\alpha_s}
\right)\right].$$

协方差为 \(vI\) 的高斯，其指数括号中二次项系数是 \(1/v\)。上面两个平方分别贡献 \(r/(1-r)\) 与 \(1/(1-\bar\alpha_s)\)，因此

$$\begin{aligned}
\frac1v&=\frac r{1-r}+\frac1{1-\bar\alpha_s}\\
&=\frac{r(1-\bar\alpha_s)+1-r}{(1-r)(1-\bar\alpha_s)}\\
&=\frac{1-\bar\alpha_t}{(1-r)(1-\bar\alpha_s)}.
\end{aligned}$$

取倒数，得到

$$v=\frac{1-\bar\alpha_s}{1-\bar\alpha_t}
\left(1-\frac{\bar\alpha_t}{\bar\alpha_s}\right).$$

## 5 选择 eta 并写出完整跳步更新

选择 \(\sigma_{t\to s}=\eta\sqrt v\)，于是

$$\boxed{\sigma_{t\to s}=\eta\sqrt{
\frac{1-\bar\alpha_s}{1-\bar\alpha_t}
\left(1-\frac{\bar\alpha_t}{\bar\alpha_s}\right)},\qquad0\leq\eta\leq1.}$$

这个公式包含两件不同的事情：根号里的方差来自 DDPM 后验推导；用 \(\eta\) 缩放标准差是 DDIM 的采样选择，不是保持边缘分布唯一要求的选择。

令 \(s=t-1\)，有 \(1-\bar\alpha_t/\bar\alpha_{t-1}=\beta_t\)，恢复相邻步的 \(\eta\sqrt{\tilde\beta_t}\)。由于 \(\bar\alpha_s>\bar\alpha_t\)，基准方差不超过 \(1-\bar\alpha_s\)，所以所选范围满足根号约束。

实际更新为

$$\begin{aligned}
x_s={}&\sqrt{\bar\alpha_s}\hat x_0^{(t)}\\
&+\sqrt{1-\bar\alpha_s-\sigma_{t\to s}^2}\hat\epsilon_t
+\sigma_{t\to s}z,
\qquad z\sim\mathcal N(0,I).
\end{aligned}$$

新噪声独立抽取。\(\eta=1\) 对应所选子序列的后验方差更新，不等于执行原 DDPM 的全部小步。跳步方差必须使用实际目标 \(s\)，不能沿用相邻步数值。

## 6 为什么跳步加速 以及代价是什么

选择原始时间子序列，倒序执行，例如

$$1000\to800\to600\to400\to200\to0.$$

每跳只调用一次网络，这个示意序列共五次调用。到达 \(800\) 后输入时间仍是 \(800\)，使用原噪声日程的累计系数，不能改成“第二站”的编号。

网络推理通常占主要计算，减少调用可以加速。上面的五步不保证足够好的图片质量；少步也意味着更少的中间重新估计机会。

最后目标 \(s=0\)，\(\bar\alpha_0=1\)，新噪声标准差为零，直接输出 \(\hat x_0^{(t)}\)，不调用时间零的网络。直接 \(T\to0\) 虽可计算，却只是一次高噪声原图估计，普通训练不保证它能一次生成高质量图片。

## 7 确定性采样伪代码与实现检查

```python
# model: 已训练的噪声预测网络；alpha_bar[0] = 1
# times: 例如 [1000, 800, 600, 400, 200, 0]
x = randn(image_shape)  # 初始 x_T
for t, s in adjacent_pairs(times):
    eps = model(x, t)  # 每到一站重新预测
    x0_hat = (x - sqrt(1 - alpha_bar[t]) * eps) / sqrt(alpha_bar[t])
    x = sqrt(alpha_bar[s]) * x0_hat + sqrt(1 - alpha_bar[s]) * eps
return x  # 最后 s = 0 时输出 x0_hat
```

模型使用推理模式，处理 dropout 等随机操作后才讨论确定性；当前累计系数保持正值；数学时刻、数组下标与模型时间编码一致。本文没有裁剪原图估计，若加入额外处理，应明确方向项是否同步重算。

[官方 generalized_steps](https://github.com/ermongroup/ddim/blob/main/functions/denoising.py)使用当前站与目标站的累计系数；其 \(-1\) 哨兵对应累计系数 \(1\)，数学上就是这里的干净端点 \(0\)。

参考：[DDIM 原论文第 4.2 节与附录 C.1、D.3](https://arxiv.org/html/2010.02502v4)。

## 8 自检

1. 跳到 \(s\) 时，网络输入当前 \(t\) 还是目标 \(s\)？
2. 确定性 DDIM 是否要求整条路径共用第一次噪声预测？
3. \(\eta=1\) 的跳步是否等于原 DDPM 全部相邻步？

### 参考答案

1. 输入当前 \(x_t,t\)，目标 \(s\) 只进入这一跳的系数；到达后才调用 \(x_s,s\)。
2. 不是。每一跳重新预测，同一份预测只在这一跳中复用。
3. 不是。所选子序列上的更新省去了中间网络调用，通常改变结果。
