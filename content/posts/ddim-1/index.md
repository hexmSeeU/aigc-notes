{
  "title": "DDIM（一）：为什么可以换一条生成路径",
  "description": "分清训练题与采样规则，用独立噪声和共享噪声说明边缘相同、路径不同。",
  "summary": "分清训练题与采样规则，用独立噪声和共享噪声说明边缘相同、路径不同。",
  "date": "2026-10-14T16:23:16+08:00",
  "slug": "ddim-1",
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
  "episode": 1
}

一次确定性 DDIM 更新可以先看成三个动作：预测噪声、估计原图、用目标噪声等级的系数重新组合。为什么这个规则值得尝试？这一篇先讲动机，不急着证明公式。

## 1 网络预测与采样更新是两个环节

沿用 DDPM 的符号：原图是 \(x_0\)，当前状态是 \(x_t\)，图片视为 \(d\) 维向量。

$$\alpha_t=1-\beta_t,\qquad
\bar\alpha_t=\prod_{k=1}^{t}\alpha_k,\qquad \bar\alpha_0=1.$$

\(0<\beta_t<1\)，所以累计信号系数随时间减小。DDIM 原论文的 \(\alpha_t\) 对应本文的 \(\bar\alpha_t\)，不要与本文单步系数混用。

DDPM 训练时，抽原图、时间和标准高斯噪声，直接制造一个带噪输入：

$$x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,
\qquad\epsilon\sim\mathcal N(0,I).$$

网络输入是 \(x_t,t\)，输出 \(\epsilon_\theta(x_t,t)\)，训练监督是实际抽取的 \(\epsilon\)。这一道题不要求网络输出 \(x_{t-1}\)，也不需要先制造所有中间状态。

生成时则不同：网络预测之后，还需要一个公式，决定怎样从当前状态产生下一状态。所谓“连接方式”，就是这个状态更新的规则。它规定保留多少当前信息、要不要加入新噪声、目标时刻是哪一个。

## 2 为什么作者想改变连接

原始 DDPM 采样沿相邻时间步执行，需要多次网络调用。作者想保留已经训练好的噪声预测能力，重新设计更新规则，使生成可以只经过一部分时间点。

这不是说原 DDPM 路径内部的连接可以随意改。指定 DDPM 正向过程后，连接已经确定。DDIM 的自由来自重新选择参考路径，同时保留每个时刻的训练分布。

目标是始终保持

$$q_\sigma(x_t\mid x_0)=\mathcal N\!\left(
\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I\right).$$

下标 \(\sigma\) 表示待构造的新过程；这里的“边缘”指固定原图、只观察一个时刻的分布。

## 3 单时刻分布相同 不代表路径相同

固定同一张原图，选择 \(0<s<t\)。为缩短表达，记 \(a_r=\sqrt{\bar\alpha_r}\)、\(b_r=\sqrt{1-\bar\alpha_r}\)。下面是两个明确的参考构造。

构造 A：独立抽取两份噪声。

$$\epsilon_s,\epsilon_t\overset{\mathrm{ind}}{\sim}\mathcal N(0,I),
\qquad x_s=a_sx_0+b_s\epsilon_s,\quad x_t=a_tx_0+b_t\epsilon_t.$$

构造 B：只抽取一份噪声，两个时刻共同使用。

$$\epsilon\sim\mathcal N(0,I),
\qquad x_s=a_sx_0+b_s\epsilon,\quad x_t=a_tx_0+b_t\epsilon.$$

分别观察任意一个时刻，两种构造都是“同一原图项加同样尺度的标准高斯”，因此单时刻分布相同。但知道 \(x_0,x_t\) 后，A 的 \(x_s\) 仍随机；B 的 \(x_s\) 可以通过反解共享噪声确定。

**构造 A 不是 DDPM 同一条路径。** DDPM 每一步新增噪声独立，但后续状态继承前面噪声，所以不同时刻的累计噪声相关。例如

$$\begin{aligned}
x_1&=\sqrt{\alpha_1}x_0+\sqrt{\beta_1}z_1,\\
x_2&=\sqrt{\alpha_2\alpha_1}x_0+
\sqrt{\alpha_2\beta_1}z_1+\sqrt{\beta_2}z_2.
\end{aligned}$$

两式都含 \(z_1\)。独立制造不同时间的训练样本，与制造同一条 DDPM 路径，是不同事情。

## 4 保留边缘分布 为什么可以沿用训练

时间编号代表训练时规定的噪声等级。保持原图数据分布、时间抽样规则以及上述条件分布，就保持 \((x_0,x_t,t)\) 的训练联合分布。监督噪声又能反解：

$$\epsilon=\frac{x_t-\sqrt{\bar\alpha_t}x_0}{\sqrt{1-\bar\alpha_t}}.$$

因此，输入与监督答案的联合分布保持不变，简化噪声损失也保持不变：

$$L_{\mathrm{simple}}(\theta)=
\mathbb E_{x_0,t,\epsilon}
\left[\left\|\epsilon-\epsilon_\theta\!\left(
\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,t
\right)\right\|^2\right].$$

这里原图按数据分布抽取，时间均匀抽取，噪声是标准高斯，三者独立。这项损失只使用单时刻样本，不使用整条路径的连接。

这只说明简化目标相同。不同参考过程的完整变分目标一般有不同时间权重；原论文关于权重不影响最优解的讨论还有参数不跨时间共享等条件，不能扩大成所有 ELBO 无条件完全相同。

## 5 看一眼最终要得到的更新

$$\hat\epsilon_t=\epsilon_\theta(x_t,t),\qquad
\hat x_0^{(t)}=\frac{x_t-\sqrt{1-\bar\alpha_t}\hat\epsilon_t}
{\sqrt{\bar\alpha_t}},$$

$$x_s=\sqrt{\bar\alpha_s}\hat x_0^{(t)}+
\sqrt{1-\bar\alpha_s}\hat\epsilon_t,\qquad 0\leq s<t.$$

它在一跳内复用当前预测，只改变混合系数。我们还没有证明它的来源；下一篇先假装知道真实原图，构造参考连接，再推正向条件。

参考：[DDIM 原论文第 3 节](https://arxiv.org/html/2010.02502v4)。


[继续阅读下一篇]({{< relref "posts/ddim-2" >}})
