{
  "title": "DDIM（三）：从真实原图到实际生成公式",
  "description": "逐步代入原图估计，解释精确化简、预测误差与确定性更新。",
  "summary": "逐步代入原图估计，解释精确化简、预测误差与确定性更新。",
  "date": "2026-10-14T16:23:16+08:00",
  "slug": "ddim-3",
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
  "episode": 3
}

前一篇的参考规则知道真实原图。生成时只有当前状态和训练好的网络，这一篇说明怎样替换未知量，以及哪些等式仍然精确成立。

## 1 用噪声预测计算原图估计

已知原图的加噪关系是

$$x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon.$$

若知道真实噪声，减掉噪声项、除以原图缩放系数即可恢复原图。生成时不知道它，先调用网络：

$$\hat\epsilon_t=\epsilon_\theta(x_t,t).$$

再定义

$$\hat x_0^{(t)}=\frac{x_t-\sqrt{1-\bar\alpha_t}\hat\epsilon_t}
{\sqrt{\bar\alpha_t}}.$$

这是当前时刻的原图估计，不保证已经是清晰图片。定义本身保证

$$x_t=\sqrt{\bar\alpha_t}\hat x_0^{(t)}+
\sqrt{1-\bar\alpha_t}\hat\epsilon_t.$$

这是代数恒等式，不代表估计一定正确。

## 2 将两处真实原图一起替换

令 \(c_t=\sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\)。参考更新为

$$x_{t-1}=\sqrt{\bar\alpha_{t-1}}x_0+
 c_t\frac{x_t-\sqrt{\bar\alpha_t}x_0}{\sqrt{1-\bar\alpha_t}}+\sigma_tz.$$

原图项和噪声残差中的 \(x_0\) 都要换成 \(\hat x_0^{(t)}\)。其中残差精确化简：

$$\begin{aligned}
\frac{x_t-\sqrt{\bar\alpha_t}\hat x_0^{(t)}}{\sqrt{1-\bar\alpha_t}}
&=\frac{x_t-\left(x_t-\sqrt{1-\bar\alpha_t}\hat\epsilon_t\right)}
{\sqrt{1-\bar\alpha_t}}\\
&=\hat\epsilon_t.
\end{aligned}$$

于是实际生成更新为

$$\boxed{x_{t-1}=\sqrt{\bar\alpha_{t-1}}\hat x_0^{(t)}+
\sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\hat\epsilon_t+\sigma_tz.}$$

对中间时刻，这定义了模型条件 \(p_{\theta,\sigma}(x_{t-1}\mid x_t)\)：把参考条件中的原图替换成当前网络估计。生成从 \(x_T\sim\mathcal N(0,I)\) 出发，因为参考末端在 \(\bar\alpha_T\) 很小时接近标准高斯。

## 3 为什么参考边缘证明不能直接用于生成

参考证明使用真实原图与真实标准高斯噪声。实际生成中，原图是估计，预测噪声不保证服从标准高斯，而且生成得到的当前状态也可能偏离参考分布。

可以在真实加噪样本上做一个误差分析。记

$$\delta=\hat\epsilon_t-\epsilon.$$

则

$$\hat x_0^{(t)}=x_0-\sqrt{\frac{1-\bar\alpha_t}{\bar\alpha_t}}\delta.$$

用同一份新噪声比较参考与模型更新：

$$\begin{aligned}
x_{t-1}^{\mathrm{model}}={}&
\underbrace{\sqrt{\bar\alpha_{t-1}}x_0+c_t\epsilon+\sigma_tz}_{\text{参考更新}}\\
&+\left(c_t-\sqrt{\bar\alpha_{t-1}}
\sqrt{\frac{1-\bar\alpha_t}{\bar\alpha_t}}\right)\delta.
\end{aligned}$$

多出的误差项可能改变分布。因此“参考构造精确保留边缘”与“网络采样精确生成数据分布”是不同结论。网络优化噪声平方误差，理想输出是给定输入的平均噪声估计，也不是辨认某次唯一历史噪声。

## 4 确定性更新怎样理解

令 \(\sigma_t=0\)：

$$x_{t-1}=\sqrt{\bar\alpha_{t-1}}\hat x_0^{(t)}+
\sqrt{1-\bar\alpha_{t-1}}\hat\epsilon_t.$$

对照当前状态的分解，一次更新就是用同一份预测原图和预测噪声，按更低噪声等级重新组合。到达下一状态后，需要重新调用网络，重新计算两份估计；不能把第一次预测一直使用到最后。

固定初始噪声、模型、时间日程及确定的推理计算，轨迹才确定。初始噪声仍然随机，所以不同起点可以产生不同图片。改变时间日程也通常会改变中间预测和最终结果。

实现时当前 \(\bar\alpha_t\) 必须为正，不能为了表示“纯噪声”直接设为零再除以其平方根。若裁剪原图估计，残差等于原预测噪声的恒等式一般不再自动成立，需要明确如何处理后续方向项。

参考：[DDIM 原论文第 3.2、4.1 节](https://arxiv.org/html/2010.02502v4)。


[前一篇]({{< relref "posts/ddim-2" >}})

[继续阅读下一篇]({{< relref "posts/ddim-4" >}})
