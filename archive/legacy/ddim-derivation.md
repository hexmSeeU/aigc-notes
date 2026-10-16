{
  "title": "DDIM：从边缘分布到跳步采样",
  "description": "完整保留五篇 DDIM 学习路线：训练自由度、参考过程、实际生成、DDPM 对应与跳步方差。",
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
  "summary": "按动机与构造一步一步推导，区分参考过程和生成模型，附完整跨步 sigma 推导。",
  "archiveOrder": 4,
  "lastmod": "2026-10-14T16:23:16+08:00",
  "hiddenInHomeList": true,
  "hiddenInRss": true,
  "hiddenInLists": true,
  "legacyCollection": "ddim"
}

本页保留原链接，汇总更新后的五篇推导。建议从[DDIM 学习合集]({{< relref "collections/ddim" >}})按篇阅读。

## DDIM（一）：为什么可以换一条生成路径

一次确定性 DDIM 更新可以先看成三个动作：预测噪声、估计原图、用目标噪声等级的系数重新组合。为什么这个规则值得尝试？这一篇先讲动机，不急着证明公式。

### 1 网络预测与采样更新是两个环节

沿用 DDPM 的符号：原图是 \(x_0\)，当前状态是 \(x_t\)，图片视为 \(d\) 维向量。

$$\alpha_t=1-\beta_t,\qquad
\bar\alpha_t=\prod_{k=1}^{t}\alpha_k,\qquad \bar\alpha_0=1.$$

\(0<\beta_t<1\)，所以累计信号系数随时间减小。DDIM 原论文的 \(\alpha_t\) 对应本文的 \(\bar\alpha_t\)，不要与本文单步系数混用。

DDPM 训练时，抽原图、时间和标准高斯噪声，直接制造一个带噪输入：

$$x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,
\qquad\epsilon\sim\mathcal N(0,I).$$

网络输入是 \(x_t,t\)，输出 \(\epsilon_\theta(x_t,t)\)，训练监督是实际抽取的 \(\epsilon\)。这一道题不要求网络输出 \(x_{t-1}\)，也不需要先制造所有中间状态。

生成时则不同：网络预测之后，还需要一个公式，决定怎样从当前状态产生下一状态。所谓“连接方式”，就是这个状态更新的规则。它规定保留多少当前信息、要不要加入新噪声、目标时刻是哪一个。

### 2 为什么作者想改变连接

原始 DDPM 采样沿相邻时间步执行，需要多次网络调用。作者想保留已经训练好的噪声预测能力，重新设计更新规则，使生成可以只经过一部分时间点。

这不是说原 DDPM 路径内部的连接可以随意改。指定 DDPM 正向过程后，连接已经确定。DDIM 的自由来自重新选择参考路径，同时保留每个时刻的训练分布。

目标是始终保持

$$q_\sigma(x_t\mid x_0)=\mathcal N\!\left(
\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I\right).$$

下标 \(\sigma\) 表示待构造的新过程；这里的“边缘”指固定原图、只观察一个时刻的分布。

### 3 单时刻分布相同 不代表路径相同

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

### 4 保留边缘分布 为什么可以沿用训练

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

### 5 看一眼最终要得到的更新

$$\hat\epsilon_t=\epsilon_\theta(x_t,t),\qquad
\hat x_0^{(t)}=\frac{x_t-\sqrt{1-\bar\alpha_t}\hat\epsilon_t}
{\sqrt{\bar\alpha_t}},$$

$$x_s=\sqrt{\bar\alpha_s}\hat x_0^{(t)}+
\sqrt{1-\bar\alpha_s}\hat\epsilon_t,\qquad 0\leq s<t.$$

它在一跳内复用当前预测，只改变混合系数。我们还没有证明它的来源；下一篇先假装知道真实原图，构造参考连接，再推正向条件。

参考：[DDIM 原论文第 3 节](https://arxiv.org/html/2010.02502v4)。

## DDIM（二）：怎样构造参考正向与反向过程

这一篇暂时假设真实原图 \(x_0\) 已知。我们先设计反向参考条件，再验证它保留单时刻分布，最后用贝叶斯公式推导对应的正向条件。这里还不是实际生成算法。

### 1 从当前状态提取标准化噪声

保留 DDPM 的单时刻分布后，定义

$$e_t=\frac{x_t-\sqrt{\bar\alpha_t}x_0}{\sqrt{1-\bar\alpha_t}}.$$

固定原图、让当前状态按参考分布变化，有 \(e_t\sim\mathcal N(0,I)\)。但同时固定 \(x_0,x_t\) 时，\(e_t\) 已经是一个确定的向量。

### 2 把目标噪声方差分成两份

目标状态需要原图项 \(\sqrt{\bar\alpha_{t-1}}x_0\)，以及总方差 \((1-\bar\alpha_{t-1})I\) 的噪声。考虑 \(t\geq2\)，选择

$$x_{t-1}=\sqrt{\bar\alpha_{t-1}}x_0+c_te_t+\sigma_tz,
\qquad z\sim\mathcal N(0,I).$$

新噪声 \(z\) 独立于已有状态。复用 \(e_t\) 让两时刻建立联系，加入 \(z\) 让连接有可调随机性。两份标准高斯独立，方差随系数平方缩放，因此要求

$$c_t^2+\sigma_t^2=1-\bar\alpha_{t-1}.$$

选择非负系数：

$$c_t=\sqrt{1-\bar\alpha_{t-1}-\sigma_t^2},
\qquad0\leq\sigma_t^2\leq1-\bar\alpha_{t-1}.$$

这是一种设计，不是“保持边缘”唯一能推出的答案。

### 3 验证只固定原图时的目标分布

只固定 \(x_0\)，两份噪声都随机且均值为零：

$$\mathbb E[x_{t-1}\mid x_0]=\sqrt{\bar\alpha_{t-1}}x_0,$$

$$\operatorname{Cov}(x_{t-1}\mid x_0)
=(c_t^2+\sigma_t^2)I=(1-\bar\alpha_{t-1})I.$$

独立高斯的线性组合仍是高斯，所以目标边缘分布得到保留。再同时固定 \(x_t\)，旧噪声已确定，条件方差则只有 \(\sigma_t^2I\)。两种方差没有矛盾，因为固定的条件不同。

把 \(e_t\) 的定义代入，得到参考反向分布：

$$\begin{aligned}
q_\sigma(x_{t-1}\mid x_t,x_0)
=\mathcal N\!\Bigg(&\sqrt{\bar\alpha_{t-1}}x_0\\
&+c_t\frac{x_t-\sqrt{\bar\alpha_t}x_0}{\sqrt{1-\bar\alpha_t}},
\sigma_t^2I\Bigg).
\end{aligned}$$

先规定正确的末端边缘，再倒序应用这些条件，得到整条路径：

$$q_\sigma(x_{1:T}\mid x_0)
=q_\sigma(x_T\mid x_0)\prod_{t=2}^{T}q_\sigma(x_{t-1}\mid x_t,x_0).$$

刚才的证明保证“当前边缘正确，就得到前一时刻的正确边缘”，因此可以逐步递推到所有时刻。

### 4 为什么引入 rho 以及怎样换回图片

定义标准化目标噪声：

$$e_{t-1}=\frac{x_{t-1}-\sqrt{\bar\alpha_{t-1}}x_0}
{\sqrt{1-\bar\alpha_{t-1}}}.$$

将图片构造除以目标标准差，得到

$$e_{t-1}=\rho_te_t+\sqrt{1-\rho_t^2}z,
\qquad\rho_t=\sqrt{1-\frac{\sigma_t^2}{1-\bar\alpha_{t-1}}}.$$

\(\rho_t\) 不是独立参数，只是标准化后的旧噪声系数。反过来乘回目标标准差：

$$\begin{aligned}
x_{t-1}={}&\sqrt{\bar\alpha_{t-1}}x_0\\
&+\sqrt{1-\bar\alpha_{t-1}}\rho_te_t
+\sqrt{1-\bar\alpha_{t-1}}\sqrt{1-\rho_t^2}z.
\end{aligned}$$

两个系数分别化简为

$$\sqrt{1-\bar\alpha_{t-1}}\rho_t=c_t,\qquad
\sqrt{1-\bar\alpha_{t-1}}\sqrt{1-\rho_t^2}=\sigma_t.$$

这就回到前面的图片构造。

### 5 反方向的标准噪声条件怎样推出来

先设 \(\rho=\rho_t<1\)。根据构造，\(e_t\) 是标准高斯，给定它后，\(e_{t-1}\) 是均值 \(\rho e_t\)、方差 \((1-\rho^2)I\) 的高斯。用贝叶斯公式：

$$p(e_t\mid e_{t-1})\propto
\exp\!\left[-\frac12\left(
\frac{\|e_{t-1}-\rho e_t\|^2}{1-\rho^2}+\|e_t\|^2
\right)\right].$$

固定 \(e_{t-1}\)，展开并配方：

$$\begin{aligned}
&\|e_{t-1}-\rho e_t\|^2+(1-\rho^2)\|e_t\|^2\\
&=\|e_t\|^2-2\rho e_{t-1}^{\mathsf T}e_t+\|e_{t-1}\|^2\\
&=\|e_t-\rho e_{t-1}\|^2+(1-\rho^2)\|e_{t-1}\|^2.
\end{aligned}$$

除以 \(1-\rho^2\) 后，最后一项不依赖正在讨论的变量 \(e_t\)，并入归一化常数。于是

$$e_t\mid e_{t-1}\sim\mathcal N(\rho e_{t-1},(1-\rho^2)I).$$

抽样写法是

$$e_t=\rho_te_{t-1}+\sqrt{1-\rho_t^2}u,
\qquad u\sim\mathcal N(0,I).$$

\(u\) 独立于条件输入，不是原反向构造里的同一份 \(z\)。我们推导的是条件分布，不是在倒用那次随机抽样。

### 6 对应的正向条件仍然依赖原图

将标准噪声换回图片，定义

$$\begin{aligned}
m_t^+(x_{t-1},x_0)={}&\sqrt{\bar\alpha_t}x_0\\
&+\rho_t\sqrt{\frac{1-\bar\alpha_t}{1-\bar\alpha_{t-1}}}
\left(x_{t-1}-\sqrt{\bar\alpha_{t-1}}x_0\right).
\end{aligned}$$

因此

$$q_\sigma(x_t\mid x_{t-1},x_0)=\mathcal N\!\left(
m_t^+(x_{t-1},x_0),
\frac{1-\bar\alpha_t}{1-\bar\alpha_{t-1}}\sigma_t^2I\right).$$

也可以直接写贝叶斯关系：

$$q_\sigma(x_t\mid x_{t-1},x_0)=
\frac{q_\sigma(x_{t-1}\mid x_t,x_0)q_\sigma(x_t\mid x_0)}
{q_\sigma(x_{t-1}\mid x_0)}.$$

一般不能去掉条件 \(x_0\)。原图本来属于整个随机过程的起点；新过程的转移还显式依赖这个起点，所以通常不再是 DDPM 那种只依赖上一状态的正向马尔可夫过程。

### 7 零新噪声与起点单独处理

\(\sigma_t=0\) 时，\(\rho_t=1\)，有 \(e_t=e_{t-1}\)。正反向连接在已知原图后都确定。零方差应理解为点质量；不能代入含倒方差或对数方差的普通高斯密度、KL 公式。

上面的标准化目标公式要求 \(t\geq2\)。正向起点仍为 \(q_\sigma(x_1\mid x_0)\)，而不是对零分母强行使用同一表达式。

参考：[DDIM 原论文第 3.1 节及附录 B](https://arxiv.org/html/2010.02502v4)。

## DDIM（三）：从真实原图到实际生成公式

前一篇的参考规则知道真实原图。生成时只有当前状态和训练好的网络，这一篇说明怎样替换未知量，以及哪些等式仍然精确成立。

### 1 用噪声预测计算原图估计

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

### 2 将两处真实原图一起替换

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

### 3 为什么参考边缘证明不能直接用于生成

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

### 4 确定性更新怎样理解

令 \(\sigma_t=0\)：

$$x_{t-1}=\sqrt{\bar\alpha_{t-1}}\hat x_0^{(t)}+
\sqrt{1-\bar\alpha_{t-1}}\hat\epsilon_t.$$

对照当前状态的分解，一次更新就是用同一份预测原图和预测噪声，按更低噪声等级重新组合。到达下一状态后，需要重新调用网络，重新计算两份估计；不能把第一次预测一直使用到最后。

固定初始噪声、模型、时间日程及确定的推理计算，轨迹才确定。初始噪声仍然随机，所以不同起点可以产生不同图片。改变时间日程也通常会改变中间预测和最终结果。

实现时当前 \(\bar\alpha_t\) 必须为正，不能为了表示“纯噪声”直接设为零再除以其平方根。若裁剪原图估计，残差等于原预测噪声的恒等式一般不再自动成立，需要明确如何处理后续方向项。

参考：[DDIM 原论文第 3.2、4.1 节](https://arxiv.org/html/2010.02502v4)。

## DDIM（四）：sigma、eta 与 DDPM 的关系

我们已有通用生成公式。本篇选择新噪声大小，逐步验证与 DDPM 的关系，并解释为什么直接删除 DDPM 随机项不等于确定性 DDIM。

### 1 用后验标准差作为基准

\(\sigma_t\) 是新噪声标准差。改变它时，新噪声项与复用噪声系数必须一起变化：

$$x_{t-1}=\sqrt{\bar\alpha_{t-1}}\hat x_0+
\sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\hat\epsilon_t+\sigma_tz.$$

DDPM 已知原图的后验方差是

$$\tilde\beta_t=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t.$$

因此常用选择为

$$\sigma_t=\eta\sqrt{\tilde\beta_t},\qquad0\leq\eta\leq1.$$

\(\eta\) 缩放标准差，方差是 \(\eta^2\tilde\beta_t\)。例如 \(\eta=0.5\) 时是四分之一方差。\(\eta=0\) 不加新噪声，\(\eta=1\) 与 DDPM 后验方差一致，中间值加入部分新噪声。这个范围满足根号约束，但不是参考构造唯一允许的范围。

### 2 eta 等于 1 先化简复用噪声系数

方差相同还不够，需要检查均值。代入 \(\sigma_t^2=\tilde\beta_t\)：

$$\begin{aligned}
1-\bar\alpha_{t-1}-\tilde\beta_t
&=(1-\bar\alpha_{t-1})
\frac{1-\bar\alpha_t-\beta_t}{1-\bar\alpha_t}\\
&=\frac{\alpha_t(1-\bar\alpha_{t-1})^2}{1-\bar\alpha_t}.
\end{aligned}$$

最后一步使用

$$1-\bar\alpha_t-\beta_t
=\alpha_t-\alpha_t\bar\alpha_{t-1}
=\alpha_t(1-\bar\alpha_{t-1}).$$

因此

$$\sqrt{1-\bar\alpha_{t-1}-\tilde\beta_t}
=\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{\sqrt{1-\bar\alpha_t}}.$$

### 3 展开原图项 再合并两个噪声系数

因为 \(\bar\alpha_t=\alpha_t\bar\alpha_{t-1}\)，

$$\begin{aligned}
\sqrt{\bar\alpha_{t-1}}\hat x_0
&=\sqrt{\frac{\bar\alpha_{t-1}}{\bar\alpha_t}}
\left(x_t-\sqrt{1-\bar\alpha_t}\hat\epsilon_t\right)\\
&=\frac{x_t}{\sqrt{\alpha_t}}-
\frac{\sqrt{1-\bar\alpha_t}}{\sqrt{\alpha_t}}\hat\epsilon_t.
\end{aligned}$$

加上第二项后，预测噪声的总系数为

$$\begin{aligned}
&-\frac{\sqrt{1-\bar\alpha_t}}{\sqrt{\alpha_t}}
+\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{\sqrt{1-\bar\alpha_t}}\\
&=\frac{-(1-\bar\alpha_t)+\alpha_t(1-\bar\alpha_{t-1})}
{\sqrt{\alpha_t}\sqrt{1-\bar\alpha_t}}\\
&=-\frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1-\bar\alpha_t}}.
\end{aligned}$$

分子展开后，\(\bar\alpha_t\) 与 \(\alpha_t\bar\alpha_{t-1}\) 抵消，剩下 \(\alpha_t-1=-\beta_t\)。所以完整更新是

$$\boxed{x_{t-1}=\frac1{\sqrt{\alpha_t}}
\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\right)
+\sqrt{\tilde\beta_t}z.}$$

这恢复了使用后验方差的 DDPM 更新。对应关系要求同一套相邻时间步、同一网络预测和相同额外处理；它不代表等同于选择 \(\beta_t\) 或学习方差的所有 DDPM 实现。

### 4 为什么直接删随机项不等于 DDIM

将 DDPM 写回原图估计形式，只删除最后一项，会得到

$$x_{t-1}^{\mathrm{delete}}=
\sqrt{\bar\alpha_{t-1}}\hat x_0+
\sqrt{1-\bar\alpha_{t-1}-\tilde\beta_t}\hat\epsilon_t.$$

而令通用公式的 \(\sigma_t=0\)，得到

$$x_{t-1}^{\mathrm{DDIM}}=
\sqrt{\bar\alpha_{t-1}}\hat x_0+
\sqrt{1-\bar\alpha_{t-1}}\hat\epsilon_t.$$

原图项相同，复用噪声系数不同。参考构造把目标噪声方差分为旧噪声和新噪声。只删除新噪声会少一份方差；确定性 DDIM 则把目标方差全部分给复用噪声。

例如目标方差 \(0.4\)，原本旧噪声 \(0.3\)、新噪声 \(0.1\)：

$$\sqrt{0.3}\epsilon+\sqrt{0.1}z
\quad\longrightarrow\quad
\begin{cases}
\sqrt{0.3}\epsilon&\text{直接删除},\\
\sqrt{0.4}\epsilon&\text{确定性参考构造}.
\end{cases}$$

方差解释针对真实标准噪声的参考过程；实际网络预测不必标准高斯，但两条实际更新公式的系数差异仍是精确的。

参考：[DDIM 原论文第 4.1 节及附录 D.3](https://arxiv.org/html/2010.02502v4)。本文按原论文与官方代码使用标准差参数 \(\eta\)，不要与用 \(\eta\) 直接缩放方差的约定混用。

## DDIM（五）：跳步采样与跨步方差的推导

本篇从相邻步推广到任意 \(t\to s\)，再推导跳步的新噪声系数。确定性控制的是每跳随机性；步数控制的是调用网络的次数，两者是不同选择。

### 1 为什么可以直接选择更早的目标

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

### 2 跨步 sigma 为什么要以 DDPM 后验方差为基准

和相邻步一样，我们选择跨步后验标准差的 \(\eta\) 倍。但首先需要计算基准方差。

下面临时使用的是**原 DDPM 的正向过程**，不是任意 \(\eta\) 下 DDIM 的正向连接。令

$$r=\frac{\bar\alpha_t}{\bar\alpha_s}.$$

我们将证明

$$q_{\mathrm{DDPM}}(x_t\mid x_s)=\mathcal N(\sqrt r\,x_s,(1-r)I).$$

### 3 多个小步怎样合成一个大步

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

### 4 用贝叶斯与配方求跨步后验方差

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

### 5 选择 eta 并写出完整跳步更新

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

### 6 为什么跳步加速 以及代价是什么

选择原始时间子序列，倒序执行，例如

$$1000\to800\to600\to400\to200\to0.$$

每跳只调用一次网络，这个示意序列共五次调用。到达 \(800\) 后输入时间仍是 \(800\)，使用原噪声日程的累计系数，不能改成“第二站”的编号。

网络推理通常占主要计算，减少调用可以加速。上面的五步不保证足够好的图片质量；少步也意味着更少的中间重新估计机会。

最后目标 \(s=0\)，\(\bar\alpha_0=1\)，新噪声标准差为零，直接输出 \(\hat x_0^{(t)}\)，不调用时间零的网络。直接 \(T\to0\) 虽可计算，却只是一次高噪声原图估计，普通训练不保证它能一次生成高质量图片。

### 7 确定性采样伪代码与实现检查

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

### 8 自检

1. 跳到 \(s\) 时，网络输入当前 \(t\) 还是目标 \(s\)？
2. 确定性 DDIM 是否要求整条路径共用第一次噪声预测？
3. \(\eta=1\) 的跳步是否等于原 DDPM 全部相邻步？

#### 参考答案

1. 输入当前 \(x_t,t\)，目标 \(s\) 只进入这一跳的系数；到达后才调用 \(x_s,s\)。
2. 不是。每一跳重新预测，同一份预测只在这一跳中复用。
3. 不是。所选子序列上的更新省去了中间网络调用，通常改变结果。
