{
  "title": "MiniMax H3 FL2VA：从首尾帧到音视频联合生成",
  "description": "沿着一条具体样本，拆解 MiniMax H3 FL2VA 的双路径条件编码、token 尺寸、联合注意力、AdaLN、SwiGLU 与双调度采样。",
  "summary": "从 124 帧的具体维度出发，追踪首尾帧、音频和视频如何进入同一个 Transformer，再还原为两种 latent 的更新量。",
  "date": "2026-10-28T09:30:00+00:00",
  "slug": "minimax-h3-fl2va",
  "categories": ["前沿"],
  "tags": ["MiniMax H3", "音视频生成", "DiT", "Flow Matching"],
  "draft": false,
  "math": true
}

给定一张“人拿着杯子”的首帧、一张“杯子放在桌上”的尾帧，再加一句“他放下杯子，说了一句‘好了’，伴随杯子碰到桌面的声音”，模型需要同时生成动作、说话声与碰撞声，并安排它们发生的时间。

MiniMax H3 的 FL2VA 模式提供了这样一条生成路径。本文沿着**输入条件 → latent → token → 共享 Transformer → 两种更新量 → 迭代解码**展开。重点不是罗列模块名，而是把每一步的形状、计算顺序和条件作用讲清楚。

**阅读范围。** 本文是模型卡与公开推理代码的结构阅读笔记，不将它们称作已经核实的论文实验结论。任务定义参考 [MiniMax 官方模型卡](https://huggingface.co/MiniMaxAI/MiniMax-H3#model-variants-and-input-specifications)；具体计算以 DiffSynth-Studio 提交 [e5a44c4](https://github.com/modelscope/DiffSynth-Studio/tree/e5a44c42776978457b61d51d595a1f6746c3d87a) 为准，核查日期为 2026 年 10 月 28 日。以下数值是该实现的配置与形状推导，不是性能测量，也不代表所有 H3 变体。

## 1. FL2VA 的输入与输出

FL2VA 表示由首尾帧条件生成音视频。官方模型卡中的 H3-Base-FL2VA 支持零张、一张或两张输入图像：

| 已知条件 | 对应任务 |
|---|---|
| 文字 | 纯文字生成音视频 |
| 文字＋首帧 | 从给定画面开始 |
| 文字＋尾帧 | 向给定画面结束 |
| 文字＋首尾帧 | 生成连接两个画面的过程 |

“从给定画面开始”描述的是条件任务；是否达到像素级一致，需要另行判断，不能仅从任务名称推出。后文会回到这一点。

生成并不直接发生在像素和音频采样点上。两个 VAE 分别定义视频与音频的压缩表示，也就是 latent。模型先生成 latent，再由各自的解码器还原：

~~~text
文字、可选首尾帧 → 上下文编码器 → 上下文特征
          首尾帧 → 视频 VAE    → 关键帧条件 latent

上下文 + 关键帧条件 + 音频噪声 + 视频噪声
                     ↓
             共享 Transformer
                     ↓
          音频更新量 / 视频更新量
                     ↓
          分别更新 latent，反复迭代
                     ↓
          音频 VAE / 视频 VAE 解码
                     ↓
               带声音的视频
~~~

统一主干让两种待生成内容在采样过程中交换信息。这为声音与动作的配合提供了计算途径，但架构本身不保证每个样本都能精确同步。

## 2. 首尾帧为什么需要两条编码路径？

首尾帧既进入上下文编码器，也进入视频 VAE。两条路径的结果不能互相替代。

**上下文路径**把文字与图像一起处理，输出供主干读取的上下文特征。代码把组件命名为 text_encoder，但传入关键帧时，PromptEmbedder 也会传递图像张量与网格信息。因此不能把这里的上下文简单理解成纯文字 embedding。

**VAE 路径**把每张关键帧编码到生成所用的视觉 latent 空间，再切成条件 token。这些 token 带有首帧或尾帧对应的位置坐标，进入统一序列。

可以将两者的分工理解为：一条帮助解释“画面内容与指令的关系”，另一条提供“生成空间里的画面表示”。这是对计算路径的解释，不是将两种能力严格分离的实验结论。相关实现见 [PromptEmbedder 与 KeyframeEncoder](https://github.com/modelscope/DiffSynth-Studio/blob/e5a44c42776978457b61d51d595a1f6746c3d87a/diffsynth/pipelines/minimax_h3_audio_video.py)。

## 3. 用一组具体尺寸追踪 token

下面固定单个样本：**124 帧、480×832、24 fps，提供首尾两张图片**。分辨率用于说明尺寸换算，不是声称这是官方推荐的默认输出。另作一个教学假设：上下文编码器输出 \(L_c=1000\) 个 token。实际 \(L_c\) 由输入与处理器决定，并不固定为 1000。

统一记号如下：

| 记号 | 含义 |
|---|---|
| \(z_v,z_a\) | 视频、音频 latent |
| \(L_c,L_k,L_a,L_v\) | 上下文、关键帧、音频、视频 token 数 |
| \(L\) | 有效序列长度 |
| \(d=5376\) | 主干隐藏宽度 |
| \(k\) | 外层采样迭代编号 |
| \(\tau_v,\tau_a\) | 传给主干的音视频生成进度 |
| \((p_t,p_h,p_w)\) | token 的时空位置坐标 |

**生成进度 \(\tau\) 和视频中的时间位置 \(p_t\) 是两件事。** 前者描述当前去噪走到哪里，后者描述内容发生在哪里。

### 3.1 视频：从五维 latent 到二维 token 表

这份实现先对帧数做对齐，使其满足 \(17n+5\)。124 已经满足 \(124=17\times7+5\)，无需向上调整。NoiseInitializer 使用的 latent 时间长度为：

$$
T_v=\left\lfloor\frac{124-5}{17}\right\rfloor\times5+2=37.
$$

空间维度压缩 16 倍，因此：

$$
z_v\in\mathbb R^{1\times24\times37\times30\times52},
\qquad 30=480/16,\quad52=832/16.
$$

这五个维度依次为 batch、通道、latent 时间、高、宽。**37 来自上述分段对齐规则，不能直接替换成 \(124/4\)。**

再使用 \(1\times2\times2\) 的 latent patch。每个 patch 展开后的特征宽度为：

$$
D_v=24\times1\times2\times2=96.
$$

token 数量为：

$$
L_v=37\times(30/2)\times(52/2)
=37\times15\times26=14430.
$$

所以形状变化是：

~~~text
视频 latent：[1, 24, 37, 30, 52]
       ↓ patchify
视频 token： [14430, 96]
~~~

96 是每个 token 的特征宽度，14430 是 token 数量，两者不能混淆。

### 3.2 首尾帧：每张提供 390 个条件 token

每张图片先调整到目标画布大小，再编码为：

~~~text
[1, 24, 1, 30, 52]
~~~

同样使用 \(1\times2\times2\) patch，一张图得到 \(15\times26=390\) 个 96 维 token；两张图合计：

$$
L_k=2\times390=780,\qquad
\text{关键帧 token 形状}=[780,96].
$$

它们是额外条件，不占用待生成视频的 14430 个 token。

### 3.3 音频：两个声道按时间展开

该视频按帧数除以帧率计算的时长约为 \(124/24=5.17\) 秒。代码按每秒 40 个音频 latent 位置取整：

$$
T_a=\operatorname{round}\left(\frac{124}{24}\times40\right)=207.
$$

音频 latent 的形状为：

~~~text
[2, 32, 207]
声道 特征 时间
~~~

这里最前面的 2 是两个声道，不是两个样本。pack_audio 按声道优先排列，将每个声道的每个时间位置变成一个 token：

$$
L_a=2\times207=414,\qquad
\text{音频 token 形状}=[414,32].
$$

### 3.4 相同宽度，才能沿长度拼接

该 DiT 的上下文输入宽度为 5120，视觉 patch 宽度为 96，音频宽度为 32。输入投影将它们统一到 \(d=5376\)：

| 来源 | 投影前 | 投影后 |
|---|---|---|
| 上下文（教学假设） | \([1000,5120]\) | \([1000,5376]\) |
| 首尾帧 | \([780,96]\) | \([780,5376]\) |
| 音频 | \([414,32]\) | \([414,5376]\) |
| 视频 | \([14430,96]\) | \([14430,5376]\) |

首尾帧与目标视频使用同一个视觉输入投影。**上下文投影后还经过默认两层的 Token Refiner**，再放入共享序列；因此将整个输入端概括成“几次 Linear 后直接拼接”会省略这一层处理。

FL2VA 的序列顺序是：

~~~text
[上下文 | 首尾帧条件 | 音频 | 视频 | padding]
~~~

有效长度为：

$$
L=L_c+L_k+L_a+L_v
=1000+780+414+14430=16624.
$$

实现补齐到 64 的倍数，得到 \([16640,5376]\)，其中有 16 个 padding token。有效内容和 padding 被划分为不同注意力区段，不能将 padding 也当作有效条件。

以上尺寸可对照 [NoiseInitializer、PackedSequenceBuilder](https://github.com/modelscope/DiffSynth-Studio/blob/e5a44c42776978457b61d51d595a1f6746c3d87a/diffsynth/pipelines/minimax_h3_audio_video.py) 与 [DiT 输入投影、Token Refiner](https://github.com/modelscope/DiffSynth-Studio/blob/e5a44c42776978457b61d51d595a1f6746c3d87a/diffsynth/models/minimax_h3_dit.py)。后文为便于阅读，使用有效长度 \(L=16624\)，省略 padding。

## 4. 联合注意力怎样交换信息？

设某层送入注意力的特征为 \(Y\in\mathbb R^{L\times5376}\)。该实现使用 56 个 head，每个 head 的维度为 128。Q、K、V 投影并拆分后，分别为：

$$
Q,K,V\in\mathbb R^{L\times56\times128}.
$$

这里注意力内部总宽度是：

$$
56\times128=7168,
$$

**它并不等于主干的 5376。** 线性投影可以改变宽度，不能套用“head 数乘 head 维度必等于输入宽度”的假设。

对一个 head，先对 Q、K 做 RMSNorm，再施加 RoPE，记结果为 \(\widehat Q,\widehat K\)。注意力计算为：

$$
A=\operatorname{softmax}
\left(\frac{\widehat Q\widehat K^\top}{\sqrt{128}}\right),
\qquad O=AV.
$$

这里 softmax 沿每一行的 key 维度归一化。数学上，\(A\) 的形状是 \([16624,16624]\)：第 \(i\) 行表示第 \(i\) 个 token 读取其他 token 时使用的权重。某个视频 token 可以读取关键帧、文字上下文与音频 token；音频 token 也可以读取视频信息。

每个 head 输出 \([L,128]\)，56 个 head 拼接后，再投影回 \([L,5376]\)。这样注意力改变了内容，却保持主干的序列长度和输出宽度。

这个平方矩阵是数学表达，不代表高效注意力内核必须将它完整保存。本文也不据此估算运行显存。计算细节见 [MiniMaxH3Attention](https://github.com/modelscope/DiffSynth-Studio/blob/e5a44c42776978457b61d51d595a1f6746c3d87a/diffsynth/models/minimax_h3_dit.py)。

## 5. 展平以后，位置与模态怎样保留？

视频 token 展平前对应一个 \([37,15,26]\) 网格。展平仅改变存放顺序，三个位置坐标 \((p_t,p_h,p_w)\) 仍会传入三维 RoPE，用于旋转注意力的 Q、K。该配置实际旋转每个 head 的前 96 维，剩余 32 维直通；V 不做这次旋转。上下文 Token Refiner 的注意力也不使用这套 RoPE。

实际位置不是直接照抄整数网格。实现对空间与时间坐标做了缩放；首帧条件放在目标时间轴开头，尾帧条件放在末端。音频也有时间坐标，两个声道还通过空间轴位置区分。因此不能将“首尾帧排在音频之前”误解为它们只约束视频最早的一段。

位置编码回答“这个 token 对应哪里”；模态标签与时间调制回答“应该按哪类表示、哪种生成进度处理它”。关键帧与目标视频同属视觉模态，但它们可以具有不同的生成进度。上下文区域也包含编码器产生的多模态特征，不应只凭序列区域名称猜测每个 token 的实际标签。

## 6. AdaLN：完整写出一个共享 block

默认主干包含 50 个 block，不包括前面的两层上下文 Token Refiner。每个 block 都包含注意力和 SwiGLU FFN 两个残差分支。两者共享跨模态的主干计算权重，再通过依赖模态与生成进度的调制参数调整各类 token。

代码沿用 AdaLN 命名，但这里实际的归一化操作是 **RMSNorm**。不能仅凭名称把它写成标准 LayerNorm。

### 6.1 注意力分支：先调制，后门控残差

令一个 block 的输入为 \(X\)。对第 \(i\) 个 token，调制分支给出三个 \(d\) 维向量：平移 \(b_{\mathrm{attn},i}\)、缩放 \(s_{\mathrm{attn},i}\)、门控 \(g_{\mathrm{attn},i}\)。它们由该 token 的模态与生成进度选择，同一组合共享一组向量。

为了简洁，把按 token 选出的向量堆成与 \(X\) 同形状的矩阵，得到：

$$
\widetilde X
=\operatorname{RMSNorm}_{1}(X)
\odot(1+s_{\mathrm{attn}})
+b_{\mathrm{attn}},
$$

$$
X_1=X+g_{\mathrm{attn}}\odot
\operatorname{Attention}(\widetilde X).
$$

\(\odot\) 表示逐元素相乘。scale 与 shift 在注意力之前作用于归一化结果；gate 在注意力之后作用于分支输出，然后加回残差。gate 是学习到的调制量，不应自动当作限定在 0 到 1 的概率。

### 6.2 FFN 分支：scale、shift 并没有消失

第二个分支接收的是**已经完成注意力残差相加的 \(X_1\)**：

$$
\widetilde X_1
=\operatorname{RMSNorm}_{2}(X_1)
\odot(1+s_{\mathrm{ffn}})
+b_{\mathrm{ffn}},
$$

$$
X_{\mathrm{out}}
=X_1+g_{\mathrm{ffn}}\odot
\operatorname{FFN}(\widetilde X_1).
$$

因此准确顺序是：

~~~text
X → RMSNorm₁ → scale、shift → Attention
                                  ↓
                             gate、残差相加
                                  ↓
                                  X₁
                                  ↓
    RMSNorm₂ → scale、shift → FFN → gate、残差相加
                                  ↓
                                X_out
~~~

FFN 前的 RMSNorm 处理 \(X_1\)，不是直接处理尚未乘 gate、尚未加回残差的注意力输出。每个 block 共使用六组调制向量：两个分支各有 shift、scale、gate。完整顺序见 [MiniMaxH3DiTBlock](https://github.com/modelscope/DiffSynth-Studio/blob/e5a44c42776978457b61d51d595a1f6746c3d87a/diffsynth/models/minimax_h3_dit.py)。

### 6.3 SwiGLU：逐 token 加工特征

FFN 对每个 token 独立计算，token 之间的信息交换由注意力完成。这里的 SwiGLU 首先把 5376 维投影为 28672 维，再拆成两个 14336 维分支：

~~~text
输入：     [16624, 5376]
线性投影： [16624, 28672]
拆分：     G、U 各为 [16624, 14336]
~~~

然后计算：

$$
H=\operatorname{SiLU}(G)\odot U,
\qquad
\operatorname{FFN}(\widetilde X_1)=HW_2,
$$

其中 \(W_2\in\mathbb R^{14336\times5376}\)，把特征投影回主干宽度。这里的 \(G\) 是 SwiGLU 内部的门控分支，和上一节位于 FFN 之后的 \(g_{\mathrm{ffn}}\) 不是同一个量。

音频、视频和上下文 token 使用同一套 FFN 权重，但它们进入 FFN 的内容与调制结果不同，所以共享参数不等于相同输出。

## 7. 输出头怎样还原两种更新量？

经过 50 个主干 block，隐藏特征仍为 \([L,5376]\)。输出层再次使用 RMSNorm 与时间调制；这里按时间索引选择 shift、scale，没有块内的残差 gate，也没有另按三种模态展开调制分支。随后分别投影：

~~~text
视频输出投影：5376 → 96
音频输出投影：5376 → 32
~~~

实现按位置提取对应行，并在 Pipeline 的模型函数中去掉关键帧条件的预测，只保留需要生成的部分：

~~~text
视频：[14430, 96] → unpatchify → [1, 24, 37, 30, 52]
音频：[  414, 32] → unpack    → [2, 32, 207]
~~~

这两种输出与各自的 latent 同形状，因而可以用于更新当前状态。此时还没有得到像素或波形。

把原始主干预测统一写成：

$$
(u_v^k,u_a^k)
=F_\theta(z_v^k,z_a^k,c;\tau_v^k,\tau_a^k),
$$

其中 \(c\) 表示固定的上下文与关键帧输入，\(k\) 是外层采样编号。不能因为变量名叫 noise_pred，就不加区别地把它等同于 [DDPM]({{< relref "posts/ddpm-2" >}}) 中的噪声预测：这里采用 Flow Matching 调度，还需要跟踪参数化与时间方向。

## 8. 两条调度怎样把噪声变成音视频？

### 8.1 每轮联合预测，分别更新

一开始，目标 \(z_v^0,z_a^0\) 是高斯噪声。每一轮重新送入当前状态，计算两种预测，再分别交给视频与音频 scheduler：

~~~text
当前 z_v、z_a + 上下文 + 关键帧条件
                    ↓
             主干联合预测
                    ↓
         按实现约定转换预测方向
                    ↓
        视频 scheduler / 音频 scheduler
                    ↓
              下一轮 z_v、z_a
~~~

该 Pipeline 默认视频 flow shift 为 12，音频为 3。**同一次联合预测不要求两种模态具有相同的噪声进度或步长。**

为避免把方向约定藏在公式里，记调度器使用的输出为 \(\widehat u_m^k\)，其中 \(m\in\{v,a\}\)。在本文核对的 model_fn_minimax_h3 中，主干输出还原成 latent 形状后取负号返回：

$$
\widehat u_m^k=-u_m^k.
$$

主干使用的生成进度与 scheduler 的时间值 \(t_m^k\) 则满足：

$$
\tau_m^k=1-\frac{t_m^k}{1000}.
$$

因此最稳妥的采样表达是：

$$
z_m^{k+1}
=\operatorname{Step}_m(z_m^k,\widehat u_m^k,k).
$$

若用“当前状态加步长乘方向”的示意式，步长、方向都必须遵守具体 scheduler 的约定，不能把原始 \(u_m^k\) 与一个任意正数相乘。这里补全的是推理代码中的符号转换，不是另行提出一种采样算法。见 [model_fn_minimax_h3 与联合迭代循环](https://github.com/modelscope/DiffSynth-Studio/blob/e5a44c42776978457b61d51d595a1f6746c3d87a/diffsynth/pipelines/minimax_h3_audio_video.py)。

### 8.2 50 层和 50 次采样不是一回事

50 层指一次主干前向内部依次经过的 block 数；50 次采样指外层循环执行多少轮，后者可以配置。每轮都要重新运行主干，因为待生成 latent 已经改变。

若启用需要正负条件分支的 CFG，一轮还可能包含额外的模型求值。因此不能把“50 次迭代”无条件等同于“恰好 50 次前向”。

全部迭代完成后，视频 VAE 解码 \(z_v\)，音频 VAE 解码 \(z_a\)，才得到画面与立体声音频。

## 9. 首尾帧约束究竟固定了什么？

KeyframeEncoder 先编码关键帧，再构造固定的条件 anchor。默认 imgvid_cond_noise_aug 配置值为 0.999，但实现先将它转换成 pipe.torch_dtype 的张量。记转换后的实际系数为 \(\rho\)，混合形式为：

$$
c_k=\rho\,c_{\mathrm{clean}}+(1-\rho)\,\epsilon_k.
$$

忽略数值舍入时，名义上的两个系数是 0.999 与 0.001；实际扰动取决于 dtype。特别是 BF16 的常规舍入会把 0.999 舍入为 1.0，此时 \(1-\rho=0\)，不能声称默认 BF16 一定加入 0.1% 噪声。这个配置也**不是“99.9% 像素保真率”**，更不是独立测得的质量指标。条件 anchor 在外层采样过程中复用，不像目标音视频 latent 那样被 scheduler 逐轮更新。

“条件固定”也不表示条件 token 在每个 Transformer block 中的隐藏特征不变。它们参与共同的注意力与 FFN，内部特征会被计算和更新；下一轮前向仍从固定的输入条件重新开始。

这条 FL2VA 路径并没有在最终解码结果上把输入图片直接覆盖到第一帧或最后一帧。因此，从该条件机制只能得出“模型持续接收首尾帧约束”，不能推出输出与输入图片必然逐像素相等。由固定 anchor 到“不保证像素一致”的判断是基于代码路径的分析，不是生成效果实测。

## 10. 阅读来源与核对说明

本文整理时去除了对话确认、重复解释和本地工作区路径，保留了尺寸计算、注意力、两套调制、SwiGLU、输出还原与采样主线。数值按以下公开来源核对：

- [MiniMax H3 官方模型卡](https://huggingface.co/MiniMaxAI/MiniMax-H3)：用于确认 FL2VA 输入组合与整体架构定位。模型卡会更新，应与具体实现版本区分。
- [固定版本的音视频 Pipeline](https://github.com/modelscope/DiffSynth-Studio/blob/e5a44c42776978457b61d51d595a1f6746c3d87a/diffsynth/pipelines/minimax_h3_audio_video.py)：查看 NoiseInitializer、PromptEmbedder、KeyframeEncoder、PackedSequenceBuilder 与 model_fn_minimax_h3。
- [同版本 DiT](https://github.com/modelscope/DiffSynth-Studio/blob/e5a44c42776978457b61d51d595a1f6746c3d87a/diffsynth/models/minimax_h3_dit.py)：查看默认维度、Token Refiner、Q/K 归一化、AdaLN、SwiGLU 与输出头。

特别补清了容易被简化叙述掩盖的细节：上下文投影后的 Token Refiner；Q/K 在 RoPE 前的 RMSNorm；主干与调度器之间的时间和符号转换；固定条件输入与可变化的层内特征之间的区别；以及条件噪声系数受 dtype 舍入影响的边界。

本文没有运行模型权重或进行生成质量对比，因此不报告吞吐、显存、对齐精度或保真率。结构上的信息通路与实际效果之间，仍需要实验验证。

## 11. 小结

FL2VA 的核心过程可以沿着维度一路追踪：首尾帧通过上下文与视觉 latent 两条路径提供条件；音频、视频和条件 token 投影到同一宽度后进入共享主干；联合注意力交换信息，AdaLN 根据模态与生成进度调制计算，SwiGLU 逐 token 加工特征；最后两个输出头还原出与各自 latent 同形状的预测，并交给两条调度迭代更新。

贯穿全文的例子中，视频有 14430 个 token，首尾帧有 780 个，音频有 414 个。只有上下文的 1000 个 token 是教学假设。记住这些数量的来源，比只记住总长度 16624 更有用。

理解这套结构时，最需要分清的是：**token 数量与特征宽度、时空位置与生成进度、层数与采样次数，以及条件约束与像素级硬保证。** 这四组区别把从输入到输出的计算过程连在了一起。

