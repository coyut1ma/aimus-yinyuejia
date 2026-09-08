# 多分枝 ACE-Step 第一版设计方案及执行计划

## 1. 方案定位

本版本采用最小改动原则：只在 ACE-Step 的第一个 DiT block 附近增加多轨处理和一次跨轨信息交流，后续网络、输出头、VAE、文本/歌词条件和 diffusion scheduler 尽量保持原样。

本版本选择方案 A：四条轨道共享同一套文本/歌词条件。也就是说，所有 stem 使用同一个 `caption`、`lyrics` 和原 ACE-Step 条件；轨道之间的区别主要通过输入顺序和简单的 `stem embedding` 表示。

本版本的目标不是立即完成完整的联合多轨生成，而是验证一个核心假设：

> 在 ACE-Step 的早期特征处理后，让四条轨道交换一次 latent 特征，是否能改善目标轨道编辑时的节拍、和声和整体一致性。

本版本可以称为：

```text
Multi-Branch ACE-Step
或
Stem-aware ACE-Step with Early Cross-Stem Attention
```

它不是完整的 SlowFast 网络，也不是完整的联合多轨扩散模型。

---

## 2. 当前基线与改造位置

现有 `music_edit_demo` 已经能完成：

```text
自然语言指令
    ↓
EditPlan
    ↓
ACE-Step HTTP Generator
    ↓
编辑目标 stem
    ↓
crossfade、混音和候选排序
```

当前 demo 的 HTTP 请求已经包含全部输入 stem、编辑区域、目标轨道、prompt、preserve 和 seeds。主要改动应放在远端 ACE-Step 服务内部，而不是 parser、compositor 或任务存储层。

当前 ACE-Step DiT 的基本结构为：

```text
输入 latent
    ↓
多个 DiT block
    ├─ Self-Attention
    ├─ Text/Lyric Cross-Attention
    └─ MLP
    ↓
输出投影
    ↓
噪声预测
```

本方案只改造第一个 block 前后：

```text
多轨输入 latent
    ↓
四个共享参数的 Block 1 分支
    ↓
一次 Cross-Stem Attention
    ↓
选择目标轨道
    ↓
原始 ACE-Step Block 2...N
    ↓
原始 ACE-Step 输出头
```

---

## 3. 整体结构

```text
vocals latent ──┐
drums latent  ──┤
bass latent   ──┼─ 共享 ACE-Step Block 1
other latent  ──┘
                         │
                         ▼
                  Cross-Stem Attention
                         │
               保留四条轨道的独立特征
                         │
                   选择目标轨道
                         │
                         ▼
              原 ACE-Step Block 2...N
                         │
                         ▼
                原 ACE-Step 输出投影
                         │
                         ▼
                  目标轨道噪声预测
```

以编辑 vocals 为例：

```text
四条轨道共同输入
    ↓
四条轨道分别通过共享 Block 1
    ↓
Cross-Stem Attention
    ↓
取出 vocals 融合后的特征
    ↓
进入原 ACE-Step Block 2...N
    ↓
输出 vocals
```

以编辑 vocals 和 bass 为例，可以对两个目标轨道分别执行同一联合前端，或在后续将目标轨道并入 batch 维并行执行。第一版优先保证接口简单和原模型兼容。

---

## 4. 多轨输入表示

### 4.1 Latent 组织

四条输入轨道先分别经过现有 ACE-Step 音频编码流程，得到：

```text
z_vocals
z_drums
z_bass
z_other
```

每条 latent 的形状为：

```text
[B, T, C]
```

堆叠后得到：

```text
Z = [B, 4, T, C]
```

轨道顺序固定为：

```text
0 → vocals
1 → drums
2 → bass
3 → other
```

其中 `T` 是 latent 特征序列的时间帧数，不是音频秒数；四条轨道的实际时长仍然相同。

### 4.2 统一文本/歌词条件

本版本使用方案 A：四条轨道共享同一套：

```text
caption
lyrics
parsed metadata
tempo/key 等已有条件
```

不为每个 stem 单独构造局部文本 prompt。这样可以避免同时改动 ACE-Step 的文本条件接口，也能先单独验证音频 latent 之间的跨轨交流是否有效。

### 4.3 简单 Stem Embedding

使用四个可学习向量表示轨道身份：

```text
e_vocals
e_drums
e_bass
e_other
```

将对应向量加到每条轨道的输入特征上：

\[
H_s^0=Z_s+E_s^{stem}
\]

实现上可以使用：

```python
self.stem_embedding = nn.Embedding(4, hidden_size)
stem_ids = torch.arange(4, device=z.device)
stem_emb = self.stem_embedding(stem_ids)       # [4, D]
h = h + stem_emb[None, :, None, :]              # [B, 4, T, D]
```

Stem embedding 只表示“这是哪一条轨道”，不表示该轨道当前具体演奏了什么。具体音符、节奏和音色仍由 audio latent 表示。

---

## 5. 四分支 Block 1

### 5.1 共享参数，而不是复制四套模型

第一版不为 vocals、drums、bass、other 各复制一套完整 Block 1，而是让四条分支共享同一个 ACE-Step Block 1：

```python
h_vocals = block1(h_vocals, shared_condition)
h_drums  = block1(h_drums, shared_condition)
h_bass   = block1(h_bass, shared_condition)
h_other  = block1(h_other, shared_condition)
```

实际实现可以把输入从 `[B, 4, T, D]` reshape 为 `[B*4, T, D]`，用同一个 Block 1 一次并行计算，再 reshape 回 `[B, 4, T, D]`。

### 5.2 文本条件处理

四条分支使用相同的 `encoder_hidden_states` 和 `encoder_attention_mask`。文本条件可以在 batch 维复制四次：

```text
同一首歌的 Global caption / lyrics
        ↓
复制给 vocals、drums、bass、other
```

本版不加入 per-stem local caption，以保持 ACE-Step 原条件路径不变。

### 5.3 第一版是否修改原 Block 1

推荐先加载原 Block 1 权重，并保持其主体参数不变。新增内容只有：

```text
多轨输入 reshape
Stem Embedding
Cross-Stem Attention
```

如果原 Block 1 直接接收的输入通道与多轨组织方式不一致，只增加一个轻量输入适配层，不修改后续 Block 2...N 的输入维度。

---

## 6. Cross-Stem Attention

### 6.1 作用

Cross-Stem Attention 放在四条轨道通过 Block 1 之后。它让每条轨道读取其他轨道的中间特征：

```text
生成 vocals：参考 drums、bass、other
生成 drums：参考 vocals、bass、other
生成 bass：参考 vocals、drums、other
生成 other：参考 vocals、drums、bass
```

它不是把音频直接混在一起，而是在 latent 特征层进行信息交流。

### 6.2 第一版计算方式

第一版保留轨道维度：

```text
H：[B, 4, T, D]
```

对每条轨道 (s)，当前轨道作为 Query，其他轨道作为 Key/Value：

\[
H'_s=H_s+\mathrm{CrossStemAttn}(Q=H_s,K/V=H_{j\neq s})
\]

推荐先使用同一时间位置的其他轨道特征：

```text
bass(t) 读取 drums(t)、vocals(t)、other(t)
vocals(t) 读取 drums(t)、bass(t)、other(t)
```

后续如果显存允许，再改成附近时间窗口。

### 6.3 Cross-Stem 的输出

Cross-Stem 后仍然保留四条独立轨道：

```text
H'：[B, 4, T, D]
```

第一版默认选择目标轨道：

```python
target_hidden = h_cross[:, target_stem_id, :, :]  # [B, T, D]
```

这样从此处开始，输入形状恢复为原 ACE-Step 的单轨格式。

---

## 7. 后续原 ACE-Step 主干

Cross-Stem 后选择目标轨道特征：

```text
[B, 4, T, D]
        ↓ 选择 target_stem
[B, T, D]
        ↓
原 ACE-Step Block 2
        ↓
原 ACE-Step Block 3
        ↓
...
        ↓
原 ACE-Step Block N
        ↓
原 ACE-Step 输出投影
```

以下部分第一版全部保持不变：

- 后续 DiT block；
- 文本/歌词 cross-attention 结构；
- timestep embedding 和 AdaLN；
- 原输出投影层；
- VAE 解码器；
- diffusion scheduler；
- 外部输出格式。

因此，原模型仍然输出一条目标轨道的噪声预测：

```text
[B, T, C]
```

不需要新增多轨输出头。

---

## 8. 局部编辑和保持区域

虽然第一版不修改 ACE-Step 的输出结构，但仍要在服务端保留现有的编辑/保持逻辑。

对于任务：

```text
编辑 vocals 的 42–58 秒
drums、bass、other 作为参考
其他时间保持
```

模型内部：

```text
四条轨道全部输入 Block 1 和 Cross-Stem Attention
只有 vocals 作为目标轨道进入后续生成
```

在 diffusion 更新后：

```text
编辑区域：使用 ACE-Step 生成结果
编辑区域外：使用原始参考 latent
非目标轨道：使用原始参考音频/latent
```

可用 mask 表示：

```text
M=0：允许生成
M=1：必须保持
```

保持区域的替换形式为：

\[
z^{n-1}=(1-M)\odot z_{generated}^{n-1}+M\odot z_{reference}^{n-1}
\]

第一版可以继续复用现有 repaint/clamp 实现，不要求把 mask 改造成新的神经网络模块。

---

## 9. 一次请求的推理流程

### 9.1 单轨目标编辑

```text
1. 从 payload 读取四条 stem
2. 对四条 stem 做统一长度、采样率和编码处理
3. 得到 Z：[B,4,T,C]
4. 加入 stem embedding
5. 四条轨道共享 Block 1
6. 执行一次 Cross-Stem Attention
7. 选择目标轨道
8. 进入原 ACE-Step Block 2...N
9. 使用原输出头得到目标轨道噪声
10. scheduler 更新目标轨道 latent
11. 对非编辑区域和非目标轨道执行 reference clamp
12. 重复 diffusion steps
13. 解码目标轨道并返回原有输出格式
```

### 9.2 多目标编辑

对于 vocals 和 bass 同时编辑，第一版可以先采用两次目标选择：

```text
同一组四轨输入
    ↓
前端 Block 1 + Cross-Stem Attention
    ↓
分别选择 vocals 和 bass
    ↓
使用共享的原 ACE-Step 后续主干
```

如果第一版需要严格同步生成，可以将目标轨道在后续主干中合并到 batch 维并行处理：

```text
[B, 2, T, D] → [B*2, T, D]
```

但这一步不是结构验证的必要条件。第一版应先确认 Cross-Stem 前端能够改善单轨目标编辑。

---

## 10. 训练策略

### 10.1 第一阶段：代码和形状验证

目标：验证结构能运行，不评价音质。

做法：

- 加载原 ACE-Step checkpoint；
- 新增 Stem Embedding 和 Cross-Stem Attention；
- 新增模块 residual gate 初始化为 0；
- 冻结原 ACE-Step 所有参数；
- 使用随机或零初始化 Cross-Stem 参数；
- 检查张量形状、显存、推理时延和输出一致性。

此时输出质量没有参考意义，因为新增 Cross-Stem 模块尚未训练。

### 10.2 第二阶段：只训练新增模块

冻结：

```text
原 ACE-Step Block 1
原 ACE-Step Block 2...N
原输出头
文本编码器
VAE
```

训练：

```text
Stem Embedding
Cross-Stem Attention
输入/输出适配层（如果需要）
```

训练样本来自 MUSDB18 四轨音频，随机生成：

- 单轨局部编辑；
- 双轨目标编辑；
- 非目标轨道作为参考；
- 编辑区域和非编辑区域 mask；
- 不同随机种子。

基础损失先使用目标轨道的 diffusion loss：

\[
L_{diff}=\|\epsilon_{target}-\hat\epsilon_{target}\|_2^2
\]

第一版不必立即加入复杂的和声或节拍辅助损失，先观察 Cross-Stem 是否带来收益。

### 10.3 第三阶段：小范围解冻

如果第二阶段有效，再低学习率解冻：

- ACE-Step Block 1 的部分参数；
- 中间一到两个 DiT block；
- 必要的输入投影。

保持后续 block 和输出头尽量冻结，以减少对原模型音质的破坏。

---

## 11. Demo/API 执行改造

### 11.1 保持现有任务层接口

`music_edit_demo` 的 `EditPlan`、候选 seed、crossfade、混音和评估流程可以继续使用。

现有 HTTP payload 已经包含：

```text
stems
region
targets
prompt
preserve
seeds
```

第一版只建议增加一个开关：

```json
{
  "joint_frontend": true,
  "target_stems": ["vocals"]
}
```

如果不传该字段，服务端继续走旧的单轨路径；如果传 `true`，走新的多分枝前端。

### 11.2 服务端内部新增接口

建议将新增逻辑封装成独立模块：

```text
MultiStemFrontend
    ├─ prepare_stem_latents()
    ├─ add_stem_embedding()
    ├─ run_shared_block1()
    ├─ cross_stem_attention()
    └─ select_target_stem()
```

原 ACE-Step 后续生成函数只接收：

```text
target_hidden：[B,T,D]
```

这样可以尽量减少对现有生成代码的侵入。

---

## 12. 执行步骤

### 步骤 1：确认 ACE-Step Block 1 的真实接口

检查并记录：

- Block 1 的输入 shape；
- hidden size；
- `encoder_hidden_states` shape；
- timestep embedding shape；
- attention mask 要求；
- 输出是否包含 hidden states 和 attention weights。

### 步骤 2：实现多轨 latent 包装

新增一个包装函数，将四条轨道组织成：

```text
[B,4,T,C]
```

并确保四条轨道的采样率、长度、切片位置和 latent 时间轴一致。

### 步骤 3：加入 Stem Embedding

实现四个可学习 embedding，并确认：

```text
[B,4,T,D] + [1,4,1,D]
```

可以正常广播。

### 步骤 4：实现共享 Block 1

将 `[B,4,T,D]` reshape 为 `[B*4,T,D]`，复用原 ACE-Step Block 1，再恢复轨道维度。

### 步骤 5：实现 Cross-Stem Attention

第一版只做同时间位置的跨轨 attention，并加入 residual gate：

```text
h_cross = h + alpha * cross_stem(h)
```

其中 `alpha` 初始为 0。

### 步骤 6：接回原始后续主干

选择目标轨道后，确认形状恢复为：

```text
[B,T,D]
```

再调用原 ACE-Step Block 2...N 和原输出头。

### 步骤 7：做无训练 smoke test

检查：

- 单轨输入路径仍然可用；
- 多轨前端输出 shape 正确；
- 输出音频时长和采样率不变；
- 新旧路径在 `alpha=0` 时结果接近；
- 目标轨道选择正确；
- 非目标轨道没有被错误输出或覆盖。

### 步骤 8：构造 MUSDB18 训练样本

使用四条 stem 的同一时间片段，随机生成目标轨道和编辑 mask。训练/验证/测试按歌曲隔离。

### 步骤 9：只训练新增模块

先训练 Stem Embedding 和 Cross-Stem Attention，保存独立 adapter checkpoint，不覆盖原 ACE-Step checkpoint。

### 步骤 10：对比评测

至少比较：

```text
旧版逐轨 ACE-Step
新版多轨前端 + ACE-Step
```

重点观察：

- 跨轨 onset/beat 对齐；
- 和声/HPCP 一致性；
- 编辑区域音质；
- 非编辑区域保持率；
- 整体混音 MOS；
- 推理时延和显存。

---

## 13. 预期收益与限制

### 预期收益

- 改动范围小，原 ACE-Step 主干高度复用；
- 可以让目标轨道在早期生成阶段参考其他轨道；
- 不改变现有文本/歌词条件接口；
- 不改变原输出头和外部音频格式；
- 适合先验证多轨协同是否有效。

### 当前限制

- 只在第一个 block 后交流一次，后续轨道之间不再持续交互；
- 默认还是目标轨道输出，不是完整的四轨联合输出；
- 共享文本条件无法表达非常细的轨道专属指令；
- 新增模块需要训练，未训练时不能期待质量提升；
- 它属于多轨条件增强，不是完整联合概率建模。

---

## 14. 后续扩展顺序

如果第一版有效，建议按以下顺序扩展：

```text
第一版：Block 1 后一次 Cross-Stem
    ↓
第二版：在中间 block 再增加一次 Cross-Stem
    ↓
第三版：支持 Slow/Fast 多时间尺度
    ↓
第四版：多轨同步输出和联合 diffusion
    ↓
第五版：加入 beat、harmony、mix consistency loss
```

不要在第一版同时引入完整 SlowFast、独立 per-stem prompt、复杂图结构和四套独立 ACE-Step 主干。

---

## 15. 最终版本总结

第一版最终结构为：

```text
四条 stem
    ↓
分别编码为 latent
    ↓
加入简单 stem embedding
    ↓
共享 ACE-Step Block 1
    ↓
一次 Cross-Stem Attention
    ↓
选择目标轨道
    ↓
完全复用 ACE-Step Block 2...N
    ↓
完全复用 ACE-Step 输出头
    ↓
原有 diffusion、VAE 和输出流程
```

一句话概括：

> 在 ACE-Step 第一个 block 后增加一个轻量的多轨信息交流前端，让目标轨道在进入原始 ACE-Step 主干之前先读取其他轨道的特征；除此之外尽量保持原模型不变。
