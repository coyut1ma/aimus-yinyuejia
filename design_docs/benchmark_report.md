# MUSDB18 音乐可控编辑项目 Benchmark 报告

**版本**：v1.0（建议作为评测协议草案）  
**数据集**：MUSDB18 及项目自建编辑任务标注  
**参考基线**：ACE-Step 1.5（arXiv:2602.00744）、DiffRhythm 2（arXiv:2510.22950）

## 1. 执行摘要

本项目的目标不是单纯生成一首“听起来不错”的歌曲，而是在指定片段、指定轨道和自然语言意图下进行可控编辑，同时保持节拍、段落结构、和声、音色及多轨同步。因此 benchmark 必须同时覆盖：

1. **编辑是否正确**：编辑范围和文本意图是否被执行；
2. **衔接是否自然**：编辑片段与原曲边界是否出现断裂、爆音或节拍漂移；
3. **音乐结构是否保持**：节拍、段落、和声是否满足约束；
4. **多轨是否协同**：各 stem 是否对齐、音量是否平衡；
5. **听感是否达标**：专家对质量、自然度、风格/情绪匹配度的评分；
6. **效率与稳定性**：时延、显存/算力、失败率和多次采样方差。

图片中的阈值（MOS≥4.5、匹配度≥4.0、段落误差≤2 s、节拍误差≤20 ms、和声匹配度≥90%、多轨对齐≤20 ms、音量偏差≤3 dB、语义匹配率≥85%）应视为**业务验收目标**，而不是声学领域普遍统一的阈值。评测算法应尽量采用现有标准和公开工具，再冻结项目化的输入、统计口径和阈值。

## 2. 任务定义与数据协议

### 2.1 单条编辑任务格式

每条任务包含：

```text
song_id
stem_files: vocals / drums / bass / other
input_mix
edit_type
edit_start, edit_end
instruction                 # 自然语言指令
preserve_constraints         # 例如 preserve_tempo=true
beat_grid, downbeat_times
section_boundaries           # verse/chorus 等
chord_labels
lyrics_word_or_syllable_times（若有歌词任务）
loudness_reference
```

编辑类型建议至少包括：局部重生成、歌词/句子级编辑、旋律编辑、和声编辑、风格/情绪编辑、单轨编辑、多轨协同编辑。每个样本同时标注哪些属性**必须保持**、哪些属性**必须改变**；例如“变得更欢快”不能同时把情绪变化当作音色保持约束。

### 2.2 数据划分

- `dev`：开发和调参；
- `test`：最终公开测试；
- `hidden-test`（推荐）：时间外或歌曲/歌手隔离测试，防止排行榜过拟合。

必须按歌曲或歌手划分，不能让同一首歌的相邻片段跨越 train/test。MUSDB18 原始文件不天然提供所有段落、和弦、歌词和编辑意图标签，建议先用自动工具产生初始标注，再进行人工抽检和版本化发布。

### 2.3 统一运行协议

固定采样率、声道、响度预处理、随机种子、采样次数、最大推理时间、硬件、显存限制、外部数据/工具许可、失败和超时处理方式。严禁对每个系统单独做会影响指标的响度归一化后再比较音量偏差。

## 3. 指标总览与推荐口径

| 图片要求 | 推荐主指标 | 辅助指标 | 推荐验收口径 |
|---|---|---|---|
| 衔接自然 | 带参考 MOS/MUSHRA | 边界响度跳变、频谱距离 | MOS 均值≥4.5，并报告 95% CI |
| 曲风/情绪/乐器音色保持一致 | 专家匹配度评分 | CLAP/MuLan、属性分类 F1 | 专家均值≥4.0；自动指标只作辅助 |
| 主歌/副歌误差≤2 s | 结构边界时间误差 | Boundary F1、segment overlap | verse/chorus 边界 P95≤2 s |
| 节拍误差≤20 ms | beat-to-beat 时间误差 | Beat F1、Cemgil/continuity | 匹配节拍 P95≤20 ms |
| 和声匹配度≥90% | Weighted Chord Symbol Recall | Root accuracy、HPCP 相似度 | 编辑区 WCSR≥90% |
| 多轨对齐≤20 ms | stem onset/beat P95 | 互相关 lag、跨轨 beat 偏差 | 各 stem P95≤20 ms |
| 音量偏差≤3 dB | 短时 LUFS 偏差 | RMS dB、边界 loudness jump | 各 stem LUFS 偏差 P95≤3 dB |
| 多轨编辑后专家音质≥4.0 | 多轨整体 MOS | 每轨 MOS、伪影率 | MOS 均值≥4.0 |
| 语义匹配率≥85% | 结构化任务成功率 | CLAP/MuLan、属性分类 | 必要条件全部满足的样本率≥85% |

## 4. 备选方法调研（非最终执行规范）

本节保留方法选择过程，便于解释为何没有采用其他常见指标。**正式运行 benchmark 时，每项只使用第 11 节选定的一种主方法；本节中的其他方法不得混入主分数。**

### 4.1 音频质量 MOS 与衔接自然度

#### 方法 A：ITU-T P.800 / P.808 的 ACR MOS

评测者对单个音频按 1–5 分评分：

$$
MOS=\frac{1}{N}\sum_{j=1}^{N}r_j
$$

**公式来源**：算术平均定义，采用 ITU-T P.800 的 Mean Opinion Score 计算口径；P.800 规定主观评分流程，但该式本身是均值的统计定义，不是某个生成模型论文提出的公式。

可分别评价音频质量、编辑边界自然度、伪影严重程度。P.800 是经典主观质量规范，P.808 更适合远程/众包评测。

**优点**：实施简单、结果容易解释，适合项目验收。  
**缺点**：无原曲参考时，评测者容易把“音乐偏好”混入质量分；不同评测者尺度差异较大。

#### 方法 B：ITU-R BS.1534 MUSHRA

同一试次同时播放原曲参考、多个系统结果和隐藏参考/低质量锚点，评测者在 0–100 分连续量表上评价相对质量。适合“编辑后是否接近参考、哪个系统更好”的场景。

**优点**：区分度高，有参考且能同时比较多个系统；适合高质量音乐。  
**缺点**：实验设计和播放控制更复杂，样本数与听评成本更高。

#### 方法 C：成对偏好（pairwise preference）+ Bradley–Terry/分组统计

让评测者在 A/B 中选择更自然或更高质量的编辑结果，再估计系统胜率或 Bradley–Terry 分数。

**优点**：对细微差异和模型排序较稳健，不要求评测者理解绝对分数。  
**缺点**：不能直接得到 MOS≥4.5 这种绝对验收分数。

**推荐**：主验收采用 P.800/P.808 风格 1–5 分 MOS；开发阶段或模型对比增加 MUSHRA/成对偏好。至少报告均值、标准差、95% 置信区间和评测者一致性（Krippendorff’s α 或 Fleiss’ κ）。MOS≥4.5 时建议同时要求 95% CI 下界不低于 4.5，或明确仅以点估计判定。

### 4.2 曲风、情绪、乐器与音色保持一致

#### 方法 A：音频-文本嵌入相似度（CLAP、MuLan、MusicCLIP）

将编辑指令和输出音频编码后计算 cosine similarity，或比较原曲与编辑结果的音频 embedding。

**适用**：大规模自动筛选、风格/情绪匹配的趋势分析。  
**局限**：分数受模型训练域影响，不能等价于专家判断；不建议直接把 cosine=0.8 映射成“80% 匹配”。

#### 方法 B：属性分类器的准确率/F1

为每个样本提供风格、情绪、乐器等标签，用预训练分类器或人工验证的分类器预测输出，报告 Accuracy、Macro-F1 或 multi-label F1。

**优点**：可以逐项回答“是否有鼓/钢琴/摇滚/欢快”等属性。  
**局限**：分类器本身可能有偏差，细粒度音色和生成新风格不易覆盖。

#### 方法 C：专家匹配度评分（1–5）

盲化模型名称，分别评价“曲风匹配、情绪匹配、乐器/音色一致性”。

**推荐**：专家评分作为最终验收（均值≥4.0），CLAP/MuLan 和分类器作为可复现的辅助指标；三类评分不得合并成一个无法解释的总分。若要求“保持原音色”，应在非编辑区和编辑区分别评分。

### 4.3 主歌/副歌结构边界误差

#### 方法 A：边界时间误差

设原曲边界为 (t_s)，编辑结果边界为 \(\hat t_s\)：

$$
E_s=|\hat t_s-t_s|
$$

**公式来源**：本项目定义，参考音乐结构分析中的 boundary hit/距离评测（可用 `mir_eval` 的结构评测思路实现）；它不是 ACE-Step 1.5 或 DiffRhythm 2 论文中的专有公式。

对 verse、chorus 起点和终点分别统计 MAE、median、P95。

**优点**：直接对应图片中的“≤2 s”。  
**局限**：依赖可靠的段落标注；一个边界错位可能影响多个片段。

#### 方法 B：Boundary Precision/Recall/F-measure

采用音乐结构分析常用的容差匹配：在 ±δ 秒内视为命中，报告 boundary P/R/F1。可参考 `mir_eval` 的结构评测思路。

**优点**：适合边界数目不同或存在漏检的结果。  
**局限**：需要先固定容差 δ，且 F1 不能直接表示偏移量大小。

#### 方法 C：段落重叠（IoU/segment overlap）

以时间区间计算预测段落与参考段落的 Intersection-over-Union。

**优点**：同时反映起点和终点。  
**局限**：对短片段和边界小偏差较敏感。

**推荐**：主指标用 verse/chorus 边界 P95 时间误差，辅助报告 boundary F1 和 segment overlap。将图片阈值固化为“所有必要边界 P95≤2 s”，不要使用未说明的最大值或平均值。

> 如果“主副歌误差”实际指歌词/人声时间误差，则应另设词/音节级对齐指标（见 4.8），不能和结构边界混用。

### 4.4 节拍误差

#### 方法 A：beat-to-beat 时间误差

使用同一 beat tracker 提取原曲节拍 (b_i) 和编辑结果节拍 \(\hat b_i\)，经允许的全局偏移/线性时间映射后做最近邻匹配：

$$
e_i=|\hat b_i-b_i|\quad (ms)
$$

**公式来源**：本项目定义，基于 MIREX/`mir_eval.beat` 的参考节拍与估计节拍匹配框架增加逐拍绝对时间偏差；MIREX/`mir_eval` 同时提供 Precision、Recall、F-measure、Cemgil 等标准节拍指标。

报告 MAE、median、P95、漏检率和误检率。

**优点**：直接对应“≤20 ms”，适合保持速度的编辑。  
**局限**：依赖 beat tracker；若任务允许改 BPM，则不能把原曲 beat 当作唯一真值。

#### 方法 B：`mir_eval.beat` / MIREX Beat F-measure

报告 Beat Precision、Recall、F-measure、Cemgil Accuracy、continuity 等。

**优点**：社区使用广泛，便于与音乐信息检索工作比较。  
**局限**：F1 高不代表逐拍时间偏差一定小，不能单独替代 20 ms 误差。

#### 方法 C：downbeat/beat-grid 偏差

分别计算小节首拍和一般拍的偏差，或拟合 BPM 后比较 beat grid 的相位与漂移。

**推荐**：主指标采用 beat-to-beat P95≤20 ms；辅助报告 Beat F1 和 downbeat accuracy。对于允许变速的任务，改为比较输出与目标 BPM/目标 beat grid 的误差。

### 4.5 和声匹配度

#### 方法 A：`mir_eval.chord` 的 Weighted Chord Symbol Recall（WCSR）

将原曲和编辑结果表示为带时间的和弦标签，按持续时间加权：

$$
WCSR=\frac{\text{和弦标签一致的时间长度}}{\text{评测总时间}}\times100\%
$$

**公式来源**：`mir_eval.chord`/MIREX chord evaluation 的 Weighted Chord Symbol Recall（WCSR）定义；实现时应采用该工具的和弦等价类、时间加权和未标注区间处理规则。

**优点**：是和弦识别/比较中最常用的参考指标之一，和业务的“90%”容易对齐。  
**局限**：依赖和弦标注质量；等价和弦、转位和装饰音需要统一标签规范。

#### 方法 B：Root Accuracy / Chord Symbol Recall

只比较根音，或比较完整和弦符号。根音准确率对复杂和弦更宽松；完整符号更严格。

**优点**：可分别诊断“根音保持”和“完整和弦保持”。  
**局限**：单独使用 Root Accuracy 可能掩盖三和弦/七和弦质量差异。

#### 方法 C：chroma/HPCP cosine similarity

对齐时间帧后比较 chroma 或 HPCP 向量余弦相似度。

**优点**：没有完整和弦标签时仍可运行，对复杂和声较灵活。  
**局限**：相似度不是和弦正确率，不能直接把 0.9 解释为 90%。

**推荐**：有可靠和弦标签时以 WCSR≥90% 为主，Root Accuracy 和 HPCP 作为诊断；若任务明确要求改变和声，则建立目标和弦标注，比较输出与目标而非原曲。

### 4.6 多轨对齐误差

本节中的“对齐”需要区分：跨轨协同同步，以及编辑结果相对于原曲的时间保持。正式主指标采用第 11.6 节的跨轨相对事件误差；原曲—编辑误差只作为辅助保持性指标。

#### 方法 A：stem onset 对齐

对编辑后的 vocals、drums、bass、other 各轨提取对应 onset/瞬态，比较轨道之间的相对事件时间与任务规定的目标关系。若任务要求保持原曲协同关系，目标关系可由原曲标注提供；不能把“编辑结果 vs 原曲同一轨”直接称为跨轨协同主指标。

$$
E^{cross}_{k,l}=P95_i\left|\left(\hat t_{i,k}-\hat t_{i,l}\right)-d^{target}_{i,k,l}\right|
$$

**公式来源**：本项目定义，基于 onset detection 与跨轨事件匹配；它不是一个单独的统一行业公式。P95 是本项目为对应“≤20 ms”验收要求选择的统计汇总方式。

**优点**：能发现局部接拍、鼓点错位和人声进入点漂移。  
**局限**：弱音轨或持续音的 onset 不稳定。

#### 方法 B：互相关估计整体时延

对原轨和编辑轨的包络或短时能量做 cross-correlation，取最大相关对应的 lag。

**优点**：实现简单，适合检查整体平移。  
**局限**：只能发现整体延迟，无法发现随时间累积的漂移或局部错位。

#### 方法 C：跨轨 beat/downbeat 偏差

比较各 stem 的 beat/onset 相对于公共 beat grid 的偏差。

**推荐**：采用“跨轨相对 onset P95 为主、互相关 lag 为辅、beat-grid 偏差作诊断”的组合；验收写为各必需 stem 对的相对同步 P95≤20 ms，同时报告最大 lag 和失败轨道比例。

### 4.7 音量偏差

#### 方法 A：BS.1770 / EBU R128 短时 LUFS

在固定窗口 (w) 内计算原曲和编辑结果的响度：

$$
\Delta L_w=|L_{edited}(w)-L_{original}(w)|
$$

**公式来源**：本项目定义的“响度差”；其中 $L$ 应按 ITU-R BS.1770（及 EBU R128）计算 LUFS，BS.1770 规定响度测量算法，但不规定本项目这种编辑前后差值的验收形式。

**优点**：符合广播/流媒体响度规范，比波形振幅更接近听感。  
**局限**：窗口和 gating 对短片段影响较大。

#### 方法 B：RMS/均方根电平 dB

$$
L_{RMS}=20\log_{10}(RMS)
$$

**公式来源**：离散信号 RMS 电平的标准信号处理定义；本项目仅将其作为诊断指标，不作为最终响度验收标准。最终响度优先采用 BS.1770/EBU R128 LUFS。

**优点**：易实现，便于诊断单个 stem。  
**局限**：对频率加权和听感不如 LUFS，不能作为唯一验收指标。

#### 方法 C：边界 loudness jump / Loudness Range（LRA）

评价编辑边界前后响度跳变，或比较整段的 LRA。

**优点**：专门捕捉拼接处突然变大/变小。  
**局限**：不是编辑区内每个窗口的音量保持度。

**推荐**：以短时 LUFS 偏差为主，RMS 作调试，边界 jump 作自然衔接诊断。固定 400 ms 或 1 s 窗口、步长和静音阈值；图片阈值定义为各 stem 偏差 P95≤3 dB。

### 4.8 歌词—人声对齐（若任务包含歌词编辑）

#### 方法 A：强制对齐后的词/音节 onset 误差

利用歌词和音频强制对齐得到词或音节时间戳 (t_i,\hat t_i)，报告 MAE、P95 和容差内命中率。

#### 方法 B：ASR 时间戳 + 容差 F1

用歌唱/语音识别模型得到词级时间戳，在 ±δ ms 内匹配参考词。

**优点**：流程自动化。  
**局限**：歌唱 ASR 本身可能产生较大时间戳误差，需要人工抽检。

#### 方法 C：DTW/音素或音节路径偏差

用动态时间规整寻找参考与编辑人声的最优路径，再统计局部时间偏差。

**优点**：对速度轻微变化更鲁棒。  
**局限**：可能把错误演唱“对齐”得很好，必须同时检查歌词识别正确率。

**推荐**：词/音节级强制对齐 onset MAE/P95 为主，ASR/DTW 作辅助；将“歌词内容正确”和“时间对齐正确”分开报告。

### 4.9 自然语言意图理解/语义匹配率

#### 方法 A：结构化人工 checklist

将每条指令拆成必要属性。例如“将副歌改成更欢快的摇滚风格，保留原歌词和人声音色”拆成：编辑范围、风格、情绪、歌词保持、音色保持。只有必要条件全部满足，样本才算成功：

$$
SuccessRate=\frac{\#\text{完全满足指令的样本}}{N}\times100\%
$$

**公式来源**：本项目定义的结构化任务成功率；它借鉴分类任务的准确率形式，但不是某篇基线论文规定的音乐编辑专用公式。必要条件、样本分母和标注者一致性必须在任务协议中冻结。

**优点**：最接近业务定义，可解释、可审计。  
**缺点**：需要专家标注，成本较高。

#### 方法 B：CLAP/MuLan 文本—音频相似度

对指令中的风格、情绪、乐器描述分别计算相似度。

**优点**：可批量运行。  
**局限**：无法可靠判断“只编辑副歌”“保留歌词”等精确约束。

#### 方法 C：属性分类器或 LLM-as-judge

对风格、情绪、乐器、歌词保持、编辑区域等属性进行分类/判定。

**优点**：可以得到逐属性 Precision、Recall、F1。  
**局限**：分类器或评审模型需要校准，不能无条件当作 ground truth。

**推荐**：以人工 checklist 的完全成功率作为 85% 的正式指标；CLAP/MuLan 和属性分类器只作自动辅助。至少由 3 名评测者盲评，并报告 Fleiss’ κ 或 Krippendorff’s α。

## 5. 人工评测设计

### 5.1 评分维度

不要把所有问题压成一个 MOS。至少分为：

1. 音频质量/伪影；
2. 边界过渡自然度；
3. 风格匹配；
4. 情绪匹配；
5. 乐器/音色一致性；
6. 和声合理性；
7. 多轨整体自然度；
8. 指令完成度。

### 5.2 评测者与统计

- 专业音乐制作/音频背景评测者：建议 ≥5 人；
- 每个样本至少 3 次独立评分；
- 模型名称隐藏、顺序随机；
- 统一播放设备和响度；
- 报告均值、标准差、95% CI；
- 报告评测者一致性；
- 公开评分表、锚点样本和异常评分剔除规则。

图片中的“MOS≥4.5”和“专家匹配度≥4.0”应分别使用不同问题，避免质量高但不符合指令的样本被错误判为通过。

## 6. 基线比较方案

### 6.1 基线系统

至少包括：

1. 项目方法；
2. ACE-Step 1.5；
3. DiffRhythm 2；
4. 当前已有系统/旧版本；
5. 简单基线：原片段复制、硬拼接、交叉淡化或传统 stem 编辑。

ACE-Step 1.5 和 DiffRhythm 2 的论文重点是歌曲生成及一定的可控/编辑能力，不应假定它们支持所有句子级、多轨局部编辑。对不支持的任务标为 `N/A`，不能用不等价的替代输入填表。

### 6.2 公平性要求

固定输入、指令、采样次数、硬件、最大时延、采样率、后处理权限和外部数据权限。对随机生成模型报告至少 3 个随机种子或每条指令多次采样的均值和方差。

## 7. 统计与通过规则

### 7.1 自动指标

每项报告：均值、median、P95、标准差、样本数、分任务类型结果和 95% bootstrap CI。对于“≤”指标，优先使用 P95；对于“≥”指标，报告总体均值以及最差任务切片。

### 7.2 主观指标

报告 MOS/匹配度均值、标准差、95% CI、样本级通过率和评测者一致性。推荐总体验收条件写成：

```text
主指标达到阈值，且关键子集（歌词、局部边界、多轨）没有明显失效。
```

不能用一个加权总分掩盖某一项关键能力失败。

### 7.3 推荐验收表

| 能力 | 主指标 | 阈值 |
|---|---|---|
| 衔接自然 | 音频质量/过渡 MOS | ≥4.5 |
| 风格、情绪、音色 | 专家匹配度 MOS | ≥4.0 |
| 主歌/副歌结构 | 边界误差 P95 | ≤2 s |
| 节拍 | beat-to-beat P95 | ≤20 ms |
| 和声 | WCSR | ≥90% |
| 多轨同步 | 编辑后跨轨相对事件时间 P95 | ≤20 ms |
| 音量 | 短时 LUFS 偏差 P95 | ≤3 dB |
| 多轨整体质量 | 专家 MOS | ≥4.0 |
| 意图理解 | checklist 完全成功率 | ≥85% |

## 8. 推荐实现工具栈

- `mir_eval`：beat、chord、部分结构/音乐信息检索指标；
- `madmom`、`librosa`：beat/downbeat、onset、谱特征；
- `Essentia`：响度、节拍、和声、音频描述符；
- BS.1770/EBU R128 兼容响度实现：LUFS、LRA；
- `museval`/BSS Eval：stem 分离质量诊断（不替代同步指标）；
- 强制对齐/歌唱 ASR 工具：词/音节时间戳；
- CLAP、MuLan、MusicCLIP：文本—音乐相似度辅助；
- 统计脚本：bootstrap CI、配对检验、效应量、Fleiss’ κ/Krippendorff’s α。

所有工具、模型版本和参数应写入 `environment.yml` 或容器配置，并在 benchmark 发布时锁定版本。

## 9. 结果报告模板

每个模型至少报告以下字段：

```text
model, task_type, song_split, seed, sample_count
section_error_mae, section_error_p95
beat_mae_ms, beat_p95_ms, beat_f1
harmony_wcsr, root_accuracy
stem_sync_p95_ms, max_crosscorr_lag_ms
lufs_bias_mean_db, lufs_bias_p95_db
audio_mos_mean, audio_mos_ci95
style_emotion_match_mean, match_ci95
semantic_success_rate, annotator_agreement
latency_p50, latency_p95, vram_gb, failure_rate
```

同时发布失败案例：错拍、段落错位、音量跳变、音色漂移、歌词错唱、多轨不同步等，而不只发布总分。

## 10. 结论与执行顺序

这些指标大多有成熟的测量基础，但没有一套现成 benchmark 能直接覆盖“局部、多轨、自然语言音乐编辑”的全部要求。推荐采用：

1. **标准方法作为测量层**：P.800/P.808/MUSHRA、BS.1770/EBU R128、`mir_eval` beat/chord、onset/互相关、CLAP；
2. **任务标注作为真值层**：编辑区间、段落、beat、和弦、歌词、保持/改变属性；
3. **项目阈值作为验收层**：20 ms、3 dB、90%、85%、MOS 4.5/4.0；
4. **人工盲评作为最终听感校验**：自动指标不能替代自然度、音乐性和意图完成度。

下一步应先冻结任务 JSON schema 和《指标定义表》，用 20–50 首歌曲做 pilot，检查自动指标与专家听感的一致性，再开始全量跑 ACE-Step 1.5、DiffRhythm 2 和项目模型。

## 11. 最终主方法规范（每项指标只保留一种）

本节是可直接执行的主评测协议，优先级高于第 4 节中列出的备选方法。选择原则是：能直接对应图片中的验收阈值、能在 MUSDB18 上获得可复现真值、实现成本可控，并且结果对音乐编辑失败模式敏感。除特别说明外，所有自动指标都在编辑区和编辑边界扩展区（前后各 1 s）上计算；扩展区用于检测拼接伪影，但不把整首歌未编辑部分重复计入分数。

### 11.1 音频质量/衔接自然度：ACR MOS（1–5 分）

**主方法**：采用 ITU-T P.800 的 Absolute Category Rating（ACR）式 MOS，而不采用 MUSHRA 作为主验收分数。

**评测问题**：只评价“音频质量与编辑衔接自然度”，不评价是否符合文字指令；后者由 11.9 单独评测。

**流程**：

1. 从每条结果截取“编辑前 2 s + 编辑区 + 编辑后 2 s”，统一采样率、声道和播放响度；
2. 隐藏模型和文件名，随机化样本顺序；
3. 每个样本由至少 5 名具有音乐制作/音频背景的评测者评分；
4. 使用 1–5 分锚定量表：1=不可接受，3=基本可听但有明显缺陷，5=专业质量且无明显拼接痕迹；
5. 对样本和模型分别计算平均分，并报告标准差、95% bootstrap CI、样本级通过率。

**计算**：

$$
MOS=\frac{1}{N}\sum_{j=1}^{N}r_j
$$

**来源**：ITU-T P.800（主观语音/音频质量评分方法）和 ITU-T P.808（远程/众包主观评测流程）。上式是评分的算术均值；P.800/P.808 规定的是评测设计和量表，不是音乐生成模型论文中的专有公式。

**选择原因**：图片直接给出 MOS≥4.5，ACR 的 1–5 分量表可直接对应该阈值；MUSHRA 的 0–100 分不能直接与 4.5 比较。MUSHRA可作为论文附加实验，但不能替代主验收 MOS。

**通过规则**：模型总体 MOS 均值≥4.5；同时报告 95% CI 下界。若项目需要高置信度验收，采用“CI 下界≥4.5”，否则必须明确“按点估计判定”。

### 11.2 曲风、情绪、乐器和音色匹配：分维度专家匹配度 MOS

**主方法**：专家盲评的 1–5 分匹配度评分，分别对曲风、情绪、主要乐器和主体音色评分，不使用 CLAP 分数直接替代人工判断。

**流程**：每条自然语言指令先转为结构化属性；评测者分别回答“结果是否符合指定曲风/情绪”“指定乐器是否存在且未被错误替换”“编辑前后主体音色是否连续”。每个维度使用相同 1–5 锚定量表，至少 5 名专家独立评分。

**计算**：对维度 (d) 单独求 MOS；若业务必须得到一个匹配分，使用四个维度的等权平均，并同时公布分维度结果：

$$
MatchMOS_d=\frac{1}{N}\sum_{j=1}^{N}r_{j,d}
$$

$$
MatchMOS=\frac{1}{4}\sum_{d\in\{style,emotion,instrument,timbre\}}MatchMOS_d
$$

**来源**：主观评分流程参考 ITU-T P.800/P.808；音乐风格和音色没有一个公认的单一 MOS 标准，因此这里是基于该标准的项目化评分协议。CLAP（Wu et al., 2023）和 MuLan（Huang et al., 2022）仅作为自动相关性诊断，不作为 4.0 阈值的唯一依据。

**选择原因**：图片要求的是“专家评价匹配度≥4.0”，人工分维度最能覆盖“只改副歌、保留音色但改变情绪”等复合意图；单一 embedding 相似度不能判断保留约束。

**通过规则**：每个必要维度均值≥4.0；不建议只看四维平均值，因为某一关键维度失败可能被其他维度抵消。

### 11.3 主歌/副歌边界：时间边界 P95 误差

**主方法**：采用人工校正的 verse/chorus 边界时间戳，计算编辑结果边界经全局时间对齐后的绝对误差；以 P95 作为主统计量。

**前处理与真值**：两名标注者独立标注主歌、副歌的起点和终点；冲突超过 1 s 时由第三名专家仲裁。对结果和原曲使用未编辑区估计全局偏移 β（若存在速度变化则估计线性映射 αt+β）。

**计算**：

$$
E_s=\left|\frac{\hat t_s-\beta}{\alpha}-t_s\right|
$$

其中 (s) 是 verse/chorus 的某个起点或终点；若未允许变速，固定 α=1。

**来源**：音乐结构分割评测中的 boundary distance/tolerance 思路（例如 Nieto and Bello, 2016；`mir_eval` 的结构评测实现）。上述加入 α、β 的形式是本项目针对局部编辑拼接的适配定义，不是某篇基线论文的原式。

**选择原因**：图片阈值以秒为单位，直接报告时间误差比 Boundary F1 更可解释；F1 作为附加诊断即可。

**通过规则**：所有必需主歌/副歌边界的 P95≤2 s；同时报告漏检率。

### 11.4 节拍：逐拍匹配的 P95 时间误差

**主方法**：使用同一个固定 beat/downbeat tracker（建议 `madmom` 或 `librosa` 的锁定版本）提取原曲和编辑结果节拍，进行一对一最近邻匹配后计算时间误差。

**匹配规则**：先用编辑区前后未编辑音频估计全局偏移；在相同 beat 序号或最近邻候选中匹配，候选距离超过 100 ms 的视为漏检；一对多匹配禁止。

**计算**：

$$
e_i=\left|\hat b_i-(\alpha b_i+\beta)\right|\times1000\quad(ms)
$$

主指标为 (P95(e_i))，同时报告 beat recall 和误检率。

**来源**：MIREX Beat Tracking Evaluation 和 `mir_eval.beat`（Raffel et al., 2014）提供 beat Precision/Recall/F-measure、Cemgil 和 continuity 等标准评测；逐拍绝对时间差和 P95 是本项目为“≤20 ms”编辑验收增加的适配统计。

**选择原因**：F1 只能说明检测事件是否命中，不能直接回答每拍偏移多少；逐拍 P95 与业务 20 ms 阈值一一对应。若任务明确允许改变 BPM，改用目标 beat grid 而不是原曲 beat grid。

**通过规则**：preserve_tempo=true 的任务中，匹配节拍 P95≤20 ms，beat recall≥95%。

### 11.5 和声：Weighted Chord Symbol Recall（WCSR）

**主方法**：采用 `mir_eval.chord` 的 Weighted Chord Symbol Recall，比较编辑结果与参考和弦在时间上的一致性。

**前处理**：统一 chord vocabulary、转位和等价和弦规则；优先使用人工校正的 MUSDB18 和弦标注，自动和弦识别结果只能作为初始标注。编辑指令要求改变和声时，建立“目标和弦标注”，与目标而非原曲比较。

**计算**：

$$
WCSR=\frac{\sum_{q=1}^{Q}\operatorname{dur}(q)\,\mathbf{1}[c_q=\hat c_q]}{\sum_{q=1}^{Q}\operatorname{dur}(q)}\times100\%
$$

**来源**：`mir_eval.chord`/MIREX chord evaluation 的 Weighted Chord Symbol Recall；其实现包含和弦等价类、时间加权和无和弦区间处理规范。上式是对应实现的简化表达。

**选择原因**：图片给出的是百分比阈值，WCSR天然以时间加权百分比表示，比 chroma cosine 更容易解释；Root Accuracy 只比较根音，过于宽松，适合作为诊断而非主指标。

**通过规则**：preserve_harmony=true 的编辑区 WCSR≥90%，并报告 Root Accuracy 作为错误分析。

### 11.6 多轨协同：编辑后跨轨同步误差（主指标）

这里需要区分两个问题：

1. **跨轨协同同步**：编辑后的 vocals、drums、bass、other 是否彼此同步；这是图片“多轨协同”中“对齐误差”的主含义。
2. **时间保持误差**：编辑后的某条 stem 是否仍处于原曲对应的时间位置；这是辅助指标，不应冒充跨轨同步指标。

#### 主指标：编辑后各 stem 之间的相对事件时间误差

**前处理**：

- 对编辑后的每条 stem 使用同一版本的 onset detector 和阈值；鼓轨优先使用 percussive onset，人声使用音节/音符 onset；
- 对鼓、bass 等预期与节拍对齐的轨道，使用编辑后混音的 beat/downbeat grid 或任务规定的目标 beat grid；
- 对 vocals 等不一定落在拍点上的轨道，只比较任务标注的对应事件（例如同一音节、同一音符或同一段落进入点），不能要求所有 stem 的 onset 都落在同一个 beat 上；
- 对每个有定义的对应事件 (i)，在相关 stem 中采用一对一匹配，最大允许距离 50 ms；静音、低能量和任务中不存在该事件的 stem 标记为“不适用”，不能当作错位；
- 事件数量不足时报告覆盖率，避免少量 onset 得出虚假的高分。

令 \(\hat t_{i,k}\) 为编辑后第 (k) 条 stem 的对应事件时间，\(d^{target}_{i,k,l}\) 为任务规定的两轨目标相对时间差（若任务要求保持原曲协同关系，则由原曲标注得到）。定义跨轨同步误差：

$$
E^{cross}_{k,l}=P95_i\left|\left(\hat t_{i,k}-\hat t_{i,l}\right)-d^{target}_{i,k,l}\right|\times1000\quad(ms)
$$

对于节拍对齐的事件，也可以将目标相对时间差写成公共网格形式：

$$
E^{grid}_k=P95_i\left|\hat t_{i,k}-\hat g_i\right|\times1000\quad(ms)
$$

**来源**：onset detection 的基础方法来自 Bello et al. (2005) 的音乐起音检测综述，实际检测可用锁定版本的 `librosa`/Essentia；beat/downbeat 评测可参考 MIREX/`mir_eval.beat`。上面“编辑后轨道之间的相对时间差相对目标关系”是本项目对多轨编辑任务的适配定义，不是 ACE-Step 1.5 或 DiffRhythm 2 的原始公式。

**选择原因**：它直接测量“编辑后不同轨之间是否保持目标协同关系”，不会错误地要求 vocals 等非节拍事件与鼓点同时发生。互相关只能检测整体平移，不能发现局部轨道错位；相对 onset/目标 beat grid 可以检查鼓点、人声进入点和 bass 瞬态的局部协同。

**主通过规则**：每个必需 stem 对或任务规定的同步关系均满足 (E^{cross}_{k,l}\le20) ms；事件覆盖率≥95%；同时报告最大相对误差和无法检测事件比例。

#### 辅助指标：编辑后相对于原曲的时间保持误差

只有在 `must_preserve` 包含 `tempo` 或 `non_edit_region` 时计算。令 (t_{i,k}) 为原曲对应事件，\(\alpha t_{i,k}+\beta\) 为根据未编辑区估计的原曲参考位置：

$$
E^{ref}_k=P95_i\left|\hat t_{i,k}-(\alpha t_{i,k}+\beta)\right|\times1000\quad(ms)
$$

**解释**：(E^{ref}_k) 衡量编辑是否破坏原曲的时间位置，不等同于跨轨协同误差。建议作为辅助结果单独报告，不将它与 (E^{cross}_{k,l}) 混成一个分数。

### 11.7 音量偏差：短时 LUFS P95

**主方法**：按 ITU-R BS.1770/EBU R128 计算短时 LUFS，在固定滑动窗口内比较原轨和编辑轨的响度差；以 stem 级 P95 为主指标。

**前处理**：统一采样率和声道；窗口 400 ms、步长 100 ms；对低于 -70 LUFS 的静音窗不计算差值；不对每个结果独立归一化。

**计算**：

$$
\Delta L_w=\left|L_{edited,w}^{LUFS}-L_{original,w}^{LUFS}\right|
$$

主指标为每条 stem 的 (P95(\Delta L_w))。

**来源**：ITU-R BS.1770 和 EBU R128 规定 LUFS 的 K-weighting、能量聚合和 gating；编辑前后差值及 P95 是本项目适配。RMS dB 只用于调试，不用于最终验收。

**选择原因**：LUFS 是广播/流媒体广泛采用的响度单位，比波形峰值或 RMS 更接近感知音量；P95 能避免平均值掩盖局部突然变大/变小。

**通过规则**：每条必需 stem 的 LUFS 偏差 P95≤3 dB；边界窗口另报告 loudness jump。

### 11.8 分轨编辑后音质：stem 级 ACR MOS

**主方法**：沿用 11.1 的 ITU-T P.800/P.808 ACR MOS，但将播放对象换成单独 stem；混音后的整体 MOS 不替代 stem 级分数。

**流程**：每条 stem 独立响度匹配后播放编辑前后上下文；评测者评价噪声、断裂、音色伪影和编辑自然度，1–5 分，至少 5 名专家。

**计算**：

$$
MOS_k=\frac{1}{N}\sum_{j=1}^{N}r_{j,k}
$$

**来源**：ITU-T P.800/P.808；stem 级应用是本项目的评测对象适配。

**选择原因**：图片明确要求“分轨编辑后各轨专家音质评分≥4.0”，因此必须逐轨报告，不能只报告最终混音 MOS。

**通过规则**：每条必需 stem 的 MOS_k≥4.0；同时报告最差 stem，禁止只报告四轨平均值。

### 11.9 自然语言意图理解：结构化专家 checklist 完全成功率

**主方法**：将每条自然语言指令拆成可观察的必要条件，由至少 3 名专家盲评；样本只有在全部必要条件满足时才计为成功。

**示例**：“将副歌改成更欢快的摇滚风格，保留原歌词和人声音色”拆成：编辑范围正确、摇滚风格满足、情绪更欢快、歌词保持、主体音色保持。

**计算**：

$$
SuccessRate=\frac{\sum_{n=1}^{N}\mathbf{1}[\text{样本 }n\text{ 的全部必要条件满足}]}{N}\times100\%
$$

**来源**：这是本项目的任务成功率定义，形式上借鉴分类准确率；不是 ACE-Step 1.5 或 DiffRhythm 2 论文中的专用公式。标注规范、必要条件和“不适用”样本处理必须公开。

**选择原因**：CLAP/MuLan 可以评估整体文本—音频相关性，却无法可靠判断“只编辑副歌”“保留歌词”等离散约束；结构化 checklist 可审计、可解释，最适合图片中的“语义匹配率≥85%”。

**通过规则**：总体完全成功率≥85%，并分别报告编辑范围、风格、情绪、歌词、音色等子属性成功率及 Fleiss’ κ/Krippendorff’s α。

## 参考规范与工具

1. ITU-T P.800：Methods for subjective determination of transmission quality。
2. ITU-T P.808：Subjective evaluation of speech quality with a crowdsourcing approach。
3. ITU-R BS.1534：MUSHRA method for the subjective assessment of intermediate quality level of audio systems。
4. ITU-R BS.1116：Methods for the subjective assessment of small impairments in audio systems。
5. ITU-R BS.1770、EBU R128：响度与 LUFS 测量。
6. `mir_eval`：https://mir-eval.readthedocs.io/
7. `museval`/BSS Eval：https://github.com/multimediaeval/museval
8. ACE-Step 1.5：https://arxiv.org/abs/2602.00744
9. DiffRhythm 2：https://arxiv.org/abs/2510.22950

## 12. 最终主方法的来源文献对照

下表只列第 11 节最终采用的方法。引用“标准”表示该组织规定了测量/听评程序；引用“论文”表示论文提出或系统化实现了相关评测；“项目适配”表示公式中的比较对象、P95 聚合或验收阈值由本 benchmark 定义，不能误称为来源论文的原公式。

| 主指标 | 直接来源 | 来源性质 | 本项目新增部分 |
|---|---|---|---|
| ACR MOS | ITU-T P.800；ITU-T P.808 | 国际标准 | 音乐编辑片段截取方式、专家人数和 4.5 阈值 |
| 曲风/情绪/乐器/音色专家评分 | P.800/P.808；CLAP、MuLan 仅作方法论参照 | 标准 + 相关论文 | 四维评分表、各维度≥4.0 |
| 主歌/副歌边界误差 | Nieto & Bello (2016)；McFee et al. (2015) | 结构分析论文 + 评测工具论文 | α/β 对齐、绝对误差 P95、2 s 阈值 |
| 逐拍时间误差 | MIREX Beat Tracking；Raffel et al. (2014) | 社区协议 + 工具论文 | 逐拍毫秒误差、P95、20 ms 阈值 |
| WCSR | MIREX ACE；McFee et al. (2015) `mir_eval.chord` | 社区协议 + 工具论文 | 在编辑区/目标和弦上计算、90% 阈值 |
| stem onset 同步误差 | Bello et al. (2005) | onset detection 综述论文 | stem 事件匹配、P95、20 ms 阈值 |
| 短时 LUFS 偏差 | ITU-R BS.1770；EBU R128 | 国际/行业标准 | 编辑前后 LUFS 差、P95、3 dB 阈值 |
| stem 级 MOS | ITU-T P.800/P.808 | 国际标准 | 逐 stem 播放和每轨≥4.0 |
| checklist 完全成功率 | 分类准确率的一般统计形式 | 项目定义 | 必要条件拆解、完全成功判定、85% 阈值 |

### 12.1 完整参考文献

1. **ITU-T Recommendation P.800**. *Methods for Subjective Determination of Transmission Quality*. International Telecommunication Union.
2. **ITU-T Recommendation P.808**. *Subjective Evaluation of Speech Quality with a Crowdsourcing Approach*. International Telecommunication Union.
3. **ITU-R Recommendation BS.1534**. *Method for the Subjective Assessment of Intermediate Quality Level of Audio Systems (MUSHRA)*. International Telecommunication Union.
4. **ITU-R Recommendation BS.1770**. *Algorithms to Measure Audio Programme Loudness and True-Peak Audio Level*. International Telecommunication Union.
5. **EBU R 128**. *Loudness Normalisation and Permitted Maximum Level of Audio Signals*. European Broadcasting Union.
6. **Raffel, C., McFee, B., Humphrey, E. J., Salamon, J., Nieto, O., Liang, D., & Ellis, D. P. W. (2014)**. *mir_eval: A Transparent Implementation of Common MIR Metrics*. Proceedings of ISMIR.
7. **McFee, B., Raffel, C., Liang, D., Ellis, D. P. W., McVicar, M., Battenberg, E., & Nieto, O. (2015)**. *librosa: Audio and Music Signal Analysis in Python*. Proceedings of the Python in Science Conference.（用于相关音频分析实现；正式分数仍按锁定的 benchmark 代码计算。）
8. **Nieto, O., & Bello, J. P. (2016)**. *Systematic Exploration of Computational Music Structure Research*. Proceedings of ISMIR.（结构边界评测背景。）
9. **Bello, J. P., Daudet, L., Abdallah, S., Duxbury, C., Davies, M., & Sandler, M. B. (2005)**. *A Tutorial on Onset Detection in Music Signals*. IEEE Transactions on Speech and Audio Processing, 13(5), 1035–1047.
10. **Wu, Y., Chen, K., Zhang, T., Hui, Y., Berg-Kirkpatrick, T., & Dubnov, S. (2023)**. *Large-Scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation*. ICASSP.（CLAP；仅作自动辅助。）
11. **Huang, Q., Jansen, A., Lee, J., Ganti, R., Li, J. Y., & Ellis, D. P. W. (2022)**. *MuLan: A Joint Embedding of Music Audio and Natural Language*. Proceedings of ISMIR.（仅作自动辅助。）
12. **Gong, J., et al. (2026)**. *ACE-Step 1.5: Pushing the Boundaries of Open-Source Music Generation*. arXiv:2602.00744.（基线模型；不是上述项目适配公式的来源。）
13. **Jiang, Y., et al. (2025/2026)**. *DiffRhythm 2: Efficient and High Fidelity Song Generation via Block Flow Matching*. arXiv:2510.22950.（基线模型；不是上述项目适配公式的来源。）

### 12.2 引用使用原则

- 若正文写“按 BS.1770 计算 LUFS”，表示底层响度算法来自标准；“编辑前后差值 P95≤3 dB”仍应标注为本项目定义。
- 若正文写“按 `mir_eval.chord` 计算 WCSR”，应锁定软件版本、和弦 vocabulary 和时间区间；不要只复写简化公式后声称完全复现工具行为。
- ACE-Step 1.5 与 DiffRhythm 2 是比较基线。除非其论文原文明确给出并被本项目原样采用，否则不能把本项目的 P95、2 s、20 ms、3 dB 或 checklist 公式署名给这两篇论文。
- 所有阈值都应在 pilot 后冻结。若 pilot 结果显示阈值与专家感知显著不一致，应先修订协议版本，再测试 hidden-test，不能看完最终结果后修改阈值。

## 13. P.800/P.808 原始五级量表与项目适配边界

### 13.1 ACR（Absolute Category Rating）的原始等级标签

P.800 的 ACR listening-quality scale 使用以下英文等级标签（从高到低）：

| 分值 | P.800 原始英文标签 | 本报告建议的中文对译 |
|---:|---|---|
| 5 | Excellent | 优秀 |
| 4 | Good | 良好 |
| 3 | Fair | 一般 |
| 2 | Poor | 较差 |
| 1 | Bad | 很差 |

这些是标准量表的**等级标签**。P.800 并没有为音乐编辑中的“爆音、节拍错位、音色漂移、过渡自然度”等每一种缺陷提供一套专门的音乐锚点描述。因此，本报告第 11.1 节中“接近专业制作质量”“存在明显拼接痕迹”等文字是本项目为帮助评测者理解任务而写的**项目化说明**，不能引用为 P.800 原文。

P.800 的 ACR 问题通常是围绕“对刚才听到的整体质量进行评分”来组织；具体问句会随被测对象（语音、编解码器、传输条件）和试验语言而变化。用于本项目时，应写成明确的音乐问题，例如：

> 请评价这段音频的声音质量和编辑衔接自然度。只考虑噪声、失真、断裂、音色突变和明显拼接痕迹，不考虑您是否喜欢该曲风。

这句话是项目问句，不是 P.800 的逐字原文。

### 13.2 DCR（Degradation Category Rating）的原始五级标签

如果要比较“编辑结果相对于参考音频的劣化程度”，P.800 的 DCR 使用另一套 1–5 标签（从几乎无劣化到劣化严重）：

| 分值 | P.800 DCR 原始英文标签 | 中文对译 |
|---:|---|---|
| 5 | Imperceptible | 感觉不到劣化 |
| 4 | Perceptible, but not annoying | 能察觉，但不令人烦恼 |
| 3 | Slightly annoying | 略令人烦恼 |
| 2 | Annoying | 令人烦恼 |
| 1 | Very annoying | 非常令人烦恼 |

DCR 必须先听参考，再听处理结果，评价的是“相对于参考的劣化”，不是独立的整体质量。因此它不应与 ACR MOS 的 1–5 分混用。

### 13.3 CCR（Comparison Category Rating）的原始等级标签

P.800 还定义了比较两个条件的 CCR。它不是 1–5 分，而是 -3 到 +3：

| 分值 | 原始英文标签 | 中文对译 |
|---:|---|---|
| +3 | Much better | 好很多 |
| +2 | Better | 更好 |
| +1 | Slightly better | 稍好 |
| 0 | About the same | 大致相同 |
| −1 | Slightly worse | 稍差 |
| −2 | Worse | 更差 |
| −3 | Much worse | 差很多 |

CCR 适合项目模型与 ACE-Step 1.5、DiffRhythm 2 做 A/B 比较，但不能直接产生图片要求的 MOS≥4.5。

### 13.4 P.808 的作用

P.808 主要是**远程/众包主观评测的执行与质量控制建议**，不是另一套替代 P.800 的 1–5 语义量表。实际采用时仍使用 ACR 的 1–5 等级标签，同时增加：

- 评测者资格和耳机/环境检查；
- 训练样本和明确的任务说明；
- gold/trap questions；
- 评分时间、异常模式和一致性检查；
- 预先声明的无效评测剔除规则。

因此，本项目报告中准确的写法应是：

> 采用 P.800 的 ACR 五级量表（Excellent/Good/Fair/Poor/Bad），并按照 P.808 的远程/众包质量控制流程执行；音乐缺陷锚点和具体问题句为本项目化适配。
