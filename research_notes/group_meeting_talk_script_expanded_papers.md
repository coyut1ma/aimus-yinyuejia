# 音乐可控编辑项目相关进展组会讲稿：扩展论文版

## 汇报标题

从音乐生成到歌曲级可控编辑：相关进展与项目切入点

## 0. 开场

各位老师、同学好。今天我汇报的是我们这个音乐可控编辑项目的相关进展。

这次调研不是做泛泛的 AI 音乐生成综述，而是围绕我们项目要解决的问题来组织：我们希望对已有歌曲做可控局部编辑，例如改歌词、改旋律、改风格、改人声或伴奏，同时保持原曲的节拍、和声、音色、结构和衔接自然。

所以我把相关工作分成四条线来讲：

1. 可控音乐生成基础模型
2. 歌曲级生成与编辑统一
3. 文本驱动音乐属性编辑
4. 歌词、人声、旋律和分轨协同编辑

最后我会给出项目切入点建议。

## 1. 项目定位：不是泛音乐生成，而是歌曲级可控编辑

我们项目的目标不是从一句 prompt 直接生成一首歌，而是在已有音乐上进行局部修改。

这类任务有几个关键约束：

- 改动要局部：用户只想改一句歌词或一段伴奏，其他部分尽量不变。
- 结构要保持：节拍、段落、主副歌位置不能明显错位。
- 音乐性要保持：旋律、和声、调性、音色不能被无意破坏。
- 多轨要协同：人声、伴奏、鼓、贝斯等不能各改各的。
- 意图要准确：用户说“副歌更燃”“这一句换词”，系统要知道对应改什么。

因此，相关进展不能只看生成质量，而要重点看可编辑性、可控性、对齐能力和真实音频稳定性。

## 2. 论文一：MusicGen

### 论文基本信息

论文全名是 **Simple and Controllable Music Generation**，系统名通常叫 **MusicGen**。  
作者来自 Meta 等机构，发表于 **NeurIPS 2023**，arXiv 编号是 2306.05284。

### 它解决什么问题

MusicGen 主要解决的是条件音乐生成问题，也就是如何根据文本描述或旋律条件生成高质量音乐。

在它之前，很多音乐生成系统会采用多阶段级联结构，例如先生成低分辨率表示，再逐级上采样。这样系统复杂、误差容易累积。MusicGen 的目标是用一个相对简洁的单阶段 Transformer 语言模型完成音乐 token 生成。

### 方法思路

MusicGen 使用压缩后的离散音乐 token 作为建模对象，然后用单个 Transformer LM 建模多个 token stream。它的关键点是设计 token interleaving pattern，让模型能够高效处理多个音频 token 序列。

它支持两类条件：

- 文本条件：根据自然语言描述生成音乐。
- 旋律条件：根据给定旋律引导生成结果。

### 实现效果

论文报告中，MusicGen 在标准 text-to-music benchmark 上优于当时多个基线，并且提供了开源代码和模型，所以它后来成为很多音乐生成与编辑工作的基础对照模型。

### 对我们项目的意义

MusicGen 对我们项目不是直接的编辑方案，但它是一个重要基线：

- 对应项目中的“音乐生成基础模型”部分。
- 可作为 text-to-music 或 melody-conditioned generation 的对照。
- 说明文本和旋律条件确实可以进入音乐生成模型。

但它的局限也很明确：MusicGen 主要是生成，不是编辑。它不能稳定完成“只替换已有歌曲中的一句歌词，同时保持其他部分不变”这种任务。

## 3. 论文二：Mustango

### 论文基本信息

论文全名是 **Mustango: Toward Controllable Text-to-Music Generation**。  
发表于 **NAACL 2024 Long Papers**，arXiv 编号是 2311.08355。

### 它解决什么问题

Mustango 解决的是 text-to-music 里的“音乐理论可控性不足”问题。

很多早期音乐生成模型虽然能根据文本生成音乐，但文本通常只是风格、情绪、乐器等粗粒度描述，很难控制更具体的音乐要素，比如和弦、节拍、速度和调性。

### 方法思路

Mustango 是一个基于 diffusion 的音乐生成系统。它引入了音乐领域知识指导模块 **MuNet**，将文本中的音乐理论信息转成模型可以使用的控制条件。

它还构建了 **MusicBench** 数据集，包含 5.2 万多个带有音乐理论描述的样本。数据增强时会修改音频的和声、节奏、动态等属性，并用 MIR 方法抽取音乐特征，再把这些特征写入文本描述。

### 实现效果

论文报告称，Mustango 在音乐质量和音乐特定文本控制能力上优于 MusicGen 和 AudioLDM2 等模型，尤其是在 chord、beat、tempo、key 等可控属性上更强。

### 对我们项目的意义

Mustango 对应我们项目里的“意图理解到音乐参数控制”部分。

它说明一个重要方向：用户的自然语言需求不能只作为普通文本 prompt，而应该解析出音乐结构参数。例如：

- “更快一点”对应 tempo。
- “更激昂”可能对应节奏密度、动态、配器和和声走向。
- “更忧伤”可能对应调式、速度、乐器和音区。

但 Mustango 仍然主要是生成模型，不是已有歌曲编辑模型。因此它更适合给我们提供“音乐属性控制”的思路，而不是直接解决局部编辑。

## 4. 论文三：Seed-Music

### 论文基本信息

论文全名是 **Seed-Music: A Unified Framework for High Quality and Controlled Music Generation**。  
这是字节 Seed 团队 2024 年发布的 **arXiv 技术报告**，arXiv 编号是 2409.09214，并有官方 demo 页面。

### 它解决什么问题

Seed-Music 试图解决的是音乐创作系统的统一性问题。

传统系统往往把几个任务拆开：文本生成音乐、旋律控制、人声生成、后期编辑分别做。Seed-Music 希望用一个统一框架支持两类 workflow：

- 可控音乐生成
- 生成后的后期编辑

### 方法思路

Seed-Music 结合了自回归语言模型和 diffusion 方法。

在输入条件上，它支持多模态控制，包括：

- 风格描述
- 音频参考
- 乐谱
- voice prompt

在后期编辑上，它支持对生成音频中的歌词和人声旋律进行交互式修改。

### 实现效果

从技术报告和官方样例看，Seed-Music 展示了较强的高质量音乐生成、细粒度风格控制、人声生成和后期编辑能力。它的意义不只是单个指标，而是说明工业级系统已经开始把“生成”和“编辑”合并成一个统一音乐创作框架。

### 对我们项目的意义

Seed-Music 对应我们项目里的“统一编辑系统架构”。

它给我们的启发是：后续系统不能简单地把生成模型、分离模型、编辑模型、混音模块硬拼起来，而应该有统一的音乐表示和统一的控制接口。

但它的局限是：这是工业技术报告，复现门槛高，训练数据、模型规模和完整工程链路都不容易获得。因此它更适合作为系统目标参考，而不是短期可复现基线。

## 5. 论文四：SongEditor

### 论文基本信息

论文全名是 **SongEditor: Adapting Zero-Shot Song Generation Language Model as a Multi-Task Editor**。  
发表于 **AAAI 2025**，会议论文集信息是 AAAI Conference on Artificial Intelligence 39(24): 25597-25605，DOI 为 10.1609/aaai.v39i24.34750。

### 它解决什么问题

SongEditor 是目前和我们项目最直接相关的论文之一。

它指出，已有歌曲生成模型已经可以同时生成几分钟的人声和伴奏，但对已有歌曲做局部调整和编辑的研究还不足。真实音乐制作中，用户经常需要局部修改，而不是每次重新生成整首歌。

### 方法思路

SongEditor 把 zero-shot song generation language model 改造成多任务编辑器。

它支持两类核心编辑：

- **segment-wise modification**：片段级修改，例如生成整段、补全 masked lyrics。
- **track-wise modification**：分轨级修改，例如调整 lyrics、vocals、accompaniments。

系统核心组件包括：

- music tokenizer
- autoregressive language model
- diffusion generator

它既可以从零生成歌曲，也可以编辑已有歌曲中的片段、人声或背景音乐。

### 实现效果

论文报告中，SongEditor 在端到端歌曲编辑任务上通过客观指标和主观指标验证了效果，强调其在 song editing 上表现优于相关基线。更重要的是，它把“歌词、人声、伴奏、片段编辑”放进了同一个编辑框架。

### 对我们项目的意义

SongEditor 对应我们项目中的三个核心方向：

- 局部片段编辑
- 多轨协同编辑
- 歌词、人声和伴奏修改

它是最值得我们优先精读和对标的工作。

如果老师问“哪篇论文最像我们项目”，我会回答：SongEditor 最像，因为它直接把歌曲生成模型改造成多任务歌曲编辑器，并且明确处理 segment-wise 和 track-wise 编辑。

它的局限是：这类系统对数据、tokenizer、语言模型和 diffusion generator 的协同要求很高，复现成本较高；同时，论文效果不等于能在任意真实歌曲上稳定达到专业后期制作质量。

## 6. 论文五：MusicMagus

### 论文基本信息

论文全名是 **MusicMagus: Zero-Shot Text-to-Music Editing via Diffusion Models**。  
2024 年发布在 **arXiv**，编号是 2402.06178。

### 它解决什么问题

MusicMagus 解决的是文本驱动音乐编辑问题，尤其是如何在不额外训练的情况下，对已有或生成音乐的某些属性进行修改。

它关注的编辑对象主要是属性级编辑，例如：

- genre
- mood
- instrument
- timbre

### 方法思路

MusicMagus 把文本编辑转化为 latent space manipulation。

简单说，就是模型不直接在波形上硬改，而是在扩散模型的潜空间中改变和目标文本相关的方向，同时加入一致性约束，尽量保持不该变的内容。

它不需要为每类编辑重新训练一个模型，可以接入已有 text-to-music diffusion model。

### 实现效果

论文报告称，MusicMagus 在 style transfer 和 timbre transfer 等任务上优于一些 zero-shot 和 supervised baseline，并展示了实际音乐编辑场景中的可用性。

### 对我们项目的意义

MusicMagus 对应我们项目里的“风格/情绪/乐器属性编辑”模块。

如果用户说：

- “这段改成摇滚风”
- “伴奏更轻快”
- “乐器换成钢琴为主”

MusicMagus 这类方法可以作为技术基线。

但它不是精确歌曲编辑方案。它对“把某一句歌词换掉”“保持原旋律和节拍”“只改人声不动伴奏”这类需求支持不够。因此它适合放在属性编辑模块，而不是项目主线。

## 7. 论文六：MEDIC

### 论文基本信息

论文全名是 **MEDIC: Zero-shot Music Editing with Disentangled Inversion Control**。  
2024 年发布在 **arXiv**，编号是 2407.13220；也有 OpenReview 页面和项目 demo 页面。OpenReview 页面显示其为 ICLR 2025 withdrawn submission，因此正式汇报时建议称为 arXiv 预印本，而不是已录用会议论文。

### 它解决什么问题

MEDIC 关注 zero-shot 音乐编辑中一个很关键的问题：inversion 误差。

很多扩散编辑方法要先把原始音频反推回扩散模型的隐空间，也就是 DDIM inversion 一类方法。但这个过程会逐步积累误差，导致编辑后：

- 原始内容保持不好
- 编辑目标不准确
- 音乐结构被破坏

MEDIC 试图在实现目标编辑的同时，保持原音乐的核心内容。

### 方法思路

MEDIC 提出 **Disentangled Inversion Control**。

它把扩散过程拆成多个分支，用来纠正 source branch 在 inversion 过程中偏离的问题。

另外，它提出 **Harmonized Attention Control**，把 self-attention control 和 cross-attention control 结合起来，希望逐步获得目标音乐中的和声和旋律信息，同时保留源音乐内容。

### 实现效果

MEDIC 的一个重要贡献是提出 **ZoME-Bench**，包含 1100 个样本和 10 类音乐编辑任务，用于评估 zero-shot 和 instruction-based music editing。

论文报告称，它在 edit fidelity 和 essential content preservation 上优于当时的 inversion 技术。

### 对我们项目的意义

MEDIC 对应我们项目中的“编辑评测”和“内容保持”部分。

它给我们的启发是：音乐编辑不能只看目标 prompt 是否满足，还要同时评估原始内容是否保持。对我们项目来说，应该至少评估：

- 编辑成功率
- 原旋律保持
- 节拍保持
- 音色保持
- 未编辑区域保持
- 边界自然度

但 MEDIC 主要还是属性级和语义级编辑，不是完整歌曲的歌词替换或分轨协同方案。

## 8. 论文七：Editing Music with Melody and Text: Using ControlNet for Diffusion Transformer

### 论文基本信息

论文全名是 **Editing Music with Melody and Text: Using ControlNet for Diffusion Transformer**。  
2024 年发布在 **arXiv**，编号是 2410.05151，2025 年有更新版本。

### 它解决什么问题

这篇论文解决的是 melody-control music generation/editing 问题。

已有可控音乐生成和编辑方法常受限于 Mel-spectrogram 表示和 UNet 架构，生成长度、音质和控制精度都有问题。作者希望用 Diffusion Transformer 加 ControlNet 的方式，让模型同时受文本和旋律控制。

### 方法思路

它基于 StableAudio 一类 text-controlled DiT 模型，在 DiT 上加入 ControlNet 分支。

核心设计有两个：

第一，用 text prompt 控制音乐风格和语义。

第二，用 melody prompt 控制旋律走向。论文没有简单使用 chroma，而是提出 top-k CQT 表示，以减少多轨和宽音域音乐中的旋律歧义。

另外，论文使用 progressive curriculum masking，让模型逐步学习如何平衡文本和旋律条件。

### 实现效果

论文在 text-to-music generation 和 music-style transfer 上做实验。结果显示，该方法在 melody-controlled editing 上优于 MusicGen stereo melody baseline，同时保留较好的文本生成能力。

### 对我们项目的意义

这篇论文对应我们项目里的“旋律保持”和“旋律条件编辑”模块。

如果我们做“改歌词但保持原旋律”，或者“改伴奏风格但保持主旋律”，旋律条件非常关键。

不过它的实验主要集中在无歌声 instrumental music 和风格迁移任务上，还不是完整歌曲的人声歌词编辑方案。因此它适合作为旋律控制技术参考，而不是直接落地方案。

## 9. 论文八：REFFLY

### 论文基本信息

论文全名是 **REFFLY: Melody-Constrained Lyrics Editing Model**。  
发表于 **NAACL 2025 Long Papers**，论文集页码 11295-11315，DOI 为 10.18653/v1/2025.naacl-long.564。

### 它解决什么问题

REFFLY 解决的是旋律约束下的歌词编辑问题。

传统 melody-to-lyric 方法通常是从零生成歌词，但实际创作中，用户可能已经有一句话、一个主题、一个翻译草稿或一段文本，只是需要把它改得更适合旋律演唱。

REFFLY 的核心问题是：如何把普通文本改写成符合给定旋律约束的歌词，同时尽量保留原语义。

### 方法思路

REFFLY 是一个歌词修订框架，全名可以理解为 REvision Framework For LYrics。

它训练 lyric revision module，把 plain text draft 转换成 melody-aligned lyrics。

同时，论文还使用一些 training-free heuristics 来保持语义和音乐一致性，例如控制音节数、重音位置和旋律结构匹配。

### 实现效果

论文在 song translation 等任务上验证效果，并报告模型相对 Lyra 和 GPT-4 等强基线，在 musicality 和 text quality 上都有明显提升，摘要中提到提升约 25%。

### 对我们项目的意义

REFFLY 对应我们项目里的“歌词编辑前处理”模块。

如果用户说“把这一句歌词换成某个意思”，我们不能直接把普通文本扔给人声编辑模型，因为新歌词可能不适合原旋律，比如音节数不对、重音位置不对、句长不对。

REFFLY 可以先把用户输入改写成“可唱、可对齐、适合原旋律”的歌词，再交给后续 singing voice editing 模块。

它的局限是：REFFLY 本身不生成音频，只解决歌词文本和旋律约束之间的匹配问题。

## 10. 论文九：YingMusic-Singer

### 论文基本信息

论文全名是 **YingMusic-Singer: Controllable Singing Voice Synthesis with Flexible Lyric Manipulation and Annotation-free Melody Guidance**。  
2026 年发布在 **arXiv**，编号是 2603.24589。

### 它解决什么问题

YingMusic-Singer 解决的是改歌词但保持旋律的问题。

已有 singing voice synthesis 或 singing voice editing 方法往往有两个问题：

- 控制性有限
- 需要人工对齐歌词和旋律

而真实应用中，用户希望输入一个旋律参考片段和修改后的歌词，就能重新生成歌声，不想手动标注对齐信息。

### 方法思路

YingMusic-Singer 是一个完全基于 diffusion 的歌声合成模型。

它的输入包括：

- 可选的音色参考
- 提供旋律的歌唱片段
- 修改后的歌词

它不要求人工对齐。训练上使用 curriculum learning 和 Group Relative Policy Optimization，以增强旋律保持和歌词依从性。

### 实现效果

论文报告称，它在 melody preservation 和 lyric adherence 上优于 Vevo2 这个可比基线，并提出 **LyricEditBench**，用于评估 melody-preserving lyric modification。

### 对我们项目的意义

YingMusic-Singer 对应我们项目里最核心的“改歌词但保持旋律”模块。

它非常适合支撑我们的早期验证任务：输入原歌曲或原人声片段，替换歌词，同时尽量保持原旋律和音色。

它的局限是：它主要针对 singing voice synthesis/editing，不直接解决混音后整首歌中的伴奏同步和多轨混音问题。

## 11. 论文十：MeloDISinger

### 论文基本信息

论文全名是 **MeloDISinger: Melody-Aware & Duration-Preserving Singing Voice Editing with Audio Infilling**。  
2026 年发布在 **arXiv**，编号是 2606.30580。

### 它解决什么问题

MeloDISinger 解决的是文本驱动歌声编辑中一个很实际的问题：改了歌词之后，如何保持旋律、总时长和未编辑区域。

这和我们的项目高度相关。因为如果用户只想改一句歌词，系统不能因为新歌词长短不同就改变整段时长，也不能让前后句衔接断裂。

### 方法思路

MeloDISinger 是一个基于 flow-matching 的 singing voice editing 模型。

它的核心模块是 **MeloDRP**，用于预测固定预算的 duration ratios，从而实现 span-wise duration control。

同时，它融合 phonetic cues 和 pseudo-MIDI melodic context，让 duration 分配和旋律结构相关。最后用 flow-matching mel decoder 做 audio infilling，只合成编辑区域，同时保留周围上下文。

### 实现效果

论文报告称，它在客观和主观评价中达到 state-of-the-art。关键不只是音质，而是它专门处理了 duration-preserving 和 context-preserving 这两个真实编辑场景中的核心问题。

### 对我们项目的意义

MeloDISinger 是最贴近我们“衔接自然”诉求的工作之一。

它对应我们项目中的：

- 局部歌词替换
- 时长保持
- 未编辑区域保持
- 边界自然衔接

如果我们要设计第一个 demo，我建议重点参考它的任务定义和评价方式。

局限是它偏 singing voice editing，并不直接解决伴奏同步和整曲混音后处理。

## 12. 论文十一：AUDEDIT

### 论文基本信息

论文全名是 **AUDEDIT: Inversion-Free Text-Guided Editing with Pretrained Audio Flow Models**。  
2026 年发布在 **arXiv**，编号是 2606.15149。

### 它解决什么问题

AUDEDIT 解决的是真实音频文本编辑中的 inversion 问题。

很多 audio-to-audio editing 方法会先给原音频加噪，再按新 prompt 去噪。但这种 inversion-style route 常常需要在两件事之间做取舍：

- 是否遵循目标文本
- 是否保持原音频的节奏、瞬态、音色和长程结构

AUDEDIT 希望不走传统 inversion 路线，而是用 flow 模型直接做 source-to-target 编辑。

### 方法思路

它基于 pretrained rectified-flow audio generator，设计了 inversion-free 的编辑机制。

在 flow step 中，它比较 target-conditioned 和 source-conditioned velocity fields，并用差值更新编辑后的 latent。

它不需要训练、不需要 paired edit data、不需要优化过程，也不依赖内部 attention map。

### 实现效果

论文在 sound-effect 和 music editing sets 上做实验。报告显示，相比 SDEdit、ODE inversion 和 FireFlow，AUDEDIT 在 CLAP text alignment 和 audio preservation 上都有提升。例如在 sound effects 上，相比最强基线将 target-text CLAP similarity 从 0.42 提升到 0.52，同时 FAD 从 65.70 降到 50.37。

### 对我们项目的意义

AUDEDIT 对应我们项目中的“真实音频编辑”和“结构保持”模块。

它提醒我们：如果直接在真实歌曲上做文本编辑，inversion 误差和结构保持会是很大的问题。未来如果我们做混音后音频编辑，可以关注 inversion-free flow editing。

但它是通用音频编辑方法，不是专门的歌曲级歌词替换或多轨编辑方案。

## 13. 论文十二：VersBand

### 论文基本信息

论文全名是 **Versatile Framework for Song Generation with Prompt-based Control**，系统名是 **VersBand**。  
2025 年发布在 **arXiv**，编号是 2504.19062，并有项目 demo 页面。

### 它解决什么问题

VersBand 解决的是多任务歌曲生成中人声和伴奏对齐不足的问题。

现有歌曲生成方法往往难以同时生成高质量人声和伴奏，并且保证二者在节奏、旋律、风格和情绪上对齐。

### 方法思路

VersBand 是一个多任务歌曲生成框架，由多个模块组成：

- VocalBand：生成带风格控制的人声。
- AccompBand：生成与人声对齐的伴奏。
- LyricBand：生成歌词。
- MelodyBand：生成旋律。

其中 AccompBand 使用 flow-based transformer，并引入 Band-MOE，以提升质量、对齐和控制能力。

### 实现效果

论文报告称，VersBand 在多个歌曲生成任务上通过客观和主观指标优于基线，尤其强调 aligned vocals and accompaniments，也展示了 vocal-to-song generation 和 music style transfer 等样例。

### 对我们项目的意义

VersBand 对应我们项目里的“多轨协同”和“人声伴奏对齐”模块。

虽然它偏歌曲生成，而不是已有歌曲编辑，但它说明多轨协同不能只靠最后混音解决，而应该在生成或编辑阶段就显式建模人声和伴奏之间的关系。

它的局限是：项目目标如果是编辑已有真实歌曲，VersBand 不能直接完成“只修改某一轨并保持其他轨不变”，但它对系统架构设计很有参考价值。

## 14. 小结：这些论文分别支撑项目的哪一部分

可以把上述工作映射到我们项目的不同模块：

| 项目模块 | 代表论文 | 能提供什么 |
|---|---|---|
| 基础音乐生成 | MusicGen | 文本/旋律条件生成基线 |
| 音乐理论控制 | Mustango | chord、beat、tempo、key 等显式控制 |
| 统一生成与编辑框架 | Seed-Music | 生成和后期编辑一体化设计 |
| 歌曲级编辑 | SongEditor | 片段级、分轨级歌曲编辑 |
| 属性编辑 | MusicMagus、MEDIC、AUDEDIT | 风格、情绪、乐器、真实音频属性编辑 |
| 旋律控制 | ControlNet for DiT | 文本 + 旋律双条件编辑 |
| 歌词适配 | REFFLY | 把普通文本改写成适合旋律的歌词 |
| 歌声改词 | YingMusic-Singer、MeloDISinger | 改歌词并保持旋律、时长和上下文 |
| 多轨协同 | VersBand | 人声和伴奏对齐、模块化歌曲系统 |

## 15. 当前项目建议

基于这些论文，我认为项目不应该一开始就做“任意歌曲全功能编辑”。

更合理的路线是：

第一阶段，做人声歌词局部替换。

具体任务是：输入一首含人声歌曲，指定一句歌词替换，输出改词后的歌曲；要求原旋律、节拍、音色和未编辑区域尽量保持，编辑边界自然。

这个阶段可以重点参考：

- REFFLY：做歌词适配旋律。
- YingMusic-Singer：做改词并保持旋律。
- MeloDISinger：做局部 infilling、时长保持和边界自然。

第二阶段，做伴奏和属性编辑。

可以参考：

- MusicMagus：风格、情绪、乐器属性编辑。
- MEDIC：zero-shot 编辑和内容保持评价。
- AUDEDIT：真实音频结构保持。
- ControlNet for DiT：旋律和文本双条件控制。

第三阶段，做多轨协同系统。

可以参考：

- SongEditor：segment-wise 和 track-wise 歌曲编辑。
- Seed-Music：统一生成与后期编辑框架。
- VersBand：人声和伴奏对齐。

最终系统更像一个编排式音乐编辑工作流，而不是单一模型。它需要包括意图解析、歌词适配、人声编辑、伴奏编辑、对齐、边界平滑、响度匹配和质量检查。

## 16. 结尾总结

最后总结一下。

第一，相关研究已经从纯音乐生成走向可控编辑，尤其是 SongEditor 和 Seed-Music 说明歌曲级编辑正在成为独立任务。

第二，文本驱动属性编辑已经相对成熟，可以作为风格、情绪和乐器编辑基线，但还不能解决精确改词和多轨协同。

第三，歌词、人声和旋律编辑是目前最适合作为我们项目入口的方向，尤其是 REFFLY、YingMusic-Singer 和 MeloDISinger。

第四，多轨协同和无痕衔接仍然是主要难点，需要结合 SongEditor、VersBand 以及后处理评测机制逐步推进。

因此，项目最稳妥的切入点是先做“人声歌词局部替换”，再扩展到伴奏属性编辑和完整多轨协同编辑系统。

## 参考链接

- MusicGen / Simple and Controllable Music Generation: https://arxiv.org/abs/2306.05284
- Mustango: Toward Controllable Text-to-Music Generation: https://arxiv.org/abs/2311.08355
- Seed-Music: A Unified Framework for High Quality and Controlled Music Generation: https://arxiv.org/abs/2409.09214
- SongEditor: Adapting Zero-Shot Song Generation Language Model as a Multi-Task Editor: https://arxiv.org/abs/2412.13786
- MusicMagus: Zero-Shot Text-to-Music Editing via Diffusion Models: https://arxiv.org/abs/2402.06178
- MEDIC: Zero-shot Music Editing with Disentangled Inversion Control: https://arxiv.org/abs/2407.13220
- Editing Music with Melody and Text: Using ControlNet for Diffusion Transformer: https://arxiv.org/abs/2410.05151
- REFFLY: Melody-Constrained Lyrics Editing Model: https://aclanthology.org/2025.naacl-long.564/
- YingMusic-Singer: https://arxiv.org/abs/2603.24589
- MeloDISinger: https://arxiv.org/abs/2606.30580
- AUDEDIT: https://arxiv.org/abs/2606.15149
- VersBand: https://arxiv.org/abs/2504.19062

