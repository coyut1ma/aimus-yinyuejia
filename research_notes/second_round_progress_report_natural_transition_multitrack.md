# 衔接自然与多轨协同：二次调研报告

调研目标：只聚焦两个问题。  
1. **衔接自然**：局部改完以后，前后听起来不能断。  
2. **多轨协同**：对同一个时间片段里的多条轨做修改时，它们之间不能乱。

这份报告尽量少用术语。必须用术语时，会先解释它是什么意思。

## 1. 总体结论

目前没有一个公开方法可以直接稳定完成“任意真实歌曲的高质量局部多轨编辑”。更现实的做法是把问题拆成一个流程：

1. **先定位要改的时间片段**，比如第 35 秒到第 39 秒。
2. **对这个片段里的多条轨一起做修改**，而不是只看某一条轨。
3. **改完以后把边界接顺**，不能让人听出断层。
4. **最后检查节拍、音高、音色、响度和轨道之间的关系**，看有没有被改乱。

所以，这个方向不是“找一个万能模型”，而是“做一套能协调多条轨的编辑流程”。

## 2. 痛点 A：衔接自然

### 2.1 这个痛点到底是什么

衔接自然不是简单说“音质好”，而是指：

> 如果我们只改歌曲中间几秒钟，改完后这几秒要能自然地接在原曲前后，不能让人听出明显断层。

比如原曲第 35 秒到 39 秒的人声要改歌词。系统需要生成新的 4 秒内容，但这 4 秒必须满足：

- 进入时不能突然变响或变小；
- 人声音色不能突然变成另一个人；
- 节拍不能偏快或偏慢；
- 旋律不能突然跳到不该去的音高；
- 34-35 秒和 39-40 秒的边界不能有明显断裂。

所以“衔接自然”的核心不是“生成一个好听片段”，而是“生成一个能和左右上下文接上的片段”。

### 2.2 音频补全：把被挖掉的一段补回来

这里的“补全”指的是：把原音频中间某一小段拿掉，让模型根据前后内容补出中间这段。

这和我们的项目很像，因为我们做局部编辑时，本质上也是：

1. 把原来要改的片段拿掉；
2. 生成一个新片段；
3. 把新片段放回原位置；
4. 要求前后接得自然。

#### MAID

**MAID: A Conditional Diffusion Model For Long Music Audio Inpainting**，ICASSP 2023。

它解决的问题是：音乐中间缺了一段，如何根据前后内容补回来。

这里的 **扩散模型** 可以简单理解成一种生成模型：它先从很乱的噪声开始，逐步还原成听起来像音乐的音频。  
这里的 **音频补全** 就是“音频中间缺一段，让模型补上”。

MAID 还可以用 **钢琴卷帘** 作为条件。钢琴卷帘不是声音，而是一种“时间 - 音高”表格：横轴是时间，纵轴是音高，哪个时间点有哪个音就标出来。它告诉模型“这段应该弹哪些音”。

对项目的用处：

- 可以作为“局部缺口补全”的基础参考；
- 适合研究改完片段如何和前后内容接上；
- 但它主要是补缺口，不是专门做歌词替换。

#### Token-Based Audio Inpainting via Discrete Diffusion

**Token-Based Audio Inpainting via Discrete Diffusion**，arXiv 2025，后续收录到 ICLR 2026。

它解决的问题也是音频补全，但做法不同。它先把音频压缩成一串 **token**。这里的 token 可以理解成“音频编码后的基本单位”，类似把连续声音变成一串离散符号。模型不是直接改波形，而是在这些符号上补缺口。

这样做的好处是：对较长缺口更容易保持整体结构。论文还加入了平滑约束，让补出来的片段不要和前后突然断开。

对项目的用处：

- 如果我们要改 3 秒、5 秒甚至更长片段，可以参考这种“先编码再补全”的方式；
- 重点参考它如何避免补出来的内容和前后不连贯；
- 局限是它仍偏“补缺口”，不直接解决“改歌词但保音色”。

### 2.3 歌声局部改词：更接近我们项目

音频补全只是基础。我们项目更常见的任务是：不是随便补一段，而是要把歌词改成指定内容，同时保持旋律、时长和音色。

#### MeloDISinger

**MeloDISinger: Melody-Aware & Duration-Preserving Singing Voice Editing with Audio Infilling**，arXiv 2026，Interspeech 2026 接收。

它解决的问题是：改歌声里的某几个字或某一句歌词，但不破坏原来的旋律和时长。

这里的 **保持时长** 指的是：原来这一句唱 4 秒，改完后也尽量还是 4 秒。否则和伴奏就对不上。  
这里的 **音频填充** 和音频补全类似，意思是只重新生成被编辑的区域，周围没改的部分尽量保持原样。

对项目的用处：

- 很适合我们第一阶段做“指定一句歌词替换”；
- 可以参考它怎么让新歌词塞进原来的时间长度；
- 可以参考它怎么只改编辑区域，尽量不动前后区域；
- 局限是它主要处理人声，不直接处理伴奏和整曲混音。

#### Audio Editing with Non-Rigid Text Prompts

**Audio Editing with Non-Rigid Text Prompts**，Interspeech 2024。

它解决的问题是：用文本指令编辑音频时，既要实现目标变化，又要保住原音频的主要内容。

这里的“不那么死板的文本指令”可以理解成：不是只写一个固定标签，而是用更自由的文本描述编辑目标。比如不是只写“钢琴”，而是写“让这段更明亮、更有空间感”。

对项目的用处：

- 适合作为“文本驱动局部编辑”的参考；
- 可用于风格调整、补音、局部修改；
- 但它不专门解决“歌词精确替换”和“多轨同步”。

#### AUDEDIT

**AUDEDIT: Inversion-Free Text-Guided Editing with Pretrained Audio Flow Models**，arXiv 2026。

它解决的问题是：很多音频编辑方法会先把原音频反推回模型内部表示，再进行修改。这个反推过程容易带来误差，导致编辑后原来的节奏、音色或结构被破坏。AUDEDIT 试图绕开这一步。

这里的 **不做反推** 指的是：不先把原音频硬塞回模型内部，而是直接用模型已有的生成方向来改。  
这里的 **流模型** 可以简单理解成另一类生成模型，它学习如何把一种声音状态逐步变成另一种声音状态。

对项目的用处：

- 如果后续做真实歌曲编辑，这类方法值得关注；
- 它重点解决“编辑后原音频被带偏”的问题；
- 局限是它是通用音频编辑，不是专门为歌曲多轨编辑设计。

### 2.4 不只是最后接一下，模型也要尽量别改坏原结构

有些问题不是最后做平滑能解决的。比如模型生成的新片段节奏已经错了，后面再怎么淡入淡出也很难救。

所以除了边界处理，模型本身也要尽量保住原来的节奏、旋律和和声。

#### Melodia

**Melodia: Training-Free Music Editing Guided by Attention Probing in Diffusion Models**，AAAI 2026。

它研究的是：在扩散模型里，哪些内部机制会影响音乐结构保持。它发现某些注意力模块和节奏、旋律结构关系更大，所以编辑时只动必要部分，减少对原结构的破坏。

这里的 **注意力机制** 可以理解成模型在生成时“重点看哪些信息”。如果模型在编辑时乱改注意力，就可能把原来的节奏和旋律结构改坏。

对项目的用处：

- 提醒我们不要只看文本指令有没有满足，也要看原曲结构有没有被破坏；
- 可作为风格编辑时“保结构”的参考；
- 不适合直接作为歌词替换方案。

#### InstructME

**InstructME: An Instruction Guided Music Edit And Remix Framework with Latent Diffusion Models**，arXiv 2023。

它解决的问题是：按自然语言指令做音乐编辑和 remix 时，如何保住和声和长程结构。

这里的 **潜空间扩散** 指的是不直接在原始波形上生成，而是在模型压缩后的中间表示上生成或编辑。这样计算更方便，也更容易控制整体结构。  
这里的 **remix** 指的是重编配、调整乐器或混音结构，不一定是重新作曲。

对项目的用处：

- 适合参考“编辑时怎么保和声”；
- 对多轮编辑和乐器调整有参考价值；
- 但它不专门处理人声歌词替换。

#### Audio Prompt Adapter

**Audio Prompt Adapter: Unleashing Music Editing Abilities for Text-to-Music with Lightweight Finetuning**，ISMIR 2024。

它解决的问题是：已有文本生成音乐模型本来不擅长编辑已有音频，能不能用很小的改动让它具备编辑能力。

它给已有模型加一个轻量模块，让模型同时看输入音频和文本指令，从而做音色转换、风格转换、伴奏生成等任务。

对项目的用处：

- 适合做低成本 baseline；
- 可以验证“不训练大模型，只加小模块”是否够用；
- 局限是编辑精度和多轨协同能力有限。

### 2.5 衔接自然部分的小结

这一块可以拆成四个具体问题：

1. **补什么**：是补缺失音频，还是替换歌词、人声、伴奏。
2. **怎么保时长**：改完后长度不能乱，否则和伴奏对不上。
3. **怎么保上下文**：没改的前后部分不能被模型顺手改掉。
4. **怎么接边界**：新片段和原曲之间不能有响度、音色、节拍突变。

对项目最有用的验证任务是：

> 输入一段人声或歌曲，指定 3-8 秒局部歌词替换，输出改后的版本，并检查旋律、时长、音色和边界是否保持。

## 3. 痛点 B：多轨协同

### 3.1 这个痛点到底是什么

多轨协同指的是：一首歌里通常有多条轨道，比如人声、鼓、贝斯、吉他、钢琴、其他伴奏。我们对同一个时间片段做修改时，这些轨道不能各自为政，必须一起协调。

举例：

- 这个片段的人声改了，伴奏和和声不能冲突；
- 这个片段的鼓点改了，贝斯和节拍不能错位；
- 某个片段的人声变大，整体混音不能失衡；
- 某个片段的某条轨改完后，其他轨也要跟着保持配合。

所以多轨协同不是“把轨道分开再各改各的”，而是要保证时间、和声、响度和音色关系仍然协调。

### 3.2 第一步通常是先看清每条轨

**音轨分离** 指的是把一首混在一起的歌拆成几条轨，比如：

- vocals：人声；
- drums：鼓；
- bass：贝斯；
- other：其他伴奏。

这样做的目的很直接：如果我们要对一个时间片段里的多条轨做协调修改，先要知道这些轨大概各自是什么样。

#### HTDemucs

**Hybrid Transformers for Music Source Separation**，arXiv 2022，常被称为 HTDemucs。

它解决的问题是：如何从混合歌曲中分离出人声、鼓、贝斯和其他伴奏。

它同时利用两种信息：

- 时间域信息：直接看声音随时间怎么变化；
- 频域信息：看不同频率上的能量分布。

这样做能提高分离质量。论文在 MUSDB 等数据集上取得了很强的 SDR 指标。SDR 可以粗略理解为“分离出来的轨道和真实轨道有多接近”，数值越高越好。

对项目的用处：

- 可以作为多轨协同修改的基础工具；
- 先把一个片段里的各条轨看清楚，再考虑怎么一起改；
- 局限是分离结果不完美，可能有串音和漏音。

#### Sound Demixing Challenge 2023

**The Sound Demixing Challenge 2023 – Music Demixing Track**，TISMIR 2024。

它不是单个模型，而是一个比赛和总结，反映音乐源分离领域的主流水平。

它提到一个实际问题：训练数据和真实音乐里常有 label noise 和 bleeding。  
这里的 **label noise** 是指标注不完全准确。  
这里的 **bleeding** 是指一条轨里混进了其他轨的声音，比如人声轨里还有鼓或吉他残留。

对项目的用处：

- 提醒我们不能把分离结果当成完美轨道；
- 后续编辑和混回时要考虑分离误差；
- 否则编辑后的伪影会更明显。

### 3.3 只知道每条轨还不够，还要让它们一起改

如果只把人声、伴奏拆开，然后分别改，容易出现“各改各的”。更合理的是：在同一个时间片段里，让多条轨一起受到约束，一起修改，一起保持对齐和和声关系。

#### Multi-Source Diffusion Models

**Multi-Source Diffusion Models for Simultaneous Music Generation and Separation**，ICLR 2024。

它解决的问题是：能不能在同一个模型里同时处理“多轨生成”和“多轨分离”。

论文里有一个思路叫 **source imputation**。可以理解成：给模型一部分轨道，让它补出其他轨道。比如给人声，让模型生成匹配的人声伴奏；或者给伴奏，让模型补出匹配的人声。

对项目的用处：

- 和“一个时间片段里多条轨一起协调修改”很接近；
- 说明多轨关系最好在模型内部就考虑，而不是最后混音时才补救；
- 对我们最有价值的是它的“多轨一起建模”的思路。

#### Music ControlNet

**Music ControlNet: Multiple Time-varying Controls for Music Generation**，arXiv 2023。

它解决的问题是：只用一句文本很难精确控制音乐。比如“更有节奏感”太模糊，模型不知道具体哪个时间点该强、哪个时间点该弱。

它引入了随时间变化的控制条件，包括：

- melody：旋律；
- rhythm：节奏；
- dynamics：强弱变化。

这些条件会告诉模型“每个时间段应该怎么走”，比一句文本更精确。

对项目的用处：

- 多轨协同需要这种时间级控制；
- 比如一个片段里多条轨要同时变化，就不能只靠文本；
- 需要把节拍、旋律、力度这些信息显式喂给模型。

#### Editing Music with Melody and Text

**Editing Music with Melody and Text: Using ControlNet for Diffusion Transformer**，ICASSP 2025。

它解决的问题是：如何同时用文本和旋律来控制音乐生成或编辑。

这里的 **ControlNet** 可以理解成给生成模型加一条“控制支路”。主模型负责生成声音，控制支路负责告诉它“旋律应该怎么走”或“风格应该是什么”。  
这里的 **DiT** 是一种把扩散模型和 Transformer 结合的生成模型。  
这里的 **CQT** 是一种频率表示方式，比普通频谱更适合表达音乐里的音高关系。

对项目的用处：

- 如果我们要让同一时间片段里的多条轨一起变化，这类方法很重要；
- 如果我们要改旋律，同时让其他轨跟得上，也需要类似机制；
- 局限是它不等于完整歌曲多轨编辑系统。

### 3.4 已经有更接近歌曲编辑的系统

#### InstructME

**InstructME: An Instruction Guided Music Edit And Remix Framework with Latent Diffusion Models**，arXiv 2023。

它支持按指令做乐器编辑、remix 和多轮编辑。

对项目的用处：

- 可以参考它如何把“编辑指令”和“音乐结构保持”结合；
- 适合研究多次修改后如何避免整首歌越来越乱；
- 但它不是专门面向“一个片段里多条轨同时协调修改”的完整方案。

#### SongEditor

**SongEditor: Adapting Zero-Shot Song Generation Language Model as a Multi-Task Editor**，AAAI 2025。

它解决的问题很贴近我们项目：已有歌曲生成模型可以生成歌曲，但不擅长对已有歌曲做局部修改。SongEditor 把歌曲生成模型改造成多任务编辑器。

它支持两类编辑：

- segment-wise editing：片段级编辑，也就是改歌曲中的某一段；
- track-wise editing：分轨级编辑，也就是改歌词、人声或伴奏这一类轨道。

对项目的用处：

- 它是“歌曲级多轨编辑”的重点参考；
- 可以重点看它怎么表示歌曲、怎么定位片段、怎么把片段里的多条轨一起改；
- 局限是复现成本较高，系统依赖 tokenizer、语言模型和扩散生成器。

#### VersBand

**VersBand: Versatile Framework for Song Generation with Prompt-based Control**，arXiv 2025。

它偏歌曲生成，但很重视不同轨道之间的对齐。它把歌声、伴奏、歌词、旋律拆成不同模块处理。

对项目的用处：

- 可以参考它怎么把不同轨道作为相关模块处理；
- 说明多轨协同不能只放到最后混音阶段；
- 局限是它主要是生成，不是编辑已有真实歌曲。

### 3.5 多轨协同部分的小结

这一块可以拆成四个具体问题：

1. **怎么看清每条轨**：先知道一个片段里每条轨大概在做什么。
2. **怎么一起改**：不是只改某一条轨，而是让片段里的多条轨一起变化。
3. **怎么处理分离误差**：拆出来的轨不一定干净，可能有串音。
4. **怎么混回去**：检查时间对齐、响度、和声是否还合理。

对项目最有用的验证任务是：

> 输入一首含人声歌曲，对其中一个时间片段里的多条轨一起做协调修改，再检查这些轨之间是否还对齐、还协调。

### 3.6 为什么这类文献看起来少

你看到的情况基本属实：**直接把“多轨协同”写成“编辑”的论文不多**。原因主要有三点：

1. **数据难**：真正对齐好的多轨歌曲数据不多，尤其是“同一首歌的可编辑版本”更少。
2. **任务难拆**：多轨问题通常先被拆成“分离”“生成伴奏”“生成人声”“安排编配”几个子任务。
3. **评估难**：很难像图像任务那样直接说改哪里就改哪里，音乐里还要看节拍、和声、响度和整体听感。

所以，很多论文虽然没有直接写“多轨编辑”，但其实已经在解决“多轨怎么协调”的关键部件。对我们来说，应该把这些论文当成协同机制来源，而不是只看标题里有没有 editing。

### 3.7 更贴近“协同”的几篇补充论文

#### Multi-Track MusicLDM

**Multi-Track MusicLDM: Towards Versatile Music Generation with Latent Diffusion Model**，arXiv 2024。

它解决的问题是：多条乐器轨怎么一起生成，并且彼此听起来是配套的。

论文里明确说，多轨编曲本身就要求每一轨和别的轨在节拍、力度、和声、旋律上相互匹配。它把多轨放在一个联合模型里学，让模型学习“共享上下文”的轨道关系，还支持给定一部分轨道、补另外一部分轨道。

对项目的用处：

- 这是“多轨怎么一起配合”的直接建模；
- 和我们要做的“一个时间片段里多条轨一起协调修改”很接近；
- 重点参考它怎么做 arrangement generation，也就是“给定一些轨道，补出配套轨道”。

#### AccompGen

**AccompGen: Hierarchical Autoregressive Vocal Accompaniment Generation with Dual-Rate Codec Tokenization**，arXiv 2026。

它解决的问题很直接：给定人声，如何生成能直接混进去的伴奏。

这里的 **vocal-conditioned** 指的是“伴奏以人声为条件生成”。  
这里的 **time-aligned** 指的是伴奏和人声在时间上要对齐。  
它把人声和伴奏放在不同速率的编码里建模，再用分层自回归方式一步步生成伴奏。

对项目的用处：

- 如果我们做的是“一个时间片段里的人声和伴奏一起协调修改”，这篇的对齐思路很有参考价值；
- 它说明多轨协调不只是同时生成，而是要让不同轨道在时间上和结构上互相配合；
- 也能参考它对齐时间和不同采样率的处理方式。

#### MIDI-Informed Singing Accompaniment Generation in a Compositional Song Pipeline

**MIDI-Informed Singing Accompaniment Generation in a Compositional Song Pipeline**，arXiv 2026。

它解决的问题是：连续歌唱和间歇歌唱的场景里，伴奏怎么跟着人声和旋律一起变。

论文里强调两点：

- 伴奏要和 **symbolic vocal-melody MIDI** 对齐，也就是人声旋律的符号表示；
- 对于“有唱和没唱”交替出现的段落，要用明确的节奏和和声控制，让 backing track 在前后段都保持一致。

对项目的用处：

- 这篇特别适合“一个时间片段内有人声段、伴奏段、过门段一起变化”的真实歌曲结构；
- 它告诉我们多轨协同不仅是对齐，还要处理“有声段”和“无声段”之间的连续性。

#### Sing-On-Your-Beat

**Sing-On-Your-Beat: Simple Text-Controllable Accompaniment Generations**，arXiv 2024。

它解决的问题是：如何根据人声和文本控制生成伴奏，让伴奏既跟着歌，又符合想要的乐器和风格。

它的重点是简单但有效的文字控制，不只是生成“能听”的伴奏，而是让伴奏更贴合输入人声和目标风格。

对项目的用处：

- 可以作为“伴奏跟人声协调”的轻量参考；
- 适合看文本控制如何作用到伴奏生成；
- 但它更多是生成伴奏，不是编辑已有多轨。

#### Efficient Vocal-Conditioned Music Generation via Soft Alignment Attention and Latent Diffusion

**Efficient Vocal-Conditioned Music Generation via Soft Alignment Attention and Latent Diffusion**，arXiv 2025。

它解决的问题是：给定人声，怎样生成和它配得上的伴奏，同时让模型足够轻量、足够快。

它提出 **soft alignment attention**，可以理解成“一个软性的对齐注意力机制”，让模型自己决定哪些地方要紧跟人声，哪些地方可以更自由地展开。

对项目的用处：

- 可以作为“人声驱动伴奏生成”的轻量基线；
- 对实时或交互式编辑更有参考价值；
- 重点在于“跟人声对齐”，不是单纯把伴奏单独生成出来。

#### BandControlNet

**BandControlNet: Parallel Transformers-based Steerable Popular Music Generation with Fine-Grained Spatiotemporal Features**，arXiv 2024。

它解决的问题是：多轨流行音乐怎么在结构、节拍、和声上一起受控生成。

它用 **Cross-Track Transformer** 去学不同轨道之间的关系，用 **structure-enhanced self-attention** 去保结构。

对项目的用处：

- 这是“多轨之间怎么互相协调”的非常直接的机制；
- 它不是靠最后混音补救，而是在模型里显式建轨道关系；
- 如果我们后面做多轨协同，这类“跨轨交互”很值得参考。

## 4. 项目可执行建议

### 4.1 可以先作为 baseline 的工作

这里的 **baseline** 指的是“先拿来做对照的基础方法”，不是最终方案。

衔接自然：

- MAID：看音频补全怎么做；
- Token-Based Audio Inpainting：看较长缺口怎么补；
- MeloDISinger：看改词时怎么保旋律和时长；
- AUDEDIT：看真实音频编辑怎么减少原结构跑偏。

多轨协同：

- HTDemucs：先把一个片段里的各条轨看清楚；
- Multi-Track MusicLDM：看多条轨怎么一起配合；
- Music ControlNet：看怎么用旋律、节奏、强弱控制生成；
- SongEditor：看歌曲级片段编辑和分轨编辑。

### 4.2 系统里应该保留的机制

衔接自然部分：

- 局部编辑，不整首重做；
- 保时长；
- 保未编辑区域；
- 边界平滑；
- 音高、音色、节拍、响度检查。

多轨协同部分：

- 先看清同一片段里的各条轨；
- 对这个片段里的多条轨一起协调修改；
- 混回前检查时间对齐；
- 混回后检查音量和和声；
- 不把分离结果当成完美真值。

### 4.3 目前不宜直接承诺

- 任意真实歌曲都能无痕编辑；
- 完全没有误差的多轨对齐；
- 不分轨、不后处理，直接端到端改完整首歌；
- 只靠一句文本就精确完成专业音乐编辑。

## 5. 下一步建议

下一轮建议只精读 6 篇，不要继续无限扩：

衔接自然：

1. MAID
2. Token-Based Audio Inpainting
3. MeloDISinger

多轨协同：

1. HTDemucs
2. Multi-Track MusicLDM
3. SongEditor

精读时每篇只回答五个问题：

1. 它解决的具体问题是什么？
2. 它输入什么、输出什么？
3. 它怎么保证局部不乱？
4. 它怎么处理多轨一起变化？
5. 它能不能直接放进我们的项目流程？

## 参考

- [MAID](https://ieeexplore.ieee.org/abstract/document/10095769)
- [Token-Based Audio Inpainting via Discrete Diffusion](https://arxiv.org/abs/2507.08333)
- [MeloDISinger](https://papers.cool/arxiv/2606.30580)
- [Audio Editing with Non-Rigid Text Prompts](https://www.isca-archive.org/interspeech_2024/paissan24b_interspeech.html)
- [AUDEDIT](https://arxiv.org/abs/2606.15149)
- [Melodia](https://papers.cool/arxiv/2511.08252)
- [InstructME](https://arxiv.org/abs/2308.14360)
- [Audio Prompt Adapter](https://arxiv.org/abs/2407.16564)
- [HTDemucs](https://arxiv.org/abs/2211.08553)
- [Sound Demixing Challenge 2023 – Music Demixing Track](https://transactions.ismir.net/articles/10.5334/tismir.171)
- [Multi-Source Diffusion Models for Simultaneous Music Generation and Separation](https://arxiv.org/abs/2302.02257)
- [Music ControlNet](https://arxiv.org/abs/2311.07069)
- [Editing Music with Melody and Text](https://arxiv.org/abs/2410.05151)
- [SongEditor](https://arxiv.org/abs/2412.13786)
- [VersBand](https://arxiv.org/abs/2504.19062)
- [Multi-Track MusicLDM](https://arxiv.org/abs/2406.07985)
- [AccompGen](https://arxiv.org/abs/2601.10132)
- [MIDI-Informed Singing Accompaniment Generation in a Compositional Song Pipeline](https://arxiv.org/abs/2604.08218)
- [Sing-On-Your-Beat](https://arxiv.org/abs/2401.14742)
- [Efficient Vocal-Conditioned Music Generation via Soft Alignment Attention and Latent Diffusion](https://arxiv.org/abs/2501.06381)
- [BandControlNet](https://arxiv.org/abs/2404.10365)

