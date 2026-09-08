# 衔接自然 / 多轨协同 文献对照表

这张表用于后续继续补论文。每一列都尽量问具体问题，避免只写术语。

## 表格模板

| 论文 | 年份 / 出处 | 属于哪个痛点 | 它具体想解决什么 | 输入是什么 | 输出是什么 | 它怎么避免改坏原曲 | 它怎么让多个部分协同变化 | 论文怎么评估效果 | 对我们项目有什么用 | 局限 |
|---|---|---|---|---|---|---|---|---|---|---|

## 怎么填

### 1. 属于哪个痛点

只填两类：

- 衔接自然
- 多轨协同

如果两类都相关，就写“两者都相关”，但要说明主要相关哪一类。

### 2. 它具体想解决什么

不要只写“音频编辑”或“音乐生成”。要写成具体问题，例如：

- 音频中间缺一段，如何补回来；
- 改歌词后，如何保持原来的旋律和时长；
- 把混合歌曲拆成人声、鼓、贝斯和伴奏；
- 改人声时，如何让伴奏仍然对得上；
- 用文本和旋律一起控制音乐生成。

### 3. 输入是什么、输出是什么

这一列很重要，因为它能判断论文能不能直接接到我们项目里。

例子：

- 输入：原音频 + 被遮住的时间区域；输出：补全后的音频。
- 输入：原人声 + 新歌词；输出：改词后的人声。
- 输入：混合歌曲；输出：人声、鼓、贝斯、其他伴奏。
- 输入：伴奏 + 文本条件；输出：匹配伴奏的人声或完整歌曲。

### 4. 它怎么避免改坏原曲

重点看这些问题：

- 有没有只改局部；
- 有没有保持时长；
- 有没有保持旋律；
- 有没有保持未编辑区域；
- 有没有把其他轨作为参考；
- 有没有用节拍、音高、和声、响度做约束。

### 5. 它怎么让多个部分协同变化

衔接自然方向重点看：

- 前后边界有没有平滑；
- 有没有控制响度突变；
- 有没有控制音高跳变；
- 有没有控制音色变化；
- 有没有保留上下文。

多轨协同方向重点看：

- 同一个时间片段里的多条轨是不是一起改；
- 轨道之间有没有共享上下文；
- 旋律、节奏、强弱是不是一起受控；
- 混回后人声和伴奏是否还对齐；
- 是否考虑分离误差和串音。

## 重点论文清单

### 衔接自然

- MAID
- Token-Based Audio Inpainting via Discrete Diffusion
- MeloDISinger
- Audio Editing with Non-Rigid Text Prompts
- AUDEDIT
- Melodia
- InstructME
- Audio Prompt Adapter

### 多轨协同

- HTDemucs
- Sound Demixing Challenge 2023 – Music Demixing Track
- Multi-Source Diffusion Models for Simultaneous Music Generation and Separation
- Music ControlNet
- Editing Music with Melody and Text
- SongEditor
- VersBand
- Multi-Track MusicLDM
- AccompGen
- MIDI-Informed Singing Accompaniment Generation in a Compositional Song Pipeline
- Sing-On-Your-Beat
- Efficient Vocal-Conditioned Music Generation via Soft Alignment Attention and Latent Diffusion
- BandControlNet
