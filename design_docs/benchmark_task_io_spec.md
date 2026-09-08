# 音乐可控编辑 Benchmark：任务与输入输出规范

**版本**：v1.0  
**数据集**：MUSDB18 及项目补充标注  
**基线**：ACE-Step 1.5、DiffRhythm 2

## 1. Benchmark 的基本单位

Benchmark 的一个样本不是一首普通歌曲，而是一条**编辑任务**：

```text
给定原始歌曲、可选的分轨和编辑指令，系统输出编辑后的完整歌曲/分轨。
```

每条任务必须明确：

- 修改什么（`must_change`）；
- 保持什么（`must_preserve`）；
- 修改哪一段、哪一轨；
- 输出什么格式；
- 哪些指标适用。

## 2. 任务类型

### T1：局部片段重生成

修改指定时间片段，保持片段外内容和音乐结构自然。

- 典型指令：将 42–58 秒副歌改得更欢快。
- `must_change`：指定片段中的目标属性。
- `must_preserve`：非编辑区、节拍、段落位置、指定音色等。
- 主要评测：音频 MOS、衔接自然度、节拍、段落边界、音量。

### T2：歌词/句子级编辑

修改指定歌词句子或音节对应的人声片段。

- 典型指令：将第二句歌词替换为给定文本。
- `must_change`：指定歌词和对应演唱内容。
- `must_preserve`：其他歌词、节拍、曲风、非编辑区。
- 主要评测：歌词正确率、歌词—人声对齐误差、衔接 MOS。

### T3：旋律编辑

修改指定片段的旋律，同时保持歌词、节拍或风格约束。

- 典型指令：保持歌词不变，将副歌旋律改得更有上行感。
- `must_change`：旋律。
- `must_preserve`：歌词、节奏、指定音色和非编辑区。
- 主要评测：语义/意图成功率、节拍、音频 MOS、音色一致性。

### T4：和声/调式编辑

修改指定片段的和弦、调式或和声进行。

- 典型指令：将副歌改为小调和声。
- `must_change`：目标和弦或调式。
- `must_preserve`：歌词、段落、节拍和非编辑区。
- 主要评测：输出与**目标和弦标注**的 WCSR，而不是与原曲和弦比较。

### T5：风格/情绪编辑

修改曲风、情绪或主要乐器表现。

- 典型指令：将片段改成更欢快的摇滚风格，加入电吉他。
- `must_change`：风格、情绪或指定乐器。
- `must_preserve`：编辑范围、歌词、节拍和必要的主体音色。
- 主要评测：专家风格/情绪匹配度、乐器匹配度、音频 MOS。

### T6：单轨编辑

只修改一个 stem，例如 vocals、drums、bass 或 other。

- 典型指令：只重写鼓轨，其他轨道不变。
- `must_change`：指定 stem。
- `must_preserve`：其他 stem、整体节拍和多轨同步。
- 主要评测：stem 音质 MOS、对齐误差、音量偏差、非目标轨保持率。

### T7：多轨协同编辑

同时修改两个或多个 stem，要求修改后仍然同步、平衡、可混音。

- 典型指令：同时修改人声和伴奏，保持节拍一致。
- `must_change`：指定多个 stem 的目标属性。
- `must_preserve`：跨轨时间关系、段落结构和指定音色。
- 主要评测：各轨 onset 对齐、LUFS 偏差、分轨 MOS、整体 MOS。

## 3. 输入定义

推荐使用 JSON 描述一条任务：

```json
{
  "task_id": "musdb18_001_t1_0001",
  "song_id": "musdb18_001",
  "task_type": "local_regeneration",
  "input_mix": "mix.wav",
  "input_stems": {
    "vocals": "vocals.wav",
    "drums": "drums.wav",
    "bass": "bass.wav",
    "other": "other.wav"
  },
  "edit_region": {
    "start_sec": 42.0,
    "end_sec": 58.0
  },
  "instruction": "将副歌改得更欢快，保持原歌词和人声音色",
  "must_change": ["emotion"],
  "must_preserve": [
    "lyrics",
    "tempo",
    "vocal_timbre",
    "non_edit_region"
  ],
  "reference": {
    "beat_times": "beats.json",
    "section_boundaries": "sections.json",
    "chords": "chords.lab",
    "lyrics_alignment": "lyrics.json"
  }
}
```

### 必填输入字段

| 字段 | 类型 | 含义 |
|---|---|---|
| `task_id` | string | 任务唯一编号 |
| `song_id` | string | MUSDB18 歌曲编号 |
| `task_type` | enum | T1–T7 之一 |
| `input_mix` | audio | 原始混音；多轨任务必填 |
| `edit_region` | object | 编辑起止时间，单位秒 |
| `instruction` | string | 自然语言编辑指令 |
| `must_change` | list | 必须发生变化的属性 |
| `must_preserve` | list | 必须保持的属性 |

### 条件输入字段

| 字段 | 适用任务 | 用途 |
|---|---|---|
| `input_stems` | 单轨/多轨编辑 | 提供 vocals、drums、bass、other |
| `target_lyrics` | 歌词编辑 | 指定替换后的歌词 |
| `target_chords` | 和声编辑 | 指定目标和弦或调式 |
| `target_style` | 风格编辑 | 结构化风格标签 |
| `target_emotion` | 情绪编辑 | 结构化情绪标签 |
| `reference` | 自动评测 | 提供 beat、段落、和弦、歌词对齐真值 |

## 4. 输出定义

系统必须返回完整歌曲；如果任务涉及分轨，还必须返回编辑后的分轨。

```json
{
  "task_id": "musdb18_001_t1_0001",
  "status": "success",
  "edited_mix": "edited_mix.wav",
  "edited_stems": {
    "vocals": "edited_vocals.wav",
    "drums": "edited_drums.wav",
    "bass": "edited_bass.wav",
    "other": "edited_other.wav"
  },
  "metadata": {
    "sample_rate": 44100,
    "channels": 2,
    "duration_sec": 180.0,
    "seed": 1234,
    "latency_sec": 8.6
  },
  "error": null
}
```

### 输出要求

1. `edited_mix` 的时长应与输入歌曲一致，除非任务明确允许改变时长；
2. 音频采样率、声道数和文件格式必须符合协议；
3. 编辑区外原则上保持不变；若系统修改了非编辑区，必须记录；
4. 单轨/多轨任务必须输出指定 stem，且 stem 时长一致；
5. 不得通过单独响度归一化掩盖真实音量偏差；
6. 必须记录随机种子、推理时延和失败原因；
7. `status` 只能为 `success` 或 `failed`。

## 5. 保持与改变约束

每条任务都必须包含 `must_change` 和 `must_preserve`。二者不能出现逻辑冲突。

| 任务 | 必须改变示例 | 必须保持示例 |
|---|---|---|
| 局部重生成 | 风格/情绪/局部内容 | 非编辑区、节拍 |
| 歌词编辑 | 目标歌词、人声内容 | 其他歌词、曲风 |
| 旋律编辑 | 旋律 | 歌词、节拍 |
| 和声编辑 | 和弦/调式 | 歌词、段落 |
| 风格编辑 | 曲风/情绪/乐器 | 编辑范围、必要音色 |
| 单轨编辑 | 指定 stem | 其他 stem、同步 |
| 多轨编辑 | 指定多个 stem | 跨轨时间关系、整体平衡 |

## 6. 失败和不适用规则

以下情况计为任务失败：

- 输出文件缺失、无法解码或时长严重错误；
- 未返回要求的 stem；
- 超过规定时间或显存限制；
- 输出与输入完全相同，但任务要求必须改变；
- 发生严重爆音、断裂或无法播放。

以下情况标记为 `N/A`，不强行比较：

- 基线不支持该任务类型；
- 任务未提供所需真值，例如没有目标和弦就不计算目标和声准确率；
- 任务本身允许改变某项属性，例如和声编辑不使用“保持原和声”指标。

## 7. 数据划分与统一协议

- 按歌曲或歌手划分 `dev/test/hidden-test`，同一歌曲不得跨集合；
- 固定采样率、声道、随机种子、采样次数、硬件和最大时延；
- 固定外部数据、额外模型和后处理权限；
- 自动标注的 beat、段落、和弦和歌词时间戳必须人工抽检并版本化；
- 每个模型都使用完全相同的任务 JSON、输入音频和评测脚本。

## 8. 任务与指标映射

| 任务 | 必评指标 | 可选指标 |
|---|---|---|
| T1 局部重生成 | MOS、衔接、节拍、音量、段落 | 和声、音色 |
| T2 歌词编辑 | 歌词正确率、歌词对齐、MOS | 音色、节拍 |
| T3 旋律编辑 | 意图成功率、MOS、节拍 | 和声、音色 |
| T4 和声编辑 | 目标 WCSR、意图成功率、MOS | 原曲 WCSR |
| T5 风格/情绪编辑 | 风格/情绪匹配、MOS | 乐器、音色 |
| T6 单轨编辑 | stem MOS、对齐、LUFS | 非目标轨保持 |
| T7 多轨编辑 | 各轨 MOS、对齐、LUFS、整体 MOS | 和声、段落 |

## 9. 最小可执行版本

如果项目需要先做小规模验证，建议优先实现：

1. T1、T2、T6、T7 四类任务；
2. JSON 输入/输出协议；
3. 音频质量 MOS、节拍误差、LUFS 偏差、多轨对齐四项自动/人工评测；
4. 20–50 首歌曲 pilot；
5. 冻结协议后再跑 ACE-Step 1.5、DiffRhythm 2 和项目模型。

