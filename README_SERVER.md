# 多轨协同编辑服务器上传包

## 目录说明

- `ACE-Step-1.5/`：ACE-Step 1.5 源码、依赖配置、训练与服务入口。
- `music_edit_demo/`：端到端多轨编辑 demo 的源码和测试。
- `third_party/MuseCPEval/`：可选的音乐上下文保持评测工具。
- `design_docs/`：多分枝第一版方案、结构报告和 benchmark 协议。
- `research_notes/`：项目已有 Markdown 调研和汇报材料；PDF/PPTX 已排除。

## 本包未包含

为控制上传体积，本包不包含：

- PDF 参考文献和 PPTX；
- MUSDB18 数据集及压缩包；
- ACE-Step checkpoint；
- Python 虚拟环境；
- `.cache`、`__pycache__`；
- demo 的 `runtime*`、生成音频和评测产物；
- ACE-Step 的 `output/`；
- 原始 ZIP 压缩包。

## 服务器上需要另行准备

1. 根据 `ACE-Step-1.5/README.md` 安装系统依赖、CUDA/PyTorch 和项目依赖。
2. 下载所需 ACE-Step checkpoint，并按 ACE-Step 配置放置或指定模型路径。
3. 将 MUSDB18 放到服务器数据盘，不要复制进源码目录。
4. 通过环境变量把 demo 指向 MUSDB18 和 ACE-Step 服务。
5. 如需 MuseCPEval 上下文保持评测，在 music 环境中执行
   `python -m pip install -e ../third_party/MuseCPEval`。

示例：

```bash
cd ACE-Step-1.5
uv sync

cd ../music_edit_demo
python -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[api]" --no-build-isolation
```

运行 demo 前设置：

```bash
export MUSIC_EDIT_RUNTIME_DIR=/data/music-edit-runtime
export MUSDB_ROOT=/data/musdb18
export MUSIC_EDIT_GENERATOR=ace_step_http
export ACE_STEP_BASE_URL=http://127.0.0.1:8000
```

## 第一版开发入口

模型改造方案见：

```text
design_docs/多分枝ACE-Step第一版设计方案及执行计划.md
```

## 多分枝消融对照

若要运行“与多分枝第一版基本相同、但关闭 Cross-Stem Attention”的对照，使用同一个本地
ACE-Step pilot repaint 脚本并增加 `--disable-cross-stem-attention`：

```bash
cd ACE-Step-1.5
../.conda-env/bin/python scripts/run_multistem_pilot_repaint.py \
  --manifest ../music_edit_demo/runtime_real/pilot_tasks_v1.jsonl \
  --cache-root ../music_edit_demo/runtime_real/musdb_cache_fulltest_20260909 \
  --output-dir ../outputs/multistem_adapter/pilot20_joint_no_cross_stem \
  --adapter-checkpoint ../outputs/multistem_adapter/run_long_20260908_2125/adapter_latest.pt \
  --disable-cross-stem-attention \
  --evaluate
```

该路径仍使用四轨上下文、共享 Block 1、目标轨选择、同样的 splice/mix/evaluation；
与多分枝第一版的主要差别是跳过 Cross-Stem Attention 残差交换。

生成完成后，用 MuseCPEval 对比两个输出目录的编辑边界：

```bash
../.conda-env/bin/python scripts/compare_pilot_musecpeval_boundaries.py \
  --left-root ../outputs/multistem_adapter/pilot20_joint_adapter_fulltest_20260909 \
  --right-root ../outputs/multistem_adapter/pilot20_joint_no_cross_stem \
  --left-label multibranch_v1 \
  --right-label demo_no_cross_stem \
  --output-dir ../outputs/musecpeval_boundary_compare_no_cross
```

实施顺序：

1. 确认 ACE-Step 第一个 DiT block 的真实输入输出接口。
2. 将四条 stem 编码并组织为 `[B,4,T,C]`。
3. 加入简单 stem embedding。
4. 四轨共享原 Block 1。
5. 在 Block 1 后加入一次 Cross-Stem Attention。
6. 选择目标 stem 后接回原 Block 2...N 和原输出头。
7. 新增模块 gate 置零，先做无训练兼容性测试。
8. 冻结 ACE-Step，只训练 stem embedding 和 Cross-Stem Attention。

## 上传前检查

运行以下命令确认包内没有大文件或敏感配置：

```bash
find . -type f -size +100M -print
find . -type f \( -name '*.pdf' -o -name '*.zip' -o -name '*.wav' -o -name '*.pyc' \) -print
```

`.env`、API key 和私有模型地址不应上传；服务器上从 `.env.example` 单独配置。
