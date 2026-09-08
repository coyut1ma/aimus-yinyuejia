# 多轨协同编辑服务器上传包

## 目录说明

- `ACE-Step-1.5/`：ACE-Step 1.5 源码、依赖配置、训练与服务入口。
- `music_edit_demo/`：端到端多轨编辑 demo 的源码和测试。
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
