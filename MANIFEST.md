# 上传包清单与排除规则

## 已包含

- ACE-Step 核心 Python 源码与测试
- ACE-Step `pyproject.toml`、`uv.lock`、requirements、Docker 配置和常用 Linux 启动脚本
- Demo 的 `src/`、`tests/`、`README.md`、`pyproject.toml`
- `third_party/MuseCPEval/` 上下文保持评测工具源码
- 第一版多分枝模型设计与执行计划
- 多轨协同技术/通俗报告、讲解稿和 benchmark 规范
- Markdown 研究笔记

## 已排除

```text
*.pdf
*.pptx
*.zip
*.wav
*.mp4
*.pyc
.venv/
venv_xpu/
.cache/
__pycache__/
checkpoints/
runtime*/
output/
musdb18/
musdb18_preview/
```

数据集和 checkpoint 应放在服务器独立数据目录，通过配置路径引用。
