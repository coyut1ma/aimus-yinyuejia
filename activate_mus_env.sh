#!/usr/bin/env bash
# Source from the repository root: source ./activate_mus_env.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CONDA_PREFIX="$SCRIPT_DIR/.conda-env"
export PATH="$CONDA_PREFIX/bin:$PATH"

export CONDA_PKGS_DIRS="$SCRIPT_DIR/.conda-pkgs"
export PIP_CACHE_DIR="$SCRIPT_DIR/.pip-cache"
export UV_CACHE_DIR="$SCRIPT_DIR/.uv-cache"
export HF_HOME="${HF_HOME:-$SCRIPT_DIR/.hf-cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$SCRIPT_DIR/.matplotlib-cache}"

export ACESTEP_CONFIG_PATH="${ACESTEP_CONFIG_PATH:-acestep-v15-turbo}"
export ACESTEP_INIT_LLM="${ACESTEP_INIT_LLM:-false}"
export ACESTEP_USE_FLASH_ATTENTION="${ACESTEP_USE_FLASH_ATTENTION:-false}"

export MUSDB_ROOT="${MUSDB_ROOT:-$SCRIPT_DIR/musdb18}"
export MUSIC_EDIT_RUNTIME_DIR="${MUSIC_EDIT_RUNTIME_DIR:-$SCRIPT_DIR/music_edit_demo/runtime_real}"
export MUSIC_EDIT_GENERATOR="${MUSIC_EDIT_GENERATOR:-ace_step_http}"
export ACE_STEP_BASE_URL="${ACE_STEP_BASE_URL:-http://127.0.0.1:8001}"
