#!/usr/bin/env bash
# Configure workspace-local cache/temp/install paths.
#
# Usage:
#   source /workspace/workspace_env.sh
#   source /workspace/workspace_env.sh /workspace

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "This script must be sourced to persist environment variables."
  echo "Example: source ${BASH_SOURCE[0]} /workspace"
  exit 1
fi

workspace_root="${1:-/workspace}"
if [[ ! -d "${workspace_root}" ]]; then
  echo "Workspace path does not exist: ${workspace_root}" >&2
  return 1
fi

workspace_root="$(cd "${workspace_root}" && pwd)"

# Base locations
export XDG_CACHE_HOME="${workspace_root}/.cache"
export TMPDIR="${workspace_root}/tmp"
export PYTHONUSERBASE="${workspace_root}/.local"

# pip
export PIP_CACHE_DIR="${XDG_CACHE_HOME}/pip"

# Hugging Face / Transformers / Datasets
export HF_HOME="${XDG_CACHE_HOME}/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"

# PyTorch / Triton / CUDA compile caches
export TORCH_HOME="${XDG_CACHE_HOME}/torch"
export TORCH_EXTENSIONS_DIR="${XDG_CACHE_HOME}/torch_extensions"
export TORCHINDUCTOR_CACHE_DIR="${XDG_CACHE_HOME}/torchinductor"
export TRITON_CACHE_DIR="${XDG_CACHE_HOME}/triton"
export CUDA_CACHE_PATH="${XDG_CACHE_HOME}/nv"

# Common tool caches
export MPLCONFIGDIR="${XDG_CACHE_HOME}/matplotlib"
export WANDB_DIR="${workspace_root}/wandb"
export WANDB_CACHE_DIR="${XDG_CACHE_HOME}/wandb"

# Create all directories
mkdir -p \
  "${TMPDIR}" \
  "${PYTHONUSERBASE}" \
  "${PIP_CACHE_DIR}" \
  "${HF_HOME}" \
  "${HUGGINGFACE_HUB_CACHE}" \
  "${TRANSFORMERS_CACHE}" \
  "${HF_DATASETS_CACHE}" \
  "${TORCH_HOME}" \
  "${TORCH_EXTENSIONS_DIR}" \
  "${TORCHINDUCTOR_CACHE_DIR}" \
  "${TRITON_CACHE_DIR}" \
  "${CUDA_CACHE_PATH}" \
  "${MPLCONFIGDIR}" \
  "${WANDB_DIR}" \
  "${WANDB_CACHE_DIR}"

echo "Workspace environment configured for: ${workspace_root}"
echo "TMPDIR=${TMPDIR}"
echo "PIP_CACHE_DIR=${PIP_CACHE_DIR}"
echo "HF_HOME=${HF_HOME}"
echo "TRITON_CACHE_DIR=${TRITON_CACHE_DIR}"
echo "TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR}"


source venv/bin/activate
