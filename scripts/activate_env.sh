#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# BioChemInsight conda activation hook
#
# PURPOSE
#   Ensures that cuDNN and CUDA shared libraries installed inside the conda
#   environment are visible to TensorFlow, PyTorch, and PaddlePaddle at
#   runtime.  Without this, TensorFlow cannot find cuDNN even when it is
#   present in the environment's lib/ directory.
#
# INSTALLATION (run once per environment)
#
#   For a conda environment named  chem_ocr :
#
#     CONDA_PREFIX=$(conda run -n chem_ocr python -c "import sys; print(sys.prefix)")
#     mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
#     cp scripts/activate_env.sh "$CONDA_PREFIX/etc/conda/activate.d/biocheminsight.sh"
#     chmod +x "$CONDA_PREFIX/etc/conda/activate.d/biocheminsight.sh"
#
#   After that, every  `conda activate chem_ocr`  will source this file
#   automatically.
#
# MANUAL USE (without conda hooks)
#
#   source scripts/activate_env.sh
#
# ---------------------------------------------------------------------------

# Guard: only run if CONDA_PREFIX is set (i.e., a conda env is active).
if [ -z "${CONDA_PREFIX:-}" ]; then
    CONDA_PREFIX="$(python -c 'import sys; print(sys.prefix)' 2>/dev/null || true)"
fi

if [ -n "${CONDA_PREFIX}" ]; then
    # Prepend the environment's lib directory so cuDNN/CUDA .so files are found
    # before any system-wide copies.
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

# Also include common CUDA installation paths used by NVIDIA driver packages.
for _cuda_lib in \
    /usr/local/cuda/lib64 \
    /usr/local/nvidia/lib \
    /usr/local/nvidia/lib64 \
    /usr/local/lib \
    /usr/lib/x86_64-linux-gnu
do
    if [ -d "${_cuda_lib}" ]; then
        export LD_LIBRARY_PATH="${_cuda_lib}:${LD_LIBRARY_PATH:-}"
    fi
done
unset _cuda_lib
