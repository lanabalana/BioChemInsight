#!/usr/bin/env bash
# ==============================================================================
# BioChemInsight — Unattended setup script for VAST AI (PyTorch template)
# ==============================================================================
# Usage:
#   chmod +x setup.sh && ./setup.sh
#
# Assumes: mamba/conda available in PATH (PyTorch Vast template provides this)
# ==============================================================================
set -euo pipefail

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

step()  { echo -e "\n${CYAN}${BOLD}▶ $*${NC}"; }
ok()    { echo -e "${GREEN}✔  $*${NC}"; }
warn()  { echo -e "${YELLOW}⚠  $*${NC}"; }
die()   { echo -e "${RED}✘  ERROR: $*${NC}" >&2; exit 1; }

# ── Locate conda ──────────────────────────────────────────────────────────────
CONDA_BASE=""
for _try in \
    "$(conda info --base 2>/dev/null || true)" \
    /opt/conda \
    "$HOME/miniconda3" \
    "$HOME/anaconda3"
do
    if [ -f "${_try}/etc/profile.d/conda.sh" ]; then
        CONDA_BASE="$_try"
        break
    fi
done
[ -n "$CONDA_BASE" ] || die "Cannot find conda installation. Is the VAST AI PyTorch template active?"

step "Initialising conda shell integration from $CONDA_BASE"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Initialise mamba if available
if [ -f "$CONDA_BASE/etc/profile.d/mamba.sh" ]; then
    # shellcheck disable=SC1091
    source "$CONDA_BASE/etc/profile.d/mamba.sh"
fi

ok "conda initialised (version: $(conda --version))"

# ── Helper: run a command inside the chem_ocr env ─────────────────────────────
run_in_env() {
    conda run --no-capture-output -n chem_ocr "$@"
}

# ==============================================================================
# 1. Clone repository
# ==============================================================================
step "Cloning BioChemInsight repository"
if [ -d "BioChemInsight/.git" ]; then
    warn "BioChemInsight/ already exists — skipping clone."
    cd BioChemInsight
else
    git clone https://github.com/lanabalana/BioChemInsight.git
    cd BioChemInsight
fi
REPO_DIR="$(pwd)"
ok "Repository ready at $REPO_DIR"

# ==============================================================================
# 2. Create conda environment
# ==============================================================================
step "Creating conda environment 'chem_ocr' (Python 3.10)"
if conda env list | grep -qE '^chem_ocr\s'; then
    warn "Environment 'chem_ocr' already exists — skipping creation."
else
    mamba create -n chem_ocr python=3.10 -y
    ok "Environment created."
fi
conda activate chem_ocr
ok "Environment activated: $(which python)"

# ==============================================================================
# 3. Install PyTorch (CUDA 11.8)
# ==============================================================================
step "Installing PyTorch 2.7.1 (cu118)"
pip install torch==2.7.1 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu118
ok "PyTorch installed: $(python -c 'import torch; print(torch.__version__)')"

# ==============================================================================
# 4. Clone ChemSAM and install its Python dependencies
# ==============================================================================
step "Cloning ChemSAM (chemical structure segmentation model)"
CHEMSAM_DIR="$REPO_DIR/vendor/ChemSAM"
mkdir -p "$REPO_DIR/vendor"
if [ -d "$CHEMSAM_DIR/.git" ]; then
    warn "ChemSAM already cloned at $CHEMSAM_DIR — skipping."
else
    git clone https://github.com/mindrank-ai/ChemSAM.git "$CHEMSAM_DIR"
    ok "ChemSAM cloned to $CHEMSAM_DIR"
fi

step "Installing ChemSAM Python dependencies"
pip install scipy scikit-image imageio
ok "ChemSAM dependencies installed."

step "ChemSAM model checkpoint"
CKPT_DIR="$CHEMSAM_DIR/logs/chemseg_pix_sdg_2023_07_10_17_34_25/Model"
CKPT_FILE="$CKPT_DIR/last_660_checkpoint.pth"
mkdir -p "$CKPT_DIR"
if [ -f "$CKPT_FILE" ]; then
    ok "ChemSAM checkpoint already present."
else
    warn "ChemSAM checkpoint NOT found."
    warn "Contact the ChemSAM authors (mindrank-ai) to obtain last_660_checkpoint.pth"
    warn "and place it at:"
    warn "  $CKPT_FILE"
    warn "Setup will continue; the server will fail at startup until the checkpoint is present."
fi

# ==============================================================================
# 5. Install molscribe (SMILES recognition engine)
# ==============================================================================
step "Installing molscribe"
pip install molscribe
ok "molscribe installed."

# ==============================================================================
# 6. Re-install correct PyTorch (molscribe may downgrade it)
# ==============================================================================
step "Re-pinning PyTorch 2.7.1 (cu118) after molscribe install"
pip install torch==2.7.1 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu118
ok "PyTorch re-pinned: $(python -c 'import torch; print(torch.__version__)')"

# ==============================================================================
# 7. Install conda packages
# ==============================================================================
step "Installing jupyter, pytesseract, transformers via mamba/conda"
mamba install -c conda-forge jupyter pytesseract transformers -y
ok "conda packages installed."

# ==============================================================================
# 8. Install remaining pip packages
# ==============================================================================
step "Installing remaining pip packages"
pip install \
    PyMuPDF \
    PyPDF2 \
    openai \
    Levenshtein \
    mdutils \
    google-generativeai \
    tabulate \
    python-multipart \
    fastapi \
    uvicorn \
    modelscope \
    huggingface_hub
ok "pip packages installed."

# ==============================================================================
# 9. Install PaddlePaddle (GPU, CUDA 12.6 wheel)
# ==============================================================================
step "Installing PaddlePaddle-GPU"
pip install paddlepaddle-gpu \
    -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
ok "PaddlePaddle installed."

# ==============================================================================
# 10. Install paddlex and paddleocr
# ==============================================================================
step "Installing paddlex[ocr] and paddleocr 3.4.0"
pip install "paddlex[ocr]"
pip install paddleocr==3.4.0
ok "paddlex and paddleocr installed."

# ==============================================================================
# 11. Configure constants.py
# ==============================================================================
step "Generating constants.py from constants_example.py"

MOLVEC_JAR="$REPO_DIR/bin/molvec-0.9.9-SNAPSHOT-jar-with-dependencies.jar"
if [ ! -f "$MOLVEC_JAR" ]; then
    warn "MolVec JAR not found at expected path: $MOLVEC_JAR"
    warn "Update MOLVEC in constants.py manually after setup."
fi

python - <<PYEOF
import re, sys

src = open("constants_example.py").read()

# --- Gemini API key: replace placeholder with empty string + prominent comment
src = src.replace(
    "GEMINI_API_KEY = 'YOUR_GEMINI_API_KEY_HERE'",
    (
        "# TODO ──────────────────────────────────────────────────────────────────\n"
        "# ACTION REQUIRED: paste your Gemini API key below before starting the\n"
        "# backend service.  Leave it empty and the server will refuse to start.\n"
        "# ────────────────────────────────────────────────────────────────────────\n"
        "GEMINI_API_KEY = ''  # <-- fill in your key here"
    )
)

# --- MOLVEC path
molvec_path = "$MOLVEC_JAR"
src = re.sub(
    r"MOLVEC\s*=\s*'[^']*'",
    f"MOLVEC = '{molvec_path}'",
    src,
)

# --- OCR engine: enable paddleocr as default + uncomment server URL
src = src.replace(
    "DEFAULT_OCR_ENGINE = 'dots_ocr'",
    "DEFAULT_OCR_ENGINE = 'paddleocr'",
)
src = src.replace(
    "# PADDLEOCR_SERVER_URL = 'http://your_paddleocr_server:8010'",
    "PADDLEOCR_SERVER_URL = 'http://localhost:8010'",
)

open("constants.py", "w").write(src)
print("constants.py written.")
PYEOF

ok "constants.py created."
echo ""
echo "  MOLVEC  → $MOLVEC_JAR"
echo "  OCR     → paddleocr  (server at http://localhost:8010)"
echo "  Gemini  → (EMPTY — fill in constants.py before starting)"

# ==============================================================================
# 12. Set up LD_LIBRARY_PATH hook in conda activate.d
# ==============================================================================
step "Installing conda activate.d hook for cuDNN / LD_LIBRARY_PATH"

# Derive the real prefix of the active environment
ENV_PREFIX="$(python -c 'import sys; print(sys.prefix)')"
ACTIVATE_D="$ENV_PREFIX/etc/conda/activate.d"
mkdir -p "$ACTIVATE_D"

# Copy the project's own activation script
cp "$REPO_DIR/scripts/activate_env.sh" "$ACTIVATE_D/cudnn.sh"
chmod +x "$ACTIVATE_D/cudnn.sh"

ok "Hook installed at $ACTIVATE_D/cudnn.sh"
ok "LD_LIBRARY_PATH will be extended automatically on every 'conda activate chem_ocr'"

# Apply immediately for the rest of this session
# shellcheck disable=SC1090
source "$ACTIVATE_D/cudnn.sh"

# ==============================================================================
# 13. Build frontend
# ==============================================================================
step "Building frontend (npm install + npm run build)"

if ! command -v node &>/dev/null; then
    warn "node/npm not found in PATH. Attempting to install Node.js via conda..."
    conda install -c conda-forge nodejs -y
fi

cd "$REPO_DIR/frontend/ui"
npm install
npm run build
cd "$REPO_DIR"
ok "Frontend built successfully."

# ==============================================================================
# Done
# ==============================================================================
echo ""
echo -e "${GREEN}${BOLD}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}${BOLD}║          BioChemInsight setup complete!                      ║${NC}"
echo -e "${GREEN}${BOLD}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BOLD}Next steps:${NC}"
echo ""
echo -e "  ${YELLOW}1. Fill in your Gemini API key:${NC}"
echo -e "     ${CYAN}nano $REPO_DIR/constants.py${NC}"
echo -e "     (search for  GEMINI_API_KEY  and paste your key)"
echo ""
echo -e "  ${YELLOW}2. Activate the environment (new terminal / re-login):${NC}"
echo -e "     ${CYAN}conda activate chem_ocr${NC}"
echo ""
echo -e "  ${YELLOW}3. Start the three services (each in its own terminal):${NC}"
echo ""
echo -e "     ${BOLD}Service 1 — PaddleOCR server (port 8010):${NC}"
echo -e "     ${CYAN}conda activate chem_ocr && cd $REPO_DIR/DOCKER_PADDLE_OCR${NC}"
echo -e "     ${CYAN}uvicorn app.server:app --host 0.0.0.0 --port 8010${NC}"
echo ""
echo -e "     ${BOLD}Service 2 — Main backend API (port 8000):${NC}"
echo -e "     ${CYAN}conda activate chem_ocr && cd $REPO_DIR${NC}"
echo -e "     ${CYAN}uvicorn frontend.backend.main:app --host 0.0.0.0 --port 8000${NC}"
echo ""
echo -e "     ${BOLD}Service 3 — Frontend dev server (port 5173) — optional:${NC}"
echo -e "     ${CYAN}(the production build in frontend/ui/dist/ can be served by${NC}"
echo -e "     ${CYAN} any static-file server, e.g. 'npx serve frontend/ui/dist')${NC}"
echo ""
echo -e "  ${YELLOW}Tip:${NC} use tmux or screen to keep services alive after disconnect."
echo -e "       ${CYAN}tmux new -s biochem${NC}"
echo ""
