FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3-pip \
        python3-dev \
        build-essential \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgl1 \
        wget \
        git \
        ccache \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

RUN python -m pip install --upgrade pip

# Install PaddleOCR and runtime dependencies
RUN pip install --no-cache-dir \
        paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/ && \
    pip install --no-cache-dir \
        paddleocr[all]==3.2.0 \
        paddlenlp>=2.7.2 \
        fastapi \
        uvicorn[standard] \
        python-multipart \
        PyMuPDF \
        numpy \
        langchain==0.3.27 \
        opencv-python-headless \
        pillow 

WORKDIR /workspace

# ---------------------------------------------------------------------------
# Patch paddlex processors.py: guard nms() calls against empty / 1-D box
# arrays that cause IndexError when no boxes are detected (Issue 4).
# ---------------------------------------------------------------------------
RUN python - <<'EOF'
import pathlib, re, sys
try:
    import paddlex
    pkg_dir = pathlib.Path(paddlex.__file__).parent
except ImportError:
    print("paddlex not installed, skipping nms patch"); sys.exit(0)

NMS_PATTERN = re.compile(
    r"^(\s*\w+\s*=\s*)(nms\([^)]+\))(\s*)$",
    re.MULTILINE,
)
for p in pkg_dir.rglob("processors.py"):
    text = p.read_text()
    new_text = NMS_PATTERN.sub(
        r"\1(\2 if boxes.ndim > 1 and boxes.shape[0] > 0 else [])\3",
        text,
    )
    if new_text != text:
        p.write_text(new_text)
        print(f"PATCHED {p}")
    else:
        print(f"OK (no changes needed) {p}")
EOF

COPY app /workspace/app

EXPOSE 8010

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8010", "--workers", "1"]
