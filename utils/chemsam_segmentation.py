"""
ChemSAM segmentation wrapper — drop-in replacement for DECIMER Segmentation.

Provides:
    get_chemsam_segments(image_bgr) → (segments, bboxes, masks)
    warmup()                        → pre-initialise model on the main thread

Return types match what BioChemInsight's process_page() expects:
    segments : list[np.ndarray]         cropped BGR images (h, w, 3) uint8
    bboxes   : list[[x1, y1, x2, y2]]  pixel coords on the original page
    masks    : np.ndarray (H, W, N)     per-structure binary masks uint8

ChemSAM repo is expected at vendor/ChemSAM/ inside the BioChemInsight root.
Checkpoint: vendor/ChemSAM/logs/chemseg_pix_sdg_2023_07_10_17_34_25/Model/last_660_checkpoint.pth
"""

import os
import sys
import threading

import cv2
import numpy as np
import scipy.ndimage
import torch
import torchvision.transforms as transforms
from PIL import Image as PILImage
from skimage.morphology import binary_dilation, binary_erosion

# ---------------------------------------------------------------------------
# Locate the vendored ChemSAM source and inject it into sys.path
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CHEMSAM_DIR = os.path.join(_REPO_ROOT, "vendor", "ChemSAM")

if not os.path.isdir(_CHEMSAM_DIR):
    raise ImportError(
        f"ChemSAM not found at {_CHEMSAM_DIR}.\n"
        "Clone it with:\n"
        f"  git clone https://github.com/mindrank-ai/ChemSAM.git {_CHEMSAM_DIR}"
    )

if _CHEMSAM_DIR not in sys.path:
    sys.path.insert(0, _CHEMSAM_DIR)

import cfg as _chemsam_cfg
from utils import get_network as _get_network
from complete_structure import expand_masks as _expand_masks

# ---------------------------------------------------------------------------
# Checkpoint path (relative to _CHEMSAM_DIR)
# ---------------------------------------------------------------------------
_CHECKPOINT_REL = os.path.join(
    "logs", "chemseg_pix_sdg_2023_07_10_17_34_25",
    "Model", "last_660_checkpoint.pth",
)
_CHECKPOINT_ABS = os.path.join(_CHEMSAM_DIR, _CHECKPOINT_REL)

# ---------------------------------------------------------------------------
# Singleton model state — loaded once, reused across pages
# ---------------------------------------------------------------------------
_model_lock = threading.Lock()   # serialise GPU inference (not re-entrant)
_net = None
_device = None
_transform = None
_args = None

_PIX_CUT = 75          # threshold matching inferencing2.py
_STRUCT = np.ones((3, 3))  # connectivity structure for scipy.ndimage.label


def _load_model() -> None:
    """Initialise ChemSAM exactly once.  Must be called from the main thread."""
    global _net, _device, _transform, _args

    if not os.path.exists(_CHECKPOINT_ABS):
        raise FileNotFoundError(
            f"ChemSAM checkpoint not found: {_CHECKPOINT_ABS}\n"
            "Download last_660_checkpoint.pth from the ChemSAM authors and "
            f"place it at:\n  {_CHECKPOINT_ABS}"
        )

    args = _chemsam_cfg.parse_args()
    args.net = "sam_adaptered"
    args.image_size = 512
    args.out_size = 128
    args.distributed = False

    use_gpu = torch.cuda.is_available()
    args.gpu = use_gpu
    args.gpu_device = 0

    device = torch.device("cuda", 0) if use_gpu else torch.device("cpu")

    # When loading from the full fine-tuned checkpoint, sam_ckpt must be None
    # so get_network() builds an uninitialised skeleton that we then fill below.
    args.sam_ckpt = None
    args.loadSaved_point = _CHECKPOINT_ABS

    net = _get_network(args, args.net, use_gpu=use_gpu,
                       gpu_device=device, distribution=False)
    net.to(device)

    loc = f"cuda:{args.gpu_device}" if use_gpu else "cpu"
    checkpoint = torch.load(_CHECKPOINT_ABS, map_location=loc)
    net.load_state_dict(checkpoint["state_dict"], strict=True)
    net.eval()

    _net = net
    _device = device
    _args = args
    _transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])


def _ensure_loaded() -> None:
    if _net is None:
        _load_model()


# ---------------------------------------------------------------------------
# Internal helpers (ported from ChemSAM/inferencing2.py)
# ---------------------------------------------------------------------------

def _get_bounding_box(contour: np.ndarray):
    """Return [y1, x1, y2, x2] from an (N, 2) array of (row, col) pixel coords."""
    return [
        int(np.min(contour[:, 0])),
        int(np.min(contour[:, 1])),
        int(np.max(contour[:, 0])),
        int(np.max(contour[:, 1])),
    ]


def _bbox_filter(bboxes, w_=400, h_=400, area_size_threshold=8050 * 3):
    """Remove bboxes that are too small or fully contained in a larger one."""
    unique = [list(x) for x in set(tuple(x) for x in bboxes)]
    filtered, outer = [], []
    for i, b1 in enumerate(unique):
        h = b1[2] - b1[0]
        w = b1[3] - b1[1]
        if w < w_ and h < h_:
            continue
        if h * w < area_size_threshold:
            continue
        is_inner = False
        for j, b2 in enumerate(unique):
            if i == j:
                continue
            if (b1[0] >= b2[0] and b1[2] <= b2[2]
                    and b1[1] >= b2[1] and b1[3] <= b2[3]):
                is_inner = True
                if b2 not in outer:
                    outer.append(b2)
                break
        if not is_inner:
            filtered.append(b1)
    filtered = [b for b in filtered if b not in outer]
    return filtered, outer


def _boxscalar(boxes, ori_size, new_size):
    """Scale bboxes from new_size-space back to ori_size-space."""
    y_scale = ori_size[0] / new_size[0]
    x_scale = ori_size[1] / new_size[1]
    return [
        [
            int(b[0] * y_scale), int(b[1] * x_scale),
            int(b[2] * y_scale), int(b[3] * x_scale),
        ]
        for b in boxes
    ]


def _seedpix(bool_arry: np.ndarray, contour_mask: np.ndarray):
    """Return pixels that lie inside the contour *and* on the white background."""
    overlap = bool_arry & contour_mask
    return list(map(tuple, np.argwhere(overlap)))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def warmup() -> None:
    """
    Pre-initialise ChemSAM on the main thread before ThreadPoolExecutor
    launches workers.  Mirrors the DECIMER GPU-warmup pattern so that the
    PyTorch/CUDA context is established in the main thread, not a worker.
    """
    _ensure_loaded()
    dummy = np.zeros((64, 64, 3), dtype="uint8")
    get_chemsam_segments(dummy)


def get_chemsam_segments(image_bgr: np.ndarray):
    """
    Detect and extract chemical structures from a single page image.

    Parameters
    ----------
    image_bgr : np.ndarray
        Full page image as returned by cv2.imread() — shape (H, W, 3), uint8.

    Returns
    -------
    segments : list[np.ndarray]
        Cropped BGR images, one per detected structure.
    bboxes   : list[[x1, y1, x2, y2]]
        Bounding boxes in BioChemInsight's x-first convention.
    masks    : np.ndarray, shape (H, W, N), uint8
        Per-structure binary masks stacked along axis 2 (N = len(segments)).
        Rectangular masks are used since ChemSAM exposes bboxes, not raw masks.
    """
    _ensure_loaded()

    H, W = image_bgr.shape[:2]
    new_size = (_args.image_size, _args.image_size)

    # BGR → RGB → PIL (ChemSAM operates on PIL images)
    image_pil = PILImage.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

    # --- Preprocessing (mirrors chemsam() in inferencing2.py) ---------------
    gray_image = image_pil.convert("L")
    scaled_gray = gray_image.resize(new_size)
    threshold = np.mean(np.array(scaled_gray))
    binary_image = scaled_gray.point(lambda px: 255 if px > threshold else 0, mode="1")
    binary_array = np.array(binary_image)

    blur_factor = max(2, int(binary_array.shape[1] / 185))
    bool_arry = binary_erosion(binary_array, footprint=np.ones((blur_factor, blur_factor)))

    # --- Model inference (serialised with _model_lock) -----------------------
    with _model_lock:
        x1 = _transform(image_pil).unsqueeze(0).to(dtype=torch.float32, device=_device)
        with torch.no_grad():
            out_ = _net.image_encoder(x1)
            bs = out_.size(0)
            sparse_emb = torch.empty(
                (bs, 0, _net.prompt_encoder.embed_dim),
                device=_net.prompt_encoder._get_device(),
            )
            dense_emb = (
                _net.prompt_encoder.no_mask_embed.weight
                .reshape(1, -1, 1, 1)
                .expand(
                    bs, -1,
                    _net.prompt_encoder.image_embedding_size[0],
                    _net.prompt_encoder.image_embedding_size[1],
                )
            )
            pred_masks, _ = _net.mask_decoder(
                image_embeddings=out_,
                image_pe=_net.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False,
            )

    pred_bhw = torch.sigmoid(pred_masks).cpu().detach().numpy()[:, 0, :, :]

    # --- Post-processing -------------------------------------------------------
    # Rescale prediction from network output size to 512×512, then threshold
    rescaled = cv2.resize(
        pred_bhw[0], (new_size[1], new_size[0]), interpolation=cv2.INTER_LINEAR
    )
    predicted_mask = rescaled * 255
    predicted_mask[predicted_mask >= _PIX_CUT] = 255
    predicted_mask[predicted_mask < _PIX_CUT] = 0
    predicted_mask = binary_dilation(predicted_mask, footprint=np.ones((5, 5)))
    predicted_mask = predicted_mask.astype("int8") * 255

    # Connected-component labelling — find individual structure blobs
    labels, num_labels = scipy.ndimage.label(predicted_mask, structure=_STRUCT)
    ar_masks = []
    for lbl in range(1, num_labels + 1):
        contour_mask = labels == lbl
        n_pixels = int(np.count_nonzero(contour_mask))
        ar_masks.append((n_pixels, contour_mask))
    # Process largest blobs first (matches ChemSAM's l2s_masks ordering)
    ar_masks.sort(key=lambda a: a[0], reverse=True)

    expanded_masks_list = []
    for _, contour_mask in ar_masks:
        seeds = _seedpix(bool_arry, contour_mask)
        if seeds:
            expanded = _expand_masks(bool_arry, seeds, contour_mask)
            expanded_masks_list.append(expanded)

    # Extract bboxes from the expanded masks (still in 512-space)
    raw_bboxes = []
    for mask_arr in expanded_masks_list:
        pixels = np.argwhere(mask_arr.astype("int8") != 0)
        if len(pixels):
            raw_bboxes.append(_get_bounding_box(pixels))

    if not raw_bboxes:
        empty = np.zeros((H, W, 0), dtype=np.uint8)
        return [], [], empty

    # Scale area/dimension thresholds from original-image-space to 512-space
    y_scale_fwd = H / new_size[0]
    x_scale_fwd = W / new_size[1]
    area_threshold = 8050 / (y_scale_fwd * x_scale_fwd)
    w_thresh = 400 / x_scale_fwd
    h_thresh = 400 / y_scale_fwd

    filtered, _ = _bbox_filter(
        raw_bboxes, w_=w_thresh, h_=h_thresh, area_size_threshold=area_threshold
    )

    if not filtered:
        empty = np.zeros((H, W, 0), dtype=np.uint8)
        return [], [], empty

    # Scale filtered bboxes from 512-space back to original image dimensions
    scaled = _boxscalar(filtered, ori_size=(H, W), new_size=new_size)

    # --- Build return values in BioChemInsight's expected format ---------------
    # ChemSAM bbox convention: [y1, x1, y2, x2]
    # BioChemInsight convention: [x1, y1, x2, y2]
    segments = []
    bboxes_out = []
    masks_out = []

    for chemsam_bbox in scaled:
        y1, x1, y2, x2 = chemsam_bbox
        # Clamp to image bounds
        y1 = max(0, min(y1, H - 1))
        y2 = max(y1 + 1, min(y2, H))
        x1 = max(0, min(x1, W - 1))
        x2 = max(x1 + 1, min(x2, W))

        crop = image_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        segments.append(crop)
        bboxes_out.append([x1, y1, x2, y2])

        # Rectangular binary mask — ChemSAM doesn't expose per-pixel masks
        # externally, so we reconstruct from the bbox for use in save_box_image
        mask = np.zeros((H, W), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255
        masks_out.append(mask)

    if masks_out:
        masks_3d = np.stack(masks_out, axis=-1)
    else:
        masks_3d = np.zeros((H, W, 0), dtype=np.uint8)

    return segments, bboxes_out, masks_3d
