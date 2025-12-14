import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import torch
import os
import tempfile
from io import BytesIO
from scipy.spatial import Delaunay

# SAM imports
from segment_anything import sam_model_registry, SamPredictor

# Stable Diffusion imports
try:
    from diffusers import AutoPipelineForInpainting
except ImportError:
    # Fallback: import from specific module path
    from diffusers.pipelines.auto_pipeline import AutoPipelineForInpainting

# IDM VITON API imports
from gradio_client import Client, file as handle_file

# Page configuration
st.set_page_config(
    page_title="Virtual Try-On Suite",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
    <style>
    .main { 
        background-color: #f8f9fa; 
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        background-color: #2c3e50;
        color: white;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #34495e;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    h1 {
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .iteration-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #34495e;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Utility functions
def download_sam_weights():
    """Downloads the SAM ViT-B weights if they don't exist."""
    url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    filename = "sam_vit_b_01ec64.pth"
    
    if not os.path.exists(filename):
        with st.spinner(f"Downloading SAM weights... This may take a few minutes."):
            import urllib.request
            urllib.request.urlretrieve(url, filename)
    return filename

def get_line_intersection(p1, p2, p3, p4):
    """Finds the intersection point of two lines (Shoulder-Hip diagonals)."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None  # Parallel lines
    
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return int(px), int(py)

def get_sam_mask(image_np, points, labels, predictor, box=None):
    """Generates a mask using the SAM Predictor."""
    predictor.set_image(image_np)
    
    if box is not None:
        masks, scores, logits = predictor.predict(
            point_coords=np.array(points) if points and len(points) > 0 else None,
            point_labels=np.array(labels) if points and len(points) > 0 else None,
            box=box[None, :] if box is not None else None,
            multimask_output=True
        )
    else:
        masks, scores, logits = predictor.predict(
            point_coords=np.array(points),
            point_labels=np.array(labels),
            multimask_output=True
        )
    
    # Return the best mask
    best_idx = np.argmax(scores)
    return masks[best_idx].astype(np.uint8)

def sam_point_mask(img, point_xy, predictor):
    """Simple point-based mask generation using intersection point (from iteration2.ipynb)."""
    predictor.set_image(img)
    pts = np.array([[point_xy[0], point_xy[1]]], dtype=np.float32)
    lbl = np.array([1], dtype=np.int32)
    masks, scores, _ = predictor.predict(point_coords=pts, point_labels=lbl, multimask_output=True)
    return masks[np.argmax(scores)].astype(np.uint8)

# Smart shirt mask generation helpers
def _lm_xy(lm, idx, W, H):
    """Get landmark xy coordinates."""
    return (int(np.clip(lm[idx].x * W, 0, W-1)),
            int(np.clip(lm[idx].y * H, 0, H-1)))

def _build_torso_polygon_and_box(lm, W, H, pad_ratio=0.12):
    """Build torso polygon and bounding box from landmarks."""
    LS, RS, LH, RH = 11, 12, 23, 24
    p_ls = _lm_xy(lm, LS, W, H)
    p_rs = _lm_xy(lm, RS, W, H)
    p_lh = _lm_xy(lm, LH, W, H)
    p_rh = _lm_xy(lm, RH, W, H)

    # Inset shoulders slightly to avoid grabbing arms
    shoulder_vec = np.array(p_rs) - np.array(p_ls)
    inset = max(4, int(0.06 * (np.linalg.norm(shoulder_vec) + 1e-6)))
    shoulder_dir = shoulder_vec / (np.linalg.norm(shoulder_vec) + 1e-6)
    p_ls_in = (int(p_ls[0] + inset*shoulder_dir[0]), int(p_ls[1]))
    p_rs_in = (int(p_rs[0] - inset*shoulder_dir[0]), int(p_rs[1]))

    torso_poly = np.array([p_ls_in, p_rs_in, p_rh, p_lh], dtype=np.int32)

    x0 = max(0, min(p_ls[0], p_rs[0], p_lh[0], p_rh[0]))
    y0 = max(0, min(p_ls[1], p_rs[1], p_lh[1], p_rh[1]))
    x1 = min(W-1, max(p_ls[0], p_rs[0], p_lh[0], p_rh[0]))
    y1 = min(H-1, max(p_ls[1], p_rs[1], p_lh[1], p_rh[1]))
    pad_x = int((x1 - x0) * pad_ratio)
    pad_y = int((y1 - y0) * pad_ratio)
    box = np.array([max(0, x0-pad_x), max(0, y0-pad_y),
                    min(W-1, x1+pad_x), min(H-1, y1+pad_y)], dtype=np.int32)
    return torso_poly, box, (p_ls, p_rs, p_lh, p_rh)

def _points_inside_polygon(poly, W, H, kx=4, ky=5):
    """Generate positive points inside torso polygon."""
    xs = np.linspace(poly[:,0].min(), poly[:,0].max(), kx+2, dtype=int)[1:-1]
    ys = np.linspace(poly[:,1].min(), poly[:,1].max(), ky+2, dtype=int)[1:-1]
    pos = []
    mask_poly = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(mask_poly, poly, 255)
    for y in ys:
        for x in xs:
            if mask_poly[int(y), int(x)]:
                pos.append((int(x), int(y)))
    return pos

def _negative_points_from_landmarks(lm, W, H):
    """Generate negative points from head and legs to avoid full-body masks."""
    NOSE, L_ANK, R_ANK = 0, 27, 28
    neg = []
    try:
        neg.append(_lm_xy(lm, NOSE, W, H))
    except:
        pass
    try:
        neg.append(_lm_xy(lm, L_ANK, W, H))
    except:
        pass
    try:
        neg.append(_lm_xy(lm, R_ANK, W, H))
    except:
        pass
    return neg

def smart_shirt_mask_with_sam(person_rgb, mp_results, predictor):
    """Build a robust t-shirt mask using SAM with torso guidance."""
    H, W = person_rgb.shape[:2]
    if not mp_results.pose_landmarks:
        raise RuntimeError("No pose landmarks for smart shirt mask.")

    lm = mp_results.pose_landmarks.landmark
    torso_poly, box, (p_ls, p_rs, p_lh, p_rh) = _build_torso_polygon_and_box(lm, W, H)
    pos_pts = _points_inside_polygon(torso_poly, W, H, kx=4, ky=5)
    neg_pts = _negative_points_from_landmarks(lm, W, H)

    point_coords = np.array(pos_pts + neg_pts, dtype=np.float32)
    point_labels = np.array([1]*len(pos_pts) + [0]*len(neg_pts), dtype=np.int32)

    predictor.set_image(person_rgb)
    masks, scores, _ = predictor.predict(
        point_coords=point_coords if len(point_coords) > 0 else None,
        point_labels=point_labels if len(point_coords) > 0 else None,
        box=box[None, :],
        multimask_output=True
    )

    torso_mask = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(torso_mask, torso_poly, 1)
    best_score, best_idx = -1, 0
    torso_area = torso_mask.sum()

    for i, m in enumerate(masks):
        m_u8 = (m > 0).astype(np.uint8)
        inter = (m_u8 & torso_mask).sum()
        union = (m_u8 | torso_mask).sum()
        iou = inter / (union + 1e-8)
        area = m_u8.sum()
        size_penalty = np.exp(-max(0.0, (area/(torso_area+1e-8) - 1.3)))  # penalize too-large masks
        score = iou * size_penalty
        if score > best_score:
            best_score, best_idx = score, i

    raw = (masks[best_idx] > 0).astype(np.uint8)
    refined = (raw & torso_mask).astype(np.uint8)
    refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, np.ones((9,9), np.uint8))

    return refined

# Helper functions for triangulation warping
def biggest_contour(mask):
    """Find the largest contour in a mask."""
    mask_uint8 = mask.astype(np.uint8)
    if mask_uint8.sum() == 0:
        return None
    cnts, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    # Filter out very small contours
    cnts = [c for c in cnts if cv2.contourArea(c) > 100]
    if not cnts:
        return None
    return max(cnts, key=cv2.contourArea).reshape(-1, 2)

def sample_contour(cnt, n=300):
    """Sample n points uniformly along a contour."""
    if cnt is None or len(cnt) < 3:
        return np.array([], dtype=np.float32)
    
    pts = cnt.astype(np.float32)
    d = np.sqrt(np.sum((np.roll(pts, -1, axis=0) - pts) ** 2, axis=1))
    s = np.concatenate([[0], np.cumsum(d)])
    total = float(s[-1]) if s[-1] > 0 else 1.0
    target = np.linspace(0, total, num=n, endpoint=False)
    out = []
    j = 0
    for t in target:
        while not (s[j] <= t < s[j+1] if j+1 < len(s) else s[j] <= t):
            j = (j + 1) % (len(pts) - 1 if len(pts) > 1 else 1)
        a = (t - s[j]) / (s[j+1] - s[j] + 1e-8) if j+1 < len(s) else 0
        p = (1 - a) * pts[j] + a * pts[(j + 1) % len(pts)]
        out.append(p)
    return np.array(out, dtype=np.float32)

def kabsch(A, B):
    """Find optimal rotation and scale using Kabsch algorithm."""
    A0 = A - A.mean(0, keepdims=True)
    B0 = B - B.mean(0, keepdims=True)
    H = A0.T @ B0
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    varA = (A0 ** 2).sum()
    s = np.trace(np.diag(S)) / (varA + 1e-8)
    t = B.mean(0) - s * (A.mean(0) @ R)
    return R, s, t

def best_circular_alignment(src_xy, dst_xy, coarse_steps=32):
    """Find best circular shift alignment between two contours."""
    if len(src_xy) == 0 or len(dst_xy) == 0:
        return src_xy
    N = len(src_xy)
    shifts = np.linspace(0, N, coarse_steps, endpoint=False, dtype=int)
    best_k, best_err = 0, 1e18
    for k in shifts:
        s_shift = np.roll(src_xy, -k, axis=0)
        R, sc, t = kabsch(s_shift, dst_xy)
        pred = sc * (s_shift @ R) + t
        err = np.mean(np.sum((pred - dst_xy) ** 2, axis=1))
        if err < best_err:
            best_err, best_k = err, k
    return np.roll(src_xy, -best_k, axis=0)

def bbox_from_mask(mask):
    """Get bounding box from mask."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 0, 0, mask.shape[1], mask.shape[0]
    return xs.min(), ys.min(), xs.max(), ys.max()

def grid_anchors_in_mask(mask, nx=6, ny=8):
    """Generate grid of anchor points inside mask."""
    x0, y0, x1, y1 = bbox_from_mask(mask)
    W = max(1, x1 - x0)
    H = max(1, y1 - y0)
    us = np.linspace(0.05, 0.95, nx)
    vs = np.linspace(0.05, 0.95, ny)
    pts = []
    for v in vs:
        for u in us:
            x = int(x0 + u * W)
            y = int(y0 + v * H)
            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x] > 0:
                pts.append((x, y))
    return np.array(pts, dtype=np.float32)

def map_grid_anchors_by_uv(src_mask, dst_mask, pts_src):
    """Map source grid anchors to destination using UV coordinates."""
    sx0, sy0, sx1, sy1 = bbox_from_mask(src_mask)
    dx0, dy0, dx1, dy1 = bbox_from_mask(dst_mask)
    sW = max(1, sx1 - sx0)
    sH = max(1, sy1 - sy0)
    dW = max(1, dx1 - dx0)
    dH = max(1, dy1 - dy0)
    uv = np.stack([(pts_src[:, 0] - sx0) / sW, (pts_src[:, 1] - sy0) / sH], axis=1)
    dst_pts = np.stack([dx0 + uv[:, 0] * dW, dy0 + uv[:, 1] * dH], axis=1)
    return dst_pts.astype(np.float32)

def feather_blend(base_img, overlay_img, mask, feather=15):
    """Blend overlay onto base with feathered edges."""
    mask_f = cv2.GaussianBlur(mask.astype(np.float32), (feather * 2 + 1, feather * 2 + 1), 0)
    mask_f = mask_f / (mask_f.max() + 1e-8)
    out = overlay_img.astype(np.float32) * mask_f[..., None] + base_img.astype(np.float32) * (1 - mask_f[..., None])
    return np.clip(out, 0, 255).astype(np.uint8)

def seamless_clone(src, dst, mask, center):
    """Seamless cloning using Poisson blending for better edge integration."""
    try:
        # Ensure center is within bounds
        h, w = dst.shape[:2]
        center_x = int(np.clip(center[0], 0, w - 1))
        center_y = int(np.clip(center[1], 0, h - 1))
        
        # Convert to BGR for OpenCV seamlessClone
        src_bgr = cv2.cvtColor(src, cv2.COLOR_RGB2BGR)
        dst_bgr = cv2.cvtColor(dst, cv2.COLOR_RGB2BGR)
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Ensure mask has valid region
        if mask_uint8.sum() == 0:
            return dst
        
        # Use seamless cloning
        result_bgr = cv2.seamlessClone(src_bgr, dst_bgr, mask_uint8, (center_x, center_y), cv2.NORMAL_CLONE)
        result = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
        return result
    except Exception as e:
        # Fallback to feather blend if seamless clone fails
        return feather_blend(dst, src, mask, feather=20)

def color_correction(src_img, target_img, mask):
    """Apply color correction to match target image's color statistics."""
    mask_3d = mask[..., None].astype(np.float32) / 255.0
    
    # Get non-zero pixels
    mask_bool = mask > 0
    if mask_bool.sum() == 0:
        return src_img
    
    # Filter out white/bright pixels from statistics calculation
    src_pixels_all = src_img[mask_bool]
    target_pixels_all = target_img[mask_bool]
    
    # Remove white pixels (bright in all channels) from statistics
    not_white_src = ~((src_pixels_all[:, 0] > 240) & (src_pixels_all[:, 1] > 240) & (src_pixels_all[:, 2] > 240))
    not_white_target = ~((target_pixels_all[:, 0] > 240) & (target_pixels_all[:, 1] > 240) & (target_pixels_all[:, 2] > 240))
    
    if not_white_src.sum() == 0 or not_white_target.sum() == 0:
        return src_img
    
    src_pixels = src_pixels_all[not_white_src]
    target_pixels = target_pixels_all[not_white_target]
    
    # Compute statistics
    src_mean = src_pixels.mean(axis=0)
    src_std = src_pixels.std(axis=0) + 1e-8
    target_mean = target_pixels.mean(axis=0)
    target_std = target_pixels.std(axis=0) + 1e-8
    
    # Apply color transfer
    corrected = src_img.copy().astype(np.float32)
    for c in range(3):
        corrected[:, :, c] = (corrected[:, :, c] - src_mean[c]) * (target_std[c] / src_std[c]) + target_mean[c]
    
    # Blend only in masked region, and don't apply to white pixels
    result = src_img.copy().astype(np.float32)
    mask_bool_not_white = mask_bool & ~((src_img[:, :, 0] > 240) & (src_img[:, :, 1] > 240) & (src_img[:, :, 2] > 240))
    result[mask_bool_not_white] = corrected[mask_bool_not_white]
    
    return np.clip(result, 0, 255).astype(np.uint8)

def refine_mask(mask, iterations=2):
    """Refine mask with morphological operations."""
    # Remove small holes and noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    refined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Smooth edges
    refined = cv2.GaussianBlur(refined, (5, 5), 0)
    refined = (refined > 127).astype(np.uint8) * 255
    
    return refined

def edge_aware_blend(base_img, overlay_img, mask):
    """Edge-aware blending using guided filter approach."""
    # Convert to float
    base = base_img.astype(np.float32)
    overlay = overlay_img.astype(np.float32)
    mask_f = mask.astype(np.float32) / 255.0
    
    # Filter out white pixels from overlay before blending
    overlay_gray = cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    not_white = overlay_gray < 240
    not_white_3d = not_white[..., None].astype(np.float32)
    
    # Create edge-aware mask using bilateral filter
    mask_3d = mask_f[..., None]
    
    # Multi-scale blending for smoother transitions
    mask_smooth = cv2.bilateralFilter((mask_f * 255).astype(np.uint8), 9, 75, 75).astype(np.float32) / 255.0
    mask_smooth = mask_smooth[..., None]
    
    # Only blend where overlay is not white
    effective_mask = mask_smooth * not_white_3d
    
    # Blend
    result = overlay * effective_mask + base * (1 - effective_mask)
    
    return np.clip(result, 0, 255).astype(np.uint8)

# Load models
@st.cache_resource
def load_sam():
    """Load the SAM model."""
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        
    checkpoint_path = download_sam_weights()
    sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return predictor

@st.cache_resource
def load_inpainting_pipe():
    """Load Stable Diffusion Inpainting pipeline."""
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Try ReV_Animated first, but use a more reliable fallback
    try:
        pipe = AutoPipelineForInpainting.from_pretrained(
            "redstonehero/ReV_Animated_Inpainting",
            torch_dtype=torch.float16 if device != "cpu" else torch.float32
        )
        # Use CPU offload for memory efficiency (as in notebook)
        pipe.enable_model_cpu_offload()
        return pipe
    except Exception as e:
        # Fallback to a more standard model
        try:
            pipe = AutoPipelineForInpainting.from_pretrained(
                "runwayml/stable-diffusion-inpainting",
                torch_dtype=torch.float16 if device != "cpu" else torch.float32
            )
            if device == "mps":
                pipe.enable_attention_slicing()
                pipe = pipe.to(device)
            elif device == "cuda":
                pipe = pipe.to(device)
            else:
                pipe.enable_model_cpu_offload()
            return pipe
        except Exception as e2:
            # Final fallback to SDXL
            st.warning(f"Could not load ReV_Animated or SD 1.5, using SDXL: {e2}")
            pipe = AutoPipelineForInpainting.from_pretrained(
                "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                torch_dtype=torch.float16 if device != "cpu" else torch.float32,
                use_safetensors=True
            )
            if device == "mps":
                pipe.enable_attention_slicing()
                pipe = pipe.to(device)
            elif device == "cuda":
                pipe = pipe.to(device)
            else:
                pipe.enable_model_cpu_offload()
            return pipe

# Iteration 1: Smart Masking and Warping
def run_iteration_1(person_pil, cloth_pil):
    """Run Iteration 1: Mask cloth, extract only cloth pixels, warp to person mask, replace t-shirt."""
    predictor = load_sam()
    
    # MediaPipe Pose Detection
    mp_pose = mp.solutions.pose
    person_np = np.array(person_pil)
    h, w, _ = person_np.shape
    
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        results = pose.process(person_np)
        
        if not results.pose_landmarks:
            st.error("No pose detected in the person image.")
            return None, None, None
            
        lm = results.pose_landmarks.landmark
        
        # Landmarks: 11=L_Shoulder, 12=R_Shoulder, 23=L_Hip, 24=R_Hip
        p11 = (int(lm[11].x * w), int(lm[11].y * h))
        p12 = (int(lm[12].x * w), int(lm[12].y * h))
        p23 = (int(lm[23].x * w), int(lm[23].y * h))
        p24 = (int(lm[24].x * w), int(lm[24].y * h))
        
        # Calculate intersection center
        center = get_line_intersection(p11, p24, p12, p23)
        if not center:
            center = ((p11[0] + p12[0] + p23[0] + p24[0]) // 4,
                     (p11[1] + p12[1] + p23[1] + p24[1]) // 4)
    
    # SAM Mask for Person's T-shirt
    person_mask = sam_point_mask(person_np, center, predictor)
    person_mask_refined = refine_mask(person_mask.astype(np.uint8) * 255)
    person_mask = (person_mask_refined > 127).astype(np.uint8)
    
    # SAM Mask for Cloth - extract only the cloth
    cloth_np = np.array(cloth_pil)
    cw, ch = cloth_np.shape[1], cloth_np.shape[0]
    cloth_mask = get_sam_mask(cloth_np, [[cw//2, ch//2]], [1], predictor)
    cloth_mask_refined = refine_mask(cloth_mask.astype(np.uint8) * 255)
    cloth_mask = (cloth_mask_refined > 127).astype(np.uint8)
    
    # Extract only the masked cloth pixels (remove background)
    cloth_masked = cloth_np.copy()
    cloth_masked[cloth_mask == 0] = [0, 0, 0]  # Set background to black
    
    # Get bounding boxes with padding to ensure we capture edges
    cloth_ys, cloth_xs = np.where(cloth_mask > 0)
    if len(cloth_xs) == 0:
        st.error("Could not extract cloth from image.")
        return None, None, None
    
    cloth_x_min, cloth_x_max = cloth_xs.min(), cloth_xs.max()
    cloth_y_min, cloth_y_max = cloth_ys.min(), cloth_ys.max()
    
    # Add padding to ensure we capture edge pixels
    pad = 5
    cloth_x_min = max(0, cloth_x_min - pad)
    cloth_x_max = min(cw - 1, cloth_x_max + pad)
    cloth_y_min = max(0, cloth_y_min - pad)
    cloth_y_max = min(ch - 1, cloth_y_max + pad)
    
    # Crop to cloth region only
    cloth_cropped = cloth_masked[cloth_y_min:cloth_y_max+1, cloth_x_min:cloth_x_max+1]
    cloth_mask_cropped = cloth_mask[cloth_y_min:cloth_y_max+1, cloth_x_min:cloth_x_max+1]
    
    # Ensure cloth_mask_cropped is valid
    if cloth_mask_cropped.sum() == 0:
        st.error("Cloth mask became empty after cropping.")
        return None, None, None
    
    # Store original mask for safety
    cloth_mask_cropped_original = cloth_mask_cropped.copy()
    original_mask_area = cloth_mask_cropped.sum()
    
    # Try to dilate cloth mask slightly to ensure we capture edge pixels
    # But only if it's safe to do so
    if original_mask_area > 50:  # Only dilate if mask has sufficient area
        try:
            kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            # Convert to uint8 with proper scaling
            cloth_mask_cropped_uint8 = (cloth_mask_cropped * 255).astype(np.uint8)
            cloth_mask_cropped_dilated = cv2.dilate(cloth_mask_cropped_uint8, kernel_small, iterations=1)
            cloth_mask_cropped_new = (cloth_mask_cropped_dilated > 127).astype(np.uint8)
            
            # Only use dilated version if it still has reasonable content
            new_mask_area = cloth_mask_cropped_new.sum()
            if new_mask_area > 0 and new_mask_area >= original_mask_area * 0.3:  # At least 30% of original area
                cloth_mask_cropped = cloth_mask_cropped_new
            else:
                # Dilation caused issues, keep original
                cloth_mask_cropped = cloth_mask_cropped_original
        except Exception as e:
            # If dilation fails for any reason, keep original mask
            cloth_mask_cropped = cloth_mask_cropped_original
    
    # Final safety check - ensure mask still has content
    if cloth_mask_cropped.sum() == 0:
        st.error("Cloth mask became empty. Using original mask.")
        cloth_mask_cropped = cloth_mask_cropped_original
        if cloth_mask_cropped.sum() == 0:
            return None, None, None
    
    # Get person t-shirt bounding box
    person_ys, person_xs = np.where(person_mask > 0)
    if len(person_xs) == 0:
        st.error("Could not extract person t-shirt mask.")
        return None, None, None
    
    # Triangulation Warping - map cloth mask to person mask
    # Ensure masks are valid
    if cloth_mask_cropped.sum() == 0:
        st.error("Cloth mask is empty after processing.")
        return None, None, None
    
    if person_mask.sum() == 0:
        st.error("Person t-shirt mask is empty.")
        return None, None, None
    
    cloth_cnt = biggest_contour(cloth_mask_cropped.astype(np.uint8) * 255)
    person_cnt = biggest_contour(person_mask.astype(np.uint8) * 255)
    
    if cloth_cnt is None:
        st.error("Could not extract cloth contour. Cloth mask may be too fragmented.")
        # Try using the full cloth mask instead
        cloth_cnt = biggest_contour(cloth_mask.astype(np.uint8) * 255)
        if cloth_cnt is None:
            return None, None, None
        # Use full cloth for warping
        cloth_cropped = cloth_masked
        cloth_mask_cropped = cloth_mask
        cloth_x_min, cloth_y_min = 0, 0
    
    if person_cnt is None:
        st.error("Could not extract person t-shirt contour.")
        return None, None, None
    
    # Cloth contour is already in cropped space, no adjustment needed
    cloth_cnt_adjusted = cloth_cnt.copy()
    
    # Sample contours
    N_BOUND = 300
    try:
        src_boundary = sample_contour(cloth_cnt_adjusted, n=N_BOUND)
        dst_boundary = sample_contour(person_cnt, n=N_BOUND)
        if len(src_boundary) == 0 or len(dst_boundary) == 0:
            raise ValueError("Empty contours after sampling")
        src_boundary = best_circular_alignment(src_boundary, dst_boundary)
    except Exception as e:
        st.error(f"Contour sampling failed: {str(e)}")
        return None, None, None
    
    # Interior grid anchors
    src_grid = grid_anchors_in_mask(cloth_mask_cropped, nx=6, ny=8)
    if len(src_grid) == 0:
        # Fallback: use contour points as grid
        if len(cloth_cnt_adjusted) > 20:
            src_grid = cloth_cnt_adjusted[::max(1, len(cloth_cnt_adjusted)//20)]  # Sample every 20th point
        else:
            src_grid = cloth_cnt_adjusted
    
    # Grid is already in cropped space coordinates
    src_grid_adjusted = src_grid.copy()
    
    # Map to person mask - need to map from cropped cloth space to person space
    # Create a temporary full-size mask for UV mapping
    cloth_mask_full = np.zeros((ch, cw), dtype=np.uint8)
    cloth_mask_full[cloth_y_min:cloth_y_max+1, cloth_x_min:cloth_x_max+1] = cloth_mask_cropped
    dst_grid = map_grid_anchors_by_uv(cloth_mask_full, person_mask, src_grid)
    
    # Pose anchors
    pose_dst = []
    for (x, y) in [p11, p12, p23, p24]:
        x = int(np.clip(x, 0, w - 1))
        y = int(np.clip(y, 0, h - 1))
        if person_mask[y, x] == 0:
            found = False
            for r in range(3, 25, 3):
                for dx in (-r, 0, r):
                    for dy in (-r, 0, r):
                        xx = int(np.clip(x + dx, 0, w - 1))
                        yy = int(np.clip(y + dy, 0, h - 1))
                        if person_mask[yy, xx] > 0:
                            pose_dst.append((xx, yy))
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
        else:
            pose_dst.append((x, y))
    
    if len(pose_dst) > 0:
        pose_dst = np.array(pose_dst, dtype=np.float32)
        dx0, dy0, dx1, dy1 = bbox_from_mask(person_mask)
        dW = max(1, dx1 - dx0)
        dH = max(1, dy1 - dy0)
        uv = np.stack([(pose_dst[:, 0] - dx0) / dW, (pose_dst[:, 1] - dy0) / dH], axis=1)
        sx0, sy0, sx1, sy1 = 0, 0, cloth_mask_cropped.shape[1]-1, cloth_mask_cropped.shape[0]-1
        sW = max(1, sx1 - sx0)
        sH = max(1, sy1 - sy0)
        pose_src = np.stack([sx0 + uv[:, 0] * sW, sy0 + uv[:, 1] * sH], axis=1).astype(np.float32)
    else:
        pose_src = np.zeros((0, 2), dtype=np.float32)
    
    src_pts = np.vstack([src_boundary, src_grid_adjusted, pose_src])
    dst_pts = np.vstack([dst_boundary, dst_grid, pose_dst])
    
    # Delaunay triangulation and warping
    canvas_rgb = np.zeros_like(person_np)
    acc_mask = np.zeros((h, w), dtype=np.uint8)
    tri = Delaunay(dst_pts)
    simplices = tri.simplices
    
    for (i, j, k) in simplices:
        dst_tri = np.float32([dst_pts[i], dst_pts[j], dst_pts[k]])
        src_tri = np.float32([src_pts[i], src_pts[j], src_pts[k]])
        
        x_min = int(np.floor(dst_tri[:, 0].min()))
        x_max = int(np.ceil(dst_tri[:, 0].max()))
        y_min = int(np.floor(dst_tri[:, 1].min()))
        y_max = int(np.ceil(dst_tri[:, 1].max()))
        
        if x_max <= x_min or y_max <= y_min:
            continue
        
        x_min = np.clip(x_min, 0, w - 1)
        x_max = np.clip(x_max, 0, w - 1)
        y_min = np.clip(y_min, 0, h - 1)
        y_max = np.clip(y_max, 0, h - 1)
        
        dst_tri_local = dst_tri.copy()
        dst_tri_local[:, 0] -= x_min
        dst_tri_local[:, 1] -= y_min
        tri_mask = np.zeros((y_max - y_min + 1, x_max - x_min + 1), dtype=np.uint8)
        cv2.fillConvexPoly(tri_mask, np.int32(dst_tri_local), 255)
        
        # Warp the cropped cloth (only masked pixels)
        M = cv2.getAffineTransform(src_tri, dst_tri)
        warped_full = cv2.warpAffine(cloth_cropped, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
        warped_roi = warped_full[y_min:y_max + 1, x_min:x_max + 1]
        
        # Also warp the mask to know which pixels are valid
        warped_mask_full = cv2.warpAffine(cloth_mask_cropped.astype(np.uint8) * 255, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        warped_mask_roi = warped_mask_full[y_min:y_max + 1, x_min:x_max + 1]
        
        shirt_roi = person_mask[y_min:y_max + 1, x_min:x_max + 1]
        # Keep pixels that are: in triangle AND in person mask
        # Use warped mask as secondary check, but prioritize person mask coverage
        keep = (tri_mask > 0) & (shirt_roi > 0)
        
        # Prefer pixels that are in warped cloth mask, but also fill gaps
        preferred = keep & (warped_mask_roi > 127)
        fallback = keep & (warped_mask_roi <= 127)  # Fill gaps even if not in original cloth mask
        
        # Apply preferred pixels first
        canvas_rgb[y_min:y_max + 1, x_min:x_max + 1][preferred] = warped_roi[preferred]
        acc_mask[y_min:y_max + 1, x_min:x_max + 1][preferred] = 255
        
        # Fill gaps with warped pixels (even if not in original mask, they're still valid cloth pixels)
        if fallback.sum() > 0:
            # Use the warped pixels to fill gaps
            canvas_rgb[y_min:y_max + 1, x_min:x_max + 1][fallback] = warped_roi[fallback]
            acc_mask[y_min:y_max + 1, x_min:x_max + 1][fallback] = 255
    
    # Remove person's original t-shirt and replace with warped cloth
    final_img = person_np.copy()
    
    # Fill gaps in canvas_rgb by inpainting missing areas
    # Find areas in person mask that don't have cloth
    missing_areas = (person_mask > 0) & (acc_mask == 0)
    
    if missing_areas.sum() > 0:
        # Use inpainting to fill missing areas with nearby cloth pixels
        canvas_rgb_uint8 = np.clip(canvas_rgb, 0, 255).astype(np.uint8)
        missing_mask = missing_areas.astype(np.uint8) * 255
        
        # Inpaint missing areas using the existing cloth pixels
        canvas_rgb_filled = cv2.inpaint(canvas_rgb_uint8, missing_mask, 5, cv2.INPAINT_TELEA)
        canvas_rgb = canvas_rgb_filled
        
        # Update acc_mask to include filled areas
        acc_mask = person_mask.astype(np.uint8) * 255
    
    # Ensure we fill the entire person mask area
    # For any remaining gaps, use the nearest cloth pixel
    if (person_mask > 0).sum() > acc_mask.sum() * 0.9:  # If less than 90% coverage
        # Dilate the cloth slightly to fill edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        acc_mask_dilated = cv2.dilate(acc_mask, kernel, iterations=2)
        
        # For newly added areas, use the nearest cloth color
        new_areas = (acc_mask_dilated > 0) & (acc_mask == 0) & (person_mask > 0)
        if new_areas.sum() > 0:
            # Get the average color from existing cloth pixels
            existing_cloth = canvas_rgb[acc_mask > 0]
            if len(existing_cloth) > 0:
                avg_color = existing_cloth.mean(axis=0)
                canvas_rgb[new_areas] = avg_color
                acc_mask = acc_mask_dilated
    
    # Apply warped cloth to entire person mask area
    final_img[person_mask > 0] = canvas_rgb[person_mask > 0]
    
    # Apply feather blending at edges for smooth transition
    # Use person_mask for blending to ensure full coverage
    final_img = feather_blend(person_np, final_img, person_mask.astype(np.uint8) * 255, feather=20)
    
    # Create visualization with intersection point and landmarks
    vis_img = person_np.copy()
    cv2.line(vis_img, p11, p24, (255, 0, 0), 2)
    cv2.line(vis_img, p12, p23, (255, 0, 0), 2)
    cv2.circle(vis_img, center, 10, (0, 255, 0), -1)
    # Draw landmark points
    cv2.circle(vis_img, p11, 5, (0, 0, 255), -1)  # Left shoulder - red
    cv2.circle(vis_img, p12, 5, (0, 0, 255), -1)  # Right shoulder - red
    cv2.circle(vis_img, p23, 5, (255, 0, 255), -1)  # Left hip - magenta
    cv2.circle(vis_img, p24, 5, (255, 0, 255), -1)  # Right hip - magenta
    landmark_vis = Image.fromarray(vis_img)
    
    # Create cloth mask visualization
    cloth_mask_vis = cloth_np.copy()
    cloth_mask_vis[cloth_mask == 0] = [255, 255, 255]  # White background for visibility
    
    return Image.fromarray(final_img), Image.fromarray(person_mask.astype(np.uint8) * 255), landmark_vis, Image.fromarray(cloth_mask_vis)

# Iteration 2: Stable Diffusion
def run_iteration_2(person_pil, prompt_text):
    """Run Iteration 2: Stable Diffusion Inpainting."""
    predictor = load_sam()
    
    # MediaPipe & SAM (same logic as Iteration 1)
    mp_pose = mp.solutions.pose
    person_np = np.array(person_pil)
    h, w, _ = person_np.shape
    
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        results = pose.process(person_np)
        if not results.pose_landmarks:
            st.error("No pose detected.")
            return None, None, None
        
        lm = results.pose_landmarks.landmark
        p11 = (int(lm[11].x * w), int(lm[11].y * h))
        p12 = (int(lm[12].x * w), int(lm[12].y * h))
        p23 = (int(lm[23].x * w), int(lm[23].y * h))
        p24 = (int(lm[24].x * w), int(lm[24].y * h))
        center = get_line_intersection(p11, p24, p12, p23)
        if not center:
            center = ((p11[0] + p12[0] + p23[0] + p24[0]) // 4,
                     (p11[1] + p12[1] + p23[1] + p24[1]) // 4)
    
    # Use simple point-based mask with intersection point (same as Iteration 1)
    mask = sam_point_mask(person_np, center, predictor)
    # For inpainting: white areas are inpainted, black areas are kept
    # SAM mask: 1 = shirt area, 0 = background, so multiply by 255 to get white for shirt
    mask_array = (mask.astype(np.uint8) * 255)
    mask_pil = Image.fromarray(mask_array).convert("L")
    
    # Inpainting
    pipe = load_inpainting_pipe()
    
    # Resize for SD (must be multiple of 8)
    w_orig, h_orig = person_pil.size
    w_new = (w_orig // 8) * 8
    h_new = (h_orig // 8) * 8
    
    img_resized = person_pil.resize((w_new, h_new))
    mask_resized = mask_pil.resize((w_new, h_new))
    
    # Verify mask has white pixels (for debugging)
    mask_check = np.array(mask_resized)
    if mask_check.max() == 0:
        st.error("Mask is completely black! SAM may have failed to generate a mask.")
        return None, mask_pil, Image.fromarray(vis_img)
    
    # Match parameters from iteration1.ipynb
    try:
        with torch.inference_mode():
            output = pipe(
                prompt=prompt_text,
                image=img_resized,
                mask_image=mask_resized,
                width=w_new,
                height=h_new,
                num_inference_steps=28,
                strength=1.0,
                guidance_scale=3.0
            ).images[0]
        
        result = output.resize((w_orig, h_orig))
        
        # Verify result is not black
        result_array = np.array(result)
        if result_array.max() < 10:  # If all pixels are very dark
            st.error("Pipeline returned a black image. This may be a model loading issue.")
            return None, mask_pil, Image.fromarray(vis_img)
            
    except Exception as e:
        st.error(f"Inpainting failed: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None, mask_pil, Image.fromarray(vis_img)
    
    # Create visualization with intersection point
    vis_img = person_np.copy()
    cv2.line(vis_img, p11, p24, (255, 0, 0), 2)
    cv2.line(vis_img, p12, p23, (255, 0, 0), 2)
    cv2.circle(vis_img, center, 10, (0, 255, 0), -1)
    
    return result, mask_pil, Image.fromarray(vis_img)

# Iteration 3: IDM-VTON API Only (with retry logic)
def run_iteration_3(person_pil, cloth_pil):
    """Run Iteration 3: IDM-VTON API only with retry logic."""
    predictor = load_sam()
    
    # Generate SAM mask and landmarks for visualization
    mp_pose = mp.solutions.pose
    person_np = np.array(person_pil)
    h, w, _ = person_np.shape
    
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        results = pose.process(person_np)
        
        if not results.pose_landmarks:
            st.error("No pose detected.")
            return None, None, None, None
        
        lm = results.pose_landmarks.landmark
        p11 = (int(lm[11].x * w), int(lm[11].y * h))
        p12 = (int(lm[12].x * w), int(lm[12].y * h))
        p23 = (int(lm[23].x * w), int(lm[23].y * h))
        p24 = (int(lm[24].x * w), int(lm[24].y * h))
        center = get_line_intersection(p11, p24, p12, p23)
        if not center:
            center = ((p11[0] + p12[0] + p23[0] + p24[0]) // 4,
                     (p11[1] + p12[1] + p23[1] + p24[1]) // 4)
    
    # Use simple point-based mask with intersection point
    person_mask = sam_point_mask(person_np, center, predictor)
    
    # Create visualization images
    mask_vis = Image.fromarray(person_mask.astype(np.uint8) * 255)
    
    vis_img = person_np.copy()
    cv2.line(vis_img, p11, p24, (255, 0, 0), 2)
    cv2.line(vis_img, p12, p23, (255, 0, 0), 2)
    cv2.circle(vis_img, center, 10, (0, 255, 0), -1)
    landmark_vis = Image.fromarray(vis_img)
    
    # DensePose visualization with green landmarks
    dense_pose_vis = person_np.copy()
    # Draw all pose landmarks in green
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(dense_pose_vis, (x, y), 3, (0, 255, 0), -1)
        # Draw key connections
        mp_drawing = mp.solutions.drawing_utils
        mp_pose_draw = mp.solutions.pose
        # Draw pose connections in green
        for connection in mp_pose_draw.POSE_CONNECTIONS:
            start_idx = connection[0]
            end_idx = connection[1]
            if start_idx < len(results.pose_landmarks.landmark) and end_idx < len(results.pose_landmarks.landmark):
                start = results.pose_landmarks.landmark[start_idx]
                end = results.pose_landmarks.landmark[end_idx]
                x1, y1 = int(start.x * w), int(start.y * h)
                x2, y2 = int(end.x * w), int(end.y * h)
                if 0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h:
                    cv2.line(dense_pose_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    dense_pose_vis = Image.fromarray(dense_pose_vis)
    
    # IDM-VTON API call with retry logic
    result_img = None
    person_path = None
    cloth_path = None
    max_retries = 5
    retry_delay = 3  # seconds
    
    for attempt in range(max_retries):
        try:
            # Create temporary files
            if person_path is None or not os.path.exists(person_path):
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f1:
                    person_pil.save(f1.name, "JPEG")
                    person_path = f1.name
            if cloth_path is None or not os.path.exists(cloth_path):
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f2:
                    cloth_pil.save(f2.name, "JPEG")
                    cloth_path = f2.name
            
            # API call (silent retries - no error messages shown)
            client = Client("yisol/IDM-VTON")
            result = client.predict(
                dict={"background": handle_file(person_path), "layers": [], "composite": None},
                garm_img=handle_file(cloth_path),
                garment_des="A cool t-shirt",
                is_checked=True,
                is_checked_crop=False,
                denoise_steps=30,
                seed=42,
                api_name="/tryon"
            )
            
            # Result[0] is usually the processed image path
            if result and len(result) > 0 and result[0]:
                result_img = Image.open(result[0]).convert("RGB")
                break  # Success!
            else:
                raise Exception("Empty result")
                
        except Exception as e:
            if attempt < max_retries - 1:
                # Wait before retrying silently
                import time
                time.sleep(retry_delay)
                continue
            else:
                # Final attempt failed - return None silently (no error messages)
                result_img = None
    
    # Cleanup temp files
    if person_path and os.path.exists(person_path):
        try:
            os.remove(person_path)
        except:
            pass
    if cloth_path and os.path.exists(cloth_path):
        try:
            os.remove(cloth_path)
        except:
            pass
    
    return result_img, mask_vis, landmark_vis, dense_pose_vis

# Main UI
def main():
    st.title("Virtual Try-On Suite")
    st.markdown("<p style='text-align: center; color: #7f8c8d; margin-bottom: 2rem;'>Advanced Virtual Try-On Technology</p>", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("Select Iteration")
        iteration = st.radio(
            "Choose Method:",
            ["Iteration 1", "Iteration 2", "Iteration 3"],
            label_visibility="collapsed"
        )
        
        st.divider()
        st.markdown("### Settings")
        
        if iteration == "Iteration 2":
            prompt = st.text_area(
                "Garment Prompt",
                "a bright red graphic t-shirt, high quality, realistic texture",
                height=100
            )
        
        st.info("Upload images and click Process to begin.")
    
    # Main content area
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Person Image")
        person_file = st.file_uploader("Upload Person Image", type=['png', 'jpg', 'jpeg'], key="person")
        if person_file:
            person_img = Image.open(person_file).convert("RGB")
            st.image(person_img, use_container_width=True)
        else:
            person_img = None
            
    with col2:
        if iteration != "Iteration 2":
            st.markdown("### Cloth Image")
            cloth_file = st.file_uploader("Upload Cloth Image", type=['png', 'jpg', 'jpeg'], key="cloth")
            if cloth_file:
                cloth_img = Image.open(cloth_file).convert("RGB")
                st.image(cloth_img, use_container_width=True)
            else:
                cloth_img = None
        else:
            st.markdown("### Prompt Based Generation")
            if 'prompt' in locals():
                st.write(f"**Prompt:** {prompt}")
            cloth_img = None

    # Process button
    st.markdown("<br>", unsafe_allow_html=True)
    process_button = st.button(
        f"Process - {iteration}",
        type="primary",
        use_container_width=True
    )
    
    if process_button:
        if not person_img:
            st.error("Please upload a person image.")
        elif iteration != "Iteration 2" and not cloth_img:
            st.error("Please upload a cloth image.")
        else:
            with st.spinner("Processing... This may take a moment."):
                try:
                    if iteration == "Iteration 1":
                        result_img, mask_vis, landmark_vis, cloth_mask_vis = run_iteration_1(person_img, cloth_img)
                        
                        if result_img:
                            st.success("Processing completed successfully!")
                            st.divider()
                            
                            # Display results
                            res_col1, res_col2 = st.columns([2, 1])
                            with res_col1:
                                st.markdown("### Result")
                                st.image(result_img, use_container_width=True)
                                
                                # Download button
                                buf = BytesIO()
                                result_img.save(buf, format="PNG")
                                st.download_button(
                                    "Download Result",
                                    buf.getvalue(),
                                    "iteration1_result.png",
                                    "image/png"
                                )
                            
                            with res_col2:
                                st.markdown("### Process Visualizations")
                                st.image(landmark_vis, caption="Landmark Intersection Points", use_container_width=True)
                                st.image(mask_vis, caption="Person T-shirt SAM Mask", use_container_width=True)
                                st.image(cloth_mask_vis, caption="Cloth SAM Mask", use_container_width=True)
                        else:
                            st.error("Processing failed. Please check your images.")
                    
                    elif iteration == "Iteration 2":
                        result_img, mask_vis, landmark_vis = run_iteration_2(person_img, prompt)
                        
                        if result_img:
                            st.success("Processing completed successfully!")
                            st.divider()
                            
                            # Display results
                            res_col1, res_col2 = st.columns([2, 1])
                            with res_col1:
                                st.markdown("### Result")
                                st.image(result_img, use_container_width=True)
                                
                                # Download button
                                buf = BytesIO()
                                result_img.save(buf, format="PNG")
                                st.download_button(
                                    "Download Result",
                                    buf.getvalue(),
                                    "iteration2_result.png",
                                    "image/png"
                                )
                            
                            with res_col2:
                                st.markdown("### Process Visualizations")
                                st.image(landmark_vis, caption="Landmark Intersection", use_container_width=True)
                                st.image(mask_vis, caption="SAM Mask", use_container_width=True)
                        else:
                            st.error("Processing failed. Please check your images.")
                    
                    elif iteration == "Iteration 3":
                        result_img, mask_vis, landmark_vis, dense_pose_vis = run_iteration_3(person_img, cloth_img)
                        
                        st.divider()
                        
                        # Always show visualizations
                        viz_col1, viz_col2, viz_col3 = st.columns(3)
                        with viz_col1:
                            st.markdown("#### SAM Mask")
                            st.image(mask_vis, use_container_width=True)
                        with viz_col2:
                            st.markdown("#### Landmark Intersection")
                            st.image(landmark_vis, use_container_width=True)
                        with viz_col3:
                            st.markdown("#### Dense Pose")
                            st.image(dense_pose_vis, use_container_width=True)
                        
                        if result_img:
                            st.success("Processing completed successfully!")
                            st.markdown("### Final Result")
                            st.image(result_img, use_container_width=True)
                            
                            # Download button
                            buf = BytesIO()
                            result_img.save(buf, format="PNG")
                            st.download_button(
                                "Download Result",
                                buf.getvalue(),
                                "iteration3_result.png",
                                "image/png"
                            )
                        else:
                            # No result available - just show visualizations without any message
                            pass
                        
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
