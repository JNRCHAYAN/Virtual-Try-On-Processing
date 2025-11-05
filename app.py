import os
import io
import numpy as np
import cv2
from PIL import Image, ImageOps
import streamlit as st

# Optional pose (for smarter masks). We handle absence gracefully.
try:
    import mediapipe as mp
    MP_AVAILABLE = True
except Exception:
    MP_AVAILABLE = False

# Stable Diffusion Inpainting
import torch
from diffusers import AutoPipelineForInpainting

# ---------------------------
# Utilities
# ---------------------------

def load_image(file) -> Image.Image:
    img = Image.open(file).convert("RGB")
    return img

def resize_with_padding(im: Image.Image, target=768):
    """
    Resize the image so the longer side == target, pad to a square (target x target).
    Returns padded_image, scale info for later unpadding.
    """
    w, h = im.size
    if max(w, h) == target:
        base = im.copy()
    else:
        if w >= h:
            new_w = target
            new_h = int(h * (target / w))
        else:
            new_h = target
            new_w = int(w * (target / h))
        base = im.resize((new_w, new_h), Image.LANCZOS)

    # pad to square
    pad_w = target - base.size[0]
    pad_h = target - base.size[1]
    left = pad_w // 2
    top = pad_h // 2
    padded = ImageOps.expand(base, border=(left, top, pad_w - left, pad_h - top), fill=(0,0,0))
    return padded, (left, top, base.size[0], base.size[1], (w, h))

def unpad_and_resize_back(padded_img: Image.Image, meta):
    left, top, base_w, base_h, orig_wh = meta
    cropped = padded_img.crop((left, top, left + base_w, top + base_h))
    return cropped.resize(orig_wh, Image.LANCZOS)

def landmarks_to_points(landmarks, image_w, image_h, names):
    """ Extract pixel coordinates for requested landmark names. """
    pose_idx = {
        "left_shoulder": 11, "right_shoulder": 12,
        "left_hip": 23, "right_hip": 24,
        "left_knee": 25, "right_knee": 26,
        "left_ankle": 27, "right_ankle": 28,
        "nose": 0
    }
    pts = {}
    for n in names:
        i = pose_idx[n]
        lm = landmarks[i]
        x = int(lm.x * image_w)
        y = int(lm.y * image_h)
        pts[n] = (x,y)
    return pts

def fallback_mask(H, W, region: str) -> np.ndarray:
    """Simple rectangle mask when pose not detected/available."""
    mask = np.zeros((H, W), dtype=np.uint8)
    if region == "Upper":
        cv2.rectangle(mask, (W//4, H//8), (3*W//4, H//2), 255, -1)
    else:
        cv2.rectangle(mask, (W//3, H//2), (2*W//3, int(0.95*H)), 255, -1)
    mask = cv2.GaussianBlur(mask, (31,31), 0)
    return mask

def make_region_mask(padded_img: Image.Image, region: str) -> Image.Image:
    """
    Create a white (255) mask for the area to inpaint (upper or lower body) using pose landmarks.
    The rest is black (0). Also soft-expand the region for better blending.
    """
    img = np.array(padded_img)
    H, W = img.shape[:2]

    if not MP_AVAILABLE:
        return Image.fromarray(fallback_mask(H, W, region))

    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True, model_complexity=1, enable_segmentation=False) as pose:
        res = pose.process(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if not res.pose_landmarks:
            return Image.fromarray(fallback_mask(H, W, region))

        lms = res.pose_landmarks.landmark
        need = [
            "left_shoulder","right_shoulder","left_hip","right_hip",
            "left_knee","right_knee","left_ankle","right_ankle","nose"
        ]
        pts = landmarks_to_points(lms, W, H, need)

        mask = np.zeros((H, W), dtype=np.uint8)

        if region == "Upper":
            ls, rs = pts["left_shoulder"], pts["right_shoulder"]
            lh, rh = pts["left_hip"], pts["right_hip"]
            expand_x = int(0.10 * W)
            expand_up = int(0.06 * H)
            expand_down = int(0.10 * H)
            poly = np.array([
                (rs[0]-expand_x, rs[1]-expand_up),
                (ls[0]+expand_x, ls[1]-expand_up),
                (lh[0]+expand_x, lh[1]+expand_down),
                (rh[0]-expand_x, rh[1]+expand_down)
            ], dtype=np.int32)
            cv2.fillPoly(mask, [poly], 255)
        else:  # Lower
            lh, rh = pts["left_hip"], pts["right_hip"]
            lk, rk = pts["left_knee"], pts["right_knee"]
            la, ra = pts["left_ankle"], pts["right_ankle"]
            y_bottom = max(la[1], ra[1])
            y_mid = int(0.5*(lk[1] + rk[1]))
            expand_x_top = int(0.06 * W)
            expand_x_mid = int(0.08 * W)
            expand_x_bot = int(0.10 * W)
            poly = np.array([
                (rh[0]-expand_x_top, rh[1]),
                (lh[0]+expand_x_top, lh[1]),
                (lh[0]+expand_x_mid, y_mid),
                (rh[0]-expand_x_mid, y_mid),
                (rh[0]-expand_x_bot, y_bottom),
                (lh[0]+expand_x_bot, y_bottom)
            ], dtype=np.int32)
            cv2.fillPoly(mask, [poly], 255)

        kernel = np.ones((21,21), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.GaussianBlur(mask, (51,51), 0)
        return Image.fromarray(mask)

@st.cache_resource
def load_inpaint_pipe():
    """
    Free, local Stable Diffusion inpainting model.
    We don't include torch in requirements to avoid wheel issues; install torch separately.
    """
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = AutoPipelineForInpainting.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch_dtype,
        safety_checker=None,
    )
    pipe = pipe.to(device)
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    return pipe

def run_inpaint(pipe, padded_img: Image.Image, mask_img: Image.Image, prompt: str, negative_prompt: str, guidance=7.0, steps=30, seed=42):
    generator = torch.Generator(device=pipe.device.type).manual_seed(int(seed))
    result = pipe(
        prompt=prompt,
        image=padded_img,
        mask_image=mask_img,
        negative_prompt=negative_prompt,
        guidance_scale=float(guidance),
        num_inference_steps=int(steps),
        generator=generator,
        strength=0.95,
    )
    return result.images[0]

# ---------------------------
# Streamlit UI
# ---------------------------

st.set_page_config(page_title="Virtual Try-On (Free Demo)", page_icon="🧥", layout="wide")
st.title("🧥 Virtual Try-On — Inpainting Demo (Free)")

st.markdown("""
Upload a photo of a **single person** (frontal or 3/4 view works best), choose a region, and describe the clothing item.
If pose detection isn't available, the app will use a reasonable fallback mask.
""")

col_left, col_right = st.columns([1,1], gap="large")

with col_left:
    uploaded = st.file_uploader("Upload an image (PNG/JPG)", type=["png","jpg","jpeg"])
    region = st.radio("Select region", ["Upper", "Lower"], horizontal=True)
    default_prompt = "a Hawaiian shirt with bright floral patterns, short sleeves, casual fit"
    user_prompt = st.text_area("Clothing prompt", value=default_prompt, height=90,
                               help="Describe the clothing to apply to the selected region.")
    negative_prompt = "deformed, disfigured, extra limbs, blurry, text, logo, watermark, low quality"
    seed = st.number_input("Seed (for reproducibility)", value=42, min_value=0, step=1)
    steps = st.slider("Sampling steps", min_value=10, max_value=60, value=30, step=5)
    guidance = st.slider("Guidance scale", min_value=3.0, max_value=12.0, value=7.0, step=0.5)

    run = st.button("Run Virtual Try-On", type="primary")

with col_right:
    st.markdown("### Results")

    if uploaded and run:
        try:
            img = load_image(uploaded)
            padded, meta = resize_with_padding(img, target=768)

            mask = make_region_mask(padded, region)

            pipe = load_inpaint_pipe()
            out = run_inpaint(
                pipe, padded, mask, user_prompt, negative_prompt,
                guidance=guidance, steps=steps, seed=int(seed)
            )

            out_full = unpad_and_resize_back(out, meta)
            mask_full = unpad_and_resize_back(mask.convert("RGB"), meta)

            # Use use_column_width=True for older Streamlit versions
            st.image(np.array(img), caption="Original", use_column_width=True)
            st.image(np.array(mask_full), caption="Inpainting Mask (white = edited)", use_column_width=True)
            st.image(np.array(out_full), caption="Final Try-On", use_column_width=True)

            buf_out = io.BytesIO()
            out_full.save(buf_out, format="PNG")
            st.download_button("Download Final Image", buf_out.getvalue(), file_name="tryon_result.png", mime="image/png")

            buf_mask = io.BytesIO()
            mask_full.save(buf_mask, format="PNG")
            st.download_button("Download Mask", buf_mask.getvalue(), file_name="tryon_mask.png", mime="image/png")

        except Exception as e:
            st.error(f"Something went wrong: {e}")
    else:
        st.info("Upload an image and click **Run Virtual Try-On**.")
