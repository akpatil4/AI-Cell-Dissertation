import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import io

st.set_page_config(layout="wide")
st.title("Ground Truth Annotation Tool — Full Image")

# ─────────────────────────────────────────────
# 1. LOAD IMAGE
# ─────────────────────────────────────────────
IMG_PATH   = "Image4.png"
img_bgr    = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)
img_rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
H, W       = img_rgb.shape[:2]

# ─────────────────────────────────────────────
# 2. DISPLAY SETTINGS
#    Canvas width fixed — height scales to maintain aspect ratio
# ─────────────────────────────────────────────
CANVAS_W   = 1200
CANVAS_H   = int(H * CANVAS_W / W)
SCALE_X    = W / CANVAS_W
SCALE_Y    = H / CANVAS_H

# ─────────────────────────────────────────────
# 3. SIDEBAR CONTROLS
# ─────────────────────────────────────────────
st.sidebar.header("Annotation Settings")
stroke_width = st.sidebar.slider("Brush size", 1, 20, 4)
stroke_color = st.sidebar.color_picker("Draw colour", "#00FF00")
drawing_mode = st.sidebar.selectbox("Drawing mode", ["freedraw", "polygon"])
st.sidebar.markdown("---")
st.sidebar.markdown("**How to use:**")
st.sidebar.markdown("- Draw over the structures you want to annotate")
st.sidebar.markdown("- Use freedraw for smooth outlines")
st.sidebar.markdown("- Use polygon for precise boundaries")
st.sidebar.markdown("- Click Export when done")

# ─────────────────────────────────────────────
# 4. CANVAS
# ─────────────────────────────────────────────
img_pil    = Image.fromarray(img_rgb).resize((CANVAS_W, CANVAS_H))

canvas_result = st_canvas(
    background_image=img_pil,
    stroke_color=stroke_color,
    stroke_width=stroke_width,
    drawing_mode=drawing_mode,
    width=CANVAS_W,
    height=CANVAS_H,
    key="annotation_canvas"
)

# ─────────────────────────────────────────────
# 5. MASK EXTRACTION
#    Scale canvas drawing back to full image resolution
# ─────────────────────────────────────────────
if canvas_result.image_data is not None:
    # Canvas returns RGBA — extract drawn pixels
    canvas_rgba  = canvas_result.image_data.astype(np.uint8)
    canvas_alpha = canvas_rgba[:, :, 3]   # alpha channel — nonzero where drawn

    # Anything drawn = foreground
    drawn_mask_small = (canvas_alpha > 0).astype(np.uint8) * 255

    # Scale back to full image resolution
    full_mask = cv2.resize(
        drawn_mask_small,
        (W, H),
        interpolation=cv2.INTER_NEAREST
    )

    # ── Preview ──
    st.subheader("Mask Preview")
    col1, col2 = st.columns(2)
    with col1:
        st.image(img_rgb, caption="Original Image", use_column_width=True)
    with col2:
        # Overlay mask on image for preview
        overlay     = img_rgb.copy()
        overlay[full_mask > 0] = [0, 255, 0]
        blended     = cv2.addWeighted(img_rgb, 0.6, overlay, 0.4, 0)
        st.image(blended, caption="Annotation Overlay", use_column_width=True)

    st.metric("Annotated pixels", int((full_mask > 0).sum()))
    st.metric("Coverage (%)", f"{100 * (full_mask > 0).mean():.2f}%")

    # ─────────────────────────────────────────────
    # 6. EXPORT
    # ─────────────────────────────────────────────
    st.subheader("Export Ground Truth Mask")

    buffer = io.BytesIO()
    Image.fromarray(full_mask).save(buffer, format="PNG")
    buffer.seek(0)

    st.download_button(
        label="Download Full Resolution Mask (PNG)",
        data=buffer,
        file_name="ground_truth_mask.png",
        mime="image/png"
    )

    # Also export binary (0/1) numpy array option
    buffer_npy = io.BytesIO()
    np.save(buffer_npy, (full_mask > 0).astype(np.uint8))
    buffer_npy.seek(0)

    st.download_button(
        label="Download as NumPy array (.npy)",
        data=buffer_npy,
        file_name="ground_truth_mask.npy",
        mime="application/octet-stream"
    )