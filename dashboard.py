import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity

st.title("Lung Histology Cell Segmentation")
st.sidebar.header("Settings")

# ── Channel selector ──
channel = st.sidebar.selectbox("Channel", ["Orange (Image 4)", "Green (Image 5)"])
n_descriptors = st.sidebar.slider("Fourier Descriptors (N_d)", 1, 50, 20)
min_len = st.sidebar.slider("Min Contour Length", 10, 200, 20)

# ── Load image based on selection ──
img_path = "Image4.png" if "Orange" in channel else "Image5.png"
img_bgr  = cv2.imread(img_path, cv2.IMREAD_COLOR)
img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
gray     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# ── Run pipeline ──
clahe     = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16,16))
gray_eq   = clahe.apply(gray)
blurred   = cv2.GaussianBlur(gray_eq, (5,5), 0)
laplace   = cv2.Laplacian(blurred, cv2.CV_8U, ksize=3)
_, thresh = cv2.threshold(laplace, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

dist      = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist, 0.2 * dist.max(), 255, 0)
sure_fg   = np.uint8(sure_fg)
kernel    = np.ones((3,3), np.uint8)
sure_bg   = cv2.dilate(thresh, kernel, iterations=3)
unknown   = cv2.subtract(sure_bg, sure_fg)
_, markers = cv2.connectedComponents(sure_fg)
markers   = markers + 1
markers[unknown == 255] = 0
img_ws    = img_bgr.copy()
cv2.watershed(img_ws, markers)

# ── Fourier smoothing + contour drawing ──
img_overlay = img_rgb.copy()
contour_count = 0

for label in np.unique(markers):
    if label == -1 or label == 1:
        continue
    mask = np.zeros_like(markers, dtype=np.uint8)
    mask[markers == label] = 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        continue
    contour = contours[0]
    if len(contour) < min_len or len(contour) > 3000:
        continue
    contour_complex = np.array([complex(p[0][0], p[0][1]) for p in contour])
    fourier_result  = np.fft.fft(contour_complex)
    fourier_result[n_descriptors:-n_descriptors] = 0
    recon = np.fft.ifft(fourier_result)
    recon_pts = np.array([[int(p.real), int(p.imag)] for p in recon], dtype=np.int32)
    cv2.drawContours(img_overlay, [recon_pts], -1, (0, 255, 0), 2)
    contour_count += 1

# ── Display ──
st.subheader("Segmentation Result")
col1, col2 = st.columns(2)
with col1:
    st.image(img_rgb, caption="Original", use_column_width=True)
with col2:
    st.image(img_overlay, caption=f"Segmented (N_d = {n_descriptors})", use_column_width=True)

st.metric("Structures Detected", contour_count)

# ── Metrics section ──
st.subheader("Quantitative Evaluation")
uploaded_mask = st.file_uploader("Upload ground truth mask (PNG)", type=["png"])

if uploaded_mask is not None:
    gt_bytes  = np.frombuffer(uploaded_mask.read(), np.uint8)
    gt_mask   = cv2.imdecode(gt_bytes, cv2.IMREAD_GRAYSCALE)
    _, gt_bin = cv2.threshold(gt_mask, 127, 1, cv2.THRESH_BINARY)

    # Build predicted mask from contours
    pred_mask = np.zeros(gray.shape, dtype=np.uint8)
    for label in np.unique(markers):
        if label == -1 or label == 1:
            continue
        mask = np.zeros_like(markers, dtype=np.uint8)
        mask[markers == label] = 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours) == 0:
            continue
        if len(contours[0]) < min_len or len(contours[0]) > 3000:
            continue
        cv2.drawContours(pred_mask, contours, -1, 1, -1)  # filled

    pred_bin = (pred_mask > 0).astype(np.uint8)
    gt_bin   = (gt_bin > 0).astype(np.uint8)

    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union        = np.logical_or(pred_bin,  gt_bin).sum()
    iou          = intersection / union if union > 0 else 0

    dice = (2 * intersection) / (pred_bin.sum() + gt_bin.sum()) \
           if (pred_bin.sum() + gt_bin.sum()) > 0 else 0

    precision = intersection / pred_bin.sum() if pred_bin.sum() > 0 else 0
    recall    = intersection / gt_bin.sum()   if gt_bin.sum()  > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("IoU",       f"{iou:.3f}")
    col2.metric("Dice",      f"{dice:.3f}")
    col3.metric("Precision", f"{precision:.3f}")
    col4.metric("Recall",    f"{recall:.3f}")