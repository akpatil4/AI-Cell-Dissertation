import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
from PIL import Image

st.set_page_config(layout="wide", page_title="Cell Segmentation Dashboard")

DEFAULT_ORANGE = "Image4.png"
DEFAULT_GREEN  = "Image5.png"

# ─────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────

def load_image_color(uploaded_file=None, default_path=None):
    """Load image in BGR then convert to RGB (colour)."""
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    elif default_path and os.path.exists(default_path):
        img = cv2.imread(default_path, cv2.IMREAD_COLOR)
    else:
        return None
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_image_gray(uploaded_file=None, default_path=None):
    """Load image as grayscale — matches cv2.imread(..., IMREAD_GRAYSCALE)."""
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    elif default_path and os.path.exists(default_path):
        img = cv2.imread(default_path, cv2.IMREAD_GRAYSCALE)
    else:
        return None
    return img


def separate_channels(img):
    return {
        "Red":       img[:, :, 0],
        "Green":     img[:, :, 1],
        "Blue":      img[:, :, 2],
    }


def channel_display(name, ch):
    """Return an RGB array colourised for a single-channel array."""
    if name == "Red":
        d = np.zeros((*ch.shape, 3), dtype=np.uint8); d[:, :, 0] = ch; return d
    if name == "Green":
        d = np.zeros((*ch.shape, 3), dtype=np.uint8); d[:, :, 1] = ch; return d
    if name == "Blue":
        d = np.zeros((*ch.shape, 3), dtype=np.uint8); d[:, :, 2] = ch; return d
    return cv2.cvtColor(ch, cv2.COLOR_GRAY2RGB)


def fill_mask(contours, shape):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    for c in contours:
        arr = c.reshape(-1, 1, 2) if c.ndim == 2 else c
        cv2.fillPoly(mask, [arr], 255)
    return mask > 0


def _iou(pred, gt):
    i = np.logical_and(pred, gt).sum()
    u = np.logical_or(pred, gt).sum()
    return float(i / u) if u else 0.0


def _dice(pred, gt):
    i = np.logical_and(pred, gt).sum()
    d = pred.sum() + gt.sum()
    return float(2 * i / d) if d else 0.0


# ─────────────────────────────────────────────────────────────────
#  WATERSHED-FOURIER PIPELINE (matches notebooks exactly)
# ─────────────────────────────────────────────────────────────────

def preprocess_green(gray):
    """
    Matches watershed_green.ipynb Cell 1 exactly:
    - Zero pixels below 40
    - Binarise remaining
    - Morphological open (2 iters) + close (1 iter)
    - Return cleaned binary image (replaces gray for all downstream steps)
    """
    img = gray.copy()
    img[img < 40] = 0
    binary = np.zeros_like(img, dtype=np.uint8)
    binary[img > 0] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
    return cleaned


def run_watershed_fourier(gray, K, otsu_frac, min_pts, color_overlay_img=None):
    """
    Core watershed + Fourier pipeline.
    
    gray:              grayscale input (already preprocessed if green)
    K:                 number of Fourier descriptors to retain per side
    otsu_frac:         fraction of distance transform max for foreground seeds
    min_pts:           minimum contour length to keep
    color_overlay_img: original colour image to draw contours on (matches notebook behaviour)
    """
    laplace = cv2.Laplacian(gray, cv2.CV_64F)
    laplace = cv2.convertScaleAbs(laplace)

    _, thresh = cv2.threshold(laplace, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, otsu_frac * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(thresh, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers += 1
    markers[unknown == 255] = 0

    img_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    markers = cv2.watershed(img_color, markers)

    # --- Fourier descriptor refinement (Cell 3/4 in notebooks) ---
    contour_list, length_list = [], []
    MAX_LEN = 3000

    for label in np.unique(markers):
        if label == -1 or label == 1:
            continue
        mask = np.zeros_like(markers, dtype=np.uint8)
        mask[markers == label] = 255
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            continue
        cnt = cnts[0]

        contour_complex = np.empty(cnt.shape[0], dtype=complex)
        for i in range(cnt.shape[0]):
            contour_complex[i] = complex(cnt[i][0][0], cnt[i][0][1])
        fourier_result = np.fft.fft(contour_complex)
        fourier_result[K:-K] = 0
        contour_reconstructed = np.fft.ifft(fourier_result)
        rec = np.array([[int(p.real), int(p.imag)] for p in contour_reconstructed], dtype=np.int32)

        if len(rec) < MAX_LEN and len(rec) > min_pts:
            contour_list.append(rec)
            length_list.append(len(rec))

    if not contour_list:
        overlay = color_overlay_img.copy() if color_overlay_img is not None else cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        return [], overlay

    # --- Remove nested contours (Cell 5/6 in notebooks) ---
    length_list, contour_list = zip(*sorted(
        zip(length_list, contour_list), key=lambda x: x[0], reverse=True
    ))

    outer_contour_list = []
    for i in range(len(length_list)):
        redundant = False
        if len(outer_contour_list) == 0:
            outer_contour_list.append(contour_list[i])
        else:
            for contour in outer_contour_list:
                for point in contour_list[i]:
                    if cv2.pointPolygonTest(contour, [int(point[0]), int(point[1])], False) != -1:
                        redundant = True
                        break
                if redundant:
                    break
            if not redundant:
                outer_contour_list.append(contour_list[i])

    # --- Draw on colour image (matches notebook: reload original colour) ---
    if color_overlay_img is not None:
        overlay = color_overlay_img.copy()
    else:
        overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(overlay, outer_contour_list, -1, (0, 255, 0), 2)

    return outer_contour_list, overlay


# ─────────────────────────────────────────────────────────────────
#  CELLPOSE
# ─────────────────────────────────────────────────────────────────

def run_cellpose(gray, color_overlay_img=None):
    try:
        from cellpose import models
        model = models.CellposeModel(gpu=False)
        masks, _, _ = model.eval(gray, diameter=None)
        overlay = color_overlay_img.copy() if color_overlay_img is not None else cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        contours_out = []
        for label in np.unique(masks):
            if label == 0:
                continue
            m = (masks == label).astype(np.uint8)
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, cnts, -1, (255, 80, 0), 2)
            contours_out.extend(cnts)
        return overlay, contours_out, int(masks.max()), None
    except Exception as e:
        overlay = color_overlay_img.copy() if color_overlay_img is not None else cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        return overlay, [], 0, str(e)


# ═════════════════════════════════════════════════════════════════
#  SIDEBAR NAVIGATION
# ═════════════════════════════════════════════════════════════════

st.sidebar.title("Cell Segmentation Dashboard")
page = st.sidebar.radio(
    "Navigate",
    [
        "1 — Upload & Channels",
        "2 — Segmentation",
        "3 — Annotation",
        "4 — Evaluation",
    ],
)

# Persistent session state
for key, default in [
    ("img_color", None), ("img_gray", None),
    ("selected_channel", "Grayscale"),
    ("contours", []), ("overlay", None), ("gt_df", None),
    ("image_source", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ═════════════════════════════════════════════════════════════════
#  PAGE 1 — UPLOAD & CHANNEL SEPARATION
# ═════════════════════════════════════════════════════════════════
if page == "1 — Upload & Channels":
    st.title("Page 1 — Upload & Channel Separation")

    st.subheader("Select image source")
    source = st.radio(
        "Choose an image",
        ["Upload custom image", f"Default orange ({DEFAULT_ORANGE})", f"Default green ({DEFAULT_GREEN})"],
    )

    uploaded = None
    if source == "Upload custom image":
        uploaded = st.file_uploader(
            "Upload a histology image",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
        )
        if uploaded is not None:
            # Need to read twice — once for color, once for gray
            file_bytes = uploaded.read()
            arr = np.asarray(bytearray(file_bytes), dtype=np.uint8)
            img_color = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
            img_gray = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
            st.session_state.image_source = "custom"
        else:
            img_color, img_gray = None, None
    elif "orange" in source.lower():
        img_color = load_image_color(default_path=DEFAULT_ORANGE)
        img_gray = load_image_gray(default_path=DEFAULT_ORANGE)
        st.session_state.image_source = "orange"
    else:
        img_color = load_image_color(default_path=DEFAULT_GREEN)
        img_gray = load_image_gray(default_path=DEFAULT_GREEN)
        st.session_state.image_source = "green"

    if img_color is None or img_gray is None:
        st.error("Could not load the image. Upload one or ensure default images exist in the working directory.")
        st.stop()

    st.session_state.img_color = img_color
    st.session_state.img_gray = img_gray

    st.subheader("Original Image")
    st.image(img_color, use_column_width=True)

    st.subheader("Colour Channel Separation")
    channels = separate_channels(img_color)
    cols = st.columns(3)
    for col, (name, ch) in zip(cols, channels.items()):
        with col:
            st.markdown(f"**{name}**")
            st.image(channel_display(name, ch), use_column_width=True)

    st.success(
        f"Image loaded ({st.session_state.image_source}). "
        "Head to **Page 2** to run segmentation."
    )


# ═════════════════════════════════════════════════════════════════
#  PAGE 2 — SEGMENTATION PIPELINE
# ═════════════════════════════════════════════════════════════════
elif page == "2 — Segmentation":
    st.title("Page 2 — Segmentation Pipeline")

    if st.session_state.img_color is None:
        st.warning("Load an image on Page 1 first.")
        st.stop()

    img_color = st.session_state.img_color
    img_gray = st.session_state.img_gray
    src = st.session_state.image_source

    # Sidebar controls
    st.sidebar.subheader("Method")
    method = st.sidebar.selectbox(
        "Segmentation method", ["Watershed-Fourier", "Cellpose", "CPN (placeholder)"]
    )

    # ── Watershed-Fourier ──
    if method == "Watershed-Fourier":
        st.sidebar.subheader("Parameters")

        # Set defaults based on image source
        default_K = 20 if src == "green" else 50
        default_preprocess = True if src == "green" else False

        K = st.sidebar.slider(
            "K — Fourier descriptors", 2, 150, default_K,
            help="Frequency components to keep per side. Orange: 50 · Green: 20.",
        )
        otsu_frac = st.sidebar.slider(
            "Distance threshold (fraction of max)", 0.05, 0.50, 0.20, 0.05,
            help="Sets foreground seed certainty. Lower → more seeds → more segments.",
        )
        min_pts = st.sidebar.slider(
            "Minimum contour length (pts)", 5, 200, 20,
            help="Contours shorter than this are rejected as noise.",
        )
        preprocess = st.sidebar.checkbox(
            "Green preprocessing (noise removal + morphology)",
            value=default_preprocess,
            help="Zeros pixels < 40, binarises, then morphological open/close. "
                 "Required for Image5/green channel. Do NOT use for orange.",
        )

        with st.expander("Parameter guide"):
            st.markdown("""
| Parameter | Orange (Image4) | Green (Image5) |
|---|---|---|
| **K** | 50 | 20 |
| **Distance threshold** | 0.20 | 0.20 |
| **Min contour length** | 20 | 20 |
| **Preprocessing** | Off | **On** |
""")

        st.markdown(
            f"**Input:** Full grayscale image loaded via `cv2.IMREAD_GRAYSCALE` "
            f"(matches notebook pipeline). Shape: `{img_gray.shape}`"
        )

        if st.button("▶  Run Segmentation"):
            with st.spinner("Running Watershed-Fourier…"):
                # Apply green preprocessing if selected (modifies the grayscale input)
                if preprocess:
                    gray_input = preprocess_green(img_gray)
                else:
                    gray_input = img_gray.copy()

                contours, overlay = run_watershed_fourier(
                    gray_input, K, otsu_frac, min_pts,
                    color_overlay_img=img_color  # draw on original colour image
                )
            st.session_state.contours = contours
            st.session_state.overlay = overlay
            st.rerun()

        st.markdown("---")
        if st.session_state.overlay is not None:
            st.subheader("Result")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original image**")
                st.image(img_color, use_column_width=True)
            with col2:
                st.markdown("**Segmentation overlay**")
                st.image(st.session_state.overlay, use_column_width=True)
            st.metric("Detected structures", len(st.session_state.contours))
            st.caption(
                "Adjust **K** in the sidebar and click **Run Segmentation** again "
                "to see the effect of changing Fourier descriptor count."
            )
        else:
            st.info("Click **▶ Run Segmentation** to start.")

    # ── Cellpose ──
    elif method == "Cellpose":
        st.sidebar.subheader("Parameters")
        st.sidebar.info("Cellpose uses automatic parameter detection via its pretrained model.")

        if st.button("▶  Run Cellpose"):
            with st.spinner("Running Cellpose (may take ~30 s on CPU)…"):
                ov, cnts, n, err = run_cellpose(img_gray, color_overlay_img=img_color)
            if err:
                st.error(f"Cellpose error: {err}")
            else:
                st.session_state.overlay = ov
                st.session_state.contours = cnts
                st.rerun()

        if st.session_state.overlay is not None:
            col1, col2 = st.columns(2)
            col1.image(img_color, use_column_width=True, caption="Original")
            col2.image(st.session_state.overlay, use_column_width=True, caption="Cellpose")
            st.metric("Detected cells", len(st.session_state.contours))

    # ── CPN placeholder ──
    else:
        st.info(
            "CPN integration is a placeholder — paste your CPN inference code into "
            "`dashboard.py` under this branch."
        )


# ═════════════════════════════════════════════════════════════════
#  PAGE 3 — ANNOTATION & GROUND TRUTH
# ═════════════════════════════════════════════════════════════════
elif page == "3 — Annotation":
    st.title("Page 3 — Annotation & Ground Truth")

    if st.session_state.img_color is None:
        st.warning("Load an image on Page 1 first.")
        st.stop()

    img_color = st.session_state.img_color
    img_gray = st.session_state.img_gray

    try:
        from streamlit_drawable_canvas import st_canvas
        CANVAS = True
    except ImportError:
        CANVAS = False

    left, right = st.columns([3, 1])

    with left:
        h, w = img_gray.shape

        st.markdown("**Crop region** — select the area to annotate")
        c1, c2, c3, c4 = st.columns(4)
        x1  = c1.number_input("X start",  0, w - 10, 0)
        y1  = c2.number_input("Y start",  0, h - 10, 0)
        cw  = c3.number_input("Width",   50, min(600, w - int(x1)), min(400, w))
        ch_ = c4.number_input("Height",  50, min(600, h - int(y1)), min(400, h))
        x1, y1, cw, ch_ = int(x1), int(y1), int(cw), int(ch_)

        crop_rgb = img_color[y1:y1 + ch_, x1:x1 + cw]

        if CANVAS:
            st.markdown(
                "**Draw ground-truth contours** — polygon mode: click to place vertices, "
                "double-click to close each contour"
            )
            canvas_result = st_canvas(
                fill_color="rgba(0,255,0,0.08)",
                stroke_width=2,
                stroke_color="#00FF00",
                background_image=Image.fromarray(crop_rgb),
                update_streamlit=True,
                height=ch_,
                width=cw,
                drawing_mode="polygon",
                key="canvas",
            )

            if canvas_result.json_data:
                objects = canvas_result.json_data.get("objects", [])
                rows = []
                for cid, obj in enumerate(objects):
                    for pt in obj.get("path", []):
                        if len(pt) == 3:
                            rows.append({
                                "contour_id": cid,
                                "x": int(pt[1]) + x1,
                                "y": int(pt[2]) + y1,
                                "colour": st.session_state.image_source or "unknown",
                            })
                if rows:
                    df_ann = pd.DataFrame(rows)
                    st.dataframe(df_ann.head(30))
                    st.download_button(
                        "Export annotations as CSV",
                        df_ann.to_csv(index=False),
                        "annotations.csv",
                        "text/csv",
                    )
        else:
            st.image(crop_rgb, caption="Crop preview", use_column_width=True)
            st.warning(
                "Install `streamlit-drawable-canvas` to enable drawing:\n"
                "```\npip install streamlit-drawable-canvas\n```"
            )

    with right:
        st.markdown("**Load existing annotations**")
        gt_file = st.file_uploader("Upload GT CSV", type=["csv"])
        if gt_file:
            df = pd.read_csv(gt_file)
            st.session_state.gt_df = df
            st.dataframe(df.head(40))
        else:
            # Auto-detect GT files
            for fname in ["orange_gt.csv", "green_gt.csv"]:
                if os.path.exists(fname):
                    if st.checkbox(f"Load {fname}"):
                        df = pd.read_csv(fname)
                        st.session_state.gt_df = df
                        st.dataframe(df.head(40))
                        break


# ═════════════════════════════════════════════════════════════════
#  PAGE 4 — EVALUATION
# ═════════════════════════════════════════════════════════════════
elif page == "4 — Evaluation":
    st.title("Page 4 — Evaluation")

    if st.session_state.img_color is None:
        st.warning("Load an image on Page 1 first.")
        st.stop()

    img_color = st.session_state.img_color
    img_gray = st.session_state.img_gray
    h, w = img_gray.shape

    left, right = st.columns(2)

    # ── ground truth ──
    with left:
        st.subheader("Ground Truth")
        gt_src = st.radio(
            "GT source",
            ["Upload CSV", "Loaded from Page 3", "Auto-detect file"],
        )
        gt_df = None
        if gt_src == "Upload CSV":
            f = st.file_uploader("CSV file", type=["csv"])
            if f:
                gt_df = pd.read_csv(f)
                st.session_state.gt_df = gt_df
        elif gt_src == "Loaded from Page 3":
            gt_df = st.session_state.gt_df
        elif gt_src == "Auto-detect file":
            for fname in ["orange_gt.csv", "green_gt.csv"]:
                if os.path.exists(fname):
                    gt_df = pd.read_csv(fname)
                    st.session_state.gt_df = gt_df
                    st.success(f"Loaded {fname}")
                    break

        gt_contours = []
        if gt_df is not None:
            for cid in gt_df["contour_id"].unique():
                pts = gt_df[gt_df["contour_id"] == cid][["x", "y"]].values.astype(np.int32)
                if len(pts) >= 3:
                    gt_contours.append(pts)
            st.info(f"{len(gt_contours)} ground-truth contour(s) loaded.")

    # ── predictions ──
    with right:
        st.subheader("Predictions")
        pred_contours = st.session_state.contours
        st.info(f"{len(pred_contours)} predicted contour(s) from Page 2 segmentation.")

        # Optional: filter to a crop region (matches notebook behaviour)
        st.markdown("**Optional: filter predictions to a crop region**")
        use_crop = st.checkbox("Filter contours to bounding box", value=False)
        if use_crop:
            cc1, cc2, cc3, cc4 = st.columns(4)
            tl_x = cc1.number_input("Top-left X", 0, w, 0)
            tl_y = cc2.number_input("Top-left Y", 0, h, 0)
            br_x = cc3.number_input("Bottom-right X", 0, w, w)
            br_y = cc4.number_input("Bottom-right Y", 0, h, h)

            filtered = []
            for con in pred_contours:
                first_pt = con[0] if con.ndim == 2 else con[0][0]
                if tl_x <= first_pt[0] <= br_x and tl_y <= first_pt[1] <= br_y:
                    filtered.append(con)
            pred_contours = filtered
            st.info(f"{len(pred_contours)} contour(s) after filtering.")

        if not pred_contours:
            st.warning("Run segmentation on Page 2 first (or adjust crop filter).")

    # ── metrics ──
    if gt_contours and pred_contours:
        gt_mask   = fill_mask(gt_contours, (h, w))
        pred_mask = fill_mask(pred_contours, (h, w))

        iou_score  = _iou(pred_mask, gt_mask)
        dice_score = _dice(pred_mask, gt_mask)
        precision  = float(np.logical_and(pred_mask, gt_mask).sum() / pred_mask.sum()) if pred_mask.sum() else 0.0
        recall     = float(np.logical_and(pred_mask, gt_mask).sum() / gt_mask.sum()) if gt_mask.sum() else 0.0

        st.markdown("---")
        st.subheader("Metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("IoU",       f"{iou_score:.4f}")
        m2.metric("Dice",      f"{dice_score:.4f}")
        m3.metric("Precision", f"{precision:.4f}")
        m4.metric("Recall",    f"{recall:.4f}")

        st.subheader("Overlay Comparison")
        comparison = img_color.copy()

        # Ground truth = blue, predictions = green
        for c in gt_contours:
            cv2.polylines(comparison, [c.reshape(-1, 1, 2)], True, (0, 80, 255), 2)
        for c in pred_contours:
            arr = c.reshape(-1, 1, 2) if c.ndim == 2 else c
            cv2.drawContours(comparison, [arr], -1, (0, 220, 0), 2)

        st.image(comparison, caption="Green = Predicted  |  Blue = Ground Truth", use_column_width=True)

    elif gt_contours or pred_contours:
        st.info("Load both ground truth and predictions to compute metrics.")
