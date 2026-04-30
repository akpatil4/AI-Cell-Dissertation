
# Importing necessary libraries for dashboard
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
from PIL import Image

# Setting Streamlit page configuration for better layout
st.set_page_config(layout="wide", page_title="Cell Segmentation Dashboard")

# Setting default image paths that are used for the model
DEFAULT_ORANGE = "Image4.png"
DEFAULT_GREEN  = "Image5.png"


# Function to load an image in colour (BGR → RGB)
def load_image_color(uploaded_file=None, default_path=None):
    """Load image in BGR then convert to RGB (colour)."""
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    elif default_path and os.path.exists(default_path):
        # OpenCV loads in BGR by default, so it converted to RGB for correct display in Streamlit
        img = cv2.imread(default_path, cv2.IMREAD_COLOR)
    else:
        return None
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# Loading the image in grayscale to be used for watershed
def load_image_gray(uploaded_file=None, default_path=None):
    """Load image as grayscale."""
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    elif default_path and os.path.exists(default_path):
        img = cv2.imread(default_path, cv2.IMREAD_GRAYSCALE)
    else:
        return None
    return img


# Filling the mask with contours to create binary masks for evaluation metrics (IoU, Dice, etc.)
def fill_mask(contours, shape):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    for c in contours:
        arr = c.reshape(-1, 1, 2) if c.ndim == 2 else c
        cv2.fillPoly(mask, [arr], 255)
    return mask > 0

# IoU calculations
def _iou(pred, gt):
    # Intersection over Union (IoU) = (Area of Overlap) / (Area of Union)
    i = np.logical_and(pred, gt).sum()
    u = np.logical_or(pred, gt).sum()
    return float(i / u) if u else 0.0

# Dice coefficient calculations 
# Dice = (2 * Area of Overlap) / (Total number of pixels in both masks) 
def _dice(pred, gt):
    i = np.logical_and(pred, gt).sum()
    d = pred.sum() + gt.sum()
    return float(2 * i / d) if d else 0.0


# ─────────────────────────────────────────────────────────────────
#  WATERSHED-FOURIER PIPELINE 
# ─────────────────────────────────────────────────────────────────

# Preprocessing for the green image: zero out pixels < 30 to reduce noise intensity
def preprocess_green(gray):
    img = gray.copy()
    img[img < 30] = 0
    return img
    
# Preprocessing for the orange image
def preprocess_orange(gray):
    img = gray.copy()
    img[img < 30] = 0
    return img


# Running the full watershed + Fourier pipeline and adding parameters
def run_watershed_fourier(gray, K, otsu_frac, min_pts, color_overlay_img=None):
    """
    Core watershed + Fourier pipeline.
    
    gray:              grayscale input (already preprocessed if green)
    K:                 number of Fourier descriptors to retain per side
    otsu_frac:         fraction of distance transform max for foreground seeds
    min_pts:           minimum contour length to keep
    color_overlay_img: original colour image to draw contours on 
    """

    # Laplacian filtering to enhance edges for watershed
    laplace = cv2.Laplacian(gray, cv2.CV_64F)
    laplace = cv2.convertScaleAbs(laplace)

    #Negating the laplacian to preserve outer boundaries and conducting Otsu's thresholding 
    _, thresh = cv2.threshold(-laplace, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

  # Distance transform- each pixel's value is replaced by its distance to the nearest zero pixel (background).
    dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    #Threshold at otsu_frac = 0.2 to keep pixels confidently inside the structures
    _, sure_fg = cv2.threshold(dist, otsu_frac * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

 # Dilating the edges to get background region
    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(thresh, kernel, iterations=3)
    # Subtracting foreground from background to get the unknown region (where watershed will be applied)
    unknown = cv2.subtract(sure_bg, sure_fg)

   # Labelling each foreground marker with unique number
    _, markers = cv2.connectedComponents(sure_fg)
    markers += 1
    markers[unknown == 255] = 0

    # Running watershed
    img_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    markers = cv2.watershed(img_color, markers)

    #  Fourier descriptor refinement 
    # defining contour_list and length_list to store the refined contours and their lengths for filtering
    contour_list, length_list = [], []
    # Maximun contour length
    MAX_LEN = 3000

   # Iterating through each unique marker label (except -1 for boundaries and 1 for background) to extract contours, apply Fourier descriptors
    for label in np.unique(markers):
        if label == -1 or label == 1:
            continue
        # Creating a binary mask for the current marker label to find contours
        mask = np.zeros_like(markers, dtype=np.uint8)
        mask[markers == label] = 255
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            continue
        cnt = cnts[0]
        
        # Converting contour points to complex numbers for Fourier transform
        contour_complex = np.empty(cnt.shape[0], dtype=complex)
        for i in range(cnt.shape[0]):
            contour_complex[i] = complex(cnt[i][0][0], cnt[i][0][1])
        # Applying Fourier transform, zeroing high-frequency components, and reconstructing the contour with inverse Fourier transform
        fourier_result = np.fft.fft(contour_complex)
        fourier_result[K:-K] = 0
        # Reconstructing the contour using the inverse Fourier transform and converting back to integer coordinates
        contour_reconstructed = np.fft.ifft(fourier_result)
        rec = np.array([[int(p.real), int(p.imag)] for p in contour_reconstructed], dtype=np.int32)

      # Filtering contours based on length to remove noise 
        if len(rec) < MAX_LEN and len(rec) > min_pts:
            contour_list.append(rec)
            length_list.append(len(rec))

    if not contour_list:
        overlay = color_overlay_img.copy() if color_overlay_img is not None else cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        return [], overlay

 
   

    # Drawing contours on overlay 
    if color_overlay_img is not None:
        overlay = color_overlay_img.copy()
    else:
        overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(overlay, contour_list, -1, (0, 255, 0), 2)

    return contour_list, overlay



#  SIDEBAR NAVIGATION

#Title and navigation
st.sidebar.title("Cell Segmentation Dashboard")
page = st.sidebar.radio(
    "Navigate",
    [
        "1 — Upload & Channels",
        "2 — Segmentation",
        "3 — Evaluation",
    ],
)

# Initialising session state variables to store images, contours, and ground truth across pages
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
   # Allowing the user to select the image source: either upload a custom image or use one of the default images (orange or green)
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
            # Needs to read twice — once for color, once for gray
            file_bytes = uploaded.read()
            arr = np.asarray(bytearray(file_bytes), dtype=np.uint8)
            img_color = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
            img_gray = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
            st.session_state.image_source = "custom"
        else:
            img_color, img_gray = None, None
    elif "orange" in source.lower():
        # Loading the default orange image in both colour and grayscale modes. The colour image is converted from BGR to RGB for correct display in Streamlit.
        img_color = load_image_color(default_path=DEFAULT_ORANGE)
        img_gray = load_image_gray(default_path=DEFAULT_ORANGE)
        st.session_state.image_source = "orange"
    else:
        # Loading the default green image
        img_color = load_image_color(default_path=DEFAULT_GREEN)
        img_gray = load_image_gray(default_path=DEFAULT_GREEN)
        st.session_state.image_source = "green"

    if img_color is None or img_gray is None:
        st.error("Could not load the image. Upload one or ensure default images exist in the working directory.")
        st.stop()

   # Storing the loaded images and source information in session state for access on other pages
    st.session_state.img_color = img_color
    st.session_state.img_gray = img_gray

    st.subheader("Original Image")
    st.image(img_color, use_column_width=True)

   # Directing the user to page 2
    st.success(
        f"Image loaded ({st.session_state.image_source}). "
        "Head to **Page 2** to run segmentation."
    )

#  PAGE 2 — SEGMENTATION PIPELINE
elif page == "2 — Segmentation":
    st.title("Page 2 — Segmentation Pipeline")

    if st.session_state.img_color is None:
        st.warning("Load an image on Page 1 first.")
        st.stop()

    img_color = st.session_state.img_color
    img_gray = st.session_state.img_gray
    src = st.session_state.image_source

    # Sidebar controls for segmentation method and parameters
    st.sidebar.subheader("Method")
    method = st.sidebar.selectbox(
        "Segmentation method", ["Watershed-Fourier"]
    )

    # ── Watershed-Fourier ──
    if method == "Watershed-Fourier":
        st.sidebar.subheader("Parameters")

        # Set default parameters
        default_K = 20 if src == "green" else 50
        default_preprocess = True if src == "green" else False
       # Slider for number of fourier descriptors
        K = st.sidebar.slider(
            "K — Fourier descriptors", 2, 150, default_K,
            help="Frequency components to keep per side. Orange: 50 · Green: 20.",
        )
        # Slider for distance threshold which is otsu_frac
        otsu_frac = st.sidebar.slider(
            "Distance threshold (fraction of max)", 0.05, 0.50, 0.20, 0.05,
            help="Sets foreground seed certainty. Lower → more seeds → more segments.",
        )
        # Slider for minimum contour length
        min_pts = st.sidebar.slider(
            "Minimum contour length (pts)", 5, 200, 20,
            help="Contours shorter than this are rejected as noise.",
        )
        # Slider for toggling the pixel intensity preprocessing filter
        preprocess = st.sidebar.checkbox(
            "Preprocessing (noise removal + morphology)",
            value=default_preprocess,
            help="Zeros pixels < 30, binarises, then morphological open/close. "
                 ,
        )

       # Displaying the input image and how it was loaded
        st.markdown(
            f"**Input:** Full grayscale image loaded via `cv2.IMREAD_GRAYSCALE` "
            f". Shape: `{img_gray.shape}`"
        )
        # Button to run the watershed-fourier pipeline with a spinner to indicate processing.
        if st.button("▶  Run Segmentation"):
            with st.spinner("Running Watershed-Fourier…"):
                # Apply green preprocessing if selected 
                if preprocess:
                    gray_input = preprocess_green(img_gray)
                else:
                    gray_input = img_gray.copy()
               # Running the model pipeline and storing the resulting contours and overlay in session state for display and access on the evaluation page
                contours, overlay = run_watershed_fourier(
                    gray_input, K, otsu_frac, min_pts,
                    color_overlay_img=img_color  # draw on original colour image
                )
            st.session_state.contours = contours
            st.session_state.overlay = overlay
            st.rerun()

       # Displaying the original image alongside the segmented image
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

# ═════════════════════════════════════════════════════════════════
#  PAGE 3 — EVALUATION
# ═════════════════════════════════════════════════════════════════

# Evaluation page to compare predicted contours against grounf-truth contours
elif page == "3 — Evaluation":
    st.title("Page 3 — Evaluation")

    if st.session_state.img_color is None:
        st.warning("Load an image on Page 1 first.")
        st.stop()
# Receiving the predicted contours
    img_color = st.session_state.img_color
    img_gray = st.session_state.img_gray
    h, w = img_gray.shape

    # Creating two columns left for ground truth and right for predictions
    left, right = st.columns(2)

    # Allowing the user to upload a ground truth CSV file or auto-detect it since orangee was tested a lot
    with left:
        st.subheader("Ground Truth")
        gt_src = st.radio(
            "GT source",
            ["Upload CSV", "Auto-detect file"],
        )
        gt_df = None
        if gt_src == "Upload CSV":
            f = st.file_uploader("CSV file", type=["csv"])
            if f:
                gt_df = pd.read_csv(f)
                st.session_state.gt_df = gt_df
       # Auto-detecting file if user selects that option
        elif gt_src == "Auto-detect file":
            for fname in ["orange_gt.csv", "green_gt.csv"]:
                if os.path.exists(fname):
                    gt_df = pd.read_csv(fname)
                    st.session_state.gt_df = gt_df
                    st.success(f"Loaded {fname}")
                    break

#  Extracting contours from the ground truth dataframe. 
        gt_contours = []
        # Assuming the gt CSV has columns: contour_id, x, y. Each unique contour_id corresponds to one contour, and the (x, y) pairs are the points of that contour.
        if gt_df is not None:
            for cid in gt_df["contour_id"].unique():
                pts = gt_df[gt_df["contour_id"] == cid][["x", "y"]].values.astype(np.int32)
                # Only keeping contours with 3 or more points to ensure they are valid for evaluation
                if len(pts) >= 3:
                    gt_contours.append(pts)
            st.info(f"{len(gt_contours)} ground-truth contour(s) loaded.")

    # auto-detect crop region from GT coordinates to give accurate results
    gt_bbox = None
    # Adding a bounding box around the gt contours to define the evaluation region
    if gt_df is not None and len(gt_contours) > 0:
        all_x = gt_df["x"].values
        all_y = gt_df["y"].values
        # Adding a small padding around the GT region to catch the resulting contours near edges
        pad = 20
        gt_bbox = {
            "x_min": max(0, int(all_x.min()) - pad),
            "y_min": max(0, int(all_y.min()) - pad),
            "x_max": min(w, int(all_x.max()) + pad),
            "y_max": min(h, int(all_y.max()) + pad),
        }

    # Displaying the predicted contours for the cell count
    with right:
        st.subheader("Predictions")
        pred_contours = list(st.session_state.contours)
        st.info(f"{len(pred_contours)} total predicted contour(s) from Page 2.")

        if not pred_contours:
            st.warning("Run segmentation on Page 2 first.")

    # filtering predictions to Ground Truth region & compute metrics ──
    if gt_contours and pred_contours and gt_bbox is not None:
        # Filter predicted contours to what falls within the gt box
       
        filtered_preds = []
        for con in pred_contours:
            first_pt = con[0] if con.ndim == 2 else con[0][0]
            if (gt_bbox["x_min"] <= first_pt[0] <= gt_bbox["x_max"] and
                gt_bbox["y_min"] <= first_pt[1] <= gt_bbox["y_max"]):
                filtered_preds.append(con)
        
    

       #Compute metrics only if there are predictions in the GT region
        if filtered_preds:
            # creating binary mask for the gt and the predicted contours to compute metrics
            gt_mask   = fill_mask(gt_contours, (h, w))
            pred_mask = fill_mask(filtered_preds, (h, w))
            
            # Calculating IoU, Dice, Precision, and Recall based on the binary masks. The metrics are computed using the logical operations on the masks to find intersections and unions.
            iou_score  = _iou(pred_mask, gt_mask)
            dice_score = _dice(pred_mask, gt_mask)
            # Computing precision and recall
            precision  = float(np.logical_and(pred_mask, gt_mask).sum() / pred_mask.sum()) if pred_mask.sum() else 0.0
            recall     = float(np.logical_and(pred_mask, gt_mask).sum() / gt_mask.sum()) if gt_mask.sum() else 0.0

           #Displaying the computed metrics using streamlit's metric component
            st.subheader("Metrics")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("IoU",       f"{iou_score:.4f}")
            m2.metric("Dice",      f"{dice_score:.4f}")
            m3.metric("Precision", f"{precision:.4f}")
            m4.metric("Recall",    f"{recall:.4f}")

            #  Overlay comparison (cropped to gt region for clarity on results)
            st.subheader("Overlay Comparison")
            comparison = img_color.copy()

            # Drawing the Ground truth bounding box in white
            cv2.rectangle(
                comparison,
                (gt_bbox["x_min"], gt_bbox["y_min"]),
                (gt_bbox["x_max"], gt_bbox["y_max"]),
                (255, 255, 255), 2
            )
            # using blue for ground truth regions on overlay
            for c in gt_contours:
                cv2.polylines(comparison, [c.reshape(-1, 1, 2)], True, (0, 80, 255), 2)
            # using green for predicted contours on overlay
            for c in filtered_preds:
                arr = c.reshape(-1, 1, 2) if c.ndim == 2 else c
                cv2.drawContours(comparison, [arr], -1, (0, 220, 0), 2)

            # Showing full overlay with gt box
            st.image(comparison, caption="White box = evaluation region  |  Green = Predicted  |  Blue = Ground Truth", use_column_width=True)

            # Also showing cropped view of overlay for more detail and focus on the evaluation region
            crop_comp = comparison[gt_bbox["y_min"]:gt_bbox["y_max"], gt_bbox["x_min"]:gt_bbox["x_max"]]
            st.image(crop_comp, caption="Cropped evaluation region (detail view)", use_column_width=True)
        else:
            st.warning("No predicted contours fall within the ground truth region. Check your segmentation results.")

    elif gt_contours or pred_contours:
        st.info("Load both ground truth and predictions to compute metrics.")
