# Automated Cell Segmentation for counting in Lung Histology Images


A classical image processing pipeline for automated cell and tissue segmentation for cell counting in fluorescence histology images of lung tissue, developed as a final-year dissertation project. The main pipeline combines watershed segmentation with Fourier descriptor boundary refinement to produce smooth, biologically plausible contours without requiring any labelled training data.

---


## Project Structure

```
├── watershed_orange.ipynb
├── watershed_green.ipynb
├── cpn_red.ipynb
├── cpn_blue.ipynb
├── cpn_orange.ipynb
├── cpn_green.ipynb
├── cellpose.ipynb
├── Image1.png
├── Image2.png
├── Image3.png
├── Image4.png
├── Image5.png
├── dashboard.py
                 # Streamlit segmentation dashboard

├── orange_gt.csv                 # Histologist ground-truth annotations (orange)
├── green_gt.csv                  # Histologist ground-truth annotations (green)
└── README.md
```

---

## Segmentation Dashboard

##Setup
```bash
pip install streamlit opencv-python numpy pandas matplotlib
```
##Running the Dashboard
```bash
streamlit run dashboard.py
```

An interactive Streamlit dashboard combining the full pipeline into a single application — no coding required.

**Page 1 — Upload & Channels:** Load any histology image, view channel separation.

**Page 2 — Segmentation:** Run the watershed-Fourier pipeline with adjustable parameters. K slider, distance threshold, intensity filter — results update in real time.

**Page 3 — Evaluation:** Load ground-truth CSV files, auto-detect the annotated region, filter predictions to that region, compute IoU, Dice, Precision and Recall with full overlay comparison.


## Running the Notebooks
### Watershed-Fourier Pipeline (VS Code / Jupyter)
Open in VS Code or Jupyter:
- `watershed_orange.ipynb` — orange channel pipeline and evaluation
- `watershed_green.ipynb` — green channel pipeline and evaluation

Ensure `Image4.png` and `Image5.png` are in the same directory.

### CPN and Cellpose (Google Colab)
- `cpn_red.ipynb`, `cpn_blue.ipynb`, `cpn_orange.ipynb`, `cpn_green.ipynb`
Each notebook runs CPN on the corresponding channel. Upload the relevant channel image when needed.

- `cellpose.ipynb`

In this case CPN and Cellpose notebooks were developed and tested on Google Colab however they can be run on any notebook format desired.

Upload the relevant channel image when prompted.





## Results

Evaluated against expert ground-truth annotations obtained from a histologist at Glenfield Hospital:

| Channel | Method | IoU | Dice | Precision | Recall |
|---------|--------|-----|------|-----------|--------|
| Orange (K=50) | Watershed-Fourier | 0.7327 | 0.8458 | 0.8727 | 0.8204 |
| Green (K=70) | Watershed-Fourier | 0.6130 | 0.7600 | 0.8958 | 0.6527 |

### Structure Counts

| Channel | Method | Structures | Total Area (px) | Mean Area (px) | Std Area (px) |
|---------|--------|------------|-----------------|----------------|---------------|
| Red | CPN | 189 | 34090 | 180.37 | 313.46 |
| Blue | CPN | 2107 | 208548 | 98.98 | 42.27 |
| Orange | Watershed-Fourier | 161 | 172405 | 1070.84 | 2848.34 |
| Green | Watershed-Fourier | 153 | 186896 | 1221.54 | 3609.65 |

---







## References

- Upschulte et al. (2022). Contour Proposal Networks for Biomedical Instance Segmentation. Medical Image Analysis.
- Stringer et al. (2021). Cellpose: a generalist algorithm for cellular segmentation. Nature Methods.
- Gonzalez & Woods (2018). Digital Image Processing, 4th ed. Pearson.
- Vincent, L. & Soille, P. (1991). Watersheds in digital spaces: an efficient algorithm based on immersion simulations. IEEE Transactions on Pattern Analysis and Machine Intelligence, 13(6), 583–598.
- Otsu, N. (1979). A threshold selection method from gray-level histograms. IEEE Transactions on Systems, Man, and Cybernetics, 9(1), 62–66.
- Kuhl, F.P. & Giardina, C.R. (1982). Elliptic Fourier features of a closed contour. Computer Graphics and Image Processing, 18(3), 236–258.




