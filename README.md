Automated Cell Segmentation in Lung Histology Images:
A classical image processing pipeline for segmenting cells and tissue structures from fluorescence histology images of lung tissue, developed as a final-year dissertation project.

The problem:
Histology images are used in pathology to diagnose disease, but manual cell counting is time-consuming, error-prone, and subjective. This project works with a single unlabelled fluorescence image of stained lung tissue containing four colour channels. No pre-existing ground-truth annotations were available.

Deep learning methods(CPN,Cellpose,U-Net) were evaluated but struggled with the larger, irregular tissue structures in the orange and green channels - and U-Net could not be trained without labelled data. This motivated the development of a classical pipeline that operates without supervision.
