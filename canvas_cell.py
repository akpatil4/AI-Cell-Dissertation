import streamlit as st
import pandas as pd
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import cv2
import numpy as np
canvas_result = st_canvas(background_image=Image.open("Image5.png"), stroke_width=2, stroke_color="#FF0000", update_streamlit=True, height=500, width=500)
if canvas_result.image_data is not None:
    st.image(canvas_result.image_data)
if canvas_result.json_data is not None:
    st.json(canvas_result.json_data)
    objects = pd.json_normalize(canvas_result.json_data["objects"])
    if len(objects) > 0:
      st.write(objects.loc[len(objects) - 1, "path"])

if canvas_result.image_data is not None:

    # Extract drawing
    drawn = canvas_result.image_data[:, :, 0]

    # Convert to binary mask
    gt_mask = (drawn > 0).astype(np.uint8)

    st.write("Preview of Ground Truth Mask:")
    st.image(gt_mask * 255)

    # Save button
    if st.button("Save Ground Truth Mask"):
        st.download_button(
    label="Download Ground Truth Mask",
     data= buffer.getvalue(),
    file_name="gt_mask_image5.png",
    mime="image/png"
)
        cv2.imwrite("gt_mask_image5.png", gt_mask * 255)
        st.success("Mask saved successfully!")