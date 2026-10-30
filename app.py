import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from inference import run_gigapixel_inference

st.set_page_config(layout="wide", page_title="Digital Pathology AI")

st.title("🔬 AI Pathology: Gigapixel Tissue Segmentation")

uploaded_file = st.file_uploader("Upload WSI file (.tif, .svs, .ndpi)", type=['tif', 'svs', 'ndpi'])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        wsi_path = tmp_file.name
        
    if st.button("Run Segmentation Analysis"):
        with st.spinner('Processing...'):
            heatmap = run_gigapixel_inference(wsi_path, "dummy_weights.pth")
            
            heatmap_normalized = np.uint8(255 * heatmap)
            colored_heatmap = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
            
            st.image(colored_heatmap, use_column_width=True)
            
    os.unlink(wsi_path)
