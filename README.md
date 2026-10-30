# AIRA Matrix: Patch-Based Histopathology Segmentation Pipeline

This project demonstrates a production-styled pipeline for segmenting ultra-high-resolution Whole Slide Images (WSIs), mirroring industry standards in Digital Pathology.

## Features
* **Gigapixel Memory Management:** Extracts tiles (patches) sequentially via OpenSlide without loading the multi-gigabyte image into RAM.
* **Deep Learning Inference:** Scaffolds a PyTorch U-Net for tissue/tumor segmentation.
* **Spatial Stitching:** Reconstructs local patch predictions into a global anatomical heatmap.
* **Interactive UI:** Provides a Streamlit frontend for pathologists to upload and view analyses.

## System Prerequisites
Before installing Python dependencies, you must install the OpenSlide C-library on your system:

**Ubuntu / Debian / WSL:**
```bash
sudo apt-get update
sudo apt-get install openslide-tools
```

**Windows:**
Download the Windows binaries from [OpenSlide's website](https://openslide.org/download/) and add the `bin` directory to your system's PATH.

## Installation
1. Clone/unzip this repository.
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
2. Upload a WSI format file (`.tif`, `.svs`, `.ndpi`).
3. Click **Run Segmentation Analysis**.

*(Note: This scaffold uses a randomly initialized U-Net. For real deployment, train the model on a dataset like CAMELYON16 and replace `dummy_weights.pth` with your trained `.pth` file).*
