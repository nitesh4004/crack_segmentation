# Geotechnical Desiccation Crack Analysis

This repository provides a **Streamlit**-based web application for automatic segmentation and quantitative analysis of desiccation crack patterns in clayey soils using a YOLO-based pipeline and classical image processing. [page:1][page:2]

## Features

- YOLO-based crack segmentation with automatic model download from Google Drive (`best.pt`). [page:2]  
- Adaptive thresholding and morphological cleaning to refine crack networks. [page:2]  
- Skeletonization of the crack network and extraction of nodes, segments, and crack centerlines. [page:2]  
- Computation of key geometric descriptors such as surface crack ratio, clod statistics, crack density, and estimated crack volume following Tang et al. (2012) logic. [page:2]

## Crack Metrics

The app computes the following parameters from a calibrated crack image: [page:2]

- **Surface crack ratio (R_sc)**: Ratio of crack area to total specimen surface area (%). [page:2]  
- **Number of clods (N_c)**: Count of independent closed soil areas surrounded by cracks. [page:2]  
- **Average area of clods (A_av)**: Mean clod area (cm²). [page:2]  
- **Number of nodes per unit area (N_n)**: Intersection and end nodes per unit area (cm⁻²). [page:2]  
- **Crack segments per unit area (N_seg)**: Distinct crack segments per unit area (cm⁻²). [page:2]  
- **Average length of cracks (L_av)**: Mean crack centerline length (cm). [page:2]  
- **Crack density (D_c)**: Total crack length per unit area (cm⁻¹). [page:2]  
- **Average width of cracks (W_av)**: Mean crack width estimated from distance transform (cm). [page:2]  
- **Estimated crack volume (V_cr)**: Crack area multiplied by soil layer thickness (cm³). [page:2]

## How It Works

1. **Model handling**  
   - On first run, the app downloads the YOLO model from Google Drive using `gdown` (file ID stored as `MODEL_ID`) and caches the loaded model with `st.cache_resource`. [page:2]

2. **Image processing pipeline**  
   - Converts the uploaded image to grayscale and enhances contrast via CLAHE. [page:2]  
   - Applies YOLO segmentation to obtain a structural crack map and combines it with adaptive thresholding to form a connectivity map. [page:2]  
   - Cleans the binary mask using morphological closing, small object removal, and hole filling. [page:2]  
   - Skeletonizes the cleaned network and identifies nodes, segments, and crack centerlines using `skimage` and `scipy.ndimage`. [page:2]

3. **Visualization**  
   - A 2×2 grid shows original image, binary crack map, skeleton with nodes, and an overlay with the surface crack ratio. [page:2]  
   - A metrics tab displays all geometric parameters in a table, and a definitions tab explains each metric. [page:2]

## Usage

1. **Install dependencies**

Using `requirements.txt`: [page:1][page:2]

