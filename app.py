import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from skimage.morphology import skeletonize, remove_small_objects, remove_small_holes
from scipy import ndimage
import os
import gdown

# =============================================================================
# 1. PAGE CONFIGURATION & STYLING
# =============================================================================
st.set_page_config(
    page_title="Geotechnical Crack Analysis | Amit Kumar",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    h1, h2, h3 {
        color: #2c3e50;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .stButton>button {
        width: 100%;
        background-color: #2c3e50;
        color: white;
    }
    .stTable {
        font-size: 1.1rem !important;
    }
    </style>
""", unsafe_allow_html=True)

# =============================================================================
# 2. SYSTEM CONFIGURATION & MODEL LOADING
# =============================================================================

# Google Drive File ID for best.pt
MODEL_ID = '1bQUE7ZL8luHRPWPX9P2u3zdtyAtWbQNP'
MODEL_URL = f'https://drive.google.com/uc?id={MODEL_ID}'
MODEL_FILENAME = 'best.pt'

@st.cache_resource
def download_and_load_model():
    """
    Downloads the YOLO model from Google Drive if not present, then loads it.
    Uses st.cache_resource to ensure this only happens once.
    """
    if not os.path.exists(MODEL_FILENAME):
        with st.spinner("Downloading Model from Drive... (This may take a moment)"):
            try:
                gdown.download(MODEL_URL, MODEL_FILENAME, quiet=False)
            except Exception as e:
                st.error(f"Failed to download model: {e}")
                return None
    
    try:
        model = YOLO(MODEL_FILENAME)
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return None

# =============================================================================
# 3. ANALYSIS LOGIC
# =============================================================================

def process_image(image_file, model, px_per_mm, thickness_mm):
    """
    Executes the full image analysis pipeline (Tang et al., 2012 logic).
    """
    # Convert uploaded file to numpy array
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # ---------------------------------------------------------
    # A. Pre-processing
    # ---------------------------------------------------------
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_enhanced = clahe.apply(gray)
    img_input = cv2.cvtColor(gray_enhanced, cv2.COLOR_GRAY2BGR)

    # ---------------------------------------------------------
    # B. Crack Detection (YOLO + Thresholding)
    # ---------------------------------------------------------
    # 1. YOLO Prediction
    results = model.predict(img_input, conf=0.05, save=False, verbose=False)
    
    if results[0].masks is None:
        structure_map = np.zeros(gray.shape, dtype=np.uint8)
    else:
        masks = results[0].masks.data.cpu().numpy()
        structure_map = np.zeros(gray.shape, dtype=np.uint8)
        for m in masks:
            m_resized = cv2.resize(m, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
            structure_map = np.maximum(structure_map, m_resized)

    # 2. Adaptive Thresholding
    # Smoothing block size optimized for clay textures
    connectivity_map = cv2.adaptiveThreshold(
        gray_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 85, 15
    )
    connectivity_clean = remove_small_objects(connectivity_map.astype(bool), min_size=250).astype(np.uint8)

    # ---------------------------------------------------------
    # C. Fusion & Cleaning
    # ---------------------------------------------------------
    combined_map = cv2.bitwise_or(structure_map.astype(np.uint8), connectivity_clean)
    
    kernel_bridge = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed_map = cv2.morphologyEx(combined_map, cv2.MORPH_CLOSE, kernel_bridge, iterations=2)
    clean_map = remove_small_objects(closed_map.astype(bool), min_size=200).astype(np.uint8)
    clean_map = remove_small_holes(clean_map.astype(bool), area_threshold=200).astype(np.uint8)

    # ---------------------------------------------------------
    # D. Skeletonization
    # ---------------------------------------------------------
    skeleton_base = skeletonize(clean_map)
    # Dilate to ensure connectivity before final skeletonization
    kernel_thick = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    final_binary_map = cv2.dilate(skeleton_base.astype(np.uint8), kernel_thick, iterations=1)
    final_skeleton = skeletonize(final_binary_map)

    # ---------------------------------------------------------
    # E. Metric Calculations
    # ---------------------------------------------------------
    h, w = final_binary_map.shape
    total_area_cm2 = (h * w) / (px_per_mm ** 2) / 100

    # 1. Clod Analysis (N_c, A_av)
    clod_mask = 1 - final_binary_map
    clod_mask = remove_small_objects(clod_mask.astype(bool), min_size=20).astype(np.uint8)
    num_clods, _ = cv2.connectedComponents(clod_mask)
    num_clods -= 1  # remove background
    avg_clod_area = (np.sum(clod_mask) / num_clods) / (px_per_mm ** 2) / 100 if num_clods > 0 else 0

    # 2. Surface Crack Ratio (R_sc)
    crack_pixels = np.sum(final_binary_map)
    surface_crack_ratio = (crack_pixels / (h * w)) * 100

    # 3. Node Analysis (N_n)
    skel_int = final_skeleton.astype(int)
    conv = np.array([[1,1,1],[1,1,1],[1,1,1]])
    neighbor_count = ndimage.convolve(skel_int, conv, mode='constant', cval=0)
    raw_nodes = (skel_int == 1) & (neighbor_count > 3)
    num_node_clusters, labels, stats, centroids = cv2.connectedComponentsWithStats(raw_nodes.astype(np.uint8))
    real_node_count = num_node_clusters - 1
    node_density = real_node_count / total_area_cm2
    node_coords = centroids[1:]

    # 4. Segment Analysis (N_seg, L_av, D_c)
    skel_segments = skel_int.copy()
    skel_segments[raw_nodes] = 0 
    valid_segments = remove_small_objects(skel_segments.astype(bool), min_size=8)
    num_segments, _ = cv2.connectedComponents(valid_segments.astype(np.uint8))
    num_segments -= 1
    
    segment_density = num_segments / total_area_cm2
    total_len_cm = np.sum(final_skeleton) / px_per_mm / 10
    avg_crack_length = total_len_cm / num_segments if num_segments > 0 else 0
    crack_density = total_len_cm / total_area_cm2

    # 5. Width & Volume (W_av, Volume)
    dist_map = cv2.distanceTransform(final_binary_map, cv2.DIST_L2, 5)
    width_samples = dist_map[final_skeleton] * 2
    avg_width = (np.mean(width_samples) / px_per_mm) / 10 if len(width_samples) > 0 else 0
    
    # Volume = Crack Area * Thickness
    est_volume_cm3 = (crack_pixels / (px_per_mm ** 2) / 100) * (thickness_mm / 10)

    metrics = {
        "R_sc": surface_crack_ratio,
        "N_c": num_clods,
        "A_av": avg_clod_area,
        "N_n": node_density,
        "N_seg": segment_density,
        "L_av": avg_crack_length,
        "D_c": crack_density,
        "W_av": avg_width,
        "Volume": est_volume_cm3
    }
    
    # Prepare red overlay for visualization
    red_mask = np.zeros_like(img_rgb)
    red_mask[:, :, 0] = 255
    overlay = cv2.addWeighted(img_rgb, 1.0, 
                             cv2.bitwise_and(red_mask, red_mask, mask=final_binary_map), 0.6, 0)

    images = {
        "Original": img_rgb,
        "Binary Map": final_binary_map,
        "Skeleton": final_skeleton,
        "Overlay": overlay,
        "Nodes": node_coords
    }
    
    return metrics, images

# =============================================================================
# 4. MAIN APPLICATION UI
# =============================================================================

def main():
    st.title("🏗️ Geotechnical Desiccation Crack Analysis")
    st.markdown("""
    **Developed by:** Amit Kumar  
    **Context:** Automated quantification of desiccation crack patterns in clayey soils using image processing techniques 
    as described by *Tang et al. (2012)*.
    """)
    st.divider()

    # --- SIDEBAR: INPUTS ---
    with st.sidebar:
        st.header("1. Configuration")
        
        # Load Model Automatically
        model = download_and_load_model()
        if model:
            st.success(f"✅ Model Loaded: {MODEL_FILENAME}")
        else:
            st.error("❌ Model Failed to Load")
            st.stop()

        # Image Uploader
        image_file = st.file_uploader("Upload Soil Image", type=['jpg', 'jpeg', 'png'])
        
        st.header("2. Calibration")
        px_per_mm = st.number_input("Pixels per mm", min_value=1.0, value=4.4333, format="%.4f",
                                   help="Calibration factor to convert pixels to metric units.")
        thickness_mm = st.number_input("Layer Thickness (mm)", min_value=1.0, value=8.0, format="%.1f",
                                      help="Thickness of the soil layer for volume estimation.")
        
        run_btn = st.button("🚀 Run Analysis")

    # --- MAIN EXECUTION ---
    if run_btn:
        if not image_file:
            st.error("Please upload an Image file to proceed.")
        else:
            if model:
                with st.spinner("Processing Crack Network..."):
                    metrics, images = process_image(image_file, model, px_per_mm, thickness_mm)
                
                # --- RESULTS SECTION ---
                st.success("Analysis Complete")
                
                # TAB 1: VISUALIZATION
                tab1, tab2, tab3 = st.tabs(["📊 2x2 Visual Grid", "📋 Geometric Metrics", "📑 Definitions"])
                
                with tab1:
                    # Creating a 2x2 Matplotlib Figure
                    # figsize=(6, 6) ensures the image size is significantly reduced (compact)
                    fig, axes = plt.subplots(2, 2, figsize=(6, 6))
                    
                    # 1. Original Image
                    axes[0, 0].imshow(images["Original"])
                    axes[0, 0].set_title("1. Original Image", fontsize=8)
                    axes[0, 0].axis('off')
                    
                    # 2. Binary Map (Black & White)
                    axes[0, 1].imshow(images["Binary Map"], cmap='gray')
                    axes[0, 1].set_title("2. Binary Crack Map", fontsize=8)
                    axes[0, 1].axis('off')
                    
                    # 3. Skeleton & Nodes
                    axes[1, 0].imshow(images["Skeleton"], cmap='gray_r')
                    node_coords = images["Nodes"]
                    if len(node_coords) > 0:
                        axes[1, 0].scatter(node_coords[:, 0], node_coords[:, 1], c='red', s=5)
                    axes[1, 0].set_title("3. Skeleton & Nodes", fontsize=8)
                    axes[1, 0].axis('off')
                    
                    # 4. Overlay (Segmentation)
                    axes[1, 1].imshow(images["Overlay"])
                    axes[1, 1].set_title(f"4. Overlay (R_sc={metrics['R_sc']:.1f}%)", fontsize=8)
                    axes[1, 1].axis('off')

                    plt.tight_layout()
                    
                    # Center the figure in Streamlit
                    col_spacer1, col_fig, col_spacer2 = st.columns([1, 2, 1])
                    with col_fig:
                        st.pyplot(fig)

                # TAB 2: METRICS
                with tab2:
                    st.markdown("#### Complete Geometric Parameters")
                    data = {
                        "Parameter": [
                            "Surface crack ratio ($R_{sc}$)", 
                            "Number of clods ($N_c$)", 
                            "Average area of clods ($A_{av}$)", 
                            "Number of nodes per unit area ($N_n$)", 
                            "Crack segments per unit area ($N_{seg}$)", 
                            "Average length of cracks ($L_{av}$)", 
                            "Crack density ($D_c$)", 
                            "Average width of cracks ($W_{av}$)",
                            "Estimated Crack Volume ($V_{cr}$)"
                        ],
                        "Value": [
                            f"{metrics['R_sc']:.2f} %",
                            f"{metrics['N_c']}",
                            f"{metrics['A_av']:.2f} cm²",
                            f"{metrics['N_n']:.2f} cm⁻²",
                            f"{metrics['N_seg']:.2f} cm⁻²",
                            f"{metrics['L_av']:.2f} cm",
                            f"{metrics['D_c']:.2f} cm⁻¹",
                            f"{metrics['W_av']:.4f} cm",
                            f"{metrics['Volume']:.2f} cm³"
                        ]
                    }
                    st.table(data)

                # TAB 3: DEFINITIONS
                with tab3:
                    st.markdown("### 📚 Terminology & Definitions")
                    st.info("The following definitions are based on **Tang et al. (2012)**.")
                    
                    st.markdown("""
                    * **Surface Crack Ratio ($R_{sc}$):** Defined as the ratio of the crack area to the total surface area of the soil specimen. It is an indicator of the extent of surficial cracking.
                    * **Number of Clods ($N_c$):** The clod is defined as the independent closed area that is split by cracks (the closed soil area between cracks).
                    * **Average Area of Clods ($A_{av}$):** The mean surface area of the identified soil clods.
                    * **Number of Nodes ($N_n$):** The number of intersection nodes (where crack segments meet) or end nodes (dead ends) per unit area.
                    * **Number of Crack Segments ($N_{seg}$):** The count of distinct crack segments defining the outline of the soil crack pattern per unit area.
                    * **Average Length of Cracks ($L_{av}$):** The average trace length of the medial axis of crack segments.
                    * **Crack Density ($D_c$):** Calculated as the total crack length per unit area.
                    * **Average Width of Cracks ($W_{av}$):** Determined by calculating the shortest distance from a randomly chosen point on one boundary to the opposite boundary of the crack segment.
                    * **Estimated Crack Volume ($V_{cr}$):** A derived volumetric estimation calculated as the Crack Area multiplied by the specimen thickness.
                    """)

if __name__ == "__main__":
    main()
