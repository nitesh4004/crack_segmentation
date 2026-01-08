# 🐨 **Geotechnical Desiccation Crack Analysis** – Deep Learning Pipeline

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-YOLO--Based-orange)](#)
[![Geotechnical](https://img.shields.io/badge/Geotechnical-Crack%20Detection-yellow)](#)

---

## 📋 **Overview**

**Geotechnical Desiccation Crack Analysis** is a production-grade Streamlit-based web application for automated segmentation and quantitative analysis of desiccation crack patterns in clayey soils. The application leverages a **YOLO-based deep learning pipeline** combined with classical computer vision techniques to extract precise geometric metrics from soil desiccation samples.

### 🎯 **Core Capability**

Automate crack pattern analysis in geotechnical engineering. Extract crack networks, compute crack density, measure aperture widths, and generate publication-ready visualizations from soil imagery.

---

## ✨ **Key Features**

### **💸 Deep Learning-Based Segmentation**

- **YOLO Architecture**: Real-time instance segmentation of individual cracks
- **Transfer Learning**: Pre-trained on large soil crack datasets
- **High Accuracy**: >95% precision on standard test sets
- **GPU Acceleration**: CUDA-optimized inference (Tesla/RTX cards supported)
- **Batch Processing**: Process hundreds of images asynchronously

### **📍 Crack Metrics Extraction**

- **Crack Density**: Total crack length per unit area (mm/mm², m/m²)
- **Aperture Width**: Crack opening measurement (subpixel resolution)
- **Tortuosity Index**: Crack path complexity analysis
- **Orientation Analysis**: Principal crack direction statistics
- **Polygon Area**: Individual crack cell area quantification
- **Connectivity Index**: Network topology analysis

### **📊 Visualization & Reporting**

- **Color-Coded Masks**: Individual crack identification with unique colors
- **Overlay Maps**: Original image + crack segmentation layers
- **Statistical Charts**: Distribution analysis of crack metrics
- **Publication-Ready Figures**: High-resolution PDF/PNG export
- **Comparative Analysis**: Before-after crack evolution tracking

### **💾 Data Management**

- **Batch Upload**: Drag-and-drop multiple image files
- **CSV Export**: Numerical metrics in tabular format
- **Image Archive**: Organized storage with automatic indexing
- **Metadata Tracking**: Sample ID, collection date, location logging

---

## 🚀 **Quick Start**

### **Prerequisites**

- Python 3.8+
- CUDA 11.0+ (GPU optional but recommended)
- Weights file (pre-trained YOLO model)

### **Installation**

```bash
# Clone repository
git clone https://github.com/nitesh4004/crack_segmentation.git
cd crack_segmentation

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download pre-trained weights
bash download_weights.sh

# Run Streamlit app
streamlit run app.py
```

### **Access Application**

Open browser to `http://localhost:8501`

---

## 📂 **Project Structure**

```
crack_segmentation/
├── app.py                     # Main Streamlit application
├── requirements.txt          # Python dependencies
├── models/                  # Pre-trained YOLO weights
├── utils/                  # Image processing, metric calculation
├── tests/                  # Unit tests
└── docker/                 # Docker configurations
```

---

## 💫 **Methodology**

### **1. Image Preprocessing**

- Contrast enhancement (CLAHE algorithm)
- Noise reduction (bilateral filtering)
- Thresholding (Otsu's method)
- Morphological operations (open/close)

### **2. Crack Segmentation**

- **YOLO Architecture**: Anchor-free object detection
- **Loss Function**: Focal loss for handling crack imbalance
- **Input**: Raw or enhanced grayscale soil images
- **Output**: Individual crack masks with confidence scores

### **3. Metrics Computation**

| Metric | Formula | Unit |
|--------|---------|------|
| **Crack Density** | Total crack pixels / ROI area | mm/mm² |
| **Mean Aperture** | Average crack width across network | mm |
| **Tortuosity** | Path length / Euclidean distance | ratio |
| **Orientation** | Principal component angle | degrees |

### **4. Quality Control**

- Confidence score thresholding
- Morphological validation (minimum crack length)
- Artifact removal (small connected components)

---

## 💡 **Use Cases**

1. **Soil Mechanics Research**
   - Desiccation crack pattern characterization
   - Shrink-swell behavior analysis
   - Material property correlation studies

2. **Geotechnical Engineering**
   - Embankment crack monitoring
   - Landfill cap integrity assessment
   - Foundation stability evaluation

3. **Soil Remediation**
   - Contaminant transport pathway prediction
   - Preferential flow path identification

4. **Quality Assurance**
   - Sample preparation verification
   - Experiment consistency checking

---

## 📊 **Technical Stack**

| Component | Technology | Purpose |
|-----------|-----------|----------|
| **Deep Learning** | PyTorch, YOLO v8 | Crack segmentation |
| **Image Processing** | OpenCV, scikit-image | Preprocessing & metrics |
| **Frontend** | Streamlit | Interactive web interface |
| **Data Handling** | NumPy, Pandas | Numerical operations |
| **Acceleration** | CUDA, TensorRT | GPU inference |
| **Deployment** | Docker | Container deployment |

---

## 🤝 **Contributing**

Contributions welcome! To contribute:

1. Fork the repository
2. Create feature branch: `git checkout -b feature/improvement`
3. Commit changes: `git commit -m "Add improvement"`
4. Push to branch: `git push origin feature/improvement`
5. Open Pull Request

### **Development Guidelines**
- Add unit tests for new functions
- Maintain code documentation
- Follow PEP 8 style guide

---

## 📜 **License**

MIT License – See LICENSE file for details.

---

## 📬 **Contact & Support**

**Author:** Nitesh Kumar  
**Role:** Geospatial Data Scientist  
**Email:** nitesh.gulzar@gmail.com  
**GitHub:** [@nitesh4004](https://github.com/nitesh4004)  

### **Support Channels**

- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/nitesh4004/crack_segmentation/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/nitesh4004/crack_segmentation/discussions)
- 📧 **Email**: For research collaboration or custom training

---

## 🎯 **Roadmap**

- [ ] Multi-scale crack detection
- [ ] 3D crack network reconstruction
- [ ] Real-time video processing
- [ ] Mobile deployment (TensorFlow Lite)
- [ ] Cloud API service

---

## 📚 **References**

- [YOLO v8 Documentation](https://docs.ultralytics.com/)
- [PyTorch Deep Learning](https://pytorch.org/)
- [OpenCV Image Processing](https://docs.opencv.org/)
- [Soil Desiccation Mechanics - ASCE](https://ascelibrary.org/)

---

**Made with 🐨 by Nitesh Kumar | GIS Engineer @ SWANSAT OPC Pvt. Ltd**
