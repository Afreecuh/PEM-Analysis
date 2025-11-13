# PEM Fuel Cell Catalyst Analysis Toolkit

A Streamlit application that supports proton-exchange membrane (PEM) imaging workflows by guiding users from scale-bar calibration through porosity and platinum (Pt) particle analysis to automated report generation with AI commentary: https://pem-analysis-biqblqe52kwqhnpcbqfatz.streamlit.app/

## Features
- **Scale bar annotation:** Upload secondary electron (SEI) and backscattered electron (BSE) images, mark two points on the scale bar, and compute the pixel-to-micron conversion while automatically cropping the scale bar from both images.
- **Porosity quantification:** Enhance the SEI image, run multi-Otsu segmentation, classify pores into primary and secondary groups, and visualize distributions alongside summary metrics (porosity %, mean diameters, ratio).
- **Pt particle detection:** Suppress background with FFT, segment Pt particles, combine connected-component labeling (CCL) with normalized cross-correlation (NCC) to capture different particle sizes, and render heatmaps and surface-area-weighted histograms.
- **3D grayscale viewer:** Downsample and smooth images before presenting an interactive 3D point-cloud representation of grayscale intensities.
- **Downloadable reports with AI insights:** Export a PDF that captures inputs, plots, computed metrics, and GPT-generated commentary tailored to observed porosity and particle characteristics.

## Requirements
- Python 3.9 or newer.
- Python dependencies listed in [`requirements.txt`](requirements.txt). Install with:
  ```bash
  pip install -r requirements.txt
  ```
- System libraries for OpenCV rendering (Ubuntu/Debian example):
  ```bash
  sudo apt-get update && sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
  ```

## OpenAI API configuration
The report generator uses the OpenAI Chat Completions API. Provide a key via Streamlit secrets:

1. Create a `.streamlit` directory next to `website.py` if it does not exist.
2. Add a `secrets.toml` file with your credentials:
   ```toml
   [openai]
   api_key = "sk-..."
   ```

## Running the app
From the repository root, launch Streamlit in headless mode (adjust the port if needed):
```bash
streamlit run website.py --server.headless true --server.port 8501
```
Then open the reported local URL in a browser to interact with the UI.

## Typical workflow
1. **Upload & calibrate:** On Page 1, upload paired SEI/BSE images and enter the real-world scale to compute µm/px for downstream measurements.
2. **Analyze porosity:** Page 2 computes pore metrics, draws contours, and plots pore area and diameter histograms for the SEI image.
3. **Assess Pt particles:** Page 3 reports particle counts, size statistics, heatmaps, and surface-area weighting derived from the BSE image.
4. **Visualize in 3D:** Page 4 renders a point cloud of grayscale intensities with adjustable Gaussian smoothing.
5. **Generate the report:** Page 5 bundles the calibrated images, analysis summaries, and AI-written commentary into a downloadable PDF.

## Troubleshooting tips
- Ensure the two scale-bar points are distinct; identical coordinates prevent calibration.
- Verify the OpenAI key has quota—otherwise the PDF will fall back to a placeholder message for the AI commentary.
- NCC-based Pt detection assumes clear template regions; adjust image preprocessing upstream if matches are sparse.

## License
This project inherits the license of its upstream source. Add license details here if you adopt a specific open-source license.
