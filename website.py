#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import numpy as np
import pandas as pd
import cv2
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import matplotlib.pyplot as plt
from skimage.filters import threshold_multiotsu
from skimage.measure import regionprops, label
from skimage.morphology import remove_small_objects
from skimage import exposure
from numpy.fft import fft2, fftshift, ifft2, ifftshift
import streamlit.components.v1 as components
import io
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import openai

openai.api_key = st.secrets["openai"]["api_key"]

# === 2. Preprocess Image: Crop Scale Bar ===
def auto_crop_scale_bar(img, threshold=80):
    h, _ = img.shape
    if np.mean(img[int(h * 0.9):]) < threshold:
        black_row = np.where(np.mean(img, axis=1) < threshold)[0][0]
        return img[:black_row, :]
    return img


# In[1]:


# === 2. 比例尺處理區塊：雙圖上傳 + 比例尺標記與裁剪 ===

# 初始化 session_state 欄位
for key in ["scale_coords", "scale_pixels", "scale_length_um", "pixel_to_um",
            "image_display", "image_bse", "image_sei"]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "scale_coords" else []

# 更新點擊座標（最多兩點）
def update_coords(click_x, click_y):
    if len(st.session_state.scale_coords) < 2:
        st.session_state.scale_coords.append((click_x, click_y))
        st.rerun()
    else:
        st.warning("⚠️ 已標註兩個點，請輸入比例長度。")

# 顯示圖像並標註座標點
def plot_image_with_annotations():
    image = st.session_state.image_display
    fig = px.imshow(np.array(image), color_continuous_scale='gray')
    for coord in st.session_state.scale_coords:
        fig.add_trace(go.Scatter(
            x=[coord[0]],
            y=[coord[1]],
            mode="markers",
            marker=dict(color="red", size=10),
            name="Annotation Point"
        ))
    return fig

# 自動裁剪比例尺（底部黑色區域）
def auto_crop_scale_bar(img, threshold=80):
    h, _ = img.shape
    if np.mean(img[int(h * 0.9):]) < threshold:
        black_row = np.where(np.mean(img, axis=1) < threshold)[0][0]
        return img[:black_row, :]
    return img

# 上傳雙圖 + 標註比例尺介面
def upload_and_mark_scale():
    st.title("📷 Upload BSE & SEI Images + Annotate Scale Bar")

    col1, col2 = st.columns(2)
    with col1:
        sei_file = st.file_uploader("🔬 Upload SEI Image (for Porosity)", type=["png", "jpg", "jpeg", "bmp"], key="sei")
    with col2:
        bse_file = st.file_uploader("⚙️ Upload BSE Image (for Pt Analysis)", type=["png", "jpg", "jpeg", "bmp"], key="bse")

    if sei_file and bse_file:
        sei_img = Image.open(sei_file).convert("RGB")
        bse_img = Image.open(bse_file).convert("RGB")

        st.session_state.image_display = sei_img  # 用 SEI 來標記比例尺
        st.image(sei_img, caption="Click to mark two points on the scale bar", use_column_width=True)

        # 標註點處理
        click = st.plotly_chart(plot_image_with_annotations(), use_container_width=True)
        click_data = st.session_state.get("plotly_click_event")
        if click_data:
            x_click = int(click_data["points"][0]["x"])
            y_click = int(click_data["points"][0]["y"])
            update_coords(x_click, y_click)

        if len(st.session_state.scale_coords) == 2:
            x1, y1 = st.session_state.scale_coords[0]
            x2, y2 = st.session_state.scale_coords[1]
            scale_pixels = abs(x2 - x1)
            st.session_state.scale_pixels = scale_pixels
            st.success(f"✅ Selected scale range: {scale_pixels:.2f} px")

            scale_length_input = st.text_input("Enter actual scale length (µm):", "10")

            if st.button("Calculate µm/px"):
                try:
                    scale_length_um = float(scale_length_input)
                    st.session_state.scale_length_um = scale_length_um
                    pixel_to_um = scale_length_um / scale_pixels
                    st.session_state.pixel_to_um = pixel_to_um
                    st.success(f"📏 Result: {scale_length_um:.2f} µm ({pixel_to_um:.4f} µm/px)")

                    # 裁剪比例尺並更新分析圖像（SEI、BSE）
                    sei_crop = auto_crop_scale_bar(np.array(sei_img.convert("L")))
                    bse_crop = auto_crop_scale_bar(np.array(bse_img.convert("L")))
                    st.session_state.image_sei = Image.fromarray(sei_crop)
                    st.session_state.image_bse = Image.fromarray(bse_crop)

                except ValueError:
                    st.error("⚠️ Invalid input. Please enter a number.")


# In[ ]:


# **Page 2: Porosity Analysis (孔隙分析)**
def analyze_porosity_page():
    inject_ga()
    st.title("🔬 Porosity Analysis (SEI Image)")

    if st.session_state.image_sei is None:
        st.error("⚠️ Please upload SEI image and set scale on Page 1!")
        return

    image_gray = np.array(st.session_state.image_sei)
    image_cropped = exposure.rescale_intensity(image_gray, in_range='image', out_range=(0, 255)).astype(np.uint8)

    # --- Multi-Otsu Segmentation ---
    thresholds = threshold_multiotsu(image_cropped, classes=4)
    segmented = np.digitize(image_cropped, bins=thresholds)

    # --- Calculate Pore Areas ---
    nm_per_pixel = st.session_state.pixel_to_um * 1000
    area_conversion = nm_per_pixel ** 2
    total_area_image_nm2 = image_cropped.shape[0] * image_cropped.shape[1] * area_conversion

    porosity_mask = (segmented == 0).astype(np.uint8)
    labeled = label(porosity_mask)
    props = regionprops(labeled)

    primary_diameters = []
    secondary_diameters = []
    all_pore_areas_nm2 = []

    for region in props:
        if region.area > 10:
            diameter_px = 2 * np.sqrt(region.area / np.pi)
            diameter_nm = diameter_px * nm_per_pixel
            area_nm2 = region.area * area_conversion
            all_pore_areas_nm2.append(area_nm2)

            if diameter_nm < 10:
                primary_diameters.append(diameter_nm)
            else:
                secondary_diameters.append(diameter_nm)

    total_pore_area = np.sum(all_pore_areas_nm2)
    pore_area_pct = (total_pore_area / total_area_image_nm2) * 100

    avg_primary_size = np.mean(primary_diameters) if primary_diameters else 0
    avg_secondary_size = np.mean(secondary_diameters) if secondary_diameters else 0
    rm_ratio = (avg_secondary_size / avg_primary_size) if avg_primary_size > 0 else 0

    # --- Detection Image ---
    st.subheader("🔍 Pore Detection Image")
    overlay = cv2.cvtColor(image_cropped, cv2.COLOR_GRAY2BGR)
    for region in props:
        if region.area > 10:
            coords = region.coords
            mask = np.zeros_like(porosity_mask, dtype=np.uint8)
            mask[tuple(zip(*coords))] = 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1)
    st.image(overlay, caption="Detected Pores (Contours)", use_container_width=True)

    # --- Pore Area Histogram ---
    st.subheader("📊 Pore Area Distribution Histogram")
    hist_area = np.histogram(all_pore_areas_nm2, bins=20)
    area_bins = hist_area[1]
    area_counts = hist_area[0]
    filtered_area = [all_pore_areas_nm2[i] for i in range(len(all_pore_areas_nm2)) if np.histogram(all_pore_areas_nm2, bins=area_bins)[0][np.digitize(all_pore_areas_nm2[i], area_bins)-1] >= 3]

    st.plotly_chart(
        px.histogram(
            x=filtered_area,
            nbins=20,
            labels={"x": "Pore Area (nm²)", "y": "Count"},
            title="Pore Area Distribution (Filtered ≥3)"
        ).update_traces(marker_color="steelblue"),
        use_container_width=True
    )

    # --- Pore Diameter Histogram (Grouped) ---
    st.subheader("📊 Pore Diameter Distribution Histogram")
    fig = go.Figure()
    if primary_diameters:
        primary_hist = np.histogram(primary_diameters, bins=15)
        primary_filtered = [v for v in primary_diameters if np.histogram(primary_diameters, bins=primary_hist[1])[0][np.digitize(v, primary_hist[1]) - 1] >= 3]
        fig.add_trace(go.Histogram(x=primary_filtered, nbinsx=15, name="Primary (≤10 nm)", marker_color="blue"))
    if secondary_diameters:
        secondary_hist = np.histogram(secondary_diameters, bins=15)
        secondary_filtered = [v for v in secondary_diameters if np.histogram(secondary_diameters, bins=secondary_hist[1])[0][np.digitize(v, secondary_hist[1]) - 1] >= 3]
        fig.add_trace(go.Histogram(x=secondary_filtered, nbinsx=15, name="Secondary (≥10 nm)", marker_color="orange"))
    fig.update_layout(
        barmode="group",
        title="Pore Diameter Distribution (Filtered ≥3)",
        xaxis_title="Diameter (nm)",
        yaxis_title="Count"
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Summary ---
    st.subheader("📄 Porosity Analysis Summary")
    summary_text = f"""
    Porosity Analysis Summary:
    - Porosity: {pore_area_pct:.2f} %
    - Average Primary Pore Size: {avg_primary_size:.2f} nm
    - Average Secondary Pore Size: {avg_secondary_size:.2f} nm
    - Rm Size Ratio (Secondary / Primary): {rm_ratio:.2f}
    """
    st.text(summary_text)

    # Save to session
    st.session_state.pore_areas_nm2 = all_pore_areas_nm2
    st.session_state.porosity_ratio = pore_area_pct
    st.session_state.porosity_summary = {
        "Porosity (%)": pore_area_pct,
        "Avg Primary (nm)": avg_primary_size,
        "Avg Secondary (nm)": avg_secondary_size,
        "Rm Ratio": rm_ratio
    }


# In[ ]:


# **Page 3: Pt Particle Analysis (CCL + NCC + Heatmap)**
def analyze_pt_particles_page():
    inject_ga()
    st.title("⚙️ Pt Particle Analysis with CCL + NCC")

    if st.session_state.image_bse is None or st.session_state.pixel_to_um is None:
        st.error("⚠️ Please upload BSE image and set the scale first!")
        return

    image_np = np.array(st.session_state.image_bse.convert("L"))

    # === 1. Multi-Otsu Segmentation ===
    img_cropped = auto_crop_scale_bar(image_np)
    thresholds = threshold_multiotsu(img_cropped, classes=4)
    segmented = np.digitize(img_cropped, bins=thresholds)

    st.subheader("🧪 Multi-Otsu Segmentation Result")
    st.image(segmented * 85, caption="Segmented Classes (4 classes)", use_container_width=True, clamp=True)

    # === 2. FFT Background Removal ===
    f = fft2(img_cropped)
    fshift = fftshift(f)
    crow, ccol = fshift.shape[0] // 2, fshift.shape[1] // 2
    fshift[crow-10:crow+10, ccol-10:ccol+10] *= 0.1
    img_cleaned = np.abs(ifft2(ifftshift(fshift)))
    img_cleaned = exposure.rescale_intensity(img_cleaned, in_range='image', out_range=(0, 255)).astype(np.uint8)

    # === 3. Particle Detection (Layer 3) ===
    layer = 3
    nm_per_pixel = st.session_state.pixel_to_um * 1000
    area_conversion = nm_per_pixel ** 2
    total_area_image_nm2 = img_cropped.shape[0] * img_cropped.shape[1] * area_conversion

    particles_mask = (segmented == layer)
    particles_mask = remove_small_objects(particles_mask, min_size=10)
    labeled_particles = label(particles_mask)
    props = regionprops(labeled_particles)

    ccl_areas_nm2 = []
    ccl_grain_sizes = []
    ccl_surface_areas = []
    ccl_mask = np.zeros_like(particles_mask, dtype=np.uint8)

    for p in props:
        area_nm2 = p.area * area_conversion
        if area_nm2 > 100:
            d = 2 * np.sqrt(area_nm2 / np.pi)
            surface_area = np.pi * d**2
            ccl_areas_nm2.append(area_nm2)
            ccl_grain_sizes.append(d)
            ccl_surface_areas.append(surface_area)
            coords = p.coords
            ccl_mask[tuple(zip(*coords))] = 1

    # === 4. NCC for Small Particles ===
    best_region, best_circ = None, 0
    for p in props:
        circ = (4 * np.pi * p.area / (p.perimeter ** 2)) if p.perimeter > 0 else 0
        if circ > best_circ:
            best_circ = circ
            best_region = p

    ncc_matches = []
    ncc_areas_nm2 = []
    ncc_grain_sizes = []
    ncc_surface_areas = []

    if best_region:
        minr, minc, maxr, maxc = best_region.bbox
        template = img_cleaned[minr:maxr, minc:maxc]
        result = cv2.matchTemplate(img_cleaned, template, cv2.TM_CCOEFF_NORMED)
        threshold_ncc = 0.6
        loc = np.where(result >= threshold_ncc)
        h, w = template.shape

        for pt in zip(*loc[::-1]):
            cy = pt[1] + h // 2
            cx = pt[0] + w // 2
            if not ccl_mask[cy, cx]:
                area_nm2 = np.count_nonzero(template) * area_conversion
                d = 2 * np.sqrt(area_nm2 / np.pi)
                surface_area = np.pi * d**2
                ncc_areas_nm2.append(area_nm2)
                ncc_grain_sizes.append(d)
                ncc_surface_areas.append(surface_area)
                ncc_matches.append(pt)

    # === 5. Detection Image ===
    st.subheader("🔍 Pt Particle Detection Image")
    detection_img = cv2.cvtColor(img_cleaned, cv2.COLOR_GRAY2BGR)
    for p in props:
        if p.area * area_conversion > 100:
            mask = np.zeros_like(labeled_particles, dtype=np.uint8)
            mask[tuple(zip(*p.coords))] = 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(detection_img, contours, -1, (0, 255, 0), 1)

    for pt in ncc_matches:
        cv2.rectangle(detection_img, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 1)

    st.image(cv2.cvtColor(detection_img, cv2.COLOR_BGR2RGB),
             caption="Green: CCL-detected | Red: NCC-matched",
             use_container_width=True)

    # === 6. Grain Size Histogram ===
    st.subheader("📊 Grain Size Histogram (Pt Particle Diameter)")
    all_grain_sizes = ccl_grain_sizes + ncc_grain_sizes
    hist_counts, bin_edges = np.histogram(all_grain_sizes, bins=20)
    filtered_bins = [(bin_edges[i], bin_edges[i+1]) for i, c in enumerate(hist_counts) if c >= 3]
    filtered_grain_sizes = [v for v in all_grain_sizes if any(start <= v < end for start, end in filtered_bins)]
    st.plotly_chart(
        px.histogram(
            x=filtered_grain_sizes,
            nbins=20,
            labels={"x": "Diameter (nm)", "y": "Count"},
            title="Grain Size Distribution of Pt Particles"
        ).update_traces(marker_color="indigo"),
        use_container_width=True
    )

    # === 7. Surface Area Histogram ===
    st.subheader("📊 Surface Area Histogram")
    all_surface_areas = ccl_surface_areas + ncc_surface_areas
    hist_counts_area, bin_edges_area = np.histogram(all_surface_areas, bins=20)
    filtered_bins_area = [(bin_edges_area[i], bin_edges_area[i+1]) for i, c in enumerate(hist_counts_area) if c >= 3]
    filtered_surface_areas = [v for v in all_surface_areas if any(start <= v < end for start, end in filtered_bins_area)]
    st.plotly_chart(
        px.histogram(
            x=filtered_surface_areas,
            nbins=20,
            labels={"x": "Surface Area (nm²)", "y": "Count"},
            title="Spherical Surface Area Distribution"
        ).update_traces(marker_color="darkorange"),
        use_container_width=True
    )

    # === 8. Heatmap ===
    st.subheader("🌡️ Pt Particle Distribution Heatmap")
    grid_rows, grid_cols = 10, 10
    cell_h = img_cleaned.shape[0] // grid_rows
    cell_w = img_cleaned.shape[1] // grid_cols
    heatmap = np.zeros((grid_rows, grid_cols), dtype=int)

    for p in props:
        if p.area * area_conversion > 100:
            cy, cx = p.centroid
            r = min(int(cy // cell_h), grid_rows - 1)
            c = min(int(cx // cell_w), grid_cols - 1)
            heatmap[r, c] += 1
    for pt in ncc_matches:
        cy = pt[1] + h // 2
        cx = pt[0] + w // 2
        r = min(int(cy // cell_h), grid_rows - 1)
        c = min(int(cx // cell_w), grid_cols - 1)
        heatmap[r, c] += 1

    fig_heat = px.imshow(heatmap, color_continuous_scale="inferno")
    fig_heat.update_layout(
        title="Pt Particle Heatmap (10x10 Grid)",
        xaxis_title="Column",
        yaxis_title="Row"
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # === 9. Summary Table ===
    st.subheader("📋 Pt Particle Summary")
    all_areas_nm2 = ccl_areas_nm2 + ncc_areas_nm2
    total_surface_area_nm2 = np.sum(all_surface_areas)
    effective_surface_area_ratio = total_surface_area_nm2 / total_area_image_nm2
    avg_grain_size = np.mean(all_grain_sizes)
    mean_area = np.mean(all_areas_nm2)

    summary = {
        "Total Pt Particles": len(all_areas_nm2),
        "CCL Particles": len(ccl_areas_nm2),
        "NCC Particles": len(ncc_areas_nm2),
        "Average Grain Size (nm)": avg_grain_size,
        "Mean Particle Area (nm²)": mean_area,
        "Total Surface Area (nm²)": total_surface_area_nm2,
        "Effective Pt particle surface (m²) per unit area of CL (m²)": effective_surface_area_ratio
    }

    df_summary = pd.DataFrame(summary, index=["Result"])
    st.dataframe(df_summary)

    # Store results in session
    st.session_state.pt_summary = summary
    st.session_state.pt_props = props
    st.session_state.img_cleaned = img_cleaned
    st.session_state.ccl_mask = ccl_mask
    st.session_state.ncc_matches = ncc_matches
    st.session_state.ccl_areas_nm2 = ccl_areas_nm2
    st.session_state.ncc_areas_nm2 = ncc_areas_nm2
    st.session_state.ccl_grain_sizes = ccl_grain_sizes
    st.session_state.ncc_grain_sizes = ncc_grain_sizes
    st.session_state.ccl_surface_areas = ccl_surface_areas
    st.session_state.ncc_surface_areas = ncc_surface_areas


# User Guide

# In[ ]:


# **Show User Guide**
def show_user_guide():
    """Display User Guide based on current page"""
    guide_content = {
        1: """
        ### **Page 1: Upload BSE & SEI Images + Scale Annotation**
        - Upload **SEI image** (used for Porosity Analysis) and **BSE image** (used for Pt Particle Detection).
        - The system will show the **SEI image**.
        - Click **two points on the SEI image** to mark the scale bar.
        - Enter the actual scale length (in µm) and the system will calculate **µm/px**.
        - This µm/px value is applied to **both images**, and the system automatically crops out the scale bar from each.
        - Click **Next** to proceed.
        """,
        2: """
        ### **Page 2: Porosity Analysis (SEI)**
        - Performs **background enhancement** and **Multi-Otsu segmentation**.
        - Detects pores in the SEI image and calculates:
          - Porosity (%)
          - Primary and secondary pore sizes
          - Size ratio and area distribution
        - Visualizes pore detection and histogram plots.
        - Click **Next** to proceed to Pt particle analysis.
        """,
        3: """
        ### **Page 3: Pt Particle Analysis (BSE)**
        - Performs **Multi-Otsu segmentation** and **FFT background suppression**.
        - Detects Pt particles in BSE image using:
          - **CCL** (Connected Component Labeling) for larger particles
          - **NCC** (Normalized Cross-Correlation) for smaller matched particles
        - Displays:
          - Particle size and surface area distributions
          - Heatmap of particle locations
        - Click **Next** to view 3D visualization of grayscale intensity.
        """,
        4: """
        ### **Page 4: 3D Visualization**
        - Visualizes grayscale intensities as a 3D point cloud structure.
        - Each pixel's **brightness defines its depth and color**.
        - Use the **Gaussian blur slider** to smooth the visualization.
        - Useful for examining surface topography or texture contrast.
        - Click **Next** to generate and download the full report.
        """,
        5: """
        ### **Page 5: Download Report**
        - Click **"Generate PDF Report"** to compile a detailed SEM report.
        - The report includes:
          - Scale bar and µm/px info
          - Porosity summary from SEI
          - Pt particle summary from BSE
          - Distribution histograms and heatmap
          - 3D visualization note
          - GPT-generated scientific commentary
        - After generating, click **"Download PDF"** to save the file.
        """
    }

    st.sidebar.title("📖 **User Guide**")
    st.sidebar.markdown(guide_content.get(st.session_state.page, "No guide available for this page."))


# In[ ]:


def generate_pdf():
    client = openai
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawCentredString(width / 2, 770, "SEM Analysis Report")
    pdf.setFont("Helvetica", 12)
    pdf.drawCentredString(width / 2, 750, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    pdf.line(50, 740, 550, 740)

    y = 720

    # === Scale Info ===
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(50, y, "Scale Information")
    y -= 20
    pdf.setFont("Helvetica", 12)
    px_um = st.session_state.get("pixel_to_um", None)
    if px_um:
        pdf.drawString(50, y, f"Pixel to µm Ratio: {px_um:.6f} µm/px")
    else:
        pdf.drawString(50, y, "⚠️ Scale not set.")
    y -= 10
    pdf.line(50, y, 550, y)
    y -= 30

    # === SEI Image & Porosity Summary ===
    if st.session_state.get("image_sei"):
        pdf.setFont("Helvetica-Bold", 14)
        pdf.drawString(50, y, "SEI Image (Used for Porosity Analysis)")
        y -= 190
        img_buffer = io.BytesIO()
        st.session_state.image_sei.save(img_buffer, format="PNG")
        img_reader = ImageReader(img_buffer)
        pdf.drawImage(img_reader, 100, y, width=400, height=180)
        y -= 30

    poro_ratio = st.session_state.get("porosity_ratio", None)
    pore_areas = st.session_state.get("pore_areas_nm2", [])
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(50, y, "Porosity Summary")
    y -= 20
    pdf.setFont("Helvetica", 12)
    if poro_ratio is not None and pore_areas:
        primary = [a for a in pore_areas if a < (10 ** 2 * np.pi / 4)]
        secondary = [a for a in pore_areas if a >= (10 ** 2 * np.pi / 4)]
        avg_primary = np.mean([2 * np.sqrt(a / np.pi) for a in primary]) if primary else 0
        avg_secondary = np.mean([2 * np.sqrt(a / np.pi) for a in secondary]) if secondary else 0

        pdf.drawString(50, y, f"Porosity: {poro_ratio:.2f} %")
        y -= 15
        pdf.drawString(50, y, f"Primary Pores (<10 nm): {len(primary)}")
        y -= 15
        pdf.drawString(50, y, f"Secondary Pores (≥10 nm): {len(secondary)}")
        y -= 15
        pdf.drawString(50, y, f"Average Primary Diameter: {avg_primary:.2f} nm")
        y -= 15
        pdf.drawString(50, y, f"Average Secondary Diameter: {avg_secondary:.2f} nm")
    else:
        pdf.drawString(50, y, "⚠️ Porosity data not available.")
    y -= 10
    pdf.line(50, y, 550, y)
    y -= 30

    # === BSE Image & Pt Particle Summary ===
    if st.session_state.get("image_bse"):
        if y < 200:  # 換頁避免蓋到
            pdf.showPage()
            y = 750
        pdf.setFont("Helvetica-Bold", 14)
        pdf.drawString(50, y, "BSE Image (Used for Pt Particle Analysis)")
        y -= 190
        img_buffer = io.BytesIO()
        st.session_state.image_bse.save(img_buffer, format="PNG")
        img_reader = ImageReader(img_buffer)
        pdf.drawImage(img_reader, 100, y, width=400, height=180)
        y -= 30

    pt_summary = st.session_state.get("pt_summary", {})
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(50, y, "Pt Particle Summary")
    y -= 20
    pdf.setFont("Helvetica", 12)
    def write_line(label, key):
        nonlocal y
        val = pt_summary.get(key, None)
        if val is not None:
            if isinstance(val, float):
                val = f"{val:.2f}"
            pdf.drawString(50, y, f"{label}: {val}")
            y -= 15
    write_line("Total Pt Particles", "Total Particles")
    write_line("CCL Particles", "CCL Particles")
    write_line("NCC Particles", "NCC Particles")
    write_line("Average Grain Size (nm)", "Average Grain Size (nm)")
    write_line("Surface Area per nm²", "Effective Surface Area per nm²")
    y -= 10
    pdf.line(50, y, 550, y)
    y -= 30

    # === AI Commentary ===
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(50, y, "AI Commentary")
    y -= 20
    pdf.setFont("Helvetica", 11)

    poro_display = f"{poro_ratio:.2f}%" if poro_ratio is not None else "N/A"
    prompt = f"""
    Based on the following SEM analysis results, write a short expert commentary on:
    - What the porosity and Pt particle size may imply
    - Implications for catalyst performance or structural integrity

    Porosity: {poro_display}
    Pt Particles: {pt_summary.get("Total Particles", "N/A")}
    Average Grain Size: {pt_summary.get("Average Grain Size (nm)", "N/A")} nm
    Effective Surface Area: {pt_summary.get("Effective Surface Area per nm²", "N/A")}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
        )
        ai_comment = response.choices[0].message.content.strip()
    except Exception as e:
        ai_comment = "*AI comment unavailable due to quota limit.*"

    for line in ai_comment.split("\n"):
        if y < 50:
            pdf.showPage()
            y = 750
            pdf.setFont("Helvetica", 11)
        pdf.drawString(50, y, line.strip())
        y -= 15

    pdf.save()
    buffer.seek(0)
    return buffer


# In[ ]:


def download_report_page():
    inject_ga()
    st.title("📄 **Download Report**")
    st.write("Click the button below to generate and download your SEM analysis report as a PDF.")

    # Visual separator
    st.markdown("---")

    # Center-aligned download button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("📥 Generate PDF Report", use_container_width=True):
            pdf_buffer = generate_pdf()
            st.success("✅ Report generated successfully! Click below to download.")
            st.download_button(
                label="📄 **Download PDF**",
                data=pdf_buffer,
                file_name="SEM_Analysis_Report.pdf",
                mime="application/pdf",
                use_container_width=True
            )

    st.markdown("---")


# In[ ]:


import numpy as np
import cv2
import streamlit as st
import plotly.graph_objects as go
from PIL import Image
from scipy.ndimage import gaussian_filter

# 3D可視化：使用 intensity 同時決定 z 軸與顏色（反轉灰階 colormap + smoothing）
def view_3d_model():
    st.title("🧊 3D Grayscale Intensity Viewer")

    if st.session_state.image is None:
        st.error("⚠️ Please upload an image first!")
        return

    # ✅ Gaussian smoothing slider
    smoothing_sigma = st.slider("🧹 Smoothing (Gaussian Blur σ)", min_value=0.0, max_value=5.0, value=0.0, step=0.1)

    # 灰階轉換與模糊處理
    image_gray = np.array(st.session_state.image.convert("L"))
    if smoothing_sigma > 0:
        image_gray = gaussian_filter(image_gray, sigma=smoothing_sigma)

    height, width = image_gray.shape
    x_vals, y_vals, z_vals = [], [], []

    for y in range(height):
        for x in range(width):
            intensity = image_gray[y, x]
            if intensity > 0:
                x_vals.append(x)
                y_vals.append(y)
                z_vals.append(intensity)

    fig = go.Figure(data=[go.Scatter3d(
        x=x_vals,
        y=y_vals,
        z=z_vals,
        mode='markers',
        marker=dict(
            size=1,
            color=z_vals,
            colorscale="Greys_r",  # ✅ 反轉灰階：0=黑，255=白
            opacity=0.8,
            colorbar=dict(title="Intensity")
        )
    )])

    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Intensity (0–255)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        title="3D Visualization Based on Grayscale Intensity"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
    🎨 Each point's depth and color are based on grayscale intensity (0–255), smoothed using Gaussian σ = `{smoothing_sigma}`.

    • Black = low intensity  
    • White = high intensity  
    • Adjust smoothing to reduce noise and enhance topography.
    """)

# debug entry point
def debug_process():
    if st.session_state.image is None:
        st.error("⚠️ Please upload an image first!")
        return
    view_3d_model()


# In[3]:


# === 第九區塊：主程式入口與初始化 ===

# **Google Analytics Tracking Code**
def inject_ga():
    """Inject Google Analytics tracking code into the Streamlit app."""
    GA_TRACKING_ID = "G-4QWR3D46SD"
    ga_code = f"""
    <script async src="https://www.googletagmanager.com/gtag/js?id={GA_TRACKING_ID}"></script>
    <script>
        window.dataLayer = window.dataLayer || [];
        function gtag(){{dataLayer.push(arguments);}}
        gtag('js', new Date());
        gtag('config', '{GA_TRACKING_ID}', {{ 'send_page_view': true }});
    </script>
    """
    components.html(ga_code, height=0)

# **Initialize Session State**
init_keys = {
    "page": 1,
    "scale_coords": [],
    "pixel_to_um": None,
    "scale_pixels": None,
    "scale_length_um": None,
    "image": None,
    "image_sei": None,
    "image_bse": None,
}
for k, v in init_keys.items():
    if k not in st.session_state:
        st.session_state[k] = v

def next_page():
    st.session_state.page += 1

def prev_page():
    st.session_state.page -= 1

# **Plot Image with Annotations**
def plot_image_with_annotations():
    if "image" not in st.session_state or st.session_state.image is None:
        st.warning("⚠️ No image available to plot.")
        return go.Figure()

    try:
        image_np = np.array(st.session_state.image)
    except Exception as e:
        st.error(f"⚠️ Failed to convert image: {e}")
        return go.Figure()

    fig = px.imshow(image_np, color_continuous_scale='gray')

    for coord in st.session_state.scale_coords:
        fig.add_trace(go.Scatter(
            x=[coord[0]],
            y=[coord[1]],
            mode="markers",
            marker=dict(color="red", size=10),
            name="Annotation Point"
        ))

    return fig

# **Page 1: Upload SEI + BSE & Annotate Scale**
def upload_and_mark_scale():
    inject_ga()

    # 封面圖中置顯示
    col_left, col_img, col_right = st.columns([1, 6, 1])
    with col_img:
        st.image("cover_image.png", use_container_width=True)

    # 上傳區
    col1, col2 = st.columns(2)
    with col1:
        sei_file = st.file_uploader("🔬 Upload SEI Image (for Porosity)", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"], key="sei")
    with col2:
        bse_file = st.file_uploader("⚙️ Upload BSE Image (for Pt Analysis)", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"], key="bse")

    if sei_file and bse_file:
        sei_img = Image.open(sei_file).convert("RGB")
        bse_img = Image.open(bse_file).convert("RGB")

        st.session_state.image = sei_img
        fig = plot_image_with_annotations()
        st.plotly_chart(fig, use_container_width=True)

        st.write("Manually input two coordinate points (X and Y):")
        col1, col2 = st.columns(2)
        with col1:
            x1 = st.number_input("First point X", min_value=0, step=1, key="x1_input")
            x2 = st.number_input("Second point X", min_value=0, step=1, key="x2_input")
        with col2:
            y1 = st.number_input("First point Y", min_value=0, step=1, key="y1_input")
            y2 = st.number_input("Second point Y", min_value=0, step=1, key="y2_input")

        if st.button("Mark Scale", key="mark_scale_button"):
            if x1 != x2 or y1 != y2:
                st.session_state.scale_coords = [(x1, y1), (x2, y2)]
                scale_pixels = abs(x2 - x1)
                st.session_state.scale_pixels = scale_pixels
                st.success(f"✅ Selected scale range: {scale_pixels:.2f} px")
                st.rerun()
            else:
                st.error("⚠️ The two coordinates cannot be identical. Please re-enter.")

        # 計算 µm/px
        if len(st.session_state.scale_coords) == 2:
            scale_length_input = st.text_input("Enter actual scale length (µm):", "10")
            if st.button("Calculate µm/px"):
                if not scale_length_input.strip():
                    st.error("⚠️ Please input a valid number before calculating.")
                else:
                    try:
                        scale_length_um = float(scale_length_input)
                        scale_pixels = st.session_state.scale_pixels
                        pixel_to_um = scale_length_um / scale_pixels
                        st.session_state.scale_length_um = scale_length_um
                        st.session_state.pixel_to_um = pixel_to_um
                        st.success(f"📏 Result: {scale_length_um:.2f} µm ({pixel_to_um:.4f} µm/px)")

                        sei_crop = auto_crop_scale_bar(np.array(sei_img.convert("L")))
                        bse_crop = auto_crop_scale_bar(np.array(bse_img.convert("L")))
                        st.session_state.image_sei = Image.fromarray(sei_crop)
                        st.session_state.image_bse = Image.fromarray(bse_crop)
                        st.rerun()

                    except ValueError:
                        st.error("⚠️ Invalid input. Please enter a number.")

# **Main Application Entry Point**
def main():
    show_user_guide()

    if st.session_state.page == 1:
        upload_and_mark_scale()
    elif st.session_state.page == 2:
        analyze_porosity_page()
    elif st.session_state.page == 3:
        analyze_pt_particles_page()
    elif st.session_state.page == 4:
        view_3d_model()
    elif st.session_state.page == 5:
        download_report_page()

    col1, col2 = st.columns([1, 5])
    with col1:
        if st.session_state.page > 1:
            if st.button("⬅️ Previous", key="prev_button"):
                prev_page()
                st.rerun()
    with col2:
        if st.session_state.page < 5:
            if st.button("Next ➡️", key="next_button"):
                next_page()
                st.rerun()

# ✅ Run
if __name__ == "__main__":
    main()

