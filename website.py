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


# In[1]:


# **Initialize Session State**
if "scale_coords" not in st.session_state:
    st.session_state.scale_coords = []
if "scale_pixels" not in st.session_state:
    st.session_state.scale_pixels = None
if "scale_length_um" not in st.session_state:
    st.session_state.scale_length_um = None
if "pixel_to_um" not in st.session_state:
    st.session_state.pixel_to_um = None
if "image" not in st.session_state:
    st.session_state.image = None

# **Update annotation points when user clicks**
def update_coords(click_x, click_y):
    """Record coordinates of user clicks, maximum of two points"""
    if len(st.session_state.scale_coords) < 2:
        st.session_state.scale_coords.append((click_x, click_y))
        st.rerun()  # Refresh UI to show red dot
    else:
        st.warning("⚠️ Two points already marked. Please input the actual scale length!")

# **Plot Image with Annotations**
def plot_image_with_annotations():
    """Display image and allow user to mark scale range."""
    image = st.session_state.image
    fig = px.imshow(np.array(image), color_continuous_scale='gray')  # 強制使用灰階顯示圖像

    # **Add annotation points**
    for coord in st.session_state.scale_coords:
        fig.add_trace(go.Scatter(
            x=[coord[0]],
            y=[coord[1]],
            mode="markers",
            marker=dict(color="red", size=10),
            name="Annotation Point"
        ))

    return fig

# **Handle scale annotation and calculate µm/px**
def handle_scale_annotation():
    if len(st.session_state.scale_coords) == 2:
        x1, y1 = st.session_state.scale_coords[0]
        x2, y2 = st.session_state.scale_coords[1]
        scale_pixels = abs(x2 - x1)  # **Only calculate X-direction distance**
        st.session_state.scale_pixels = scale_pixels

        st.success(f"✅ Selected scale range: {scale_pixels:.2f} px")
        
        # **Input actual scale length**
        scale_length_input = st.text_input("Enter actual scale length (µm):", "10")

        if st.button("Calculate µm/px"):
            try:
                scale_length_um = float(scale_length_input)
                st.session_state.scale_length_um = scale_length_um
                pixel_to_um = scale_length_um / scale_pixels
                st.session_state.pixel_to_um = pixel_to_um
                st.success(f"📏 Result: {scale_length_um:.2f} µm ({pixel_to_um:.4f} µm/px)")
            except ValueError:
                st.error("⚠️ Invalid input. Please enter a number.")


# In[ ]:


# === 2. Preprocess Image: Crop Scale Bar ===
def auto_crop_scale_bar(img, threshold=80):
    """
    Crop the image by removing the scale bar region if detected at the bottom.
    """
    h, _ = img.shape
    if np.mean(img[int(h * 0.9):]) < threshold:  # Check bottom region for scale bar
        black_row = np.where(np.mean(img, axis=1) < threshold)[0][0]  # Find first row of black region
        return img[:black_row, :]
    return img

# **Page 2: Porosity Analysis (孔隙分析)**
def analyze_porosity_page():
    inject_ga()
    st.title("🔬 Porosity Analysis")

    # Check if image is available
    if st.session_state.image is None:
        st.error("⚠️ Please upload an image first!")
        return

    # --- Preprocess Image for Porosity Analysis ---
    image_gray = np.array(st.session_state.image.convert("L"))
    image_cropped = auto_crop_scale_bar(image_gray)
    image_cropped = exposure.rescale_intensity(image_cropped, in_range='image', out_range=(0, 255)).astype(np.uint8)

    # --- Multi-Otsu Segmentation ---
    thresholds = threshold_multiotsu(image_cropped, classes=4)
    segmented = np.digitize(image_cropped, bins=thresholds)

    # --- Calculate Pore Areas ---
    nm_per_pixel = st.session_state.pixel_to_um * 1000
    area_conversion = nm_per_pixel ** 2
    total_area_image_nm2 = image_cropped.shape[0] * image_cropped.shape[1] * area_conversion

    # Extract Porosity Layer (Class 0)
    porosity_mask = (segmented == 0).astype(np.uint8)
    labeled = label(porosity_mask)
    props = regionprops(labeled)

    primary_diameters = []
    secondary_diameters = []
    primary_areas = []
    secondary_areas = []
    all_pore_areas_nm2 = []

    for region in props:
        if region.area > 10:
            diameter_px = 2 * np.sqrt(region.area / np.pi)
            diameter_nm = diameter_px * nm_per_pixel
            area_nm2 = region.area * area_conversion
            all_pore_areas_nm2.append(area_nm2)

            if diameter_nm < 10:
                primary_diameters.append(diameter_nm)
                primary_areas.append(area_nm2)
            else:
                secondary_diameters.append(diameter_nm)
                secondary_areas.append(area_nm2)

    # --- Calculate Porosity Ratio ---
    total_pore_area = np.sum(all_pore_areas_nm2)
    pore_area_pct = (total_pore_area / total_area_image_nm2) * 100

    # --- Pore Area and Size Ratio Histogram ---
    st.subheader("📊 Pore Area and Size Ratio Histogram")
    st.plotly_chart(
        px.histogram(
            x=all_pore_areas_nm2,
            nbins=20,
            labels={"x": "Pore Area (nm²)", "y": "Count"},
            title="Pore Area Distribution"
        ).update_traces(marker_color="steelblue"),
        use_container_width=True
    )

    # --- Pore Detection Image ---
    st.subheader("🔍 Pore Detection Image")
    st.image(porosity_mask * 255, caption="Porosity Mask", use_column_width=True)

    # --- Show Summary Information ---
    st.write(f"Total Pore Area: {total_pore_area:.2f} nm²")
    st.write(f"Total Image Area: {total_area_image_nm2:.2f} nm²")
    st.write(f"Porosity Ratio: {pore_area_pct:.2f}%")

    # Store porosity information in session state for next page
    st.session_state.pore_areas_nm2 = all_pore_areas_nm2
    st.session_state.pore_detection_img = porosity_mask * 255
    st.session_state.porosity_ratio = pore_area_pct

    st.success("✅ Porosity analysis completed. Proceed to the next page to analyze Pt particles.")


# In[ ]:


# **Page 3: Pt Particle Analysis (CCL + NCC + Heatmap)**
def analyze_pt_particles_page():
    inject_ga()
    st.title("⚙️ Pt Particle Analysis with CCL + NCC")

    if st.session_state.image is None or st.session_state.pixel_to_um is None:
        st.error("⚠️ Please upload an image and set the scale first!")
        return

    image_np = np.array(st.session_state.image.convert("L"))

    # === 1. FFT Background Removal ===
    f = fft2(image_np)
    fshift = fftshift(f)
    crow, ccol = fshift.shape[0] // 2, fshift.shape[1] // 2
    fshift[crow-10:crow+10, ccol-10:ccol+10] *= 0.1
    img_cleaned = np.abs(ifft2(ifftshift(fshift)))
    img_cleaned = exposure.rescale_intensity(img_cleaned, in_range='image', out_range=(0, 255)).astype(np.uint8)

    st.subheader("📷 FFT Background Removal")
    st.image(img_cleaned, caption="FFT Cleaned Image", use_column_width=True, clamp=True)

    # === 2. Multi-Otsu Segmentation ===
    thresholds = threshold_multiotsu(img_cleaned, classes=4)
    segmented = np.digitize(img_cleaned, bins=thresholds)

    st.subheader("🧪 Multi-Otsu Segmentation Result")
    st.image(segmented * 85, caption="Segmented Classes (4 classes)", use_column_width=True, clamp=True)

    # === 3. Pt Particle Detection ===
    layer = 3
    nm_per_pixel = st.session_state.pixel_to_um * 1000
    area_conversion = nm_per_pixel ** 2
    particles_mask = (segmented == layer)
    particles_mask = remove_small_objects(particles_mask, min_size=10)
    labeled_particles = label(particles_mask)
    props = regionprops(labeled_particles)

    # === 4. CCL for larger particles
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

    # === 5. NCC for smaller particles
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
                template_area_px = np.count_nonzero(template)
                area_nm2 = template_area_px * area_conversion
                d = 2 * np.sqrt(area_nm2 / np.pi)
                surface_area = np.pi * d**2
                ncc_areas_nm2.append(area_nm2)
                ncc_grain_sizes.append(d)
                ncc_surface_areas.append(surface_area)
                ncc_matches.append(pt)

    # === 6. Summary Calculation ===
    all_areas_nm2 = ccl_areas_nm2 + ncc_areas_nm2
    grain_sizes_nm = ccl_grain_sizes + ncc_grain_sizes
    spherical_surface_areas = ccl_surface_areas + ncc_surface_areas

    total_surface_area_nm2 = np.sum(spherical_surface_areas)
    effective_surface_area_per_nm2 = total_surface_area_nm2 / total_area_image_nm2
    num_particles = len(all_areas_nm2)
    avg_grain_size_nm = np.mean(grain_sizes_nm)
    mean_area_nm2 = np.mean(all_areas_nm2)

    # === 7. Display Summary Table ===
    summary = {
        "Total Particles": num_particles,
        "CCL Particles": len(ccl_areas_nm2),
        "NCC Particles": len(ncc_areas_nm2),
        "Average Grain Size (nm)": avg_grain_size_nm,
        "Mean Particle Area (nm²)": mean_area_nm2,
        "Total Surface Area (nm²)": total_surface_area_nm2,
        "Effective Surface Area per nm²": effective_surface_area_per_nm2
    }

    st.subheader("📊 Pt Particle Summary")
    df_summary = pd.DataFrame(summary, index=["Result"])
    st.dataframe(df_summary)

    # Store Pt particle information for next page
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

    st.success("✅ Pt particle analysis completed. Proceed to the next page to view heatmap and distributions.")


# User Guide

# In[ ]:


# **Show User Guide**
def show_user_guide():
    """Display User Guide based on current page"""
    guide_content = {
        1: """
        ### **Page 1: Upload Image & Scale Annotation**
        - Upload an image (PNG, JPG, BMP).
        - Click two points to **mark the scale**.
        - Enter the actual scale length (µm).
        - The system calculates **µm/px** automatically.
        - Click **Next** to proceed.
        """,
        2: """
        ### **Page 2: Porosity Analysis**
        - The system performs **background cleaning (FFT)** and **Multi-Otsu segmentation**.
        - It selects the **porosity layer** to analyze pores and calculate:
          - Total pore area
          - Pore distribution
        - Click **Next** to proceed to Pt particle analysis.
        """,
        3: """
        ### **Page 3: Pt Particle Analysis (CCL + NCC + Heatmap)**
        - The system detects **Pt particles** using:
          - **CCL** for larger particles
          - **NCC** for matching smaller ones
        - Generates **particle size distribution** and **heatmap**.
        - Click **Next** to view 3D visualization of intensity.
        """,
        4: """
        ### **Page 4: 3D Visualization**
        - Visualize grayscale intensities as a 3D structure.
        - Each pixel's **brightness determines its depth and color**.
        - Adjust **Gaussian smoothing (σ)** to reduce noise.
        - Explore internal topography layer by layer.
        - Click **Next** to generate and download the report.
        """,
        5: """
        ### **Page 5: Download Report**
        - Click **"Generate PDF Report"** to create a detailed analysis report.
        - The report includes:
          - Scale information
          - Porosity and Pt particle analysis
          - Heatmap and 3D visualization summary
          - GPT-generated expert commentary
        - Click **"Download PDF"** to save it to your device.
        """
    }

    st.sidebar.title("📖 **User Guide**")
    st.sidebar.markdown(guide_content.get(st.session_state.page, "No guide available for this page."))


# In[ ]:


def generate_pdf():
    client = OpenAI(api_key=st.secrets["openai"]["api_key"])
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawCentredString(width / 2, 770, "Pt Particle Analysis Report")
    pdf.setFont("Helvetica", 12)
    pdf.drawCentredString(width / 2, 750, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    pdf.line(50, 740, 550, 740)

    # === SEM image preview ===
    if st.session_state.image:
        img_buffer = io.BytesIO()
        st.session_state.image.save(img_buffer, format="PNG")
        img_reader = ImageReader(img_buffer)
        pdf.drawImage(img_reader, 100, 520, width=400, height=200)

    y = 500
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(50, y, "Scale Information")
    y -= 20
    pdf.setFont("Helvetica", 12)
    px_um = st.session_state.get("pixel_to_um", None)
    if px_um:
        pdf.drawString(50, y, f"Pixel to µm Ratio: {px_um:.6f} µm/px")
    else:
        pdf.drawString(50, y, "⚠️ No scale information.")
    y -= 10
    pdf.line(50, y, 550, y)
    y -= 30

    # === Summary Data ===
    pt_summary = st.session_state.get("pt_summary", {})
    summary_lines = []
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(50, y, "Pt Particle Summary")
    y -= 20
    pdf.setFont("Helvetica", 12)
    for key, value in pt_summary.items():
        if isinstance(value, (float, int)):
            line = f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}"
            pdf.drawString(50, y, line)
            summary_lines.append(line)
            y -= 15

    y -= 5
    pdf.line(50, y, 550, y)
    y -= 30

    # === AI Commentary ===
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(50, y, "AI Commentary")
    y -= 20
    pdf.setFont("Helvetica", 11)

    # GPT 調用簡評
    prompt = f"""
    Based on the following Pt particle analysis summary, provide:
    1. A brief sentence explaining what each metric may indicate.
    2. A short expert commentary on the particle distribution, surface area, and possible implications for catalytic performance.

    Summary:
    {chr(10).join(summary_lines)}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
        )
        ai_comment = response.choices[0].message.content.strip()
    except Exception as e:
        ai_comment = f"[Error from ChatGPT: {e}]"

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

    # ✅ 修正 bug：確保不會因為初次 rerun 而跳頁
    st.session_state.page = 3

    if st.session_state.image is None:
        st.error("⚠️ Please upload an image first!")
        return

    # ✅ Always visible smoothing slider
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

    # ❌ 移除 point count display
    # st.write(f"Total points to render: {len(x_vals)}")

    fig = go.Figure(data=[go.Scatter3d(
        x=x_vals,
        y=y_vals,
        z=z_vals,
        mode='markers',
        marker=dict(
            size=1,
            color=z_vals,           # 使用強度作為顏色
            colorscale="Greys_r",   # ✅ 反轉灰階：0=黑，255=白
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
if "page" not in st.session_state:
    st.session_state.page = 1
if "scale_coords" not in st.session_state:
    st.session_state.scale_coords = []
if "pixel_to_um" not in st.session_state:
    st.session_state.pixel_to_um = None
if "scale_pixels" not in st.session_state:
    st.session_state.scale_pixels = None
if "scale_length_um" not in st.session_state:
    st.session_state.scale_length_um = None
if "image" not in st.session_state:
    st.session_state.image = None

def next_page():
    st.session_state.page += 1

def prev_page():
    st.session_state.page -= 1

# **Page 1: Upload Image & Annotate Scale**
def upload_and_mark_scale():
    inject_ga()

    # ✅ 圖片置中顯示，減少左右空白
    col_left, col_img, col_right = st.columns([1, 6, 1])
    with col_img:
        st.image("cover_image.png", use_column_width=True)

    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "tif", "tiff"], key="image_upload")

    if uploaded_file:
        # 讀取圖片並確保其是灰階模式
        image = Image.open(uploaded_file).convert("L")  # 強制轉換為灰階模式
        st.session_state.image = image  # 存儲圖片至 session_state
        st.success("✅ Image uploaded successfully! Please mark the scale.")

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
                st.success(f"✅ Selected scale range: {abs(x2 - x1):.2f} px")
                st.rerun()
            else:
                st.error("⚠️ The two coordinates cannot be identical. Please re-enter.")

        fig = plot_image_with_annotations()
        st.plotly_chart(fig, use_container_width=True)

        handle_scale_annotation()

# **Main Application Entry Point**
def main():
    if "page" not in st.session_state:
        st.session_state.page = 1

    show_user_guide()  # Update user guide to reflect new pages

    # Page routing based on current page number
    if st.session_state.page == 1:
        upload_and_mark_scale()  # Handle uploading and marking scale
    elif st.session_state.page == 2:
        analyze_porosity_page()  # Update this to Porosity analysis page
    elif st.session_state.page == 3:
        analyze_pt_particles_page()  # Pt particle analysis (CCL + NCC + Heatmap)
    elif st.session_state.page == 4:
        view_3d_model()  # 3D grayscale intensity visualization
    elif st.session_state.page == 5:
        download_report_page()  # Generate and download PDF report

    # Navigation buttons for page control
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.session_state.page > 1:
            if st.button("⬅️ Previous", key="prev_button"):
                prev_page()
                st.rerun()  # Re-run to update page state
    with col2:
        if st.session_state.page < 5:
            if st.button("Next ➡️", key="next_button"):
                next_page()
                st.rerun()  # Re-run to update page state

# ✅ Run main app
if __name__ == "__main__":
    main()

