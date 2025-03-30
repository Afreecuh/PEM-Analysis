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
from skimage.filters import threshold_multiotsu
from skimage.measure import regionprops, label
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
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

# **Display image and allow user to annotate scale**
def plot_image_with_annotations():
    """Display image and allow user to mark scale range"""
    image = st.session_state.image
    fig = px.imshow(np.array(image))

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


from skimage.measure import regionprops, label

def otsu_segmentation():
    inject_ga()
    st.title("Multi-Otsu Thresholding & SEM Analysis")

    num_classes = 5  

    if st.session_state.image is None or st.session_state.pixel_to_um is None:
        st.error("⚠️ Please upload an image and set the scale first!")
        return

    image_np = np.array(st.session_state.image.convert("L"))
    image_eq = cv2.equalizeHist(image_np)
    image_blur = cv2.GaussianBlur(image_eq, (5, 5), 0)

    # Segmentation
    thresholds = threshold_multiotsu(image_blur, classes=num_classes)
    segmented_image = np.digitize(image_blur, bins=thresholds)

    # Show raw segmented preview
    st.image(segmented_image * int(255 / num_classes), caption="Segmented Classes Preview", clamp=True)

    # Always rebuild the class masks
    st.session_state.class_masks = [(segmented_image == i).astype(np.uint8) * 255 for i in range(num_classes)]

    if "selected_layer_index" not in st.session_state:
        st.session_state.selected_layer_index = 0

    labeled_masks = [label(mask) for mask in st.session_state.class_masks]
    class_properties = [regionprops(labeled_mask) for labeled_mask in labeled_masks]

    num_regions = [len(props) for props in class_properties]
    avg_area_per_region = [
        np.mean([prop.area for prop in props]) if len(props) > 0 else 0 
        for props in class_properties
    ]

    pixel_areas = [(segmented_image == i).sum() for i in range(num_classes)]
    total_area = sum(pixel_areas)
    real_physical_sizes = [area * (st.session_state.pixel_to_um ** 2) for area in pixel_areas]
    area_percentages = [(size / total_area) * 100 for size in real_physical_sizes]

    layer_labels = [
        "Porosity (Holes, Cracks)", 
        "Pollutants, Sediments", 
        "Matrix (Base Material)", 
        "Metallic Particles", 
        "High-Reflectivity Contaminants"
    ]

    df_analysis = pd.DataFrame({
        "Layer": layer_labels,
        "Pixel Area": pixel_areas,
        "Physical Area (µm²)": real_physical_sizes,
        "Area Percentage (%)": area_percentages,
        "Number of Regions": num_regions,
        "Average Area per Region (px)": avg_area_per_region
    })

    st.dataframe(df_analysis)

    fig_bar = px.bar(df_analysis, x="Layer", y="Physical Area (µm²)", title="Physical Area of Each Layer")
    fig_bar.update_traces(customdata=layer_labels, hoverinfo="x+y")
    st.plotly_chart(fig_bar, use_container_width=True)

    st.write("### Layer Visualization")
    selected_layer = st.selectbox(
        "Select Layer to Visualize", 
        layer_labels, 
        index=st.session_state.get("selected_layer_index", 0),
        key="layer_selection"
    )
    if selected_layer:
        st.session_state.selected_layer_index = layer_labels.index(selected_layer)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(st.session_state.class_masks[st.session_state.selected_layer_index], cmap="gray")
    ax.set_title(f"{selected_layer} (Layer {st.session_state.selected_layer_index})")
    ax.axis("off")
    st.pyplot(fig)

    fig_pie = px.pie(df_analysis, names="Layer", values="Area Percentage (%)", title="Area Distribution Across Layers")
    st.plotly_chart(fig_pie, use_container_width=True)

    porosity_ratio = (pixel_areas[0] / total_area) * 100 if total_area > 0 else 0
    catalyst_areas = sum(pixel_areas[2:4]) if len(pixel_areas) > 3 else 0
    catalyst_percentage = (catalyst_areas / total_area) * 100 if total_area > 0 else 0
    agglomeration_ratio = (pixel_areas[3] / catalyst_areas) * 100 if catalyst_areas > 0 else 0
    oxidation_ratio = (pixel_areas[4] / total_area) * 100 if len(pixel_areas) > 4 else 0

    st.write(f"📌 Porosity Ratio: {porosity_ratio:.2f}%")
    st.write(f"📌 Catalyst Coverage: {catalyst_percentage:.2f}%")
    st.write(f"📌 Agglomeration Ratio: {agglomeration_ratio:.2f}%")
    st.write(f"📌 Oxidation/Impurity Coverage: {oxidation_ratio:.2f}%")

    st.session_state.analysis_df = df_analysis
    st.session_state.extra_metrics = {
        "Porosity Ratio": porosity_ratio,
        "Catalyst Coverage": catalyst_percentage,
        "Agglomeration Ratio": agglomeration_ratio,
        "Oxidation/Impurity Coverage": oxidation_ratio,
    }


# In[ ]:


import cv2
import numpy as np
import streamlit as st
import plotly.express as px
from skimage.filters import threshold_multiotsu
from PIL import Image

# **Set number of Multi-Otsu classes to 4**
NUM_CLASSES = 4  

# **Calculate shape features**
def calculate_shape_features(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    x, y, w, h = cv2.boundingRect(contour)
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
    aspect_ratio = w / h if h > 0 else 0
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    return circularity, aspect_ratio, solidity

# **Classify particle shape**
def classify_shape(circularity, aspect_ratio, solidity):
    if circularity > 0.8 and 0.9 < aspect_ratio < 1.1:
        return "Circle"
    elif aspect_ratio > 1.5:
        return "Ellipse"
    elif 0.95 <= aspect_ratio <= 1.05 and solidity > 0.95:
        return "Square"
    elif solidity < 0.9:
        return "Irregular"
    else:
        return "Polygon"

# **Analyze particles**
def analyze_particles(image):
    img_gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    img_eq = cv2.equalizeHist(img_gray)
    img_blur = cv2.GaussianBlur(img_eq, (5, 5), 0)

    thresholds = threshold_multiotsu(img_blur, classes=NUM_CLASSES)
    segmented = np.digitize(img_blur, bins=thresholds)

    binary = (segmented == 2).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    circularities = []
    shape_labels = []

    img_with_contours = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 50:  # **Filter small particles**
            circularity, aspect_ratio, solidity = calculate_shape_features(contour)
            shape = classify_shape(circularity, aspect_ratio, solidity)
            circularities.append(circularity)
            shape_labels.append(shape)
            cv2.drawContours(img_with_contours, [contour], -1, (0, 255, 255), 2)

    return circularities, shape_labels, binary, img_with_contours

# **Streamlit interface**
def analyze_particles_page():
    st.title("🔬 SEM Particle Shape Analysis")

    if st.session_state.image is None:
        st.error("⚠️ Please upload an image and set the scale first!")
        return

    image = st.session_state.image
    circularities, shape_labels, binary_image, img_with_contours = analyze_particles(image)

    shape_counts = {shape: shape_labels.count(shape) for shape in set(shape_labels)}
    st.session_state.shape_analysis = shape_counts
    st.session_state.circularity_data = circularities

    st.subheader("📊 Circularity Distribution")
    if circularities:
        fig_circularity = px.histogram(
            x=circularities, nbins=20, range_x=[0, 1], 
            labels={'x': 'Circularity Score', 'y': 'Frequency'},
            title="Circularity Distribution",
            opacity=0.7
        )
        fig_circularity.update_traces(marker_color='blue')
        st.plotly_chart(fig_circularity, use_container_width=True)

        with st.expander("🔍 Show Processed Binary Image"):
            st.image(binary_image, caption="Binary Segmentation (Used for Circularity Calculation)", use_column_width=True, clamp=True)

    else:
        st.warning("⚠️ No particles detected")

    st.subheader("📊 Shape Analysis")
    if shape_labels:
        fig_shape = px.histogram(
            x=shape_labels, 
            labels={'x': 'Shape', 'y': 'Count'},
            title="Shape Analysis",
            opacity=0.7
        )
        fig_shape.update_traces(marker_color='green')
        st.plotly_chart(fig_shape, use_container_width=True)

        with st.expander("🔍 Show Shape Contour Image"):
            st.image(img_with_contours, caption="Segmented Image with Contours", use_column_width=True)
    else:
        st.warning("⚠️ No particles detected")


# User Guide

# In[ ]:


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
        ### **Page 2: Multi-Otsu Segmentation**
        - The system performs **Multi-Otsu thresholding** to segment different material layers.
        - You will see **area analysis** results including:
          - Physical area (µm²)
          - Percentage of each layer
          - Porosity & Catalyst coverage
        - Click **Next** to explore 3D visualization.
        """,
        3: """
        ### **Page 3: 3D Intensity Viewer**
        - Visualize grayscale intensities as a 3D structure.
        - Each pixel's **brightness determines its depth and color**.
        - Adjust **Gaussian smoothing (σ)** to reduce noise.
        - Explore internal topography layer by layer.
        - Click **Next** to analyze particle shape.
        """,
        4: """
        ### **Page 4: Shape & Circularity Analysis**
        - Analyze **particle shapes** using contour detection.
        - View the **circularity distribution** of particles.
        - Identify shape categories like:
          - Circle / Ellipse / Square / Irregular / Polygon
        - Preview segmentation masks and contour overlays.
        - Click **Next** to generate a final report.
        """,
        5: """
        ### **Page 5: Download Report**
        - Click **"Generate PDF Report"** to create a detailed analysis report.
        - The report includes:
          - Scale information
          - Multi-Otsu segmentation analysis
          - 3D intensity visualization
          - Particle shape and circularity analysis
        - Click **"Download PDF"** to save it to your device.
        """
    }

    st.sidebar.title("📖 **User Guide**")
    st.sidebar.markdown(guide_content.get(st.session_state.page, "No guide available for this page."))


# In[ ]:


import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from datetime import datetime
import openai
import streamlit as st

def generate_pdf():
    # ✅ 安全地從 secrets 讀取 OpenAI API Key
    openai.api_key = st.secrets["openai"]["api_key"]

    def query_gpt_for_insights(layer_data, extra_metrics):
        prompt = f"""
        Based on the following SEM analysis data:

        {layer_data}

        Additional metrics:
        {extra_metrics}

        Please generate:
        1. A one-sentence comment for each layer describing its possible implication.
        2. An overall short expert commentary on this catalyst's performance and structure based on the SEM results.
        """

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"[Error calling ChatGPT: {e}]"

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Title
    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawCentredString(width / 2, 770, "SEM Image Analysis Report")
    pdf.setFont("Helvetica", 12)
    pdf.drawCentredString(width / 2, 750, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    pdf.line(50, 740, 550, 740)

    # SEM Image
    if st.session_state.image:
        img_buffer = io.BytesIO()
        st.session_state.image.save(img_buffer, format="PNG")
        img_reader = ImageReader(img_buffer)
        pdf.drawImage(img_reader, 100, 520, width=400, height=200)

    y_offset = 500
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(50, y_offset, "Scale Information")
    pdf.setFont("Helvetica", 12)
    pixel_to_um = st.session_state.get('pixel_to_um', None)
    y_offset -= 20
    pdf.drawString(50, y_offset, f"Pixel to µm Ratio: {pixel_to_um:.6f} µm/px" if pixel_to_um else "⚠️ No scale information available.")
    y_offset -= 10
    pdf.line(50, y_offset, 550, y_offset)
    y_offset -= 20

    # Multi-Otsu Segmentation
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(50, y_offset, "Multi-Otsu Segmentation Analysis")
    pdf.setFont("Helvetica", 12)
    y_offset -= 20

    segmentation_data = st.session_state.get("analysis_df", None)
    extra_metrics = st.session_state.get("extra_metrics", {})
    layer_lines = []
    extra_lines = []

    if segmentation_data is not None:
        for _, row in segmentation_data.iterrows():
            line = f"{row['Layer']}: {row['Physical Area (µm²)']:.2f} µm² ({row['Area Percentage (%)']:.2f}%)"
            pdf.drawString(50, y_offset, line)
            layer_lines.append(line)
            y_offset -= 15
    else:
        pdf.drawString(50, y_offset, "⚠️ No segmentation data available.")
        y_offset -= 15

    y_offset -= 5
    pdf.line(50, y_offset, 550, y_offset)
    y_offset -= 30

    # Additional Metrics
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(50, y_offset, "Additional Metrics")
    pdf.setFont("Helvetica", 12)
    y_offset -= 20

    for metric, value in extra_metrics.items():
        if value is not None:
            line = f"{metric}: {value:.2f}%"
            pdf.drawString(50, y_offset, line)
            extra_lines.append(line)
            y_offset -= 15

    y_offset -= 5
    pdf.line(50, y_offset, 550, y_offset)
    y_offset -= 30

    # Shape Analysis
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(50, y_offset, "Particle Shape Analysis")
    pdf.setFont("Helvetica", 12)
    shape_analysis = st.session_state.get("shape_analysis", {})
    y_offset -= 20

    if shape_analysis:
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(50, y_offset, "Shape Type")
        pdf.drawString(250, y_offset, "Count")
        y_offset -= 15
        pdf.line(50, y_offset, 550, y_offset)
        y_offset -= 20

        pdf.setFont("Helvetica", 12)
        for shape, count in shape_analysis.items():
            pdf.drawString(50, y_offset, shape)
            pdf.drawString(250, y_offset, str(count))
            y_offset -= 15
    else:
        pdf.drawString(50, y_offset, "⚠️ No shape analysis data available.")
        y_offset -= 15

    y_offset -= 5
    pdf.line(50, y_offset, 550, y_offset)
    y_offset -= 30

    # Circularity
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(50, y_offset, "Circularity Distribution")
    pdf.setFont("Helvetica", 12)
    circularity_data = st.session_state.get("circularity_data", [])
    y_offset -= 20

    if circularity_data:
        avg_circularity = sum(circularity_data) / len(circularity_data)
        pdf.drawString(50, y_offset, f"Average Circularity: {avg_circularity:.2f}")
        y_offset -= 15
    else:
        pdf.drawString(50, y_offset, "⚠️ No circularity data available.")
        y_offset -= 15

    y_offset -= 5
    pdf.line(50, y_offset, 550, y_offset)
    y_offset -= 30

    # === AI Commentary Block ===
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(50, y_offset, "AI Commentary")
    y_offset -= 20
    pdf.setFont("Helvetica", 11)

    ai_response = query_gpt_for_insights("\n".join(layer_lines), "\n".join(extra_lines))

    for line in ai_response.split("\n"):
        if y_offset < 50:
            pdf.showPage()
            y_offset = 750
            pdf.setFont("Helvetica", 11)
        pdf.drawString(50, y_offset, line.strip())
        y_offset -= 15

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

    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], key="image_upload")

    if uploaded_file:
        st.session_state.image = Image.open(uploaded_file)
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

    show_user_guide()

    if st.session_state.page == 1:
        upload_and_mark_scale()
    elif st.session_state.page == 2:
        otsu_segmentation()
    elif st.session_state.page == 3:
        view_3d_model()
    elif st.session_state.page == 4:
        analyze_particles_page()
    elif st.session_state.page == 5:
        download_report_page()

    # ✅ Navigation buttons (集中統一管理)
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.session_state.page > 1:
            if st.button("⬅️ Previous", key="prev_button"):
                prev_page()
                st.rerun()  # 避免殘留事件
    with col2:
        if st.session_state.page < 5:
            if st.button("Next ➡️", key="next_button"):
                next_page()
                st.rerun()

# ✅ Run main app
if __name__ == "__main__":
    main()

