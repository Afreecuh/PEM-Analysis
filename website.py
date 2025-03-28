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
        - Click **Next** to proceed to shape analysis.
        """,
        3: """
        ### **Page 3: Shape & Circularity Analysis**
        - The system analyzes the **shape and circularity** of detected particles.
        - **Circularity Distribution** chart shows the spread of circularity scores.
        - **Shape Analysis** chart categorizes particles into:
          - Circle
          - Ellipse
          - Square
          - Irregular
          - Polygon
        - Hover over the charts to see **detailed values**.
        """,
        4: """
        ### **Page 4: Download Report**
        - Click **"Generate PDF Report"** to create a detailed analysis report.
        - The report includes:
          - Scale information
          - Multi-Otsu segmentation analysis
          - Particle shape analysis
          - Circularity distribution
        - Click **"Download PDF"** to save the report to your device.
        """,
    }

    st.sidebar.title("📖 **User Guide**")
    st.sidebar.markdown(guide_content.get(st.session_state.page, "No guide available for this page."))


# In[ ]:


import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from datetime import datetime

def generate_pdf():
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # **Title**
    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawCentredString(width / 2, 770, "SEM Image Analysis Report")
    pdf.setFont("Helvetica", 12)
    pdf.drawCentredString(width / 2, 750, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    pdf.line(50, 740, 550, 740)

    # **Insert SEM Image**
    if st.session_state.image:
        img_buffer = io.BytesIO()
        st.session_state.image.save(img_buffer, format="PNG")
        img_reader = ImageReader(img_buffer)
        pdf.drawImage(img_reader, 100, 520, width=400, height=200)

    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(50, 500, "Scale Information")
    pdf.setFont("Helvetica", 12)
    pixel_to_um = st.session_state.get('pixel_to_um', None)
    pdf.drawString(50, 480, f"Pixel to µm Ratio: {pixel_to_um:.6f} µm/px" if pixel_to_um else "⚠️ No scale information available.")
    pdf.line(50, 470, 550, 470)

    # **Multi-Otsu Segmentation Analysis**
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(50, 450, "Multi-Otsu Segmentation Analysis")
    pdf.setFont("Helvetica", 12)
    segmentation_data = st.session_state.get("analysis_df", None)
    y_offset = 430

    if segmentation_data is not None:
        for _, row in segmentation_data.iterrows():
            pdf.drawString(50, y_offset, f"{row['Layer']}: {row['Physical Area (µm²)']:.2f} µm² ({row['Area Percentage (%)']:.2f}%)")
            y_offset -= 20
    else:
        pdf.drawString(50, y_offset, "⚠️ No segmentation data available.")

    pdf.line(50, y_offset - 10, 550, y_offset - 10)
    y_offset -= 30

    # **Additional Metrics**
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(50, y_offset, "Additional Metrics")
    pdf.setFont("Helvetica", 12)
    additional_metrics = {
        "Porosity Ratio": st.session_state.get("porosity_ratio", None),
        "Catalyst Coverage": st.session_state.get("catalyst_coverage", None),
        "Agglomeration Ratio": st.session_state.get("agglomeration_ratio", None),
        "Oxidation/Impurity Coverage": st.session_state.get("oxidation_ratio", None)
    }
    y_offset -= 20

    for metric, value in additional_metrics.items():
        if value is not None:
            pdf.drawString(50, y_offset, f"{metric}: {value:.2f}%")
            y_offset -= 20
    pdf.line(50, y_offset - 10, 550, y_offset - 10)
    y_offset -= 30

    # **Particle Shape Analysis**
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(50, y_offset, "Particle Shape Analysis")
    pdf.setFont("Helvetica", 12)
    shape_analysis = st.session_state.get("shape_analysis", {})
    y_offset -= 20

    if shape_analysis:
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(50, y_offset, "Shape Type")
        pdf.drawString(250, y_offset, "Count")
        pdf.line(50, y_offset - 5, 550, y_offset - 5)
        y_offset -= 20

        pdf.setFont("Helvetica", 12)
        for shape, count in shape_analysis.items():
            pdf.drawString(50, y_offset, shape)
            pdf.drawString(250, y_offset, str(count))
            y_offset -= 20
    else:
        pdf.drawString(50, y_offset, "⚠️ No shape analysis data available.")

    pdf.line(50, y_offset - 10, 550, y_offset - 10)
    y_offset -= 30

    # **Circularity Analysis**
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(50, y_offset, "Circularity Distribution")
    pdf.setFont("Helvetica", 12)
    circularity_data = st.session_state.get("circularity_data", [])
    y_offset -= 20

    if circularity_data:
        avg_circularity = sum(circularity_data) / len(circularity_data)
        pdf.drawString(50, y_offset, f"Average Circularity: {avg_circularity:.2f}")
        y_offset -= 20
    else:
        pdf.drawString(50, y_offset, "⚠️ No circularity data available.")

    pdf.save()
    buffer.seek(0)
    return buffer


# In[ ]:


def download_report_page():
    inject_ga()
    st.title("📄 **Download Report**")
    st.write("Click the button below to generate and download your SEM analysis report as a PDF.")

    # **Visual separator**
    st.markdown("---")

    # **Center-aligned download button**
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

    # **Navigation button**
    col_prev, _, _ = st.columns([1, 3, 1])
    with col_prev:
        if st.button("⬅️ Previous", use_container_width=True):
            st.session_state.page -= 1


# In[ ]:


import numpy as np
import cv2
import streamlit as st
import plotly.graph_objects as go
from skimage.measure import label

# Debug: 检查原图强度范围
def check_image_intensity(image):
    image_np = np.array(image.convert("L"))
    pixel_min = np.min(image_np)
    pixel_max = np.max(image_np)
    st.write(f"Original Image Intensity Range: {pixel_min} to {pixel_max}")
    return image_np

# Debug: 检查分割后的图像（segmented_image）
def check_segmented_image(image_blur):
    thresholds = threshold_multiotsu(image_blur, classes=5)
    segmented_image = np.digitize(image_blur, bins=thresholds)
    st.write(f"Segmented Image Min: {np.min(segmented_image)} Max: {np.max(segmented_image)}")
    return segmented_image

# Debug: 检查每一层 mask 的强度范围
def check_mask_intensity(mask):
    ys, xs = np.where(mask > 0)  # 获取所有非零像素的位置
    pixel_values = mask[ys, xs]
    pixel_min = np.min(pixel_values)
    pixel_max = np.max(pixel_values)
    st.write(f"Layer {i+1} - Pixel Min: {pixel_min}, Pixel Max: {pixel_max}")
    return pixel_values

# Debug: 检查每一层的强度是否正常
def check_z_distribution(masks):
    for i, mask in enumerate(masks):
        pixel_values = check_mask_intensity(mask)  # 取出每一层的像素强度
        if np.min(pixel_values) == np.max(pixel_values):
            st.write(f"Layer {i+1}: All pixels have the same value {np.min(pixel_values)}")
        else:
            st.write(f"Layer {i+1} Intensity Range: {np.min(pixel_values)} to {np.max(pixel_values)}")

# 3D可视化函数
def view_3d_model():
    st.title("🧊 3D Layered Material Viewer")

    if "class_masks" not in st.session_state:
        st.error("⚠️ Please perform Multi-Otsu Segmentation first!")
        return

    masks = st.session_state.class_masks
    if not masks or len(masks) < 5:
        st.error("⚠️ Incomplete mask data.")
        return

    points = []

    # 定义颜色和透明度
    colors = ['rgba(255, 0, 0, 0.8)', 'rgba(0, 255, 0, 0.8)', 'rgba(0, 0, 255, 0.8)', 'rgba(255, 255, 0, 0.8)', 'rgba(255, 0, 255, 0.8)']

    # 根据原图的强度来设置 z 轴的位置
    for i, mask in enumerate(masks):
        ys, xs = np.where(mask > 0)  # 获取所有非零像素的位置

        # 读取原图的强度并进行归一化处理
        pixel_values = mask[ys, xs]  # 根据 mask 获取像素强度
        pixel_min = np.min(pixel_values)
        pixel_max = np.max(pixel_values)
        
        # 防止除以零，设置零强度为最小非零强度
        if pixel_max > pixel_min:
            normed_depth = (pixel_values - pixel_min) / (pixel_max - pixel_min)  # 强度归一化
        else:
            normed_depth = np.zeros_like(pixel_values)  # 如果最大值等于最小值，设置为零

        # 根据归一化后的强度设置 z 轴深度
        for y, x, depth in zip(ys, xs, normed_depth):
            points.append((x, y, depth, colors[i]))  # 颜色根据层次设置

    # 显示点数和范围
    x, y, z, color = zip(*points)
    total_voxels = len(x)
    st.write(f"Total points to render: {total_voxels}")

    # 使用 go.Scatter3d 显示每层颜色，增加层次感
    fig = go.Figure()

    for i, c in enumerate(colors):
        layer_points = [(x_val, y_val, z_val) for x_val, y_val, z_val, col in zip(x, y, z, color) if col == c]
        if layer_points:
            x_layer, y_layer, z_layer = zip(*layer_points)
            fig.add_trace(go.Scatter3d(
                x=x_layer,
                y=y_layer,
                z=z_layer,
                mode='markers',
                marker=dict(size=1, color=c, opacity=0.7),  # 固定粒子大小为1，并保持透明度
                name=f"Layer {i+1}"
            ))

    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Layer Depth',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        title="3D Visualization of 5-Layer Material Structure"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    🧬 This interactive 3D model shows **5 material layers** segmented from your SEM image.
    
    Each layer is visually separated in 3D space for clarity.
    
    Rotate, zoom, and explore internal structures layer-by-layer.
    """)

# 调用调试和可视化函数
def debug_process():
    if st.session_state.image is None:
        st.error("⚠️ Please upload an image first!")
        return
    
    # Step 1: 检查原始图像的强度
    image_np = check_image_intensity(st.session_state.image)
    
    # Step 2: 检查图像的分割
    image_eq = cv2.equalizeHist(image_np)  # 对图像进行直方图均衡化
    image_blur = cv2.GaussianBlur(image_eq, (5, 5), 0)  # 高斯模糊
    segmented_image = check_segmented_image(image_blur)

    # Step 3: 获取分割后的每一层
    st.session_state.class_masks = [(segmented_image == i).astype(np.uint8) * 255 for i in range(5)]
    
    # Step 4: 检查每一层的强度分布
    check_z_distribution(st.session_state.class_masks)

    # Step 5: 绘制 3D 模型
    view_3d_model()

# 调用调试和可视化过程
debug_process()


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
    
    st.image("cover_image.jpg", use_container_width=True)
    st.title("PEM Analysis")

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

        if st.button("Mark Scale"):
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

    # Navigation buttons (no container width)
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.session_state.page > 1:
            if st.button("⬅️ Previous"):
                st.session_state.page -= 1
    with col2:
        if st.session_state.page < 6:
            if st.button("Next ➡️"):
                st.session_state.page += 1

if __name__ == "__main__":
    main()

