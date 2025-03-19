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


# **初始化 Session State**
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

# **用戶點擊時更新標註點**
def update_coords(click_x, click_y):
    """記錄用戶點擊的座標，最多允許兩個點"""
    if len(st.session_state.scale_coords) < 2:
        st.session_state.scale_coords.append((click_x, click_y))
        st.rerun()  # 讓 UI 重新更新，顯示紅點
    else:
        st.warning("⚠️ 已標註兩個點，請輸入比例尺長度！")

# **顯示圖片並讓用戶標註比例尺**
def plot_image_with_annotations():
    """顯示圖片，並讓用戶點擊標註比例尺範圍"""
    image = st.session_state.image
    fig = px.imshow(np.array(image))

    # **添加標註點**
    for coord in st.session_state.scale_coords:
        fig.add_trace(go.Scatter(
            x=[coord[0]],
            y=[coord[1]],
            mode="markers",
            marker=dict(color="red", size=10),
            name="標註點"
        ))

    return fig

# **處理比例尺標註與計算 µm/px**
def handle_scale_annotation():
    if len(st.session_state.scale_coords) == 2:
        x1, y1 = st.session_state.scale_coords[0]
        x2, y2 = st.session_state.scale_coords[1]
        scale_pixels = abs(x2 - x1)  # **只計算 X 方向的距離**
        st.session_state.scale_pixels = scale_pixels

        st.success(f"✅ 你已選取比例尺範圍: {scale_pixels:.2f} px")
        
        # **輸入比例尺的實際長度**
        scale_length_input = st.text_input("請輸入比例尺的實際長度 (µm):", "10")

        if st.button("計算 µm/px"):
            try:
                scale_length_um = float(scale_length_input)
                st.session_state.scale_length_um = scale_length_um
                pixel_to_um = scale_length_um / scale_pixels
                st.session_state.pixel_to_um = pixel_to_um
                st.success(f"📏 計算結果: {scale_length_um:.2f} µm（{pixel_to_um:.4f} µm/px）")
            except ValueError:
                st.error("⚠️ 輸入格式錯誤，請輸入數字")


# In[ ]:


from skimage.measure import regionprops, label

def otsu_segmentation():
    inject_ga()
    st.title("Multi-Otsu Thresholding & SEM Analysis")
    
    # **固定分割區間數為 5**
    num_classes = 5  
    
    if st.session_state.image is None or st.session_state.pixel_to_um is None:
        st.error("⚠️ 請先上傳圖片並設定比例尺！")
        return

    image_np = np.array(st.session_state.image.convert("L"))
    pixel_to_um = st.session_state.pixel_to_um

    # **應用 Multi-Otsu 門檻分割**
    thresholds = threshold_multiotsu(image_np, classes=num_classes)
    segmented_image = np.digitize(image_np, bins=thresholds)

    # **確保 class_masks 只計算一次**
    if "class_masks" not in st.session_state:
        st.session_state.class_masks = [(segmented_image == i).astype(np.uint8) * 255 for i in range(num_classes)]

    # **確保 Layer 選擇不會影響頁面狀態**
    if "selected_layer_index" not in st.session_state:
        st.session_state.selected_layer_index = 0  # **預設選擇第一個 Layer**

    # **產生分類遮罩並計算統計數據**
    labeled_masks = [label(mask) for mask in st.session_state.class_masks]  # 標記區域
    class_properties = [regionprops(labeled_mask) for labeled_mask in labeled_masks]
    
    # **計算每個類別的區域統計資訊**
    num_regions = [len(props) for props in class_properties]
    avg_area_per_region = [
        np.mean([prop.area for prop in props]) if len(props) > 0 else 0 
        for props in class_properties
    ]
    
    # **計算面積分析**
    pixel_areas = [(segmented_image == i).sum() for i in range(num_classes)]
    total_area = sum(pixel_areas)
    real_physical_sizes = [area * (pixel_to_um ** 2) for area in pixel_areas]
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

    # **顯示表格**
    st.dataframe(df_analysis)

    # **可視化 - 可點擊的 bar chart**
    fig_bar = px.bar(df_analysis, x="Layer", y="Physical Area (µm²)", title="Physical Area of Each Layer")
    fig_bar.update_traces(customdata=layer_labels, hoverinfo="x+y")
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # **顯示 Layer Visualization**
    st.write("### Layer Visualization")

    # **使用 session_state 綁定 selectbox**
    selected_layer = st.selectbox(
        "選擇要顯示的 Layer", 
        layer_labels, 
        index=st.session_state.get("selected_layer_index", 0),
        key="layer_selection"
    )

    # **更新 session_state，確保不影響頁面跳轉**
    if selected_layer:
        st.session_state.selected_layer_index = layer_labels.index(selected_layer)

    # **顯示選定 Layer 的遮罩**
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(st.session_state.class_masks[st.session_state.selected_layer_index], cmap="gray")
    ax.set_title(f"{selected_layer} (Layer {st.session_state.selected_layer_index})")
    ax.axis("off")
    st.pyplot(fig)
    
    # **可視化 - 圖表**
    fig_pie = px.pie(df_analysis, names="Layer", values="Area Percentage (%)", title="Area Distribution Across Layers")
    st.plotly_chart(fig_pie, use_container_width=True)

    # **計算額外指標**
    porosity_ratio = (pixel_areas[0] / total_area) * 100 if len(pixel_areas) > 0 else 0
    catalyst_areas = sum(pixel_areas[2:4]) if len(pixel_areas) > 3 else 0
    catalyst_percentage = (catalyst_areas / total_area) * 100 if total_area > 0 else 0
    agglomeration_ratio = (pixel_areas[3] / catalyst_areas) * 100 if len(pixel_areas) > 3 and catalyst_areas > 0 else 0
    oxidation_ratio = (pixel_areas[4] / total_area) * 100 if len(pixel_areas) > 4 else 0

    st.write(f"📌 孔隙率: {porosity_ratio:.2f}%")
    st.write(f"📌 催化劑覆蓋率: {catalyst_percentage:.2f}%")
    st.write(f"📌 團聚比: {agglomeration_ratio:.2f}%")
    st.write(f"📌 氧化/雜質覆蓋率: {oxidation_ratio:.2f}%")

    # **存入 session_state**
    st.session_state.analysis_df = df_analysis
    st.session_state.extra_metrics = {
        "Porosity Ratio": (pixel_areas[0] / total_area) * 100 if total_area > 0 else 0,
        "Catalyst Coverage": (sum(pixel_areas[2:4]) / total_area) * 100 if total_area > 0 else 0,
        "Agglomeration Ratio": (pixel_areas[3] / sum(pixel_areas[2:4])) * 100 if sum(pixel_areas[2:4]) > 0 else 0,
        "Oxidation/Impurity Coverage": (pixel_areas[4] / total_area) * 100 if total_area > 0 else 0,
    }



# In[ ]:


import cv2
import numpy as np
import streamlit as st
import plotly.express as px
from skimage.filters import threshold_multiotsu
from PIL import Image

# **統一 Multi-Otsu 分割區間數為 4**
NUM_CLASSES = 4  

# **計算形狀特徵**
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

# **分類顆粒形狀**
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

# **分析顆粒**
def analyze_particles(image):
    img_gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    img_eq = cv2.equalizeHist(img_gray)
    img_blur = cv2.GaussianBlur(img_eq, (5, 5), 0)  # **還原為 GaussianBlur**

    # **Multi-Otsu 分割**
    thresholds = threshold_multiotsu(img_blur, classes=NUM_CLASSES)
    segmented = np.digitize(img_blur, bins=thresholds)

    # **產生二值化遮罩**
    binary = (segmented == 2).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    circularities = []
    shape_labels = []

    # **生成輪廓標記影像**
    img_with_contours = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 50:  # **過濾掉小面積顆粒**
            circularity, aspect_ratio, solidity = calculate_shape_features(contour)
            shape = classify_shape(circularity, aspect_ratio, solidity)
            circularities.append(circularity)
            shape_labels.append(shape)

            # **在影像上標註顆粒輪廓**
            cv2.drawContours(img_with_contours, [contour], -1, (0, 255, 255), 2)

    return circularities, shape_labels, binary, img_with_contours

# **Streamlit 介面**
def analyze_particles_page():
    st.title("🔬 SEM 顆粒形狀分析")

    if st.session_state.image is None:
        st.error("⚠️ 請先上傳圖片並設定比例尺！")
        return

    image = st.session_state.image

    # **執行分析**
    circularities, shape_labels, binary_image, img_with_contours = analyze_particles(image)

    # **繪製 Circularity Distribution 直方圖**
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

        # **用 expander 顯示 Binary Segmentation**
        with st.expander("🔍 Show Processed Binary Image"):
            st.image(binary_image, caption="Binary Segmentation (Used for Circularity Calculation)", use_column_width=True, clamp=True)

    else:
        st.warning("⚠️ 沒有偵測到顆粒")

    # **繪製 Shape Analysis 直方圖**
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

        # **用 expander 顯示 Segmented Image with Contours**
        with st.expander("🔍 Show Shape Contour Image"):
            st.image(img_with_contours, caption="Segmented Image with Contours", use_column_width=True)

    else:
        st.warning("⚠️ 沒有偵測到顆粒")


# User Guide

# In[ ]:


def show_user_guide():
    """顯示 User Guide 內容，根據目前頁面選擇對應說明"""
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

    # **標題**
    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawCentredString(width / 2, 770, "SEM Image Analysis Report")
    pdf.setFont("Helvetica", 12)
    pdf.drawCentredString(width / 2, 750, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    pdf.line(50, 740, 550, 740)

    # **插入 SEM 圖像**
    if st.session_state.image:
        img_buffer = io.BytesIO()
        st.session_state.image.save(img_buffer, format="PNG")
        img_reader = ImageReader(img_buffer)
        pdf.drawImage(img_reader, 100, 520, width=400, height=200)  # **調整尺寸並置中**

    # **比例尺資訊**
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(50, 500, "Scale Information")
    pdf.setFont("Helvetica", 12)
    pixel_to_um = st.session_state.get('pixel_to_um', None)
    if pixel_to_um:
        pdf.drawString(50, 480, f"Pixel to µm Ratio: {pixel_to_um:.6f} µm/px")  # **控制顯示小數位數**
    else:
        pdf.drawString(50, 480, "⚠️ No scale information available.")

    pdf.line(50, 470, 550, 470)  # **分隔線**

    # **Multi-Otsu 分割分析**
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(50, 450, "Multi-Otsu Segmentation Analysis")
    pdf.setFont("Helvetica", 12)
    analysis_df = st.session_state.get("analysis_df", None)
    y_offset = 430
    if analysis_df is not None and not analysis_df.empty:
        for _, row in analysis_df.iterrows():
            pdf.drawString(50, y_offset, f"{row['Layer']}: {row['Physical Area (µm²)']:.2f} µm² ({row['Area Percentage (%)']:.2f}%)")
            y_offset -= 20
    else:
        pdf.drawString(50, 430, "⚠️ No segmentation data available.")

    pdf.line(50, y_offset - 10, 550, y_offset - 10)  # **分隔線**
    y_offset -= 30

    # **額外分析指標**
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(50, y_offset, "Additional Metrics")
    pdf.setFont("Helvetica", 12)
    extra_metrics = st.session_state.get("extra_metrics", {})
    y_offset -= 20
    if extra_metrics:
        for key, value in extra_metrics.items():
            pdf.drawString(50, y_offset, f"{key}: {value:.2f}%")
            y_offset -= 20
    else:
        pdf.drawString(50, y_offset, "⚠️ No additional metrics available.")

    pdf.line(50, y_offset - 10, 550, y_offset - 10)  # **分隔線**
    y_offset -= 30

    # **顆粒形狀分析**
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(50, y_offset, "Particle Shape Analysis")
    pdf.setFont("Helvetica", 12)
    shape_analysis = st.session_state.get("shape_analysis", {})
    y_offset -= 20
    if shape_analysis:
        for shape, count in shape_analysis.items():
            pdf.drawString(50, y_offset, f"{shape}: {count} particles")
            y_offset -= 20
    else:
        pdf.drawString(50, y_offset, "⚠️ No shape analysis data available.")

    pdf.line(50, y_offset - 10, 550, y_offset - 10)  # **分隔線**
    y_offset -= 30

    # **Circularity 分析**
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(50, y_offset, "Circularity Distribution")
    pdf.setFont("Helvetica", 12)
    circularity_data = st.session_state.get("circularity_data", [])
    y_offset -= 20
    if circularity_data:
        avg_circularity = sum(circularity_data) / len(circularity_data)
        pdf.drawString(50, y_offset, f"Average Circularity: {avg_circularity:.2f}")
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

    # **增加區塊外觀**
    st.markdown("---")

    # **居中顯示下載按鈕**
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

    # **頁面導航按鈕**
    col_prev, _, _ = st.columns([1, 3, 1])
    with col_prev:
        if st.button("⬅️ Previous", use_container_width=True):
            st.session_state.page -= 1


# In[3]:


# **Google Analytics 追蹤碼**
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

# **初始化 Session State**
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

# **頁面 1：上傳圖片 + 標註比例尺**
def upload_and_mark_scale():
    inject_ga()  # **確保 Google Analytics 被執行**
    
    st.image("cover_image.jpg", use_container_width=True)  # **確保封面圖片仍然顯示**
    st.title("PEM Analysis")  # **保留唯一標題**


    # **單一上傳圖片區塊**
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], key="image_upload")

    if uploaded_file:
        # **儲存圖片**
        st.session_state.image = Image.open(uploaded_file)  # 存入 Session State
        st.success("✅ Image uploaded successfully! Please mark the scale.")

        # **手動輸入點擊座標**
        st.write("請手動輸入兩個點的座標（X 和 Y）：")
        col1, col2 = st.columns(2)
        with col1:
            x1 = st.number_input("第一點 X 座標", min_value=0, step=1, key="x1_input")
            x2 = st.number_input("第二點 X 座標", min_value=0, step=1, key="x2_input")
        with col2:
            y1 = st.number_input("第一點 Y 座標", min_value=0, step=1, key="y1_input")
            y2 = st.number_input("第二點 Y 座標", min_value=0, step=1, key="y2_input")

        if st.button("標註比例尺"):
            if x1 != x2 or y1 != y2:  # 確保兩點不同
                st.session_state.scale_coords = [(x1, y1), (x2, y2)]
                st.success(f"✅ 你已選取比例尺範圍: {abs(x2 - x1):.2f} px")
                st.rerun()  # **強制重新繪製紅點**
            else:
                st.error("⚠️ 兩個座標不能完全相同，請重新輸入！")

        # **顯示圖片 + 即時更新標註點**
        fig = plot_image_with_annotations()
        st.plotly_chart(fig, use_container_width=True)

        # **處理比例尺標註與計算 µm/px**
        handle_scale_annotation()


def main():
    if "page" not in st.session_state:
        st.session_state.page = 1  # 預設第一頁

    # **顯示 User Guide**
    show_user_guide()  

    if st.session_state.page == 1:
        upload_and_mark_scale()
    elif st.session_state.page == 2:
        otsu_segmentation()
    elif st.session_state.page == 3:
        analyze_particles_page()
    elif st.session_state.page == 4:
        download_report_page()  # **新增下載報告頁面**

    # **頁面導航按鈕**
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.session_state.page > 1:
            if st.button("Previous"):
                st.session_state.page -= 1
    with col2:
        if st.session_state.page < 4:  # **修改最大頁數**
            if st.button("Next"):
                st.session_state.page += 1


if __name__ == "__main__":
    main()

