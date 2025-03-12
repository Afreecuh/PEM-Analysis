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

    # **產生分類遮罩並計算統計數據**
    class_masks = [(segmented_image == i).astype(np.uint8) * 255 for i in range(num_classes)]
    labeled_masks = [label(mask) for mask in class_masks]  # 標記區域
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
    
    # **顯示對應的遮罩圖片**
    st.write("### Layer Visualization")
    selected_layer = st.selectbox("選擇要顯示的 Layer", layer_labels, key="layer_select")
    if selected_layer:
        selected_index = layer_labels.index(selected_layer)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(class_masks[selected_index], cmap="gray")
        ax.set_title(f"{selected_layer} (Layer {selected_index})")
        ax.axis("off")
        st.pyplot(fig)
    
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


# In[ ]:


import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from skimage.filters import threshold_multiotsu
from PIL import Image

# **統一 Multi-Otsu 分割區間數為 5**
NUM_CLASSES = 5  

# **產生標準圓形模板**
def generate_circular_template(size=50):
    template = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(template, (size//2, size//2), size//3, 255, -1)
    return template

CIRCULAR_TEMPLATE = generate_circular_template()

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
    return circularity, aspect_ratio, solidity, (x, y, w, h)

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
    img_blur = cv2.medianBlur(img_eq, 5)

    # **Multi-Otsu 分割**
    thresholds = threshold_multiotsu(img_blur, classes=NUM_CLASSES)
    segmented = np.digitize(img_blur, bins=thresholds)

    # **產生二值化遮罩**
    binary = (segmented == 2).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # **標記影像**
    segmented_normalized = (segmented - segmented.min()) / (segmented.max() - segmented.min())
    segmented_colored = cm.jet(segmented_normalized)
    img_without_contours = (segmented_colored[:, :, :3] * 255).astype(np.uint8)
    img_with_contours = img_without_contours.copy()

    areas = []
    shape_results = []
    ncc_scores = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 50:
            circularity, aspect_ratio, solidity, (x, y, w, h) = calculate_shape_features(contour)
            shape = classify_shape(circularity, aspect_ratio, solidity)

            # **計算 NCC 相似度**
            shape_roi = img_blur[y:y+h, x:x+w]
            if shape_roi.shape[0] > 10 and shape_roi.shape[1] > 10:
                shape_resized = cv2.resize(shape_roi, (50, 50))
                ncc_score = cv2.matchTemplate(shape_resized, CIRCULAR_TEMPLATE, cv2.TM_CCOEFF_NORMED).max()
                ncc_scores.append(ncc_score)

            # **畫出顆粒輪廓**
            cv2.drawContours(img_with_contours, [contour], -1, (0, 255, 255), 2)
            cv2.rectangle(img_with_contours, (x, y), (x + w, y + h), (255, 0, 255), 2)
            cv2.putText(img_with_contours, shape, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            shape_results.append(f"Shape: {shape} | NCC: {ncc_score:.2f} | Area: {area:.2f}px² | Circularity: {circularity:.2f}, Aspect Ratio: {aspect_ratio:.2f}, Solidity: {solidity:.2f}")
            areas.append(area)

    return img_without_contours, img_with_contours, shape_results, areas, img_blur, thresholds, ncc_scores

# **Streamlit 介面**
st.title("🔬 SEM 顆粒形狀分析（Multi-Otsu & NCC）")
st.write("上傳 SEM 影像以分析顆粒形狀、圓度（Circularity）、NCC 形狀匹配。")

uploaded_file = st.file_uploader("上傳 SEM 影像（PNG, JPG, BMP）", type=["png", "jpg", "bmp"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="原始影像", use_column_width=True)

    # **執行分析**
    img_without_contours, img_with_contours, result_text, areas, img_blur, thresholds, ncc_scores = analyze_particles(image)

    # **顯示處理後影像**
    st.subheader("📌 分割影像（未標記）")
    st.image(img_without_contours, caption="分割後影像", use_column_width=True)

    st.subheader("📌 分割影像（已標記顆粒）")
    st.image(img_with_contours, caption="顆粒分析", use_column_width=True)

    # **顯示形狀分析結果**
    st.subheader("📊 顆粒形狀分析")
    if result_text:
        for item in result_text:
            st.write(item)
    else:
        st.write("未檢測到顯著顆粒")

    # **繪製 Multi-Otsu 門檻直方圖**
    st.subheader("📉 Multi-Otsu 門檻直方圖")
    fig_otsu, ax_otsu = plt.subplots(figsize=(6, 3))
    hist, bins = np.histogram(img_blur.ravel(), bins=256, range=[0, 256])
    ax_otsu.plot(bins[:-1], hist, color='black')
    for thresh in thresholds:
        ax_otsu.axvline(thresh, color='red', linestyle='--', linewidth=1.5)
    ax_otsu.set_title("Histogram with Multi-Otsu Thresholding")
    st.pyplot(fig_otsu)

    # **繪製 NCC 分數直方圖**
    if ncc_scores:
        st.subheader("📊 NCC 分數分佈")
        fig_ncc, ax_ncc = plt.subplots(figsize=(6, 3))
        ax_ncc.hist(ncc_scores, bins=10, color='green', alpha=0.7, edgecolor='black')
        ax_ncc.set_title("NCC Score Distribution")
        st.pyplot(fig_ncc)


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

    # **側邊欄使用者指南**
    st.sidebar.header("User Guide")
    guide_text = (
        "1. Upload an image.\n"
        "2. Click two points on the image to mark the scale.\n"
        "3. Enter the actual scale length (µm).\n"
        "4. The system calculates **µm/px** automatically.\n"
        "5. Navigate between pages using the buttons below."
    )
    st.sidebar.info(guide_text)

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



# **主函式**
def main():
    if st.session_state.page == 1:
        upload_and_mark_scale()
    elif st.session_state.page == 2:
        otsu_segmentation()  # Multi-Otsu + 面積分析
    elif st.session_state.page == 3:
        analyze_particles_page()  # Multi-Otsu + NCC + 形狀分類分析

    # **頁面導航按鈕**
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.session_state.page > 1:
            if st.button("Previous"):
                st.session_state.page -= 1
    with col2:
        if st.session_state.page < 4:
            if st.button("Next"):
                st.session_state.page += 1

if __name__ == "__main__":
    main()


