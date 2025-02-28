#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import cv2
import numpy as np
import pandas as pd
import networkx as nx
from PIL import Image
import matplotlib.pyplot as plt
import re
import streamlit.components.v1 as components
import io
import math
import time
import plotly.express as px
import plotly.graph_objects as go


# In[3]:


# 儲存用戶點選的比例尺座標
if "scale_coords" not in st.session_state:
    st.session_state["scale_coords"] = []

def plot_image_with_annotations(image_path):
    """使用 Plotly 顯示圖片，並讓用戶點擊標註比例尺範圍"""
    image = Image.open(image_path)
    fig = px.imshow(np.array(image))

    # 添加標註點
    for coord in st.session_state["scale_coords"]:
        fig.add_trace(go.Scatter(
            x=[coord[0]],
            y=[coord[1]],
            mode="markers",
            marker=dict(color="red", size=10),
            name="標註點"
        ))

    fig.update_layout(dragmode="drawopenpath")
    return fig

# Streamlit 介面
st.title("比例尺標註工具（適用於 Streamlit Cloud）")
st.write("請上傳圖片，然後在圖片上點擊 **兩個點** 來標註比例尺長度，並輸入實際 µm 長度")

# **步驟 1：上傳圖片**
uploaded_file = st.file_uploader("上傳圖片", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # **儲存上傳的圖片**
    image_path = "temp_uploaded_image.png"
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # **顯示圖片（用 Plotly）**
    st.session_state["scale_coords"] = []  # 清空之前的標註點
    fig = plot_image_with_annotations(image_path)
    selected_points = st.plotly_chart(fig, use_container_width=True)

    # **步驟 2：用戶點擊圖片來選取比例尺範圍**
    st.write("請在上方圖片點擊兩個點來標註比例尺範圍（水平距離）")

    # **檢查是否已選取兩個點**
    if len(st.session_state["scale_coords"]) == 2:
        x1, y1 = st.session_state["scale_coords"][0]
        x2, y2 = st.session_state["scale_coords"][1]
        
        # 只計算 X 方向的像素長度
        scale_pixels = abs(x2 - x1)

        st.success(f"你標註的比例尺長度（像素）: {scale_pixels:.2f} px")

        # **步驟 3：用戶輸入實際比例尺長度**
        scale_length_input = st.text_input("請輸入比例尺的實際長度 (µm):", "10")

        if st.button("計算 µm/px"):
            try:
                scale_length_um = float(scale_length_input)
                pixel_to_um = scale_length_um / scale_pixels
                st.success(f"📏 解析比例尺: {scale_length_um:.2f} µm（{pixel_to_um:.4f} µm/px）")
            except ValueError:
                st.error("⚠️ 輸入格式錯誤，請輸入數字")


# In[5]:


# 使用 Otsu + 形態學處理進行圖像分割
def segment_image_otsu(image):
    img_array = np.array(image.convert("L"))  # 轉換為灰度
    
    # Otsu 阈值法
    _, binary = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 形態學操作 - 去除噪聲
    kernel = np.ones((3,3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    binary = cv2.dilate(binary, kernel, iterations=1)
    
    # Canny 邊緣檢測
    edges = cv2.Canny(binary, 100, 200)
    return edges, binary


# 預處理圖像並檢測顆粒
def detect_particles(image, pixel_to_um):
    edges, binary = segment_image_otsu(image)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    particle_data = []
    centroids = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            area_px = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area_px / (perimeter ** 2) if perimeter > 0 else 0
            area_um = area_px * (pixel_to_um ** 2)
            centroids.append((cx, cy))
            particle_data.append((cx, cy, area_um, circularity))
    return centroids, particle_data, binary

def detect_tpb_points_improved(centroids, pixel_to_um, threshold_um=0.5, angle_threshold=30):
    """
    改進版的 TPB 檢測函式：
      - 根據距離閥值建立鄰近關係。
      - 對每個與至少兩個鄰居連接的節點，計算從該節點指向每個鄰居的向量，
        並計算所有向量之間的夾角，以最小夾角作為可信度依據。
    
    參數：
      centroids: list of (x, y) 粒子質心
      pixel_to_um: 每個像素代表多少 µm
      threshold_um: 鄰近距離閥值，預設 0.5 µm
      angle_threshold: 角度過濾閥值，低於此值則可信度為 0（單位：度）
    
    回傳：
      tpb_points: 檢測到的 TPB 點列表
      confidence_scores: 每個 TPB 點對應的可信度分數（0～1）
    """
    threshold_px = threshold_um / pixel_to_um
    G = nx.Graph()
    
    # 建立鄰近關係，僅比較一次 (i < j)
    for i, (x1, y1) in enumerate(centroids):
        for j, (x2, y2) in enumerate(centroids):
            if i < j:
                distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                if distance < threshold_px:
                    G.add_edge(i, j)
    
    tpb_points = []
    confidence_scores = []
    for node in G.nodes:
        neighbors = list(G[node])
        if len(neighbors) >= 2:
            x0, y0 = centroids[node]
            vectors = []
            for n in neighbors:
                xn, yn = centroids[n]
                dx, dy = xn - x0, yn - y0
                norm = math.hypot(dx, dy)
                if norm > 0:
                    vectors.append((dx / norm, dy / norm))
            if len(vectors) < 2:
                continue
            
            angles = []
            for i in range(len(vectors)):
                for j in range(i+1, len(vectors)):
                    dot = np.clip(vectors[i][0] * vectors[j][0] + vectors[i][1] * vectors[j][1], -1.0, 1.0)
                    angle = math.degrees(math.acos(dot))
                    angles.append(angle)
            
            if angles:
                min_angle = min(angles)
                if min_angle < angle_threshold:
                    conf = 0.0
                else:
                    conf = min_angle / 90.0
                    conf = max(0.0, min(conf, 1.0))
                
                tpb_points.append(centroids[node])
                confidence_scores.append(conf)
    
    return tpb_points, confidence_scores


# In[9]:


# 可視化 TPB 結果
def visualize_tpb(image, centroids, tpb_points):
    img_array = np.array(image.convert("RGB"))  # 轉換為 RGB 格式
    img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    for (cx, cy) in centroids:
        cv2.circle(img, (cx, cy), 3, (0, 255, 0), -1)  # 綠色標記顆粒
    for (tx, ty) in tpb_points:
        cv2.circle(img, (tx, ty), 3, (0, 0, 255), -1)  # 紅色標記 TPB
    
    output_path = "tpb_output.png"
    cv2.imwrite(output_path, img)
    return output_path

# 計算 TPB 密度
def calculate_tpb_density(tpb_points, image_shape, pixel_to_um):
    img_area_um2 = (image_shape[0] * pixel_to_um) * (image_shape[1] * pixel_to_um)
    unit_count = img_area_um2 / 10
    tpb_density = len(tpb_points) / unit_count if unit_count > 0 else 0
    return tpb_density

# 保存數據到 CSV
def save_tpb_data(particle_data, tpb_points):
    df = pd.DataFrame(particle_data, columns=["Centroid_X", "Centroid_Y", "Area_um2", "Circularity"])
    df_tpb = pd.DataFrame(tpb_points, columns=["TPB_X", "TPB_Y"])
    df.to_csv("particles.csv", index=False)
    df_tpb.to_csv("tpb_points.csv", index=False)

    # 繪製可信度直方圖
def plot_confidence_histogram(confidence_scores):
    """
    繪製候選 TPB 點可信度分布直方圖，並返回 matplotlib 的 Figure 對象。
    """
    import matplotlib.pyplot as plt  # 確保導入 matplotlib
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.hist(confidence_scores, bins=10, range=(0, 1), edgecolor='black')
    ax.set_title("TPB 點可信度分布")
    ax.set_xlabel("可信度")
    ax.set_ylabel("點的數量")
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    return fig

def show_progress_overlay(duration=3000, tip_text="Tips: Use a clear, high-resolution image for better analysis results."):
    """
    在 Streamlit 顯示進度條，並提供提示資訊。
    duration: 顯示時長（毫秒）
    tip_text: 提示文字
    """
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(100):
        time.sleep(duration / 1000 / 100)  # 根據 duration 時長調整進度速度
        progress_bar.progress(i + 1)
    
    status_text.text(tip_text)
    time.sleep(1)
    progress_bar.empty()
    status_text.empty()

def hide_progress_bar():
    """ 清除進度條 """
    st.empty()


# In[17]:


def classify_shape(contour):
    """
    根據輪廓近似多邊形的頂點數和圓形度判斷顆粒的形狀。
    """
    epsilon = 0.03 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx) == 3:
        return "Triangle"
    elif len(approx) == 4:
        return "Quadrilateral"
    elif len(approx) >= 5:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return "Other"
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity > 0.7:
            return "Circle"
        else:
            return "Other"
    else:
        return "Other"

def compute_shape_ratios(binary_uint8, min_area=10):
    """
    偵測輪廓，並分類形狀，返回形狀數量和比例。
    """
    contours, _ = cv2.findContours(binary_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shape_counts = {"Triangle": 0, "Quadrilateral": 0, "Circle": 0, "Other": 0}
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        shape = classify_shape(cnt)
        shape_counts[shape] += 1
    total = sum(shape_counts.values())
    ratios = {k: (v / total) * 100 if total > 0 else 0 for k, v in shape_counts.items()}
    return shape_counts, ratios

def format_shape_composition(ratios):
    """
    格式化形狀比例結果。
    """
    return ", ".join(f"{shape}: {perc:.1f}%" for shape, perc in ratios.items())

def annotate_image_with_shapes(orig_pil, binary_uint8):
    """
    在圖片上標註顆粒形狀。
    """
    orig_cv = cv2.cvtColor(np.array(orig_pil), cv2.COLOR_RGB2BGR)
    contours, _ = cv2.findContours(binary_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        shape = classify_shape(cnt)
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(orig_cv, shape[0], (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return Image.fromarray(cv2.cvtColor(orig_cv, cv2.COLOR_BGR2RGB))

def plot_shape_composition_bar(ratios):
    """
    繪製形狀組成比例的柱狀圖。
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.bar(ratios.keys(), ratios.values(), color='skyblue', edgecolor='black')
    ax.set_title("Shape Composition", fontsize=14, fontweight='bold')
    ax.set_xlabel("Shape", fontsize=12)
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_ylim(0, 100)
    for i, v in enumerate(ratios.values()):
        ax.text(i, v + 1, f"{v:.1f}%", ha='center', fontsize=10)
    return fig


# In[8]:


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

# **用戶標註比例尺功能**
def plot_image_with_annotations(image_path):
    """顯示圖片，並讓用戶點擊兩個點來標註比例尺"""
    image = Image.open(image_path)
    fig = px.imshow(np.array(image))

    # 添加標註點
    for coord in st.session_state["scale_coords"]:
        fig.add_trace(go.Scatter(
            x=[coord[0]],
            y=[coord[1]],
            mode="markers",
            marker=dict(color="red", size=10),
            name="標註點"
        ))

    fig.update_layout(dragmode="drawopenpath")
    return fig

# **頁面 1：上傳圖片與比例尺標註**
def upload_image():
    st.image("cover_image.jpg", use_container_width=True)
    st.title("PEM Analysis")

    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], key="image_upload")

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.session_state.image = image  # 存儲圖片
        st.success("Image uploaded successfully! Please mark the scale.")

        # 儲存圖片
        image_path = "temp_uploaded_image.png"
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # **顯示圖片，讓用戶點擊標註比例尺**
        fig = plot_image_with_annotations(image_path)
        selected_points = st.plotly_chart(fig, use_container_width=True)

        # **步驟 2：用戶標註比例尺**
        st.write("請在圖片上點擊 **兩個點** 來標註比例尺範圍（水平長度）")

        # **檢查是否已選取兩個點**
        if len(st.session_state["scale_coords"]) == 2:
            x1, y1 = st.session_state["scale_coords"][0]
            x2, y2 = st.session_state["scale_coords"][1]
            scale_pixels = abs(x2 - x1)  # **只計算 X 方向的距離**
            st.session_state.scale_pixels = scale_pixels

            st.success(f"你標註的比例尺長度（像素）: {scale_pixels:.2f} px")

            # **步驟 3：用戶輸入比例尺的實際長度**
            scale_length_input = st.text_input("請輸入比例尺的實際長度 (µm):", "10")

            if st.button("計算 µm/px"):
                try:
                    scale_length_um = float(scale_length_input)
                    st.session_state.scale_length_um = scale_length_um
                    pixel_to_um = scale_length_um / scale_pixels
                    st.session_state.pixel_to_um = pixel_to_um
                    st.success(f"📏 解析比例尺: {scale_length_um:.2f} µm（{pixel_to_um:.4f} µm/px）")
                except ValueError:
                    st.error("⚠️ 輸入格式錯誤，請輸入數字")

# **頁面 2：TPB 分析結果**
def show_tpb_results():
    st.title("PEM Analysis - Results")
    if "image" in st.session_state and st.session_state.image is not None:
        st.image(st.session_state.image, caption="Processed Image", use_container_width=True)

    # **顯示 µm/px**
    if "pixel_to_um" in st.session_state and st.session_state.pixel_to_um is not None:
        st.info(f"📏 解析比例尺: **1 pixel ≈ {st.session_state.pixel_to_um:.4f} µm**")

    # **繪製 TPB 信心分佈圖**
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.hist([0.1, 0.3, 0.5, 0.7, 0.9], bins=10, range=(0, 1), edgecolor="black")
    ax.set_title("TPB confidence distribution")
    ax.set_xlabel("Confidence score")
    ax.set_ylabel("Number of TPB candidates")
    st.pyplot(fig)

# **頁面 3：形態學分析**
def show_morphology_analysis():
    st.title("Morphology Analysis")
    if "image" in st.session_state and st.session_state.image is not None:
        st.image(st.session_state.image, caption="Morphology Processed Image", use_container_width=True)
    st.write("Shape composition analysis will be displayed here.")

# **頁面 4：結束頁面**
def show_final_page():
    st.title("Final Page")
    st.write("(Empty Page)")
    if st.button("Restart"):
        st.session_state.page = 1

# **側邊欄使用者指南**
def show_user_guide():
    st.sidebar.header("User Guide")
    guide_text = (
        "1. Click **'Upload Image'** to select an image file.\n"
        "2. Click two points on the image to mark the scale.\n"
        "3. Enter the actual scale length (µm).\n"
        "4. The system calculates **µm/px** automatically.\n"
        "5. Navigate between pages using the buttons below."
    )
    st.sidebar.info(guide_text)

# **主函式**
def main():
    inject_ga()
    show_user_guide()
    
    if st.session_state.page == 1:
        upload_image()
    elif st.session_state.page == 2:
        show_tpb_results()
    elif st.session_state.page == 3:
        show_morphology_analysis()
    elif st.session_state.page == 4:
        show_final_page()

    if st.button("Next", key="next_button"):
        next_page()

if __name__ == "__main__":
    main()

