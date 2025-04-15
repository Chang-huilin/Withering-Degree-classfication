import os
import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import graycomatrix, graycoprops

# 设置目标文件夹路径
image_folder = r"D:\红茶数据2024.0423\工业相机\自然\剪裁后"
output_csv = os.path.join(image_folder, "Combined_Features.csv")

# GLCM 参数
GLCM_PROPERTIES = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']

# 初始化存储结果的列表
data = []

# 遍历文件夹内的所有 BMP 图片
for filename in os.listdir(image_folder):
    if filename.endswith(".bmp"):
        image_path = os.path.join(image_folder, filename)

        # 读取灰度图
        image = Image.open(image_path).convert("L")
        image_array = np.array(image)

        ### 计算直方图统计矩特征 ###
        hist, _ = np.histogram(image_array, bins=256, range=(0, 256))
        hist = hist / hist.sum()  # 归一化直方图
        L = len(hist)

        # 计算统计矩
        g = np.arange(L)
        mean_gray = np.sum(g * hist)  # 均值
        variance = np.sum(((g - mean_gray) ** 2) * hist)  # 方差
        std_dev = np.sqrt(variance)  # 标准差
        norm_variance = variance / ((L - 1) ** 2)  # 归一化方差
        texture_measure = 1 - 1 / (1 + norm_variance)  # 纹理度量
        third_moment = np.sum(((g - mean_gray) ** 3) * hist) / ((L - 1) ** 2)  # 三阶中心矩

        # 计算能量和熵
        energy = np.sum(hist ** 2)  # 能量
        entropy = -np.sum(hist * np.log2(hist + np.finfo(float).eps))  # 熵，避免 log(0)

        ### 计算 GLCM ###
        distances = [1]  # 计算 1 像素距离的 GLCM
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # 方向：0°, 45°, 90°, 135°
        glcm = graycomatrix(image_array, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)

        # 提取 GLCM 特征
        glcm_features = []
        for prop in GLCM_PROPERTIES:
            values = graycoprops(glcm, prop).flatten()  # 提取不同角度的均值
            glcm_features.extend(values)

        # 组合所有特征
        combined_features = [filename, mean_gray, std_dev, texture_measure, third_moment, energy, entropy] + glcm_features
        data.append(combined_features)

# 生成 DataFrame 并保存
columns = ["Filename", "Mean", "StdDev", "TextureMeasure", "ThirdMoment", "Energy", "Entropy"] + \
          [f"GLCM_{prop}_{angle}" for prop in GLCM_PROPERTIES for angle in ["0", "45", "90", "135"]]

df = pd.DataFrame(data, columns=columns)
df.to_csv(output_csv, index=False, encoding="utf-8")

print(f"特征提取完成，结果已保存至: {output_csv}")
