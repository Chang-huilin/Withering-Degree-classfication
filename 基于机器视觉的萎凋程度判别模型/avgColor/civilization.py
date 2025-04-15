import os
import cv2
import numpy as np
import cupy as cp
from skimage import color, feature, util
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.io

# ---------------------- 配置GPU设备 ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------- 加载MATLAB模型并转换为PyTorch模型 ----------------------
def load_matlab_model(mat_path):
    # 加载MATLAB模型参数
    mat_data = scipy.io.loadmat(mat_path)
    weights = torch.tensor(mat_data['weights'], dtype=torch.float32)
    bias = torch.tensor(mat_data['bias'], dtype=torch.float32)

    # 定义PyTorch模型
    class SimpleNet(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(SimpleNet, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, output_size)
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # 创建模型并加载权重
    input_size = 39  # 输入特征维度
    hidden_size = 64  # 隐藏层大小（根据MATLAB模型调整）
    output_size = 1   # 输出维度（水分预测值）
    net = SimpleNet(input_size, hidden_size, output_size)

    # 加载权重
    net.fc1.weight.data = weights
    net.fc1.bias.data = bias

    # 将模型转移到GPU
    net = net.to(device)
    net.eval()
    return net

# 加载模型
file_path = "D:\\红茶数据2024.0423\\工业相机\\自然\\DATA\\net.mat"
net = load_matlab_model(file_path)

# ---------------------- 辅助函数：逐像素提取39维特征 ----------------------
def extract39Features_pixel(rgb_image, gray_image, r, c, win_size=3):
    """GPU加速的特征提取函数"""
    half_win = win_size // 2
    
    # 1. 颜色空间特征 (9维)
    rgb_pixel = cp.asarray(rgb_image[r, c])
    R, G, B = rgb_pixel
    
    # LAB转换
    lab_pixel = cp.asarray(color.rgb2lab(rgb_image[r:r+1, c:c+1]), dtype=cp.float32).reshape(3)
    L, a, b = lab_pixel
    
    # HSI近似 (使用HSV)
    hsv_pixel = cp.asarray(color.rgb2hsv(rgb_image[r:r+1, c:c+1]), dtype=cp.float32).reshape(3)
    H, S, I = hsv_pixel
    
    # 2. 灰度统计与纹理特征 (30维)
    window = cp.array(gray_image[r-half_win:r+half_win+1, 
                                 c-half_win:c+half_win+1])
    
    # 2.1 直方图统计矩 (6维)
    counts = cp.histogram(window, bins=256, range=(0, 1))[0].astype(cp.float32)
    counts /= cp.sum(counts) + 1e-8
    
    g_vals = cp.arange(256, dtype=cp.float32) / 255.0
    mean_val = cp.sum(g_vals * counts)
    variance = cp.sum((g_vals - mean_val)**2 * counts)
    
    features = [
        mean_val.get(),                            # Mean
        cp.sqrt(variance).get(),                   # StdDev
        (1 - 1/(1 + variance/(255**2))).get(),    # TextureMeasure
        cp.sum((g_vals - mean_val)**3 * counts).get() / (255**2),  # ThirdMoment
        cp.sum(counts**2).get(),                   # Energy
        -cp.sum(counts * cp.log2(counts + 1e-8)).get()  # Entropy
    ]
    
    # 2.2 GLCM特征 (24维)
    offsets = [(0, 1), (-1, 1), (-1, 0), (-1, -1)]
    window_cpu = cp.asnumpy(window * 255).astype(np.uint8)
    for offset in offsets:
        glcm = feature.graycomatrix(window_cpu, 
                                   distances=[1], 
                                   angles=[np.arctan2(offset[0], offset[1])], 
                                   levels=256,
                                   symmetric=True)
        
        # 计算统计量
        contrast = feature.graycoprops(glcm, 'contrast')[0,0]
        correlation = feature.graycoprops(glcm, 'correlation')[0,0]
        energy = feature.graycoprops(glcm, 'energy')[0,0]
        homogeneity = feature.graycoprops(glcm, 'homogeneity')[0,0]
        
        # 自定义计算
        P = glcm / np.sum(glcm)
        i, j = np.indices(glcm.shape)
        dissimilarity = np.sum(np.abs(i - j) * P)
        ASM = np.sum(P**2)
        
        features.extend([contrast, correlation, energy, homogeneity, dissimilarity, ASM])
    
    # 合并所有特征
    color_features = cp.asarray([R, G, B, L, a, b, H, S, I])
    texture_features = cp.asarray(features)
    return cp.concatenate([color_features, texture_features]).get()

# ---------------------- 批量处理主函数 ----------------------
def process_images(input_folder, output_folder, max_images=6):
    os.makedirs(output_folder, exist_ok=True)
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.bmp')]
    
    for idx, filename in enumerate(tqdm(image_files[:max_images], desc="Processing Images")):
        img_path = os.path.join(input_folder, filename)
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        img_float = img.astype(np.float32) / 255.0
        
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        moisture_map = cp.zeros(img.shape[:2], dtype=cp.float32)
        
        rows, cols = img.shape[:2]
        for r in tqdm(range(rows), desc=f"Processing {filename}", leave=False):
            for c in range(cols):
                features = extract39Features_pixel(img_float, gray, r, c)
                features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    pred = net(features_tensor).item()
                moisture_map[r, c] = pred
        
        moisture_map = cp.clip(moisture_map, 0, 100)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(img)
        plt.imshow(cp.asnumpy(moisture_map), alpha=0.5, cmap='jet')
        plt.colorbar()
        plt.title(f'Moisture Map - {filename}')
        
        output_path = os.path.join(output_folder, f'heatmap_{filename}')
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()

# ---------------------- 执行处理 ----------------------
if __name__ == "__main__":
    input_folder = 'D:/红茶数据2024.0423/工业相机/自然/可视化'
    output_folder = 'D:/红茶数据2024.0423/工业相机/自然/可视化/Processed'
    process_images(input_folder, output_folder)