import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA

# 加载数据
file_path = r"D:\红茶数据2024.0423\工业相机\自然\DATA\pf.mat"
data = sio.loadmat(file_path)
X = data['X']  # 形状为 (140, 39)
Y = data['Y']  # 形状为 (140, 1)

# 数据扩增参数设置
augmentation_factor = 5  # 最终数据量为原始的 5 倍
noise_std = 0.01         # 高斯噪声的标准差（可根据需要调整）

# 使用 PCA 对 X 进行降维处理，保留 95% 的方差
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X)
print("PCA 后的特征维度：", X_pca.shape)

# 定义在 PCA 空间中添加噪声并逆变换得到新样本的函数
def augment_sample(x_pca, noise_std):
    noise = np.random.normal(0, noise_std, size=x_pca.shape)
    x_pca_aug = x_pca + noise
    # 逆变换回原始特征空间
    x_aug = pca.inverse_transform(x_pca_aug)
    return x_aug

# 对原始数据进行扩增
X_aug_list = [X]  # 包含原始数据
Y_aug_list = [Y]

# 每次扩增生成一个新的数据集
for _ in range(augmentation_factor - 1):
    X_aug = []
    # 对每个样本添加噪声扩增
    for i in range(X.shape[0]):
        x_sample_pca = pca.transform(X[i].reshape(1, -1))[0]
        x_aug = augment_sample(x_sample_pca, noise_std)
        X_aug.append(x_aug)
    X_aug_list.append(np.array(X_aug))
    Y_aug_list.append(Y)

# 将各次扩增数据按行拼接
X_final = np.concatenate(X_aug_list, axis=0)
Y_final = np.concatenate(Y_aug_list, axis=0)

print("扩增后 X 的形状：", X_final.shape)  # 应为 (700, 39)
print("扩增后 Y 的形状：", Y_final.shape)  # 应为 (700, 1)

# 保存扩增后的数据到 .mat 文件
save_path = r"D:\红茶数据2024.0423\工业相机\自然\DATA\PF_augmented.mat"
sio.savemat(save_path, {'X_aug': X_final, 'Y_aug': Y_final})
print("保存成功，数据保存在：", save_path)
