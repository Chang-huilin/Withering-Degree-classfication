# Withering-Degree-classfication

Online monitoring system of black tea withering process based on image

---

## 算法与模型

### 1. 自定义数据集加载

- **EnhancedDataset**：  
  根据数据集根目录下的子文件夹自动读取图像及其类别标签。  
  支持图像预处理（例如：Resize、RandomRotation、GaussianBlur、RandomHorizontalFlip、归一化等）。

### 2. 数据增强与划分

- **数据预处理**：  
  使用 `torchvision.transforms` 对图像进行 Resize（224×224）、随机旋转、随机水平翻转、GaussianBlur 等数据增强操作，提高模型的泛化能力。

- **数据集划分**：  
  数据集按照 6:2:2 的比例划分为训练集（60%）、验证集（20%）和测试集（20%）。

### 3. 模型结构

- **DeepCNN 模型**：  
  一个由 5 个卷积块组成的深度卷积神经网络。  
  每个卷积块包含多个卷积层、BatchNorm、ReLU 激活和 MaxPooling，最后通过全局平均池化和全连接层进行分类。

- **CBAM 分类模型**：  
  在基础卷积网络中嵌入了 CBAM 模块（通道注意力和空间注意力机制），增强了模型对重要特征的关注能力。

- **预训练 VGG16/19 模型**：  
  利用在 ImageNet 上预训练的 VGG16/19 模型，并替换最后的全连接层，使其适用于当前的分类任务。  
  迁移学习策略可以显著提高小数据集上的分类效果。

- **预训练 ResNet50 模型**：  
  利用在 ImageNet 上预训练的 ResNet50 模型，并替换最后的全连接层，使其适用于当前的分类任务。  
  迁移学习策略可以显著提高小数据集上的分类效果。

---

## 训练与评估

### 训练流程

1. **模型训练**：  
   使用 Adam 优化器和交叉熵损失函数进行训练，每个 epoch 内在训练集上进行参数更新。

2. **验证过程**：  
   每个 epoch 结束后在验证集上评估模型性能（损失和准确率），并记录指标以绘制训练/验证曲线。

3. **模型保存**：  
   根据验证集准确率，保存表现最佳的模型（保存为 `.pth` 文件，可用于后续加载和测试）。

### 评估指标与可视化

- **准确率 (Accuracy)**：  
  测试阶段计算总体分类准确率。

- **训练/验证曲线**：  
  记录每个 epoch 的训练和验证损失、准确率，并利用 matplotlib 绘制曲线，直观观察模型的收敛情况和是否存在过拟合现象。
  
  ![alt text](image.png)

- **混淆矩阵 (Confusion Matrix)**：  
  在测试阶段收集所有预测结果和真实标签，利用 scikit-learn 计算混淆矩阵并可视化，帮助分析模型在各个类别上的分类效果。

  ![Confusion Matrix](Confusion%20Matrix.png)

- **Grad_CAM**：  
  利用 Grad-CAM 技术生成热力图，直观展示模型在图像分类任务中关注的区域。通过对卷积层输出的梯度进行反向传播，计算每个像素的重要性，并将其叠加在原始图像上，帮助理解模型的决策过程。

  ![alt text](output93.86.png)

---

## 如何运行

1. **安装依赖**

   请确保安装了以下主要依赖：

   - Python 3.x
   - PyTorch
   - torchvision
   - scikit-learn
   - matplotlib
   - Pillow

   安装示例：

```bash
   pip install torch torchvision scikit-learn matplotlib pillow
