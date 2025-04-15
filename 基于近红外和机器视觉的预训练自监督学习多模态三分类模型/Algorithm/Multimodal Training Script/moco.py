import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import copy
import os

# ==== Enhanced Dataset with contrastive views ====
class EnhancedContrastiveDataset(Dataset):
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img_path = self.base_dataset.image_paths[idx]
        nir = self.base_dataset.nir_data[idx]
        label = self.base_dataset.labels[idx]
        img = Image.open(img_path).convert("RGB")
        x1 = self.transform(img)
        x2 = self.transform(img)
        return x1, x2, torch.tensor(nir, dtype=torch.float32), label


# ==== MoCo Contrastive Learning Module ====
class MoCoContrastive(nn.Module):
    def __init__(self, encoder, feature_dim=128, queue_size=1024, momentum=0.999, nir_dim=10):
        super().__init__()
        self.encoder_q = encoder
        self.encoder_k = copy.deepcopy(encoder)
        for param in self.encoder_k.parameters():
            param.requires_grad = False

        self.queue_size = queue_size
        self.momentum = momentum

        self.register_buffer("queue_feats", torch.randn(queue_size, feature_dim))
        self.register_buffer("queue_labels", torch.zeros(queue_size, dtype=torch.long))
        self.register_buffer("queue_nir", torch.zeros(queue_size, nir_dim))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def momentum_update(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)

    @torch.no_grad()
    def dequeue_and_enqueue(self, keys, labels, nirs):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0

        self.queue_feats[ptr:ptr+batch_size] = keys
        self.queue_labels[ptr:ptr+batch_size] = labels
        self.queue_nir[ptr:ptr+batch_size] = nirs
        self.queue_ptr[0] = (ptr + batch_size) % self.queue_size

    def forward(self, x_q, x_k, nir, labels):
        q = self.encoder_q(x_q)
        with torch.no_grad():
            self.momentum_update()
            k = self.encoder_k(x_k)

        logits = torch.mm(q, self.queue_feats.T)  # [B, K]
        labels_contrastive = (labels.view(-1, 1) == self.queue_labels.view(1, -1)).float()
        logits /= 0.07  # temperature

        self.dequeue_and_enqueue(k.detach(), labels, nir)
        return logits, labels_contrastive


# ==== Supervised Contrastive Loss ====
def supervised_contrastive_loss(logits, labels_contrastive):
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()  # 为了数值稳定

    exp_logits = torch.exp(logits)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True))

    mean_log_prob_pos = (labels_contrastive * log_prob).sum(1) / labels_contrastive.sum(1)
    loss = -mean_log_prob_pos.mean()
    return loss


# ==== Dummy Encoder for Testing ====
class SimpleEncoder(nn.Module):
    def __init__(self, feature_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 224 * 224, 512),
            nn.ReLU(),
            nn.Linear(512, feature_dim)
        )

    def forward(self, x):
        return self.net(x)


# ==== Training Loop ====
def train_moco(model, loader, optimizer, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x1, x2, nir, labels in loader:
            x1, x2, nir, labels = x1.to(device), x2.to(device), nir.to(device), labels.to(device)
            logits, labels_contrastive = model(x1, x2, nir, labels)
            loss = supervised_contrastive_loss(logits, labels_contrastive)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")


# ==== Example Usage ====
if __name__ == '__main__':
    # 假设 base_dataset 已定义并包含 image_paths, nir_data, labels
    # transform 和 encoder 初始化
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 创建 contrastive 数据集
    dataset = EnhancedContrastiveDataset(base_dataset, transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 模型
    encoder = SimpleEncoder()
    model = MoCoContrastive(encoder, feature_dim=128, nir_dim=dataset[0][2].shape[0]).cuda()

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 训练
    train_moco(model, dataloader, optimizer, device=torch.device('cuda'), epochs=10)
