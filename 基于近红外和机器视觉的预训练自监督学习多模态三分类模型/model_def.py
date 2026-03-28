import torch
import torch.nn as nn
import torch.nn.functional as F

class PyramidCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.multi_scale = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d((1,1)),
                nn.Flatten()
            ) for _ in [64, 32, 16]
        ])
        self.scale_proj = nn.Linear(128*3, 128)

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)

        features = []
        for i, pool in enumerate(self.multi_scale):
            size = [64, 32, 16][i]
            resized = F.interpolate(x3, size=(size, size)) if size != x3.shape[-1] else x3
            features.append(pool(resized))
        multi_feat = torch.cat(features, dim=1)
        return self.scale_proj(multi_feat)

class NIRAttentionExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(64)
        )
        self.se_block = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.Sigmoid()
        )
        self.pre_gru = nn.Linear(128, 64)
        self.gru = nn.GRU(input_size=64, hidden_size=64, batch_first=True, bidirectional=True)
        self.attn_fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.fusion = nn.Linear(192, 192)

    def forward(self, x):
        # x: (B, 128)
        x_cnn = x.unsqueeze(1)            # (B, 1, 128)
        x_cnn = self.conv(x_cnn)          # (B, 64, 64)
        se = self.se_block(x_cnn).unsqueeze(-1)
        x_cnn = x_cnn * se
        x_cnn = x_cnn.mean(dim=-1)        # (B, 64)

        x_gru = self.pre_gru(x)           # (B, 64)
        x_gru = x_gru.unsqueeze(1)        # (B, 1, 64)
        x_gru, _ = self.gru(x_gru)        # (B, 1, 128)

        attn_weights = torch.softmax(self.attn_fc(x_gru), dim=1)  # (B,1,1)
        x_gru = torch.sum(x_gru * attn_weights, dim=1)            # (B,128)

        features = torch.cat([x_cnn, x_gru], dim=1)               # (B,192)
        return self.fusion(features)

class CrossModalFusion(nn.Module):
    def __init__(self, img_dim=128, nir_dim=192, hidden_dim=256):
        super().__init__()
        self.importance = nn.Parameter(torch.ones(2))

        self.img_proj = nn.Sequential(
            nn.Linear(img_dim, hidden_dim),
            nn.ReLU()
        )
        self.nir_proj = nn.Sequential(
            nn.Linear(nir_dim, hidden_dim),
            nn.ReLU()
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 2),
            nn.Softmax(dim=1)
        )

        # ✅ 关键：不要写死 device='cuda:0'
        self.fusion = nn.Linear(hidden_dim*2, hidden_dim)

    def forward(self, img_feat, nir_feat):
        modal_weights = torch.softmax(self.importance, dim=0)
        img_proj = self.img_proj(img_feat * modal_weights[0])
        nir_proj = self.nir_proj(nir_feat * modal_weights[1])

        attn_weights = self.attention(torch.cat([img_proj, nir_proj], dim=1))
        fused = torch.cat([
            img_proj * attn_weights[:, 0:1],
            nir_proj * attn_weights[:, 1:2]
        ], dim=1)
        return self.fusion(fused)

class DistillTriClassModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.image_encoder = PyramidCNN()
        self.nir_encoder = NIRAttentionExtractor()
        self.fusion = CrossModalFusion(img_dim=128, nir_dim=192, hidden_dim=256)

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, img, nir):
        img_feat = self.image_encoder(img)   # (B,128)
        nir_feat = self.nir_encoder(nir)     # (B,192)
        fused = self.fusion(img_feat, nir_feat)  # (B,256)
        return self.classifier(fused)        # (B,3)
