import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d
from dist.cbam import CBAM
from dist.resnet18 import ResNet18

class DFF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.deform = DeformConv2d(in_channel * 3, out_channel, 3, 1, 1)
        self.offset = nn.Conv2d(in_channel, 2 * 3 * 3, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, ref_feat, dist_feat):
        offset = self.offset(ref_feat)
        diff = (ref_feat - dist_feat) ** 2
        feat = self.deform(torch.cat((ref_feat, dist_feat, diff), dim=1), offset)
        feat = self.relu(feat)
        return feat, diff
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ResNet18()

        channel_list = [64, 128, 256, 512]
        self.DFFs = nn.ModuleList([DFF(c, c) for c in channel_list])  # 变形卷积
        self.attens = nn.ModuleList([CBAM(c) for c in channel_list])  # 注意力
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        feat_dim = sum(channel_list)  # 960
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2),
        )

    def normalize_tensor(self, x, eps=1e-10):
        norm = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True) + eps)
        return x / norm

    def _process_stage(self, ref_feat, dist_feat, dff, atten):
        ref_feat = self.normalize_tensor(ref_feat)
        dist_feat = self.normalize_tensor(dist_feat)

        feat, diff = dff(ref_feat, dist_feat)
        att_feat = atten(feat)
        pooled_feat = torch.flatten(self.avg_pool(att_feat), start_dim=1)
        return pooled_feat

    def forward_spatial(self, ref, dis):
        ref_feats = self.backbone(ref)
        dist_feats = self.backbone(dis)

        pooled_feats = []
        for i in range(4):
            feat = self._process_stage(ref_feats[i], dist_feats[i], self.DFFs[i], self.attens[i])
            pooled_feats.append(feat)

        output = self.fc(torch.cat(pooled_feats, dim=1))
        learned_score, std = output[:, 0], output[:, 1]
        learned_score = learned_score.unsqueeze(1)
        return learned_score

    def forward(self, ref, dis):
        score = self.forward_spatial(ref, dis)
        return score





