import torch
import torch.nn as nn
import torchvision.models as models

class VGGFeatureExtractor(nn.Module):
    def __init__(self, layers=(3, 8, 17, 26, 35)):
        super().__init__()
        vgg = models.vgg19(pretrained=True).features
        self.slices = nn.ModuleList()
        prev = 0
        for l in layers:
            self.slices.append(nn.Sequential(*list(vgg.children())[prev:l]))
            prev = l
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = []
        for slice in self.slices:
            x = slice(x)
            features.append(x)
        return features

class CombinedLoss(nn.Module):
    def __init__(self, l1_weight=1.0, perceptual_weight=0.1, device='cpu'):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.perceptual = VGGFeatureExtractor().to(device)  # 移动 VGG 到同一设备
        self.device = device
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight

    def forward(self, output, target):
        output_rgb = output.repeat(1, 3, 1, 1).to(self.device)
        target_rgb = target.repeat(1, 3, 1, 1).to(self.device)

        perceptual_loss = 0.0
        output_feats = self.perceptual(output_rgb)
        target_feats = self.perceptual(target_rgb)
        for of, tf in zip(output_feats, target_feats):
            perceptual_loss += self.l1(of, tf)

        l1_loss = self.l1(output, target)
        return self.l1_weight * l1_loss + self.perceptual_weight * perceptual_loss
