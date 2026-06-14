import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.shared = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.shared(self.max_pool(x)) + self.shared(self.avg_pool(x)))


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3 = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.conv5 = nn.Conv2d(2, 1, kernel_size=5, padding=2, bias=False)
        self.conv7 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.conv1 = nn.Conv2d(3, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map = torch.amax(x, dim=1, keepdim=True)
        spatial = torch.cat([avg_map, max_map], dim=1)
        spatial = torch.cat([self.conv3(spatial), self.conv5(spatial), self.conv7(spatial)], dim=1)
        return self.sigmoid(self.conv1(spatial))


class CMFI(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.rgb_channel = ChannelAttention(channels, reduction)
        self.vis_channel = ChannelAttention(channels, reduction)
        self.rgb_spatial = SpatialAttention()
        self.vis_spatial = SpatialAttention()
        self.channel_scale = nn.Parameter(torch.zeros(1))
        self.spatial_scale = nn.Parameter(torch.zeros(1))
        self.rgb_refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.vis_refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, rgb, vis):
        b, c, h, w = rgb.shape
        c_rgb = self.rgb_channel(rgb)
        c_vis = self.vis_channel(vis)
        rgb_vec = c_rgb.flatten(1)
        vis_vec = c_vis.flatten(1)
        cross_c = torch.bmm(rgb_vec.unsqueeze(2), vis_vec.unsqueeze(1))
        rgb_gate = torch.bmm(F.softmax(cross_c, dim=-1), vis_vec.unsqueeze(2)).view(b, c, 1, 1)
        vis_gate = torch.bmm(F.softmax(cross_c.transpose(1, 2), dim=-1), rgb_vec.unsqueeze(2)).view(b, c, 1, 1)
        rgb_c = rgb * (c_rgb + self.channel_scale * rgb_gate.sigmoid())
        vis_c = vis * (c_vis + self.channel_scale * vis_gate.sigmoid())
        s_rgb = self.rgb_spatial(rgb_c)
        s_vis = self.vis_spatial(vis_c)
        rgb_sp = s_rgb.flatten(2)
        vis_sp = s_vis.flatten(2)
        rgb_sp = F.softmax(rgb_sp, dim=-1).view(b, 1, h, w)
        vis_sp = F.softmax(vis_sp, dim=-1).view(b, 1, h, w)
        rgb_out = rgb + self.rgb_refine(rgb_c * (s_rgb + self.spatial_scale * vis_sp))
        vis_out = vis + self.vis_refine(vis_c * (s_vis + self.spatial_scale * rgb_sp))
        return self.relu(rgb_out), self.relu(vis_out)


class SimAM(nn.Module):
    def __init__(self, e_lambda=1e-4):
        super().__init__()
        self.e_lambda = e_lambda
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h, w = x.shape[-2:]
        n = max(h * w - 1, 1)
        d = (x - x.mean(dim=(2, 3), keepdim=True)).pow(2)
        v = d.sum(dim=(2, 3), keepdim=True) / n
        return x * self.sigmoid(d / (4 * (v + self.e_lambda)) + 0.5)


class CMFFBranch(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.q = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.k = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.v = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        q = self.q(x).flatten(2).transpose(1, 2)
        k = self.k(x).flatten(2)
        v = self.v(x).flatten(2)
        attn = F.softmax(torch.bmm(q, k) / (c ** 0.5), dim=-1)
        specific = torch.bmm(v, attn.transpose(1, 2)).view(b, c, h, w)
        return self.proj(specific + x)


class CMFF(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.rgb_branch = CMFFBranch(channels)
        self.vis_branch = CMFFBranch(channels)
        self.shared = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.rgb_cross = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.vis_cross = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.rgb_simam = SimAM()
        self.vis_simam = SimAM()
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, rgb, vis):
        specific_rgb = self.rgb_branch(rgb)
        specific_vis = self.vis_branch(vis)
        shared = self.shared(torch.cat([specific_rgb, specific_vis], dim=1))
        cross_rgb = self.rgb_cross(torch.cat([specific_rgb, shared], dim=1))
        cross_vis = self.vis_cross(torch.cat([specific_vis, shared], dim=1))
        cross_rgb = self.rgb_simam(cross_rgb + rgb)
        cross_vis = self.vis_simam(cross_vis + vis)
        return self.fusion(torch.cat([cross_rgb, shared, cross_vis], dim=1))


class TaskHead(nn.Module):
    def __init__(self, in_channels, out_features, dropout=0.4):
        super().__init__()
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, out_features)
        )

    def forward(self, x):
        return self.head(x)


class EnhancedDualResNet18(nn.Module):
    def __init__(self, rgb_channels=3, vis_channels=19, num_classes=4):
        super().__init__()
        self.inplanes = 64
        self.branch1_conv1 = nn.Conv2d(rgb_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.branch1_bn1 = nn.BatchNorm2d(64)
        self.branch1_relu = nn.ReLU(inplace=True)
        self.branch1_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.branch2_conv1 = nn.Conv2d(vis_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.branch2_bn1 = nn.BatchNorm2d(64)
        self.branch2_relu = nn.ReLU(inplace=True)
        self.branch2_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.branch1_layer1 = self._make_layer(64, 64, 2)
        self.branch2_layer1 = self._make_layer(64, 64, 2)
        self.branch1_layer2 = self._make_layer(64, 128, 2, stride=2)
        self.branch2_layer2 = self._make_layer(64, 128, 2, stride=2)
        self.cmfi1 = CMFI(128)
        self.branch1_layer3 = self._make_layer(128, 256, 2, stride=2)
        self.branch2_layer3 = self._make_layer(128, 256, 2, stride=2)
        self.branch1_layer4 = self._make_layer(256, 512, 2, stride=2)
        self.branch2_layer4 = self._make_layer(256, 512, 2, stride=2)
        self.shared_layer4 = nn.Identity()
        self.cmfi2 = CMFI(512)
        self.cmff = CMFF(512)
        self.classifier = TaskHead(512, num_classes)
        self.regressor = TaskHead(512, 1)
        self._init_weights()

    def _make_layer(self, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
        layers = [BasicBlock(inplanes, planes, stride, downsample)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(planes, planes))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x1, x2):
        rgb = self.branch1_maxpool(self.branch1_relu(self.branch1_bn1(self.branch1_conv1(x1))))
        vis = self.branch2_maxpool(self.branch2_relu(self.branch2_bn1(self.branch2_conv1(x2))))
        rgb = self.branch1_layer1(rgb)
        vis = self.branch2_layer1(vis)
        rgb = self.branch1_layer2(rgb)
        vis = self.branch2_layer2(vis)
        rgb, vis = self.cmfi1(rgb, vis)
        rgb = self.branch1_layer3(rgb)
        vis = self.branch2_layer3(vis)
        rgb = self.branch1_layer4(rgb)
        vis = self.branch2_layer4(vis)
        rgb, vis = self.cmfi2(rgb, vis)
        fused = self.cmff(rgb, vis)
        class_out = self.classifier(fused)
        reg_out = self.regressor(fused)
        return class_out, reg_out


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EnhancedDualResNet18().to(device)
    model.eval()
    x1 = torch.randn(2, 3, 64, 64, device=device)
    x2 = torch.randn(2, 19, 64, 64, device=device)
    with torch.no_grad():
        class_out, reg_out = model(x1, x2)
    print(class_out.shape)
    print(reg_out.shape)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params / 1e6:.2f}M')
