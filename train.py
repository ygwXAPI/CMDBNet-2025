import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, cohen_kappa_score, confusion_matrix,
                             mean_squared_error, r2_score)
import seaborn as sns
import os
from tqdm import tqdm
import time
import importlib.util
from importlib.machinery import SourceFileLoader
from pathlib import Path

# from onlyresnet1 import EnhancedDualResNet18
###从这里开始
#完整模型
# from onlyresnet1 import CMDBNet

EnhancedDualResNet18 = CMDBNet.EnhancedDualResNet18
#MCAM特征融合
# from onlysimam1 import ImprovedDualResNet18
# from onlybase import EnhancedDualResNet18
from torch import nn
import torch
from torch.nn import functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """标签平滑的交叉熵损失"""

    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, x, target):
        log_probs = F.log_softmax(x, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = (1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean', smoothing=0.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.smoothing = smoothing

        if isinstance(alpha, (float, int)):
            self.alpha = torch.ones(1) * alpha
        elif isinstance(alpha, list):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)

    def forward(self, inputs, targets):
        # 应用标签平滑
        if self.smoothing > 0:
            confidence = 1.0 - self.smoothing
            log_probs = F.log_softmax(inputs, dim=-1)
            nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(1))
            nll_loss = nll_loss.squeeze(1)
            smooth_loss = -log_probs.mean(dim=-1)
            ce_loss = confidence * nll_loss + self.smoothing * smooth_loss
        else:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        pt = torch.exp(-ce_loss)
        if self.alpha is not None:
            if self.alpha.type() != inputs.data.type():
                self.alpha = self.alpha.type_as(inputs.data)
            at = self.alpha.gather(0, targets.data.view(-1))
            focal_loss = at * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class EMA:
    """指数移动平均模型"""

    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


class DualInputDataset:
    def __init__(self, fun, transform=None):
        """
        双输入数据集类

        参数:
            df: 包含文件名和标签的Data  Frame
            folder1_path: 第一个输入文件夹路径
            folder2_path: 第二个输入文件夹路径
            transform: 可选的数据转换
        """

        self.folder1_path = "E:/UAV-aphid data/2024ROI-sub-npy/rgbnpy/"  # rgb
        self.folder2_path = "E:/UAV-aphid data/2024ROI-sub-npy/datasetnpy/"  # 多波段
        self.transform = transform
        self.data, labels = [], []
        if fun == "train":
            file = os.path.join("train_set.xlsx")
        elif fun == "test":
            file = os.path.join("test_set.xlsx")
        elif fun == "val":
            file = os.path.join("val_set.xlsx")
        self.df = pd.read_excel(file)

        # 确保所有文件都存在
        self.valid_indices = []
        for idx, row in self.df.iterrows():
            filename = row['filename']
            path1 = os.path.join(self.folder1_path, f"{filename}.npy")
            path2 = os.path.join(self.folder2_path, f"{filename}.npy")
            if os.path.exists(path1) and os.path.exists(path2):
                self.valid_indices.append(idx)

        print(f"Total valid samples: {len(self.valid_indices)}/{len(self.df)}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        row = self.df.iloc[actual_idx]
        filename = row['filename']

        # 加载两个输入
        input1 = np.load(os.path.join(self.folder1_path, f"{filename}.npy"))

        input2 = np.load(os.path.join(self.folder2_path, f"{filename}.npy"))

        # 获取两个输出标签
        label1 = row['level']  # 第一个输出标签
        label2 = row['rate']  # 第二个输出标签
        label1 = torch.tensor(label1).long()
        label2 = torch.tensor(label2, dtype=torch.float32)

        # 应用转换(如果有)
        if input1.shape[0] != 64 or input1.shape[1] != 64:
            # 使用双线性插值调整大小
            input1 = torch.tensor(input1[0]).permute(2, 0, 1)  # 先转为CxHxW格式
            input1 = F.interpolate(input1.unsqueeze(0), size=(64, 64), mode='bilinear', align_corners=False)
            input1 = input1.squeeze(0)  # 去掉batch维度
        if input2.shape[0] != 64 or input2.shape[1] != 64:
            # 使用双线性插值调整大小
            input2 = torch.tensor(input2)  # 先转为CxHxW格式
            input2 = F.interpolate(input2.unsqueeze(0), size=(64, 64), mode='bilinear', align_corners=False)
            input2 = input2.squeeze(0)  # 去掉batch维度

        # 转换为张量(在训练时使用torch.from_numpy)
        # 这里我们返回numpy数组，可以在DataLoader中转换
        return (input1, input2), (label1, label2)


trainset = DualInputDataset(fun='train')
# testset = MyDataset("F:/Data/Data500×500_Mat/test")
# valset = MyDataset("F:/Data/Data500×500_Mat/val")
testset = DualInputDataset(fun='test')
valset = DualInputDataset(fun='val')
torch.manual_seed(37)
np.random.seed(37)
batch_size = 32
num_epochs = 200
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(valset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, drop_last=True)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model = EnhancedDualResNet18().to(device)
# model = ImprovedDualResNet18().to(device)


# 定义损失函数和优化器 - 使用标签平滑
# criterion_cls = FocalLoss(alpha=[1.0, 1.5, 2.0, 2.0], gamma=5, smoothing=0.5)  # 添加标签平滑
criterion_cls = FocalLoss(alpha=[1.0, 1.3, 1.8, 2.0], gamma=5, smoothing=0.4)  # 添加标签平滑最优
# criterion_cls = LabelSmoothingCrossEntropy(smoothing=0.1)  # 替代方案
criterion_reg = nn.MSELoss()
# criterion_reg = nn.HuberLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)  # 使用AdamW


# 学习率调度器
def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return optim.lr_scheduler.LambdaLR(optimizer, f)


# 创建EMA模型
ema = EMA(model, decay=0.999)
ema.register()

# 学习率预热
warmup_iters = 5
warmup_factor = 0.1
warmup_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)


# 训练和验证函数
def train_epoch(model, loader, optimizer, criterion_cls, criterion_reg, ema=None):
    model.train()
    running_cls_loss = 0.0
    running_reg_loss = 0.0
    correct = 0
    total = 0

    all_reg_preds = []
    all_reg_targets = []

    for (inputs1, inputs2), (targets_cls, targets_reg) in tqdm(loader, desc="Training"):
        inputs1 = inputs1.float().to(device)
        inputs2 = inputs2.float().to(device)
        targets_cls = targets_cls.to(device)
        targets_reg = targets_reg.float().to(device)

        optimizer.zero_grad()

        # 前向传播
        outputs_cls, outputs_reg = model(inputs1, inputs2)

        # 计算损失
        loss_cls = criterion_cls(outputs_cls, targets_cls)
        loss_reg = criterion_reg(outputs_reg.squeeze(), targets_reg)
        total_loss = loss_cls + loss_reg

        # 反向传播和优化
        total_loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # 更新EMA模型
        if ema is not None:
            ema.update()

        # 统计信息
        running_cls_loss += loss_cls.item()
        running_reg_loss += loss_reg.item()

        _, predicted = torch.max(outputs_cls.data, 1)
        total += targets_cls.size(0)
        correct += (predicted == targets_cls).sum().item()

        all_reg_preds.extend(outputs_reg.squeeze().detach().cpu().numpy())
        all_reg_targets.extend(targets_reg.cpu().numpy())

    # 计算指标
    train_cls_loss = running_cls_loss / len(loader)
    train_reg_loss = running_reg_loss / len(loader)
    train_acc = 100 * correct / total

    # 回归指标
    rmse = np.sqrt(mean_squared_error(all_reg_targets, all_reg_preds))
    r2 = r2_score(all_reg_targets, all_reg_preds)
    rpd = np.std(all_reg_targets) / rmse if rmse != 0 else 0

    return train_cls_loss, train_reg_loss, train_acc, rmse, r2, rpd


def validate_epoch(model, loader, criterion_cls, criterion_reg):
    model.eval()
    running_cls_loss = 0.0
    running_reg_loss = 0.0
    correct = 0
    total = 0

    all_cls_preds = []
    all_cls_targets = []
    all_reg_preds = []
    all_reg_targets = []

    with torch.no_grad():
        for (inputs1, inputs2), (targets_cls, targets_reg) in tqdm(loader, desc="Validating"):
            inputs1 = inputs1.float().to(device)
            inputs2 = inputs2.float().to(device)
            targets_cls = targets_cls.to(device)
            targets_reg = targets_reg.float().to(device)

            # 前向传播
            outputs_cls, outputs_reg = model(inputs1, inputs2)

            # 计算损失
            loss_cls = criterion_cls(outputs_cls, targets_cls)
            loss_reg = criterion_reg(outputs_reg.squeeze(), targets_reg)

            # 统计信息
            running_cls_loss += loss_cls.item()
            running_reg_loss += loss_reg.item()

            _, predicted = torch.max(outputs_cls.data, 1)
            total += targets_cls.size(0)
            correct += (predicted == targets_cls).sum().item()

            all_cls_preds.extend(predicted.cpu().numpy())
            all_cls_targets.extend(targets_cls.cpu().numpy())
            all_reg_preds.extend(outputs_reg.squeeze().cpu().numpy())
            all_reg_targets.extend(targets_reg.cpu().numpy())

    # 计算指标
    val_cls_loss = running_cls_loss / len(loader)
    val_reg_loss = running_reg_loss / len(loader)
    val_acc = 100 * correct / total

    # 分类指标
    precision = precision_score(all_cls_targets, all_cls_preds, average='weighted')
    recall = recall_score(all_cls_targets, all_cls_preds, average='weighted')
    f1 = f1_score(all_cls_targets, all_cls_preds, average='weighted')
    kappa = cohen_kappa_score(all_cls_targets, all_cls_preds)

    # 回归指标
    rmse = np.sqrt(mean_squared_error(all_reg_targets, all_reg_preds))
    r2 = r2_score(all_reg_targets, all_reg_preds)
    rpd = np.std(all_reg_targets) / rmse if rmse != 0 else 0

    return (val_cls_loss, val_reg_loss, val_acc, precision, recall, f1, kappa,
            rmse, r2, rpd, all_cls_preds, all_cls_targets, all_reg_preds, all_reg_targets)


# 训练循环

best_val_loss = float('inf')
history = {
    'train_cls_loss': [], 'train_reg_loss': [], 'train_acc': [], 'train_rmse': [], 'train_r2': [], 'train_rpd': [],
    'val_cls_loss': [], 'val_reg_loss': [], 'val_acc': [], 'val_precision': [], 'val_recall': [], 'val_f1': [],
    'val_kappa': [], 'val_rmse': [], 'val_r2': [], 'val_rpd': []
}

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    start_time = time.time()

    # 学习率预热
    if epoch < warmup_iters:
        warmup_scheduler.step()

    # 训练
    train_cls_loss, train_reg_loss, train_acc, train_rmse, train_r2, train_rpd = train_epoch(
        model, train_loader, optimizer, criterion_cls, criterion_reg, ema)

    # 验证 - 使用EMA模型
    ema.apply_shadow()
    (val_cls_loss, val_reg_loss, val_acc, val_precision, val_recall, val_f1, val_kappa,
     val_rmse, val_r2, val_rpd, _, _, _, _) = validate_epoch(model, val_loader, criterion_cls, criterion_reg)
    ema.restore()

    # 更新学习率
    scheduler.step(val_cls_loss + val_reg_loss)

    # 保存历史
    history['train_cls_loss'].append(train_cls_loss)
    history['train_reg_loss'].append(train_reg_loss)
    history['train_acc'].append(train_acc)
    history['train_rmse'].append(train_rmse)
    history['train_r2'].append(train_r2)
    history['train_rpd'].append(train_rpd)

    history['val_cls_loss'].append(val_cls_loss)
    history['val_reg_loss'].append(val_reg_loss)
    history['val_acc'].append(val_acc)
    history['val_precision'].append(val_precision)
    history['val_recall'].append(val_recall)
    history['val_f1'].append(val_f1)
    history['val_kappa'].append(val_kappa)
    history['val_rmse'].append(val_rmse)
    history['val_r2'].append(val_r2)
    history['val_rpd'].append(val_rpd)

    # 打印结果
    print(f"Train - Cls Loss: {train_cls_loss:.4f}, Reg Loss: {train_reg_loss:.4f}, Acc: {train_acc:.2f}%")
    print(f"Train - RMSE: {train_rmse:.4f}, R2: {train_r2:.4f}, RPD: {train_rpd:.4f}")
    print(f"Val - Cls Loss: {val_cls_loss:.4f}, Reg Loss: {val_reg_loss:.4f}, Acc: {val_acc:.2f}%")
    print(f"Val - Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}, Kappa: {val_kappa:.4f}")
    print(f"Val - RMSE: {val_rmse:.4f}, R2: {val_r2:.4f}, RPD: {val_rpd:.4f}")
    print(f"Time: {time.time() - start_time:.2f}s")

    # 保存最佳模型
    if val_cls_loss + val_reg_loss < best_val_loss:
        best_val_loss = val_cls_loss + val_reg_loss
        torch.save(model.state_dict(), 'onlyresent.pth')
        print("Saved best model!")


# 测试函数
def test_model(model, loader, criterion_cls, criterion_reg):
    model.eval()
    running_cls_loss = 0.0
    running_reg_loss = 0.0
    correct = 0
    total = 0

    all_cls_preds = []
    all_cls_targets = []
    all_reg_preds = []
    all_reg_targets = []

    with torch.no_grad():
        for (inputs1, inputs2), (targets_cls, targets_reg) in tqdm(loader, desc="Testing"):
            inputs1 = inputs1.float().to(device)
            inputs2 = inputs2.float().to(device)
            targets_cls = targets_cls.to(device)
            targets_reg = targets_reg.float().to(device)

            # 前向传播
            outputs_cls, outputs_reg = model(inputs1, inputs2)

            # 计算损失
            loss_cls = criterion_cls(outputs_cls, targets_cls)
            loss_reg = criterion_reg(outputs_reg.squeeze(), targets_reg)

            # 统计信息
            running_cls_loss += loss_cls.item()
            running_reg_loss += loss_reg.item()

            _, predicted = torch.max(outputs_cls.data, 1)
            total += targets_cls.size(0)
            correct += (predicted == targets_cls).sum().item()

            all_cls_preds.extend(predicted.cpu().numpy())
            all_cls_targets.extend(targets_cls.cpu().numpy())
            all_reg_preds.extend(outputs_reg.squeeze().cpu().numpy())
            all_reg_targets.extend(targets_reg.cpu().numpy())

    # 计算指标
    test_cls_loss = running_cls_loss / len(loader)
    test_reg_loss = running_reg_loss / len(loader)
    test_acc = 100 * correct / total

    # 分类指标
    precision = precision_score(all_cls_targets, all_cls_preds, average='weighted')
    recall = recall_score(all_cls_targets, all_cls_preds, average='weighted')
    f1 = f1_score(all_cls_targets, all_cls_preds, average='weighted')
    kappa = cohen_kappa_score(all_cls_targets, all_cls_preds)

    # 回归指标
    rmse = np.sqrt(mean_squared_error(all_reg_targets, all_reg_preds))
    r2 = r2_score(all_reg_targets, all_reg_preds)
    rpd = np.std(all_reg_targets) / rmse if rmse != 0 else 0

    return (test_cls_loss, test_reg_loss, test_acc, precision, recall, f1, kappa,
            rmse, r2, rpd, all_cls_preds, all_cls_targets, all_reg_preds, all_reg_targets)


# 加载最佳模型并测试
model.load_state_dict(torch.load('onlyresent.pth'))
(test_cls_loss, test_reg_loss, test_acc, test_precision, test_recall, test_f1, test_kappa,
 test_rmse, test_r2, test_rpd, test_cls_preds, test_cls_targets, test_reg_preds, test_reg_targets) = test_model(
    model, test_loader, criterion_cls, criterion_reg)

# 打印测试结果
print("\nTest Results:")
print(f"Classification - Loss: {test_cls_loss:.4f}, Acc: {test_acc:.2f}%")
print(f"Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}, Kappa: {test_kappa:.4f}")
print(f"Regression - Loss: {test_reg_loss:.4f}")
print(f"RMSE: {test_rmse:.4f}, R2: {test_r2:.4f}, RPD: {test_rpd:.4f}")


# 绘制混淆矩阵
def plot_confusion_matrix(y_true, y_pred, classes, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()


# 绘制回归图
def plot_regression(y_true, y_pred, title='Regression Plot'):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(title)
    plt.savefig('regression_plot.png', dpi=300, bbox_inches='tight')
    plt.show()


# 假设有4个类别
class_names = ['Class 0', 'Class 1', 'Class 2', 'Class 3']
plot_confusion_matrix(test_cls_targets, test_cls_preds, class_names, 'Test Set Confusion Matrix')
plot_regression(test_reg_targets, test_reg_preds, 'Test Set Regression Plot')


# 绘制训练曲线
def plot_training_history(history):
    plt.figure(figsize=(15, 10))

    # 分类损失和准确率
    plt.subplot(2, 3, 1)
    plt.plot(history['train_cls_loss'], label='Train')
    plt.plot(history['val_cls_loss'], label='Validation')
    plt.title('Classification Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 3, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    # 回归损失和指标
    plt.subplot(2, 3, 3)
    plt.plot(history['train_reg_loss'], label='Train')
    plt.plot(history['val_reg_loss'], label='Validation')
    plt.title('Regression Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 3, 4)
    plt.plot(history['train_rmse'], label='Train')
    plt.plot(history['val_rmse'], label='Validation')
    plt.title('RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend()

    plt.subplot(2, 3, 5)
    plt.plot(history['train_r2'], label='Train')
    plt.plot(history['val_r2'], label='Validation')
    plt.title('R2 Score')
    plt.xlabel('Epoch')
    plt.ylabel('R2')
    plt.legend()

    plt.subplot(2, 3, 6)
    plt.plot(history['train_rpd'], label='Train')
    plt.plot(history['val_rpd'], label='Validation')
    plt.title('RPD')
    plt.xlabel('Epoch')
    plt.ylabel('RPD')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()


plot_training_history(history)

# 保存完整模型
torch.save(model, 'full_model.pth')
