import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import importlib.util
from importlib.machinery import SourceFileLoader
from pathlib import Path

# 导入你的模型和数据集类
model_path = Path(__file__).resolve().with_name("CMDBNet.txt")
loader = SourceFileLoader("CMDBNet", str(model_path))
spec = importlib.util.spec_from_loader(loader.name, loader)
CMDBNet = importlib.util.module_from_spec(spec)
loader.exec_module(CMDBNet)
EnhancedDualResNet18 = CMDBNet.EnhancedDualResNet18
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader
import torch.nn.functional as F


# 重新定义数据集类（与训练时保持一致）
class DualInputDataset:
    def __init__(self, fun, transform=None):
        self.folder1_path = "E:/UAV-aphid data/2024ROI-sub-npy/rgbnpy/"  # rgb
        self.folder2_path = "E:/UAV-aphid data/2024ROI-sub-npy/datasetnpy/"  # 多波段
        self.transform = transform

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

        # 数据预处理
        if input1.shape[0] != 64 or input1.shape[1] != 64:
            input1 = torch.tensor(input1[0]).permute(2, 0, 1)
            input1 = F.interpolate(input1.unsqueeze(0), size=(64, 64), mode='bilinear', align_corners=False)
            input1 = input1.squeeze(0)
        if input2.shape[0] != 64 or input2.shape[1] != 64:
            input2 = torch.tensor(input2)
            input2 = F.interpolate(input2.unsqueeze(0), size=(64, 64), mode='bilinear', align_corners=False)
            input2 = input2.squeeze(0)

        return (input1, input2), (label1, label2), filename




class FeatureExtractor:
    """特征提取器"""

    def __init__(self, model):
        self.model = model
        self.features = {}
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """注册特征提取钩子"""

        def hook_fn(name):
            def hook(module, input, output):
                # 全局平均池化提取特征
                if isinstance(output, (tuple, list)):
                    pooled = []
                    for item in output:
                        if len(item.shape) == 4:
                            pooled.append(F.adaptive_avg_pool2d(item, 1).flatten(1))
                        else:
                            pooled.append(item.flatten(1))
                    feature = torch.cat(pooled, dim=1)
                elif len(output.shape) == 4:  # [B, C, H, W]
                    feature = F.adaptive_avg_pool2d(output, 1).squeeze()  # [B, C]
                else:  # [B, C]
                    feature = output
                self.features[name] = feature.detach().cpu()

            return hook

        # 注册不同层的钩子
        layers_to_extract = {
            'layer1_branch1': self.model.branch1_layer1,
            'layer1_branch2': self.model.branch2_layer1,
            'layer2_branch1': self.model.branch1_layer2,
            'layer2_branch2': self.model.branch2_layer2,
            'layer3_branch1': self.model.branch1_layer3,
            'layer3_branch2': self.model.branch2_layer3,
            'layer4_branch1': self.model.branch1_layer4,
            'layer4_branch2': self.model.branch2_layer4,
            'cmfi_final': self.model.cmfi2,
            'cmff_fusion': self.model.cmff,
            # 'classifier_features': self.model.classifier[1],  # 分类器的Flatten层后a
            'cmff_features': self.model.cmff
        }

        for name, layer in layers_to_extract.items():
            self.hooks.append(layer.register_forward_hook(hook_fn(name)))

    def extract_features(self, input1, input2):
        """提取特征"""
        self.model.eval()
        with torch.no_grad():
            output_cls, output_reg = self.model(input1, input2)

        return self.features.copy(), output_cls, output_reg

    def remove_hooks(self):
        """移除钩子"""
        for hook in self.hooks:
            hook.remove()




def extract_and_save_features(model, data_loader, device, save_path='features_analysis'):
    """提取特征并保存到Excel"""

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    extractor = FeatureExtractor(model)

    all_features = {}
    all_predictions = []
    all_labels = []
    all_filenames = []

    model.eval()

    print("正在提取特征...")
    for (input1, input2), (label_cls, label_reg), filenames in tqdm(data_loader):
        input1, input2 = input1.float().to(device), input2.float().to(device)

        # 提取特征
        features, pred_cls, pred_reg = extractor.extract_features(input1, input2)

        # 收集数据
        batch_size = input1.shape[0]
        for i in range(batch_size):
            # 预测结果
            pred_cls_label = pred_cls[i].argmax().item()
            pred_reg_value = pred_reg[i].item()

            all_predictions.append({
                'filename': filenames[i],
                'true_cls': label_cls[i].item(),
                'true_reg': label_reg[i].item(),
                'pred_cls': pred_cls_label,
                'pred_reg': pred_reg_value
            })

            all_labels.append({
                'filename': filenames[i],
                'level': label_cls[i].item(),
                'rate': label_reg[i].item()
            })

            all_filenames.append(filenames[i])

            # 特征数据
            for layer_name, layer_features in features.items():
                if layer_name not in all_features:
                    all_features[layer_name] = []

                if len(layer_features.shape) == 1:  # 单个样本
                    all_features[layer_name].append(layer_features.numpy())
                else:  # 批次数据
                    all_features[layer_name].append(layer_features[i].numpy())

    # 保存预测结果
    pred_df = pd.DataFrame(all_predictions)
    pred_df.to_excel(os.path.join(save_path, 'predictions_results.xlsx'), index=False)

    # 保存标签数据
    labels_df = pd.DataFrame(all_labels)
    labels_df.to_excel(os.path.join(save_path, 'labels_data.xlsx'), index=False)

    # 保存各层特征
    for layer_name, features_list in all_features.items():
        features_array = np.array(features_list)

        # 创建DataFrame
        feature_columns = [f'{layer_name}_feature_{i}' for i in range(features_array.shape[1])]
        feature_df = pd.DataFrame(features_array, columns=feature_columns)
        feature_df.insert(0, 'filename', all_filenames)
        feature_df.insert(1, 'level', [item['level'] for item in all_labels])
        feature_df.insert(2, 'rate', [item['rate'] for item in all_labels])

        # 保存到Excel
        feature_df.to_excel(os.path.join(save_path, f'{layer_name}_features.xlsx'), index=False)

    extractor.remove_hooks()
    print(f"特征提取完成，结果保存在 {save_path}")

    return all_features, all_predictions, all_labels


def visualize_feature_analysis(features_dict, labels, save_path='features_analysis'):
    """特征分析可视化，并导出T-SNE坐标"""

    # 使用分类器特征进行可视化（通常是最具代表性的）
    if 'cmff_features' in features_dict: #classifier_features,layer2_branch2_features
        features = np.array(features_dict['cmff_features']) #classifier_features,layer2_branch2_features

        # 提取标签和文件名
        cls_labels = [item['level'] for item in labels]
        reg_labels = [item['rate'] for item in labels]
        filenames = [item['filename'] for item in labels]

        # t-SNE可视化
        print("正在进行t-SNE降维...")
        tsne = TSNE(n_components=2, init='random', random_state=30, perplexity=35)
        features_tsne = tsne.fit_transform(features)

        # --- 新增代码：导出T-SNE坐标 ---
        print("正在导出t-SNE坐标...")
        tsne_df = pd.DataFrame(features_tsne, columns=['TSNE_Component_1', 'TSNE_Component_2'])
        tsne_df['filename'] = filenames
        tsne_df['true_level'] = cls_labels
        tsne_df['true_rate'] = reg_labels

        # 调整列顺序，将filename放在前面
        tsne_df = tsne_df[['filename', 'TSNE_Component_1', 'TSNE_Component_2', 'true_level', 'true_rate']]

        tsne_df.to_excel(os.path.join(save_path, 'tsne_coordinates-3-rgb.xlsx'), index=False)
        print(f"t-SNE坐标已保存到 {os.path.join(save_path, 'tsne_coordinates-3-rgb.xlsx')}")
        # ---------------------------------

        # 绘制t-SNE结果
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=cls_labels, cmap='viridis')
        plt.colorbar(scatter)
        plt.title('t-SNE Visualization (Classification Labels)')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')

        plt.subplot(1, 3, 2)
        scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=reg_labels, cmap='plasma')
        plt.colorbar(scatter)
        plt.title('t-SNE Visualization (Regression Labels)')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')

        # PCA可视化
        print("正在进行PCA降维...")
        pca = PCA(n_components=2)
        features_pca = pca.fit_transform(features)

        plt.subplot(1, 3, 3)
        scatter = plt.scatter(features_pca[:, 0], features_pca[:, 1], c=cls_labels, cmap='viridis')
        plt.colorbar(scatter)
        plt.title(f'PCA Visualization\nExplained Variance: {pca.explained_variance_ratio_.sum():.3f}')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'feature_visualization.png'), dpi=600, bbox_inches='tight')
        plt.show()

        # 特征相关性分析
        # plt.figure(figsize=(12, 8))

        # 计算特征与标签的相关性
        # feature_df = pd.DataFrame(features)
        # feature_df['cls_label'] = cls_labels
        # feature_df['reg_label'] = reg_labels

        # 计算相关性矩阵（只取前50个特征以便可视化）
        # n_features_to_show = min(100, features.shape[1])
        # corr_matrix = feature_df.iloc[:, :n_features_to_show + 2].corr()
        #
        # sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
        # plt.title('Feature Correlation Matrix')
        # plt.tight_layout()
        # plt.savefig(os.path.join(save_path, 'feature_correlation.png'), dpi=300, bbox_inches='tight')
        # plt.show()


def main():
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载训练好的模型
    model = EnhancedDualResNet18().to(device)
    # model.load_state_dict(torch.load('C:/Users/ygw98/Desktop/消融实验/最终模型1/onlyresent.pth'))
    model.load_state_dict(torch.load('C:/Users/ygw98/Desktop/最优模型！！！/onlyresent.pth'))
    print("模型加载完成")

    # 创建数据加载器
    test_dataset = DualInputDataset(fun='test')
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    print(f"测试集样本数量: {len(test_dataset)}")


    # 2. 特征提取和保存
    print("\n开始特征提取...")
    features_dict, predictions, labels = extract_and_save_features(
        model, test_loader, device, save_path='features_analysis'
    )

    # 3. 特征分析可视化
    print("\n开始特征分析可视化...")
    visualize_feature_analysis(features_dict, labels, save_path='features_analysis')

    print("\n所有任务完成!")
    print("结果文件:")
    print("- Grad-CAM可视化: gradcam_results/")
    print("- 特征和预测结果: features_analysis/")
    print("  - predictions_results.xlsx: 预测结果")
    print("  - labels_data.xlsx: 真实标签")
    print("  - *_features.xlsx: 各层特征")


if __name__ == '__main__':
    main()
