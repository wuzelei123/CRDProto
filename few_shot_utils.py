"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from collections import OrderedDict
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from torch import Tensor
from torchvision.transforms.functional import to_pil_image

from data_utils import preprocess_data_labels, get_support_noise_query_indices, get_augmentation_transforms
from noise_utils import add_noise
import clip
import torch
import json
from torchvision import transforms
from nltk.corpus import wordnet as wn
from collections import OrderedDict
# import torch
# import clip
from PIL import Image
import numpy as np
# import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
def visualize_features(features: torch.Tensor, labels: torch.Tensor, method='tsne'):
    """
    可视化特征分布，展示各类别特征的一致性或区分度

    参数：
        features: Tensor，形状为 (N, D)
        labels: Tensor，形状为 (N,)
        method: 'tsne' 或 'pca'，降维方法
    """
    features_np = features.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()

    if method == 'tsne':
        reducer = TSNE(n_components=2, init='pca', random_state=42, perplexity=30)
    elif method == 'pca':
        reducer = PCA(n_components=2)
    else:
        raise ValueError("Method must be 'tsne' or 'pca'")

    reduced_features = reducer.fit_transform(features_np)

    # 绘图
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        reduced_features[:, 0], reduced_features[:, 1],
        c=labels_np, cmap='tab10', alpha=0.7, edgecolors='k'
    )
    plt.colorbar(scatter, label="Class Label")
    plt.title(f"Feature Visualization using {method.upper()}")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("CRGD_FSL_TSN.png")


# features = torch.randn(100, 128)  # 示例特征
# labels = torch.randint(1, 6, (100,))  # 示例标签（5类）

def pairwise_distances_logits(a, b):
    n = a.shape[0]
    m = b.shape[0]
    logits = -(
            (a.unsqueeze(1).expand(n, m, -1) - b.unsqueeze(0).expand(n, m, -1)) ** 2
    ).sum(dim=2)
    return logits


def pairwise_cosine_logits(a, b, epsilon=1e-6):
    # Normalize all embeddings to unit vectors
    norm_a = a / (torch.norm(a, dim=1, keepdim=True) + epsilon)
    norm_b = b / (torch.norm(b, dim=1, keepdim=True) + epsilon)

    # Calculate cosine angle between all support samples in each class
    cos = norm_a @ norm_b.T

    return cos


def accuracy(predictions, targets, binary=False):
    if binary:
        predictions = (predictions > 0.5).float().view(targets.shape)
    else:
        predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


#####################
# Prototypes
#####################


def gen_prototypes(embeddings, ways, shots, agg_method="mean"):
    assert (
            embeddings.size(0) == ways * shots
    ), "# of embeddings ({}) doesn't match ways ({}) and shots ({})".format(
        embeddings.size(0), ways, shots
    )

    embeddings = embeddings.reshape(ways, shots, -1)
    mean_embeddings = embeddings.mean(dim=1)

    if agg_method == "mean":
        return mean_embeddings

    elif agg_method == "median":
        # Init median as mean
        median_embeddings = torch.unsqueeze(mean_embeddings, dim=1)
        c = 0.5
        for i in range(5):
            errors = median_embeddings - embeddings
            # Poor man's Newton's method
            denom = torch.sqrt(torch.sum(errors ** 2, axis=2, keepdims=True) + c ** 2)
            dw = -torch.sum(errors / denom, axis=1, keepdims=True) / torch.sum(
                1.0 / denom, axis=1, keepdims=True
            )
            median_embeddings += dw
        return torch.squeeze(median_embeddings, dim=1)

    elif (
            agg_method.startswith("cosine")
            or agg_method.startswith("euclidean")
            or agg_method.startswith("abs")
    ):
        epsilon = 1e-6

        if agg_method.startswith("cosine"):
            # Normalize all embeddings to unit vectors
            norm_embeddings = embeddings / (
                    torch.norm(embeddings, dim=2, keepdim=True) + epsilon
            )
            # Calculate cosine angle between all support samples in each class: ways x shots x shots
            # Make negative, as higher cosine angle means greater correlation
            cos = torch.bmm(norm_embeddings, norm_embeddings.permute(0, 2, 1))
            attn = (torch.sum(cos, dim=1) - 1) / (shots - 1)
        elif agg_method.startswith("euclidean"):
            # dist: ways x shots x shots
            dist = (
                    (embeddings.unsqueeze(dim=2) - embeddings.unsqueeze(dim=1)) ** 2
            ).sum(dim=-1)
            attn = -torch.sum(dist, dim=1) / (shots - 1)
        elif agg_method.startswith("abs"):
            # dist: ways x shots x shots
            dist = (
                torch.abs(embeddings.unsqueeze(dim=2) - embeddings.unsqueeze(dim=1))
            ).sum(dim=-1)
            attn = -torch.sum(dist, dim=1) / (shots - 1)

        # Parse softmax temperature (default=1)
        T = float(agg_method.split("_")[-1]) if "_" in agg_method else 1
        weights = F.softmax(attn / T, dim=1).unsqueeze(dim=2)
        weighted_embeddings = embeddings * weights
        return weighted_embeddings.sum(dim=1)

    else:
        raise NotImplementedError


def gen_subset_prototypes(
        support_embeddings, shot, ways, subset_proportion=0.4, num_ensembles=10
):
    support_prototypes_sets = []
    num_support_samples = int(subset_proportion * shot)
    mask_array = [True] * num_support_samples + [False] * (shot - num_support_samples)

    for i in range(num_ensembles):
        subsample_indices_mask_i = np.concatenate(
            [np.random.permutation(mask_array) for _ in range(ways)]
        )
        sampled_support_embeddings = support_embeddings[subsample_indices_mask_i]
        support_prototypes_i = sampled_support_embeddings.reshape(
            ways, num_support_samples, -1
        ).mean(dim=1)
        support_prototypes_sets.append(support_prototypes_i)

    return support_prototypes_sets


def subset_prototype_predictions(subset_protos, query_embeddings, ways):
    # Make prediction with all sub-prototypes
    predictions = []
    for subset_protos_i in subset_protos:
        logits_i = pairwise_distances_logits(query_embeddings, subset_protos_i)
        predictions.append(torch.argmax(logits_i, dim=1))
    ensemble_predictions = torch.vstack(predictions)

    # Pick most common prediction as final prediction
    final_predictions = []
    for preds_i in ensemble_predictions.T:
        # Count up predictions of ensemble per class
        pred_counts_i = torch.bincount(preds_i, minlength=ways)
        # Find the class(es) with the most predictions
        (max_preds_i,) = torch.where(pred_counts_i == pred_counts_i.max())
        # Randomly select one of the classes with max predictions (if tie)
        random_index = np.random.randint(len(max_preds_i))
        final_predictions.append(max_preds_i[random_index])

    return torch.hstack(final_predictions)

def compute_feature_consistency(features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    计算类内方差（特征一致性指标，越小越好）

    参数：
        features: Tensor，形状为 (N, D)，N 是样本数，D 是特征维度
        labels: Tensor，形状为 (N,)，每个样本对应的标签（整数型）

    返回：
        feature_consistency: Tensor，标量，表示平均类内方差
    """
    unique_labels = labels.unique()
    intra_class_variances = []

    for cls in unique_labels:
        cls_mask = (labels == cls)
        cls_features = features[cls_mask]
        cls_center = cls_features.mean(dim=0, keepdim=True)
        variance = ((cls_features - cls_center) ** 2).sum(dim=1).mean()
        intra_class_variances.append(variance)

    return torch.stack(intra_class_variances).mean()



#####################
# MetaTrainer
#####################

# import clip
# # import
# import torchvision.transforms as transforms
# import torch.nn.functional as F
# from torchvision.transforms.functional import resize
# clip_transform = transforms.Compose([
#     transforms.Resize(224),
#     transforms.CenterCrop(224),
#     transforms.Normalize(
#     mean=(0.48145466, 0.4578275, 0.40821073),
#     std=(0.26862954, 0.26130258, 0.27577711))
#
# ])
# def compute_clip_similarity_from_tensor(image_tensor_batch, label_indices, label_file):
#     """
#     使用 CLIP 提取图像（Tensor）和文本特征，并计算相似度，文本使用 "a photo of a {label}" 模板。
#
#     参数：
#     - image_tensor_batch: torch.Tensor，形状为 (B, 3, 224, 224)，已预处理为 CLIP 输入大小
#     - label_indices: List[int]，图像对应的标签索引
#     - label_file: str，包含标签语义（每行一个标签）
#
#     返回：
#     - similarity_scores: torch.Tensor，相似度分数 (B,)
#     - image_features: torch.Tensor，图像特征 (B, D)
#     - text_features: torch.Tensor，文本特征 (B, D)
#     """
#     device=torch.device('cuda:0')
#     model, preprocess = clip.load("/home/wuzelei/Preprocess/pretrained/ViT-B-32.pt", device=device)
#     image_tensor_batch = image_tensor_batch.to(device)
#     image_tensor_batch = torch.stack([clip_transform(img) for img in image_tensor_batch])
#     # 预处理图像数据
#     # image_tensor_batch = torch.stack([preprocess(img) for img in image_tensor_batch])
#
#     with torch.no_grad():
#         # 图像特征提取
#         image_features = model.encode_image(image_tensor_batch)
#         image_features /= image_features.norm(dim=-1, keepdim=True)
#
#         # 读取标签语义
#         with open(label_file, "r", encoding="utf-8") as f:
#             raw_labels = [line.strip() for line in f]
#
#         # 为每个标签生成文本模板
#         batch_texts = [f"an image of {label}" for label in raw_labels]
#         text_tokens = clip.tokenize(batch_texts).to(device)
#
#         # 文本特征提取
#         text_features = model.encode_text(text_tokens)
#         text_features /= text_features.norm(dim=-1, keepdim=True)
#
#         # 计算每张图像与所有标签的相似度
#         similarity_scores = (image_features @ text_features.T)  # 计算图像与所有文本的相似度，得到 (B, L) 的相似度矩阵
#
#         # 归一化相似度（可以使用 softmax 或其他方法）
#         norm_similarity_scores = F.softmax(similarity_scores, dim=-1)  # 对每张图像的相似度进行归一化
#         max_similarity_scores, max_similarity_indices = torch.max(norm_similarity_scores, dim=-1)
#         print(max_similarity_scores,max_similarity_indices)
#     return norm_similarity_scores, image_features, text_features
#
# def compute_clip_similarity_from_tensor(image_tensor_batch, label_y, label_file):
#     import torch.nn.functional as F
#     import clip
#     import torch
#     from torchvision import transforms
#     to_pil = transforms.ToPILImage()
#
#
#     device = torch.device('cuda:0')
#     model, preprocess = clip.load("/home/wuzelei/Preprocess/pretrained/ViT-B-32.pt", device=device)
#     model.eval()
#
#     # 预处理图像（先移回 CPU 再转回 GPU 是为了防止非 PIL 图像出错）
#     # image_tensor_batch = torch.stack([preprocess(img.cpu()) for img in image_tensor_batch]).to(device)
#     # Tensor 转 PIL，再用 CLIP 的 preprocess
#     image_tensor_batch = torch.stack([
#         preprocess(to_pil(img.cpu())) for img in image_tensor_batch
#     ]).to(device)
#     with torch.no_grad():
#         image_features = model.encode_image(image_tensor_batch)
#         image_features /= image_features.norm(dim=-1, keepdim=True)
#
#         # 加载标签文本
#         with open(label_file, "r", encoding="utf-8") as f:
#             raw_labels = [line.strip() for line in f]
#
#         texts = [f"an image of {label}" for label in raw_labels]
#         text_tokens = clip.tokenize(texts).to(device)
#         text_features = model.encode_text(text_tokens)
#         text_features /= text_features.norm(dim=-1, keepdim=True)
#
#         # 相似度计算
#         similarity_scores = image_features @ text_features.T  # (B, L)
#
#         # 如果你想得到对应标签的相似度：
#         true_label_similarities = similarity_scores[torch.arange(len(label_y)), label_y]
#         max_similarity_scores, max_similarity_indices = torch.max(similarity_scores, dim=-1)
#         print(max_similarity_indices,max_similarity_scores,label_y
#         # torch.argmax(similarity_scores[i])
#     return similarity_scores, true_label_similarities, image_features, text_features
# def compute_clip_similarity_from_tensor(image_tensor_batch, labels_y, label_json_file):
#     import torch.nn.functional as F
#
#
#     to_pil = transforms.ToPILImage()
#     device = torch.device('cuda:0')
#
#     # 加载 CLIP 模型
#     model, preprocess = clip.load("/home/wuzelei/Preprocess/pretrained/ViT-B-32.pt", device=device)
#     model.eval()
#
#     # 图像预处理（tensor -> PIL -> preprocess）
#     image_tensor_batch = torch.stack([
#         preprocess(to_pil(img.cpu())) for img in image_tensor_batch
#     ]).to(device)
#     # labels_y=labels_y[0]
#     # print(len(labels_y))
#     print(len(image_tensor_batch))
#     # 加载 JSON 文件中的标签
#     with open(label_json_file, "r", encoding="utf-8") as f:
#         label_dict = json.load(f)
#     # print(labels_y)
#     # 如果 labels_y 是 tensor，转换为 list
#     if isinstance(labels_y, torch.Tensor):
#         labels_y = labels_y.tolist()
#     print(labels_y)
#     # 使用 labels_y 去 JSON 中取语义描述
#     text_prompts = [f"an image of {label_dict[str(label[0])]}" for label in labels_y]
#     print(text_prompts)
#     # Tokenize 文本
#     text_tokens = clip.tokenize(text_prompts).to(device)
#
#     with torch.no_grad():
#         # 图像特征
#         image_features = model.encode_image(image_tensor_batch)
#         image_features /= image_features.norm(dim=-1, keepdim=True)
#
#         # 文本特征
#         text_features = model.encode_text(text_tokens)
#         text_features /= text_features.norm(dim=-1, keepdim=True)
#
#         # 每个图像和对应文本的相似度（逐对比较）
#         print(image_features.shape)
#         print(text_features.shape)
#         similarity_scores = torch.sum(image_features * text_features, dim=-1)
#
#     return similarity_scores, image_features, text_features
import torch
import torch.nn.functional as F
import clip
import json
from torchvision import transforms


# def compute_clip_similarity_from_tensor(image_tensor_batch, labels_y, label_json_file, clip_threshold=0.19):
#     to_pil = transforms.ToPILImage()
#     device = torch.device('cuda:0')
#
#     # 加载 CLIP 模型
#     model, preprocess = clip.load("/home/wuzelei/Preprocess/pretrained/ViT-B-32.pt", device=device)
#     model.eval()
#
#     # 图像预处理（tensor -> PIL -> preprocess）
#     image_tensor_batch = torch.stack([preprocess(to_pil(img.cpu())) for img in image_tensor_batch]).to(device)
#     # print(len(image_tensor_batch))
#
#     # 加载 JSON 文件中的标签
#     with open(label_json_file, "r", encoding="utf-8") as f:
#         label_dict = json.load(f)
#
#     # 如果 labels_y 是 tensor，转换为 list
#     if isinstance(labels_y, torch.Tensor):
#         labels_y = labels_y.tolist()
#     # print(labels_y)
#
#     # 使用 labels_y 去 JSON 中取语义描述
#     text_prompts = [f"an image of {label_dict[str(label[0])]}" for label in labels_y]
#     # print(text_prompts)
#
#     # Tokenize 文本
#     text_tokens = clip.tokenize(text_prompts).to(device)
#
#     with torch.no_grad():
#         # 图像特征
#         image_features = model.encode_image(image_tensor_batch)
#         image_features /= image_features.norm(dim=-1, keepdim=True)
#
#         # 文本特征
#         text_features = model.encode_text(text_tokens)
#         text_features /= text_features.norm(dim=-1, keepdim=True)
#
#         # 每个图像和对应文本的相似度（逐对比较）
#         similarity_scores = torch.sum(image_features * text_features, dim=-1)
#
#     # 通过相似度分数和阈值判断正确与错误标签
#     correct_labels = similarity_scores >= clip_threshold
#     incorrect_labels = similarity_scores < clip_threshold
#
#     # 返回正确标签和错误标签的索引及其相似度分数
#     correct_indices = torch.nonzero(correct_labels).squeeze()
#     incorrect_indices = torch.nonzero(incorrect_labels).squeeze()
#
#     # 确保 correct_indices 和 incorrect_indices 是可以迭代的
#     if correct_indices.dim() == 0:
#         correct_indices = correct_indices.unsqueeze(0)
#     if incorrect_indices.dim() == 0:
#         incorrect_indices = incorrect_indices.unsqueeze(0)
#
#     correct_scores = similarity_scores[correct_indices]
#     incorrect_scores = similarity_scores[incorrect_indices]
#     incorrect_prompts = [text_prompts[i] for i in incorrect_indices]
#     correct_labels = [labels_y[i] for i in correct_indices]
#     incorrect_labels = [labels_y[i] for i in incorrect_indices]
#
#     # 输出
#     # print("Correct labels:", correct_labels)
#     # print("Incorrect labels:", incorrect_labels)
#
#     return correct_labels, incorrect_labels, correct_scores, incorrect_scores, correct_indices, incorrect_indices, similarity_scores, incorrect_prompts
def compute_clip_similarity_from_tensor(image_tensor_batch, labels_y, label_json_file, clip_threshold=0.19):
    to_pil = transforms.ToPILImage()
    device = torch.device('cuda:0')

    # 加载 CLIP 模型
    model, preprocess = clip.load("/home/wuzelei/Preprocess/pretrained/ViT-B-32.pt", device=device)
    model.eval()

    # 图像预处理（tensor -> PIL -> preprocess）
    image_tensor_batch = torch.stack([preprocess(to_pil(img.cpu())) for img in image_tensor_batch]).to(device)

    # 加载 JSON 文件中的标签
    with open(label_json_file, "r", encoding="utf-8") as f:
        label_dict = json.load(f)

    # 转换 labels_y 为 list
    if isinstance(labels_y, torch.Tensor):
        labels_y = labels_y.tolist()

    # 生成文本描述
    text_prompts = [f"an image of {label_dict[str(label[0])]}" for label in labels_y]
    batch_size = len(text_prompts)  # 获取批次大小

    # Tokenize 文本
    text_tokens = clip.tokenize(text_prompts).to(device)

    with torch.no_grad():
        # 计算特征
        image_features = model.encode_image(image_tensor_batch)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # 计算相似度
        similarity_scores = (image_features * text_features).sum(dim=-1)

    # 生成布尔掩码并获取低分索引
    incorrect_mask = similarity_scores < clip_threshold
    incorrect_indices = torch.nonzero(incorrect_mask).squeeze()

    # 将张量索引转为 Python 列表（处理单元素/空张量）
    if incorrect_indices.dim() == 0:  # 单个索引
        indices = [incorrect_indices.item()]
    else:  # 多个索引或无索引
        indices = incorrect_indices.cpu().tolist()
    indices = indices if isinstance(indices, list) else []

    # 生成完整提示词列表：低分位置保留原提示，其他置空
    incorrect_prompts_full = [""] * batch_size
    for idx in indices:
        if idx < batch_size:  # 防止越界
            incorrect_prompts_full[idx] = text_prompts[idx]

    # 返回结果（保留其他原有返回项）
    return incorrect_indices, similarity_scores, incorrect_prompts_full  # 替换为完整列表

class SimpleDiffusion:
    def __init__(self, num_steps=100, beta_start=1e-4, beta_end=0.02, device='cpu'):
        self.num_steps = num_steps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, num_steps).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_hats = torch.cumprod(self.alphas, dim=0)

    def get_alpha_hat(self, t: int):
        return self.alpha_hats[t]

    def add_noise(self, x0: torch.Tensor, t: int):
        """
        Forward diffusion (add noise at timestep t)
        Args:
            x0: original clean features [N, D]
            t: timestep (int)
        Returns:
            x_t: noisy features [N, D]
            noise: added noise [N, D]
        """
        alpha_hat = self.get_alpha_hat(t)
        noise = torch.randn_like(x0)
        x_t = torch.sqrt(alpha_hat) * x0 + torch.sqrt(1 - alpha_hat) * noise
        return x_t, noise

    def denoise(self, x_t: torch.Tensor, noise: torch.Tensor, t: int):
        """
        Reverse denoising process
        Args:
            x_t: noisy features [N, D]
            noise: same noise used in forward process [N, D]
            t: timestep (int)
        Returns:
            x0_pred: denoised features [N, D]
        """
        alpha_hat = self.get_alpha_hat(t)
        x0_pred = (x_t - torch.sqrt(1 - alpha_hat) * noise) / torch.sqrt(alpha_hat)
        return x0_pred


class MetaTrainer:
    def __init__(self, model, device, args):
        self.model = model
        self.device = device
        self.args = args
        self.projection = nn.Linear(1600, 512)  # 添加到模型定义中
        self.projection.to(self.device)
        self.temperature = 0.1#在模型初始化中添加
        self.clip_threshold = nn.Parameter(torch.tensor(0.5))
        # self._init_diffusion_model()
        self.clip_model, self.clip_preprocess = clip.load("/home/wuzelei/Preprocess/pretrained/ViT-B-32.pt", device=device)
        self.clip_model = self.clip_model.float()  # Ensure CLIP model uses float32

        if args.dist_metric == "euclidean":
            self.metric = pairwise_distances_logits
        elif args.dist_metric == "cosine":
            self.metric = pairwise_cosine_logits
        else:

            raise NotImplementedError

    # def _init_diffusion_model(self):
    #     """初始化图像生成模型和预处理流程"""
    #     # 本地模型路径配置
    #     scheduler_path = "/home/wuzelei/Preprocess/stable-diffusion-2-base/models--stabilityai--stable-diffusion-2-base/snapshots/fa386bb446685d8ad8a8f06e732a66ad10be6f47/scheduler"
    #     model_path = "/home/wuzelei/Preprocess/stable-diffusion-2-base/models--stabilityai--stable-diffusion-2-base/snapshots/fa386bb446685d8ad8a8f06e732a66ad10be6f47/"
    #
    #     # 加载调度器和扩散模型
    #     try:
    #         self.scheduler = EulerDiscreteScheduler.from_pretrained(scheduler_path)
    #         self.pipe = StableDiffusionPipeline.from_pretrained(
    #             model_path,
    #             scheduler=self.scheduler,
    #             torch_dtype=torch.float16  # 使用半精度节省显存
    #         ).to(self.device)
    #     except Exception as e:
    #         raise RuntimeError(f"Failed to load diffusion model: {str(e)}")
    #
    #     # 设置与训练数据一致的预处理流程
    #     self.data_preprocess = transforms.Compose([
    #         transforms.Resize(84),  # 匹配输入尺寸
    #         transforms.ToTensor(),
    #         transforms.Normalize(
    #             mean=[0.485, 0.456, 0.406],  # ImageNet 标准归一化
    #             std=[0.229, 0.224, 0.225]
    #         )
    #     ])
    def wordnet_id_to_label(self, wnid: str) -> str:
        """
        将 WordNet ID（如 "n02119789"）转换为可读的英文标签。
        如果无法通过标准方式获取，则直接返回原 wnid。

        Args:
            wnid: WordNet ID (e.g., "n02119789")
        Returns:
            label: Readable label (e.g., "Afghan hound")
        """
        try:
            # WordNet ID 的首字母代表词性（n=名词，v=动词等），后面是整数 offset
            pos = wnid[0]
            offset = int(wnid[1:])
            syn = wn.synset_from_pos_and_offset(pos, offset)
        except Exception:
            # Fallback: 尝试直接用 wn.synsets 查找
            syns = wn.synsets(wnid)
            if len(syns) > 0:
                syn = syns[0]
            else:
                # 如果还是没找到，返回原 ID
                return wnid

        # 取第一个 lemma，并把下划线替换为空格
        return syn.lemmas()[0].name().replace('_', ' ')
    def prepare_data(
            self,
            batch,
            ways,
            shot,
            query_num,
            support_label_noise_choices,
            outlier_batch=None,
    ):
        # Format data and place on device
        data, labels, labels_id = preprocess_data_labels(batch, self.device)
        if outlier_batch:
            outlier_data, _ = preprocess_data_labels(outlier_batch, self.device)
        else:
            outlier_data = None

        # Separate query and support分割查询集和支持集
        mask_indices = get_support_noise_query_indices(ways, shot, query_num)
        support_indices = torch.where(mask_indices['support'])[0].tolist()
        # 筛选支持集对应的标签ID
        selected_labels_id = [labels_id[i] for i in support_indices]
        # Add noise
        support_label_noise = np.random.choice(support_label_noise_choices)
        noise_positions = np.zeros(data.shape[0])
        if support_label_noise > 0:
            data, labels, noise_positions = add_noise(
                data,
                labels,
                mask_indices,
                ways,
                support_label_noise,
                self.args.noise_type,
                outlier_data,
            )

        # Remove noise batch data for swapping, so conv doesn't need to process
        data = data[~mask_indices["noise"]]
        labels = labels[~mask_indices["noise"]]
        noise_positions = noise_positions[~mask_indices["noise"]]
        mask_indices["support"] = mask_indices["support"][~mask_indices["noise"]]
        mask_indices["query"] = mask_indices["query"][~mask_indices["noise"]]
        # ##new
        # if isinstance(labels_id, list):
        #     labels_y = [labels_id[i] for i in (~mask_indices["noise"]).nonzero(as_tuple=True)[0].tolist()]
        # else:  # 如果是 tensor（比如 one-hot 编码）
        #     labels_y = labels_id[~mask_indices["noise"]]
        return data, labels, mask_indices, noise_positions, selected_labels_id


    def prototype_regularization_loss(self, prototypes, margin=0.5):
        """
        原型正则化损失：最大化类原型之间的最小距离。

        Args:
            prototypes: [ways, embed_dim] 类原型
            margin: 最小距离阈值
        Returns:
            loss: 正则化损失
        """
        # 计算原型间的两两距离
        prototypes = prototypes.float()
        dists = torch.pdist(prototypes)  # [ways * (ways-1) / 2]
        # 鼓励最小距离大于 margin
        loss = torch.relu(margin - dists).mean()
        return loss
    def loss_acc(self, batch, outlier_batch=None, support_label_noise_choices=[0]):
        ways = self.args.train_way if self.model.training else self.args.test_way
        shot = self.args.train_shot if self.model.training else self.args.test_shot
        query_num = (
            self.args.train_query if self.model.training else self.args.test_query
        )

        # Format data, place on device, add noise
        data, labels, mask_indices, noise_positions,labels_id = self.prepare_data(
            batch, ways, shot, query_num, support_label_noise_choices, outlier_batch
        )
        # visualize_features(data,labels)
        # print(compute_feature_consistency(data,labels))
        support_labels1 = labels[mask_indices["support"]].to(self.device)
        support_data = data[mask_indices["support"]]
        # Compute support/query embeddings and prototypes
        outputs = self.model(data, mask_indices["support"])
        support_embeddings = outputs["conv_embeddings"][mask_indices["support"]]
        visualize_features(outputs["conv_embeddings"],labels)
        query_embeddings = outputs["conv_embeddings"][mask_indices["query"]]
        try:
            outlier_logits=outputs["outlier_logits"]
            outlier_probs = torch.sigmoid(outlier_logits)  # 转换为概率
        except:
            outlier_probs=None
        # def clip_weighted_prototypes(support_embeddings, support_labels, batch_data, class_names):
        #     """集成多提示模板的CLIP加权原型计算"""
        #     # 提示模板库（可根据任务扩展）
        #     prompt_templates = [
        #         "a photo of a {}",
        #         "a clean photo of a {}",
        #         "a cropped image of a {}",
        #         "a bright image of a {}",
        #         "a dark photo of a {}",  # 涵盖不同成像条件
        #         "a {} in natural scene",
        #         "a rendered image of a {}",
        #         "a sketch drawing of a {}",
        #         "a toy version of a {}",
        #         "a close-up photo of a {}"
        #     ]
        #
        #     # 生成多提示文本特征
        #     text_features = []
        #     for name in class_names:
        #         # 为每个类别生成多提示特征并平均
        #         texts = [template.format(name) for template in prompt_templates]
        #         text_inputs = torch.cat([clip.tokenize(t) for t in texts]).to(self.device)
        #         with torch.no_grad():
        #             class_feats = self.clip_model.encode_text(text_inputs)
        #             class_feats = class_feats.mean(dim=0, keepdim=True)  # 多提示平均
        #             class_feats /= class_feats.norm(dim=-1, keepdim=True)
        #         text_features.append(class_feats)
        #     text_features = torch.cat(text_features)  # [num_classes, feat_dim]
        #
        #     # 计算支持集样本的CLIP图像特征（与之前相同）
        #     support_images = batch_data[mask_indices["support"]].cpu().permute(0, 2, 3, 1).numpy()
        #     support_images = (support_images * 255).astype(np.uint8)
        #     image_features = []
        #     for img in support_images:
        #         img_pil = Image.fromarray(img)
        #         img_clip = self.clip_preprocess(img_pil).unsqueeze(0).to(self.device)
        #         with torch.no_grad():
        #             img_feat = self.clip_model.encode_image(img_clip)
        #             img_feat /= img_feat.norm(dim=-1, keepdim=True)
        #         image_features.append(img_feat)
        #     image_features = torch.cat(image_features)  # [num_support, feat_dim]
        #
        #     # 计算相似度得分（考虑所有类别的最大相似度）
        #     sim_matrix = image_features @ text_features.T  # [num_support, num_classes]
        #     sim_scores, pred_labels = torch.max(sim_matrix, dim=1)
        #
        #     # 动态权重分配（基于预测置信度）
        #     temperature = 0.1  # 更陡峭的置信度分布
        #     weights = F.softmax(sim_scores / temperature, dim=0)
        #
        #     # 异常值检测：过滤低置信度样本
        #     mask = sim_scores > torch.quantile(sim_scores, 0.25)  # 保留前75%样本
        #     weights = weights * mask.float()
        #     weights /= weights.sum() + 1e-8
        #
        #     # 重新计算原型（考虑CLIP预测标签）
        #     prototypes = []
        #     for cls in range(len(class_names)):
        #         # 合并原始标签和CLIP预测标签
        #         cls_mask = (support_labels == cls) | (pred_labels == cls)
        #         cls_weights = weights[cls_mask]
        #         cls_feats = support_embeddings[cls_mask]
        #         if len(cls_feats) == 0:  # 防止空类
        #             prototypes.append(support_embeddings[support_labels == cls].mean(dim=0))
        #         else:
        #             prototypes.append((cls_feats * cls_weights.view(-1, 1)).sum(dim=0))
        #
        #     return torch.stack(prototypes)
        # def clip_weighted_prototypes(image_tensor_batch, labels_id, label_json_file, clip_threshold=0.19):
        #     to_pil = transforms.ToPILImage()
        #     device = torch.device('cuda:0')
        #
        #     # 加载 CLIP 模型
        #     # model, preprocess = clip.load("/home/wuzelei/Preprocess/pretrained/ViT-B-32.pt", device=device)
        #     # model.eval()
        #
        #     # 图像预处理（tensor -> PIL -> preprocess）
        #     image_tensor_batch = torch.stack([self.clip_preprocess(to_pil(img.cpu())) for img in image_tensor_batch]).to(device)
        #
        #     # 加载 JSON 文件中的标签
        #     with open(label_json_file, "r", encoding="utf-8") as f:
        #         label_dict = json.load(f)
        #
        #     # 转换 labels_y 为 list
        #     if isinstance(labels_id, torch.Tensor):
        #         labels_id = labels_id.tolist()
        #
        #     # 生成文本描述
        #     text_prompts = [f"an image of {label_dict[str(label[0])]}" for label in labels_id]
        #     batch_size = len(text_prompts)  # 获取批次大小
        #
        #     # Tokenize 文本
        #     text_tokens = clip.tokenize(text_prompts).to(device)
        #
        #     with torch.no_grad():
        #         # 计算特征
        #         image_features = self.clip_model.encode_image(image_tensor_batch)
        #         image_features /= image_features.norm(dim=-1, keepdim=True)
        #
        #         text_features = self.clip_model.encode_text(text_tokens)
        #         text_features /= text_features.norm(dim=-1, keepdim=True)
        #
        #         # 计算相似度
        #         similarity_scores = (image_features * text_features).sum(dim=-1)
        #
        #     # 生成布尔掩码并获取低分索引
        #     incorrect_mask = similarity_scores < clip_threshold
        #     incorrect_indices = torch.nonzero(incorrect_mask).squeeze()
        #
        #     # 将张量索引转为 Python 列表（处理单元素/空张量）
        #     if incorrect_indices.dim() == 0:  # 单个索引
        #         indices = [incorrect_indices.item()]
        #     else:  # 多个索引或无索引
        #         indices = incorrect_indices.cpu().tolist()
        #     indices = indices if isinstance(indices, list) else []
        #
        #     # 生成完整提示词列表：低分位置保留原提示，其他置空
        #     incorrect_prompts_full = [""] * batch_size
        #     for idx in indices:
        #         if idx < batch_size:  # 防止越界
        #             incorrect_prompts_full[idx] = text_prompts[idx]
        #
        #     # 返回结果（保留其他原有返回项）
        #     return incorrect_indices, similarity_scores, incorrect_prompts_full  # 替换为完整列表

        # Note: support labels assumes class labels 0..(N-1), in order. labels has original labels before noise
        support_labels = torch.repeat_interleave(torch.arange(ways), shot).to(
            self.device
        )
        query_labels = labels[mask_indices["query"]].long()

        losses = {}
        accs = {}
        if self.args.comp_method == "proto":
            def wordnet_id_to_label(wnid: str) -> str:
                """
                将 WordNet ID（如 "n02119789"）转换为可读的英文标签。
                如果无法通过标准方式获取，则直接返回原 wnid。
                """
                try:
                    # WordNet ID 的首字母代表词性（n/v/a/r），后面是整数 offset
                    pos = wnid[0]
                    offset = int(wnid[1:])
                    syn = wn.synset_from_pos_and_offset(pos, offset)
                except Exception:
                    # fallback：尝试直接用 wn.synsets 查找
                    syns = wn.synsets(wnid)
                    if len(syns) > 0:
                        syn = syns[0]
                    else:
                        # 如果还是没找到，就直接返回原 ID
                        return wnid

                # 取第一个 lemma，并把下划线替换为空格
                return syn.lemmas()[0].name().replace('_', ' ')

            def clip_weighted_prototypes(
                    support_embeddings: Tensor,  # [N, D]
                    support_data: Tensor,  # [N, C, H, W]
                    labels_id: List[str],  # WordNet ID
                    support_labels: Tensor,  # [N] raw 0..ways-1
                    ways: int,
                    outlier_probs: Optional[Tensor] = None,
                    outlier_threshold: float = 0.5,
                    diffusion_t: int = 70,
                    diffusion_steps: int = 100
            ) -> Tensor:  # returns [ways, D]
                """
                Compute class prototypes by CLIP-weighted aggregation of support embeddings,
                with optional diffusion-based denoising for outliers.
                Returns:
                    Tensor of shape [ways, D]
                """
                # 1) WordNet ID -> 文本标签
                class_names = []
                for e in labels_id:
                    wnid = e[0] if isinstance(e, (list, tuple)) else e
                    class_names.append(wordnet_id_to_label(wnid))

                # 2) 唯一标签、映射
                unique_labels = list(OrderedDict.fromkeys(class_names))
                assert len(unique_labels) == ways, \
                    f"实际类别数 ({len(unique_labels)}) != ways ({ways})"

                # 3) 文本特征 batch 编码 & 统计每类 prompt 数
                prompt_templates = [
                    "an image of {}", "a photo of a {}", "a cropped image of a {}",
                    "a bright image of a {}", "a dark photo of a {}", "a {} in natural scene",
                    "a rendered image of a {}", "a sketch drawing of a {}", "a toy version of a {}",
                    "a close-up photo of a {}"
                ]

                def generate_hierarchical_prompts(label: str) -> List[str]:
                    synset = wn.synsets(label.replace(' ', '_'))[0]
                    hierarchy = []
                    while synset:
                        hierarchy.append(synset.lemmas()[0].name().replace('_', ' '))
                        synset = synset.hypernyms()[0] if synset.hypernyms() else None
                    return [f"a {parent} that is a type of {label}" for parent in hierarchy[:-1]][:3]

                per_class_counts = []
                prompts = []
                for lbl in unique_labels:
                    extra = generate_hierarchical_prompts(lbl)
                    templates = prompt_templates + extra
                    per_class_counts.append(len(templates))
                    prompts += [t.format(lbl) for t in templates]

                # Tokenize + 文本编码
                text_inputs = clip.tokenize(prompts).to(self.device)
                with torch.no_grad():
                    txt_feats = self.clip_model.encode_text(text_inputs)  # [sum(per_class_counts), D]
                D = txt_feats.size(-1)

                # 逐类分割、求均值
                offset = 0
                text_features = []
                for count in per_class_counts:
                    feats = txt_feats[offset: offset + count]  # [count, D]
                    offset += count
                    text_features.append(feats.mean(0))
                text_features = torch.stack(text_features, dim=0)  # [ways, D]
                text_features = F.normalize(text_features, dim=-1)

                # 4) 图像特征 batch 编码
                imgs = support_data.cpu().permute(0, 2, 3, 1).numpy()
                pil_imgs = [Image.fromarray((im * 255).astype(np.uint8)) for im in imgs]
                img_batch = torch.cat([
                    self.clip_preprocess(im).unsqueeze(0) for im in pil_imgs
                ]).to(self.device)
                with torch.no_grad():
                    img_feats = self.clip_model.encode_image(img_batch)
                img_feats = F.normalize(img_feats, dim=-1)  # [N, D]

                # 5) 相似度 & 权重
                sim = img_feats @ text_features.T  # [N, ways]
                sim_scores, sim_preds = sim.max(1)  # [N]
                agree = sim_preds == support_labels
                clip_w = sim_scores.clone()
                clip_w[~agree] *= 0.1
                conf_w = (1 - outlier_probs.view(-1)) if outlier_probs is not None else torch.ones_like(clip_w)
                total_w = clip_w * conf_w  # [N]

                # 6) 扩散增强（可选）
                if outlier_probs is not None:
                    noisy_mask = outlier_probs.view(-1) > outlier_threshold
                    if noisy_mask.any():
                        diffuser = SimpleDiffusion(num_steps=diffusion_steps, device=self.device)
                        x0 = support_embeddings[noisy_mask]  # [M, D]
                        x_t, noise = diffuser.add_noise(x0, diffusion_t)
                        mean_feat = support_embeddings[~noisy_mask].mean(0, keepdim=True)
                        mean_feat = mean_feat.expand_as(x0)
                        alpha_hat = diffuser.get_alpha_hat(diffusion_t)
                        gamma = 1 - torch.sqrt(alpha_hat)
                        x_denoised = (1 - gamma) * x_t + gamma * mean_feat
                        support_embeddings = support_embeddings.clone()
                        support_embeddings[noisy_mask] = x_denoised

                # 7) per-class 聚合
                prototypes = []
                for cid in range(ways):
                    mask = (support_labels == cid)
                    feats = support_embeddings[mask]
                    ws = total_w[mask]
                    if feats.size(0) == 0:
                        proto = text_features[cid]
                    else:
                        valid = ws > self.clip_threshold
                        if valid.sum() > 0:
                            w2 = ws[valid] / (ws[valid].sum() + 1e-8)
                            proto = (feats[valid] * w2.unsqueeze(1)).sum(0)
                        else:
                            proto = feats.mean(0)
                    prototypes.append(F.normalize(proto, dim=-1))

                return torch.stack(prototypes, dim=0)  # [ways, D]
            def clip_weighted_prototypes_1(

                    support_embeddings: Tensor,  # [N, D]
                    support_data: Tensor,  # [N, C, H, W]
                    labels_id: List[str],  # WordNet ID
                    support_labels: Tensor,  # [N] raw 0..ways-1
                    ways: int,
                    outlier_probs: Optional[Tensor] = None,
                    outlier_threshold: float = 0.5,
                    diffusion_t: int = 70,
                    diffusion_steps: int = 100
            ) -> Tensor:  # returns [ways, D]

                # 1) WordNet ID -> 文本标签
                class_names = []
                for e in labels_id:
                    wnid = e[0] if isinstance(e, (list, tuple)) else e
                    class_names.append(wordnet_id_to_label(wnid))

                # 2) 唯一标签、映射
                unique_labels = list(OrderedDict.fromkeys(class_names))
                assert len(unique_labels) == ways, "实际类别数 != ways"

                # 3) 文本特征 batch 编码
                def generate_hierarchical_prompts(label):
                    synset = wn.synsets(label.replace(' ', '_'))[0]
                    hierarchy = []
                    while synset:
                        hierarchy.append(synset.lemmas()[0].name().replace('_', ' '))
                        if synset.hypernyms():
                            synset = synset.hypernyms()[0]
                        else:
                            break
                    return [f"a {parent} that is a type of {label}" for parent in hierarchy[:-1]][:3]

                prompt_templates = [
                    "an image of {}", "a photo of a {}", "a cropped image of a {}",
                    "a bright image of a {}", "a dark photo of a {}", "a {} in natural scene",
                    "a rendered image of a {}", "a sketch drawing of a {}", "a toy version of a {}",
                    "a close-up photo of a {}"
                ]
                prompts = []
                for lbl in unique_labels:
                    extra_prompts = generate_hierarchical_prompts(lbl)
                    templates = (prompt_templates + extra_prompts)[:13]  # 保证每类 13 条
                    prompts += [t.format(lbl) for t in templates]
                text_inputs = clip.tokenize(prompts).to(self.device)
                with torch.no_grad():
                    txt_feats = self.clip_model.encode_text(text_inputs)  # [ways*num_temp, D]
                D = txt_feats.size(-1)
                text_features = txt_feats.view(ways, -1, D).mean(1)  # [ways, D]
                text_features = F.normalize(text_features, dim=-1)

                # 4) 图像特征 batch 编码
                imgs = support_data.cpu().permute(0, 2, 3, 1).numpy()
                pil_imgs = [Image.fromarray((im * 255).astype(np.uint8)) for im in imgs]
                img_batch = torch.cat([self.clip_preprocess(im).unsqueeze(0) for im in pil_imgs]).to(self.device)
                with torch.no_grad():
                    img_feats = self.clip_model.encode_image(img_batch)
                img_feats = F.normalize(img_feats, dim=-1)  # [N, D]

                # 5) 相似度 & 权重
                sim = img_feats @ text_features.T  # [N, ways]
                sim_scores, sim_preds = sim.max(1)  # [N]
                agree = sim_preds == support_labels
                clip_w = sim_scores.clone()
                clip_w[~agree] *= 0.1
                conf_w = (1 - outlier_probs.view(-1)) if outlier_probs is not None else torch.ones_like(clip_w)
                total_w = clip_w * conf_w  # [N]

                # —— 扩散增强：对被判定为噪声的样本，先加噪再“向非噪声均值”去噪 ——
                if outlier_probs is not None:
                    noisy_mask = outlier_probs.view(-1) > outlier_threshold
                    if noisy_mask.any():
                        diffuser = SimpleDiffusion(num_steps=diffusion_steps, device=self.device)
                        x0 = support_embeddings[noisy_mask]  # [M, D]
                        x_t, noise = diffuser.add_noise(x0, diffusion_t)  # 加噪

                        # 计算非噪声样本的均值
                        mean_feat = support_embeddings[~noisy_mask].mean(0, keepdim=True)  # [1, D]
                        mean_feat = mean_feat.expand_as(x0)  # [M, D]

                        # 计算拉向均值的权重 gamma
                        alpha_hat = diffuser.get_alpha_hat(diffusion_t)
                        gamma = 1 - torch.sqrt(alpha_hat)

                        # 往均值方向“去噪”
                        x_denoised = (1 - gamma) * x_t + gamma * mean_feat

                        # 替换回 support_embeddings
                        support_embeddings = support_embeddings.clone()
                        support_embeddings[noisy_mask] = x_denoised

                # 6) per-class 聚合
                prototypes = []
                for cid in range(ways):
                    mask = (support_labels == cid)
                    feats = support_embeddings[mask]
                    ws = total_w[mask]
                    if feats.size(0) == 0:
                        proto = text_features[cid]
                    else:
                        valid = ws > self.clip_threshold
                        if valid.sum() > 0:
                            w2 = ws[valid] / (ws[valid].sum() + 1e-8)
                            proto = (feats[valid] * w2.unsqueeze(1)).sum(0)
                        else:
                            proto = feats.mean(0)
                    prototypes.append(F.normalize(proto, dim=-1))

                return torch.stack(prototypes, dim=0)
            support_prototypes = clip_weighted_prototypes(
                support_embeddings=support_embeddings,
                support_data=support_data,
                labels_id=labels_id,
                support_labels=support_labels,
                ways=self.args.train_way,
                outlier_probs=outlier_probs
                # clip_threshold=0.2 # 与compute_clip_similarity_from_tensor相同路径
            )

            def contrastive_loss(
                    query_embeddings: torch.Tensor,  # [num_query, embed_dim]
                    support_prototypes: torch.Tensor,  # [ways, embed_dim]
                    query_labels: torch.Tensor,  # [num_query]
                    temperature: float = 0.1
            ) -> torch.Tensor:
                """
                计算查询嵌入与支持集原型之间的对比学习损失。

                Args:
                    query_embeddings: 查询集嵌入，形状 [num_query, embed_dim]
                    support_prototypes: 支持集原型，形状 [ways, embed_dim]
                    query_labels: 查询集标签，形状 [num_query]
                    temperature: 温度参数，控制相似度分布

                Returns:
                    对比损失的平均值
                """
                # 归一化嵌入
                query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
                support_prototypes = F.normalize(support_prototypes, p=2, dim=-1)

                # 计算相似度矩阵：query_embeddings 与 support_prototypes 的点积
                # 形状 [num_query, ways]
                sim_matrix = torch.matmul(query_embeddings, support_prototypes.T) / temperature

                # InfoNCE 损失
                loss = F.cross_entropy(sim_matrix, query_labels)

                return loss
                # 新增：计算对比学习损失
            contrastive_loss_value = contrastive_loss(
                query_embeddings=query_embeddings,
                support_prototypes=support_prototypes,
                query_labels=query_labels,
                temperature=0.2  # 可调参数
            )
            # support_prototypes_1 = outputs["support_prototypes"]
            # logits_1 = (
            #         self.metric(query_embeddings, support_prototypes_1)
            #         / self.args.logit_temperature
            # )
            # losses["cross_entropy_1"] = F.cross_entropy(logits_1, query_labels)
            proto_reg_loss = self.prototype_regularization_loss(support_prototypes, margin=0.5)
            logits = self.metric(query_embeddings, support_prototypes) / self.args.logit_temperature
            ce_loss = F.cross_entropy(logits, query_labels)

            # ========== 优化后的CLIP对比损失 ==========
            # distill_clip_loss=clip_knowledge_distillation(query_embeddings=query_embeddings,batch_data=data,labels_id=labels_id,query_labels=query_labels,temperature=0.2)
            # print("ce_loss:")
            # print(ce_loss)
            losses = {
                "cross_entropy": ce_loss,
                "proto_regularization": self.args.proto_reg_weight * proto_reg_loss,
                "contrastive_loss_value":contrastive_loss_value
            }
            accs = {"accuracy": accuracy(logits, query_labels)}
        # Match query embedding with support embeddings
        elif self.args.comp_method == "match":
            # Create one-hot labels for support set
            one_hot_support_labels = torch.zeros(len(support_labels), ways).to(
                self.device
            )
            one_hot_support_labels[np.arange(len(support_labels)), support_labels] = 1

            # Normalize embeddings and compute cosine distance, then softmax
            support_embeddings_unit = support_embeddings / torch.norm(
                support_embeddings, dim=1, keepdim=True
            )
            query_embeddings_unit = query_embeddings / torch.norm(
                query_embeddings, dim=1, keepdim=True
            )
            similarities = F.softmax(
                query_embeddings_unit @ support_embeddings_unit.T, dim=1
            )

            # Output probability is the similarity scores multiplied with the labels
            p_y = similarities @ one_hot_support_labels

            accs["accuracy"] = accuracy(p_y, query_labels)
            losses["cross_entropy"] = F.nll_loss(p_y.log(), query_labels)

        # Train linear classifier
        elif self.args.comp_method == "lin_cls":
            # Instantiate linear classifier
            lin_cls = nn.Linear(self.model.convnet.conv_out_dims, ways).to(self.device)
            opt = torch.optim.AdamW(lin_cls.parameters(), lr=1e-3, weight_decay=0.01)

            for i in range(101):
                opt.zero_grad()
                lin_y = lin_cls(support_embeddings)
                loss = F.cross_entropy(lin_y, support_labels)
                loss.backward()
                opt.step()

            lin_y_val = lin_cls(query_embeddings)
            accs["accuracy"] = accuracy(lin_y_val, query_labels)

        else:
            raise NotImplementedError

        # Auxiliary loss: protonet of conv embeddings
        if self.args.embed_proto_loss_weight > 0:
            embed_protos = gen_prototypes(support_embeddings, ways, shot)
            embed_protos_logits = metric(query_embeddings, embed_protos)
            embed_protos_loss = self.args.embed_proto_loss_weight * F.cross_entropy(
                embed_protos_logits, query_labels
            )
            losses["conv_embed_proto"] = embed_protos_loss

        # Binary Outlier loss: identify transformer embeddings as noisy samples or not
        if self.args.binary_outlier_loss_weight > 0:
            noise_labels = (
                torch.tensor(noise_positions[mask_indices["support"]])
                .unsqueeze(dim=1)
                .to(self.device)
            )
            accs["binary_outlier"] = accuracy(
                outputs["outlier_logits"], noise_labels, binary=True
            )
            # print("binary_outlier_loss:")
            # print(      self.args.binary_outlier_loss_weight
            #     * F.binary_cross_entropy_with_logits(
            #         outputs["outlier_logits"], noise_labels
            #     ))
            losses["binary_outlier"] = (
                self.args.binary_outlier_loss_weight
                * F.binary_cross_entropy_with_logits(
                    outputs["outlier_logits"], noise_labels
                )
            )


        return losses, accs

    def nearest_neighbor(
        self, batch, outlier_batch=None, support_label_noise_choices=[0]
    ):
        ways = self.args.train_way if self.model.training else self.args.test_way
        shot = self.args.train_shot if self.model.training else self.args.test_shot
        query_num = (
            self.args.train_query if self.model.training else self.args.test_query
        )

        assert self.args.comp_method.startswith("nearest")
        k = int(self.args.comp_method[8:])  # comp_method specified as "nearest_[k]"

        # Format data, place on device, add noise
        data, labels, mask_indices, noise_positions = self.prepare_data(
            batch, ways, shot, query_num, support_label_noise_choices, outlier_batch
        )

        # Compute support/query embeddings
        outputs = self.model(data, mask_indices["support"])
        support_embeddings = outputs["conv_embeddings"][mask_indices["support"]]
        query_embeddings = outputs["conv_embeddings"][mask_indices["query"]]

        # Note: support labels assumes class labels 0..(N-1), in order
        support_labels = torch.repeat_interleave(torch.arange(ways), shot).to(
            self.device
        )
        query_labels = labels[mask_indices["query"]].long()

        # Find distance between each query and all support samples
        distances = self.metric(query_embeddings, support_embeddings)

        # Sort queries by distance and find labels of closest samples
        sorted_idx = torch.argsort(distances, axis=1, descending=True)
        topk = support_labels[sorted_idx][:, :k]

        # Convert to one-hot
        one_hot_labels = torch.zeros((ways * query_num * k, ways)).to(self.device)
        one_hot_labels[np.arange(ways * query_num * k), topk.flatten()] = 1
        one_hot_labels = one_hot_labels.reshape(ways * query_num, k, ways)

        # Tally votes, add small noise as random tiebreaker
        tallied = (
            one_hot_labels.mean(dim=1)
            + torch.randn((ways * query_num, ways)).to(self.device) / 1e4
        )
        predictions = torch.argmax(tallied, dim=1)

        acc = (predictions == query_labels).sum().float() / query_labels.size(0)

        return {}, {"accuracy": acc}
