"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import clip
import learn2learn as l2l
import numpy as np
import torch
import torch.nn as nn

from few_shot_utils import gen_prototypes


class Convnet(nn.Module):
    def __init__(self, model="conv4", conv_proj_dim=None):
        super().__init__()
        if model == "conv4":
            self.encoder = l2l.vision.models.ConvBase(
               hidden=64, channels=3, max_pool=True
            )
            self.conv_out_dims = 1600
        else:
            raise NotImplementedError

        if conv_proj_dim:
            self.fc = nn.Linear(self.conv_out_dims, conv_proj_dim)
            self.conv_out_dims = conv_proj_dim

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        if hasattr(self, "fc"):
            x = self.fc(x)
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        ways,
        shot,
        num_layers,
        nhead,
        d_model,
        dim_feedforward,
        device,
        cls_type="cls_learn",
        pos_type="pos_learn",
        agg_method="mean",
        transformer_metric="dot_prod",
    ):
        super().__init__()
        self.ways = ways
        self.shot = shot

        self.cls_type = cls_type
        self.pos_type = pos_type
        self.agg_method = agg_method

        if self.cls_type == "cls_learn":
            self.cls_embeddings = nn.Embedding(
                max(ways["train"], ways["test"]), dim_feedforward
            )
        elif self.cls_type == "rand_const":
            self.cls_embeddings = nn.Embedding(
                max(ways["train"], ways["test"]), dim_feedforward
            ).requires_grad_(False)

        if self.pos_type == "pos_learn":
            self.pos_embeddings = nn.Embedding(
                max(ways["train"], ways["test"]), dim_feedforward
            )
        elif self.pos_type == "rand_const":
            self.pos_embeddings = nn.Embedding(
                max(ways["train"], ways["test"]), dim_feedforward
            ).requires_grad_(False)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout=0.1,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.device = device

    def forward(self, x):
        ways = self.ways["train"] if self.training else self.ways["test"]
        shot = self.shot["train"] if self.training else self.shot["test"]

        n_arng = torch.arange(ways, device=self.device)

        # Concatenate cls tokens with support embeddings
        if self.cls_type in ["cls_learn", "rand_const"]:
            cls_tokens = self.cls_embeddings(n_arng)  # (ways, dim)
        elif self.cls_type == "proto":
            cls_tokens = gen_prototypes(x, ways, shot, self.agg_method)  # (ways, dim)
        else:
            raise NotImplementedError

        cls_sup_embeds = torch.cat((cls_tokens, x), dim=0)  # (ways*(shot+1), dim)
        cls_sup_embeds = torch.unsqueeze(
            cls_sup_embeds, dim=1
        )  # (ways*(shot+1), BS, dim)

        # Position embeddings based on class ID
        pos_idx = torch.cat((n_arng, torch.repeat_interleave(n_arng, shot)))
        pos_tokens = torch.unsqueeze(
            self.pos_embeddings(pos_idx), dim=1
        )  # (ways*(shot+1), BS, dim)

        # Inputs combined with position encoding
        transformer_input = cls_sup_embeds + pos_tokens

        return self.encoder(transformer_input)


class BinaryOutlierDetector(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.dim = dim
        self.fc = nn.Linear(self.dim, 1)

    def forward(self, x):
        return self.fc(x)


class ProtoConvnet(nn.Module):
    def __init__(
        self,
        ways,
        shot,
        model="conv4",
        conv_proj_dim=None,
        agg_method="mean",
        binary_outlier_detection=False,
    ):
        super().__init__()

        self.convnet = Convnet(model, conv_proj_dim)

        self.binary_outlier_detection = binary_outlier_detection
        if self.binary_outlier_detection:
            self.binary_outlier_detector = BinaryOutlierDetector(
                self.convnet.conv_out_dims
            )

        self.ways = ways
        self.shot = shot
        self.agg_method = agg_method

    def forward(self, x, support_indices):
        ways = self.ways["train"] if self.training else self.ways["test"]
        shot = self.shot["train"] if self.training else self.shot["test"]

        outputs = {}

        # Embed inputs with conv
        embeddings = self.convnet(x)
        outputs["conv_embeddings"] = embeddings

        # Calculate prototypes
        support_embeddings = embeddings[support_indices]
        outputs["support_prototypes"] = gen_prototypes(
            support_embeddings, ways, shot, self.agg_method
        )

        if self.binary_outlier_detection:
            outputs["outlier_logits"] = self.binary_outlier_detector(support_embeddings)

        return outputs

class ProtoConvTransformer(nn.Module):
    def __init__(
        self,
        ways,
        shot,
        device,
        conv_model="conv4",
        conv_proj_dim=None,
        trans_layers=1,
        trans_d_model=None,
        nhead=8,
        ortho_proj=False,
        ortho_proj_residual=False,
        cls_type="cls_learn",
        pos_type="pos_learn",
        agg_method="mean",
        transformer_metric="dot_prod",
        output_from_cls=True,
        binary_outlier_detection=False,
    ):
        super().__init__()

        self.convnet = Convnet(conv_model, conv_proj_dim)
        if trans_d_model is None:
            trans_d_model = self.convnet.conv_out_dims
        self.transformer = Transformer(
            ways,
            shot,
            trans_layers,
            nhead,
            trans_d_model,
            trans_d_model,
            device,
            cls_type,
            pos_type,
            transformer_metric=transformer_metric,
        )

        self.ortho_proj = ortho_proj
        if self.ortho_proj:
            self.proj_trans_in = nn.Parameter(
                torch.nn.init.orthogonal_(
                    torch.empty(self.convnet.conv_out_dims, trans_d_model)
                )
            )
            self.proj_trans_out = nn.Parameter(self.proj_trans_in.data.detach())
        self.ortho_proj_residual = ortho_proj_residual

        self.out_dim = self.convnet.conv_out_dims if self.ortho_proj else trans_d_model

        self.binary_outlier_detection = binary_outlier_detection
        if self.binary_outlier_detection:
            self.binary_outlier_detector = BinaryOutlierDetector(self.out_dim)

        self.ways = ways
        self.shot = shot

        self.agg_method = agg_method
        self.output_from_cls = output_from_cls

        self.device = device

    def forward(self, x, support_indices):
        ways = self.ways["train"] if self.training else self.ways["test"]
        shot = self.shot["train"] if self.training else self.shot["test"]

        outputs = {}

        # Embed inputs with conv
        embeddings = self.convnet(x)
        outputs["conv_embeddings"] = embeddings

        # Process embeddings as a sequence input to the transformer, projecting before and after if necessary
        transformer_input = (
            embeddings[support_indices] @ self.proj_trans_in
            if self.ortho_proj
            else embeddings[support_indices]
        )
        transformer_output = self.transformer(transformer_input).squeeze(
            1
        )  # squeeze out batch size dim
        if self.ortho_proj:
            transformer_output = transformer_output @ self.proj_trans_out.T

            if self.ortho_proj_residual:
                conv_support_prototypes = gen_prototypes(
                    embeddings[support_indices], ways, shot, self.agg_method
                )
                input_skip = torch.cat(
                    [conv_support_prototypes, embeddings[support_indices]], dim=0
                )
                transformer_output = input_skip + transformer_output
        outputs["trans_output"] = transformer_output

        if self.output_from_cls:
            # first #ways outputs in seq correspond to the cls tokens, and thus the prototypes
            outputs["support_prototypes"] = transformer_output[:ways]
        else:
            # Aggregate prototypes from support sample positions of each class
            outputs["support_prototypes"] = gen_prototypes(
                transformer_output[ways:], ways, shot, self.agg_method
            )

        if self.binary_outlier_detection:
            outputs["outlier_logits"] = self.binary_outlier_detector(
                transformer_output[ways:]
            )

        return outputs
class ProtoCLIP(nn.Module):
    def __init__(
        self,
        ways,
        shot,
        clip_model="ViT-B/32",  # CLIP模型类型
        conv_proj_dim=None,
        agg_method="mean",
        binary_outlier_detection=False,
    ):
        super().__init__()

        # 加载CLIP模型
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, _ = clip.load(clip_model, device=self.device)

        # Project CLIP的输出维度到指定维度
        if conv_proj_dim:
            self.projector = nn.Linear(self.clip_model.visual.output_dim, conv_proj_dim)
        else:
            self.projector = None

        # Outlier 检测
        self.binary_outlier_detection = binary_outlier_detection
        if self.binary_outlier_detection:
            self.binary_outlier_detector = BinaryOutlierDetector(self.clip_model.visual.output_dim)

        self.ways = ways
        self.shot = shot
        self.agg_method = agg_method

    def encode_image(self, images):
        with torch.no_grad():
            features = self.clip_model.encode_image(images)  # 获取CLIP特征
        if self.projector:
            features = self.projector(features)  # 如果有投影，进行投影
        return features

    def forward(self, x, support_indices):
        ways = self.ways["train"] if self.training else self.ways["test"]
        shot = self.shot["train"] if self.training else self.shot["test"]

        outputs = {}

        # 获取图像的特征
        embeddings = self.encode_image(x)
        outputs["clip_embeddings"] = embeddings

        # 计算原型
        support_embeddings = embeddings[support_indices]
        outputs["support_prototypes"] = gen_prototypes(
            support_embeddings, ways, shot, self.agg_method
        )

        if self.binary_outlier_detection:
            outputs["outlier_logits"] = self.binary_outlier_detector(support_embeddings)

        return outputs

class ProtoCLIPTransformer(nn.Module):
    def __init__(
        self,
        ways,
        shot,
        device,
        clip_model="/home/wuzelei/Preprocess/pretrained/ViT-B-32.pt",  # CLIP模型类型
        conv_proj_dim=None,
        trans_layers=1,
        trans_d_model=None,
        nhead=8,
        ortho_proj=False,
        ortho_proj_residual=False,
        cls_type="cls_learn",
        pos_type="pos_learn",
        agg_method="mean",
        transformer_metric="dot_prod",
        output_from_cls=True,
        binary_outlier_detection=False,
    ):
        super().__init__()

        # 加载CLIP模型
        self.device = device
        self.clip_model, _ = clip.load(clip_model, device=device)

        if trans_d_model is None:
            trans_d_model = self.clip_model.visual.output_dim  # 使用CLIP的输出维度作为Transformer的输入维度
        self.transformer = Transformer(
            ways,
            shot,
            trans_layers,
            nhead,
            trans_d_model,
            trans_d_model,
            device,
            cls_type,
            pos_type,
            transformer_metric=transformer_metric,
        )

        self.ortho_proj = ortho_proj
        if self.ortho_proj:
            self.proj_trans_in = nn.Parameter(
                torch.nn.init.orthogonal_(
                    torch.empty(self.clip_model.visual.output_dim, trans_d_model)
                )
            )
            self.proj_trans_out = nn.Parameter(self.proj_trans_in.data.detach())
        self.ortho_proj_residual = ortho_proj_residual

        self.out_dim = self.clip_model.visual.output_dim if self.ortho_proj else trans_d_model

        self.binary_outlier_detection = binary_outlier_detection
        if self.binary_outlier_detection:
            self.binary_outlier_detector = BinaryOutlierDetector(self.out_dim)

        self.ways = ways
        self.shot = shot

        self.agg_method = agg_method
        self.output_from_cls = output_from_cls

        self.device = device

    def encode_image(self, images):
        print(images.shape)
        with torch.no_grad():
            features = self.clip_model.encode_image(images)  # 获取CLIP特征
        return features

    def forward(self, x, support_indices):
        ways = self.ways["train"] if self.training else self.ways["test"]
        shot = self.shot["train"] if self.training else self.shot["test"]

        outputs = {}

        # 获取图像的特征
        embeddings = self.encode_image(x)
        outputs["clip_embeddings"] = embeddings

        # 处理Transformer输入
        transformer_input = (
            embeddings[support_indices] @ self.proj_trans_in
            if self.ortho_proj
            else embeddings[support_indices]
        )
        transformer_output = self.transformer(transformer_input).squeeze(1)  # 去除batch维度
        if self.ortho_proj:
            transformer_output = transformer_output @ self.proj_trans_out.T

            if self.ortho_proj_residual:
                conv_support_prototypes = gen_prototypes(
                    embeddings[support_indices], ways, shot, self.agg_method
                )
                input_skip = torch.cat(
                    [conv_support_prototypes, embeddings[support_indices]], dim=0
                )
                transformer_output = input_skip + transformer_output
        outputs["trans_output"] = transformer_output

        if self.output_from_cls:
            # 使用CLS token来做分类输出
            outputs["support_prototypes"] = transformer_output[:ways]
        else:
            # 使用聚合方法从支持样本中提取原型
            outputs["support_prototypes"] = gen_prototypes(
                transformer_output[ways:], ways, shot, self.agg_method
            )

        if self.binary_outlier_detection:
            outputs["outlier_logits"] = self.binary_outlier_detector(
                transformer_output[ways:]
            )

        return outputs
class ProtoCLIPTransformer(nn.Module):
    def __init__(
        self,
        ways,
        shot,
        device,
        clip_model="/home/wuzelei/Preprocess/pretrained/ViT-B-32.pt",  # CLIP模型类型
        conv_proj_dim=None,
        trans_layers=1,
        trans_d_model=None,
        nhead=8,
        ortho_proj=False,
        ortho_proj_residual=False,
        cls_type="cls_learn",
        pos_type="pos_learn",
        agg_method="mean",
        transformer_metric="dot_prod",
        output_from_cls=True,
        binary_outlier_detection=False,
    ):
        super().__init__()

        # 加载CLIP模型
        self.device = device
        self.clip_model, _ = clip.load(clip_model, device=device)
        # self.device = device
        # self.clip_model, _ = clip.load(clip_model, device=device)

        # 冻结CLIP的所有参数
        for param in self.clip_model.parameters():
            param.requires_grad = False
        if trans_d_model is None:
            trans_d_model = self.clip_model.visual.output_dim  # 使用CLIP的输出维度作为Transformer的输入维度
        self.transformer = Transformer(
            ways,
            shot,
            trans_layers,
            nhead,
            trans_d_model,
            trans_d_model,
            device,
            cls_type,
            pos_type,
            transformer_metric=transformer_metric,
        )

        self.ortho_proj = ortho_proj
        if self.ortho_proj:
            self.proj_trans_in = nn.Parameter(
                torch.nn.init.orthogonal_(
                    torch.empty(self.clip_model.visual.output_dim, trans_d_model)
                )
            )
            self.proj_trans_out = nn.Parameter(self.proj_trans_in.data.detach())
        self.ortho_proj_residual = ortho_proj_residual

        self.out_dim = self.clip_model.visual.output_dim if self.ortho_proj else trans_d_model

        self.binary_outlier_detection = binary_outlier_detection
        if self.binary_outlier_detection:
            self.binary_outlier_detector = BinaryOutlierDetector(self.out_dim)

        self.ways = ways
        self.shot = shot

        self.agg_method = agg_method
        self.output_from_cls = output_from_cls

        self.device = device

    def encode_image(self, images):
        # print(f"Input image shape: {images.shape}")  # 打印输入图像的形状
        features = self.clip_model.encode_image(images)
        # print(f"Features shape after encoding: {features.shape}")  # 打印特征的形状
        return features

    def forward(self, x, support_indices):
        ways = self.ways["train"] if self.training else self.ways["test"]
        shot = self.shot["train"] if self.training else self.shot["test"]

        outputs = {}

        # 获取图像的特征
        embeddings = self.encode_image(x)
        outputs["conv_embeddings"] = embeddings

        # 处理Transformer输入
        transformer_input = (
            embeddings[support_indices] @ self.proj_trans_in.to(embeddings.dtype)
            if self.ortho_proj
            else embeddings[support_indices]
        )
        transformer_output = self.transformer(transformer_input).squeeze(1)  # 去除batch维度
        if self.ortho_proj:
            transformer_output = transformer_output @ self.proj_trans_out.T

            if self.ortho_proj_residual:
                conv_support_prototypes = gen_prototypes(
                    embeddings[support_indices], ways, shot, self.agg_method
                )
                input_skip = torch.cat(
                    [conv_support_prototypes, embeddings[support_indices]], dim=0
                )
                transformer_output = input_skip + transformer_output
        outputs["trans_output"] = transformer_output

        if self.output_from_cls:
            # 使用CLS token来做分类输出
            outputs["support_prototypes"] = transformer_output[:ways]
        else:
            # 使用聚合方法从支持样本中提取原型
            outputs["support_prototypes"] = gen_prototypes(
                transformer_output[ways:], ways, shot, self.agg_method
            )

        if self.binary_outlier_detection:
            outputs["outlier_logits"] = self.binary_outlier_detector(
                transformer_output[ways:]
            )

        return outputs