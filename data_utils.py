# """
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# """
#
# import os
#
# import numpy as np
# import torch
# from torch.utils.data import DataLoader
# from torchvision import transforms
#
# import learn2learn as l2l
# from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels
# from learn2learn.vision.datasets import TieredImagenet
#
# from mini_imagenet import MiniImagenet, MiniImagenetOutlier
#
#
# IMAGENET_MEANS = (0.485, 0.456, 0.406)
# IMAGENET_STDS = (0.229, 0.224, 0.225)
# IMAGENET_SIZE = (84, 84)
#
#
# def get_augmentation_transforms(args):
#     transform_list = []
#
#     if args.dataset == "miniimagenet":
#         transform_list.append(transforms.ToPILImage())
#
#     if args.random_horizontal_flip:
#         transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
#
#     if args.random_resized_crop:
#         if "imagenet" in args.dataset:
#             transform_list.append(
#                 transforms.RandomResizedCrop(size=IMAGENET_SIZE, scale=(0.8, 1.0))
#             )
#         else:
#             raise NotImplementedError
#
#     if args.color_jitter:
#         transform_list.append(
#             transforms.ColorJitter(
#                 brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
#             )
#         )
#
#     transform_list.append(transforms.ToTensor())
#
#     if "imagenet" in args.dataset:
#         transform_list.append(transforms.Normalize(IMAGENET_MEANS, IMAGENET_STDS))
#     else:
#         raise NotImplementedError
#
#     if args.random_erasing:
#         transform_list.append(transforms.RandomErasing(value="random"))
#
#     return transforms.Compose(transform_list)
#
#
# def get_data_loaders(args):
#     if "imagenet" in args.dataset:
#         normalize_transform = transforms.Normalize(IMAGENET_MEANS, IMAGENET_STDS)
#     else:
#         raise NotImplementedError
#
#     if args.dataset == "miniimagenet":
#         train_dataset = MiniImagenet(
#             root=args.data_path,
#             mode="train",
#             transform=get_augmentation_transforms(args),  # 使用数据增强
#         )
#         valid_dataset = MiniImagenet(
#             root=args.data_path,
#             mode="validation",
#             transform=transforms.Compose([transforms.ToTensor(), normalize_transform]),
#         )
#         test_dataset = MiniImagenet(
#             root=args.data_path,
#             mode="test",
#             transform=transforms.Compose([transforms.ToTensor(), normalize_transform]),
#         )
#     elif args.dataset == "tieredimagenet":
#         train_dataset = TieredImagenet(
#             root=args.data_path,
#             mode="train",
#             transform=get_augmentation_transforms(args),
#         )
#         valid_dataset = TieredImagenet(
#             root=args.data_path,
#             mode="validation",
#             transform=transforms.Compose([transforms.ToTensor(), normalize_transform]),
#         )
#         test_dataset = TieredImagenet(
#             root=args.data_path,
#             mode="test",
#             transform=transforms.Compose([transforms.ToTensor(), normalize_transform]),
#         )
#     else:
#         raise NotImplementedError
#
#     if args.noise_type == "outlier" and "imagenet" in args.dataset:
#         outlier_train_dataset = MiniImagenetOutlier(
#             root=os.path.join(args.data_path, "../miniimagenet_outlier"),
#             mode="train",
#             transform=transforms.Compose([transforms.ToTensor(), normalize_transform]),
#         )
#         outlier_test_dataset = MiniImagenetOutlier(
#             root=os.path.join(args.data_path, "../miniimagenet_outlier"),
#             mode="test",
#             transform=transforms.Compose([transforms.ToTensor(), normalize_transform]),
#         )
#
#     train_dataset = l2l.data.MetaDataset(train_dataset)
#     train_transforms = [
#         NWays(train_dataset, args.train_way),
#         KShots(train_dataset, args.train_query + 2 * args.train_shot),
#         LoadData(train_dataset),
#         RemapLabels(train_dataset),
#     ]
#     train_tasks = l2l.data.TaskDataset(train_dataset, task_transforms=train_transforms)
#     train_loader = DataLoader(train_tasks, pin_memory=True, shuffle=True)
#
#     valid_dataset = l2l.data.MetaDataset(valid_dataset)
#     valid_transforms = [
#         NWays(valid_dataset, args.test_way),
#         KShots(valid_dataset, args.test_query + 2 * args.test_shot),
#         LoadData(valid_dataset),
#         RemapLabels(valid_dataset),
#     ]
#     valid_tasks = l2l.data.TaskDataset(
#         valid_dataset, task_transforms=valid_transforms, num_tasks=200
#     )
#     valid_loader = DataLoader(valid_tasks, pin_memory=True, shuffle=True)
#
#     test_dataset = l2l.data.MetaDataset(test_dataset)
#     test_transforms = [
#         NWays(test_dataset, args.test_way),
#         KShots(test_dataset, args.test_query + 2 * args.test_shot),
#         LoadData(test_dataset),
#         RemapLabels(test_dataset),
#     ]
#     test_tasks = l2l.data.TaskDataset(
#         test_dataset, task_transforms=test_transforms, num_tasks=args.test_tasks
#     )
#     test_loader = DataLoader(test_tasks, pin_memory=True, shuffle=True)
#
#     if args.noise_type == "outlier":
#         outlier_train_loader = DataLoader(
#             outlier_train_dataset,
#             batch_size=(args.train_way * args.train_shot),
#             pin_memory=True,
#             shuffle=True,
#         )
#         outlier_test_loader = DataLoader(
#             outlier_test_dataset,
#             batch_size=(args.test_way * args.test_shot),
#             pin_memory=True,
#             shuffle=True,
#         )
#     else:
#         outlier_train_loader = None
#         outlier_test_loader = None
#
#     return {
#         "train": train_loader,
#         "valid": valid_loader,
#         "test": test_loader,
#         "outlier_train": outlier_train_loader,
#         "outlier_test": outlier_test_loader,
#     }
#
#
# #####################
# # Data Pre-processing
# #####################
#
#
# def preprocess_data_labels(batch, device):
#     # data, labels,labels_y = batch
#     # data = data.to(device)
#     # labels = labels.to(device)
#     #
#     # # Sort data samples by labels
#     # # TODO: Can this be replaced by ConsecutiveLabels ?
#     # sort = torch.sort(labels)
#     # data = data.squeeze(0)[sort.indices].squeeze(0)
#     # labels = labels.squeeze(0)[sort.indices].squeeze(0)
#     data, labels,label_id = batch
#     data = data.to(device)
#     labels = labels.to(device)
#
#     # 排序
#     sort = torch.sort(labels)
#     sort_indices = sort.indices
#     data = data.squeeze(0)[sort_indices].squeeze(0)
#     labels = labels.squeeze(0)[sort_indices].squeeze(0)
#     # labels_y=labels_y[0]
#     # 排序 label_y（类名字符串）
#     sort_indices = sort_indices.view(-1)  # 或者 flatten()
#     # labels_y = [labels_y[i] for i in sort_indices.tolist()]
#
#     return data, labels,label_id
#     # return data, labels,labels_y
#
#
# def get_support_noise_query_indices(ways, shot, query_num):
#     # Init support and noise arrays
#     support_indices = np.zeros(ways * (2 * shot + query_num), dtype=bool)
#     noise_indices = support_indices.copy()
#
#     # Marker for beginning of each of the ways
#     selection = np.arange(ways) * (2 * shot + query_num)
#
#     # Mark support indices, starting from the beginning of each class
#     for offset in range(shot):
#         support_indices[selection + offset] = True
#         noise_indices[selection + shot + offset] = True
#     # Query indices are those that aren't support or noise
#     query_indices = ~(support_indices | noise_indices)
#
#     # Convert to torch
#     return {
#         "support": torch.from_numpy(support_indices),
#         "noise": torch.from_numpy(noise_indices),
#         "query": torch.from_numpy(query_indices),
#     }
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import learn2learn as l2l
from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels
from learn2learn.vision.datasets import TieredImagenet
from torchvision.datasets import ImageFolder

from mini_imagenet import MiniImagenet, MiniImagenetOutlier

# IMAGENET的均值和标准差
IMAGENET_MEANS = (0.485, 0.456, 0.406)
IMAGENET_STDS = (0.229, 0.224, 0.225)
IMAGENET_SIZE = (224, 224)
class ImageFolderWithLabelID(ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]  # target 是 class_to_idx 映射后的数字
        sample = self.loader(path)          # 加载图像

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        label_id = self.classes[target]  # 这是原始类名，比如 'dog'
        return sample, target, label_id  # 返回三元组
# 数据增强变换
def get_augmentation_transforms(args):
    transform_list = []

    if args.dataset == "miniimagenet":
        transform_list.append(transforms.ToPILImage())  # 转换为PIL图片格式

    if args.random_horizontal_flip:
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))  # 随机水平翻转

    if args.random_resized_crop:
        if "imagenet" in args.dataset:
            transform_list.append(
                transforms.RandomResizedCrop(size=IMAGENET_SIZE, scale=(0.8, 1.0))  # 随机裁剪
            )
        else:
            raise NotImplementedError

    if args.color_jitter:
        transform_list.append(
            transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1  # 颜色抖动
            )
        )

    transform_list.append(transforms.ToTensor())  # 转换为Tensor

    if "imagenet" in args.dataset:
        transform_list.append(transforms.Normalize(IMAGENET_MEANS, IMAGENET_STDS))  # 归一化
    else:
        raise NotImplementedError

    if args.random_erasing:
        transform_list.append(transforms.RandomErasing(value="random"))  # 随机擦除

    return transforms.Compose(transform_list)


# 数据加载器
def get_data_loaders(args):
    if "imagenet" in args.dataset:
        normalize_transform = transforms.Normalize(IMAGENET_MEANS, IMAGENET_STDS)
    else:
        raise NotImplementedError

    # 加载数据集
    print(args.dataset)
    if args.dataset == "miniimagenet":
        train_dataset = MiniImagenet(
            root=args.data_path,
            mode="train",
            transform=get_augmentation_transforms(args),  # 使用数据增强
        )
        valid_dataset = MiniImagenet(
            root=args.data_path,
            mode="validation",
            transform=transforms.Compose([
                transforms.ToPILImage(),  # 确保将图像转换为PIL格式
                transforms.Resize((224, 224)),  # 调整大小
                transforms.ToTensor(),  # 转换为Tensor
                normalize_transform,  # 标准化
            ]),
        )

        test_dataset = MiniImagenet(
            root=args.data_path,
            mode="test",
            transform=transforms.Compose([
                transforms.ToPILImage(),  # 确保将图像转换为PIL格式
                transforms.Resize((224, 224)),  # 调整大小
                transforms.ToTensor(),  # 转换为Tensor
                normalize_transform,  # 标准化
            ]),

    )
    elif args.dataset == "tieredimagenet":
        train_dataset = ImageFolderWithLabelID(root='/home/wuzelei/noisy_fewshot_learning/noisy_few_shot-main/data/tiered_imagenet/train', transform=transforms.Compose([
                transforms.Resize((224, 224)),  # 调整大小
                transforms.ToTensor(),  # 转换为Tensor
                normalize_transform,  # 标准化
            ]))

        # 加载验证集
        valid_dataset = ImageFolderWithLabelID(root='/home/wuzelei/noisy_fewshot_learning/noisy_few_shot-main/data/tiered_imagenet/val', transform=transforms.Compose([
                transforms.Resize((224, 224)),  # 调整大小
                transforms.ToTensor(),  # 转换为Tensor
                normalize_transform,  # 标准化
            ]))

        # 加载测试集
        test_dataset = ImageFolderWithLabelID(root='/home/wuzelei/noisy_fewshot_learning/noisy_few_shot-main/data/tiered_imagenet/test', transform=transforms.Compose([
                transforms.Resize((224, 224)),  # 调整大小
                transforms.ToTensor(),  # 转换为Tensor
                normalize_transform,  # 标准化
            ]))


    else:
        raise NotImplementedError

    # 处理噪声数据集
    if args.noise_type == "outlier" and "imagenet" in args.dataset:
        outlier_train_dataset = MiniImagenetOutlier(
            root=os.path.join(args.data_path, "../miniimagenet_outlier"),
            mode="train",
            transform=transforms.Compose([transforms.ToTensor(), normalize_transform]),
        )
        outlier_test_dataset = MiniImagenetOutlier(
            root=os.path.join(args.data_path, "../miniimagenet_outlier"),
            mode="test",
            transform=transforms.Compose([transforms.ToTensor(), normalize_transform]),
        )

    # 创建任务数据集
    train_dataset = l2l.data.MetaDataset(train_dataset)
    train_transforms = [
        NWays(train_dataset, args.train_way),
        KShots(train_dataset, args.train_query + 2 * args.train_shot),
        LoadData(train_dataset),
        RemapLabels(train_dataset),
    ]
    train_tasks = l2l.data.TaskDataset(train_dataset, task_transforms=train_transforms)
    train_loader = DataLoader(train_tasks, pin_memory=True, shuffle=True)

    # 验证集数据加载
    valid_dataset = l2l.data.MetaDataset(valid_dataset)
    valid_transforms = [
        NWays(valid_dataset, args.test_way),
        KShots(valid_dataset, args.test_query + 2 * args.test_shot),
        LoadData(valid_dataset),
        RemapLabels(valid_dataset),
    ]
    valid_tasks = l2l.data.TaskDataset(valid_dataset, task_transforms=valid_transforms, num_tasks=200)
    valid_loader = DataLoader(valid_tasks, pin_memory=True, shuffle=True)

    # 测试集数据加载
    test_dataset = l2l.data.MetaDataset(test_dataset)
    test_transforms = [
        NWays(test_dataset, args.test_way),
        KShots(test_dataset, args.test_query + 2 * args.test_shot),
        LoadData(test_dataset),
        RemapLabels(test_dataset),
    ]
    test_tasks = l2l.data.TaskDataset(test_dataset, task_transforms=test_transforms, num_tasks=args.test_tasks)
    test_loader = DataLoader(test_tasks, pin_memory=True, shuffle=True)

    # 处理噪声数据加载
    if args.noise_type == "outlier":
        outlier_train_loader = DataLoader(
            outlier_train_dataset,
            batch_size=(args.train_way * args.train_shot),
            pin_memory=True,
            shuffle=True,
        )
        outlier_test_loader = DataLoader(
            outlier_test_dataset,
            batch_size=(args.test_way * args.test_shot),
            pin_memory=True,
            shuffle=True,
        )
    else:
        outlier_train_loader = None
        outlier_test_loader = None

    return {
        "train": train_loader,
        "valid": valid_loader,
        "test": test_loader,
        "outlier_train": outlier_train_loader,
        "outlier_test": outlier_test_loader,
    }


#########################
# 数据预处理标签
#########################
def preprocess_data_labels(batch, device):
    data, labels, label_id = batch
    data = data.to(device)
    labels = labels.to(device)

    # 对数据按标签进行排序
    sort = torch.sort(labels)
    sort_indices = sort.indices
    data = data.squeeze(0)[sort_indices].squeeze(0)
    labels = labels.squeeze(0)[sort_indices].squeeze(0)

    sort_indices = sort_indices.view(-1)  # 展平
    return data, labels, label_id


# 获取支持集、噪声集和查询集的索引
def get_support_noise_query_indices(ways, shot, query_num):
    support_indices = np.zeros(ways * (2 * shot + query_num), dtype=bool)
    noise_indices = support_indices.copy()

    # 标记每个类别的开始位置
    selection = np.arange(ways) * (2 * shot + query_num)

    for offset in range(shot):
        support_indices[selection + offset] = True
        noise_indices[selection + shot + offset] = True

    query_indices = ~(support_indices | noise_indices)

    return {
        "support": torch.from_numpy(support_indices),
        "noise": torch.from_numpy(noise_indices),
        "query": torch.from_numpy(query_indices),
    }

