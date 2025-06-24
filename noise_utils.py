"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
import torch

from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import clip
# scheduler_path = "/home/wuzelei/Preprocess/stable-diffusion-2-base/stable-diffusion-2-base/models--stabilityai--stable-diffusion-2-base/snapshots/fa386bb446685d8ad8a8f06e732a66ad10be6f47/scheduler"
# model_path="/home/wuzelei/Preprocess/stable-diffusion-2-base/stable-diffusion-2-base/models--stabilityai--stable-diffusion-2-base/snapshots/fa386bb446685d8ad8a8f06e732a66ad10be6f47/"
# scheduler = EulerDiscreteScheduler.from_pretrained(scheduler_path)

OUTLIER = -1


def gen_derangement(n):
    """
    Generates a derangement (random permutation without any fixed points)
    """
    in_order = np.arange(n, dtype=np.int16)
    while True:
        derangement_candidate = np.random.permutation(n)
        if 0 not in derangement_candidate - in_order:
            break
    return derangement_candidate


def gen_swap_indices(ways, num_noise_samples, num_clean_shots, indices_to_change):
    shot = num_noise_samples // ways

    # Track unpicked options
    available = np.arange(num_noise_samples)
    available_KN = available.reshape(ways, shot)

    swap_indices = {}

    # permute for loop to avoid bias of class 0 picking first
    class_order = np.arange(ways)
    np.random.shuffle(class_order)
    for c in class_order:
        # Remove samples from class c from options
        available_c = [i for i in available if i not in available_KN[c]]

        # Track number of noise samples from each class
        noise_class_count = np.zeros(ways)

        swap_indices_c = []

        for i in range(len(indices_to_change[c])):
            # Random pick, if there are enough choices
            if len(available_c) == 0:
                return None

            # Pick random sample idx from available choices; remove from availabe options
            noise_choice_i = np.random.choice(available_c)
            swap_indices_c.append(noise_choice_i)
            available_c.remove(noise_choice_i)

            # Find class of picked sample and increment class counter
            noise_choice_class_i = noise_choice_i // shot
            noise_class_count[noise_choice_class_i] += 1

            # Limit noise samples per class to ensure clean class has plurality
            # If class has 1 less than the number of clean class, remove other samples from choices
            if (noise_class_count[noise_choice_class_i] + 1) >= num_clean_shots:
                available_c = [
                    i
                    for i in available_c
                    if i not in available_KN[noise_choice_class_i]
                ]

        swap_indices[c] = swap_indices_c

        # Update unpicked options
        available = np.setdiff1d(available, swap_indices[c])

    # Put swap indices back in order
    swap_indices = np.vstack([swap_indices[c] for c in range(ways)])

    return swap_indices


def gen_valid_swap_indices(ways, shot, indices_to_change):
    num_noise_samples = ways * shot
    num_clean_shots = shot - len(indices_to_change[0])

    while True:
        swap_indices = gen_swap_indices(
            ways, num_noise_samples, num_clean_shots, indices_to_change
        )

        if swap_indices is not None:
            break

    return swap_indices


def add_noise(
    data,
    labels,
    mask_indices,
    ways,
    noise_fraction,
    noise_type="sym_swap",
    outlier_data=None,
):
    """
    Adds either label swap noise (symmetric or paired) or outlier noise
    """
    # Calculate number of noisy samples
    num_idx = np.arange(data.shape[0], dtype=np.int16)[mask_indices["support"]]
    shot = int(len(num_idx) / ways)
    noise_num = int(round(noise_fraction * shot))

    # Select the indices of samples to noise
    indices_to_change = np.empty((0, noise_num), dtype=np.int16)
    for i in range(ways):
        class_i_idx = num_idx[i * shot : (i + 1) * shot]
        indices_to_change_i = np.random.choice(class_i_idx, noise_num, replace=False)
        indices_to_change = np.vstack((indices_to_change, indices_to_change_i))
    indices_to_change_flat = indices_to_change.flatten()

    # Make copy of data and labels
    noised_data = torch.clone(data)
    noised_labels = torch.clone(labels)

    # Replace clean data with noisy data at selected positions
    if noise_type == "sym_swap":
        # Swap data positions of selected samples, ensuring plurality of clean class
        swap_indices = gen_valid_swap_indices(ways, shot, indices_to_change)
        noised_data[indices_to_change_flat] = data[mask_indices["noise"]][
            swap_indices.flatten()
        ]
        noised_labels[indices_to_change_flat] = labels[mask_indices["noise"]][
            swap_indices.flatten()
        ]

    elif noise_type == "pair_swap":
        # Randomly select class pairs for noise
        drng = gen_derangement(ways)

        # Reorganize data and labels by class
        noisy_data = data[mask_indices["noise"]].reshape(
            ways, shot, data.shape[1], data.shape[2], data.shape[3]
        )
        noisy_labels = labels[mask_indices["noise"]].reshape(ways, shot)

        # Swap in noisy data from noisy samples
        for c in range(ways):
            noised_data[indices_to_change[c]] = noisy_data[drng[c]][:noise_num]
            noised_labels[indices_to_change[c]] = noisy_labels[drng[c]][:noise_num]

    elif noise_type == "outlier":
        # Randomly select outlier indices
        # Draw new indices rather than reuse indices_to_change so outlier classes don't line up
        outlier_select_indices = np.random.choice(
            len(outlier_data), len(indices_to_change_flat), replace=False
        )
        noised_data[indices_to_change_flat] = outlier_data[outlier_select_indices]
        noised_labels[indices_to_change_flat] = OUTLIER

    else:
        raise NotImplementedError

    # Location of noised data
    noise_positions = np.zeros(data.shape[0])
    noise_positions[indices_to_change_flat] = 1

    return noised_data, noised_labels, noise_positions


# class ClipDiffusionCleaner:
#     def __init__(self, device="cuda"):
#         # Load CLIP
#         self.clip_model,self.clip_processor = clip.load("pretrained/ViT-B-32.pt").to(device)
#         # self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
#
#         # Load Stable Diffusion
#         self.diffusion  = StableDiffusionPipeline.from_pretrained(model_path,scheduler=scheduler,torch_dtype=torch.float16).to("cuda")
#
#         self.device = device
#
#     def detect_noisy(self, images: list, labels: list, class_names: list, threshold=0.25):
#         """
#         输入：PIL 图像列表，对应的标签（0-N），class_names 是类别名称
#         输出：标记哪些图片为 noisy
#         """
#         prompts = [f"A photo of a {class_names[label]}" for label in labels]
#         inputs = self.clip_processor(text=prompts, images=images, return_tensors="pt", padding=True).to(self.device)
#
#         with torch.no_grad():
#             outputs = self.clip_model(**inputs)
#             logits_per_image = outputs.logits_per_image  # shape [B, 1]
#             probs = logits_per_image.softmax(dim=1)
#
#         sim_scores = logits_per_image.squeeze().cpu().numpy()  # 越低表示越不匹配
#         is_noisy = sim_scores < np.quantile(sim_scores, threshold)
#
#         return is_noisy
#
#     def replace_with_diffusion(self, labels, class_names):
#         """
#         输入标签列表，返回生成的图像（PIL）
#         """
#         prompts = [f"A photo of a {class_names[label]}" for label in labels]
#         generated_images = []
#         for prompt in prompts:
#             image = self.diffusion(prompt).images[0]
#             generated_images.append(image)
#         return generated_images
#
#     def clean_support_set(self, images, labels, class_names, mode="replace", threshold=0.25):
#         """
#         输入图像、标签、类别名，mode 可选 replace / mask
#         返回清洗后的图像列表，以及 noisy 的索引
#         """
#         is_noisy = self.detect_noisy(images, labels, class_names, threshold=threshold)
#         noisy_idx = np.where(is_noisy)[0]
#
#         if mode == "replace":
#             replacement_images = self.replace_with_diffusion([labels[i] for i in noisy_idx], class_names)
#             for i, idx in enumerate(noisy_idx):
#                 images[idx] = replacement_images[i]
#         elif mode == "mask":
#             # 只保留干净图像
#             images = [img for i, img in enumerate(images) if not is_noisy[i]]
#             labels = [lbl for i, lbl in enumerate(labels) if not is_noisy[i]]
#
#         return images, labels, noisy_idx