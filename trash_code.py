class MetaTrainer:
    def __init__(self, model, device, args):
        self.model = model
        self.device = device
        self.args = args

        if args.dist_metric == "euclidean":
            self.metric = pairwise_distances_logits
        elif args.dist_metric == "cosine":
            self.metric = pairwise_cosine_logits
        else:
            raise NotImplementedError
            # 1) 初始化调度器
            scheduler = EulerDiscreteScheduler.from_pretrained(self.args.sd_scheduler_path)

            # 2) 加载并缓存 pipeline
            self.sd_pipe = StableDiffusionPipeline.from_pretrained(
                self.args.sd_model_path,
                scheduler=scheduler,
                torch_dtype=torch.float16
            ).to(self.device)

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
        data, labels, labels_y = preprocess_data_labels(batch, self.device)
        if outlier_batch:
            outlier_data, _ = preprocess_data_labels(outlier_batch, self.device)
        else:
            outlier_data = None

        # Separate query and support
        mask_indices = get_support_noise_query_indices(ways, shot, query_num)

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
        ##new
        if isinstance(labels_y, list):
            labels_y = [labels_y[i] for i in (~mask_indices["noise"]).nonzero(as_tuple=True)[0].tolist()]
        else:  # 如果是 tensor（比如 one-hot 编码）
            labels_y = labels_y[~mask_indices["noise"]]
        return data, labels, mask_indices, noise_positions, labels_y

    def loss_acc(self, batch, outlier_batch=None, support_label_noise_choices=[0]):
        ways = self.args.train_way if self.model.training else self.args.test_way
        shot = self.args.train_shot if self.model.training else self.args.test_shot
        query_num = (
            self.args.train_query if self.model.training else self.args.test_query
        )

        # Format data, place on device, add noise
        data, labels, mask_indices, noise_positions, labels_y = self.prepare_data(
            batch, ways, shot, query_num, support_label_noise_choices, outlier_batch
        )
        correct_labels, incorrect_labels, correct_scores, incorrect_scores, correct_indices, incorrect_indices, similarity_scores, incorrect_prompts = compute_clip_similarity_from_tensor(
            data, labels_y, "./map.json")
        # print(incorrect_prompts)
        # print(correct_labels, incorrect_labels, correct_scores, incorrect_scores, correct_indices, incorrect_indices, similarity_scores)
        # print(similarity_scores)
        # Compute support/query embeddings and prototypes
        outputs = self.model(data, mask_indices["support"])
        support_embeddings = outputs["conv_embeddings"][mask_indices["support"]]#经过conv4的支持集的特征
        query_embeddings = outputs["conv_embeddings"][mask_indices["query"]]#查询集的特征

        # Note: support labels assumes class labels 0..(N-1), in order. labels has original labels before noise
        support_labels = torch.repeat_interleave(torch.arange(ways), shot).to(
            self.device
        )
        query_labels = labels[mask_indices["query"]].long()

        losses = {}
        accs = {}

        # Prototype-based comparison
        if self.args.comp_method == "proto":
            support_prototypes = outputs["support_prototypes"]

            # Compare query embeddings with prototypes
            logits = (
                    self.metric(query_embeddings, support_prototypes)
                    / self.args.logit_temperature
            )

            accs["accuracy"] = accuracy(logits, query_labels)
            losses["cross_entropy"] = F.cross_entropy(logits, query_labels)

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
        # if self.args.use_clip_filtering:
        from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
        from PIL import Image
        # 本地模型路径
        scheduler_path = "/home/wuzelei/Preprocess/stable-diffusion-2-base/models--stabilityai--stable-diffusion-2-base/snapshots/fa386bb446685d8ad8a8f06e732a66ad10be6f47/scheduler"
        model_path = "/home/wuzelei/Preprocess/stable-diffusion-2-base/models--stabilityai--stable-diffusion-2-base/snapshots/fa386bb446685d8ad8a8f06e732a66ad10be6f47/"

        clip_threshold = 0.2  # 你可以根据需求调整阈值

        # 获取支持样本的位置
        support_mask = mask_indices["support"]

        # 使用 CLIP 计算相似度
        ncorrect_indices, similarity_scores, incorrect_prompts_full = compute_clip_similarity_from_tensor(
            data, labels_y, "./map.json")
        # print(incorrect_prompts)
        # 通过相似度筛选干净样本和噪声样本
        is_clean = similarity_scores[support_mask] >= clip_threshold
        is_noisy = ~is_clean

        # 提取干净和噪声样本的 embedding 和 label
        support_embeddings_all = outputs["conv_embeddings"][support_mask]
        support_indices = support_mask.nonzero(as_tuple=True)[0].tolist()
        support_labels_all = [labels_y[i] for i in support_indices]
        # clean_support_labels = [support_labels_all[i] for i in range(len(support_labels_all)) if is_clean[i]]
        # noisy_support_labels = [support_labels_all[i] for i in range(len(support_labels_all)) if is_noisy[i]]

        clean_support_embeddings = support_embeddings_all[is_clean]
        # clean_support_labels = support_labels_all[is_clean]
        # noisy_support_embeddings = support_embeddings_all[is_noisy]
        # noisy_support_labels = support_labels_all[is_noisy]
        # return
        scheduler = EulerDiscreteScheduler.from_pretrained(scheduler_path)
        pipe = StableDiffusionPipeline.from_pretrained(
            model_path,
            scheduler=scheduler,
            torch_dtype=torch.float16
        ).to("cuda")
        new_images = []
        for prompt in incorrect_prompts:
            image = pipe(prompt).images[0]
            new_images.append(image)
        # transform = get_augmentation_transforms(self.args)
        image_tensors = []
        for img in new_images:
            transform = transforms.Compose([

                transforms.ToTensor(),  # 将 PIL 图像转为 Tensor
                get_augmentation_transforms(self.args)
            ])
            transformed = transform(img)  # 得到 Tensor（C, H, W）
            image_tensors.append(transformed)
        # list -> Tensor
        if image_tensors is None or len(image_tensors) == 0:
            # 如果 image_tensors 为空，使用 clean_support_embeddings
            support_embeddings = clean_support_embeddings
        else:
            # 如果 image_tensors 不为空，计算 new_embeddings
            image_batch = torch.stack(image_tensors).to(self.device)  # 将图片列表转换为 Tensor，并送到设备
            with torch.no_grad():
                new_embeddings = self.model.convnet(image_batch)

            # 拼接 clean_support_embeddings 和 new_embeddings
            support_embeddings = torch.cat([clean_support_embeddings, new_embeddings], dim=0)

        # for label in noisy_support_labels:
        # prompt = f"a clean photo of {label_name_map[label.item()]}"  # 根据标签生成提示词
        # image = pipe(incorrect_prompt).images[0]  # 生成图像

        # new_img = generate_image_from_prompt(incorrect_prompts)  # 生成图像
        # new_images.append(new_img)

        # else:
        # Auxiliary loss: protonet of conv embeddings
        if self.args.embed_proto_loss_weight > 0:
            embed_protos = gen_prototypes(support_embeddings, ways, shot)
            embed_protos_logits = metric(query_embeddings, embed_protos)
            embed_protos_loss = self.args.embed_proto_loss_weight * F.cross_entropy(
                embed_protos_logits, query_labels
            )
            losses["conv_embed_proto"] = embed_protos_loss

        # Binary Outlier loss: identify transformer embeddings as noisy samples or not
        # if self.args.binary_outlier_loss_weight > 0:
        #     noise_labels = (
        #         torch.tensor(noise_positions[mask_indices["support"]])
        #         .unsqueeze(dim=1)
        #         .to(self.device)
        #     )
        #     accs["binary_outlier"] = accuracy(
        #         outputs["outlier_logits"], noise_labels, binary=True
        #     )
        #     losses["binary_outlier"] = (
        #             self.args.binary_outlier_loss_weight
        #             * F.binary_cross_entropy_with_logits(
        #         outputs["outlier_logits"], noise_labels
        #     )
        #     )
        support_indices = mask_indices["support"].nonzero(as_tuple=True)[0]
        sim_support = similarity_scores[support_indices]
        clip_threshold = 0.2
        is_clean = sim_support >= clip_threshold
        is_noisy = ~is_clean

        # 支持集所有嵌入与标签
        all_embs = outputs["conv_embeddings"][support_indices]
        all_labels = torch.repeat_interleave(torch.arange(ways), shot).to(self.device)
        sup_labels = all_labels

        # 分离干净和噪声样本嵌入及标签
        clean_embs = all_embs[is_clean]
        clean_labels = sup_labels[is_clean]
        noisy_embs = all_embs[is_noisy]
        noisy_labels = sup_labels[is_noisy]

        # 计算每类 clean 和 noisy 的原型函数
        def compute_prototypes(embs, labels, ways):
            dim = embs.size(-1)
            protos = []
            for c in range(ways):
                idxs = (labels == c).nonzero(as_tuple=True)[0]
                if idxs.numel() > 0:
                    protos.append(embs[idxs].mean(dim=0))
                else:
                    protos.append(torch.zeros(dim, device=embs.device))
            return torch.stack(protos)

        clean_protos = compute_prototypes(clean_embs, clean_labels, ways)
        noise_protos = compute_prototypes(noisy_embs, noisy_labels, ways)

        def contrastive_proto_loss(anchor, positive, negative, temperature=0.1):
            # anchor, positive, negative: [ways, dim]
            pos_sim = F.cosine_similarity(anchor, positive, dim=-1).unsqueeze(1)  # [ways,1]
            neg_sim = F.cosine_similarity(anchor, negative, dim=-1).unsqueeze(1)  # [ways,1]
            logits = torch.cat([pos_sim, neg_sim], dim=1) / temperature  # [ways,2]
            labels = torch.zeros(anchor.size(0), dtype=torch.long, device=anchor.device)
            return F.cross_entropy(logits, labels)

        pred_protos = outputs.get("support_prototypes")  # [ways, dim]
        proto_loss = contrastive_proto_loss(pred_protos, clean_protos, noise_protos, 0.1)
        losses["proto_contrastive"] = self.args.clean_proto_loss_weight * proto_loss
        # # Clean Prototype matching: loss to match predicted prototype with clean exmples (reject noisy samples)
        # if self.args.clean_proto_loss_weight > 0:
        #     clean_support_embeddings = support_embeddings[
        #         ~noise_positions[mask_indices["support"]].astype(bool)
        #     ]
        #     num_clean_shots = shot - int(
        #         noise_positions[mask_indices["support"]].sum() / ways
        #     )
        #     clean_protos = gen_prototypes(
        #         clean_support_embeddings, ways, num_clean_shots
        #     )
        #
        #     # Distance of clean protos from predicted protos
        #     if self.args.dist_metric == "euclidean":
        #         losses["dist_to_clean_protos"] = (
        #                 self.args.clean_proto_loss_weight
        #                 * ((outputs["support_prototypes"] - clean_protos) ** 2).sum()
        #                 / ways
        #         )
        #     elif self.args.dist_metric == "cosine":
        #         epsilon = 1e-6
        #         norm_protos = outputs["support_prototypes"] / (
        #                 torch.norm(outputs["support_prototypes"], dim=1, keepdim=True)
        #                 + epsilon
        #         )
        #         norm_clean_protos = clean_protos / (
        #                 torch.norm(clean_protos, dim=1, keepdim=True) + epsilon
        #         )
        #
        #         # Negative, because we want to maximize cosine angle
        #         losses["dist_to_clean_protos"] = (
        #                 -self.args.clean_proto_loss_weight
        #                 * (norm_protos.flatten() * norm_clean_protos.flatten()).sum()
        #                 / ways
        #         )
        #     else:
        #         raise NotImplementedError

        return losses, accs

    # def loss_acc(self, batch, outlier_batch=None, support_label_noise_choices=[0]):
    #     # 1. Prepare data
    #     ways = self.args.train_way if self.model.training else self.args.test_way
    #     shot = self.args.train_shot if self.model.training else self.args.test_shot
    #     query_num = self.args.train_query if self.model.training else self.args.test_query
    #     data, labels, mask_indices, noise_positions, labels_y = self.prepare_data(
    #         batch, ways, shot, query_num, support_label_noise_choices, outlier_batch
    #     )
    #
    #     # 2. Forward pass for embeddings
    #     outputs = self.model(data, mask_indices["support"])
    #     conv_embs = outputs["conv_embeddings"]
    #     query_embeddings = conv_embs[mask_indices["query"]]
    #
    #     # 3. CLIP similarity for support
    #     _, _, _, _, _, _, sim_scores, incorrect_prompts = compute_clip_similarity_from_tensor(
    #         data, labels_y, "./map.json"
    #     )
    #     support_mask = mask_indices["support"]
    #     support_indices = support_mask.nonzero(as_tuple=True)[0]
    #     all_embs = conv_embs[support_mask]
    #     # all_labels = torch.tensor([labels_y[i] for i in support_indices], device=self.device)
    #     # 原来：all_labels = torch.tensor([labels_y[i] for i in support_indices], device=self.device)
    #     # all_labels = labels[support_mask].long().to(self.device)
    #     # ✅ 替换：使用未加噪前的原始标签 labels_y 来构造 all_labels
    #     all_labels = torch.tensor([labels_y[i.item()] for i in support_indices], device=self.device).long()
    #     sim_support = sim_scores[support_mask]
    #
    #     # Filter clean vs noisy
    #     is_clean = sim_support >= self.args.clip_threshold
    #     clean_embs = all_embs[is_clean]
    #     clean_labels = all_labels[is_clean]
    #     noisy_idx = (~is_clean).nonzero(as_tuple=True)[0]
    #     noisy_labels = all_labels[~is_clean]
    #
    #     # 4. Lazy init Stable Diffusion pipeline
    #     if not hasattr(self, 'sd_pipe'):
    #         sched = EulerDiscreteScheduler.from_pretrained(self.args.sd_scheduler_path)
    #         self.sd_pipe = StableDiffusionPipeline.from_pretrained(
    #             self.args.sd_model_path,
    #             scheduler=sched,
    #             torch_dtype=torch.float16
    #         ).to(self.device)
    #
    #     # 5. Generate new embeddings for noisy samples
    #     if len(noisy_idx) > 0:
    #         prompts = [incorrect_prompts[i] for i in noisy_idx.tolist()]
    #         images = self.sd_pipe(prompts).images
    #         image_tensors = [transforms.ToTensor()(img) for img in images]
    #         image_batch = torch.stack(image_tensors).to(self.device)
    #         with torch.no_grad():
    #             new_embs = self.model.convnet(image_batch)
    #     else:
    #         new_embs = torch.empty(0, all_embs.size(1), device=self.device)
    #         noisy_labels = torch.empty(0, dtype=all_labels.dtype, device=self.device)
    #
    #     # 6. Final support embeddings & labels
    #     support_embeddings = torch.cat([clean_embs, new_embs], dim=0)
    #     support_labels = torch.cat([clean_labels, noisy_labels], dim=0)
    #
    #     # 7. Initialize losses and accuracy dicts
    #     losses, accs = {}, {}
    #
    #     # 8. Comparison
    #     if self.args.comp_method == "proto":
    #         # Recompute prototypes from updated support set
    #         support_protos = gen_prototypes(support_embeddings, support_labels, ways)
    #         logits = self.metric(query_embeddings, support_protos) / self.args.logit_temperature
    #         accs["accuracy"] = accuracy(logits, labels[mask_indices["query"]].long())
    #         losses["cross_entropy"] = F.cross_entropy(logits, labels[mask_indices["query"]].long())
    #
    #     elif self.args.comp_method == "match":
    #         # Matching networks style
    #         one_hot = torch.zeros(len(support_labels), ways, device=self.device)
    #         one_hot[torch.arange(len(support_labels)), support_labels] = 1
    #         s_unit = support_embeddings / support_embeddings.norm(dim=1, keepdim=True)
    #         q_unit = query_embeddings / query_embeddings.norm(dim=1, keepdim=True)
    #         sims = F.softmax(q_unit @ s_unit.T, dim=1)
    #         p_y = sims @ one_hot
    #         accs["accuracy"] = accuracy(p_y, labels[mask_indices["query"]].long())
    #         losses["cross_entropy"] = F.nll_loss(p_y.log(), labels[mask_indices["query"]].long())
    #
    #     elif self.args.comp_method == "lin_cls":
    #         lin = nn.Linear(self.model.convnet.conv_out_dims, ways).to(self.device)
    #         opt = torch.optim.AdamW(lin.parameters(), lr=1e-3, weight_decay=0.01)
    #         for _ in range(100):
    #             opt.zero_grad()
    #             loss_sup = F.cross_entropy(lin(support_embeddings), support_labels)
    #             loss_sup.backward()
    #             opt.step()
    #         logits = lin(query_embeddings)
    #         accs["accuracy"] = accuracy(logits, labels[mask_indices["query"]].long())
    #         losses["cross_entropy"] = F.cross_entropy(logits, labels[mask_indices["query"]].long())
    #
    #     else:
    #         raise NotImplementedError(f"Unknown comp_method {self.args.comp_method}")
    #
    #     # 9. Optional embed-proto loss
    #     if self.args.embed_proto_loss_weight > 0:
    #         embed_protos = gen_prototypes(support_embeddings, support_labels, ways)
    #         embed_logits = self.metric(query_embeddings, embed_protos)
    #         losses["conv_embed_proto"] = self.args.embed_proto_loss_weight * F.cross_entropy(
    #             embed_logits, labels[mask_indices["query"]].long()
    #         )
    #
    #     # 10. Contrastive proto loss
    #     if self.args.proto_contrastive_loss_weight > 0 and new_embs.size(0) > 0:
    #         clean_protos = gen_prototypes(clean_embs, clean_labels, ways)
    #         noise_protos = gen_prototypes(new_embs, noisy_labels, ways)
    #         pred_protos = gen_prototypes(support_embeddings, support_labels, ways)
    #
    #         def contrastive_loss(a, p, n, temp):
    #             pos = F.cosine_similarity(a, p, dim=-1).unsqueeze(1)
    #             neg = F.cosine_similarity(a, n, dim=-1).unsqueeze(1)
    #             logits = torch.cat([pos, neg], dim=1) / temp
    #             return F.cross_entropy(logits, torch.zeros(a.size(0), dtype=torch.long, device=a.device))
    #
    #         c_loss = contrastive_loss(pred_protos, clean_protos, noise_protos, self.args.contrastive_temperature)
    #         losses["proto_contrastive"] = self.args.proto_contrastive_loss_weight * c_loss
    #
    #     # 11. Return original signature
    #     return losses, accs

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
#对比损失
    def contrastive_clustering_loss(self, support_embeddings, query_embeddings, support_labels, query_labels,
                                    temperature=0.07):
        """
        计算对比聚类损失，基于支持集和查询集的嵌入。

        Args:
            support_embeddings: [num_support, embed_dim] 支持集嵌入
            query_embeddings: [num_query, embed_dim] 查询集嵌入
            support_labels: [num_support] 支持集标签
            query_labels: [num_query] 查询集标签
            temperature: 温度参数，控制 softmax 分布的锐度

        Returns:
            loss: 对比聚类损失
        """
        # 合并支持集和查询集的嵌入和标签
        all_embeddings = torch.cat([support_embeddings, query_embeddings],
                                   dim=0)  # [num_support + num_query, embed_dim]
        all_labels = torch.cat([support_labels, query_labels], dim=0)  # [num_support + num_query]
        num_samples = all_embeddings.size(0)

        # 归一化嵌入
        all_embeddings = F.normalize(all_embeddings, p=2, dim=-1)

        # 计算相似度矩阵
        similarity_matrix = torch.mm(all_embeddings, all_embeddings.T) / temperature  # [num_samples, num_samples]

        # 构造正样本对（同类）和负样本对（不同类）
        labels_matrix = (all_labels.unsqueeze(1) == all_labels.unsqueeze(0)).float()  # [num_samples, num_samples]
        pos_mask = labels_matrix - torch.eye(num_samples, device=self.device)  # 去掉自相似
        neg_mask = 1 - labels_matrix  # 负样本对

        # 计算对比损失
        exp_sim = torch.exp(similarity_matrix)
        pos_sum = (exp_sim * pos_mask).sum(dim=1)  # 正样本相似度之和
        neg_sum = (exp_sim * neg_mask).sum(dim=1)  # 负样本相似度之和

        # 避免除零
        denominator = pos_sum + neg_sum + 1e-8
        loss = -torch.log(pos_sum / denominator + 1e-8)

        # 对所有样本取平均
        loss = loss[pos_mask.sum(dim=1) > 0].mean()  # 仅考虑有正样本的样本

        return loss

    def intra_class_compactness_loss(self, support_embeddings, support_labels, prototypes, outlier_probs=None):
        """
        类内紧致性损失：最小化支持集嵌入与类原型的距离。

        Args:
            support_embeddings: [num_support, embed_dim]
            support_labels: [num_support]
            prototypes: [ways, embed_dim]
            outlier_probs: [num_support] 异常值概率
        Returns:
            loss: 类内紧致性损失
        """
        distances = []
        for cls_idx in range(prototypes.size(0)):
            mask = (support_labels == cls_idx)
            cls_embeddings = support_embeddings[mask]
            cls_proto = prototypes[cls_idx]
            # 计算嵌入与原型的欧氏距离
            dist = torch.norm(cls_embeddings - cls_proto, dim=1)
            if outlier_probs is not None:
                weights = 1 - outlier_probs[mask]
                dist = dist * weights
            distances.append(dist)

        distances = torch.cat(distances)
        loss = distances.mean()
        return loss

    def clip_knowledge_distillation(self, query_embeddings, batch_data, labels_id, query_labels, temperature=1.0):
        """
        CLIP知识蒸馏：软标签蒸馏 + 特征蒸馏。

        Args:
            query_embeddings: [num_query, embed_dim] 模型生成的查询嵌入
            batch_data: [num_total, C, H, W] 批次图像数据
            labels_id: WordNet ID 列表
            query_labels: [num_query] 查询集标签
            temperature: 温度参数
        Returns:
            loss: 蒸馏损失
        """
        # 1. 准备CLIP图像嵌入和软标签
        query_images = batch_data[mask_indices["query"]].cpu().permute(0, 2, 3, 1).numpy()
        query_images = (query_images * 255).astype(np.uint8)
        clip_image_feats = []
        for img in query_images:
            img_pil = Image.fromarray(img)
            img_clip = self.clip_preprocess(img_pil).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = self.clip_model.encode_image(img_clip)
                feat = F.normalize(feat, p=2, dim=-1)
            clip_image_feats.append(feat)
        clip_image_feats = torch.cat(clip_image_feats)  # [num_query, clip_dim]

        # 生成CLIP文本嵌入
        class_names = [self.wordnet_id_to_label(wnid) for wnid in labels_id]  # 假设已定义转换函数
        unique_labels = list(dict.fromkeys(class_names))
        text_features = []
        for name in unique_labels:
            texts = [f"a photo of a {name}"]
            text_inputs = torch.cat([clip.tokenize(t) for t in texts]).to(self.device)
            with torch.no_grad():
                tf = self.clip_model.encode_text(text_inputs).mean(dim=0)
                tf = F.normalize(tf, p=2, dim=-1)
            text_features.append(tf)
        text_features = torch.stack(text_features)  # [ways, clip_dim]

        # 计算CLIP软标签
        sim_clip = torch.mm(clip_image_feats, text_features.T) / temperature
        soft_labels = F.softmax(sim_clip, dim=1)

        # 2. 软标签蒸馏
        logits = self.metric(query_embeddings, support_prototypes) / temperature  # 假设已定义metric和support_prototypes
        kd_loss = F.kl_div(F.log_softmax(logits, dim=1), soft_labels, reduction='batchmean') * (temperature ** 2)

        # 3. 特征蒸馏
        query_embeddings_proj = self.projection(query_embeddings)
        query_embeddings_proj = F.normalize(query_embeddings_proj, p=2, dim=-1)
        feat_loss = F.mse_loss(query_embeddings_proj, clip_image_feats)

        # 综合损失
        loss = kd_loss + 0.1 * feat_loss  # 特征蒸馏权重可调整
        return loss
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
        dists = torch.pdist(prototypes)  # [ways * (ways-1) / 2]
        # 鼓励最小距离大于 margin
        loss = torch.relu(margin - dists).mean()
        return loss

    def clip_weighted_prototypes(
            self,
            support_embeddings: Tensor,  # [N, D]
            support_data: Tensor,  # [N, C, H, W]
            labels_id: List[str],
            support_labels: Tensor,  # [N]
            ways: int,
            outlier_probys, D]

    device = support_embeddings.device
    diffusion = SimpleDiffusion(num_steps=100, device=device)
    diffusion_step = 70  # 可以调s: Optional[Tensor] = None

) -> Tensor:  # returns [wa参

class_names = []
for e in labels_id:
    wnid = e[0] if isinstance(e, (list, tuple)) else e
class_names.append(wordnet_id_to_label(wnid))

unique_labels = list(OrderedDict.fromkeys(class_names))
assert len(unique_labels) == ways
lbl2idx = {lbl: i for i, lbl in enumerate(unique_labels)}


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
    templates = prompt_templates + generate_hierarchical_prompts(lbl)
    prompts += [t.format(lbl) for t in templates]
text_inputs = clip.tokenize(prompts).to(device)
with torch.no_grad():
    txt_feats = self.clip_model.encode_text(text_inputs)  # [ways*num_temp, D]
D = txt_feats.size(-1)
text_features = txt_feats.view(ways, -1, D).mean(1)  # [ways, D]
text_features = F.normalize(text_features, dim=-1)

imgs = support_data.cpu().permute(0, 2, 3, 1).numpy()
pil_imgs = [Image.fromarray((im * 255).astype(np.uint8)) for im in imgs]
img_batch = torch.cat([self.clip_preprocess(im).unsqueeze(0) for im in pil_imgs]).to(device)
with torch.no_grad():
    img_feats = self.clip_model.encode_image(img_batch)
img_feats = F.normalize(img_feats, dim=-1)  # [N, D]

sim = img_feats @ text_features.T  # [N, ways]
sim_scores, sim_preds = sim.max(1)  # [N]
agree = sim_preds == support_labels
clip_w = sim_scores.clone()
clip_w[~agree] *= 0.1
conf_w = 1 - outlier_probs.view(-1) if outlier_probs is not None else torch.ones_like(clip_w)
total_w = clip_w * conf_w  # [N]

# ✨ Step: 使用 diffusion 去噪被判定为 outlier 的特征
if outlier_probs is not None:
    noise_mask = outlier_probs.view(-1) > 0.7  # 自定义阈值，视为“噪声”
    if noise_mask.sum() > 0:
        to_denoise = support_embeddings[noise_mask]
        x_t, noise = diffusion.add_noise(to_denoise, diffusion_step)
        x0_denoised = diffusion.denoise(x_t, noise, diffusion_step)
        support_embeddings[noise_mask] = x0_denoised  # 替换原始特征

# ✨ Step: 构建类原型
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
            w2 = ws[valid]
            w2 = w2 / (w2.sum() + 1e-8)
            proto = (feats[valid] * w2.unsqueeze(1)).sum(0)
        else:
            proto = feats.mean(0)
    prototypes.append(F.normalize(proto, dim=-1))
return torch.stack(prototypes, dim=0)
def clip_weighted_prototypes(
        self,
        support_embeddings: Tensor,  # [N, D]
        support_data: Tensor,        # [N, C, H, W]
        labels_id: List[str],
        support_labels: Tensor,      # [N]
        ways: int,
        outlier_probys, D]
    device = support_embeddings.device
    diffusion = SimpleDiffusion(num_steps=100, device=device)
    diffusion_step = 70  # 可以调s: Optional[Tensor] = None
) -> Tensor:  # returns [wa参

    class_names = []
    for e in labels_id:
        wnid = e[0] if isinstance(e, (list, tuple)) else e
        class_names.append(wordnet_id_to_label(wnid))

    unique_labels = list(OrderedDict.fromkeys(class_names))
    assert len(unique_labels) == ways
    lbl2idx = {lbl: i for i, lbl in enumerate(unique_labels)}

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
        templates = prompt_templates + generate_hierarchical_prompts(lbl)
        prompts += [t.format(lbl) for t in templates]
    text_inputs = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        txt_feats = self.clip_model.encode_text(text_inputs)  # [ways*num_temp, D]
    D = txt_feats.size(-1)
    text_features = txt_feats.view(ways, -1, D).mean(1)  # [ways, D]
    text_features = F.normalize(text_features, dim=-1)

    imgs = support_data.cpu().permute(0, 2, 3, 1).numpy()
    pil_imgs = [Image.fromarray((im * 255).astype(np.uint8)) for im in imgs]
    img_batch = torch.cat([self.clip_preprocess(im).unsqueeze(0) for im in pil_imgs]).to(device)
    with torch.no_grad():
        img_feats = self.clip_model.encode_image(img_batch)
    img_feats = F.normalize(img_feats, dim=-1)  # [N, D]

    sim = img_feats @ text_features.T  # [N, ways]
    sim_scores, sim_preds = sim.max(1)  # [N]
    agree = sim_preds == support_labels
    clip_w = sim_scores.clone()
    clip_w[~agree] *= 0.1
    conf_w = 1 - outlier_probs.view(-1) if outlier_probs is not None else torch.ones_like(clip_w)
    total_w = clip_w * conf_w  # [N]

    # ✨ Step: 使用 diffusion 去噪被判定为 outlier 的特征
    if outlier_probs is not None:
        noise_mask = outlier_probs.view(-1) > 0.7  # 自定义阈值，视为“噪声”
        if noise_mask.sum() > 0:
            to_denoise = support_embeddings[noise_mask]
            x_t, noise = diffusion.add_noise(to_denoise, diffusion_step)
            x0_denoised = diffusion.denoise(x_t, noise, diffusion_step)
            support_embeddings[noise_mask] = x0_denoised  # 替换原始特征

    # ✨ Step: 构建类原型
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
                w2 = ws[valid]
                w2 = w2 / (w2.sum() + 1e-8)
                proto = (feats[valid] * w2.unsqueeze(1)).sum(0)
            else:
                proto = feats.mean(0)
        prototypes.append(F.normalize(proto, dim=-1))
    return torch.stack(prototypes, dim=0)

#加diffusion
    class DiffusionModel(nn.Module):
        def __init__(self, feature_dim: int, timesteps: int = 1000):
            super(DiffusionModel, self).__init__()
            self.timesteps = timesteps
            self.feature_dim = feature_dim

            # 一个简单的MLP作为噪声去噪模型
            self.denoiser = nn.Sequential(
                nn.Linear(feature_dim, 512),
                nn.ReLU(),
                nn.Linear(512, feature_dim)
            )

        def forward(self, x: Tensor, t: Tensor) -> Tensor:
            """
            x: 加噪后的样本特征
            t: 当前时间步
            """
            # 输入加噪样本和时间步t，返回去噪后的特征
            return self.denoiser(x)

        def add_noise(self, x: Tensor, t: Tensor) -> Tensor:
            """
            x: 输入样本特征
            t: 当前时间步
            """
            noise = torch.randn_like(x)
            return x + noise * torch.sqrt(t.float())

        def reverse_process(self, noisy_x: Tensor, t: Tensor) -> Tensor:
            """
            逆过程（去噪），根据当前噪声的程度去除噪声
            """
            return self.forward(noisy_x, t)
