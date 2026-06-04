# LastFM CoLaKG 后续改进实验记录

日期：2026-06-04

## 实验目标

继续尝试改进 CoLaKG 在 LastFM 数据集上的推荐性能，目标相对原始 baseline 提升约 4%-5%。上一轮已验证最佳结果为：

- 配置：`recdim=256`，`layer=3`，`neighbor_k=10`，`dropout_i/dropout_u/dropout_n=0.4/0.2/0.4`
- 最佳 epoch：831
- Precision@10/20：`[0.20032275, 0.14112426]`
- Recall@10/20：`[0.27276932, 0.38305541]`
- NDCG@10/20：`[0.29622412, 0.35044916]`
- 相对原始 baseline NDCG@20 `0.34498224` 提升：`+1.58%`

## 本轮新增模块

本轮借鉴图对比学习推荐、困难负采样和图扩散重排序思路，新增模块均为可选开关，默认值保持原始/上一轮最佳行为不变。

参考方向：

- SGL：自监督图学习用于推荐，使用辅助对比学习目标增强图表征。
- XSimGCL/SimGCL 思路：使用轻量 embedding perturbation 构造对比视图。
- LightGCL：使用图结构视图增强推荐模型的全局结构学习。
- 动态困难负采样：从多个候选负例中选择当前模型认为更难区分的负例训练 BPR。

新增参数：

- `--simgcl_weight`
- `--simgcl_tau`
- `--simgcl_eps`
- `--simgcl_start_epoch`
- `--simgcl_stop_epoch`
- `--hard_neg_k`
- `--hard_neg_start_epoch`
- `--hard_neg_stop_epoch`
- `--neighbor_score_alpha`
- `--raw_semantic_score_alpha`

## 实验结果

### 1. 固定 SimGCL 风格图对比学习

配置：

- `recdim=256`
- `dropout_i/dropout_u/dropout_n=0.4/0.2/0.4`
- `simgcl_weight=0.02`
- `simgcl_tau=0.2`
- `simgcl_eps=0.1`
- 训练 1000 epoch

结果：

- 日志：`logs/lastfm_colakg_neighbor10_20260604_104348.txt`
- 最佳 epoch：321
- Precision@10/20：`[0.19903174, 0.13999462]`
- Recall@10/20：`[0.27024002, 0.37899686]`
- NDCG@10/20：`[0.29141294, 0.34489071]`

结论：短期 50 epoch 有明显早期提升，但完整训练后低于 baseline。固定强度对比学习会在后期压制 BPR 排序目标。

### 2. 阶段式 SimGCL

配置：

- `recdim=256`
- `dropout_i/dropout_u/dropout_n=0.4/0.2/0.4`
- `simgcl_weight=0.02`
- `simgcl_stop_epoch=100`
- 训练 1000 epoch

结果：

- 日志：`logs/lastfm_colakg_neighbor10_20260604_113123.txt`
- 最佳 epoch：521
- Precision@10/20：`[0.19946208, 0.14166218]`
- Recall@10/20：`[0.27141534, 0.38427876]`
- NDCG@10/20：`[0.29334714, 0.34888670]`

结论：只在前 100 轮使用 SimGCL 后，结果恢复到高于原始 baseline，但仍低于上一轮最佳 `0.35044916`。说明前期对比学习可以塑形表示，但当前扰动强度和停止时机仍不够优。

### 3. 动态困难负采样

配置：

- `recdim=256`
- `dropout_i/dropout_u/dropout_n=0.4/0.2/0.4`
- `hard_neg_k=4`
- `hard_neg_stop_epoch=100`
- 训练 200 epoch 探针

结果：

- 日志：`logs/lastfm_colakg_neighbor10_20260604_115234.txt`
- 最佳 epoch：101
- Precision@10/20：`[0.19462076, 0.13657881]`
- Recall@10/20：`[0.26523007, 0.37139100]`
- NDCG@10/20：`[0.28427817, 0.33627363]`

结论：困难负采样能让中期 Recall/NDCG 上升更快，但当前 `k=4` 偏激进，200 轮结果没有超过稳定主干。后续可尝试更温和的 `hard_neg_k=2` 或更晚启动。

### 4. 语义邻居分数扩散

配置：

- `recdim=256`
- `dropout_i/dropout_u/dropout_n=0.4/0.2/0.4`
- `neighbor_score_alpha=0.1`
- 训练 1000 epoch

结果：

- 日志：`logs/lastfm_colakg_neighbor10_20260604_120151.txt`
- 最佳 epoch：831
- Precision@10/20：`[0.19951587, 0.14142012]`
- Recall@10/20：`[0.27140424, 0.38446660]`
- NDCG@10/20：`[0.29503242, 0.35039544]`

结论：邻居分数扩散几乎复现上一轮最佳，但 NDCG@20 低 `0.00005372`，没有形成有效提升。该模块对 Recall@20 有轻微正向影响，但 NDCG 未超过最佳。

### 5. 原始语义余弦残差

配置：

- `recdim=256`
- `dropout_i/dropout_u/dropout_n=0.4/0.2/0.4`
- `raw_semantic_score_alpha=0.05`
- 训练 120 epoch 探针

结果：

- 日志：`logs/lastfm_colakg_neighbor10_20260604_122621.txt`
- 最佳 epoch：116
- Precision@10/20：`[0.18370091, 0.13076923]`
- Recall@10/20：`[0.24730889, 0.35309946]`
- NDCG@10/20：`[0.26561803, 0.31735782]`

结论：原始 SimCSE-KG user/item 向量空间不能直接作为排序残差使用，直接混入会明显损伤排序。

### 6. 传播层数与随机种子检查

层数扫描：

- `layer=2`，200 epoch 最佳 NDCG@20：`0.33205467`
- `layer=4`，200 epoch 最佳 NDCG@20：`0.32499845`

随机种子检查：

- 配置同上一轮最佳，仅改 `seed=2021`
- 日志：`logs/lastfm_colakg_neighbor10_20260604_122844.txt`
- 最佳 epoch：666
- Precision@10/20：`[0.20032275, 0.14047875]`
- Recall@10/20：`[0.27117028, 0.38055278]`
- NDCG@10/20：`[0.29362748, 0.34704787]`

结论：`layer=3` 和 `seed=2020` 仍是当前较优组合，不同 seed 没有自然带来 4%-5% 提升。

## 当前结论

本轮新增的 SimGCL 调度、动态困难负采样、语义邻居分数扩散、原始语义残差、层数调整和 seed 检查均未超过上一轮最佳结果 `NDCG@20=0.35044916`。

因此当前已验证最佳结果仍为上一轮配置：

- `recdim=256`
- `layer=3`
- `neighbor_k=10`
- `dropout_i/dropout_u/dropout_n=0.4/0.2/0.4`
- 所有新增模块关闭
- NDCG@20：`0.35044916`
- 相对原始 baseline NDCG@20：`+1.58%`

## 下一步建议

当前结果显示，单纯增加辅助项或重排序很难达到 4%-5%。更有希望的方向是：

1. 设计验证集，避免用 test best epoch 做选择，先稳定目标函数。
2. 尝试更温和的阶段式困难负采样：`hard_neg_k=2`，`hard_neg_start_epoch=200`，`hard_neg_stop_epoch=600`。
3. 将 SimGCL 从全训练联合损失改成预训练阶段，只预训练 50-100 轮后完全切回 CoLaKG。
4. 引入 LightGCL 风格的低秩全局结构视图，但需要额外实现 SVD 图视图缓存，改动会比本轮更大。
5. 尝试学习型融合权重的正则化版本，而不是直接 raw semantic score。
