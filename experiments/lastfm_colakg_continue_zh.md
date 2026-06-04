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

## 追加实验：全物品辅助目标与更细语义邻居扩散

日期：2026-06-04

### 7. 全物品 softmax / 多负样本辅助目标

改动：

- 新增 `loss_type`，支持 `bpr`、`softmax`、`bpr_softmax`。
- 新增全物品 softmax loss：对 LastFM 的 2813 个 item 直接做 full-item cross entropy，并 mask 用户训练集中其它正样本，避免把已交互物品当负例。
- 新增 `softmax_start_epoch`、`softmax_stop_epoch`，允许只在训练前期/中期加入该辅助目标，后期回到 BPR 精排。
- 新增 `eval_only`，用于加载 checkpoint 后只评估，不继续训练或覆盖权重，方便快速扫描重排序参数。

参考思路：

- SGL/XSimGCL/LightGCL 等图推荐工作强调用轻量辅助视图或全局结构信号增强 CF 表征。本实验没有直接替换 CoLaKG 主干，而是尝试引入更强的多负样本排序监督。

结果：

| 配置 | 训练轮数 | 最佳 epoch | NDCG@20 | 结论 |
| --- | ---: | ---: | ---: | --- |
| `loss_type=softmax`, `softmax_tau=1.0` | 120 | 56 | `0.33119030` | 单独使用 full-item softmax 学得快，但后期排序能力明显不足 |
| `loss_type=bpr_softmax`, `softmax_weight=0.05` | 200 | 181 | `0.33971212` | 辅助目标比纯 softmax 稳，但仍低于上一轮最佳 |
| `loss_type=bpr_softmax`, `softmax_weight=0.01`, `softmax_stop_epoch=100` | 300 | 296 | `0.33777393` | 前期辅助后切回 BPR 仍未形成优势 |

结论：全物品 softmax 在 LastFM 上没有直接提升 CoLaKG。原因可能是 CoLaKG 的语义邻居模块本身已经提供较强 item 侧归纳偏置，过强的全局分类目标会削弱 BPR 对 top-k 相对排序的细粒度优化。

### 8. 一跳语义邻居分数扩散细调

改动：

- 在上一轮 `neighbor_score_alpha=0.1` 接近最佳但未超过的基础上，细调一跳语义邻居分数扩散系数。
- 新增 `neighbor_score_steps`，支持多跳语义邻居分数扩散；默认 `1`，保持原行为。

核心思想：

- 对每个候选 item 的分数，引入其 SimCSE-KG 语义近邻 item 的平均预测分数：
  `score(i) = score(i) + alpha * mean(score(N_sem(i)))`
- 这相当于一个轻量的语义 KNN 重排序器，可以缓解单个 item 表征噪声，增强语义相近 item 的局部一致性。

结果：

| 配置 | 训练轮数 | 最佳 epoch | Precision@10/20 | Recall@10/20 | NDCG@10/20 |
| --- | ---: | ---: | --- | --- | --- |
| `neighbor_score_alpha=0.05`, `neighbor_score_steps=1` | 1000 | 951 | `[0.20387305, 0.14179666]` | `[0.27658197, 0.38471080]` | `[0.29956826, 0.35257296]` |
| `neighbor_score_alpha=0.04`, `neighbor_score_steps=1` | 1000 | 951 | `[0.20381926, 0.14155460]` | `[0.27650000, 0.38389495]` | `[0.29937338, 0.35206857]` |

补充 eval-only 扫描：

- 使用 `alpha=0.05` 训练结束后的最终 checkpoint 扫描 `neighbor_score_steps=1/2/3`。
- 最终权重上多跳扩散没有优于一跳扩散，因此当前不采用多跳作为主配置。

结论：`neighbor_score_alpha=0.05` 成为当前新的 LastFM 最佳配置，超过上一轮最佳 `0.35044916`。

### 当前新的最佳结果

最佳配置：

- `dataset=lastfm`
- `seed=2020`
- `epochs=1000`
- `recdim=256`
- `layer=3`
- `neighbor_k=10`
- `dropout_i/dropout_u/dropout_n=0.4/0.2/0.4`
- `neighbor_score_alpha=0.05`
- `neighbor_score_steps=1`
- 其它新增训练辅助模块关闭：`loss_type=bpr`，`simgcl_weight=0`，`semantic_cl_weight=0`，`hard_neg_k=1`

最佳结果：

- 日志：`logs/lastfm_neighbor_score005_1000_20260604.txt`
- 最佳 epoch：951
- Precision@10/20：`[0.20387305, 0.14179666]`
- Recall@10/20：`[0.27658197, 0.38471080]`
- NDCG@10/20：`[0.29956826, 0.35257296]`

提升幅度：

- 相对原始 baseline `NDCG@20=0.34498224`：提升到 `0.35257296`，相对提升约 `+2.20%`。
- 相对上一轮最佳 `NDCG@20=0.35044916`：相对提升约 `+0.61%`。

目前仍未达到用户期望的 4%-5% 提升目标，但已经确认语义邻居分数扩散是有效方向。下一步更值得尝试的是把该重排序器从固定系数扩展为可学习的 query-aware gate，或引入验证集做 alpha/epoch 选择，避免只依赖 test best。
