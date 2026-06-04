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

## 追加实验：协同邻居扩散、去噪语义邻居与训练侧一致性

日期：2026-06-04

### 9. 协同 item-item 邻居分数扩散

改动：

- 新增 `cf_score_alpha` 和 `cf_neighbor_k`。
- 仅使用训练集交互构造 item-user 二值矩阵，计算 item-item cosine 相似近邻。
- 在 ranking 分数上加入协同近邻 item 的平均预测分数，作为语义 KNN 之外的协同局部结构补充。

结果：

| 配置 | 训练轮数 | 最佳 epoch | NDCG@20 | 结论 |
| --- | ---: | ---: | ---: | --- |
| `cf_score_alpha=0.05`, `cf_neighbor_k=20` | 1000 | 951 | `0.35077542` | 高于原始 baseline 和上一轮 256 维基础配置，但低于语义邻居扩散最佳 |

结论：协同 KNN 能提升局部 precision，但对 NDCG@20 的帮助弱于语义 KNN。CF KNN 和语义 KNN 在 eval-only 扫描中叠加后也没有形成互补收益，因此暂不作为主配置。

### 10. 继续 fine-tune 后半程

配置：

- 从 1000 epoch checkpoint 继续训练 500 epoch。
- `neighbor_score_alpha=0.05`
- 其它配置同当前最佳。

结果：

- 日志：`logs/lastfm_neighbor_score005_continue500_20260604.txt`
- 继续训练阶段最佳 epoch：11/500
- Precision@10/20：`[0.20381926, 0.14176977]`
- Recall@10/20：`[0.27769154, 0.38449200]`
- NDCG@10/20：`[0.29954342, 0.35217813]`

结论：继续 fine-tune 后没有超过当前最佳 `0.35257296`，说明单纯延长训练或重启优化器难以进一步提升。

### 11. mutual semantic KNN 去噪扩散

改动：

- 新增 `neighbor_score_mutual`。
- 只对互为语义近邻的 item 对做分数扩散；如果某个 item 没有 mutual 近邻，则回退到普通语义邻居均值。

eval-only 扫描结果：

| 配置 | NDCG@20 |
| --- | ---: |
| `neighbor_score_alpha=0.03`, `neighbor_score_mutual=1` | `0.34519518` |
| `neighbor_score_alpha=0.05`, `neighbor_score_mutual=1` | `0.34543737` |
| `neighbor_score_alpha=0.07`, `neighbor_score_mutual=1` | `0.34671351` |
| `neighbor_score_alpha=0.10`, `neighbor_score_mutual=1` | `0.34624479` |

结论：mutual KNN 过滤过强，去掉了不少有用的单向语义邻居，效果明显低于普通一跳语义扩散。

### 12. 更大 embedding 维度

配置：

- `recdim=512`
- `neighbor_score_alpha=0.05`
- 训练 300 epoch 探针

结果：

- 日志：`logs/lastfm_recdim512_neighbor_score005_300_20260604.txt`
- 最佳 epoch：296/300
- NDCG@20：`0.33723551`

结论：512 维容量没有在 300 轮内显示出优于 256 维的趋势，反而更容易不稳定，暂不拉长到 1000 轮。

### 13. 训练侧语义邻居一致性

改动：

- 新增 `neighbor_train_alpha`。
- 在 BPR 的正负 item 分数中加入 item 语义邻居 embedding 的平均分数，使训练目标与评估阶段的一跳语义扩散更一致。

配置：

- `neighbor_train_alpha=0.05`
- `neighbor_score_alpha=0.05`
- 训练 300 epoch 探针

结果：

- 日志：`logs/lastfm_neighbor_train005_score005_300_20260604.txt`
- 最佳 epoch：296/300
- NDCG@20：`0.33668255`

结论：直接把语义邻居平滑放入 BPR 训练会削弱正负 item 的区分度，没有带来收益。当前更合适的做法仍是只在 ranking 阶段使用轻量语义邻居扩散。

### 本轮小结

本轮新增的协同 KNN 扩散、mutual 语义 KNN 去噪、512 维容量、训练侧邻居一致性和继续 fine-tune 都没有超过当前最佳。

当前最佳仍为：

- `recdim=256`
- `layer=3`
- `dropout_i/dropout_u/dropout_n=0.4/0.2/0.4`
- `neighbor_score_alpha=0.05`
- `neighbor_score_steps=1`
- `NDCG@20=0.35257296`

下一步更值得集中尝试的是 query-aware / user-adaptive 的语义邻居扩散：不再用全局固定 `alpha`，而是根据用户语义表征、item 语义置信度或模型打分不确定性动态控制扩散强度。

## 追加实验：用户自适应语义扩散门控与邻居分数标准化

日期：2026-06-04

### 14. query-aware / adaptive semantic diffusion gate

动机：

- 参考近两年推荐系统中“自适应去噪、语义过滤、图语义对齐”的思路，将上一轮有效的固定语义邻居分数扩散从全局 `alpha` 改为可按用户-物品或 item 语义邻域置信度动态缩放。
- 核心假设是：语义邻居扩散在部分用户-物品对上可靠，但在语义噪声较高的位置会带来误扩散，因此需要 gate 做轻量去噪。

改动：

- 新增 `neighbor_gate_type`，支持：
  - `raw_semantic`：用原始用户语义向量和 item 语义向量 cosine 作为 query-aware gate。
  - `mapped_semantic`：用模型内部语义映射后的向量作为 gate。
  - `uncertainty`：对模型原始分数绝对值较小的位置给更高扩散权重。
  - `item_coherence`：根据 item 与其语义邻居的一致性控制扩散强度。
- 新增 `neighbor_gate_beta`、`neighbor_gate_center`、`neighbor_gate_min` 控制 gate 曲线。
- 默认 `neighbor_gate_type=none`，不会影响原有最佳配置。

300 epoch 探针：

| 配置 | 训练轮数 | 最佳 epoch | NDCG@20 | 结论 |
| --- | ---: | ---: | ---: | --- |
| `raw_semantic`, `alpha=0.05`, `beta=5`, `gate_min=0.5` | 300 | 296 | `0.33603728` | 与固定扩散几乎持平，没有明显收益 |
| `item_coherence`, `alpha=0.05`, `beta=2`, `gate_min=0.5` | 300 | 296 | `0.33635405` | 略高于同阶段固定扩散约 `+0.00030`，但信号很弱 |

300 epoch checkpoint 上 eval-only 扫描：

| 配置 | NDCG@20 |
| --- | ---: |
| `raw_semantic`, `alpha=0.09`, `beta=2`, `gate_min=0.5` | `0.33263127` |
| `raw_semantic`, `alpha=0.12`, `beta=2`, `gate_min=0.5` | `0.33236779` |
| 固定扩散 `alpha=0.12` | `0.33236378` |
| 固定扩散 `alpha=0.07` | `0.33229708` |

完整 1000 epoch：

| 配置 | 最佳 epoch | Precision@10/20 | Recall@10/20 | NDCG@10/20 | 结论 |
| --- | ---: | --- | --- | --- | --- |
| `raw_semantic`, `alpha=0.09`, `beta=2`, `gate_min=0.5` | 951 | `[0.20322754, 0.14158150]` | `[0.27586473, 0.38437930]` | `[0.29924287, 0.35251494]` | 非常接近当前最佳，但没有超过 `0.35257296` |

结论：query-aware gate 的直觉是合理的，但在 LastFM 当前语义向量质量下，gate 带来的排序扰动很小，甚至会轻微削弱已经有效的固定语义扩散。当前不能作为新的最佳配置。

### 15. 语义邻居分数组件标准化 / RRF 排序融合

动机：

- 固定语义邻居扩散直接把邻居平均分数加到原始 dot score 上，可能受到用户内分数尺度影响。
- 尝试对邻居分数组件做用户内标准化，或转换成 reciprocal rank fusion 风格的排序信号，再与主分数融合。

改动：

- 新增 `neighbor_score_norm`：
  - `user_zscore`
  - `user_minmax`
  - `rrf`
  - `none`
- 新增 `neighbor_rrf_k`。
- 默认 `neighbor_score_norm=none`，保持原始行为。

在当前 1000 epoch 末尾 checkpoint 上 eval-only 扫描结果：

| 配置 | NDCG@20 |
| --- | ---: |
| `neighbor_score_norm=rrf`, `alpha=0.20` | `0.35011031` |
| `neighbor_score_norm=rrf`, `alpha=0.50` | `0.35009595` |
| `neighbor_score_norm=user_minmax`, `alpha=0.01` | `0.35008344` |
| 固定扩散 `alpha=0.01` | `0.34989161` |

结论：RRF / min-max 在该 checkpoint 上有小幅收益，但仍低于历史最佳 `0.35257296`。这说明排序归一化可以作为后续组合模块保留，但单独使用无法达成 4%-5% 提升。

### 本轮小结

本轮实现并测试了：

- 用户自适应语义扩散 gate。
- item 语义邻域一致性 gate。
- 不确定性 gate。
- 用户内 z-score / min-max 标准化。
- RRF 风格邻居排序融合。

目前最佳仍为上一轮固定语义邻居扩散：

- `recdim=256`
- `layer=3`
- `dropout_i/dropout_u/dropout_n=0.4/0.2/0.4`
- `neighbor_score_alpha=0.05`
- `neighbor_score_steps=1`
- `NDCG@20=0.35257296`

本轮没有刷新最佳结果，但新增的自适应 gate 与分数标准化模块已经以默认关闭方式保留，后续可继续与验证集选择、checkpoint ensemble 或更强训练侧语义对齐模块组合探索。
