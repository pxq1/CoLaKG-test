# LastFM CoLaKG Improvement Experiments

Date: 2026-06-03

## Goal

Improve CoLaKG recommendation performance on LastFM by modifying the model/training framework. The target was about 4%-5% improvement.

## Baseline

Source log: `logs/lastfm_colakg_neighbor10_20260603_142617.txt`

Configuration:
- `dataset=lastfm`
- `seed=2020`
- `epochs=1000`
- `recdim=64`
- `layer=3`
- `neighbor_k=10`
- `dropout_i=0.6`
- `dropout_u=0.2`
- `dropout_n=0.6`

Best result:
- Epoch: 976
- Precision@10/20: `[0.20236686, 0.14026358]`
- Recall@10/20: `[0.27423901, 0.37873558]`
- NDCG@10/20: `[0.29353203, 0.34498224]`

## Implemented Framework Extensions

All new modules are controlled by command line flags and default to the original CoLaKG behavior.

1. Learnable semantic fusion gates
   - Args: `--fusion_gate`, `--gate_type`
   - Motivation: replace fixed averaging of ID and semantic representations with scalar/vector gates.
   - Result: short runs did not translate to better full convergence.

2. LightGCN++-style propagation controls
   - Args: `--prop_norm`, `--graph_gamma`
   - Motivation: borrow normalization and ego/residual aggregation ideas from recent LightGCN++ work.
   - Result: early gains were not stable; full run with `graph_gamma=0.2` reached only NDCG@20 `0.33405804`.

3. Social graph smoothing
   - Args: `--use_social`, `--social_alpha`
   - Motivation: LastFM contains `user_friends.dat`; user-user smoothing might help sparse users.
   - Result: short run with `social_alpha=0.1` reached NDCG@20 `0.28296981`, not enough to justify full run.

4. Semantic residual score
   - Args: `--semantic_score_alpha`
   - Motivation: add semantic user-item matching directly to ranking scores.
   - Result: short run with `semantic_score_alpha=0.1` was worse than baseline short run.

5. Semantic collaborative contrastive alignment
   - Args: `--semantic_cl_weight`, `--semantic_cl_tau`
   - Motivation: inspired by LLM-based recommendation alignment methods such as RLMRec; align CF and semantic views via in-batch contrastive loss.
   - Result: strong short run when combined with lower semantic dropout, but full run reached only NDCG@20 `0.34312148`.

6. Train-set item popularity prior
   - Args: `--pop_score_alpha`
   - Motivation: lightweight ranking calibration from training data only.
   - Result: small positive weights hurt short-run performance.

## Hyperparameter/Capacity Search

Key full experiments:

| Configuration | Best epoch | Recall@20 | NDCG@20 | Relative NDCG@20 vs baseline |
| --- | ---: | ---: | ---: | ---: |
| Baseline, `recdim=64`, dropout `0.6/0.2/0.6` | 976 | 0.37873558 | 0.34498224 | 0.00% |
| `recdim=64`, dropout `0.4/0.2/0.4` | 936 | 0.37870847 | 0.34630955 | +0.38% |
| `recdim=64`, dropout `0.2/0.2/0.2` | 936 | 0.38112571 | 0.34629315 | +0.38% |
| `recdim=128`, dropout `0.4/0.2/0.4` | 911 | 0.38044861 | 0.34913909 | +1.20% |
| `recdim=128`, dropout `0.2/0.2/0.2` | 911 | 0.38073081 | 0.34649776 | +0.43% |
| `recdim=256`, dropout `0.4/0.2/0.4` | 831 | 0.38305541 | 0.35044916 | +1.58% |
| `recdim=256`, dropout `0.2/0.2/0.2` | 751 | 0.38089304 | 0.34711957 | +0.61% |

Best short-run checks that were not promoted or did not win:
- `recdim=256`, dropout `0.4/0.2/0.4`, `neighbor_k=20`: 50-epoch NDCG@20 `0.29602921`, slightly below `neighbor_k=10`.
- `recdim=256`, dropout `0.2/0.2/0.2`: 50-epoch NDCG@20 `0.29720445`, but full convergence was worse than `dropout=0.4/0.2/0.4`.

## Best Verified Result

Best configuration:
- `dataset=lastfm`
- `seed=2020`
- `epochs=1000`
- `recdim=256`
- `layer=3`
- `neighbor_k=10`
- `dropout_i=0.4`
- `dropout_u=0.2`
- `dropout_n=0.4`
- Optional experimental modules disabled:
  - `fusion_gate=0`
  - `prop_norm=0`
  - `graph_gamma=-1.0`
  - `use_social=0`
  - `semantic_score_alpha=0.0`
  - `semantic_cl_weight=0.0`
  - `pop_score_alpha=0.0`

Best result:
- Epoch: 831
- Precision@10/20: `[0.20032275, 0.14112426]`
- Recall@10/20: `[0.27276932, 0.38305541]`
- NDCG@10/20: `[0.29622412, 0.35044916]`

Improvement over baseline:
- NDCG@20: `0.34498224 -> 0.35044916`, relative improvement `+1.58%`
- Recall@20: `0.37873558 -> 0.38305541`, relative improvement `+1.01%`
- Precision@20: `0.14026358 -> 0.14112426`, relative improvement `+0.61%`

## Conclusion

The verified best LastFM improvement comes from increasing recommendation embedding capacity to 256 dimensions while using stronger semantic/neighbor dropout. This improved NDCG@20 from `0.34498224` to `0.35044916`, but did not reach the requested 4%-5% target.

The larger framework modules were left as optional switches for future experiments, but the reproducible best command uses only the stable capacity/dropout configuration.
