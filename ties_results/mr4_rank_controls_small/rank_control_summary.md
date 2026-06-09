# Rank-control sensitivity

| Control | r_P | r_N | Relation | Merged MNLI | Merged HANS non-ent | P-only MNLI | P-only HANS non-ent | N-only MNLI | N-only HANS non-ent |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| default_differential *(default)* | 16 | 4 | rank_differential | 82.45% | 7.33% | 82.05% | 5.06% | 34.80% | 0.00% |
| equal_rank_low | 4 | 4 | equal | 80.10% | 20.95% | 80.75% | 13.84% | 68.95% | 0.00% |
| equal_rank_high | 16 | 16 | equal | 80.95% | 10.83% | 81.40% | 5.35% | 68.35% | 0.00% |
| reversed_rank_default | 4 | 16 | reversed | 81.75% | 4.60% | 80.60% | 4.20% | 32.75% | 0.00% |
