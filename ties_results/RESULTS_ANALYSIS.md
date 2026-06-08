# TIES-Unlearning Diff-LoRA 实验结果分析

> 数据来源:`ties_results/{baseline_results, ablation_results, sensitivity_results}`。
> 三套实验分别对应 reviewer item #7（baseline 对比）、#8（组件消融）、#10（敏感性分析）。
> 多 seed 鲁棒性（item #9）尚未运行 —— **本报告所有数字均为单 seed**，详见末尾"显著性"一节。

所有指标为准确率（%）。评测集:MNLI(分布内 utility)、e-SNLI(跨源 utility)、ANLI / SNLI-hard(对抗 OOD)、HANS(shortcut 鲁棒性,拆 entailment / non-entailment)。

**读数前提**:在所有实验里,真正区分方法的只有 **HANS non-entailment(H-nent)**。MNLI / e-SNLI / SNLI-hard 在各配置间几乎不动(MNLI 普遍 85±0.5,e-SNLI 80±1,SNLI-hard 73±1);ANLI 全员 ~28%(低于 3 类随机 33%,符合 ANLI 作为对抗集让 MNLI 模型失败的设计预期),无区分度。因此下文重点看 **H-nent 与 HANS overall**。

---

## 1. Baseline 对比(reviewer #7)

| Method | MNLI | e-SNLI | ANLI | SNLI-hard | HANS | H-ent | **H-nent** |
|---|---:|---:|---:|---:|---:|---:|---:|
| Standard LoRA | 85.0 | 81.5 | 28.8 | 73.5 | 59.4 | 98.5 | 20.2 |
| JTT | 81.9 | 79.1 | 30.9 | 70.3 | 62.3 | 90.1 | **34.6** |
| PoE (bias-model) | 83.9 | 80.2 | 28.6 | 73.8 | 58.8 | 96.9 | 20.7 |
| z-filtering | 84.3 | 81.2 | 28.4 | 73.7 | 60.1 | 96.2 | 23.9 |
| NegMerge | 31.8 | 32.5 | 33.3 | 35.0 | 50.0 | 0.0 | 100.0 |
| Naive Subtraction | 31.7 | 32.4 | 33.3 | 35.3 | 50.0 | 0.0 | 100.0 |
| **TIES-Unlearning (ours)** | **85.4** | 80.6 | 28.0 | 73.3 | 59.8 | 97.3 | 22.3 |

**结论:**
- **本方法的定位是"几乎不损 utility 的温和去偏"**:MNLI 85.4 是全场最高之一,相对 Standard LoRA(85.0)零损失,H-nent 小幅提升(20.2 → 22.3)。
- **去偏强度被简单方法超过**:JTT 把 H-nent 拉到 34.6(远超本方法 22.3),但代价是 MNLI 掉 3.5 个点(85.4 → 81.9)。z-filtering 也以接近的 MNLI 拿到略高的 H-nent(23.9)。**在纯去偏强度上,本方法并不领先。**
- **NegMerge / Naive Subtraction 完全塌缩**(全预测 non-entailment:H-ent 0 / H-nent 100,MNLI ≈ 随机)。这是其定义的必然结果——两者都是"纯 task-arithmetic 合并、无 Phase-3 去偏微调(phase3_epochs=0)",并非 bug(已核对 `config` 段确认 NegMerge 配置正确)。它们作为**下限**说明:没有 Phase-3 兜底的全局减法会摧毁模型。

> 一句话:本方法的卖点是 **utility-preserving**,而非 **最强去偏**。

---

## 2. 组件消融(reviewer #8)

### 2.1 塌缩归因:主因是缺 Phase-3,不是 merge 方式

| merge 方式 | 有 Phase-3 (epochs>0) | 无 Phase-3 (epochs=0) |
|---|---|---|
| 无 mask (naive) | `naive_mask`: MNLI **85.0** ✅ | `naive_subtract`: MNLI **31.7** ❌ |
| sign mask | `sign_only`: MNLI **85.2** ✅ | `negmerge`: MNLI **31.8** ❌ |
| full mask | `full`: MNLI **85.4** ✅ | `no_phase3`: MNLI **61.0** ⚠️ |

竖向对比一目了然:**同样的 mask,只要 `phase3_epochs>0` 就稳在 85% MNLI,一旦为 0 就崩。** Phase-3 debias 微调是恢复 utility 的关键组件。

### 2.2 有 Phase-3 时,mask 类型几乎无差别

| Ablation | MNLI | HANS | H-ent | H-nent |
|---|---:|---:|---:|---:|
| full (sign+trim) | 85.4 | 59.8 | 97.3 | 22.3 |
| naive_mask (无 mask) | 85.0 | 59.8 | 97.4 | 22.3 |
| sign_only | 85.2 | 59.2 | 97.5 | 21.0 |
| trim_only | 84.7 | 59.4 | 97.4 | 21.3 |

四者 H-nent 落在 21–22,差异在噪声量级。**在当前规模下,TIES 的 sign-consensus + magnitude-trimming 两个 mask 的边际贡献并不明显。**

### 2.3 真正有效的核心组件:layer localization + KL 信号

| 选层方式 | MNLI | HANS | H-ent | **H-nent** | 减法层数 |
|---|---:|---:|---:|---:|---:|
| full (KL+kNN) | 85.4 | 59.8 | 97.3 | 22.3 | 4 |
| kl_only | 85.2 | 60.2 | 96.9 | **23.4** | 4 |
| knn_only | 85.2 | 53.9 | 99.1 | 8.8 | 4 |
| random (随机选层) | 84.9 | 56.3 | 98.1 | 14.5 | 4 |
| global (全层减) | 83.9 | 51.2 | 99.2 | **3.2** | 12 |

这是消融里最强的证据,差异大到肯定显著:
- **定位 vs 全层**:global(全 12 层减)H-nent 砸到 **3.2**,full(定位 4 层)是 **22.3**。全层减法严重过减,即使有 Phase-3 也救不回。→ **layer localization 是真正关键的组件。**
- **定位 vs 随机**:random(随机 4 层)14.5 < full 22.3。→ **选对层确有增益,不是"减几层就行"。**
- **KL vs kNN**:kl_only **23.4** ≥ full 22.3 ≫ knn_only **8.8**。→ **KL 散度是定位的主力且有效;kNN 信号单独表现很差,且把它混进 KL(full)反而略微拖低 H-nent(23.4 → 22.3)。kNN 组件在当前 setup 下可能多余甚至有害。**

### 2.4 ⚠️ 需要正视的发现:subtraction 本身未带来净增益

把"减法强度"从弱到强排开,看 H-nent:

`no_subtraction`(P-only,完全不减)**27.3** → `full`(定位减 4 层)22.3 → `random`(随机减 4 层)14.5 → `global`(全层减)3.2

**完全不做 subtraction 的 P-only 模型 H-nent 最高(27.3),HANS overall 也最高(61.9)。** 也就是说,在这个 small-scale setup 下,subtraction 这个核心操作整体上在**损害** HANS 鲁棒性,layer localization 的作用主要是"把损害限制住",而非"创造增益"。这一点与方法的核心假设相悖,但数据明确支持,**必须在论文里诚实报告**(可能的解释:小规模训练下 shortcut 信号未充分建立 / HANS 评测方差;需更大规模或更强 shortcut 设定来检验)。

---

## 3. 敏感性分析(reviewer #10,OAT 单参数扫描)

**总体观察:所有 9 个超参对 utility(MNLI / e-SNLI / ANLI / SNLI-hard)几乎无影响**(MNLI 全程 84.7–85.7)。敏感性**全部集中在 HANS non-entailment**。下表只列 H-nent(默认值加 `*`):

| 参数 | grid → H-nent | 默认 | 趋势 / 最优 |
|---|---|---|---|
| **neg_rank** | 2→14.4 · 4→**22.3*** · 8→12.9 | 4 | 倒 U,**默认即甜点**,两边都掉 |
| **neg_lr_mult** | 1.0→**27.1** · 2.0→22.3* · 3.0→18.7 | 2.0 | 越小越好,**1.0 优于默认** |
| **pos_rank** | 8→15.6 · 16→**22.3*** · 32→19.3 | 16 | 默认最好,过低伤得重 |
| **phase2_mnli_mix_ratio** | 0.0→**27.4** · 0.1→22.3* · 0.2→26.3 | 0.1 | **不混 MNLI(0.0)反而最好** |
| **layer_selection_topk** | 2→**26.4** · 4→22.3* · 6→21.2 | 4 | 越聚焦越好,**2 优于默认** |
| **target_modules** | q-v→**22.3*** · q-k-v→16.6 · +out.dense→21.0 | (q,v) | 默认最好,加模块反而降 |
| **beta** | 0.3→**23.1** · 0.5→22.3* · 0.7→18.1 | 0.5 | 减太狠(0.7)伤鲁棒性 |
| **alpha** | 1.0→21.4 · 1.25→22.3* · 2.0→21.6 | 1.25 | 基本不敏感 |
| **trim_ratio** | 0.1→22.2 · 0.2→22.3* · 0.3→21.5 | 0.2 | 基本不敏感 |

**结论:**
- **稳健维度**:utility 对所有超参都极稳——方法不会因调参导致 MNLI 崩。这是可以主打的卖点。
- **最敏感参数**:`neg_rank`(倒 U,4 是甜点)、`neg_lr_mult`、`pos_rank`、`phase2_mnli_mix_ratio`、`layer_selection_topk`。
- **最不敏感**:`alpha`、`trim_ratio`(几乎不动,呼应 §2.2 trim mask 贡献小)。
- **几个默认值并非最优**:`phase2_mnli_mix_ratio=0.0`(27.4)、`neg_lr_mult=1.0`(27.1)、`layer_selection_topk=2`(26.4)单独看都优于当前默认(22.3),且都收敛到 ~27 —— 与 §2.4 的 P-only(27.3)同一水平。这再次印证"**减得越少 / 越聚焦,H-nent 越高**"的总趋势。若联合重调(不一定线性叠加),H-nent 有望从 22.3 提升到 ~27。

---

## 4. 综合结论与对写作的建议

**可以主打的点:**
1. **Utility 几乎零损失**(MNLI 85.4,全场最高之一;JTT 为去偏牺牲了 3.5 点)。
2. **对超参高度鲁棒**(utility 维度),且关键超参在合理范围。
3. **layer localization + KL 选层是有效且可验证的组件**(global 3.2 → 定位 22.3;random 14.5 → KL 23.4),消融证据强。

**需要诚实面对 / 预先准备答辩的弱点:**
1. **去偏强度不领先**:被 JTT(H-nent 34.6)和 z-filtering(23.9)超过。
2. **subtraction 核心操作未见净增益**:P-only(27.3)≥ full(22.3),核心假设在当前数据上未被支持(§2.4)。
3. **TIES 的两个 mask(sign/trim)边际贡献不明显**(§2.2);**kNN 信号可能多余甚至有害**(§2.3)。
4. **默认超参偏保守**:多个参数单独看都有更优值(§3)。

**建议的下一步:**
- 重设默认超参(`phase2_mnli_mix_ratio=0.0`、`neg_lr_mult=1.0`、`layer_selection_topk=2`)后重跑主结果,看 H-nent 能否稳定到 ~27 同时保住 MNLI。
- 考虑移除或弱化 kNN 信号(kl_only 已 ≥ full)。
- 在更大训练规模 / 更强 shortcut 设定下复核 §2.4,验证 subtraction 是否能在更充分的 shortcut 信号下体现增益。

**显著性提醒(重要):**
本报告全部为**单 seed** 结果。表中许多 1–3 点的差异(如 full 22.3 vs Standard 20.2、各 mask 之间)很可能落在噪声内,**必须跑 `run_multiseed.py` 拿到 mean ± std 才能定性**。但有几处差异大到几乎肯定显著、可直接采信:layer localization(global 3.2 vs full 22.3)、KL vs kNN(8.8 vs 23.4)、Phase-3 必要性(no_phase3 / NegMerge / Naive 的塌缩)。
