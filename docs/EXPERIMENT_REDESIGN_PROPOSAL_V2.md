# SER 一次性实验程序 v2（最终版）

日期：2026-09-03。基于 v1（`EXPERIMENT_REDESIGN_PROPOSAL.md`）、Codex 的 8 条 blocking 反馈
（`REVIEW_AND_V2_EXECUTION_REQUEST_20260903.md`）与我的逐条评价
（`RESPONSE_TO_CODEX_FEEDBACK_20260903.md`）。v2 不再只是蓝图：`v2/` 目录下是可执行的
预注册包（注册表、划分生成器、run_plan、评分器、独立验证器、突变电池、runner、审计工具），
本文件是它的说明书。真实拟合只能在作者的 GPU 机器上进行。

## 0. 结论

| | v1 | v2 |
|---|---|---|
| 实验臂 | 1 审计 + 1 因子表 + 3 附属（FT/HPO/DOSE） | 1 审计 + 1 因子表 + 机制块（含原 DOSE）+ FT/HPO + 推断层 MECHID + CPU Ridge |
| 单元数 | 未枚举 | **4,943** 个 GPU/CPU 单元，每个有配置哈希与划分哈希 |
| GPU 预算 | ≈32 h | **38.15 h + 8.0 h 条件项 = 46.15 h**，硬上限 50 h |
| 检验族 | "约 10 个" | 精确：N-decomp 4、N-mech 4、N-supp 2、R 11、T 2；另 7 个假设经功效仿真**先验降为估计** |
| 正文声明 | 7 条 | **4 条**（审计估计；溢价与分解；机制；现代管线），其余进补充 |
| 可执行性 | 无代码 | 划分表、run_plan、评分器、验证器、突变电池、runner 全部实现并在合成数据上跑通 |

## 1. v1 → v2 变更表

| # | 来源 | 变更 | 落点 |
|---|---|---|---|
| 1 | Codex-1 | `SG−GG` 不是主效应。改为说话人重叠 × prompt 重叠 **棋盘 2×2**（条件 none/spk/prm/both），四格相同测试项、相同训练样本数（暴露集 + BB 池分层填充）、相同内验证；CREMA-D（12 句）与 SUBESCO-full（10 prompt）上做，RAVDESS 不进 | `splits.mechanism_block`，MECH2X2 臂 |
| 2 | Codex-2 | HPO 改为固定说话人互斥外层测试上的 {固定, HPO} × {随机内验证, 分组内验证}，DiD 估计；另保留 RR_hpo 描述格与"在测试集上选配置"的零成本乐观偏差描述量；不再把 HPO 映射到"用测试集选模型"的构念 | HPO 臂（GR_hpo/GG_hpo/RR_hpo），N14、D08、D09 |
| 3 | Codex-3 | 机器可读注册表：28 条假设逐行给出 ID、声明、语料、模型层、对比、方向、单位、统计量、CI 方法、族与精确族大小、ties/缺失/截断规则、SESOI；功效仿真表 | `registry/hypothesis_registry.csv`、`claim_map.json`、`power_table.json` |
| 4 | Codex-4 | CREMA-D-24 加 **utterance 数匹配的 91 人面板**（5 次按性别分层抽样各自）；SUBESCO 改三面板 1400×1 / 700×1 / 700×2，固定 N 的对比明确标为 allocation effect | CTRL 次级层 cremad_24_d*/cremad_91m_d*、sub_*；N10、D04、D07 |
| 5 | Codex-5 | 微调对照改为 **同一 WavLM-base+、同头、同折、同 seed 的冻结同制度模型**；speaker-ID 探针逐 checkpoint、公共留出面板；RAVDESS 加第二 seed，CREMA-D 第二 seed 为条件项 | FT 臂（wavlm_base_plus_ft / wavlm_base_plus_frozen_sr），MECHID |
| 6 | Codex-6 | SUBESCO-980 也做 CNN 的 3 draw × 3 seed 交叉；推断单位始终是说话人，先在说话人内对 draw/seed 求均值 | CTRL 交叉单元，D14 |
| 7 | Codex-7 | 执行层只描述可执行子集并报告比例；人 + AI 如实标注；语义分歧交第二名人类 | `audit/adjudicate.py` |
| 8 | Codex-8 | tag-1 冻结全部科学配置与验证器；tag-2 只填计时探针数值 | 第 6 节两次冻结 |
| 9 | 我的修正 | DOSE 并入机制块：q=0 即 none、q=1 即 spk、半暴露两轮旋转即 spk_half_h1/h2；挤压与溢出成为 D06 与 T02 | MECH2X2 |
| 10 | 功效仿真 | 单一 15 成员 Holm 族功效太低（机制块 0.1–0.3）。改为**按声明分族**，机制块加**两次 prompt 组轮换**（每说话人测试项翻倍），功效 <0.5 的假设先验降为估计 | `registry.py` FAMILY_POLICY、`power_sim.py` |
| 11 | 真实数据 | 预注册 hygiene：标签冲突的字节重复整组删除（CREMA-D 3 组）、同标签重复保留字典序首个（RAVDESS 1 组）、登记的坏文件（CREMA-D 1 个） | `corpora.apply_hygiene`，`plan_rc1/hygiene_log.json` |
| 12 | 真实数据 | CREMA-D 的 IEO 句有强度变体（1,911 条），prompt 分组改为按语料量贪心均衡 | `splits._mass_balanced_groups` |

## 2. 最终程序

### 2.1 语料层

| 层 | 基语料 | 角色 | N | 说话人 | 来源状态 |
|---|---|---|---:|---:|---|
| ravdess | RAVDESS speech | 主 | 1,439（hygiene 后） | 24 | 真实（P1 manifest） |
| cremad | CREMA-D | 主 | 7,435 | 91 | 真实（P1 manifest） |
| subesco_980 | SUBESCO 重复清洁面板 | 主 | 980 | 20 | 真实（S3 release_v10 pinned 列表） |
| subesco_full | SUBESCO 全量 | 次 | 7,000 | 20 | **占位**（需作者 manifest） |
| sub_1400_one / sub_700_one / sub_700_two | SUBESCO 面板 | 次 | 1,400 / 700 / 1,400 | 20 | 占位 |
| cremad_24_d0..4 | CREMA-D 24 人 | 次 | ≈1,960 | 24 | 真实 |
| cremad_91m_d0..4 | CREMA-D utterance 数匹配 | 次 | 与对应 24 人面板相同 | 91 | 真实 |

### 2.2 实验臂与单元

| 臂 | 内容 | 单元 | GPU-h | 上限 |
|---|---|---:|---:|---:|
| PREP | manifest、log-mel 缓存、13 层 SSL 缓存、划分断言、合成干跑、突变电池、离主语料计时探针 | — | 3.0 | 3.5 |
| CTRL | 主层：3 scratch 模型 × RR/RG/GR/GG × 5 折 × r=0..2，3 个冻结探针同划分；次级层 RR/GG；take 分组划分；draw×seed 交叉；LOSO 对；P1 对账 | 3,139 | 8.66 | 16 |
| MECH2X2 | CREMA-D 6 条件、SUBESCO-full 7 条件（含 both_sib）× 2 轮换 × 5 组 × r=0,1 × CNN/ResNet-SE | 520 | 5.66 | 7 |
| FT | WavLM-base+ 部分微调 vs 冻结同制度，RR/GG，r=0；RAVDESS 与 SUBESCO-980 两 seed，CREMA-D 一 seed（+1 条件） | 120 | 17.33 (+8.0) | 26 |
| HPO | ResNet-SE，8 配置 × GR_hpo/GG_hpo/RR_hpo × 5 折；CREMA-D 主、RAVDESS 次 | 240 | 2.70 | 5 |
| MECHID | 推断：speaker-ID 探针与最近邻组成 | 180 | 0.30 | 0.5 |
| PROBECPU | Ridge：RO/GO（RAVDESS r=0..2）、G5 与 LOSOSUB×3（CREMA-D）作为合同单元；20 次抽样 × 13 层扫描另计（54,600 次拟合，约 1 CPU-h） | 924 | 0 | — |
| STAT | 1%+1% 逐位重训练抽查、一次性评分、验证器重放 | — | 0.5 | 0.5 |
| **合计** | | **4,943** | **38.15 + 8.0** | **50** |

截断顺序（仅当投影超过 50 h 时按序执行，砍掉的检验记"未检验"）：
① CREMA-D 微调第二 seed（8.0 h）② HPO 的 RAVDESS 格（0.44）③ MECH2X2 的 r=1（2.83）
④ cremad_24/91m 第 4–5 次抽样（0.8）⑤ RAVDESS/SUBESCO 冻结对照第二 seed（1.17）
⑥ SUBESCO-980 上 Transformer 的 r=2（0.03）。

### 2.3 假设、族与先验状态（功效仿真 1,000 次，Holm 逐步）

| 族 | 确认性成员（功效） | 先验降为估计（功效） |
|---|---|---|
| N-decomp（C2a/b） | N01 SUBESCO scratch RR−GG (1.00)；N02 探针 RR−GG (0.83)；N03 RG−GG (0.86)；N04 RR−RG (0.67) | — |
| N-mech（C3） | N05 CREMA-D prm−none (0.61)；N06 spk−none (0.58)；N09 SUBESCO-full sibling 增量 (0.67)；N10 密度调节 (0.94) | N07、N08（SUBESCO-full，n=20；0.29/0.49） |
| N-modern（C4b） | — | N11–N13 重暴露（0.34/0.49/0.18）→ C4b 成为估计性声明 |
| N-supp（S1/S2） | N14 HPO 验证乐观 (1.00)；N15 头部效应 (0.82) | — |
| R（复现） | R01–R11 全部（0.53–1.00） | — |
| T（等价） | T01 LOSO-sub≈Group5 ±1 pp；T02 无溢出 ±1.5 pp | — |

功效假设逐条写在 `power_sim.py`（均值/SD/n 及来源），先验状态在 tag-1 冻结后不再改变。

### 2.4 正文四条声明

1. **C1 审计（估计）**：Y_any 与 Y_test 比例、Wilson 与部分识别区间、有限 frame 超几何包络、U 界、可执行子集的实际重叠、人–AI kappa。
2. **C2 溢价与分解**：SUBESCO-980 上 RR−GG>0（scratch 与探针），且 RG−GG>0、RR−RG>0；RAVDESS/CREMA-D 作为同实现复现（R01–R08）。
3. **C3 机制**：prompt 重叠单独抬高（N05）、跨句说话人重叠单独抬高（N06）、sibling take 增量（N09）、同 N 下少说话人高密度抬高更多（N10）。
4. **C4 现代管线**：C4a 微调后溢价仍在（R09–R11，≥2/3）；C4b 重暴露作为估计报告（+ speaker-ID 探针描述量）。

补充：S1 HPO 验证乐观（N14）、S2 头部效应（N15）、S3 LOSO 等价（T01）、S4 无溢出（T02）、
描述量 D01–D18。禁止的表述见 `claim_map.json`。

## 3. 本环境已完成并验证的内容

| 组件 | 文件 | 验证 |
|---|---|---|
| 注册表 | `v2/registry/{hypothesis_registry.csv, claim_map.json, arms.json, power_table.json, subesco_980_pinned.json}` | 28 假设全部绑定声明；族大小精确 |
| 划分生成器 | `v2/ser_v2/splits.py` | 21 项 pytest：边界零重叠、恰好一次、机制块四格测试/验证/训练量一致、sibling 隔离、面板尺寸 |
| 计划 | `v2/plan_rc1/{run_plan.csv, split_index.json, unit_configs.json, hygiene_log.json, plan_summary.json}` | 确定性（两次生成哈希相同）；4,943 单元 |
| 评分器 | `v2/ser_v2/score.py` | 合成全量运行 28 假设全部可算，完整性 I1–I7 通过 |
| 突变电池 | `v2/ser_v2/mutations.py` | 18 例故障注入 + 2 个世界级案例（零效应世界无假阳性；符号翻转按方向读为不支持） |
| 独立验证器 | `v2/ser_v2/verify.py` | 由只读规范的子代理独立实现；与评分器逐项比对至 1e-12 |
| runner | `v2/ser_v2/train.py` | 冻结 P1 引擎（advanced_models 配置索引 2）在合成 log-mel 上 CPU 干跑；线性探针、Ridge、合成引擎、WavLM 引擎 |
| 审计 | `v2/ser_v2/audit/` | beacon 排序、顺序筛选到固定 n、脚本裁决、区间族（重现论文 K=[177,263]） |

## 4. 作者机器上的执行顺序

见 `v2/README.md`。要点：真实 SUBESCO manifest → 重建计划 → 真实 log-mel/SSL 缓存 →
合成干跑 + 突变电池 + 验证器一致 → **tag-1** → CTRL/PROBECPU/MECH2X2/HPO →
离主语料计时探针 → **tag-2**（只填计时）→ FT → MECHID → 锁定账本 → 一次性评分 → 验证器重放。
