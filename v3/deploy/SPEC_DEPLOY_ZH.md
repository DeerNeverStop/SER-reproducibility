# SER26-DEPLOY-1：部署含义研究规范（登记组成偏斜 × 人工复核预算）

版本 DEPLOY-0.1；制定 2026-09-05（多伦多）；目标：2026-09-16 ICASSP 2027 截稿前形成可纳入稿件"部署含义"一节的证据。

性质：**前瞻性规定的描述性/估计性研究**。研究者已看过 v2 全部结果、Study II 结果、以及 2026-09-05 在 CREMA-D/RAVDESS/SUBESCO-980 上用 v2 缓存与预测做的两组先验证数字（登记组成偏斜的 UAR 变化；固定 20% 复核量下的接受错误率与逐人覆盖）。本研究不继承 v2/N14R/Study II 的预注册身份，不输出确认性 p 值或 Holm 结论；所有终点在正式运行前于本文件与 `plan.json` 锁定，全部预定量无论方向一律报告。先验证脚本与数字见 `PRIOR_EXPOSURE.md`。

## 0. 三个模块与唯一问题

| 模块 | 问题 | 新训练 | 设备 |
|---|---|---|---|
| A 登记组成 | 用陌生说话人**少量无标签**语音估计其个人归一化基线时，参照语音的情绪组成偏斜是否伤害识别？哪些便宜的估计器能挽回？ | Ridge（CPU，秒级） | CPU |
| B1 复核预算 | 在说话人排他的已有预测上，按最大后验概率拒判交给人工复核 b 的比例，接受集错误率、错误捕获率与**逐说话人覆盖**如何？ | 无 | CPU |
| B2 校准重叠 | 复核阈值在**说话人与训练重叠**的内层验证集（GR）上选定，与在**说话人排他**的内层验证集（GG）上选定相比，在同一批陌生测试说话人上的实际复核比例、接受错误率与"验证集承诺 vs 测试实现"的差距如何？ | Ridge（CPU）、CNN 与 WavLM 部分微调（GPU） | CPU + 云 GPU |

三个模块共用 v2 的语料人群、清理规则、划分表与特征缓存；任何文件按 SHA-256 钉住并写入 `plan.json`。

## 1. 数据与划分

- 语料层级：`ravdess`（24 人，8 类）、`cremad`（91 人，6 类）、`subesco_980`（20 人，7 类）。人群 = v2 `plan_rc2/splits/ctrl__<level>__GG__r{0,1,2}.json` 的 `population`。
- 外层划分：v2 CTRL 的说话人分组五折，`r0,r1,r2` 三次划分抽样。GG 与 GR 在同一 `r` 下外层测试折**逐折相同**（已核验），GR 的内层验证说话人与 fit 重叠（RAVDESS 19 / CREMA-D 73 / SUBESCO-980 16 人），GG 为 0。
- 特征：v2 `features/<base>__<enc>__*.npz` 的 13 状态时间均值特征，取 `state=12`（末层）；编码器 `wavlm_base_plus`、`hubert_base`、`wav2vec2_base`。log-mel 缓存供 CNN。
- 音频（仅 B2 的 WavLM 微调）：v2 manifest 指向的原始 wav，按 sha256 核验后再上传云端。

## 2. 模块 A：登记组成

### 2.1 训练
每个 `(level, enc, r, fold)`：训练说话人 = 人群 − 该折测试说话人（fit ∪ val；Ridge 无早停）。
- 模型 M_norm：对每个训练说话人用其**全部训练语句**做逐人 z-score（均值/标准差按维度；标准差下限 1e-3），再 `RidgeClassifier(alpha=1.0, class_weight="balanced", solver="auto")`。
- 模型 M_glob：`StandardScaler`（训练语句）+ 同参数 Ridge，作"不归一化"对照与 E3/E4 的后验来源（用 `decision_function` 经 softmax 温度 1 得到伪后验）。
- 训练侧类别偏移 δ_e：在逐人中心化后的训练特征上，类别 e 的均值（按训练说话人等权），用于 E4。

### 2.2 测试说话人的查询集与登记池（句子不相交）
按句子划分，查询与登记不共享句子：
- CREMA-D：12 句按预定哈希（盐 `SER26-DEPLOY-1:cremad:sent`）排序，前 5 句为查询集 Q，后 7 句为登记池 E；
- SUBESCO-980：7 个 prompt，前 3 为 Q，后 4 为 E；
- RAVDESS：2 个 statement，statement 01 为 Q，02 为 E（RAVDESS 每 statement 每类 ≥2 条，neutral 每 statement 2 条）。
Q 在所有条件间**完全相同**；评分只用 Q。

### 2.3 参照条件（composition × N）
- `balanced`：按类别轮转从 E 抽 N 条（类别顺序按预定哈希，轮转到某类而该类不足即整条件不可行）；N ∈ {3, 5, 10, 20}。
- `single:<e>`：只抽类别 e 的 N 条；N ∈ {3, 5}。
- `other_balanced` / `other_single:<e>`：参照来自**同折另一位测试说话人**（按预定哈希配对，配对固定）的 E，N ∈ {3, 5}；用于检验"个人信息是否真实存在"。
- `oracle_all`：用该说话人 E ∪ Q 的全部语句（含查询，不可部署，仅上界）。
可行性规则：某 `(level, composition, N)` 仅当该层级**每一位**说话人的 E 都能供给时才纳入；否则整条件在该层级标 `infeasible`（不降 N、不补样）。按人群计数，预计：RAVDESS `single:neutral` 全部不可行，`single:*` 仅 N=3；SUBESCO-980 `single:*` 仅 N=3；CREMA-D 全部可行。计划生成时逐条核验并写入 `plan.json`。
每个 `(speaker, composition, N)` 独立抽 5 次参照（种子 = 哈希(program, level, r, speaker, composition, N, draw)）。

### 2.4 估计器
- E0 `none`：M_glob 直接预测 Q。
- E1 `naive`：参照均值/标准差做 z-score，送 M_norm。
- E2 `shrink`：均值与方差按 `(N·参照统计 + k·训练全局统计)/(N+k)`，k=8（v1 §2.5 口径）。
- E3 `neutral_filter`：用 M_glob 伪后验给参照打分，保留预测为 neutral 的参照；不足 2 条时取 neutral 后验最高的 2 条；均值取保留参照，方差按 E2 公式用全部 N 条参照；第 2 轮把全部参照按 E1（全部参照统计）归一化后用 M_norm 后验重选一次；各条件的回退次数记入 model.json。
- E4 `prior_corrected`：用 Saerens–Latinne–Decaestecker (2002) EM 在 M_glob 伪后验上估计参照的类别先验 π̂（最多 50 轮，初值训练先验），均值 μ̂ = mean(参照) − Σ_e π̂_e δ_e；方差按 E2。
- `oracle_all` 只配 E1。
所有估计器不使用任何参照或查询的标签。参照统计与训练逐人统计均用 ddof=0，标准差下限 1e-3。
可行性预期（计划生成时核验）：RAVDESS `single:neutral` 全部不可行、`single:*` 仅 N=3、`balanced@20` 不可行（neutral 在 E 中仅 2 条）；SUBESCO-980 `single:*` 仅 N=3；CREMA-D 全部可行。

### 2.5 预定终点（每个 level × enc，说话人为单位）
逐说话人在 Q 上按固定类别集算 UAR，先对 5 次参照抽样求平均，再对 3 次划分求平均，再说话人等权。报告全部条件的绝对 UAR，以及以下配对差（UAR 百分点）：
1. `balanced@5 − none`（E1）；2. `mean_e single:e@N − none`（E1；主值取该层级可行的最大 N，其余 N 亦报告并标 primary:false）；3. `single − balanced` 同 N（各层级至少在 N=3 可比）；4. 参照类别 e 在 `single:e` 下的**召回**与 `none` 下的召回之差；5. 估计器挽回：E2/E3/E4 − E1，条件 `single:*`；6. `other_single − single`、`other_balanced − balanced`；7. 逐说话人受害份额：`single@N` 下 UAR 比 `none` 低 >2 点的说话人比例。
不确定性：10,000 次说话人聚类 bootstrap 百分位 95% 区间（`default_rng(20260905)`，索引矩阵按层级说话人数生成一次共享），逐项、条件性。不报 p 值。

## 3. 模块 B1：复核预算（已有预测）

- 数据源：v2 `runs/main/units` 中 `arm=CTRL, cell=GG` 的全部单元（模型 cnn / resnet_se / transformer / hubert_base / wavlm_base_plus / wav2vec2_base，层级 ravdess / cremad / subesco_980，r0–r2）与 `arm=FT, cell=GG, model=wavlm_base_plus_ft`（seed 0,1）。predictions.csv 的 `logit_*` 按 `unit.json` 中 `predictions_sha256` 核验。
- 置信度 = softmax 最大概率（温度 1）。同一 `(level, model, r|seed)` 五折 OOF 预测合并后：
  - 复核比例 b ∈ {0.10, 0.20, 0.30}：全局阈值 τ_b = 置信度的 b 分位数；报告接受集错误率、错误捕获率（被拒集中错误数 / 总错误数）、接受集 UAR、逐说话人覆盖（min、p10、<70% 的人数、覆盖为 0 的人数）、被拒集类别构成；
  - 逐说话人等额配额变体（每人各拒 b）：接受集错误率；
  - AURC（最大概率）与 oracle AURC。
- 聚合：同 `(level, model)` 对 r/seed 取均值；主量（b=0.20 的接受错误率、错误捕获率、覆盖 <70% 人数）给说话人 bootstrap 区间。

## 4. 模块 B2：校准集说话人重叠

- 划分：`ctrl__<level>__GR__r` 与 `ctrl__<level>__GG__r`，r0–r2，同一 r 外层测试相同；每折 fit / val / test 按划分表。
- 模型与拟合：
  - `ridge_<enc>`（CPU，3 编码器）：M_glob 口径（StandardScaler+Ridge）在 fit 上训练，输出 val 与 test 的 decision_function。
  - `cnn`（GPU）：v2 `p1_frozen` 引擎与冻结配置（config_sha256 同 v2 CTRL cnn），fit 训练、val 早停（与 v2 相同），**额外导出 val 的 logits**；r0–r2，seed_index 0。
  - `wavlm_ft`（GPU）：v2 `wavlm_partial_ft` 引擎与冻结配置，同样导出 val logits；仅 r0，seed 0（与 v2 FT 臂一致）。
  引擎代码复制到 `v3/deploy/engines_deploy.py` 后只加导出，不改 `v2/`。
- 阈值规则（主）：固定预算 b=0.20，τ = val 置信度的 0.20 分位数；承诺量 = val 上的接受错误率 r̂ 与覆盖 0.80；实现量 = test 上的覆盖、接受错误率、逐说话人覆盖；差距 = 实现 − 承诺。次要规则：目标风险 r* = 0.5 × val 错误率，τ = 使 val 接受错误率 ≤ r* 的最小拒判比例；报告 test 实现风险与覆盖。
- 终点：同 `(level, model, r, fold)` 下 GR − GG 的：实现覆盖差、接受错误率差、承诺差距差；说话人等权（按测试说话人配对）；bootstrap 区间同上。GR 与 GG 的 test 预测各自也进入 B1 式覆盖统计作为对照。

## 5. 锁定、执行与核验

1. `plan.py` 生成 `plan.json`：全部单元（A：level×enc×r×fold 训练单元 + 条件表；B2：level×model×cell×r×fold）、可行性表、种子、划分表 SHA、特征缓存 SHA、manifest SHA、代码文件 SHA、配置。生成后 `lock.py` 写 `PLAN_LOCK.json`，提交并推送后方可正式运行；先导只允许 1 折计时与格式检查，不看科学量。
2. 输出契约（供独立核验）：
   - A：`work/A/units/<unit_id>/{model.json, predictions.csv.gz, DONE}`；`predictions.csv.gz` 列 `condition,estimator,draw,relative_path,speaker,y_true,y_pred`（gzip，仅预测标签，不存分数）；`DONE` = 该 gz 文件的 sha256。
   - B2：`work/B2/units/<unit_id>/{unit.json, val_predictions.csv, test_predictions.csv, DONE}`，预测列 `relative_path,speaker,y_true,y_pred,logit_0..logit_{C-1}`（Ridge 用 decision_function 填 logit 列）；`DONE` = "<val_sha256> <test_sha256>"。
   - B1 只读 v2 单元（predictions.csv 的 sha 与 unit.json 一致方可使用），输出 `results/B1/per_speaker.csv` 等。
   - 结果：`results/<module>/endpoints.json` = {"program","module","endpoints":[{"id","level","model","comparison","quantity"/"condition"/"estimator"（按模块）,"estimate","ci95":[lo,hi],"n"}],"absolute":[...],"inputs":{...}} 与 `per_speaker.csv`（A：level,enc,condition,estimator,speaker,uar；B1：level,model,b,speaker,coverage,accepted_error,n_utts；B2：level,model,cell,speaker,coverage,accepted_error）。
   - 单元身份：`unit_id` = sha256(canonical JSON of {program, module, level, model/enc, cell, r, fold, seed_index, train_seed, split_key, split_sha256, config_sha256/config}）。
3. `score.py` 在全部单元 DONE 且完整性检查通过后一次性评分，写 `results/*.csv` 与 `score.json`；`verify.py` 由另一实现者仅依据本文件从原始 predictions 复算全部预定量与区间（容差 1e-9），写 `verification.json`。
4. 云 GPU：仅 B2 的 cnn 与 wavlm_ft；独立环境，源码/配置/缓存 SHA 与本地一致；音频上传前后 sha256 核对；输出快照回传后本地评分。GPU 释放/停止由作者控制，代码不自动开关实例。
5. 本地 CPU 任务：BelowNormal、≤2 线程、`CUDA_VISIBLE_DEVICES=-1`，不干扰在跑的 N14R。

## 6. 写作边界

可以写：在受控演出语料、说话人排他协议下，(a) 无标签登记归一化的收益依赖参照情绪组成，偏斜参照可使 UAR 低于不归一化，便宜估计器挽回多少；(b) 固定复核预算下的接受错误率、错误捕获率与逐人覆盖分布；(c) 用说话人重叠的验证集选阈值会如何误估复核工作量与残余错误。
不可以写：在线部署已验证、真实通话/会话级预算、最优阈值或最优登记条数的普适建议、任何"首次"表述、把描述性区间当假设检验。

## 7. 解释冻结（锁定前由独立验证器提出，2026-09-05 定）

1. 逐人 UAR 的"固定类别集"按 v2 惯例：某说话人 Q 中缺失的类别跳过（真实 Q 集每人全类别齐全，验证器逐一核验并报告例外）。
2. 模块 A 中 `none` 记为 condition=none / estimator=none / draw=0；`oracle_all` 仅 naive、draw=0。
3. 终点 3 在 single 与 balanced 都可行的每个 N 上报告，主值为最大公共 N。终点 2、4、7 主值取最大可行 single N，其余 N 标 primary:false；终点 5 每个可行 N 都报告；终点 6 在 other_* 与本人条件都可行的每个 N（3、5）上报告，主值为最大 N，估计器固定为 naive（E1）。
4. 终点 4 的召回逐情绪 e 报告并给跨 e 平均；终点 7 的受害指示基于该说话人跨 e 平均的 single UAR（逐 e 变体作补充）。
5. B1 阈值 τ_b 为置信度的线性分位数，接受当且仅当置信度 ≥ τ_b；错误率与捕获率按语句合并计算；p10 用线性分位数；"覆盖 <70%" 为严格小于。聚合对全部 (r, seed) 组做等权平均（cnn 在 RAVDESS/SUBESCO-980 有 9 组，FT 2 组）；bootstrap 固定 τ 只重抽说话人。AURC：置信度并列按文件顺序，AURC = mean_k risk(k)，oracle AURC = mean_k max(0, k−n_correct)/k，不给区间。
6. B2 次要规则：在 val 上按置信度从低到高逐条增加拒判，取满足接受错误率 ≤ r* 的最小拒判数 k，τ₂ = 第 k 低的 val 置信度，接受当且仅当 ≥ τ₂。GR−GG 配对在 (level, model, r, fold) 内逐说话人计算；某说话人在 test 上无被接受语句时其接受错误率记 NaN，跨 r 取 nan-aware 均值并报告被丢弃人数；覆盖差距差与覆盖差相同，不单列。只有说话人等权的量给 bootstrap 区间。
7. 单元身份：A 的 unit_id 字段见 `plan/A.json` 的 `config.unit_id_fields`；B2 的 unit_id = sha256(canonical({program, module:"B2"} ∪ calib.py `IDENTITY_FIELDS` 所列字段))。SUBESCO 的句子划分盐使用基础语料名 `subesco`。
8. 独立验证器 `verify.py` 的统计实现随本锁定冻结；仅允许在锁定后调整其"端点名称映射层"以对齐评分器的命名（不得改动任何统计或阈值代码），每次调整以 git 提交披露并在 verification.json 中记录 verify.py 的 sha。
