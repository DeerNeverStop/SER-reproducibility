# 原v2绝对基线、FT对照与偏离记录核查

2026-09-07。仅用已完成v2的结果JSON、冻结评分器/计划及旧偏离与重训收据。没有读取新384程序的结果，没有新增统计检验、置信区间、模型拟合或预测，也没有修改论文稿及冻结研究。

## 1. 数据来源与聚合对象

主来源：[v2/evidence/main/results/results.json](../v2/evidence/main/results/results.json)，schema=`ser-v2-results-1`，run_id=`main`，4,943 completed units、0 void。文件SHA256：

`7e09783aed6003e1fbf47e89f839d061e5d3f5d1655bb9de2c8c02508577a016`

JSON已经保存真实`cell_runs["FT|<level>|<model>|<RR/GG>|r0|s<seed>"].speaker_uar`，因此无需重建logits或从图中读数。绝对组均值没有单独顶层字段；本次只对这些既有逐人值按冻结规则求平均。

具体步骤与[score.py:235–258](../v2/ser_v2/score.py:235)、[317–327](../v2/ser_v2/score.py:317)一致：

1. 只取相应`arm=FT, level, model, cell`且`complete=true`的cell-run。逐人UAR本来已经由该seed的五fold完整OOF预测计算，不能再把五fold UAR等权平均当成同一指标。
2. 取FT与匹配冻结模型、RR与GG四个组合共同拥有的seed交集。本次三个语料都恰为`{0,1}`，r均为0；因此这是**一个划分draw、两个训练seed**，不是两个新划分draw。
3. 对每个人先平均两个seed的UAR，再对人物等权平均：`A(model,cell) = mean_s mean_seed U(model,cell,seed,s)`。人物数分别24、91、20。
4. FT premium先作逐人`V_FT,RR(s)−V_FT,GG(s)`再取人物均值；增量再减同一个人的冻结premium。这与绝对均值之差在本次完整相同人物集下代数相等。
5. 该聚合也与原[paper_tables.py:50–58、149–157](../v2/tools/paper_tables.py:50)生成FT表的方式一致。它不是整体录音accuracy、whole-fold UAR或训练seed间不确定性估计。

对60组`level×cell×fold×seed`模型配对核对了split SHA、train_seed及fit/val/test条数，全部相等。计划共120个FT-arm拟合，即每库20次partial FT加20次matched frozen。[run_plan.py:338–356](../v2/ser_v2/run_plan.py:338)。上述计划、unit_configs、registry及`score.py/stats.py/train.py`的当前字节均匹配原`PINS_rc2.json`。

## 2. 论文FT表应同时显示的绝对值

RR/GG列单位为逐人等权UAR百分数；最后一列是两者差，单位pp。下表保留八位小数方便校对，论文可用两位。

| 语料 | 模型 | RR UAR (%) | GG UAR (%) | RR−GG (pp) |
|---|---|---:|---:|---:|
| RAVDESS | WavLM partial FT | **78.57607887** | **59.57031250** | **19.00576637** |
| RAVDESS | WavLM frozen, same regime | **36.60249256** | **31.77083333** | **4.83165923** |
| CREMA-D | WavLM partial FT | **70.33631814** | **68.57818616** | **1.75813198** |
| CREMA-D | WavLM frozen, same regime | **50.48942206** | **51.33988417** | **−0.85046211** |
| SUBESCO-980 | WavLM partial FT | **59.89795918** | **53.11224490** | **6.78571429** |
| SUBESCO-980 | WavLM frozen, same regime | **37.04081633** | **38.82653061** | **−1.78571429** |

这六行不能将`wavlm_base_plus_frozen_sr`与原factorial的`wavlm_base_plus` standardized probe混用。匹配冻结版与FT共享原音频预处理、随机3秒crop、10秒eval上限、batch16、最多15轮/patience3及线性头训练规则；编码器冻结并不等于跳过训练态，`model.train()`仍令encoder dropout处于训练模式，且pooled representation未作fit-fold标准化。[实际engine](../v2/ser_v2/train.py:265)。它不是对冻结表征能力的通用最优基线。

同样预算规则也不意味着两个模型实际训练轮数或checkpoint相同。绝对UAR较弱以及CREMA/SUBESCO的冻结premium为负，必须与增量共同呈现；不能只报“FT premium增加”就推出微调一般性地增强了说话人记忆。负点估计也不证明冻结模型真正受益于更严格的划分。

### 原有推断字段一致性

本次绝对值导出的逐人premium与原R09–R11，及两模型premium差与原N11–N13，最大数值差为**1.78×10⁻¹⁵ pp**。以下CI/p全部来自原JSON，未重新估计：

| 原假设 | 估计对象 | mean [原95% speaker bootstrap CI]，pp | 原Holm p | 冻结身份 |
|---|---|---|---:|---|
| R09 | RAVDESS FT RR−GG | 19.00576637 [14.77864583, 23.16313244] | 0.00010907694 | confirmatory，R族m=11 |
| R10 | CREMA-D FT RR−GG | 1.75813198 [0.84574442, 2.68565035] | 0.00093624739 | confirmatory，R族m=11 |
| R11 | SUBESCO-980 FT RR−GG | 6.78571429 [3.87755102, 9.79591837] | 0.00093624739 | confirmatory，R族m=11 |
| N11 | RAVDESS FT premium−frozen premium | 14.17410714 [8.82161458, 19.59658668] | 无 | estimation-only |
| N12 | CREMA-D FT premium−frozen premium | 2.60859409 [1.47375880, 3.74400268] | 无 | estimation-only |
| N13 | SUBESCO-980 FT premium−frozen premium | 8.57142857 [5.00000000, 12.50000000] | 无 | estimation-only |

原稿FT三行的两位小数及其Holm p<.001正确。N11–N13的`p_holm=null`、`a_priori_status=estimate`；它们不因CI正向而变成确认性结论。冻结对照premium本次只补绝对表及差值，没有新增CI或显著性判断。

## 3. 主factorial绝对值备查

以下同样由既有`cell_runs.speaker_uar`聚合。CTRL只取`seed_index==r`的三个主要replicate，排除交叉重复诊断；逐人先跨replicate平均，再跨组内三个模型平均，最后跨人平均。Scratch组是CNN/ResNet-SE/Transformer；probe组是HuBERT/WavLM/wav2vec 2.0。列均为UAR百分数。

| 语料 | 模型组 | RR | RG | GR | GG |
|---|---|---:|---:|---:|---:|
| RAVDESS | scratch | 59.85449735 | 53.77914187 | 47.93010086 | 41.61809689 |
| RAVDESS | standardized probe | 73.22668651 | 70.02831515 | 62.96089616 | 61.44696594 |
| CREMA-D | scratch | 57.55653416 | 56.15867818 | 56.07926702 | 54.99822895 |
| CREMA-D | standardized probe | 68.53412833 | 67.92113903 | 66.89960257 | 66.36991925 |
| SUBESCO-980 | scratch | 47.76643991 | 44.53514739 | 43.71882086 | 40.55555556 |
| SUBESCO-980 | standardized probe | 59.92063492 | 57.66439909 | 56.33786848 | 54.58049887 |

两位小数可以生成原稿主表的RR−GG/RG−GG/RR−RG点估计；正式发表时应先以未四舍五入值相减，再格式化。不要用本表均值自行相减置信区间端点；主表CI仍应取原人级配对结果。曝光11.07/2.68/10.09的另一次核查见[EXPOSURE_CLAIM_AUDIT.md](EXPOSURE_CLAIM_AUDIT.md)。

## 4. 三条偏离实际记录的内容

[主运行deviations.jsonl](../v2/evidence/main/deviations.jsonl)与[results中的副本](../v2/evidence/main/results/deviations.jsonl)本次逐字节相同，均有entry 1、2、3。两文件SHA256：

`c8575904d6e8d1515d752aa84f9d2f82c2304b41f55e8f85ef14e81b9a116237`

它们是同一预算事件的过程、闭合和作者裁决，不是三次修改科学假设：

| 条目 | 记录内容 | 新稿必须保留的边界 |
|---|---|---|
| 1，budget | 运行中发现scratch实际耗时高于成本模型；多队列并行使单进程只看到自己队列的cap累计；执行方继续全部单元，未依预定顺序截断。曾预测总项目也会超50h。 | 预算/截断政策发生偏离，不能因源码和registry不变就写“无偏离”。当时关于总量超限的是预测。 |
| 2，budget-closure | 记录最终累计unit `gpu_seconds`：CTRL19.33h对cap16、MECH2X2 13.21对7、HPO7.20对5；FT2.69对26。主体42.42h，加MECHID约0.5h，总约42.9h。另记录重训50/51字节相同。 | **三个分实验超限，总项目50h未超**；不能沿用条1的总量超限预测。unit计时含CPU/等待且队列重叠，累计42.9h不是独占GPU忙碌时长或墙钟租期。 |
| 3，author-ruling | 事后记录作者在FT仍运行、一次性评分之前授权豁免这些分实验上限，要求全部跑完；未截断、未void，结果与验证文件不变。 | 应写“记录注明决定在评分前作出，裁决后补记”。本条的时间戳证明记录时间，不能单独当成实时公开的评分前凭证。`hypotheses:[]`仅表示不改scorer的逐假设裁决映射，不代表无需披露。 |

[冻结scorer:580–595](../v2/ser_v2/score.py:580)只把条目中`hypotheses`列出的假设映射到偏离号；因此空列表不会自动附加truncation后缀。科学数值保持原样与存在资源政策偏离可以同时成立。

### 可直接使用的短句

中文：“运行期间，CTRL、曝光和HPO三个分实验的累计计时超出预设上限。偏离记录注明作者在评分前豁免这些上限并完成全部4,943个单元，裁决随后补记；未执行预定截断，总项目累计约42.9小时，仍低于50小时上限，科学配置和结果保持原样。”

英文：“Three arm-level runtime caps were exceeded and waived; the deviation log records an author decision before scoring, documented retrospectively. All 4,943 planned units were completed without applying the planned truncation rule; aggregate recorded runtime remained below the 50-hour program cap, and scientific configurations and results were unchanged.”

时间措辞据原日志，不是本次独立重建当时全部会话可见性；可以给出日志入口，不能将其简写成“fully preregistered execution without deviations”。

## 5. 重训例外如何披露

条2提到的[retrain_report.json](../v2/evidence/main/results/retrain_report.json)实际记录`n_selected=51, identical=50, different=1, missing=0, bitwise_pass=false`。唯一例外是SUBESCO-980 / RR / fold0 / `wavlm_base_plus_frozen_sr`，UID：

`201bcd0a6b4c449ea05a2b989205dcc2658cd82d5d8709001939aba6ca903172`

原run与check均选epoch15；196个测试预测的标签一致率为97.44897959%，fold UAR为原33.67346939%、重训34.18367347%，最大logit绝对差0.10155308。该文件SHA256：

`f95fe56479febc81e5f56e3a424a50a5d70d5f54a027e1458eae1dabb5f7e471`

短句：“51个抽取单元的重训中，50个预测文件逐字节一致；一个使用FP16训练的冻结WavLM对照出现差异，原结果被保留并披露。”该收据支持这次抽样复现的具体结果，不支持宣称全FT分支逐字节可复现、所有重训误差必小于1 pp，或已实验确认唯一原因是某个CUDA内核。它与读取同一存档的数值验证通过是不同检验。
