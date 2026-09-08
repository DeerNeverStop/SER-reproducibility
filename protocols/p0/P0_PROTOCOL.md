# P0 系统检索、纳排与静态审计协议

- 协议版本：1.0（冻结于批量检索前）
- 冻结时间：2026-08-10 14:48:27 UTC-04:00
- 适用阶段：P0（CPU + 网络检索；不运行外部仓库代码，不进入 GPU 复现）
- 研究对象：在 RAVDESS、CREMA-D、IEMOCAP、EmoDB 至少一个数据集上报告或实现语音情感识别（SER）评估的公开代码库

本协议预先规定候选库如何被发现、纳入、去重和判定。批量检索开始后不得为了改变结果而回改本文件；任何必要偏离只能追加写入文末“协议偏离日志”，并同步记录到 `PROGRESS.md` 和最终 `REPORT.md`。

## 1. 研究问题与 P0 边界

P0 只回答公开实现采用了什么评估协议，以及代码中是否存在几类可审计的泄漏或选择风险：

1. 数据划分是否保证说话人互斥；
2. 数据依赖的归一化/标准化统计量是否只在训练部分拟合；
3. 最终测试集是否被用于早停、选择 epoch、选择 checkpoint 或调参；
4. 数据增强产生的样本或随机增强是否进入最终测试评估；
5. 报告来自单次划分还是多折/多 seed，是否报告方差。

P0 是静态审计，不验证仓库能否复现论文数值，不安装仓库依赖，不运行下载脚本、训练脚本或未知二进制，不做 GPU 实验。允许的本地操作限于网页/PDF阅读、`git clone`、文本搜索、读取源码和配置、记录 commit SHA，以及对审计表本身做无网络副作用的校验。

## 2. 样本单位、总体与时间界限

### 2.1 样本单位

- 主单位是“唯一代码仓库”，`survey_table.csv` 一行一个仓库。
- 同一仓库覆盖多个目标数据集时不拆行；所有可能随数据集变化的判定用 `DATASET=value` 映射记录，例如 `RAVDESS=random | IEMOCAP=LOSO`。
- 同一论文有多个实质不同的官方仓库时分别纳入并互相标注；镜像和无实质改动的 fork 只保留上游库。
- 同一仓库实现多篇论文时，记录与本次目标数据集结果直接对应的论文；不能唯一对应时标为 `multiple/unclear`，不猜测。

### 2.2 目标总体与可推断范围

目标总体是截至 2026-08-10 仍可公开访问、可被下述检索框架发现的 SER 代码库。核心统计只描述“本检索框架捕获且满足纳入标准的公开代码实现”，不外推到无公开代码的全部 SER 论文。作者关联库和第三方复现库分别统计；二者合并结果只作敏感性分析。

### 2.3 目标样本量与停止规则

- 目标为至少 25 个完成静态审计的合格仓库；达不到则报告实际数量和原因。
- 不因某个仓库看起来“协议好”或“协议差”而调整纳入顺序。
- 四个数据集、三个主来源都必须实际检索；达到 25 后仍完成已经进入全文/代码筛选的候选。
- 每个来源按第 3 节的有界规则完成后停止。若合格候选超过审计容量，先完成所有作者/论文明确关联的仓库，再按候选首次发现时间与规范化仓库 URL 排序；不得查看审计结局后抽样。

## 3. 信息源与预注册检索式

所有查询的实际执行时间、原始查询、结果页/接口 URL、检查结果数和新增候选数逐条写入 `search_log.csv`。搜索不限定发表年份和代码语言。

### 3.1 数据集名称变体

| 规范名 | 检索变体 |
|---|---|
| RAVDESS | `RAVDESS`; `Ryerson Audio-Visual Database of Emotional Speech and Song` |
| CREMA-D | `CREMA-D`; `CREMAD`; `Crowd-sourced Emotional Multimodal Actors Dataset` |
| IEMOCAP | `IEMOCAP`; `Interactive Emotional Dyadic Motion Capture` |
| EmoDB | `EmoDB`; `EMO-DB`; `Berlin Database of Emotional Speech` |

### 3.2 GitHub

对每个规范数据集名执行以下两族仓库检索，默认按 GitHub “Best match”顺序审查最多前 100 条可见结果；若接口或页面不足 100 条，则审查全部返回结果。

1. `<DATASET> "speech emotion recognition"`
2. `<DATASET> emotion recognition`

补充使用网页检索式 `site:github.com <DATASET> ("speech emotion recognition" OR "speech emotion classification")`，只用于发现 GitHub 自身检索遗漏项。每个仓库规范化为 `https://github.com/{owner}/{repo}`，去掉 `.git`、尾斜杠、大小写差异和跟踪参数。

### 3.3 Papers with Code

逐一检查四个数据集的 dataset 页、Speech Emotion Recognition task 页及其结果条目。对每个相关条目跟随 `Code`/官方实现链接，并记录：结果条目、论文链接、仓库链接、是否由 Papers with Code 标为 official。页面若失效，记录失效状态，不用搜索摘要代替原页面证据。

### 3.4 arXiv

对四个数据集分别执行：

`all:"<DATASET>" AND (all:"speech emotion recognition" OR all:"speech emotion classification")`

按 arXiv API/站点相关性默认顺序检查最多前 100 条，跟随摘要页、PDF、`Code` 链接以及论文正文中明确给出的仓库 URL。若长名称检索能找到简称检索未覆盖的论文，也进入候选框。

### 3.5 前向/后向追踪

从已纳入论文的 README、论文正文和参考文献中发现的相关公开实现可作为“追踪发现”候选，但必须单独标记 `discovery_source=snowball`，不能伪装成主检索命中。仅因为搜索引擎摘要提及仓库而无法确认实际链接的，不纳入。

## 4. 候选登记、去重与仓库冻结

每个发现项先进入 `candidate_log.csv`，先筛选、后审计。登记字段至少包括首次发现时间、发现来源、查询编号、论文、仓库 URL、目标数据集、作者关联性、筛选状态和排除理由。

去重顺序：

1. 规范化 URL 完全相同：合并来源命中；
2. GitHub 明示 fork 且没有与目标评估有关的实质修改：保留上游；
3. 仓库改名/转移：保留 GitHub 当前 canonical URL并记录旧 URL；
4. 内容镜像：以有提交历史、论文链接或作者链接的一方为主，镜像列入排除日志；
5. 仅模型名称相同但实现独立：不去重。

纳入后将仓库 clone 到 `sources/{owner}__{repo}/`，只读审计，并记录远端 URL、默认分支、完整 HEAD commit SHA、commit 日期和访问时间。所有代码证据同时写本地相对路径与冻结 commit 的永久链接；审计期间不 pull。若仓库过大或 Git LFS 阻断，允许用 GitHub commit 页面逐文件读取，但要记录限制。

## 5. 纳入与排除标准

### 5.1 纳入标准（须全部满足）

1. 截止检索日可公开访问的代码仓库；
2. 代码或其明确对应论文在四个目标数据集至少一个上进行 SER 分类/回归评估；
3. 仓库包含与目标结果有关的数据准备、划分、训练或评估代码中的至少一部分；即使关键划分脚本缺失，也保留并将相应项判为 `unknown`，避免因报告不完整造成选择偏差；
4. 能确定唯一仓库身份并冻结到具体 commit；
5. 不是仅有论文链接、仅有模型权重、仅有结果截图或完全空壳的仓库。

作者官方/作者关联实现、明确的第三方复现和未能确认关联性的实现都可纳入，但用 `repo_relation` 分层，不混称“论文官方代码”。没有正式论文的公开实现可进入补充代码库层；核心“论文关联库”比例不包含它们。

### 5.2 排除标准（使用首个成立的主理由）

- `E1_not_SER`：任务不是语音情感识别；
- `E2_target_not_evaluated`：目标数据集只用于预训练、演示或举例，没有评估；
- `E3_no_relevant_code`：只有论文、权重、截图或占位文件，没有任何相关数据/训练/评估代码；
- `E4_unavailable`：仓库私有、删除或检索日无法访问；
- `E5_duplicate_or_fork`：上游重复、镜像或无实质改动 fork；
- `E6_not_repository`：单独 gist、论坛粘贴或不可冻结的临时代码片段；
- `E7_non_target_modality`：只做面部/视频/文本情感且没有可分离的语音评估；
- `E8_other`：必须附具体、可复核说明。

不得因为准确率低、无方差、协议不清楚、代码质量差、仓库不可运行或涉嫌泄漏而排除。

## 6. 两阶段筛选流程

### 阶段 A：标题/README/论文摘要筛选

判断是否可能满足纳入标准。信息不足时保守进入阶段 B，不在阶段 A 猜测排除。记录筛选证据 URL。

### 阶段 B：代码资格筛选

确认目标数据集和 SER 评估代码的存在，执行去重，冻结 commit。资格筛选与风险判定分开：只有资格确定后才查看并编码协议结局。

筛选状态只能是 `pending`、`include`、`exclude`、`awaiting_access`。最终报告给出候选流程计数：检索命中、去重后、全文/代码筛选、纳入、各排除理由。

## 7. 代码审计顺序与证据规则

每库按固定顺序审计，以减少只寻找“有问题”证据的确认偏差：

1. README、论文链接、运行入口和配置；
2. 数据集读取器与标签/说话人 ID 解析；
3. 数据划分或预生成 split 加载；
4. 预处理、特征缩放和模型输入归一化；
5. 增强定义及其调用位置；
6. 训练循环、验证循环、checkpoint 与 early stopping；
7. 最终测试调用；
8. seed、折数、重复运行和汇总/方差代码；
9. README/论文报告数值及其口径。

推荐文本检索词包括但不限于：`train_test_split`、`KFold`、`GroupKFold`、`LeaveOneGroupOut`、`speaker`、`actor`、`session`、`fold`、`split`、`StandardScaler`、`MinMaxScaler`、`normalize`、`mean`、`std`、`fit_transform`、`augment`、`noise`、`shift`、`stretch`、`pitch`、`validation_data`、`val_`、`test_`、`early_stop`、`checkpoint`、`best`、`seed`。

### 7.1 证据最低标准

- 每个非 `unknown` 判定必须有直接支撑该判定的 `path:Lx-Ly` 代码证据，并尽量附冻结 commit 的永久链接和不超过必要长度的代码摘录。
- 证据必须覆盖“定义”和“调用/作用域”。例如只看到 `StandardScaler` import 不能判泄漏；要追到 `fit` 的输入来自哪个 split。
- README/论文可以证明作者声称和报告数字，但不能覆盖相反的代码行为。冲突时“实现判定”以代码为准，并单列 `paper_code_discrepancy=yes`。
- `no` 不是“没有搜到 yes”。只有追清活动代码路径，看到训练集专属拟合/增强、独立验证集或固定 epoch 等正面证据后才能判 `no`。
- `unknown` 也要记录已检查路径、搜索词以及缺失环节；不能只留空。
- 只审计与仓库文档所指主结果/默认配置对应的活动路径。示例、废弃和未调用代码不能作为主判定，除非主路径无法确定，此时标 `unknown` 并解释。

### 7.2 证据强度

- `high`：定义、调用和数据流均由活动代码直接闭合；
- `medium`：关键代码明确，但活动配置、预生成文件来源或论文对应关系有一处需合理推断；
- `low`：只能从不完整代码/文档间接判断。`low` 结论在主统计中按 `unknown` 做敏感性分析，不伪装成确定结论。

## 8. 判定字典

所有字段允许按数据集分别编码。原始判定保留细粒度值；汇总时如需二分，必须公开合并规则并做 `unknown` 敏感性界限。

### 8.1 划分方式 `split_category`

- `random`：样本/utterance 级随机切分或普通 KFold，代码未用说话人/演员/会话组约束；包括先增强再随机切分。
- `speaker_independent`：训练、验证、测试按说话人组互斥的 holdout、GroupShuffleSplit 或 GroupKFold；须证明 group 确为 speaker ID。
- `LOSO`：每次完整留出一个说话人，或 IEMOCAP 中完整留出一个 session 且该 session 的两位说话人不在训练中。leave-one-session-out 另在细节字段注明，不把普通随机 session 切分算 LOSO。
- `predefined_speaker_independent`：加载预定义 split，且仓库中的列表/生成代码能证明说话人互斥；汇总时并入 `speaker_independent`。
- `mixed`：同一仓库不同目标数据集或不同主结果使用不同类别；必须给出逐数据集映射。
- `unknown`：缺少 split 生成、预生成列表不可见、说话人 ID 数据流追不清，或存在多个实现而主结果无法对应。

另记 `speaker_disjointness=enforced/not_enforced/observed_overlap/unknown`。普通随机切分可判 `not_enforced`；只有代码、清单或可重建 split 明确显示同一说话人跨集合时才判 `observed_overlap`，不把概率推断写成已观察事实。

### 8.2 归一化泄漏 `normalization_leakage`

- `yes`：数据依赖的 scaler/均值/方差/量化校准在切分前对全量样本拟合，或显式对 train+val/test 联合拟合后用于训练/测试。
- `no`：统计量只在训练部分拟合，并原样应用于 val/test；或仅使用固定常数、预训练模型固定统计量、单样本内部归一化，且不存在跨样本测试信息流。
- `not_applicable`：活动路径没有数据依赖的归一化/标准化；须有活动路径证据。
- `unknown`：预计算特征/统计量来源不公开，或数据流不足以判断。

数据集官方全局常数与利用本次测试集合估计统计量分开记录；前者不自动视为泄漏。

### 8.3 测试集用于选择 `test_selection`

- `yes_explicit`：最终测试指标直接驱动 early stopping、最佳 epoch/checkpoint、超参或模型选择。
- `test_exposed_each_epoch`：代码每轮/频繁计算最终测试指标，但没有自动选择语句；这是人工选择风险，不能武断声称实际选模。
- `no_separate_validation`：有与最终测试集分离的验证集负责选择，测试只在选择完成后调用。
- `no_fixed_training`：固定训练日程且最终测试仅在训练结束调用，没有基于最终测试的选择。
- `unknown`：`validation_data`/变量命名与真实集合对应不清、外部训练日志缺失，或主入口不明。

主统计至少分别报告 `yes_explicit` 和 `yes_explicit + test_exposed_each_epoch`，不把二者混为同一个事实陈述。

### 8.4 增强混入测试 `augmentation_leakage`

- `yes`：增强在划分前生成并与原样本共同随机切分；增强管道作用于 test/eval loader；或 train 中样本的增强近重复明确进入最终测试。
- `no_train_only`：增强只挂在训练 loader/训练索引上，val/test 明确使用无增强路径。
- `not_applicable`：主结果活动路径没有数据增强，且调用链可闭合。
- `unknown`：增强后的预生成数据来源或 split 时序不公开，或路径不明。

只在测试时做固定的标准信号预处理不算“数据增强”；test-time augmentation 单列 `test_time_augmentation`，不自动当作训练泄漏。

### 8.5 重复评估与方差

`evaluation_repetition` 取以下一个或组合：

- `single_split_single_seed`
- `single_split_multi_seed`
- `kfold_single_run`
- `kfold_multi_seed`
- `LOSO`
- `repeated_holdout`
- `unknown`

分别记录 `n_folds`、`n_seeds`、seed 值/生成方式、是否跨折汇总。`variance_reported` 只能是 `sd`、`se`、`ci`、`other`、`none`、`unknown`；论文只给一个平均值而没有离散度记 `none`。跨折标准差与跨 seed 标准差不可互换，在备注中注明来源。

### 8.6 论文报告数字

记录与目标数据集和代码路径最接近的主要指标，至少包含数据集、任务/类别数、指标名、数值、单位、论文表/页或 README 行号、论文对协议的文字描述。若有 UAR/WA/accuracy 等多个指标，原样分别记录，不自行换算。若无法把数字唯一对应到冻结代码路径，标 `result_code_link=uncertain`。

## 9. 复核、分歧处理与质量控制

1. 初审者填写证据与判定，不先看跨库汇总比例。
2. 复核者按冻结 commit 重新打开全部 `yes`、全部 `unknown/low`，以及其余项目中按规范化 URL 排序每 3 个抽 1 个的至少 33% 样本。
3. 复核只看原始文件和协议，不以初审理由代替代码阅读；记录 `agree/disagree` 和新证据。
4. 分歧由主审回到定义、调用与数据流三层裁决；仍不能闭合则降为 `unknown`，不采用更“确定”或更戏剧性的标签。
5. 自动校验 CSV：仓库 URL、commit SHA、必填枚举、每项 evidence 字段、重复 URL、逐数据集映射一致性。
6. 对 3 个仓库做全字段试编码后，只允许澄清文字或增加字段，不得改变既有结局以迎合观察结果；任何实质修订进入偏离日志，并保留旧值。

本项目由自动化研究代理执行，复核表示不同审计遍次/代理的交叉核对，不宣称为两名独立人类编码者；这是最终报告必须披露的方法学局限。

## 10. 汇总计划

- 给出检索流程计数和各排除理由。
- 分母显式区分：全部纳入库、论文/作者关联库、第三方实现库、可判定子集。
- 每类风险同时报告 `n/N` 与比例；`unknown` 不从总分母静默删除。
- 对关键二分类给界限：保守下界（unknown 全按无风险）与上界（unknown 全按有风险），并给 Wilson 95% CI；由于样本不是概率抽样，CI 仅表示样本比例的不确定性，不代表完整文献总体的抽样覆盖误差。
- 分数据集、年份和仓库关联性做描述性分层；若单元过小不做显著性检验。
- P0 不用观察到的相关性声称因果，也不估算“协议溢价”；后者留给经作者另行启动的 P1 配对实验。
- 明确报告公开代码选择偏差、搜索排序/索引覆盖偏差、静态代码与实际运行配置不一致风险、同一论文多库/同一库多数据集的依赖性，以及自动化审计误差。

## 11. 预定产出与目录

- `P0_PROTOCOL.md`：本协议；
- `search_log.csv`：每次实际检索记录；
- `candidate_log.csv`：所有候选、去重和排除轨迹；
- `sources/`：冻结的只读审计副本（不运行）；
- `survey_table.csv`：一行一个纳入仓库的主表；
- `survey_notes.md`：逐库证据链、冲突与复核记录；
- `PROGRESS.md`：按阶段只追加流水；
- `REPORT.md`：P0 样本框结果、局限和是否具备进入 P1 的条件。

## 12. 协议偏离日志

当前无偏离。后续只能在此节末尾追加：时间、原规则、偏离内容、原因、受影响候选/字段及敏感性处理。

- `2026-08-10 15:49:24 UTC-04:00`：冻结名册曾把 `nhattruongpham/Light-SERNet`、`mnamvarpour/Neural_Network_Emotion_Recognition` 的同 owner 替代库误记为 canonical redirect，并把 `hellolzc/SpeechEmotionRecognition-emodb` 暂时归并到 `hellolzc/speech-emotion-recognition`。Stage-B GitHub 身份核对显示前两个旧 URL 均为 404 且无 redirect，后两个 hellolzc URL 是独立仓库，而当前连字符 URL 是 `Renovamen/Speech-Emotion-Recognition` 的 fork。处理：保留旧精确 URL 的 E4/E5 轨迹，把三个实际审计库明确标作“独立替代/独立发现”，并将名册第 33 项恢复为最初发现的独立 EmoDB 仓库。该修订只纠正仓库身份与纳排轨迹，不改变预定顺序、五项风险定义或任何审计结局；候选统计同时保留 fork 上游为 pending，以便敏感性追踪。
- `2026-08-10 16:26:25 UTC-04:00`：按预注册交叉复核流程对 32/32 个纳入库完成第二遍静态核对。21 个库全部同意；11 个库的 30 个字段依据冻结 commit 证据修正，不能闭合的 `no/direct` 保守降为 `unknown/uncertain`。Mak-Sim 的复核建议写作“5 个折内 seed 值和 1 个重复外层 schedule”；主审按字段定义将 `n_seeds` 细化为 `1`，并把五个折内固定值保留在 `repetition_by_dataset`，避免把折内初始化误算为独立外层重复。处理：旧值、复核建议、证据、裁决值和理由分别永久保存在四份 `review_*_cross.csv`、`work/review_adjudications.csv/md` 与 `survey_notes.md`；不更改预注册风险定义、纳排或样本顺序，所有汇总均在裁决后重算。
- `2026-08-10 18:32:45 UTC-04:00`（P0.6 框边界追加）：Papers with Code 的 RAVDESS、
  CREMA-D、IEMOCAP、EmoDB dataset URL 与 Speech Emotion Recognition task URL 在检索日均
  从原站 HTTP 200 跳转到与查询无关的 Hugging Face `papers/trending`，可检查原结果条目为 0、
  新候选为 0；旧缓存/搜索摘要未用于替代。因此 P0 的实际捕获框是 GitHub + arXiv，加上明确
  标记的限定网页解析/雪球追踪，不包含有效的 Papers with Code 结果层。预注册替代源
  Semantic Scholar 的 8 条查询各请求并重试一次，16/16 均为 HTTP 429、结果槽为 0，故本轮
  替代源也不可用；这不是“没有遗漏工作”的证据。详情与诊断偏离见 `P0_6_PROTOCOL.md`、
  `p0_6_search_log.csv` 和 `P0_6_REPORT.md`；不回填已冻结 P0.5 框。

## 13. P0.6 检索框边界补记

- `2026-08-10 18:23:44 UTC-04:00`：原计划中的 Papers with Code 四个数据集页与 SER task 页在检索日均返回 HTTP 200 后跳转至无关的 Hugging Face trending 页面，故该来源实际贡献为零；缓存摘要没有替代官方页面证据。原 P0 的可操作检索框因此应准确表述为“GitHub 仓库检索 + arXiv 结果筛查”，而不是“GitHub + arXiv + Papers with Code”。
- 该框只覆盖可由上述路径发现、且最终可规范化到公开 GitHub 仓库的实现；不覆盖闭源论文、未被索引或排序上限截断的项目、只托管在其他代码平台的实现，也不能从 383 个候选反推出全部 SER 文献的覆盖概率。
- 替代来源 Semantic Scholar Graph API 的查询、上限、纳排、论文/仓库两层去重及增量判定已在任何请求前冻结于 `P0_6_PROTOCOL.md`。P0.6 的新增记录不回填 P0.5 已冻结的 331 项随机抽样框，也不事后改变原 P0 的纳排或估计；它只用于量化另一文献索引源相对于既有可观察框的增量与边界。
