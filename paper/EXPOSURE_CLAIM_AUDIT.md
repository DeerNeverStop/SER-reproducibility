# 原v2匹配曝光棋盘：实际控制与结果核查

2026-09-07。本次只查旧MECH2X2的冻结源码、真实元数据、计划索引和已完成结果，并作CPU元数据重建；没有拟合或推理模型，没有读取新final program的科学结果。原submission及冻结研究保持原样；只将当前定位文件的“控制验证条件”改为下述更精确的范围。

**结论：同一配对面板内，test及val确为相同录音，fit条数及选模规则也匹配。但是，加入prompt曝光会使原本未见的val人物在fit中可见；fit类别分布变化还会改变加权val CE。因此“共同val”成立，“验证条件全部不变”不成立。三个已报告效果与CI、Holm判定均正确，应解释为这些指定训练替换及选模程序的总对照，不能解释为纯prompt/声纹/重复录音机制。**

## 1. 冻结证据是否对应真实计划

查读[原稿§2.3、§3.2](submission-20260906/english/main.tex:43)，追溯到[v2/PINS_rc2.json](../v2/PINS_rc2.json)、[split索引](../v2/plan_rc2/split_index.json)、[run_plan.csv](../v2/plan_rc2/run_plan.csv)及[原始结果](../v2/evidence/main/results/results.json)。

- 逐文件重算SHA，`splits.py/train.py/score.py/stats.py/run_plan.py/corpora.py`、计划CSV、split索引、unit_configs和假设registry均与PINS相符。
- 真实manifest来自同workspace的`SER-deploy/v2/manifests/{cremad,subesco}_manifest.csv`；其字节SHA均匹配PINS。按冻结hygiene得到CREMA-D **7,435**、SUBESCO **6,998**条，没有使用synthetic fixture。
- 当前checkout未保存大体积split表，这是[原验证记录已说明的分发方式](../v2/VALIDATION_RECORD_rc2.md:38)。本次调用冻结`mechanism_block`在内存重建CREMA的12张、SUBESCO的14张表，按原`SplitStore`的路径、顺序、meta及canonical JSON序列化；**全部26张SHA与冻结索引完全相等**。这是原计划重建，不是新设计或新随机面板。
- 原结果`plan_sha256`为`2acc658143333a2cc65b21ffaa73c7e78e355cf73276c6de97655b45898a1865`，与实际计划CSV字节相等；`registry_sha256`为`c6a00bb0fdbcd2f9ec827140611605b19c6d5213c16219e3981292d620880ce7`，同样相等。
- 原结果的**52个MECH cell-run全部complete，每个10fold**；计划共520个MECH拟合，其中CREMA240、SUBESCO280。80个`corpus×model×replicate×fold`组内，各条件的train_seed、config_sha及fit/val/test条数完全一致。原结果既有[独立验证](../v2/evidence/main/results/verification.json)记录全研究1,320项、0失败；本次没有把该既有记录冒充重新从原音频训练。

## 2. test固定到哪个层级

[splits.py:273–361](../v2/ser_v2/splits.py:273)把人物分为5组，把prompt按组内录音数量尽量平衡地分为5组。对每个replicate `r∈{0,1}`、speaker组`i`及rotation `rot∈{0,1}`，测试交集为人物组`i`与prompt组`(i+rot) mod 5`，存储fold号为`rot×5+i`。每个speaker×prompt×emotion格按固定哈希选一条take作test。

**同一`(corpus,r,fold)`跨条件、跨CNN/ResNet-SE使用完全相同的test路径和顺序。**代码还断言同一replicate的10个测试集合没有重复录音。它不是对原始全库每条录音作一次OOF，也不是所有replicate共用一个全局不变test：

| 原生数据层 | 每个replicate的10fold测试并集 | 两replicate测试路径交集 | 每replicate被评分人物 |
|---|---:|---:|---:|
| CREMA-D | r0：2,600；r1：2,595 | 947 | 全部91人 |
| SUBESCO-full | r0：560；r1：560 | 52 | 全部20人 |

每个人在每replicate贡献两个rotation的测试录音，即出现在两个存储fold中；两rotation覆盖其两个prompt组。冻结[SPEC_SCORING_CONTRACT §3](../v2/SPEC_SCORING_CONTRACT.md)仍有“each speaker … exactly one fold”的旧文字，这与实际双rotation计划不符。实际scorer将10fold的OOF录音先拼接、再算逐人UAR，执行及主要数值未依赖这句单fold描述。新论文应写“两rotation的OOF合并”，不复制该过时句。

## 3. fit、val与曝光的真实构造

记`S`为当前test人物，`P`为当前test prompt。先从同时排除`S`和`P`的录音池，按人物划分约25%的组作val；余下为背景`fill_pool`，设`N_fit=len(fill_pool)`。这里25%是`GroupShuffleSplit`的人物组比例，非精确25%录音比例。[实现](../v2/ser_v2/splits.py:77)。

每臂完整保留相应曝光集合，再用背景池的、按条件独立哈希选择的类别轮转补样填满同一`N_fit`：

| 条件 | 必须进入fit的曝光集合 | 对test人物/文本的关系 |
|---|---|---|
| none | 空；使用全部背景fit池 | 人物和prompt均未见 |
| spk | `S`人物在非`P`的全部录音 | 只见test人物，不见test prompt |
| prm | 非`S`人物在`P`的全部录音 | 只见test prompt，不见test人物 |
| both | spk与prm的并集 | 人物和prompt均见，但test的确切speaker×prompt×emotion格余下take仍排除 |
| both_sib，仅SUBESCO | both并加测试格余下take | 明确加入同人物、同文本、同情绪的其他take；不包含test原文件 |
| spk_half_h1/h2 | 仅指定半数test人物的spk曝光 | 半曝光描述；两半固定跨rotation，replicate间重新划分 |

实际fit并非在每条轨迹都用单一常数3440。匹配成立于同一配对面板，各面板容量不同：

| 数据层 | fit条数范围；20面板均值 | val条数范围 | test每fold条数 |
|---|---:|---:|---:|
| CREMA-D | **3,439–3,767；3,528.05** | 1,178–1,330 | 108–324 |
| SUBESCO-full | **3,358–3,360；3,359.10** | 1,118–1,120 | 56 |

因此原稿“approximately 3,440 and 3,359”宜在未来修稿改为真实范围或明确面板匹配；3,440接近CREMA下端，不是其均值或所有面板的固定预算。更精确的算法也不是泛指“最小合法池”：实际N取**both-exclusive池划走val后的背景fit容量**，其余臂以曝光替换背景。

代码注释称`_stratified_fill`按比例抽样，但[实际循环](../v2/ser_v2/splits.py:254)是按类别轮转，每轮各取一条直至补足，且完整曝光集合先被保留。**总fit类别直方图并未跨臂匹配**；不能从“class stratification”推导整臂类别数、人物数、时长或音质都相同。

## 4. “common inner rule”究竟控制了什么

支持的强事实来自[splits.py:350–356](../v2/ser_v2/splits.py:350)：每个面板跨条件显式断言test字节、val字节和fit长度相等。val不含test人物或test prompt，且fit/val/test之间无同一文件交叉。原稿只写common inner rule较保守，写“相同val录音”也有直接证据。

但是`prm_exposure`包括所有非test人物，其中也包括背景划分时留给val的人物。这些人的test prompt录音不在val原路径中，因此可以合法进入fit，却改变了fit–val人物关系：

| 条件 | CREMA每面板val人物在fit可见 | SUBESCO每面板val人物在fit可见 |
|---|---:|---:|
| none / spk / 两个half臂 | **0/18–19人** | **0/4人** |
| prm / both / both_sib（若适用） | **全部18–19人** | **全部4人** |

这是完整26表重建的实际结果，不只是静态代码推测。它不是test原文件泄漏，也不使原对照失效；它限制了**N05 prm−none**能否被解读为独立于选模过程的纯prompt效应。**N06 spk−none**保持val人物均未见；**N09 both_sib−both**保持val人物均已见，但仍有下面的训练组成差异。

模型侧，[run_plan.py:321–337](../v2/ser_v2/run_plan.py:321)固定每模型的同一超参和配对训练seed；无曝光臂内超参搜索。训练采用max100轮、patience15、严格更小val loss才更新并恢复best checkpoint。[train.py:96–151](../v2/ser_v2/train.py:96)。**相同选模规则不等于同一选中epoch、同样实际训练轮数或同一训练轨迹。**

此外，训练loss及val CE的类别权重都按该臂`y_fit`计数计算，`weight_c=N_fit/(K×count_c)`；相同val路径不保证数值上是同一加权评价函数。在CREMA的prm−none和spk−none中，**各20/20面板**的fit类别直方图不同；SUBESCO both_sib−both则**2/20**不同。[权重及val CE实现](../v2/ser_v2/train.py:108)。

补样还使用含condition的哈希，故N09不是仅用223–224条siblings逐条替换背景、其余背景全部保持不变：实际每面板还新增**586–613条非sibling背景**、移除**810–837条非sibling背景**，总N不变。原both曝光集合在两臂都保留。可以称“在规定匹配采样下允许sibling的增量”，不能称严格只改变重复take这一变量。

## 5. 三个主要结果及Holm身份

以下是读取原[results.json](../v2/evidence/main/results/results.json:40057)的现存数值，没有增补检验：

| 冻结假设 | 定义 | 人级n | 平均UAR差（pp） | 95%人级百分位bootstrap CI | 两侧Wilcoxon p | Holm p |
|---|---|---:|---:|---|---:|---:|
| N05 | CREMA prm−none | 91 | **11.074481074481074** | **[9.815323565323569, 12.340544871794869]** | 3.0670102801729415e−16 | **1.2268041120691766e−15** |
| N06 | CREMA spk−none | 91 | **2.6831501831501834** | **[1.7856952075702077, 3.5386141636141635]** | 1.258303748433401e−7 | **2.516607496866802e−7** |
| N09 | SUBESCO both_sib−both | 20 | **10.089285714285714** | **[7.991071428571426, 12.410714285714283]** | 8.757494073814436e−5 | **8.757494073814436e−5** |

原稿两位小数**11.07 [9.82,12.34]、2.68 [1.79,3.54]、10.09 [7.99,12.41]**及三项Holm p<.001均正确。Holm族**N-mech的大小是4**：N05、N06、N09、N10；第四项N10为匹配人数/密度对照（4.9032384 pp，n=72，Holm p=2.27385935e−11）。不能把这三项单独重新调整，也不能把SUBESCO的N07/N08当该族内显著确认结果。

N07/N08在冻结registry即为estimation-only；原结果分别5.267857和15.892857 pp，`p_holm=null`、`verdict=estimate (not tested)`。其结果JSON内部虽仍计算原始Wilcoxon值，不改变预定身份。[registry第6–11行](../v2/registry/hypothesis_registry.csv:6)、[scorer家族处理](../v2/ser_v2/score.py:513)。

## 6. 真实统计单位和结论范围

每个`model×condition×r`构成一个cell-run：先合并10fold、两rotation的测试预测，再按人物计算其所有这些OOF录音的类宏召回；缺失类按原定义跳过。随后对同一个人的两个replicate UAR取均值，再对CNN和ResNet-SE取均值，最后对条件作人级配对差。[score.py:148–180、212–258、359–360](../v2/ser_v2/score.py:148)。

推断单位因此是**91位CREMA人物或20位SUBESCO人物**，不是20次fit、两rotation、两个replicate、两个模型或测试录音条数。CI对既定人级差作10,000次全人物重抽，seed20260903、百分位2.5/97.5；Wilcoxon按冻结规则处理近零并列。[stats.py:37–70](../v2/ser_v2/stats.py:37)。这些CI不重训、不重抽全部训练面板，也不覆盖两个随机replicate不足以描述的完整训练变化；共享模型导致的依赖不能靠把fit数叫样本量消除。

**可直接用于新稿的限定句：**“Within each paired exposure panel, the conditions shared the exact test and validation recordings, fit-set size, model configuration and training seed. The comparison nevertheless changed training composition; prompt exposure also changed validation-speaker familiarity, and fit-derived class weights could change the validation-loss criterion. We therefore interpret the contrasts as effects of the specified exposure-and-selection pipelines, rather than isolated memorization mechanisms.”

以上不否定曝光块的增量：相对不同test成员的已有四格先例，它确实排除了配对比较中的test/val录音难度变化和训练录音总数差。但它并未实现后来DUAL的共同fit轨迹设计；新程序也不能反向把旧N05/N06/N09重新命名为已识别的纯机制。

## 7. 本次核查摘要指纹

原结果JSON SHA256：`7e09783aed6003e1fbf47e89f839d061e5d3f5d1655bb9de2c8c02508577a016`。split索引SHA256：`b0f675288dbb898c734754f7e9c37f985a2642727f630c1c95a9d123c092cce6`。其余12项本次字节对照全部通过PINS；26张split重建全部匹配。执行只调用元数据清洗、`mechanism_block`及canonical序列化，不调用任何fit函数。表中面板数量、val曝光比例及补样变化来自这次元数据审计，应标为事后设计核查；不作为新增科学假设或新训练结果。
