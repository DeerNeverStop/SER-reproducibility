# 最终中英双版论文：交付要求与证据清单

核对日期：2026-09-07。本文件只整理要求和现有证据，不是最终稿、执行完成证明或投稿批准。核查时分支为 `codex/final-experiments-paper-20260907`，HEAD 为 `c6386d0bb95a4b1bfe84e8868bd715671918d455`；这是本轮起点，不能冒充各研究的预执行冻结提交。新增实验正在另行规划，本文件没有读取新实验结果或执行训练。

## 要求从哪里来

当前用户要求由主任务传达为“独自跑完剩余实验并最终中英双版论文，要求同以前”。这授权继续完成研究和可审阅交付；历史文件中的“本轮仅审阅、不重跑”限定的是当时审阅任务，不应据此阻止当前已授权工作。此记录没有把“完成论文”解释成已获得作者对最终科学内容的批准、代签声明或实际向会议提交。

可核查的旧交付基线依次是：[当前稿件入口](../../paper/README.md)、[两版审阅包 README](../../paper/submission-20260906/README.md)、[交付与投稿检查](../../paper/submission-20260906/README_交付与投稿检查.md)、[Claude 审阅要求](../../paper/submission-20260906/CLAUDE_REVIEW_REQUEST.md)，以及[英文正文](../../paper/submission-20260906/english/main.tex)和[中文源文](../../paper/submission-20260906/chinese/中文解读.md)。这些文件记录了已提供的作者信息、格式和双语对应关系；没有查到另一个要求中文必须恰好八页、参考文献必须恰好十一条或必须提供 Word 的有效约定。未读取私人聊天数据库，也未将 Claude 的评价或未提供的私人记忆当作用户要求。

正式目标是 **ICASSP 2027 regular paper**。用户最初用于对照的 ICASSP 2026 论文年份不是本稿投稿年份。2026-09-07 重新查看了官方[Author Guidelines](https://2027.ieeeicassp.org/author-guidelines/)、[Submission Instructions](https://2027.ieeeicassp.org/paper-submission-instructions/)和[Paper Kit](https://cmsworkshops.com/ICASSP2027/papers/paper_kit.php)，其 regular/special-session/OJSP 稿件截止日期当前为 **2026-09-16**；本记录不自行推断网页未明确给出的截止时区。

## 最终两版应交付什么

| 项目 | 已有基线 | 本轮必须保持或更新 |
|---|---|---|
| 英文正式稿 | `english/main.tex`、`english/xie.pdf`，官方 `spconf.sty`，四页技术内容加第五页参考文献 | 在同一会议格式下重写最终叙事、结果与局限；正文、图表、摘要与结论必须反映最终已验收研究。旧标题可改，不能只往旧摘要追加新数值。 |
| 中文版 | `chinese/中文解读.md`、`chinese/explainer.pdf`，旧版八页逐节意译与解释 | 提供完整可读、与英文逐节对应的中文版本；可保留通俗解释，但实验、公式、数值、推断身份和结论强弱必须相同。八页是旧成品状态，不是发现的固定页数要求；中文不承担英文以外的新科学主张。 |
| 作者 | **Tian Xie; University of Toronto; tianjack.xie@mail.utoronto.ca** | 沿用用户已提供信息，PDF 元数据和两版署名一致。不新增作者、中文姓名写法、资助编号、IRB 批准或伦理豁免。ORCID、邮寄地址及任何真实新增披露仍须由真实信息补足，不能猜填。 |
| 排版与摘要 | US Letter、双栏、正文 10 pt、表格 9 pt、无印刷页码；英文旧摘要 142 词、五个关键词 | 保留官方模板及尺寸。官方最低字级为 9 pt（含图注），摘要约 100–150 词、最多五个关键词。图在灰度下仍须可辨识。论文标题、作者次序及摘要须与最终投稿表单一致。 |
| 页数与第五页 | 旧版第五页仅参考文献 | 官网 Author Guidelines/Submission Instructions 及 Paper Kit 的 REFERENCES 段均写第五页仅参考文献；Paper Kit 开头另允许资助致谢和伦理声明，存在内部表述差异。继续把技术内容、AI 披露和必要声明放在前四页，第五页仅参考文献，可满足较严格共同要求；不能借第五页增加结果或技术附录。 |
| 引用 | 旧英文 11 条，包括数据、模型、SER 划分先例和访问受限 artifact | 十一条不是上限或固定要求。必须补直接碰撞工作，核对原文任务、指标、划分和归因；正文每项关键比较都有真实来源，参考文献与正文引用双向对应，不保留虚构或错版本条目。 |
| AI 披露 | 旧稿已说明 Claude/Codex 在代码、研究设计、执行、审计及全稿写作中的作用 | 按实际新增贡献更新系统名称、涉及章节及使用程度，在致谢中披露，中文同步。不能声称只有语法润色，不能凭文风给“AI 使用率”；AI 不列为作者。官方要求见 [Author Guidelines 的 AI 条款](https://2027.ieeeicassp.org/author-guidelines/)。 |
| 可编辑源与构建 | 英文 LaTeX、中文 Markdown/ReportLab 脚本、模板、结构检查脚本和 `source_bundle.zip` | 新版 PDF 与新版源码、全部引用和图源一起交付；写明编译器/依赖/字体来源及实际命令。不要让新 PDF 搭配旧源码包。保留旧交付目录/manifest 的历史身份，使用明确的新版本入口和文件清单。 |
| PDF 验收 | 旧两版均有结构 QA 和全页目检记录，旧 PDF 为 58,633 / 203,354 字节 | 对**新字节**重新编译、检查、逐页渲染目检，确认无截断、重叠、乱码、破表、缺字；全部字体嵌入、无加密、英文不超页数、每份低于旧检查器的 5,000,000 字节阈值。不把旧 QA 当新稿通过证明。 |
| 最终用户可审阅包 | 已有 README、源包、PDF QA、UPLOAD_MANIFEST | 新增明确的双版入口、研究/主张/证据映射、构建与重放命令、文件 SHA/字节清单；结果图表和正文最后一次修订后统一封存。未决真实作者资料可单列投稿待办，不能伪填，也不应阻止先完成稿件与证据。 |

官方允许 Letter 或 A4，旧方案选 Letter；不需要改动已验证的版式选择。Paper Kit 要求作者署名、所有作者 ORCID、嵌入字体、无安全限制以及最多 5 MB 文件；作者名与系统次序一致。其推荐按第一作者姓氏命名 PDF，因此 `xie.pdf` 可沿用。上述投稿系统事项见 [Paper Kit](https://cmsworkshops.com/ICASSP2027/papers/paper_kit.php)。

旧 [PDF QA](../../paper/submission-20260906/pdf_qa.json)明确 `visual_review_performed_by_this_script=false`：程序没有代替人工目检。旧编译路线为 Tectonic 0.17.0、中文 ReportLab、检查使用 pypdf/pdfplumber，Poppler 渲染全页。是否继续同一工具是工程选择；最终要记录实际可用且已成功的命令，不能复制未执行的成功记录。不得为挤页数篡改官方 `spconf.sty` 或缩到最小字级以下。

## 各 Study 的实际证据与正文取舍

下表是根据当前入口进行的证据盘点，未重新训练或重验大检查点。编号沿用原项目标识，最终稿可统一重编号，但须建立映射，尤其不能把“Study II”预算配置与后来的“E2”声纹选样混为同一研究。正文/附录位置是基于当前主问题的编辑建议，不是按显著性筛掉结果；凡声称执行过的研究，其完整预定结果仍须有可追溯入口。

| 研究及来源 | 已完成、可引用的事实 | 正文/附录建议及最后需要判断的边界 |
|---|---|---|
| 原 v2 多语料程序：[旧结果](../../v2/evidence/main/results/results.json)、[核验](../../v2/evidence/main/results/verification.json)、[当前英文方法](../../paper/submission-20260906/english/main.tex) | 当前稿记录 4,943 个完成单位；三语料 outer/inner RR/RG/GR/GG、计数匹配曝光、密度与 FT 对照。原推断为逐人配对/整人 bootstrap，与后续 draw 推断不同。 | 用作历史基础或精简一张背景表；保留配额、训练成员变化和弱 frozen 对照限制。不能把代数分解说成声纹因果机制，不能让全部旧实验挤掉新主问题的可理解方法。旧 `paper/v2` 中失效 N14 结果不得沿用。 |
| N14R2：[完成 release](https://github.com/DeerNeverStop/SER/releases/tag/SER26-N14R2-complete-20260906)、[当前稿](../../paper/submission-20260906/english/main.tex) | 24 个完整 draw、1,920 fits；ResNet-SE，random/grouped 内层两臂 fit 成员会变化。主差 **1.419 pp，95% t CI [0.901, 1.937]**；D09R2 描述量 −0.076，不能解释成无损。 | 若主线是验证报告，可保留为历史直接前导。必须说明新 DUAL 解决的共同 fit 边界，而非独立语料复制；保留原 N14R 中断排除、planned 字段 24→28 单字节勘误及原失败记录。不能只替换旧宏而沿用错误样本量/检验族。 |
| Study II 固定录音预算：[完成 README](../data_design/README.md)、[英文完整稿](../../paper/study2/STUDY2_DRAFT.md)、[完整表](../../paper/study2/results/tables.md) | 已完成 **1,440** 单位（CNN/Ridge 各 720），288/576 条 × 12/48 人 × seen/new 句；六句对合并后逐人 UAR，91 人条件 bootstrap。未见句时人数配置差 CNN +1.08/+2.16 pp，Ridge +0.73/+0.06 pp。 | 若最终主线聚焦内层验证，通常放补充背景/附录而非另辟正文主结论。人数增加同时降低每人句覆盖，估计联合配置取舍；录音数不是总成本或同音频小时；四项句条件交互区间均跨零。不得重跑已完成工作冒充剩余实验。 |
| E2 声纹选样：[完整报告](../speaker_coverage/reports/RUN_REPORT.md)、[结果](../speaker_coverage/reports/data/e2/results.json)、[新外测几何复核](../review_response_20260907/e2_review_notes.md) | **720 formal + 19 pilot**；核心 CNN/Ridge 三 draw，FT 仅 draw 0 两 seed。CNN 主 C−U **−0.109 pp [−0.715, 0.509]**，未检出收益；C 降低池内最坏 NN1，但未使外测平均 NN1/NN3 更接近。 | 可完整保留于附录或关联研究说明。不能称覆盖策略等效/无效或 gap 已消除；Ridge best/last 同模型不是两次复现。预算固定 576 条而非时长/全流程采集量，ASV/neutral 参考、标签与配额也是策略前提。外测几何补算属于事后描述。 |
| DUAL 共同 fit 双验证：[完整报告](../inner_validation/reports/RUN_REPORT.md)、[结果](../inner_validation/reports/results/results.json)、[完整 gate](../inner_validation/evidence/completed/formal_gate.json) | **480 formal + 4 pilot**，24 draw × 5 fold × 4 cfg；共同轨迹、共同 SI test，两种 own-val CE checkpoint + own-val UAR config 规则。预定 θ **2.706 pp [1.802, 3.610]**。固定 cfg3/末轮同模型验证差 **2.838542 pp**。 | 当前最直接正文候选。θ是不同验证人物曝光下所选模型的验证−SI 外测差异，不能全归为“选择导致”的偏差。whole-fold 六类 UAR、先五折后 24 draw；主 t CI 条件于固定语料/随机化。两种规则外测点差 +0.847 不能改写成 θ 导致的损失。 |
| DIAG 双向交叉报告：[SPEC](../validation_reporting/SPEC.md)、[完整报告](../validation_reporting/REPORT.md)、[加权复核](../review_response_20260907/diag_review.md) | 复用 DUAL 四配置固定末轮存档预测，A/B 各 12 人，双向选择/报告，无新增训练或推理。选后报告曝光差约 2.937/2.992 pp；只是同一资料上的探索。 | 优先方法说明/附录；不称第三次独立重复或新的盲测。报告半区只未参与本方向选择，另一方向会使用；保留原整角色 batch/padding 上下文。真实分解含选择与报告差协方差；独立噪声模拟不是适配真实数据的正式 null，不能据其断言无信息。 |
| PR16 后 DUAL/E2 再分析：[审稿回应](../review_response_20260907/RESPONSE.md)、[DUAL 复核](../review_response_20260907/dual_science_review.md) | 91/91 人已有两种角色；人物/context FE、仅 checkpoint 规则对比及新外测几何均已计算。有探索性区间，但未纳入原主检验家族。 | 可作为设计下一确认的动机或敏感性附录，明确“看过结果与评价后”。不得把 FE 当逐人纯身份因果、把被选轮次平均当完整学习曲线，也不能把事后显著区间追认为确认性结果。 |
| E0/E1 与 TTS 技术探针：[诊断](../speaker_coverage/reports/DIAGNOSTICS.md)、[音频预算](../speaker_coverage/reports/AUDIO_BUDGET_AUDIT.md) | E1 是旧结果上的描述关联；TTS 100 条技术尝试，四锚检索 51/96=53.125%，当前报告记录人评 0/3，正式 E3 未通过准入。 | 只写探索/可行性或不进入主论文。RMS/裁剪比例不是人类质量，ASV 四锚命中不是身份真实性，技术生成成功不是有效合成训练。若当前新授权调整路线，另立前瞻性规范；不得以代理替代尚未完成的人评并声称合成方案已验证。 |
| 本轮 final program | 此文件核查时只有新输入盘点实现等准备工作；尚无本轮新 formal 完成 gate、主评分或论文。 | 主任务需先确定剩余研究清单与冻结分析，再执行并按完整 gate/独立复算验收。新候选与旧“延期240次”草案的继承/变化必须明确，不能从历史建议直接推断已执行或沿用旧结果。最终正文主线依据完整结果决定，不预写胜负或按显著性继续加量。 |

跨研究不得合并人物 bootstrap 与 draw t 区间、不能把 fits/折/重复种子当独立样本，也不能以共享 CREMA-D 的多次再分析声称多次独立人群复制。跨语料若新增，只能检验所设计的语料/流程稳健性；不同语言、人数、类别和语料同时变化，不能给出语言因果结论。

## 最终改稿前必须补齐的引用与证据

相关工作必须包含 Ibrahim 等的直接先例：DOI `10.1038/s41598-026-58836-w`，官方[原文](https://www.nature.com/articles/s41598-026-58836-w?error=cookies_not_supported)和[表 6](https://www.nature.com/articles/s41598-026-58836-w/tables/6?error=cookies_not_supported)已核查，审计定位见 [source note](../review_response_20260907/diag_ibrahim_source.md)。其 audio-only 分支已有 random 与 speaker-independent 比较，不能仅称其多模态而排除；accuracy 数值不能拿来与本稿 UAR 直接相减。SERAB 的人物隔离验证、Atmaja/Sasou 的划分研究、Antoniou 的评估建议，以及 Cawley/Talbot 的选择偏差先例需据原文准确归因，入口见[研究定位](../validation_reporting/RESEARCH_DIRECTION.md)。有界检索不能支持“全球首次”保证；旧稿文献尚未覆盖这些新修订。

每条进入正文的主张应能追溯到 **study ID → 原计划/源码固定提交与哈希 → 完成 gate → 主评分表及精确未舍入值 → 独立复算 → 图表生成来源**。报告可舍入，但双语、图、表须由同一结果源出发；抽样数、分析单位、预定/估计/事后身份及支持范围另列，不用论文版本号代替科学冻结身份。新研究的失败/重试/偏离、pilot 排除和有限恢复覆盖亦须据实保留。

DUAL 的科学冻结是 `de39f2ae5bad4540da9d13ab4cc6196bec786400` 和计划 SHA `570cae10577e13bc17a5838d872251beeb52d8b216886d19a8dcb6418a7f5bbd`；来源证明中的 `7bbe14b` 是检查时交付 HEAD，后来论文提交也不是新的“事前冻结”。E2、DUAL 的私有 Git 历史支持版本追溯，不自动成为公开预注册；N14R2 原规范要求的执行前公开收据未建立公众可访问证据，原限制须继续披露。旧标签名含 prereg 不能改变此事实，也不能从当前私有倒推创建时私有。

轻量材料适合随交付保存：冻结源码与规范、计划/数据清单哈希、完整汇总 CSV/JSON、图源、独立审计、环境/计时定义、匿名化运行收据和文件 manifest。原始音频、特征、预训练基座、逐单位预测/检查点、完整账本及失败记录须在受控本地归档保留；它们体积大或有数据使用约束，不应默认塞入 Git。DUAL 已有约 125.12 GB formal/pilot 结果保留在本地，Git 完成附件是审计入口而非这些大文件的替身。仓库相对路径、固定提交或 release 链接用于论文；本机绝对路径、设备/实例身份、账户和凭据不写进公开文稿。

若交付一个无需大权重的预测重放包，需另写其明确验收范围和命令；现有完整 gate 依赖模型工件，不能宣传为下载小 CSV 就完成原始音频重训或全部权重验证。资源段区分 fit 墙时之和、租机起止跨度、已入账费用与尚未结算估计；停止证明来自服务商控制面，不能仅凭训练退出。费用与大量工程检查宜放补充材料，正文只保留理解研究计算预算所需内容。

## 本轮完成判据

剩余实验以新的明确清单为准全部封存，或把真正无法完成的项与原因单列，不能在最终摘要中当已完成；独立数值核查后再更新正文全部相关段落、摘要、图表、引用及 AI 披露。随后生成内容一致的中英文 PDF 和匹配源码，重做结构 QA 与全页视觉检查，检查本地/固定 Git 引用可达性，生成新交付清单，并给用户一个能直接阅读的入口及简短科学结论与局限。

实际投稿前的 ORCID、真实作者资料/批准、需提交的伦理与资助信息以及审稿人对材料的访问安排，应单独列为真实待办；不能假称已处理，也不能把历史清单中的待办扩大成写稿、审计、编译等已授权工作的额外审批门槛。本次资料核查没有发现必须先取得“导师同意”的用户要求或提供的政策，不应引用未给出的私有 memory 新设此条件。
