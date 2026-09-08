# P0.5 随机补样与估计协议 v1.0

- 冻结时间：`2026-08-10 17:52:18 UTC-04:00`
- 冻结状态：在生成随机顺序、访问任何被抽中仓库或查看 P0.5 结局之前冻结
- 执行边界：CPU、网络与静态代码阅读；不安装/运行候选仓库代码，不使用 GPU
- 继承协议：除本文件明确补充的抽样与比较规则外，全部资格、去重、风险字典、
  证据门槛和代码优先规则原样继承 `P0_PROTOCOL.md`，不因 P0 结果或抽样结果改动

## 1. 冻结抽样框

抽样框是冻结时 `candidate_log.csv` 中 `screening_status=pending` 的全部记录：

- `candidate_log.csv` SHA-256：
  `9f46ef0e0dc3a40910fa7bc030cbe393f1f60f0401ec30254a2a2d5514306938`
- 全表 383 个唯一 canonical GitHub URL；状态为 include 32、exclude 20、pending 331
- 本次框大小：`N_frame=331`
- P0 目的性样本的 32 个 include 不在 pending 框中，因此不会重复抽入

框内记录保持原始 `candidate_id`、发现来源、检索 ID 和备注。备注可能已包含资格线索，
但随机顺序只由 canonical URL 和下述种子决定，不读取或利用备注、仓库质量、论文关系、
数据集、stars、更新时间或任何风险结局。

P0.6 后续替代检索发现的新候选不追溯加入本已冻结的 331 项框；它们进入独立的
`p0_6_alternative_candidates.csv`，作为框覆盖敏感性材料。这避免在看见抽样结果后改变分母。

## 2. 随机种子与跨语言确定性排序

- 预注册种子：`202608101744`（来自最新版任务书写入时间 2026-08-10 17:44）
- URL 规范化：沿用 P0 canonical 规则；去 query/fragment/尾斜线/`.git`，host 固定
  `github.com`，owner/repo 用 Unicode `casefold()` 后的字符串参与哈希
- 对每个 pending URL 计算：

```text
SHA256("P0.5|202608101744|" + canonical_url_casefolded_utf8)
```

- 按 64 位十六进制摘要升序排列；理论碰撞时按 casefold 后 URL、原 URL 依次升序破平
- `draw_rank=1` 起顺序检查，不使用依赖 Python 版本的 `shuffle()`，也不允许人工换位

生成脚本必须同时核对输入 SHA、331 行、URL 唯一性和状态；任一不符即停止。完整 331 项顺序
在首次 Stage-B 检查前一次性写入 `p0_5_random_order.csv`，未抽到的尾部也保留，防止选择性补抽。

## 3. 顺序筛选、补抽与停止规则

按 `draw_rank` 逐项做 P0 Stage-B 资格筛选：

1. 仓库在检查时可公开访问并能冻结到具体 commit；
2. 至少在 RAVDESS、CREMA-D、IEMOCAP、EmoDB 之一做 SER 评估；
3. 有与目标结果相关的数据准备、split、训练或评估代码，不因协议差、结果低或路径不完整排除；
4. fork/镜像/同一实现按 P0 规则只保留一个分析单位；
5. 入选后完成与 P0 相同的 38 字段静态审计和 file:line + frozen-commit 证据链。

不合格项使用 P0 的首个成立 E-code。任务书预期 E1–E5；若实际出现 E6–E8，同样如实记录并
继续补抽。不得用风险结局作为资格或补抽理由。每排除一项立即检查下一 rank，直到得到
**30 个完整审计的合格唯一仓库**；若 331 项耗尽仍不足 30，则以实际 n 结束并报告失败原因，
不从框外方便补样。

抽样日志至少包含：`draw_rank`、hash key、candidate ID、canonical URL、访问时间、资格状态、
E-code、资格证据、冻结 commit、是否计入 30 个分析单位、累计 eligible n。所有已检查项保留；
停止位之后的 URL 明确标 `not_reached`，不伪装为筛选过。

## 4. 静态审计与交叉复核

- 主表：`p0_5_random_survey.csv`，与 `survey_table.csv` 的 38 列语义完全相同；
  repository ID 使用 `P05R001`–`P05R030`，按抽中合格顺序编号
- 证据：`p0_5_random_notes.md`；每一项非 unknown 判定要有活动路径 `file:Lx-Ly`
  和 frozen commit permalink，unknown 要写已查路径和缺失环节
- clone：只读保存到本课题 `sources_p0_5/`；不执行 notebook、训练、评估、安装或下载脚本
- 初审与复核分离到不同审计遍次/代理；**30/30 全量交叉复核**，不是只抽 yes/unknown
- 分歧先写 `p0_5_review.csv`，再由主审依据定义—调用—数据流裁决到
  `p0_5_adjudications.csv`；旧值永久保留
- 对外措辞必须写“同一自动化代理系统的不同审计遍次”，不得冒充两名独立人类编码者

作者盲编码 8–10 个随机库不由 Codex 代做，作为 `REPORT.md` 顶部待作者事项保留；在作者结果
到来前，不计算或暗示人类 inter-rater reliability。

## 5. 七项主估计量

分析单位为 30 个合格唯一仓库。逐数据集先按 P0 原字典得到 yes/no/unknown，再在仓库内按
`yes > unknown > no` 合并；`evidence_strength=low` 在主分析中降为 unknown。

七项结局与 P0 完全相同：

1. `split`：说话人非互斥/随机划分风险；
2. `normalization`：全量统计量归一化泄漏；
3. `test_explicit`：最终测试集显式驱动选择；
4. `test_any_exposure`：显式选择或每 epoch 暴露最终测试指标；
5. `augmentation`：增强进入最终测试路径；
6. `single_split_single_seed`：至少一个目标数据集只做一次 split/seed；
7. `variance_absent`：主结果没有离散度。

**论文主数字预先指定为随机样本。** 每项报告：

- `yes/n`：unknown 保留在总分母、按未确认风险处理的保守主率；
- `(yes+unknown)/n`：unknown 全按风险的上界；
- 两个端点各自的 Wilson score 95% 区间；
- 可判定子集 `yes/(yes+no)` 及 Wilson 95% 区间；
- unknown 的单独比例。

因为顺序是对冻结 pending 框的种子化随机排列，前 30 个合格项在“资格状态与风险结局分离、
可访问性不由风险决定”的条件下可视为该框内合格仓库的简单随机样本。Wilson 区间因此不再是
目的性名册上的装饰，但仍只针对 GitHub/arXiv 捕获框；完整合格总体大小未知，故不做有限总体
修正，也不外推到无公开代码工作或未被检索捕获的工作。

## 6. 目的性 P0 与随机 P0.5 的并排比较

目的性样本固定为已封版的 32 库 `survey_table.csv`：
SHA-256 `27a552cedfe8361049c6c0c346fa89a410d481a9f8b0bcb454432f8f1028fa99`。

对七项结局并排报告两个样本的 `yes/no/unknown`、保守主率、上界和 known-only 率。比较规则：

- 主效应量：`random_yes/n_random - purposive_yes/n_purposive`，正值表示随机框中确定风险更多；
- 95% CI：两个独立 Wilson score 区间的 Newcombe hybrid score 差区间；
- 检验：每项做双侧 Fisher exact，2×2 为 `yes` 对 `no+unknown`；七个 p 值组成一个 family，
  用 Holm 方法校正；同时单列 unknown 率，避免把信息不足误读为无风险；
- 辅助敏感性：比较两边的 `(yes+unknown)/n` 上界差，不另做第二套显著性宣称；
- 不把两样本差异解释成“论文关联导致协议更好/更差”，只作选择机制相关的描述。

若无差异，也不能证明目的性样本无偏；报告效应 CI 和统计功效有限。若有差异，论文主估计仍
使用随机样本，目的性 32 库降为深度案例层。

## 7. 合并与权重

**不把 32 个目的性库与 30 个随机库简单池化成总体主估计。** 目的性库没有已知抽样概率，
无法构造有效 inverse-probability 权重。默认输出：

- 主估计：P0.5 随机样本 n=30；
- 深度案例与证据库：P0 目的性样本 n=32；
- 并排敏感性比较：两者差及 CI；
- 如展示 62 库合并计数，只标“未加权描述性合集”，不附总体 Wilson CI，不称总体率。

## 8. 偏离、失败与封版

任何输入框、种子、排序、停止规则、资格定义、结局编码、比较方法或 n 的变化，都只能在本节
末尾追加：时间、原规则、偏离、原因、影响项和敏感性处理；不改历史。网络/API/仓库失效属于
观测结果，按 E4 和抽样日志记录，不构成换样授权。

当前无偏离。

### 2026-08-10 23:45:53 UTC-04:00 — 推断口径澄清（不改变计算）

在最终裁决表和 P0.5 主汇总生成前的独立统计实现复核指出：目的性 P0 样本没有已知抽样概率，
故 §6 预注册的目的性—随机样本 Newcombe/Fisher/Holm 比较虽然照算并完整保留，却不具有
设计型总体覆盖率或Ⅰ类错误保证。最终报告把这些量限定为“暂把目的性样本视作 iid binomial
时的工作模型诊断”，不使用“两个总体显著不同”的措辞，也不把 `p<0.05` 升级为总体推断。
随机样本自身的七项主率与 Wilson 区间、比较公式、七项 Holm family、样本顺序、纳排、停止位、
字段编码和权重规则均未改变；此澄清不使用任何 P0.5 最终汇总数字作决定依据。
