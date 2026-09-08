# DEPLOY-2：正式分析独立验收

本目录是**看到正式结果后新增、明确版本化的验收补充层**，不属于执行前冻结的科学源码。原始预测、模型、条件、预算、评分器和 `EXECUTION_LOCK.json` 均不改动。冻结 `verify.py` / `verify_deploy2.py` 的原始失败记录必须保留，不可以用补充验收覆盖为 PASS。

## 入口与输入

在 SER-deploy 仓库根目录运行，使用 `E:/科研/SER/ser_gpu/Scripts/python.exe`。各验收器在 CPU 上限制为两线程，不连接云端、不训练模型。

```powershell
& 'E:/科研/SER/ser_gpu/Scripts/python.exe' -m v3.deploy.verify --module all --plan-dir v3/deploy/work/plan --run-dir v3/deploy/work --results-dir v3/deploy/results --v2-data 'E:/科研/essay/SER-v2/v2' --out v3/deploy/evidence/analysis_verification_all_frozen.json --tolerance 1e-9
& 'E:/科研/SER/ser_gpu/Scripts/python.exe' v3/deploy/analysis/deploy2_b1_supplemental_audit.py --summary-sidecar v3/deploy/results/B1/summary_complete.csv
& 'E:/科研/SER/ser_gpu/Scripts/python.exe' v3/deploy/analysis/deploy2_export_audit.py
```

原始布局为 `work/A/units`、`work/B2_cpu/units`、`work/B2_gpu/units`、`work/B2_FIXED/units`。`B2_gpu` 是指向完整本地备份 evidence 根目录的 junction，含 120 个实际 checkpoint；不是轻量备份。B2 验收拒绝同一 UID 同时出现在 `B2` 和 `B2_cpu` / `B2_gpu`。B1 使用 v2 数据根目录中 `runs/main/units` 的 360 个既有单元，完整保留 72 个 OOF 重复和额外 seed 组。

结果根目录为 `v3/deploy/results`；A/B1/B2/B2-FIX 各有 endpoints.json。B1 完整列和逐重复附表由独立导出步骤保存，原始 summary.csv 保留原样。

## 原始验收与补充验收的区别

冻结全部验收的收据是 `evidence/analysis_verification_all_frozen.json`。A、B2、B2-FIX 通过；B1 在完成 84 个主端点比较后因 summary.csv 缺少 accepted_support_6 列失败，整体必须记为 FAIL。该导出器使用首个六类语料的列名，漏掉较多类别语料需要的支持列；原始逐重复 JSON 和折内结果具有完整类别支持。

另一个冻结验证器缺陷是 B1 逐人风险跨重复使用 nanmean，跳过无定义风险；评分器及严格重复解释使用 mean，任一重复无接受量则该人的风险为 NA。真实数据中 SUBESCO-980 / CNN / b=.30 / F09 的 9 个重复只有 8 个风险有定义。补充验收对**所有**说话人、三个预算、全局与折内两种模式执行同一严格规则，保留这个定义性差异的清单。它不根据该人的数值选择预期值。

`deploy2_b1_supplemental_audit.py` 不导入主评分器，从原始预测重新计算；除已披露的严格重复风险外，冻结独立统计期望保持原样。它覆盖全局与折内各 84 端点、63 汇总行、2835 逐人行、441 类别构成行，以及 72 个逐重复 JSON/CSV 和完整支持、主表等全部数值附表。原始 frozen B1 失败必须已存在才能运行。

`deploy2_export_audit.py` 补查冻结主验收没有逐字段比较的导出：A 绝对类别召回从原始预测按 draw→repeat→speaker 重建；A harm/主表/N5 表核对已经独立验收的端点；B2 units.csv 的 390 行从各自 val/test 原始 logits 单独重算，避免逐单元阈值错误在汇总中抵消。

每份补充收据记录实际脚本路径、脚本 SHA、原始失败 SHA、被比较附表和来源 SHA。最终综合状态见 `evidence/analysis_verification_summary.json`；原始冻结状态与补充接受状态分别列出。

## 解释边界

- A 先平均五次参照、再平均三划分、最后说话人等权；主比较固定 N3，N5 不可行条件保留。oracle_all 使用查询特征，不是可部署方案或保证上界。
- B1 的 10%/20%/30% 预算是在可见最终批次中固定排序，不能当作已校准在线阈值；接受风险与错误捕获为录音池化量，人等权覆盖和风险另列。接受类不全的固定类 UAR 为 NA，跨重复 UAR 明确以有完整支持的重复为条件。
- B2 GR/GG 更改完整训练/验证协议；B2-FIX 才固定拟合模型与测试预测，仅替换熟悉/陌生校准来源。不能把 B2 的差异全部归因于校准泄漏。
- 风险任一重复无定义时，B2/FIX 严格保留该人 NA；报告可用人数和剔除人数。次要风险目标因模型而异，不能解释成统一绝对风险保证。
- 10000 次 bootstrap 使用共同的完整说话人索引，保留人内重复配对，重复划分与参照抽样不是新增独立样本。区间以当前模型和阈值为条件，不包含重新拟合/重新校准或真实通话迁移的不确定性。
- 先前结果已被查看，研究属于描述性/估计性；所有预定条件必须报告，不能用本次区间代替未经查看的确认性检验。
