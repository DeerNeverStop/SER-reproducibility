# 旧 N14R2：选中配置与固定 c4 的 V/T 分解核查

2026-09-07。独立CPU描述性复核；没有新增训练、推理、假设检验或置信区间，没有读取新final-program的formal结果，没有修改任何冻结源码、原结果或论文稿。

**结论：**原主结果1.419049 pp复现。选中配置的GR−GG验证差为2.554590 pp，外测差为1.135540 pp，两者相减即主结果。固定c4时，同一口径的验证−外测差之差为1.596196 pp；所以不能把1.419 pp全部归于八配置搜索。c4自身仍按验证损失选检查点，不是完全没有选择的反事实。

## 1. 来源与核验范围

使用本地此前已保存的release staging：`n14r2-cloud-20260905/v2_1/n14r2/release_staging/SER26-N14R2-complete-20260906`，未重新下载。对应[完成release](https://github.com/DeerNeverStop/SER/releases/tag/SER26-N14R2-complete-20260906)；本次不将本地哈希检查冒充执行前公开注册或匿名访问成功的证明。

- `RELEASE_SHA256SUMS`列出的**16个资产**逐个核对SHA，包括486,594,560字节的原`n14r2-closed-run.tar`；另记录manifest自己的SHA。
- 核对analysis-lock规范化payload、completion的锁引用、24个完整draw（0–23）与1,920个锁定单位；无备用替换、无blocked/void条目。原正确版本verification为pass、differences为空，其reconstructed对象与corrected result相同。本次检查这一既有验收证据，未重新运行原10万bootstrap和全符号翻转检验。
- `score.py`、`verify.py`、`PINS.json`当前历史工作树字节与source-bindings及Git提交`4fe0bb52f15b56a6626363be31a378b0f9661293`相同；原spec与run_plan字节匹配analysis lock。计划2240身份中的24个主要draw产生1920个正式单位。
- 重新读取**1,920份history及1,920份unit**，按原规则找最早最低验证损失epoch，并核对unit的best_epoch、val_loss_best和val_uar_best；在每个draw/fold/cell的8配置内以验证UAR最大、并列取最小0-based index重新选出全部240个获胜配置。
- 从原tar读取selected与固定c4所需的**393份不同outer predictions.csv**，按六类完整宏召回独立重算T，逐条核y_pred等于六维logit的最早argmax。selected与c4有87个重复身份，因此不是480份不同预测。直接读取的4,233个tar成员均另核analysis-lock中的SHA；没有将tar解压到旧路径或修改它。
- 120个draw/fold内，GR/GG的selected与c4均具有完全相同的outer路径及标签集合。重新选出的配置/单位/V和重新计算的T，逐fold及draw均与原结果一致，核对容差为绝对`1e-10` pp。

验证V来自锁定history中最低CE检查点的完整验证折UAR；本次没有原始验证logit可重新从录音计算V，不能说进行了新的验证预测推理。外部音频/特征缓存和大模型权重亦未重新验收。这是有来源绑定的旧结果分解与有限预测重算，不是原始数据重训。

| 关键文件 | SHA-256 |
|---|---|
| RELEASE_SHA256SUMS | `a1439db6185d25b67645b6487adefcac0fea8bb571d9223bbce6c8534dedece6` |
| n14r2_results.metadata-corrected.json | `92146e7dcf01cc8fd923f78d08017e638d501732ff23110008caa64b5f6ef6c6` |
| n14r2_verification.metadata-corrected.json | `e82f088d88770e6a201ad93b581a5460aa9fba27658a9810ba63a9ed115cf843` |
| analysis_lock.json | `e471cb3c2a3b26e799d8f85cbab86f9818b656db356b86eb3524a0e0ec0c3b82` |
| completion.json | `4d19ba026381983d0c593728f104fbdf391ad8f99bf2bac1638e3f44559978ff` |
| n14r2-closed-run.tar | `162e209301252397c6d7bf79ae8c3a781fca93f8ef7dbe2babace94d4f44f9e9` |

原始结果与勘误版的对象仅在`n_planned_draws`由24改28这一项不同，本次再次核对；这没有改变任何科学值。原始失败不能被该勘误副本的通过状态覆盖。

## 2. 按原分析单位计算的可入稿数值

每fold先对唯一train_rep计算whole-fold六类UAR；每draw等权平均五fold，再对24个draw等权平均。V和T为UAR百分数，V−T及组间差为百分点。不能把120折或1920拟合当独立统计样本，也没有按录音数或说话人数重新加权。

固定`c4`是原0-based index 4：ResNet-SE，学习率0.001、weight decay 0.0001、dropout 0.1。它仍使用相应GR/GG验证损失最低的checkpoint。selected是在每个cell内从8个这样的候选模型中按完整验证折UAR选最大者。

| 配置规则 | 内层规则 | V（%） | T（%） | V−T（pp） |
|---|---|---:|---:|---:|
| 选中配置 | GR：随机内验证 | 62.365972 | 59.607115 | 2.758857 |
| 选中配置 | GG：人物分组内验证 | 59.811383 | 58.471575 | 1.339808 |
| 固定c4 | GR：随机内验证 | 62.076897 | 59.751471 | 2.325426 |
| 固定c4 | GG：人物分组内验证 | 59.269571 | 58.540341 | 0.729230 |

| 组间差，统一GR−GG | 选中配置（pp） | 固定c4（pp） |
|---|---:|---:|
| ΔV | 2.554590 | 2.807326 |
| ΔT | 1.135540 | 1.211130 |
| Δ(V−T) | **1.419049** | **1.596196** |

两个附加代数描述：

- 原预定D09R2 = selected ΔT − fixed c4 ΔT = **−0.075589851639 pp**，与原报告−0.076一致。
- selected Δ(V−T) − fixed c4 Δ(V−T) = **−0.177146734373 pp**。这是本次事后描述，未给新CI/p，也不能据此宣称搜索会减小偏差或两者等效。

GR/GG在23/120个面板选中相同配置，97个不同；GR中48次、GG中39次选中c4。这些是选择频数，不是新的推断单位。

可直接用于英文结果段的描述（保留本报告作为来源）：

> Selected-model validation/test UAR averaged 62.37/59.61% under random inner validation and 59.81/58.47% under grouped inner validation. Thus the 2.555-point validation difference exceeded the 1.136-point test difference by 1.419 points. With the prespecified fixed configuration, the corresponding validation-minus-test contrast was 1.596 points; this comparator still selected its checkpoint by validation loss.

中文对应：选中模型在随机内层验证下的验证／测试UAR均值为62.37%／59.61%，分组内层验证下为59.81%／58.47%。因此，2.555点验证差减去1.136点外测差得到1.419点的相对乐观差距。固定预定配置时，对应差距仍为1.596点；该对照仍通过验证损失选检查点。

解释必须同时保留：两臂实际fit成员、验证人群与所选模型变化；以上外测优势不能独自归因于人物覆盖或某个选择机制。原主结果是验证估计的相对乐观差距，不是外测损失。无须为补全这张表重训，也不应据它给出未经测试的“分组后全量重拟合一定更好”建议。

## 3. 旧v2 N14失效：可直接追溯的源码与原纠正记录

这里区分**旧v2 N14**、未完成的**N14R**与最终完成的**N14R2**。旧N14的失效原因有直接证据，不必引用Claude或本轮红队作为唯一依据：

1. 冻结[原计分合同，第106行](https://github.com/DeerNeverStop/SER/blob/4fe0bb52f15b56a6626363be31a378b0f9661293/v2/SPEC_SCORING_CONTRACT.md#L106)写出`val_sel(f(s)) − U(s)`的N14定义；第89行把对照单位指定为speaker。一个验证fold的V被该fold多个测试speaker重复使用。
2. [旧score.py，第333行起](https://github.com/DeerNeverStop/SER/blob/4fe0bb52f15b56a6626363be31a378b0f9661293/v2/ser_v2/score.py#L333)的`n14()`逐speaker构造`out[s]`，第339–340行按`fold_of_speaker[s]`取共同的`val_sel`；[第267行起](https://github.com/DeerNeverStop/SER/blob/4fe0bb52f15b56a6626363be31a378b0f9661293/v2/ser_v2/score.py#L267)的`summarize()`将speaker数组传给paired_test和speaker_bootstrap_ci。旧结果的N14有91项speaker观测及n=91。
3. [原纠正状态记录，第9行起](https://github.com/DeerNeverStop/SER/blob/186b40d4a9db720d246f92dfef5ba73020903b9c/paper/v2_1/STATUS.md#L9)已明确将历史N14列为不能作确认性推断，并给出“fold-level validation information was expanded into pseudoreplicated speaker observations”的原因。本次确认该文件实际字节与该Git提交相同，不根据当前评价倒写历史。

最保守的入稿句：

> The earlier v2 N14 confirmatory inference was set aside because fold-level validation scores were reused across speaker-level observations; N14R2 uses complete randomized outer-split draws as its inferential units.

中文：旧v2的N14确认性推断被排除，因为折级验证分数被重复用于多个说话人层面的观测；N14R2改用完整随机外层划分draw作为推断单位。

这个问题针对N14中共享fold验证量的伪重复推断，不说明旧模型拟合失败，也不将所有v2人物层指标一并判无效。新N14R2的结果不能反过来使旧n=91检验有效。本报告不另裁定23/22假设计数或S1全局状态。

上述直接证据SHA：旧`v2/ser_v2/score.py`为`196ee61203f6732acb309d9cf18adfc834fb1d3636f9e969a29d8871a532cd5a`；`v2/SPEC_SCORING_CONTRACT.md`为`14c4696e3093347307f645021f72ed68b4022bf20f147f7d6e9ebe9b612827f6`；历史`paper/v2_1/STATUS.md`为`529503a349ba67e437dd22eb45b60f636b0732a1feb8114352ce0fa33111db3f`。

## 4. 本地重放与审计产物

Git审阅入口为[原字节重算脚本](audit_20260907/n14r2/recompute_n14r2_decomposition.py)和[完整逐fold/draw描述JSON](audit_20260907/n14r2/decomposition.json)。两份副本与下列已验SHA一致；复制前检查未发现凭据，内容为代码、来源哈希与标量分解，不含原音频、波形数组、logit数组或模型权重。仅分发这两份文件不会替代重放所需的原release归档。

D盘原小证据副本、脚本和JSON仍保存在`D:/SER-final-program-20260907/old_evidence_checks/n14r2`，原字节未改；原tar仍留在旧归档位置，没有额外复制近0.5GB。任何同名已有文件只核对，不覆盖。

- `recompute_n14r2_decomposition.py` SHA：`6871a996c92cf7e9ad89d989953a97f4e10ce0506bcaca12731387adc1c91919`。
- `decomposition.json` SHA：`f1b7f9dc45f3cc3020b613959c8c595cd899ede18e44a49469622311ec0915c6`；998,951字节，含16资产SHA、4,233个直接读取成员SHA、120fold/24draw完整描述及口径限制。
- 重放必须选择尚不存在的新输出名，示例：

```powershell
E:/科研/SER/ser_gpu/Scripts/python.exe -B paper/audit_20260907/n14r2/recompute_n14r2_decomposition.py --source-repo C:/Users/jock8/Documents/ChatGPT/论文/n14r2-cloud-20260905 --evidence D:/SER-final-program-20260907/old_evidence_checks/n14r2 --out D:/SER-final-program-20260907/old_evidence_checks/n14r2/decomposition.replay.json
```

本次实际运行使用D盘的同字节脚本，正常退出并通过所有来源、选择及逐层数值一致性断言；上面给出从仓库根目录调用副本的重放命令。没有用Claude给出的均值、CI或p值作为计算输入；未新增科学推断。
