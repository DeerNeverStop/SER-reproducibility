# Study II：固定录音预算配置实验

> **公开投影（2026-09-08）**：本文件删除了账户、运维身份或私有审阅入口；科学方法与结果保留。原始文稿存于受控私有归档，原 SHA-256 为 `bcac92ea4f0ca1bcf7b6f3defb87952ec76b9e6bfa516b706700060b0c7e8b2d`。公开版不是原冻结文件的字节副本；原科学代码、结果与计划未在此改写。

## 当前结果（2026-09-05）

**完整 1,440 单元已完成并独立核验通过。** Ridge 与 CNN 各720，所有单元第一次尝试成功。16个冻结源码/规范文件保持原哈希，没有重新生成计划或改变模型设置。科学结果在完整合并闭合后才首次计算、读取；独立复算检查6,199个数值，最大误差2.13×10⁻¹⁴个百分点。

陌生说话人＋未见句条件下，将12人配置换为48人配置，在288/576条预算上的UAR差：CNN为+1.08/+2.16个百分点，Ridge–WavLM为+0.73/+0.06个百分点。两模型的结论强度不同；四项句子条件交互区间均跨零，不能声称未见句特别受益、普遍最优配置或真实客服收益。完整绝对值、区间和三次draw必须一并报告。

- [英文完整方法与结果稿](../../paper/study2/STUDY2_DRAFT.md)
- [中文结果解释](../../paper/study2/RESULTS_BRIEF_ZH.md)
- [完整结果表](../../paper/study2/results/tables.md)、[全部估计CSV](../../paper/study2/results/results.csv)、[三次draw补表](../../paper/study2/results/draw_estimates.csv)
- 完整执行与核验回执（未随此公开快照分发；原件另行保留）、云端备份与停机回执（未随此公开快照分发；原件另行保留）
- [当天进度](progress/2026-09-05.md)、[冻结规范](SPEC_CORE_ZH.md)、[执行契约](CONTRACT.md)、[既往结果暴露说明](PRIOR_EXPOSURE.md)

## 云端已停止，保留备份

CNN 已完成并退出，无活跃运行器；最终快照在独立目录恢复并逐一核验 2,168 个文件，计算实例已确认停止。具体实例身份、账户充值与费用记录保留于受控原件。

最终CNN源：工作区 `runpod_setup/cloud_core/snapshots/20260905T154235Z/core`；原包 `cloud_core/archives/cnn_20260905T154235Z.tar.gz`，SHA `f51f872da8f702380a9256cd40bb0176c80fe987c5bb788456d3922dd49ac385`。历史143单元活动快照继续保留，其锁不能删除或用于合并。操作记录在工作区 `Runpod_5090当前实例.json` 与 `runpod_setup/运行说明.md`。

不要在本机重复启动CNN，不要自动重启Pod，不再等待旧N14R释放GPU。旧 `E:/科研/essay/SER-v2` 由原任务管理，本研究只读其进度，不控制或修改其运行。

## 固定身份与设计

程序 `SER26-STUDY2-CORE-1`；预执行提交 `6bf62b9be3ffebd669bec1dc312e724566d42d48` 已先推送再训练。计划SHA `f258cabe582d97b3a386a666b566730d00241a5ded0e14ad61d590cb3c861503`，原始JSON与gzip身份见 [PLAN_LOCK.json](evidence/PLAN_LOCK.json)。不要为方便修改16个冻结文件或重新生成此计划。

2预算（288/576）×2人数（12/48）×2句子条件（seen/new）×5说话人折×6不交叠句对×3draw×2模型，共1,440。测试覆盖91人和12句；每句曝光人数、类别和性别元数据配额固定。增加人数同时减少每人的句子覆盖，这是联合配置取舍；预算是录音条数，不是金钱或时长。代表录音MD优先、其次XX，不回退HI/LO。

这是受既往v2结果启发后的前瞻性估计扩展，不继承确认性预注册身份。噪声、合成、跨语料、建议器和配置搜索均不在本轮范围。

## 完整离线分析接口

本地Ridge原目录 `v3/data_design/work/core` 与最终CNN备份保留原状。合并器把两份各720单元的原账本、环境、回执和预测复制到全新 `work/merged_core`，核验完成后写completion/analysis_lock；不解析预测数组。不完整、损坏、有活动锁、超尝试次数或混合环境的来源均拒绝。

```shell
python -m v3.data_design.core_merge --repo . --plan v3/data_design/work/plan/core_plan.json --ridge-run v3/data_design/work/core --cnn-run /path/to/final-cloud-snapshot/core --out v3/data_design/work/merged_core
python -m v3.data_design.core_score --repo . --plan v3/data_design/work/plan/core_plan.json --run v3/data_design/work/merged_core --out v3/data_design/work/analysis
python -m v3.data_design.result_verify --repo . --plan v3/data_design/work/plan/core_plan.json --run v3/data_design/work/merged_core --analysis v3/data_design/work/analysis --out v3/data_design/work/analysis/verification.json
python -m v3.data_design.core_present --repo . --plan v3/data_design/work/plan/core_plan.json --run v3/data_design/work/merged_core --analysis v3/data_design/work/analysis --out v3/data_design/work/presentation
```

这些命令已用于本次独立本地目录；正常无需重跑。新本地CPU任务必须CUDA不可见、至多两线程、Windows BelowNormal。工作区 `runpod_setup/run_cpu_module.py` 是设置这些限制的操作包装器：`python run_cpu_module.py MODULE [arguments]`，从仓库根目录启动并将其加入PYTHONPATH。`core_merge`/`core_present` 自身设置优先级；评分与独立核验使用包装器。

独立结果复核器不调用评分器统计函数，从原始预测混淆计数和独立bootstrap权重复算。展示工具先核验完整1440闭合、独立批准和每个输入SHA，再解析已批准的成绩字节；不重新计算统计、不筛选结果。生成16个绝对量、8个人数差、4个差中之差，全部draw补表及PNG/PDF图。输出固定在 `work/presentation`，同身份完整结果可复用，损坏或不同身份输出拒绝覆盖。

评分为六句对合并→固定六类逐人UAR→三draw平均→91人等权；10,000次共享配对bootstrap、种子20260905。95%区间是当前训练与有限数据下的近似条件性、逐项区间；不作p值或全族显著性声明，三个draw不能当作三个独立样本。

## 后续与截稿

新增核心计算与独立核验已在9月5日完成，早于9月10日晚目标。接下来进行图表/文字审阅，与旧研究主线及会议篇幅整合，9月13–15日留给人审。没有因为任何结果好坏追加实验或修改范围。不会自动投稿。

历史先导、CPU、云端阶段和工具测试见progress与evidence。旧 `deadline_status.py` 只适用于原本机排队方案，不能用于判断已完成的云端CNN或再次建议租GPU。
