# 角色互换实验：独立科学自审

2026-09-07。只读依据：`NEXT_EXPERIMENT.md`、既有 `inner_validation` 训练/评分实现及已封存资源汇总。本次没有取得或读取新的 Claude 评价，不能将以下判断归于 Claude；没有生成新面板、运行训练或追加结果检验。

## 判断与推荐

保留 **CREMA-D、固定配置 3、固定第 15 轮、24 个新 draws × 5 folds × 2 arms = 240 次正式拟合**。这是当前最有辨识力且适合收窄预算的延伸。它把相同报告人物与录音置于两种整组训练人群条件下，直接约束此前“不同人群自身难度”的解释。无需这轮增加模型、学习率搜索、曝光剂量或跨语料。

每个配对面板 A/B 各 24 人，来源性别各 12；各人相同 4 个 fit 文本 × 6 类 = 576 条/臂。A、B 和 outer 人物互斥；两臂报告完全相同的 A query、B query、outer query，query 两文本与 fit 四文本不相交。完整六文本格、同类/性别/文本配额及每外测人的六类支持，仍须由新 metadata planner 对全部 120 面板验收。旧方案容量可作可行性依据，不能替代新盐生成后的验收；禁止科学分数驱动重抽。

## 主终点与推断

令 `Q_A(M)` 是在 A query 上的六类 UAR（百分数），预定

`E = [(Q_A(M_A) − Q_A(M_B)) + (Q_B(M_B) − Q_B(M_A))] / 2`。

等价写法是 `E = [(Q_A(M_A)−Q_B(M_A)) − (Q_A(M_B)−Q_B(M_B))] / 2`。因此 E 是完整 2×2 交互差分的**一半**，也是两组平均的训练包含优势，单位 pp；报告时不能漏掉 1/2。

在完全平衡 query 下，whole-panel 六类 UAR 与逐人六类 UAR 的等权平均代数等价，但建议继续锁 whole-panel 定义，先在每个 draw 内平均五折，再等权平均 24 draws。不能把 120 个 fold、240 次拟合或重复人物当作独立样本。新盐独立产生角色/文本/种子时，draw 层级区间描述的是**条件于已固定 CREMA-D 的随机化与训练程序**，不是新招募人物或其他语料的总体区间。

建议唯一主推断为 24 个 draw 均值的双侧 t 检验与 95% t 区间（df=23）；同时展示全部 24 draw 点。若保留 percentile bootstrap，仅重抽完整 draw，固定一个种子和次数，标为同一主终点的区间敏感性，不新增第二次显著性裁决。预算精简时完全可以不加 bootstrap。不能用看见分布后挑 t/boot 的做法决定结论。

**不能用 A/B 整体互换标签做 E 的置换零分布**：这会使 E 保持不变。随意翻 E 符号也不是由该设计直接保证的精确随机化检验。固定语料上的重抽程序与对新人物的抽样推断应分清。

E 的正区间支持这套整组替换操作存在平均训练包含优势；跨零是证据不够精确，不能宣布等效或无效；负区间表示该规定操作下反向差异，不能直接宣布“训练见过声音有害”。共同 outer UAR、`T_A−T_B`、两组各自贡献、各 fold/draw 分布均为预定描述，不另立优胜检验。`T_A−T_B` 在每个面板并非恒等于零。

## 可识别范围

同一人的静态难度及对全部报告人相同的模型分数平移，在上述差分中抵消。训练两臂的**全部 24 人都被换掉**，学习到的情绪表示、泛化关系及人与模型交互也会变化；E 仍不是加入某一个人的边际因果效应，更不是纯声纹记忆效应。这是整组训练语料替换对相同报告录音的平均作用，不能叫“仅改变单个人的曝光”。

固定配置和轮次使报告标签不参与选模，但也意味着本实验不估计 hyperparameter/checkpoint 搜索的复用偏差。它是此前验证偏差研究的机制边界检查，而非旧主结果 +2.706 pp 的数值复现或因果分解。CREMA-D 及旧配置已被研究过；新随机化不应包装为从未接触过的新语料确认。

## 旧实现的两个关键检查

1. **相同种子能否给相同输出头？** `speaker_coverage/run.py:330–356` 的 `build_ft_model` 先 `set_seed(seed)`，再构建相同 WavLM、加载固定 base，最后创建 `Linear(768,6)`。在同一冻结环境/代码/种子下，输出头的初始化序列应相同；`inner_validation/engine.py:227` 使用该入口，116 行训练循环又重设训练 RNG。新 pair 应在首个梯度前记录并比较**完整 initial state 和 head 的 SHA**、base SHA、可训练参数列表，不能只比较排除了 head 的 frozen-parameter hash。这里只做源码逻辑核验，没有实际构建新模型来宣称已通过配对初始化验收。

2. **是否存在 eval 后忘恢复 Dropout？** 未发现。`inner_validation/engine.py:131–147` 每轮开始明确 `model.train()`，训练完成才 `model.eval()`；下一轮重新启用 train 模式。`speaker_coverage/run.py:438–445` 的预测函数会切到 eval，但最终预测发生在训练结束之后。旧测试 `tests/test_engine.py` 使用含 Dropout 的 TinyFT，检验验证标签变化不影响 last state；这里没有重新运行测试。需要注意 `requires_grad=False` 不等于 eval：旧程序冻结底层参数时仍让底层 Dropout 按 train 模式工作，这是实际训练定义，不能在新代码里无说明地把整个 encoder 常驻 eval。

另一个必须收窄的表述是“同种子 = 同步随机操作”。旧 `_make_batch`（engine.py:171–177）把同一 NumPy RNG 用于 shuffle 与随机裁剪；`engines_deploy.py:218–220` 仅对超过 3 秒的录音取随机起点。两臂时长不同会使 RNG 消耗分叉，所以相同初始种子不保证以后同一排列和裁剪。新实现至少拆开 order 与 crop RNG，固定抽象 slot 顺序；若声明 common random numbers，则按 `(pair_seed, epoch, slot)` 生成裁剪的均匀位置量，再映射各自合法起点。两臂共享初始化和算法随机化方案，不意味着训练轨迹相同。技术验收应查初始化/种子/模式/调用日志，不能看 query UAR 来挑实现。

## 推理、存储与运行精简

- 新训练循环只执行 15 轮 fit，不每轮计算 A/B/outer query 损失或预测，没有 scheduler/early-stop 对 query 的依赖，也不保存 best checkpoints。保留每轮 train loss、36 次 optimizer update **尝试**、AMP skip 数、耗时、异常、模式审计。576/16×15=540 次尝试/拟合，240 次共 129,600 次；不能将 AMP skip 算作实际完成更新。
- 末轮仅推理 A、B、outer 的固定 query，各模型使用完全一致的音频顺序、预处理、截断和 batch 边界。旧 `wavlm_batch` 按每批最长录音补零，Head 不传长度 mask 且平均全部 frame，因此“同 batch”是**控制两模型的 padding 上下文**，不是消除 padding 本身或 model×padding 交互。最小方案可保留明确封存的 batch16/cap10s 流程；若本次主张需要排除同批其他录音的影响，则应在正式冻结前选定逐条、有效长度、cap10s 推理，只在末轮做。不要临时增加未验收的 mask/pooling 改动；推理方案只按技术吞吐/有限性验收预先确定，不能看到科学分数再择优。
- 每拟合保留一个 lossless last delta、三个 role 的 logits/paths/labels、一份 15 行训练 history、receipt 和 DONE。base 权重全程一份 SHA 绑定，不给每个 fit 再存完整 base，不重复存 best/last 别名数组，不存全部 15 轮状态。若不承诺中途 optimizer 恢复，正常最终包不需要 optimizer moments；中断保留失败证据并按预定整 fit 重试，不伪称无失败。
- 保留从磁盘重建后对末轮三个 role 的一次预测复核，以及预定首个 pair 的独立进程恢复验收；不再为三个 best/last 状态各重复预测。模型参数和 receipt 的技术检查可以读取标签结构，但训练/技术选择程序不得读取报告准确率。
- 传输仅搬已 DONE 的最终 artifacts，以单个末轮 delta 为大文件；沿用先校验备份、再精确清理的队列，不造重复巨大 tar。可一批备份与下一批训练重叠，但不能用 DONE 数量代替完整 240 manifest/receipts/ledger gate。完成归档后关 GPU，CPU 完整性复核、评分与图表放本地。

## 资源估算依据与启动门槛

已封存 `inner_validation/evidence/completed/resource_summary.compact.json` 给出旧 cfg3 的 120 次 fit 平均 **34.15784 秒**，其区间包括加载/15轮训练/逐轮双验证/多状态保存与恢复推理，不是单纯 GPU busy 时间。机械按 240 次乘算为约 **2.28 小时 fit 区间和**，只能作同卡旧流程参照，不能承诺新租期；去逐轮验证可能节省，但新的单条报告推理、启动/传输会改变总时长。应由新双臂技术试运行测量后再报租期，计时不接触科学分数。

旧 formal 保存 1,093 个去重 epoch deltas，共 checkpoint 124,007,337,154 bytes，约 **113.46 MB/状态**。由此估计新 240 个 last deltas 约 **27.2 GB**（含序列化开销的粗估，需 pilot stat 确认），显著小于照搬三个状态；仍需额外 raw/base/cache/暂存余量。实际 GPU 型号、现价和可租容量由另项实时资源核验决定，本自审不报价。

启动前门槛：新 120 面板 metadata 容量及泄漏验收；固定 query 推理合同；配对 initial/head SHA；无 query 的 15 轮模式与 RNG 测试；240 完整任务清单和唯一主终点；最后状态保存/恢复与分批归档试运行。到此为止，不应为这轮再加多语料或第二模型。
