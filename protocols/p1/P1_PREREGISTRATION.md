# P1 协议溢价受控测量：预注册

版本：1.0（冻结）  
冻结时间：2026-08-10 18:02:55 UTC-04:00  
状态：**仅预注册，尚未进入 P1；禁止在 GPU 排队许可前训练**

## 1. 时间边界与研究问题

本文件在读取任何本课题 P1 新训练结果之前写定。已知的 `0.555 → 0.736`（约 18.1 个百分点）来自姊妹项目早先完成的实验，是形成假设的先验事实，不是 P1 结果，也不会并入 P1 推断样本。

P1 的唯一核心问题是：在模型、输入、优化器、增强和评估样本固定时，将同一语料的评估协议从说话人无关改为 utterance-level 随机划分，会把 UAR、accuracy 和 macro-F1 乐观抬高多少？本阶段不复现论文，不比较他人代码，不调出最好看的模型。

主要估计量是 UAR 的协议溢价；accuracy 和 macro-F1 为次要估计量。所有差均按“较宽松协议减较严格协议”定义：

- `ΔRG = UAR(random utterance-level) − UAR(GroupKFold)`；
- `ΔRL = UAR(random utterance-level) − UAR(LOSO)`；
- `ΔGL = UAR(GroupKFold) − UAR(LOSO)`。

正值表示较宽松协议给出更高数字。绝不把负值改写成零，也不因结果不显著而改变定义。

## 2. 数据、许可与固定分析总体

| 语料 | 本地只读位置 | 分析总体 | 标签/组 | 官方来源与许可 |
|---|---|---:|---|---|
| RAVDESS speech | `E:\科研\SER\data` | 1,440 段、24 位演员、8 情绪 | 文件名中的 emotion / actor | Zenodo DOI `10.5281/zenodo.1188976`，官方元数据显示 `CC BY-NC-SA 4.0` |
| CREMA-D | `E:\科研\SER\AudioWAV` | 7,442 段、91 位演员、6 情绪 | 文件名中的 emotion / actor | `CheyneyComputerScience/CREMA-D`；数据库 `ODbL 1.0`，单项内容 `DbCL 1.0` |

> 上表两处路径于 2026-08-12 由 `C:\Users\jock8\Desktop\科研\SER\...` 改写为迁移后的
> 实际位置，分析总体未变；换算依据与核验见文末偏离日志同日条目。

固定纳入本地上述目录在运行清单中列出的全部可读 WAV；不加入 RAVDESS song，不删除难例，不做按长度或置信度的事后筛选。运行前只允许做文件可读性与标签解析门检；任何损坏/无法解析项必须逐文件记录并保持跨协议一致。数据与标签都只读；不修改 `SER/`。

正式运行必须输出一份 byte-level manifest（相对路径、文件字节数、SHA-256、标签、speaker ID），并将其哈希写入每个结果文件。若届时文件集合或标签数与上表不同，先在本文件末尾追加偏离说明，再决定是否运行；不得静默替换总体。

## 3. 统一预处理

所有模型、协议和 seed 使用同一份原始 log-mel 缓存：

- `librosa.load(..., sr=22050, mono=True)`；`librosa.effects.trim(top_db=30)`；少于 0.1 s 时右侧补零；
- 64 个 mel bin，`n_fft=1024`，`hop_length=512`，`fmax=11025`（Nyquist）；功率谱经 `librosa.power_to_db(ref=np.max)`；
- 固定 128 帧；不足在右侧以该谱最小值填充，超出保留前 128 帧；
- 每段谱单独以全谱一个均值和标准差做 z-score（`eps=1e-6`）。不使用跨 utterance、跨 speaker 或全量数据统计量，因此没有把测试折统计量带入训练折；
- 缓存名必须至少编码 `corpus/version/sr22050/mel64/nfft1024/hop512/fmax11025/frames128/trim30/sample-zscore/code-hash`。与该完整签名不符的缓存一律拒绝读取；
- 仅训练子集按 epoch 在线做 SpecAugment：2 个 time mask（最大 16 帧）、2 个 frequency mask（最大 8 bin）及高斯噪声 `σ=0.1`。validation/test 不增强。

除非发现实现错误，上述参数不因某一模型或协议表现而改变。

## 4. 固定模型与训练设置

四种模型都由姊妹项目同一构建入口 `advanced_models.build_neural_model` 创建，避免把不同来源实现混入协议差异。固定使用 `tuned_standard_experiment.candidate_configs(model)[2]` 的结构/训练字典；这里把值完整展开，防止上游以后漂移：

| 名称 | 固定结构 | 固定优化设置 |
|---|---|---|
| CNN | width 48；3 blocks；kernel 5；dilation base 1；dropout 0.1 | AdamW, lr 1e-3, weight decay 1e-4 |
| ResNet-SE | width 48；4 blocks；kernel 5；max dilation 4；SE reduction 4；dropout 0.1 | AdamW, lr 1e-3, weight decay 1e-4 |
| Transformer | d_model 96；4 heads；2 layers；FF multiplier 2；normalized sinusoidal position；attentive pooling；dropout 0.1 | AdamW, lr 3e-4, weight decay 1e-4 |
| FNO | width 48；16 Fourier modes；4 layers；dropout 0.1 | AdamW, lr 1e-3, weight decay 1e-4 |

共同设置：class-balanced cross-entropy（权重只由当前 fit 子集计算）、batch size 64、最多 100 epochs、CosineAnnealingLR、validation loss 早停 patience 15、恢复 validation loss 最低的 checkpoint；不查看 outer test 选择 epoch、阈值或超参数。随机 seeds 固定为 `{0,1,2}`。模型初始化、batch 顺序、增强 RNG 与 inner-validation 分配均由当前 seed 和稳定的 corpus/protocol/outer-fold 派生键确定；同一键跨模型复用相同数据分配。

冻结的姊妹项目版本为 git commit `d11768ed31a91f611472b89e52f3f6bb97030c40`；关键文件 SHA-256：

- `advanced_models.py`: `c1e18011dbb6aeb1717a4ee5f88afb5e8352ec61044837aeb5b7d70f6fe26ad5`
- `advanced_experiment_utils.py`: `ea175e3f7366feeee505d2441796268f58e6daae7aa6b625e6d0cb78a4b0e04e`
- `fno_data.py`: `9762eeb91268258443dd8a1908336d2cffeed9097f6e2c66461f4419ce0f3c88`
- `tuned_standard_experiment.py`: `865f60a6d27f28514b57ac174c9a3432382718f300ebe4d75d1f280d81402bff`

正式 P1 代码留在本课题目录并 import/read-only 复用这些实现；不改 `SER/`。若为支持三协议必须写 wrapper，须用小型合成数据先证明索引、speaker 互斥和 test-once 性质，但合成测试不算 P1 数字。

## 5. 三种 outer 协议（写死）

所有 outer split 在任何模型训练前一次性生成、落盘并哈希；同一语料/协议的 split 对四模型及三个 seed 完全相同。每种协议最终都有覆盖全部语料一次的 out-of-fold（OOF）测试预测。

### 5.1 Random utterance-level

`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`，只按 emotion 分层，不把 speaker 传给 splitter。每段 utterance 恰在一个 outer test fold。此臂故意允许 speaker overlap，是待测的泄漏对照，不作为推荐协议。

### 5.2 Speaker-independent GroupKFold

`GroupKFold(n_splits=5)`，`groups=speaker_id`；输入样本按 manifest 的规范相对路径排序。任一 speaker 的全部 utterance 只能出现在一个 outer fold，train/test speaker 交集必须为空，否则该运行硬失败。每段 utterance 恰在一个 outer test fold。

### 5.3 Leave-one-speaker-out (LOSO)

按规范化 speaker ID 升序逐一留出：RAVDESS 24 折、CREMA-D 91 折。每折 test 是一位 speaker 的全部 utterance，其余 speaker 为 outer train；speaker 交集必须为空，否则硬失败。

### 5.4 Inner validation

outer test 在训练完成前不可参与任何选择。每个 outer train 内预留约 25% 作 validation：

- random 臂：`train_test_split(test_size=0.25, stratify=emotion)`，允许 speaker overlap，以保持该对照臂定义；
- GroupKFold 与 LOSO 臂：`GroupShuffleSplit(n_splits=1, test_size=0.25)` 按 speaker 分组；fit/validation speaker 交集必须为空。

inner split 的 random state 由 `SHA256("P1-inner|corpus|protocol|outer_fold|seed")` 的前 32 bit 无符号整数确定。若一次 GroupShuffleSplit 使 fit 或 validation 缺任一情绪，则顺序尝试 `base+1, base+2, ...`，最多 100 次，采用第一个两侧类别齐全的分配；全部失败则该单元记为预先定义的 `inner_split_infeasible`，不改协议、不借用 outer test。

## 6. 输出、指标与配对统计

每个训练单元必须原子写出配置、split 哈希、seed、最佳 epoch、完整 validation history、outer-test 逐 utterance logits/prediction、耗时及失败码。主表从逐 utterance 文件重新计算，不信任训练日志里的汇总数字。

对每个 `corpus × model × protocol × seed`，把所有 outer fold 的 test 预测拼成一次完整 OOF 预测，计算：

- 主指标：UAR（各 emotion recall 的非加权均值）；
- 次指标：accuracy、macro-F1；
- 报告三个 seed 的均值 ± 样本 SD，并逐 seed 保留原值；不把 outer fold 当独立重复。

协议溢价的配对单位是 speaker，而不是不相干的 fold：先对每个 `speaker × protocol × model × corpus × seed` 用该 speaker 的同一批 utterance 算 UAR，再跨 3 seeds 求 speaker 内均值；随后对同一 speaker 做 `ΔRG/ΔRL/ΔGL`。因此比较两边拥有完全相同的 speaker 和 utterance，只是训练协议不同。主效应为 speaker 差值的均值；95% CI 用固定 seed `202608101744` 的 10,000 次 speaker-cluster percentile bootstrap，整位 speaker 重抽。另报告双侧 Wilcoxon signed-rank；全为零时 p=1。若 Wilcoxon 因定义域失败，报告失败原因并用精确 sign test 作敏感性分析，不替换主估计。

确认性 family 是 2 corpora × 4 models × 3 UAR contrasts = 24 个双侧检验，Holm 校正 `α=0.05`。accuracy 和 macro-F1 的相同配对差及 95% cluster-bootstrap CI 为次要/描述性，不追加显著性星号。若需任何事后探索，必须标 `exploratory`，与确认性表分开；不因多重比较结果改变主方向或删模型。

同时输出随机臂每个 outer fold 的 `n_train_speakers/n_test_speakers/n_overlap/overlap_fraction`，以及严格两臂为零交集的断言。报告必须同时给 point、SD、CI、原始 n speakers 与有效失败数；差距落入噪声范围写 `not conclusive`。

## 7. 先验点预测（看到 P1 数字前）

下表是每个模型 OOF UAR 的主观点预测与合理区间，不是接受区间，也不用于删异常结果。

| corpus | model | random 点 [区间] | GroupKFold 点 [区间] | LOSO 点 [区间] |
|---|---|---|---|---|
| RAVDESS | CNN | .70 [.58,.80] | .52 [.40,.63] | .50 [.38,.62] |
| RAVDESS | ResNet-SE | .75 [.63,.84] | .57 [.45,.68] | .55 [.43,.66] |
| RAVDESS | Transformer | .72 [.60,.82] | .54 [.42,.65] | .52 [.40,.64] |
| RAVDESS | FNO | .76 [.64,.85] | .58 [.46,.69] | .56 [.44,.67] |
| CREMA-D | CNN | .62 [.50,.73] | .52 [.40,.63] | .50 [.38,.62] |
| CREMA-D | ResNet-SE | .70 [.58,.79] | .60 [.48,.70] | .58 [.46,.69] |
| CREMA-D | Transformer | .66 [.54,.76] | .56 [.44,.67] | .54 [.42,.65] |
| CREMA-D | FNO | .69 [.57,.79] | .59 [.47,.69] | .57 [.45,.68] |

压缩成协议溢价先验：RAVDESS `ΔRG=.18 [.12,.24]`、`ΔRL=.20 [.15,.28]`、`ΔGL=.02 [-.03,.05]`；CREMA-D `ΔRG=.10 [.05,.20]`、`ΔRL=.12 [.06,.22]`、`ΔGL=.02 [-.02,.06]`。落在区间外不是失败，只触发实现审计；实现无误则如实报告并更新理论。

## 8. 算力上限、停止规则与缺失

计划训练单元数固定为：

- RAVDESS：`4 models × 3 seeds × (5 random + 5 GroupKFold + 24 LOSO) = 408`；
- CREMA-D：`4 × 3 × (5 + 5 + 91) = 1,212`；
- 合计 1,620 个 fit，最坏 162,000 epochs。

正式阶段上限：单卡 GPU 累计 200 小时、墙钟 14 天、P1 产物 20 GiB。开跑前必须再次检查 `nvidia-smi` 与队列；当前 GPU 明确交给课题 02，所以本轮不得启动。若触及任一上限，停止新增训练，保留所有完成/失败单元，按预注册矩阵报告完成率与缺失位置，结论写 `not conclusive`；不得为了凑齐而减 seed、减折、只留表现较好的模型或放宽 speaker 互斥。

单元因确定性 OOM/异常失败时，可在相同配置下重试一次；仍失败记缺失。任何 batch-size 降低会改变优化轨迹，必须先追加偏离记录并对该模型所有协议/语料统一重启，不能只救某一臂。不得用测试表现作停止依据。

## 9. 冻结与偏离政策

冻结后正文只追加、不删改。允许在运行前补充不改变以上估计对象的工程细节（文件格式、CLI、校验脚本），但必须在下方以时间、原因、是否在看到 P1 数字前、影响范围和新文件哈希记账。凡改变数据总体、模型/超参、split、seed、指标、contrast、CI、检验 family 或预算的改动均为实质偏离，必须保留原方案结果并单列；不能把偏离后的分析冒充预注册主分析。

### 偏离日志

- 2026-08-10 18:02:55 UTC-04:00：v1.0 首次冻结；尚无 P1 新训练结果，当前 GPU 由课题 02 使用，P1 未启动。
- 2026-08-11 14:15:02 UTC-04:00：在任何 P1 真实训练或指标出现前，按本任务书 §9 追加非实质工程锁定；课题任务书 2026-08-11 §6 已确认旧 GPU 排队前提失效、允许启动 P1。逐字节键、manifest/cache/split 序列化、严格全 validation weighted-CE loss、test-once/重试状态机、原子 attempt 预算账、full-cell 缺失传播、linear percentile、Wilcoxon 参数与缺失项在固定 24 项 Holm 中以 `p=1` 占位的规则详见 `P1_ENGINEERING_LOCK.md`（8,768 bytes；SHA-256 `d0d3b6f9130d35b6ec1f9f90d16ab7fdc7062714f88f5e23594ddbf0c00aed31`）。该追加不改变数据总体、四模型、三协议、seeds、超参、指标、contrast、先验或 200 GPU-hour/14-day/20-GiB 上限；追加时 `formal_start.json`、`units/`、`attempt_ledger/` 均不存在，1,620 单元全部 pending，尚未生成或读取 P1 真实数字。
- 2026-08-11 14:34:21 UTC-04:00：RAVDESS 正式训练已开始，但在读取任何 outer-test 性能、OOF 指标、协议差值或显著性结果前，修复汇总器预算审计的派生文件自计数：`results/protocol_premium/summary/` 不再计入 runner 正式产物 `output_bytes`。修订前/后汇总器 SHA-256 分别为 `13af00739dc18d7c84b6e48735541ec4df5b1d82fdef9d3e7c738825739d3404` / `a3a43530f2e65bec6c97757a583c1549302d476b1a0a40cd6d56ce82056bc181`；验证器未改。合成生命周期测试的 formal bytes 为 `19→19→19`。这是只影响汇总—复核幂等性的非实质修订，不改变任何 fit、预测、估计对象、统计方法或三项预算上限；详见追加后的 `P1_ENGINEERING_LOCK.md`（10,070 bytes；SHA-256 `caf367af06d4c160bfd61663c71a60105833ef7bf84f5c6e3621063c3fa718c5`）。
- 2026-08-12 17:01:14 UTC-04:00（**P1 已在运行、RAVDESS 阶段数字已存在**，故如实标注本条不属于"看到数字前"）：项目作者把整个工作区从
  `C:\Users\jock8\Desktop\科研\` 迁移到 `E:\科研\`，§2 表中两处只读语料路径按新位置改写
  （`SER\data`、`SER\AudioWAV` 两项，其余单元格未动）。这是**存储位置变更，不是分析总体变更**：
  新旧路径指向同一批字节——核验为 RAVDESS `data` 下
  1,440 个 WAV、CREMA-D `AudioWAV` 下 7,442 个 WAV，与本表预注册数字一致；
  抽样 `data/Actor_05/03-01-01-01-01-01-05.wav` 的 SHA-256 为
  `d6ae1eaf5f0638d0fbbb6ea9ba040e089e741c13caca902d91059541c24c093a`。
  **（同日 17:15 更正本条措辞）** 本条初稿把旧位置描述为"整目录复制出的第二份副本"，
  经 inode 比对与写入探针核验，实为 `C:\Users\jock8\Desktop\科研` 是**指向 `E:\科研`
  的符号链接**：数据自始至终只有一份，新旧路径解析到同一批字节，
  "两份是否一致"这个问题根本不成立。结论不变，机制更正如上。
  manifest 以**相对路径**记录逐字节键，故已落盘的 manifest 哈希、已完成单元与 attempt 账目均不受影响，
  无需重跑。不改变数据总体、四模型、三协议、seeds、超参、指标、contrast、先验或
  200 GPU-hour / 14-day / 20-GiB 上限。后续新写路径统一用 `E:\科研\`；
  旧路径仍可正常解析，**不要删除该符号链接**。修订人：Claude。
