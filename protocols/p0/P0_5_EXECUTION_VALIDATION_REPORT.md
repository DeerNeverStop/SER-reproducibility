# P0.5 冻结随机样本：按证据强度分档的执行验证

## 结论

对冻结概率样本的 30 个仓库逐库核验后：

- **25 个执行确认**：只在本课题 `.venv_p0_8` 中复刻决定性的划分操作，把带明确
  speaker ID 的合成清单送入该操作，并直接求训练/最终评估两侧的 speaker 集合交集；
- **1 个静态执行推理**：冻结环境没有 fastai，且禁止安装；完整读完相关代码路径后，
  可由 `valid_pct=0.2` 的 item-level 调用确定没有 speaker 分组约束；
- **4 个 unknown**：冻结树缺训练划分、外部 section 生成器或外部 Kaggle Train/Test
  清单，继续猜测不能缩小不确定性。

最终 speaker-shortcut 划分风险为 **20 yes / 6 no / 4 unknown**。原先的
**21/30** 需要更正为 **20/30**；唯一改判是 P05R015。旧汇总把方法名
`random_two_speakers_outer_test` 中的 `random` 误聚合成 utterance-level random，
但代码实际先解析 speaker，再随机抽取两个**完整 speaker group**作为测试集，
因此训练/测试 speaker 交集为 0。

这里的 `yes` 表示划分规则允许或直接产生 speaker 重叠，`no` 表示最终评估对明确
speaker group 互斥，`unknown` 表示冻结证据不足。合成探针确认的是程序机制，不是
原作者真实数据上重叠人数的估计。

## 安全与复现边界

- 未 import 或执行任何候选仓库模块；未运行整仓、notebook、安装脚本、下载器、
  pickle/二进制或模型；未联网、未训练、未使用 GPU。
- `E:\科研\SER` 全程只读；没有向 `ser_gpu` 安装依赖；没有重跑 P1/P2。
- 合成清单固定为 4,444 行：RAVDESS 1,440、CREMA-D 2,184、IEMOCAP 400、
  EmoDB 420；speaker ID 带语料前缀，避免跨语料数字 ID 假碰撞。
- 265 个候选文本/代码文件均作完整 byte-read、SHA-256 清单；`candidate_code_executed`
  全为 `false`。
- 人类盲编码与一致率口径已撤销；`实验包.zip` 明确作废、未发送、也未删除。

## 逐库证据

“交集”是合成探针各最终评估 pair/fold 的最大 speaker 交集；`—` 表示没有声称直接
执行交集。完整 speaker 集合、参数、永久链接和限制说明见
`results/execution_validation/repository_evidence.csv` 与 `per_repository/*.json`。

| ID | 层级 | 最终判定 | 最大交集 | 决定性证据 / 边界 |
|---|---|---:|---:|---|
| P05R001 | 执行确认 | yes | 24 | `notebooks/emotion_classifier_final.ipynb:L510-L522`：普通 80/20 row split |
| P05R002 | unknown | unknown | — | `app.py:L733-L752` 只加载模型/编码器/scaler；冻结树无训练划分 |
| P05R003 | 执行确认 | yes | 34 | `train.py:L148-L159`：仅按 emotion stratify，不传 speaker |
| P05R004 | unknown | unknown | — | `scripts/emotionthinker_infer.py:L1-L39` 仅下载模型并推理；无目标语料训练划分 |
| P05R005 | 执行确认 | yes | 89 | CREMA-D notebook cell 15：ordinary shuffled row split |
| P05R006 | 执行确认 | yes | 10 | `ML_with_MFCC_Normalized.ipynb:L1563-L1572`：仅按 target stratify |
| P05R007 | 执行确认 | yes | 24 | RAVDESS notebook cell 39：ordinary shuffled row split |
| P05R008 | 执行确认 | no | 0 | 训练 actor 1–19，最终测试 actor 20–24；内层随机验证不改变外层互斥 |
| P05R009 | 执行确认 | yes | 24 | `emotionravdess2.ipynb:L537-L560`：label-stratified row split |
| P05R010 | 执行确认 | yes | 24 | `train.py:L17-L41,L119-L139`：先无分组 shuffle，再按行切片 |
| P05R011 | 执行确认 | yes | 24 | `utils.py:L73-L88`：RAVDESS 文件名 90/10 row split；另两目标路径缺失 |
| P05R012 | unknown | unknown | — | `run.py:L38-L45,L140-L184` 依赖仓外 section list/数组，无法恢复 speaker 映射 |
| P05R013 | 执行确认 | yes | 24 | `train_model.py:L18-L31`：特征表已丢 actor，普通 80/20 row split |
| P05R014 | 执行确认 | yes | 113 | `scripts/train_emotion_classifier.py:L52-L60`：合并特征表按行切两次 |
| P05R015 | 执行确认 | **no** | **0** | `modules/formatter.py:L18-L64`：解析 speaker，抽两个完整 speaker，再以 speaker 筛行 |
| P05R016 | 执行确认 | no | 0 | `train_val_set.py:L51-L60,L105-L114`：actor 23–24 整体测试 |
| P05R017 | 执行确认 | yes | 91 | notebook 先为每 utterance 生成多行，再对增强后行做 80/20 split |
| P05R018 | 执行确认 | no | 0 | `iemocap_emotion_recognition_csv.ipynb:L793-L841`：10-fold LOSO |
| P05R019 | 执行确认 | yes | 10 | `Model.py:L90-L101`：emotion-stratified utterance indices |
| P05R020 | 执行确认 | no | 0 | `04_dataset_split.ipynb:L232-L286`：先切 unique speakers，再按 speaker 选行 |
| P05R021 | 执行确认 | yes | 97 | `SER_v2_kaggle.ipynb:L191`：合并表按 emotion 做 85/7.5/7.5 row split |
| P05R022 | 执行确认 | yes | 24 | `crnn/train.ipynb:L4686-L4706`：4-fold StratifiedKFold，无 groups |
| P05R023 | 执行确认 | yes | 91 | `Mel_framing_window.py:L64-L125` 先分窗；`Inception_framing.py:L156-L176` 再切 image rows |
| P05R024 | 执行确认 | yes | 24 | `train.py:L53-L79,L103-L109`：actor 未传入 stratified split |
| P05R025 | 静态执行推理 | yes | — | `prep_data.py:L38-L45`→`melspec_extrac.py:L42-L57`→`emo_rec.ipynb:L122-L133`；`valid_pct` 只切 items，无 speaker groups |
| P05R026 | 执行确认 | yes | 24 | `main.ipynb:L77-L110`：75/25 row split，random_state=9 |
| P05R027 | 执行确认 | no | 0 | loader `core.py:L531-L547,L775-L785,L820-L824`：Ses03–05 / Ses02 / Ses01 固定分组 |
| P05R028 | 执行确认 | yes | 24 | RAVDESS notebook `L12719`：70/30 row split，random_state=1 |
| P05R029 | 执行确认 | yes | 114 | `DCNN_model.ipynb:L898-L904`：合并表仅按 emotion stratify |
| P05R030 | unknown | unknown | — | notebook `L408-L418,L565-L566` 直接消费外部 Kaggle Train/Test 目录，清单/生成规则缺失 |

## 唯一更正的程序语义

P05R015 的 `modules/formatter.py` 执行顺序是：

1. 文件名的前两字符被解析为 `speaker`（L29–L33）；
2. 从 `np.unique(df['speaker'])` 中无放回抽两个 ID（L56–L59）；
3. `df_train` 选择不在该 speaker 集的所有行，`df_test` 选择在该 speaker 集的所有行
   （L60–L61）。

隔离探针以 10 位 EmoDB 合成 speaker 复现同一逻辑，得到训练 8 位、测试 2 位、交集
`[]`。因此随机的是“挑哪两位说话人”，不是“逐 utterance 随机分到哪一侧”。

## 对既有结论的影响与停止点

- 旧 P0.5 主风险点估计 `21/30` 已被本阶段的更强证据取代为 `20/30`；
  unknown-as-risk 由 `25/30` 相应变为 `24/30`。
- 既有 P2 主情景读取了旧 `21/30`，因此其情景负担数值与当前证据不再同步。
  本阶段按派单**没有重跑或改写 P2**；在后续获准重新传播前，旧 P2 情景量只能视为
  历史冻结结果，不能当作采用本次更正后的当前数字。P1 的同实现协议差不受影响。
- 本阶段不删除或重写历史 P0/P1/P2 表；更正通过独立 change registry、REPORT 追加段
  和 SER 对话末尾记录公开保留谱系。

## 产物与复核

- 主脚本：`tools/run_p0_5_execution_validation.py`
- 独立结构验证器：`tools/verify_p0_5_execution_validation.py`
- 结果目录：`results/execution_validation/`
- 关键入口：`repository_evidence.csv`、`tier_summary.csv`、`change_registry.csv`、
  `synthetic_manifest.csv`、`source_inventory.csv`、`analysis_manifest.json`、
  `verification.json`
- 验证结果：30/30 仓库、25/1/4 层级、20/6/4 判定、4,444 行合成清单、265 个
  代码/文本文件哈希、30 个 per-repository JSON 全部通过；CSV 另经 artifact-tool 导入、
  矩形/表头/设计行列与渲染检查。

本阶段到此停止，等待 Claude 做独立交叉复核；不进入其他研究阶段。

## 2026-08-13 §12 证据分档时序缺口更正

上述“25 个执行确认”是按修订前的 §10.2 写成：当时执行的是我方复刻的 split primitive，
没有执行冻结候选源。按后续 §11.1/§12 的收紧定义，该口径不足以作第一档证据。
本节取代上述层级口径，但保留旧文以维持谱系。

补强后：

- **24 个 `tier1_execution_confirmed`**：从冻结 `.py`/`.ipynb` 解析决定性 split AST
  语句，只在隔离命名空间执行这些语句；不 import 候选模块，不运行整仓。
  每库对完整合成目标语料清单与 15 个固定行序置换做有序 train/test 索引对照，
  与独立写成的 adapter 共 **384/384** 案例一致。
- **1 个 `tier2_static_execution_inference`**：P05R025 仍为 fastai `valid_pct=0.2`
  静态程序语义证据；禁止安装缺失依赖，不声称直接执行交集。
- **5 个 `tier3_unknown`**：旧四项加 P05R008。P05R008 的训练和最终评估依赖仓外
  Kaggle `Train`/`Test` 目录；冻结代码只留有“Actor 20–24”注释与 300 条测试输出，
  并无目录清单或构建规则。人造一个本地目录树只能验证人造输入，不能证明原外部目录互斥，
  故由 `tier1/no` 诚实降为 `tier3/unknown`。

有限 property test 只是**在明示输入域、种子与案例覆盖下的经验等价证据**，
不是一般数学证明。其输入合约是冻结合成 manifest 及其行序置换，输出合约是有序
train/最终评估索引；上游音频/特征生成、仓外清单、模型路径与性能均不在覆盖内。

最终统计为 **20 yes / 5 no / 5 unknown**。P05R008 的 `no→unknown` 不改变主分子
`20/30`，但使 unknown-as-risk 由上一阶段的 `24/30` 回到 `25/30`。对应 Wilson
95% CI 为：主估计 66.7% [48.8%, 80.8%]，unknown-as-risk 83.3% [66.4%, 92.7%]。

补强产物为 `results/execution_validation/equivalence_registry.csv` 和
`equivalence/P05R001.json`–`P05R030.json`；`repository_evidence.csv`、`tier_summary.csv`、
`change_registry.csv`、`analysis_manifest.json` 与 `verification.json` 已同步。验证器不 import
主分析，独立重算记录中的 speaker 交集、对账 384 个案例、冻结 source inventory 和全部哈希，
结果 PASS。
