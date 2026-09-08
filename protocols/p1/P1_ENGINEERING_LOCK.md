# P1 工程实现锁定（首个真实训练结果前）

锁定时间：2026-08-11 14:14:00 UTC-04:00  
状态：**已锁定；此刻 `formal_start.json`、`units/`、`attempt_ledger/` 均不存在，1,620 个训练单元全部 pending，尚未产生或读取任何 P1 真实模型数字。**

本文件只补齐 `P1_PREREGISTRATION.md` 已允许的工程细节，不改变数据总体、四模型、三协议、seeds、超参、指标、contrast、统计 family 或算力上限。

## 1. 冻结身份与固定顺序

- `P1_PREREGISTRATION.md` 原始冻结前缀：12,426 bytes，SHA-256 `91db3edc544f76fc412f4dfc5f35adef23934635ca3ede36c8089c38bd14b782`；今后只允许在该前缀后追加偏离日志。
- SER commit：`d11768ed31a91f611472b89e52f3f6bb97030c40`；预注册四个关键文件及传递依赖 `fno_model.py` 均在 import 前逐字节核验。SER 模块只在哈希门通过后延迟 import，禁写 bytecode，并核对模块 `__file__` 指向只读 SER 根目录。
- 运行顺序固定为 corpus `ravdess → cremad`，每个语料内 model `cnn → resnet_se → transformer → fno`，protocol `random → groupkfold → loso`，seed `0 → 1 → 2`，outer fold 升序。结果不能改变顺序或触发择优停止。
- `run_plan.csv` 固定 1,620 行：RAVDESS 408，CREMA-D 1,212；SHA-256 `ffac41581a09a769d7f299c2e27a94f75d7ca6b594196b3c81843cbe4d01296a`。

## 2. 数据、缓存与 split 序列化

| corpus | manifest SHA-256 | log-mel cache SHA-256 | outer assignment SHA-256 | inner inventory SHA-256 |
|---|---|---|---|---|
| RAVDESS | `259b453ce6135242b03a34d113209a4b8a0234687c9717e56fbe2bd75acb7ab2` | `0e73160e1a5b88b13b437091332ec9479722e8feecb92061b9b6c0fb571a79f8` | `6cfa5aea57eb03b4fa1f1e8596bd081ee83866436775948d0632defaa91b953b` | `8c3d7f6907d098195f3e4f153e01e34db29fcbe054a25636380e6376994a6af7` |
| CREMA-D | `8b628265bcea9ed36ac96a0fffe753423a8c4d8c87befb5775a2ca36c673c9b8` | `c49136162c65e38a1f39a62e4de6878ec490a40947e02ab82f35da53c9e06fef` | `81cf1f28685582b92a0ad9c58e7a743dbc992416fa9d1576a11bd388a0595f93` | `57afd4e7c90b61a68e293f7e7e5bf6dfccccc612310e990bc648f6e2ff787a0b` |

- canonical manifest 顺序是语料根目录下 POSIX 相对路径的字典序；CSV 使用 UTF-8、逗号分隔、LF/标准 `csv` 转义。RAVDESS 标签顺序固定为 `neutral, calm, happy, sad, angry, fearful, disgust, surprised`；CREMA-D 固定为 `angry, disgust, fearful, happy, neutral, sad`。
- 预处理函数与完整参数的 canonical JSON 共同定义 code hash `33bc3b5e4b6a7a9693c3ffa41d63166bb9ed827e662ff83ee747cfcd9a883d61`；缓存名含其前 16 位并含所有预注册内容参数。缓存 NPZ 内再次保存完整签名与 manifest hash；formal run 同时硬核 manifest、cache、run-plan 和 inner-NPZ 哈希。
- inner key 的逐字节形式为 UTF-8 `P1-inner|{corpus}|{protocol}|{outer_fold}|{seed}`；`outer_fold` 为 0-based 十进制，token 只允许 `ravdess|cremad` 与 `random|groupkfold|loso`。随机状态定义为 `int.from_bytes(SHA256(key).digest()[:4], "big")`；重试为该无符号 32-bit 基值依次 `+0...+99 (mod 2^32)`。
- model/loader/augment RNG key 固定为 `P1-train|{corpus}|{protocol}|{outer_fold}|{seed}`，使用同一 big-endian 前 32-bit 规则；不含 model token，所以相同 corpus/protocol/fold/seed 跨模型复用相同 split、batch 与增强随机流起点。
- 405 个 inner split 全部 offset 0 即类别齐全；合成门检 141 项通过。随机 outer 五折均为 RAVDESS `24/24`、CREMA-D `91/91` 位说话人重叠；GroupKFold/LOSO outer 与 grouped inner 的 speaker overlap 均为 0；三个 outer protocol 都逐 utterance exactly-once OOF。

## 3. 训练循环的未歧义实现

- 四模型只由冻结的 `advanced_models.build_neural_model` 构造，结构/训练字典严格等于 `candidate_configs(model)[2]`；输入 `[B,64,128]`，输出 corpus 固定类别顺序的 logits。
- float32、无 AMP、TF32 关闭；`torch.use_deterministic_algorithms(True)`、cuDNN deterministic 开、benchmark 关；DataLoader `num_workers=0`，batch size 64。模型 build 前设稳定 train seed，build 后将同一 seed 复位给 dropout、shuffle 与增强流。
- class weights 只由 fit 子集计算。训练梯度使用 PyTorch 标准 batch weighted-mean CE；checkpoint 的 validation loss 明确定义为**整个 validation 集** `sum(weighted per-sample CE) / sum(target class weights)`，不采用 SER 旧 helper 的 batch-size 加权近似，也不以 UAR 打破近似并列。
- checkpoint 只在 validation loss 严格 `<` 历史最小值时更新；patience 15、最多 100 epochs、CosineAnnealingLR `T_max=100`，恢复该严格最小 loss checkpoint。validation/test 均无增强；outer test 仅在恢复 checkpoint 后做一次完整 loader pass。
- 任何 outer-test access 后的推理或落盘异常都禁止重新训练/再次访问 test，单元记未提交并停止；只有发生在 outer-test access 之前的模型/训练异常才允许相同配置重试一次。成功计算后的文件提交只对同一内存结果重试，不重训。
- 全局独占锁防止两个 formal runner 并发处理同一单元。每次 attempt 在训练前原子建 ledger，每 epoch heartbeat；计时保守覆盖该 attempt 的训练和 outer-test 段。200 GPU-hour/14-day 在单元前和 epoch 边界检查，20 GiB 在单元前检查；触顶立即停止新增训练并保留 pending/失败位置。

## 4. 原始产物与缺失传播

- 每个成功 fit 原子目录必须包含 `config.json`、连续逐 epoch `history.csv`、outer-test 逐 utterance `predictions.csv`（全部 logits）和 `run.json`；三个文件 SHA 写回 `run.json`。失败两次则保留两次异常、traceback、耗时与 ledger，不伪装成成功。
- `corpus × model × protocol × seed` 只有在全部计划 outer folds success、每个 unit 自洽且 OOF 覆盖恰好一次时才是完整 cell。任一 fold missing/failed/invalid 使整个 cell 缺失；不得按预测交集删样本。三-seed protocol 汇总和任何依赖该 cell 的 contrast 同步缺失并写 `not conclusive`。
- 所有主表从逐 utterance CSV 重新计算，不信任训练日志中的汇总；outer fold 绝不作为独立重复。

## 5. 统计实现锁定

- UAR 使用 corpus 固定全部标签顺序；任一 OOF 或 speaker 子集缺真实类别即硬失败，不事后缩小类别分母。每 speaker 在每 protocol/seed 上使用完全相同的全部 utterances；三 seed 先在 speaker 内平均，再计算 A−B 的 `RG/RL/GL`。
- 95% CI：每个 `corpus × model × contrast × metric` 独立把 RNG 重置到 `202608101744`，整 speaker 有放回重抽 10,000 次，取 NumPy linear percentile 的 2.5/97.5 分位。accuracy/macro-F1 也给相同 CI，但不做显著性检验。
- UAR Wilcoxon 固定 `alternative="two-sided"`、`zero_method="wilcox"`、`correction=False`、`method="auto"`；全零差固定 `p=1`。只有 SciPy 定义域失败才给双侧 exact binomial sign-test 敏感性，不能替换主检验。
- Holm family 始终为预注册 24 项。缺失确认性检验在排序/校正中以保守 `p=1` 占位以维持 family=24，但公开 raw/adjusted p 留空且不拒绝；不得把 family 缩成“完成的比较”。
- 最终独立 QA 必须另外从逐 utterance 源重算关键数值；当前验证器对结构、哈希、OOF、方向、缺失、history、attempt 和预算是交叉检查，但 bootstrap/Wilcoxon 与主汇总共享部分实现，不能冒充完全独立第二统计实现。

## 6. 已知输入事实（不改总体）

- RAVDESS 含 1 组同 actor/同 label exact duplicate 与 5 个双声道文件；CREMA-D 含 3 组同 speaker、冲突 label 的 exact duplicates，以及 `1040_ITH_SAD_X.wav` 官方内容一致的命名变体。全部按固定“可读 WAV 全纳入”规则原样保留。
- CREMA-D `1076_MTI_SAD_XX.wav`（manifest index 6187）为可读但全零音频，预处理后是唯一零方差/全零谱；不删除、不改名、不单独调整模型。

## 7. 工具哈希

- `tools/p1_protocol_core.py`: `16a4d2ed96892674f6132c4ea42dea1634bcd605f2467cb88551314984f86180`
- `tools/run_p1_protocol_premium.py`: `8cf2232854919eacd676c860ce8ff990ab74e2fa69be2f2826db645cd17b2cbc`
- `tools/summarize_p1_protocol_premium.py`: `13af00739dc18d7c84b6e48735541ec4df5b1d82fdef9d3e7c738825739d3404`
- `tools/verify_p1_protocol_premium.py`: `c80b6d81acbed47b6a9dd49b8728b6f56593132256a9dda3240d4d2c98b03e75`
- 独立首训前 QA：`work/p1_preflight_review.md`，结论 PASS；没有查看或生成 P1 真实指标。

锁定后不得因运行数字修改上述内容；若实现错误迫使改变，保留旧结果并按 `P1_PREREGISTRATION.md` §9 追加实质/非实质偏离说明。

## 8. 运行中非实质校验修订（追加，不覆盖原锁定）

- 2026-08-11 14:34:21 UTC-04:00：首批 RAVDESS 正式单元已开始生成，但尚未读取任何 outer-test 性能、OOF 指标、协议差值或显著性结果。独立静态复核发现原汇总器把其自身写入的 `results/protocol_premium/summary/` 派生文件计入 `output_bytes`；因此第一次汇总后，验证器的 fresh reanalysis 会因派生文件自计数而产生确定性不一致。
- 修订仅令预算体积扫描排除派生 `summary/` 子树，正式 runner 管辖的 manifests、cache、splits、plan、ledger、units 与 logs 仍全部计入。它不改变语料、模型、协议、split、seed、训练、测试、预测、指标、contrast、CI、Wilcoxon、Holm family 或预算上限，也不改变已经运行或随后运行的任何 fit。
- 修订前汇总器 SHA-256 保留为 `13af00739dc18d7c84b6e48735541ec4df5b1d82fdef9d3e7c738825739d3404`；修订后为 `a3a43530f2e65bec6c97757a583c1549302d476b1a0a40cd6d56ce82056bc181`。验证器 SHA-256 未变，仍为 `c80b6d81acbed47b6a9dd49b8728b6f56593132256a9dda3240d4d2c98b03e75`。clean synthetic tree 的“单次汇总→fresh verify→再次计数”测试保持 formal bytes `19→19→19`，未读取正式预测或正式性能。
