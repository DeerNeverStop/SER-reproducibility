# AutoDL 补充实验：独立数值复核器（完整正式结果已通过）

**完成附注，2026-09-08：** 已对真实完整 720 单元归档和冻结评分执行独立复核，通过全部十项主检验及六张 CSV 的比较；199,780 个标量的最大绝对差为 7.105427357601002e−15。见[中文结果报告](../../../docs/autodl-supplement-20260908/RESULTS_REPORT_ZH.md)、[原评分结果](../../../docs/autodl-supplement-20260908/results/results.json)及[核验证据](../../../docs/autodl-supplement-20260908/evidence/)。

`verify_supplement.py` 是冻结科学源码之外的一条数值复核路径。它不导入原 `score.py` 或 `engine.py`，不训练、不做 GPU 推理、不反序列化检查点权重；只在完整归档成立后读取预测数组。真实运行前已用明确标为 **SYNTHETIC** 的自建数据进行验证，下面保留方法、合成测试和独立性边界的记录，不把合成答案当成真实实验发现。

## 使用门槛与命令

需要先具备：720 个正式单元的 `BACKUP_COMPLETE.json`、全部永久小文件、12 个保留检查点的真实本地字节、对应冻结源码 checkout，以及原冻结评分器已经产出的结果目录。完整归档缺一个 UID 就拒绝，随后逐一检查文件 SHA、计划/源码/环境身份、COMPLETE/receipt/attempt、备份 ACK 与非样本权重释放收据。小 ACK 的 `checkpoint_in_backup:false` 允许保持原样，但样本检查点必须另有本地完整字节与归档哈希。

```powershell
& 'E:/科研/SER/ser_gpu/Scripts/python.exe' `
  'D:/SER-final-program-20260908/independent_supplement_check/verify_supplement.py' `
  --backup 'E:/SER-autodl-supplement-20260908/backup' `
  --scores '<冻结评分输出目录>' `
  --repo 'C:/Users/jock8/Documents/ChatGPT/论文/SER-speaker-execution-20260906' `
  --out 'D:/SER-final-program-20260908/independent_supplement_check/FINAL_NEW'
```

输出目录必须全新，且不能位于任何输入目录内或包含输入。复核过程不修改输入；完成前再次核对已用文件 SHA。检查点只做字节 SHA，可能需要读数 GB 的本地数据。脚本目前要求原评分器与复核器使用相同本地归档位置，以精确绑定 `input_audit.json` 中的绝对路径，不接受悄然迁移或另一套同计划结果。

## 独立核对的内容

1. 四规则：分别用 A/B 全验证角色的 float64 CE 和逐类正确计数/Fraction UAR，在每个完整 15/45 窗口重选最优轮；并列用最早轮。逐项比较已提交的 `selected_epochs`；外测不参与任何选点。
2. 对同一外测角色算原生类 macro UAR。先以有理数作 D_CE、D_UAR、J 及同面板骨干/窗口配对差，再把每个 draw 的五折等权平均；统计单位固定为 24 个完整 draw。
3. 方差用有理数中心化后 `math.fsum`，不调用原评分器的 `numpy.std` 路径。复算双侧 t、df=23、pointwise 95% CI 及固定十槽 Holm；零方差的 t/p/CI 保留 NA，槽位不删除。
4. 逐项比较原结果 JSON，以及 `units.csv`（960 行）、`draws.csv`（312 行）、`primary_tests.csv`（10 行）、`descriptive.csv`（104 行）、`selected_epochs.csv`（4,800 行）、`curves.csv`（18,000 行）。保留反向、跨零和不显著结果，不根据效果筛选输出。
5. 另列 16 个描述恒等式：每个模型/语料/窗口的 CE 与 UAR 选点分歧比例 `q`，乘以仅分歧面板的平均外测差 `e`，应精确等于全 120 面板的平均差。无分歧时 `e=NA`、乘积按零处理。**此项是代数分解，不是频率的因果解释，不把 120 个面板当独立 Bernoulli 样本，不增加检验或区间。**

绝对 T 值的单位是 `UAR_percent`；效应、窗口差与骨干差为 `percentage_points`。为兼容原评分文件，保留原 `*_pp` 数字键，解释以显式 `unit` 与比较类别为准。

## 合成验证与独立性边界

14 项测试覆盖：完整网格、已知负主效应及正窗口差、窗口/并列选点、外测改变不影响选择、不平衡 macro recall、错标签/错路径/缺轮/NaN、正负双侧 p 对称、Fraction 精确抵消与零方差 NA、固定 Holm10、无额外描述性 p、缺归档时不读取预测、JSON/六 CSV 的读取与错误符号拒绝，以及 JSON 解析后被替换时不能用新哈希掩盖旧解释。JSON 解析和绑定使用同一次读取的原始字节。

运行测试：

```powershell
Set-Location 'D:/SER-final-program-20260908/independent_supplement_check'
& 'E:/科研/SER/ser_gpu/Scripts/python.exe' -m unittest test_verify_supplement -v
```

合成数值的已知答案检验独立计算；六 CSV 的往返测试只验证接口，不能被当成第二次科学复现。此工具是独立代码路径，不是独立数据、独立训练或另一研究团队的盲复现；它和原评分共享冻结研究定义及基础 NumPy/SciPy 依赖。通过时输出 `independent_replay.json`（全部十项原检验、全部描述与恒等式、比较误差）和 `input_sha256.json`；只有全量门槛与所有数值比较通过才标 `passed:true`。
