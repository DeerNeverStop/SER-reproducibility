# 新程序 Methods 准备稿来源映射

2026-09-07。仅描述冻结方法；未读取正式 384 单元研究的科学结果，不宣称已经完成。英文 `final_program_methods_en.tex` 与中文 `final_program_methods_zh.md` 的 M1–M8 段逐点对应。以下源码路径相对于当前仓库根；行号来自本次只读核对。实际实现优先于设计中未启用的可选项目。

| 对应段 | 核对内容 | 直接来源 |
|---|---|---|
| M1 | 360 A＋24 B；三库原生类别与说话人数；既往数据接触；私有执行前冻结，非公开预注册 | `v3/final_program_20260907/SCIENCE_DESIGN.md:3–27`；`plan.py:18–49,180–193`；`run.py:93–116` |
| M2、表 | 按来源性别分层的 outer 五折；A/B完整代表格、三组共同query文本且与fit文本不交；人数/文本/录音预算；缺格边界 | `plan.py:25–34,82–134,137–177,205–239` |
| M3 | cfg3数值；top4＋原生分类头；mean pooling；AdamW、无权CE、完整15轮、FP16训练/FP32评估；波形处理 | `SCIENCE_DESIGN.md:13`；`plan.py:21–34`；`engine.py:36–85,163–218`；旧依赖 `v3/deploy/engines_deploy.py:187–208` |
| M4 | 四条规则、exact ties、whole-role UAR；三组分开batch16；不消除padding；RNG/训练模式 | `engine.py:88–128,190–230`；`score.py:51–77,97–122`；`plan.py:175,235`；`SCIENCE_DESIGN.md:55–59` |
| M5 | deltaCE、deltaUAR、J的符号/单位与算法对照范围 | `plan.py:36–49`；`score.py:183–203,264–271`；`SCIENCE_DESIGN.md:33–37,53` |
| M6 | fold→draw聚合；24 draw/df23；六项Holm；pointwise CI；NA；1pp非等效；跨库不合并 | `score.py:131–160,194–203,218–255,264–271`；`plan.py:40–48`；`SCIENCE_DESIGN.md:39–45` |
| M7 | 仅CREMA每draw固定fold0追加B；同初始化/分流seed/匹配槽；B仅last；E整组替换且query已参与A选择 | `plan.py:137–193,240–249`；`engine.py:163–233`；`run.py:439–447`；`score.py:204–217,261–270`；`SCIENCE_DESIGN.md:61–65` |
| M8 | pilot排除；完整gate后评分；全epoch logits、去重states、磁盘回载；曲线/epoch/middle-late/oracle仅描述 | `plan.py:180–193`；`engine.py:213–272`；`run.py:421–458`；`score.py:174–203,238–255,284–338` |

边界说明：

- 外测人数 CREMA-D 17/19、SUBESCO 4、RAVDESS 4/6 由来源性别人数与 `plan.py:141–142` 的五折分配确定；不把剩余所有开发人误称为 outer。CREMA-D 外测query有可用代表缺格，故不写统一固定条数；每人全部类别仍须支持（`plan.py:230–231`）。
- `SCIENCE_DESIGN.md:41` 提及的 Bonferroni 同时区间仅是可选讨论，冻结 `score.py` 并未输出该区间，准备稿只写实际 pointwise 95% t CI。
- 旧 `v3/deploy/engines_deploy.py` 仅作为 WaveCache 的实际预处理来源；不继承该文件旧训练器的类别权重或提前停止。新训练器是 `final_program_20260907/engine.py`。
- 稿件不用“已完成384”“结果证实”“独立全新语料确认”等措辞；不加入硬件、租费、实际耗时或平台迁移等尚在进行的运行事实。
