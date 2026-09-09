# 中文解释版：对应2026年9月9日英文稿

[中文PDF](论文中文解释版.pdf) · [可编辑正文](中文解读.md) · [对应英文稿](../english/xie.pdf) · [全部绝对成绩](../ABSOLUTE_RESULTS.md)

本解释版按研究问题、实验方法、指标读法、结果和结论边界重新组织内容，帮助读者理解英文论文。它不是逐句翻译，也没有加入新的实验、显著性检验或录用概率评估。英文稿及原始结果仍是正式科学表述的依据。

全部16项主检验、11种设置的绝对成绩和必要的反向例子均保留。中文图直接使用[英文图的180行汇总数据](../figures/window_selection_summary.csv)，只翻译标注；其区间仍是24次抽样、每次先合并5折后的逐点t区间。

本版对应的英文源码SHA-256：`043ebfaddb925910b74005fff77b61192d50b5d798931e09fa2512fff6b957a7`。旧日期的中文PDF保留为历史版本，不代表这份英文修订稿。

## 生成与检查

构建需要Python、ReportLab、pypdf；机器检查另需pdfplumber。使用本机合法安装的宋体与黑体字体文件，字体文件不随仓库分发。

从本目录执行，输出到新目录：

```text
python -B build_chinese.py --source 中文解读.md --output /path/to/new-build/中文解释版.pdf --font-dir /path/to/fonts --figure-manifest figures/manifest.json
python -B check_chinese.py --pdf /path/to/new-build/中文解释版.pdf --out /path/to/new-build/checks.json
```

字体目录需包含`simsun.ttc`与`simhei.ttf`；Windows默认读取`C:/Windows/Fonts`。中文图已包含在仓库中。重新绘图使用`make_chinese_figure.py`，输入仍是既有汇总CSV，不调用模型或GPU。图片字体和渲染版本可能影响文件SHA，数值来源不随之改变。

交付文件的数值、字体、版式和逐页视觉检查记录见[机器检查](qa/MACHINE_CHECKS.json)、[交付记录](qa/DELIVERY_QA.json)。这些检查不能替代对科学结论的审阅，也不代表论文已投稿或录用。

图清单中的`manifest_sha256`是规范化JSON载荷的哈希：去掉该字段后，用`json.dumps(sort_keys=True, separators=(',', ':'), ensure_ascii=False)`序列化并编码为UTF-8，再计算SHA-256。它不是清单完整文件的哈希；后者记录在`figures/layout_qa.json`和`qa/BUILD_RECEIPT.json`中。
