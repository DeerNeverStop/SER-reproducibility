# 图1可读性修订

2026-09-07，仅修订 `complete_epoch_curves` 的绘制方式。英文和中文正文、图注与官方模板逐字节沿用 `1c533520503f069172e4c8bcd6926538d2f77d2d`；本目录不代表其他评审意见已经处理。

- [图1矢量PDF](figures/complete_epoch_curves.pdf) · [PNG预览](figures/complete_epoch_curves.png) · [SVG](figures/complete_epoch_curves.svg)
- [替图后的英文PDF](english/xie.pdf) · [替图后的中文PDF](chinese/explainer.pdf)
- [图数据与输入绑定](figures/FIGURE_MANIFEST.json) · [字号检查](qa/figure_geometry.json) · [修改范围检查](qa/scope_verification.json)

黑色外测实线先绘制，橙色未见验证点虚线最后绘制，并在全部15个epoch均值处显示空心圆。圆点是同一条均值曲线的逐轮标记，不是额外样本或区间。蓝色已见验证虚线、图例顺序、0–100共同纵轴、画布尺寸与全部浅灰轨迹保持原样。没有平移、平滑或人为分开近重合的数据。

核对了5400个单独轨迹点和135个均值点，均与原CSV完全相同。中英文仍为8页/5页；图内最小文字分别为9.205/9.533物理PDF点。重建后的全部提取文字相同，只有英文第4页和中文第5页的图区域像素改变；其他11页逐像素一致。已目检新版彩色图、灰度图以及两版实际嵌图。该检查由助手执行，不代表作者批准投稿。

原 `final-20260907` 的106项交付文件、源码包、发布标签和数值结果保持原字节，未被本次替换覆盖。

在仓库根目录重新生成图时，需要原评分目录及与其路径和字节绑定的独立数值审计，输出目录必须全新：

```text
python -B paper/figure-revision-20260907/build_figure.py --results SCORE_DIRECTORY --audit NUMERIC_AUDIT.json --out NEW_FIGURE_DIRECTORY
```

当前双语预览构建命令（工具需安装；中文字体不分发）：

```text
tectonic -X compile paper/figure-revision-20260907/english/main.tex --outdir NEW_ENGLISH_BUILD --keep-logs --keep-intermediates
python -X utf8 -B paper/final-20260907/chinese/build_chinese.py --source paper/figure-revision-20260907/chinese/中文解读.md --output NEW_CHINESE_BUILD/explainer.pdf --font-dir C:/Windows/Fonts --figure-manifest paper/figure-revision-20260907/figures/FIGURE_MANIFEST.json
```
