# AutoDL 价格补充比较

2026-09-06。仅查价和重算，没有登录、充值或租机；不修改原已复核的历史预算。AutoDL 为官方公开标价，尚未验证账户内可租实例及吞吐。

结论：同型号、同运行时间假设下，AutoDL 比上份预算采用的 Runpod Secure Cloud 明显便宜；相对 Runpod Community Cloud 的优势较小。

| 单卡 | AutoDL 元/小时 | Runpod Community 元/小时 | Runpod Secure 元/小时 |
|---|---:|---:|---:|
| RTX4090 24GB | 1.88 | 2.28 | 4.96 |
| RTX5090 32GB | 2.78 | 4.62 | 6.63 |

Runpod 单价复用同日目录查询，美元换算统一按 1 USD≈6.7 CNY 作预算近似，不是支付结算价。[AutoDL 官方公开标价](https://www.autodl.com/home)、[Runpod 价格](https://www.runpod.io/pricing)、[近期汇率参考](https://wise.com/us/currency-converter/usd-to-cny-rate/history/04-09-2026)。会员优惠未计入。

5090 每小时相对 Runpod Secure 约省58%，相对 Community 约省40%；4090 对应约62%和17%。实际完成同一任务的费用仍取决于实际卡型、CPU/RAM配额、功耗及I/O和软件环境，需同脚本先导核验。

| 沿用原5090时长情景 | AutoDL GPU费用（元） | Runpod Secure折算（元） |
|---|---:|---:|
| E0/E1/E2含FT，12.725–29.125h | 35–81 | 84–193 |
| E3生成RTF=1，95.65h | 266 | 634 |
| E3生成RTF=3，253.36h | 704 | 1681 |

上表只替换小时单价，不代表已证明AutoDL与历史5090等速。E3的RTF和生成合格率仍未测，不是报价。存储、人工、模型/数据访问费用另计；准备和回传如果改用无卡模式，实际费用结构还可优化。

AutoDL无卡模式0.10元/h、0.5核/2GB，适合传输/准备，不作为完整训练或大规模解压内存配置；释放GPU后恢复可能要等库存。[官方省钱说明](https://www.autodl.com/docs/save_money/)

默认实例数据盘50GB免费，扩容关机后仍收费且随主机定价；独立文件存储免费20GB、超额0.01元/GB/日，用满200GB约54元/30天。两种存储不是同一产品。[实例数据盘](https://www.autodl.com/docs/local_disk/)、[计费规则](https://api.autodl.com/docs/price/)

建议：将AutoDL5090列为后续新实验首选候选；先短时验证环境、峰值显存及单位任务耗时，再决定正式平台。现有数据和权重已在本地，先传新实验所需输入，不必为了开工迁移全部历史模型备份。
