# Paper number audit

- Draft: `E:\科研\claudework\07-ser-repro-protocol-audit\paper\icassp2027_demo\main_numbercheck.tex`
- Fact catalog: `E:\科研\claudework\07-ser-repro-protocol-audit\paper\icassp2027_demo\paper_facts_07.json`
- Matched: 102
- Red: 0
- Unreferenced: 0
- Excluded diagnostics: 62
- Multi-value fact bindings downgraded: 0 fact group(s) / 0 token(s)

## 匹配 ✅

| Location | Draft | Fact | Expected | Delta | Provenance |
|---|---:|---|---:|---:|---|
| L15:C23 | 66.7 | `audit.rate_pct` | 66.7 | 0.0 | `tier_summary:20/30 displayed` |
| L15:C47 | 48.8 | `audit.ci_low_pct` | 48.8 | 0.0 | `tier_summary:Wilson interval from 20/30` |
| L15:C55 | 80.8 | `audit.ci_high_pct` | 80.8 | 0.0 | `tier_summary:Wilson interval from 20/30` |
| L17:C32 | 15.39 | `ravdess.fno.delta_pct` | 15.386284722222223 | 0.003715277777777 | `premium:ravdess/fno/RG/uar/point_mean_difference` |
| L18:C27 | 19.68 | `ravdess.resnet_se.delta_pct` | 19.68315972222222 | -0.00315972222222 | `premium:ravdess/resnet_se/RG/uar/point_mean_difference` |
| L19:C25 | 1.82 | `cremad.fno.delta_pct` | 1.8169152784537408 | 0.0030847215462592 | `premium:cremad/fno/RG/uar/point_mean_difference` |
| L20:C29 | 3.06 | `cremad.transformer.delta_pct` | 3.061168583146604 | -0.001168583146604 | `premium:cremad/transformer/RG/uar/point_mean_difference` |
| L72:C31 | 384 | `audit.property_checks` | 384 | 0 | `tier_summary:24*16 derived` |
| L73:C29 | 16 | `audit.property_checks_per_repository` | 16 | 0 | `tier_summary:384/24 derived` |
| L76:C27 | 30 | `audit.n` | 30 | 0 | `tier_summary:split_risk_final denominator` |
| L83:C21 | 1,440 | `dataset.ravdess.utterances` | 1440 | 0 | `prereg:RAVDESS utterances` |
| L84:C18 | 24 | `dataset.ravdess.speakers` | 24 | 0 | `prereg:RAVDESS speakers` |
| L85:C21 | 7,442 | `dataset.cremad.utterances` | 7442 | 0 | `prereg:CREMA-D utterances` |
| L86:C18 | 91 | `dataset.cremad.speakers` | 91 | 0 | `prereg:CREMA-D speakers` |
| L113:C34 | 24 | `stats.holm_family` | 24 | 0 | `premium:fixed family size` |
| L126:C36 | 66.7 | `audit.rate_pct` | 66.7 | 0.0 | `tier_summary:20/30 displayed` |
| L126:C60 | 48.8 | `audit.ci_low_pct` | 48.8 | 0.0 | `tier_summary:Wilson interval from 20/30` |
| L126:C68 | 80.8 | `audit.ci_high_pct` | 80.8 | 0.0 | `tier_summary:Wilson interval from 20/30` |
| L127:C75 | 25 | `audit.sensitivity_count` | 25 | 0 | `tier_summary:20+5 unknown` |
| L128:C26 | 30 | `audit.n` | 30 | 0 | `tier_summary:split_risk_final denominator` |
| L129:C26 | 83.3 | `audit.sensitivity_pct` | 83.3 | 0.0 | `tier_summary:25/30 displayed` |
| L129:C50 | 66.4 | `audit.sensitivity_ci_low_pct` | 66.4 | 0.0 | `tier_summary:Wilson interval from 25/30` |
| L129:C58 | 92.7 | `audit.sensitivity_ci_high_pct` | 92.7 | 0.0 | `tier_summary:Wilson interval from 25/30` |
| L142:C21 | 15.39 | `ravdess.fno.delta_pct` | 15.386284722222223 | 0.003715277777777 | `premium:ravdess/fno/RG/uar/point_mean_difference` |
| L143:C27 | 19.68 | `ravdess.resnet_se.delta_pct` | 19.68315972222222 | -0.00315972222222 | `premium:ravdess/resnet_se/RG/uar/point_mean_difference` |
| L144:C25 | 1.82 | `cremad.fno.delta_pct` | 1.8169152784537408 | 0.0030847215462592 | `premium:cremad/fno/RG/uar/point_mean_difference` |
| L145:C29 | 3.06 | `cremad.transformer.delta_pct` | 3.061168583146604 | -0.001168583146604 | `premium:cremad/transformer/RG/uar/point_mean_difference` |
| L148:C67 | 19 | `stats.holm_significant` | 19 | 0 | `premium:count holm_reject_0_05=true for UAR` |
| L149:C34 | 24 | `stats.holm_family` | 24 | 0 | `premium:fixed family size` |
| L158:C24 | 56.86 | `ravdess.cnn.random.mean_pct` | 56.85763888888889 | 0.00236111111111 | `metric_summary:ravdess/cnn/random/uar/mean` |
| L158:C36 | 1.17 | `ravdess.cnn.random.sd_pct` | 1.1652251526613557 | 0.0047748473386443 | `metric_summary:ravdess/cnn/random/uar/sample_sd` |
| L158:C43 | 39.15 | `ravdess.cnn.groupkfold.mean_pct` | 39.14930555555555 | 0.00069444444445 | `metric_summary:ravdess/cnn/groupkfold/uar/mean` |
| L158:C55 | 0.52 | `ravdess.cnn.groupkfold.sd_pct` | 0.522187908659548 | -0.002187908659548 | `metric_summary:ravdess/cnn/groupkfold/uar/sample_sd` |
| L158:C62 | 45.79 | `ravdess.cnn.loso.mean_pct` | 45.78993055555556 | 0.00006944444444 | `metric_summary:ravdess/cnn/loso/uar/mean` |
| L158:C74 | 1.81 | `ravdess.cnn.loso.sd_pct` | 1.8093024606853068 | 0.0006975393146932 | `metric_summary:ravdess/cnn/loso/uar/sample_sd` |
| L158:C81 | 17.71 | `ravdess.cnn.delta_pct` | 17.708333333333336 | 0.001666666666664 | `premium:ravdess/cnn/RG/uar/point_mean_difference` |
| L158:C88 | 15.04 | `ravdess.cnn.ci_low_pct` | 15.0390625 | 0.0009375 | `premium:ravdess/cnn/RG/uar/ci95_low` |
| L158:C95 | 20.55 | `ravdess.cnn.ci_high_pct` | 20.551215277777775 | -0.001215277777775 | `premium:ravdess/cnn/RG/uar/ci95_high` |
| L159:C30 | 66.04 | `ravdess.resnet_se.random.mean_pct` | 66.03732638888889 | 0.00267361111111 | `metric_summary:ravdess/resnet_se/random/uar/mean` |
| L159:C42 | 1.53 | `ravdess.resnet_se.random.sd_pct` | 1.530063345231974 | -0.000063345231974 | `metric_summary:ravdess/resnet_se/random/uar/sample_sd` |
| L159:C49 | 46.35 | `ravdess.resnet_se.groupkfold.mean_pct` | 46.35416666666667 | -0.00416666666667 | `metric_summary:ravdess/resnet_se/groupkfold/uar/mean` |
| L159:C61 | 2.50 | `ravdess.resnet_se.groupkfold.sd_pct` | 2.502067558668704 | -0.002067558668704 | `metric_summary:ravdess/resnet_se/groupkfold/uar/sample_sd` |
| L159:C68 | 50.76 | `ravdess.resnet_se.loso.mean_pct` | 50.75954861111111 | 0.00045138888889 | `metric_summary:ravdess/resnet_se/loso/uar/mean` |
| L159:C80 | 2.51 | `ravdess.resnet_se.loso.sd_pct` | 2.5099605358725667 | 0.0000394641274333 | `metric_summary:ravdess/resnet_se/loso/uar/sample_sd` |
| L159:C87 | 19.68 | `ravdess.resnet_se.delta_pct` | 19.68315972222222 | -0.00315972222222 | `premium:ravdess/resnet_se/RG/uar/point_mean_difference` |
| L159:C94 | 16.62 | `ravdess.resnet_se.ci_low_pct` | 16.623263888888893 | -0.003263888888893 | `premium:ravdess/resnet_se/RG/uar/ci95_low` |
| L159:C101 | 22.94 | `ravdess.resnet_se.ci_high_pct` | 22.938368055555554 | 0.001631944444446 | `premium:ravdess/resnet_se/RG/uar/ci95_high` |
| L160:C32 | 57.36 | `ravdess.transformer.random.mean_pct` | 57.356770833333336 | 0.003229166666664 | `metric_summary:ravdess/transformer/random/uar/mean` |
| L160:C44 | 1.73 | `ravdess.transformer.random.sd_pct` | 1.7261814561930846 | 0.0038185438069154 | `metric_summary:ravdess/transformer/random/uar/sample_sd` |
| L160:C51 | 41.02 | `ravdess.transformer.groupkfold.mean_pct` | 41.015625 | 0.004375 | `metric_summary:ravdess/transformer/groupkfold/uar/mean` |
| L160:C63 | 1.33 | `ravdess.transformer.groupkfold.sd_pct` | 1.3294647042097745 | 0.0005352957902255 | `metric_summary:ravdess/transformer/groupkfold/uar/sample_sd` |
| L160:C70 | 42.53 | `ravdess.transformer.loso.mean_pct` | 42.53472222222223 | -0.00472222222223 | `metric_summary:ravdess/transformer/loso/uar/mean` |
| L160:C82 | 2.40 | `ravdess.transformer.loso.sd_pct` | 2.3976843129112684 | 0.0023156870887316 | `metric_summary:ravdess/transformer/loso/uar/sample_sd` |
| L160:C89 | 16.34 | `ravdess.transformer.delta_pct` | 16.341145833333336 | -0.001145833333336 | `premium:ravdess/transformer/RG/uar/point_mean_difference` |
| L160:C96 | 12.59 | `ravdess.transformer.ci_low_pct` | 12.586805555555555 | 0.003194444444445 | `premium:ravdess/transformer/RG/uar/ci95_low` |
| L160:C103 | 20.33 | `ravdess.transformer.ci_high_pct` | 20.334201388888886 | -0.004201388888886 | `premium:ravdess/transformer/RG/uar/ci95_high` |
| L161:C24 | 55.79 | `ravdess.fno.random.mean_pct` | 55.794270833333336 | -0.004270833333336 | `metric_summary:ravdess/fno/random/uar/mean` |
| L161:C36 | 1.24 | `ravdess.fno.random.sd_pct` | 1.2369791666666718 | 0.0030208333333282 | `metric_summary:ravdess/fno/random/uar/sample_sd` |
| L161:C43 | 40.41 | `ravdess.fno.groupkfold.mean_pct` | 40.407986111111114 | 0.002013888888886 | `metric_summary:ravdess/fno/groupkfold/uar/mean` |
| L161:C55 | 2.13 | `ravdess.fno.groupkfold.sd_pct` | 2.129944610644959 | 0.000055389355041 | `metric_summary:ravdess/fno/groupkfold/uar/sample_sd` |
| L161:C62 | 42.51 | `ravdess.fno.loso.mean_pct` | 42.51302083333333 | -0.00302083333333 | `metric_summary:ravdess/fno/loso/uar/mean` |
| L161:C74 | 0.69 | `ravdess.fno.loso.sd_pct` | 0.6889977372564066 | 0.0010022627435934 | `metric_summary:ravdess/fno/loso/uar/sample_sd` |
| L161:C81 | 15.39 | `ravdess.fno.delta_pct` | 15.386284722222223 | 0.003715277777777 | `premium:ravdess/fno/RG/uar/point_mean_difference` |
| L161:C88 | 12.28 | `ravdess.fno.ci_low_pct` | 12.28244357638889 | -0.00244357638889 | `premium:ravdess/fno/RG/uar/ci95_low` |
| L161:C95 | 18.82 | `ravdess.fno.ci_high_pct` | 18.815104166666664 | 0.004895833333336 | `premium:ravdess/fno/RG/uar/ci95_high` |
| L162:C17 | 58.63 | `cremad.cnn.random.mean_pct` | 58.63045152990628 | -0.00045152990628 | `metric_summary:cremad/cnn/random/uar/mean` |
| L162:C29 | 0.59 | `cremad.cnn.random.sd_pct` | 0.5937091377579127 | -0.0037091377579127 | `metric_summary:cremad/cnn/random/uar/sample_sd` |
| L162:C36 | 55.84 | `cremad.cnn.groupkfold.mean_pct` | 55.83963913058129 | 0.00036086941871 | `metric_summary:cremad/cnn/groupkfold/uar/mean` |
| L162:C48 | 0.84 | `cremad.cnn.groupkfold.sd_pct` | 0.8409847705411085 | -0.0009847705411085 | `metric_summary:cremad/cnn/groupkfold/uar/sample_sd` |
| L162:C55 | 57.59 | `cremad.cnn.loso.mean_pct` | 57.5862341850412 | 0.0037658149588 | `metric_summary:cremad/cnn/loso/uar/mean` |
| L162:C67 | 0.31 | `cremad.cnn.loso.sd_pct` | 0.3062937571569566 | 0.0037062428430434 | `metric_summary:cremad/cnn/loso/uar/sample_sd` |
| L162:C74 | 2.80 | `cremad.cnn.delta_pct` | 2.800480959821619 | -0.000480959821619 | `premium:cremad/cnn/RG/uar/point_mean_difference` |
| L162:C80 | 2.07 | `cremad.cnn.ci_low_pct` | 2.0657170210741618 | 0.0042829789258382 | `premium:cremad/cnn/RG/uar/ci95_low` |
| L162:C86 | 3.53 | `cremad.cnn.ci_high_pct` | 3.5272175931516605 | 0.0027824068483395 | `premium:cremad/cnn/RG/uar/ci95_high` |
| L163:C23 | 61.24 | `cremad.resnet_se.random.mean_pct` | 61.23599657814543 | 0.00400342185457 | `metric_summary:cremad/resnet_se/random/uar/mean` |
| L163:C35 | 0.64 | `cremad.resnet_se.random.sd_pct` | 0.6420205573611136 | -0.0020205573611136 | `metric_summary:cremad/resnet_se/random/uar/sample_sd` |
| L163:C42 | 58.79 | `cremad.resnet_se.groupkfold.mean_pct` | 58.78934402900132 | 0.00065597099868 | `metric_summary:cremad/resnet_se/groupkfold/uar/mean` |
| L163:C54 | 0.79 | `cremad.resnet_se.groupkfold.sd_pct` | 0.7883961867451099 | 0.0016038132548901 | `metric_summary:cremad/resnet_se/groupkfold/uar/sample_sd` |
| L163:C61 | 60.34 | `cremad.resnet_se.loso.mean_pct` | 60.34194177298037 | -0.00194177298037 | `metric_summary:cremad/resnet_se/loso/uar/mean` |
| L163:C73 | 0.43 | `cremad.resnet_se.loso.sd_pct` | 0.43122844523366244 | -0.00122844523366244 | `metric_summary:cremad/resnet_se/loso/uar/sample_sd` |
| L163:C80 | 2.46 | `cremad.resnet_se.delta_pct` | 2.4555979226308895 | 0.0044020773691105 | `premium:cremad/resnet_se/RG/uar/point_mean_difference` |
| L163:C86 | 1.65 | `cremad.resnet_se.ci_low_pct` | 1.6522310615717202 | -0.0022310615717202 | `premium:cremad/resnet_se/RG/uar/ci95_low` |
| L163:C92 | 3.27 | `cremad.resnet_se.ci_high_pct` | 3.2736965669657976 | -0.0036965669657976 | `premium:cremad/resnet_se/RG/uar/ci95_high` |
| L164:C25 | 52.01 | `cremad.transformer.random.mean_pct` | 52.00507584207516 | 0.00492415792484 | `metric_summary:cremad/transformer/random/uar/mean` |
| L164:C37 | 0.58 | `cremad.transformer.random.sd_pct` | 0.58352240794394 | -0.00352240794394 | `metric_summary:cremad/transformer/random/uar/sample_sd` |
| L164:C44 | 48.95 | `cremad.transformer.groupkfold.mean_pct` | 48.94520697885259 | 0.00479302114741 | `metric_summary:cremad/transformer/groupkfold/uar/mean` |
| L164:C56 | 0.31 | `cremad.transformer.groupkfold.sd_pct` | 0.30686102935906806 | 0.00313897064093194 | `metric_summary:cremad/transformer/groupkfold/uar/sample_sd` |
| L164:C63 | 51.54 | `cremad.transformer.loso.mean_pct` | 51.54295497906459 | -0.00295497906459 | `metric_summary:cremad/transformer/loso/uar/mean` |
| L164:C75 | 0.40 | `cremad.transformer.loso.sd_pct` | 0.3964106048770779 | 0.0035893951229221 | `metric_summary:cremad/transformer/loso/uar/sample_sd` |
| L164:C82 | 3.06 | `cremad.transformer.delta_pct` | 3.061168583146604 | -0.001168583146604 | `premium:cremad/transformer/RG/uar/point_mean_difference` |
| L164:C88 | 2.29 | `cremad.transformer.ci_low_pct` | 2.2939043303054283 | -0.0039043303054283 | `premium:cremad/transformer/RG/uar/ci95_low` |
| L164:C94 | 3.82 | `cremad.transformer.ci_high_pct` | 3.8218198417374234 | -0.0018198417374234 | `premium:cremad/transformer/RG/uar/ci95_high` |
| L165:C17 | 53.37 | `cremad.fno.random.mean_pct` | 53.37302147393079 | -0.00302147393079 | `metric_summary:cremad/fno/random/uar/mean` |
| L165:C29 | 0.48 | `cremad.fno.random.sd_pct` | 0.4751411187622234 | 0.0048588812377766 | `metric_summary:cremad/fno/random/uar/sample_sd` |
| L165:C36 | 51.55 | `cremad.fno.groupkfold.mean_pct` | 51.55277869661505 | -0.00277869661505 | `metric_summary:cremad/fno/groupkfold/uar/mean` |
| L165:C48 | 0.54 | `cremad.fno.groupkfold.sd_pct` | 0.5355315402864387 | 0.0044684597135613 | `metric_summary:cremad/fno/groupkfold/uar/sample_sd` |
| L165:C55 | 53.86 | `cremad.fno.loso.mean_pct` | 53.85639421874825 | 0.00360578125175 | `metric_summary:cremad/fno/loso/uar/mean` |
| L165:C67 | 0.11 | `cremad.fno.loso.sd_pct` | 0.10832182118440006 | 0.00167817881559994 | `metric_summary:cremad/fno/loso/uar/sample_sd` |
| L165:C74 | 1.82 | `cremad.fno.delta_pct` | 1.8169152784537408 | 0.0030847215462592 | `premium:cremad/fno/RG/uar/point_mean_difference` |
| L165:C80 | 1.09 | `cremad.fno.ci_low_pct` | 1.0888677612578719 | 0.0011322387421281 | `premium:cremad/fno/RG/uar/ci95_low` |
| L165:C86 | 2.51 | `cremad.fno.ci_high_pct` | 2.505388511569832 | 0.004611488430168 | `premium:cremad/fno/RG/uar/ci95_high` |
| L213:C76 | 20 | `audit.yes` | 20 | 0 | `tier_summary:split_risk_final=yes` |

## 草稿有但数据源没有或不一致 🔴

None.

## 数据源有但草稿未引用 ℹ️

None.

## Excluded diagnostics

`citation`: 16, `confidence_level`: 4, `configuration`: 2, `identifier`: 1, `protocol_metadata`: 36, `year`: 3

| Location | Token | Reason | Context |
|---|---:|---|---|
| L15:C38 | 95 | `confidence_level` | The observed rate was 66.7\% (Wilson 95\% CI [48.8\%, 80.8\%]). |
| L34:C76 | 1 | `citation` | Prior studies show that SER performance depends on partition construction [1]-[5], while benchmark efforts increasingly distribute fixed speaker-independent splits [7], [8]. |
| L34:C80 | 5 | `citation` | Prior studies show that SER performance depends on partition construction [1]-[5], while benchmark efforts increasingly distribute fixed speaker-independent splits [7], [8]. |
| L34:C166 | 7 | `citation` | Prior studies show that SER performance depends on partition construction [1]-[5], while benchmark efforts increasingly distribute fixed speaker-independent splits [7], [8]. |
| L34:C171 | 8 | `citation` | Prior studies show that SER performance depends on partition construction [1]-[5], while benchmark efforts increasingly distribute fixed speaker-independent splits [7], [8]. |
| L35:C26 | 2026 | `year` | The closest collision, a 2026 study, already contrasts random and speaker-independent evaluation on RAVDESS and CREMA-D and reports a larger change on RAVDESS [5]. |
| L35:C161 | 5 | `citation` | The closest collision, a 2026 study, already contrasts random and speaker-independent evaluation on RAVDESS and CREMA-D and reports a larger change on RAVDESS [5]. |
| L52:C92 | 1 | `citation` | SER evaluation has long distinguished speaker-dependent from speaker-independent settings [1]. |
| L53:C109 | 2 | `citation` | Direct comparisons later showed that fold criteria alter absolute performance and sometimes model rankings [2]-[5]. |
| L53:C113 | 5 | `citation` | Direct comparisons later showed that fold criteria alter absolute performance and sometimes model rankings [2]-[5]. |
| L54:C112 | 6 | `citation` | Reproduction work has also found that public SER systems can be hard to reconstruct under one common protocol [6]. |
| L55:C18 | 6 | `citation` | Antoniou et al. [6] audit a purposive IEMOCAP case study; our sampling estimand is instead a frozen public-code candidate frame. |
| L57:C8 | 7 | `citation` | SERAB [7] and Open-Emotion/EMO-SUPERB [8] reduce ambiguity by publishing standardized speaker-independent partitions. |
| L57:C40 | 8 | `citation` | SERAB [7] and Open-Emotion/EMO-SUPERB [8] reduce ambiguity by publishing standardized speaker-independent partitions. |
| L59:C15 | 9 | `citation` | Meyer et al. [9] provide a related warning that dataset splits can expose non-emotion shortcuts. |
| L68:C19 | 256 | `identifier` | Deterministic SHA-256 ordering with seed 202608101744 was applied to 331 pending candidates from a 383-item deduplicated frame; screening ranks 1-34 yielded 30 eligible repositories and four exclusions. |
| L83:C39 | 10 | `citation` | RAVDESS contributed 1,440 utterances [10]. |
| L85:C39 | 11 | `citation` | CREMA-D contributed 7,442 utterances [11]. |
| L126:C51 | 95 | `confidence_level` | The observed positive fraction was 66.7\% (Wilson 95\% CI [48.8\%, 80.8\%]). |
| L129:C41 | 95 | `confidence_level` | The sensitivity rate was 83.3\% (Wilson 95\% CI [66.4\%, 92.7\%]). |
| L156:C67 | 95 | `confidence_level` | Corpus & Model & Random UAR & Grouped UAR & LOSO UAR & Delta UAR [95\% CI] \\ |
| L185:C100 | 5 | `citation` | The RAVDESS-CREMA-D pattern is directionally consistent with the closest exact-corpus prior study [5]. |
| L227:C8 | 2027 | `year` | ICASSP 2027 uses single-anonymous review. |
| L228:C81 | 2027 | `year` | The placeholder author and affiliation above must be replaced, and the official 2027 template must supersede this provisional layout before submission. |
| L12:C45 | 30 | `protocol_metadata` | A preregistered probability sample included 30 eligible repositories. |
| L16:C13 | 1,620 | `protocol_metadata` | We then ran 1,620 fits on two corpora with four architectures, three protocols, and three seeds. |
| L43:C68 | 30 | `protocol_metadata` | We contribute a preregistered audit whose eligible sample size was 30 public SER repositories. |
| L44:C32 | 1,620 | `protocol_metadata` | We add a controlled study with 1,620 valid fits. |
| L68:C42 | 202608101744 | `protocol_metadata` | Deterministic SHA-256 ordering with seed 202608101744 was applied to 331 pending candidates from a 383-item deduplicated frame; screening ranks 1-34 yielded 30 eligible repositories and four exclusions. |
| L68:C70 | 331 | `protocol_metadata` | Deterministic SHA-256 ordering with seed 202608101744 was applied to 331 pending candidates from a 383-item deduplicated frame; screening ranks 1-34 yielded 30 eligible repositories and four exclusions. |
| L68:C100 | 383 | `protocol_metadata` | Deterministic SHA-256 ordering with seed 202608101744 was applied to 331 pending candidates from a 383-item deduplicated frame; screening ranks 1-34 yielded 30 eligible repositories and four exclusions. |
| L68:C145 | 1 | `protocol_metadata` | Deterministic SHA-256 ordering with seed 202608101744 was applied to 331 pending candidates from a 383-item deduplicated frame; screening ranks 1-34 yielded 30 eligible repositories and four exclusions. |
| L68:C147 | 34 | `protocol_metadata` | Deterministic SHA-256 ordering with seed 202608101744 was applied to 331 pending candidates from a 383-item deduplicated frame; screening ranks 1-34 yielded 30 eligible repositories and four exclusions. |
| L68:C158 | 30 | `protocol_metadata` | Deterministic SHA-256 ordering with seed 202608101744 was applied to 331 pending candidates from a 383-item deduplicated frame; screening ranks 1-34 yielded 30 eligible repositories and four exclusions. |
| L87:C24 | 22.05 | `configuration` | Audio was converted to 22.05 kHz mono and trimmed at 30 dB. |
| L87:C54 | 30 | `protocol_metadata` | Audio was converted to 22.05 kHz mono and trimmed at 30 dB. |
| L88:C13 | 64 | `protocol_metadata` | Inputs were 64-bin log-mel features (FFT 1,024; hop 512; upper frequency 11,025 Hz; 128 frames) with per-utterance standardization. |
| L88:C42 | 1,024 | `protocol_metadata` | Inputs were 64-bin log-mel features (FFT 1,024; hop 512; upper frequency 11,025 Hz; 128 frames) with per-utterance standardization. |
| L88:C53 | 512 | `protocol_metadata` | Inputs were 64-bin log-mel features (FFT 1,024; hop 512; upper frequency 11,025 Hz; 128 frames) with per-utterance standardization. |
| L88:C74 | 11,025 | `configuration` | Inputs were 64-bin log-mel features (FFT 1,024; hop 512; upper frequency 11,025 Hz; 128 frames) with per-utterance standardization. |
| L88:C85 | 128 | `protocol_metadata` | Inputs were 64-bin log-mel features (FFT 1,024; hop 512; upper frequency 11,025 Hz; 128 frames) with per-utterance standardization. |
| L95:C23 | 64 | `protocol_metadata` | AdamW used batch size 64, at most 100 epochs, early stopping patience 15, and seeds 0, 1, and 2. |
| L95:C35 | 100 | `protocol_metadata` | AdamW used batch size 64, at most 100 epochs, early stopping patience 15, and seeds 0, 1, and 2. |
| L95:C71 | 15 | `protocol_metadata` | AdamW used batch size 64, at most 100 epochs, early stopping patience 15, and seeds 0, 1, and 2. |
| L95:C85 | 0 | `protocol_metadata` | AdamW used batch size 64, at most 100 epochs, early stopping patience 15, and seeds 0, 1, and 2. |
| L95:C88 | 1 | `protocol_metadata` | AdamW used batch size 64, at most 100 epochs, early stopping patience 15, and seeds 0, 1, and 2. |
| L95:C95 | 2 | `protocol_metadata` | AdamW used batch size 64, at most 100 epochs, early stopping patience 15, and seeds 0, 1, and 2. |
| L99:C45 | 42 | `protocol_metadata` | Random used five-fold StratifiedKFold (seed 42) without a speaker grouping constraint. |
| L101:C38 | 24 | `protocol_metadata` | LOSO held out one speaker at a time (24 RAVDESS folds; 91 CREMA-D folds). |
| L101:C56 | 91 | `protocol_metadata` | LOSO held out one speaker at a time (24 RAVDESS folds; 91 CREMA-D folds). |
| L105:C87 | 72 | `protocol_metadata` | Full out-of-fold predictions were retained for every corpus-model-protocol-seed cell: 72/72 complete cells and 1,620/1,620 successful fits (408 for RAVDESS; 1,212 for CREMA-D). |
| L105:C90 | 72 | `protocol_metadata` | Full out-of-fold predictions were retained for every corpus-model-protocol-seed cell: 72/72 complete cells and 1,620/1,620 successful fits (408 for RAVDESS; 1,212 for CREMA-D). |
| L105:C112 | 1,620 | `protocol_metadata` | Full out-of-fold predictions were retained for every corpus-model-protocol-seed cell: 72/72 complete cells and 1,620/1,620 successful fits (408 for RAVDESS; 1,212 for CREMA-D). |
| L105:C118 | 1,620 | `protocol_metadata` | Full out-of-fold predictions were retained for every corpus-model-protocol-seed cell: 72/72 complete cells and 1,620/1,620 successful fits (408 for RAVDESS; 1,212 for CREMA-D). |
| L105:C141 | 408 | `protocol_metadata` | Full out-of-fold predictions were retained for every corpus-model-protocol-seed cell: 72/72 complete cells and 1,620/1,620 successful fits (408 for RAVDESS; 1,212 for CREMA-D). |
| L105:C158 | 1,212 | `protocol_metadata` | Full out-of-fold predictions were retained for every corpus-model-protocol-seed cell: 72/72 complete cells and 1,620/1,620 successful fits (408 for RAVDESS; 1,212 for CREMA-D). |
| L111:C27 | 10,000 | `protocol_metadata` | Confidence intervals used 10,000 whole-speaker bootstrap replicates. |
| L123:C22 | 30 | `protocol_metadata` | The sample contained 30 repositories. |
| L153:C128 | 95 | `protocol_metadata` | \caption{UAR (\%) mean $\pm$ sample SD over three seeds. The final column is the speaker-paired Random-Grouped difference with 95\% cluster-bootstrap CI.} |
| L199:C65 | 30 | `protocol_metadata` | The audit covers one GitHub/arXiv-dominated candidate frame and 30 sampled repositories, with five unresolved. |
| L214:C22 | 30 | `protocol_metadata` | The sample contained 30 repositories, with five verified negatives and five unresolved cases. |
| L215:C17 | 1,620 | `protocol_metadata` | In a controlled 1,620-fit study, Random five-fold evaluation produced higher UAR than GroupKFold for all four probes on both corpora, but the premium was much larger on RAVDESS than CREMA-D. |
