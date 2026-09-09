# Final scientific consistency review

**Result: no unresolved scientific consistency issue identified within the scope below.** This was a read-only review of the revised manuscript, archived numerical tables, figure-generation record, and relevant protocol statements. It did not rerun training, authenticate the entire controlled archive again, or establish generalizability beyond the reported design.

The reviewed [manuscript source](../english/main.tex) has SHA-256 `043ebfaddb925910b74005fff77b61192d50b5d798931e09fa2512fff6b957a7`. Subsequent source changes require a corresponding review update. PDF layout, ethics, funding, author declarations, and publication-link availability are checked separately.

## Primary results and inference

All **16 primary numerical rows** are unchanged, including their displayed means, interval endpoints, raw p values, and Holm p values, from the [preceding supplement manuscript](../../supplement-results-20260908/english/main.tex). The original six-test family and supplemental ten-test family remain distinct. The supplemental result file confirms **six of ten** Holm rejections. Uncertain RAVDESS interactions and both SUBESCO window effects remain included; the RAVDESS interaction-window effect is correctly described as not surviving Holm10 despite its pointwise interval excluding zero.

The manuscript retains five-fold averaging within each of 24 draws, followed by draw-level inference with 23 degrees of freedom. It does not treat 120 folds or 1,104 fits as independent participants. Absolute scores, UAR contrasts, paired backbone descriptions, and earlier group-exchange analyses are not promoted to additional confirmatory tests. Different native tasks are not pooled into a language-effect claim.

## Figure and selection semantics

The [window figure and its audit](../figures/window_selection_audit.json) use the complete archived WavLM 45-epoch trajectories for SUBESCO and RAVDESS. Selection uses each role's own validation CE or UAR, with the earliest tied epoch; outer UAR is consulted only after selection. Each plotted point first averages five paired fold differences per draw and then 24 draw means. Shading uses pointwise t intervals, not simultaneous bands. The complete W1–45 scan is expressly post hoc and adds no hypothesis tests.

The generation record checks all 1,920 W15/W45 selections, their outer values, 960 unit differences, 192 draw values, and 24 summary/interval values against the archived scoring output. The maximum numerical difference is approximately `8.9e-15` pp. The figure caption correctly identifies W15 as the prefix of the current long trajectories, rather than the older independently executed study. It distinguishes eligible checkpoints from training termination. The text describes observed window profiles without claiming proven saturation, a threshold, or a causal mechanism.

The final source now correctly limits per-epoch prediction retention to main A-trained fits and states that the original B-trained controls retain only epoch 15. The distinction between whole-role native-class UAR and mean speaker UAR is explicit. Fixed role batching and the unmasked mean-pooling limitation remain disclosed.

## Absolute results and interpretation

Independent projection of the [supplement descriptive aggregates](../../../docs/autodl-supplement-20260908/results/results.json) confirms the following manuscript statements:

| Statement | Check |
|---|---|
| UAR selection exceeds CE for both validation roles in five of eight new settings | Confirmed; these are descriptive mean rankings |
| Opposite rankings across roles | SUBESCO WavLM15 and RAVDESS HuBERT15 |
| CE exceeds UAR for both roles in the remaining setting | RAVDESS WavLM15 |
| New four-policy outer-UAR range | 33.3246528–58.7826504%, displayed as 33.325–58.783% |
| New UAR exposure-contrast range | 0.1109477–1.1111111 pp, displayed as 0.111–1.111 pp |
| CREMA-D HuBERT fixed last | 56.446%, between seen CE 57.002% and unseen CE 55.426% |
| CREMA-D WavLM fixed last | 53.240%, below seen CE 55.121% and unseen CE 54.103% |

The [complete absolute-results page](../ABSOLUTE_RESULTS.md) retains all 11 original/supplement settings, all four selection policies, the fixed-last baseline, and UAR contrasts. Its last-minus-unseen comparisons are marked post hoc and descriptive, with no added p values. The manuscript does not turn a larger exposure contrast into a claim that a policy is universally best.

The DUAL discussion correctly calls the fixed-last-epoch comparison descriptive and separates its validation-optimism estimand from selected-model outer-UAR contrasts. It does not attribute the entire earlier optimism gap to selection. The limitations also retain different development identities/recordings, reused corpora, the incomplete horizon matrix, fixed hyperparameters, and the absence of isolated voiceprint or language causality.

## Artifact and release scope

The manuscript distinguishes public tables, curves, scientific code and verification summaries from the controlled supplemental logits and retained weights. It describes the original 384-fit prediction replay bundle separately. Public table replay is not presented as equivalent to the full controlled-archive audit or as independent retraining. The quoted 199,780 comparisons and approximately `7.1e-15` maximum difference agree with the [existing numerical audit](../../../docs/autodl-supplement-20260908/evidence/independent_replay.json); that audit was inspected, not re-executed for this review.

Private pre-execution source/plan freezes are explicitly distinguished from public preregistration. The scientific programs retain their separate identities. This report does not assert that the planned `v2026.09.09` release is already reachable: **publication link checked separately at release**.

## Reviewed supporting identities

| File | SHA-256 |
|---|---|
| Previous supplement manuscript | `8c8d28974e6facc46f6d5a08375360ba455bb64e842c309839c119ccb5eb3ae8` |
| Complete absolute-results page | `59f0c878a1c2c3d4803e18441201610c74722aeea56b271038ef451f90e79ce1` |
| Window-selection PDF | `c35f78f7c294803c9d72f64856beb515d7eea5c0dd75461ee40204052223d53a` |
| Supplemental aggregate results | `bf6e41c9ef39465c566da737add9e2cf0c2b4e707e751fa42f3fe2991131630f` |

These identities identify the material read during this review; they are not a substitute for the separate source-freeze, archive, numerical-replay, or release-publication checks.
