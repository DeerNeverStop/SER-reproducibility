# Prior exposure and prospective identity

Study: `SER26-STUDY2-CORE-1`, planned on 2026-09-05 after prior SER results.

The researchers have already seen the v2 manuscript and its result summaries,
including N05/N10 prompt/speaker mechanisms, dataset-size descriptors, and the
Claude review in PR #7. The underlying public corpus was used in the earlier
study. None is an untouched new population, and none of these earlier results
is reclassified as a newly collected Study II observation.

Historical source: v2 runtime at `fb28c28004fa4f33322736995a63651775dceecb`;
planning documents at `6b6e1a3a6d5fff7d7c12bb7a24b200e79964780d`;
Claude review at `8edac2d61be04ed71e141cbdde7bde26d2e27ff9`.

These sources motivated a fixed-budget speaker/within-speaker coverage contrast,
sentence rotations, and reporting uncertainty. The 3 percentage-point practical
reference is a planning constant, not a value inferred from desired power.
The study is a prospectively specified estimation extension, not an expansion
of v2's confirmatory hypothesis registry. v2 and N14R records are unchanged.

The operational CPU pilot stores selection predictions without computing or
viewing UAR/rankings. It is used solely to check data identity, execution and
timing. Core design is independent of its scientific predictions. Final core
outcomes remain hidden until all planned units and integrity checks close.

The CREMA-D demographic source was retrieved read-only from the corpus authors:
[VideoDemographics.csv](https://github.com/CheyneyComputerScience/CREMA-D/blob/master/VideoDemographics.csv),
Git blob `94a2db22d3b359d3ad90d936640d4cff1ade5b91`. Only ActorID and the
provided Sex category are used to construct balanced strata. The categories
are dataset metadata, not inferred identities or a claim about deployment
demographics. File bytes and source/runtime identities are pinned by the plan.

Publishing the plan and code to GitHub before core training provides an auditable
version history. It is not an OSF registration or an independent preregistration
review, and it does not erase historical data exposure.
