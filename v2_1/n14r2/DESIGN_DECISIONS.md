# N14R2 decisions made before formal execution

N14R2 is the user's requested full rerun on four separate RTX 4090 pods, with
no reuse of any N14R completed fit or draw. Its code lives in a new namespace;
the old N14R preregistration and incomplete-run ledger remain immutable.

1. **New study and new randomization, retained scientific question.**
   `N14R2` / `SER26-N14R2` and contract version `2.0.0` distinguish this execution
   from N14R. There are 24 primary plus four reserve outer draws, five folds,
   two cells, eight configurations, and one paired training replicate. The
   bootstrap seed is `202609050001`. Reusing the corpus, hygiene, fixed feature
   cache, and model source does not reuse a fitted outcome. All 16 complete
   N14R draws and all 1,298 successful fits are excluded, including the 18
   successes belonging to an incomplete draw.
2. **Four-node task parallelism, not a change to the unit estimator.**
   `draw_id % 4` fixes the node before results. Eight independent spawned
   persistent processes share each node's one GPU; a unit resets its random
   streams and trains from initialization. All paired cells/configurations
   stay on the draw's assigned node. No DDP, AMP, shortened epoch limit,
   reduced grid, or changed early stopping is introduced for speed. CPU thread
   count one and the Linux RTX 4090 package/runtime receipts are prospectively
   bound. The synthetic benchmark informed only operational capacity; it did
   not measure formal convergence or justify outcome-based design changes.
3. **Separate infrastructure availability from scientific missingness.**
   N14R's frozen failure loop exhausted attempts after a CUDA-context incident.
   N14R2 instead circuit-breaks on infrastructure/unknown failures, preserves
   every dispatched attempt, and requires a fresh healthy process plus global
   authorization before an eligible retry. Maximum two charged attempts per
   unit still applies. Exhausted infrastructure or mixed failures remain
   blocked; they are not silently converted into missing draws or reserves.
   Only two pre-allowlisted training failures can void a draw under this
   contract. Any expanded recovery rule requires an explicit prospective
   amendment before outcome inspection.
4. **Global reserves and immutable provenance.**
   All primaries must be complete or validly training-void before the first
   reserve is considered. A global single-writer authorization activates at
   most one reserve at a time in ascending order, only as needed for 24 full
   draws. A local node's speed or failure cannot consume the reserve pool.
   Separate immutable attempt directories prevent successful retries from
   overwriting failed-attempt evidence. A conservative dispatch boundary means
   an uncertain-start attempt still consumes an attempt, whereas pre-dispatch
   setup/health failure does not.
5. **Resource limits pause operations; they do not select a result.**
   The USD 25 agent guard requests direction without turning a partial study
   into a completed scientific test. Runtime/cost estimates are not sample-size
   guarantees or scientific stopping rules. A pause/blocked state has no p-value.
6. **Inference and audit remain draw-level and prospective.**
   The sole primary family member uses the two-sided Student t test over 24
   complete draws. Positive mean plus two-sided adjusted p <= .05 is required
   for a directional claim. Bootstrap and sign-flip sensitivity assumptions
   are explicit. `D09R2` is descriptive only. The equal draw/fold weighting,
   eight-way validation selection, minimum-loss checkpoint, and tie-breaking
   are unchanged. No outcome parsing precedes the 24-draw analysis lock.

The t-test is a prespecified approximation over random pipeline draws from a
fixed corpus, not a distribution-free guarantee. Four hosts can have different
drivers and CPU throughput; same-GPU pairing and pinned core packages reduce
confounding but do not establish numerical identity across nodes. This
deployment and the prior incomplete study must be disclosed in the report.

The preregistration commit/tag, full pin inventory, four environment receipts,
and server-timestamped public receipt must all precede the first formal dispatch.
