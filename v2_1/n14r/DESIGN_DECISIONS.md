# N14R final design decisions before execution

The review comment posted on PR #6 before implementation proposed at least ten
outer draws, at least two training seeds per fold, a primary exact sign-flip
test, and retention of the historical two-member `N-supp` family.  That comment
was a request for a prospective repair, not a frozen registration.  The final
contract changes three details after methods review and before any N14R fit:

1. **Twenty-four independent outer draws and one paired training stream per
   draw/fold.**  Split draw and training RNG are jointly sampled pipeline
   randomness.  With a fixed fit budget, additional outer draws add independent
   inferential units whereas repeated seeds within the same draw do not.  The
   final design has 24 independently seeded draws and 1,920 primary planned
   unit identities before retries, compared with the review comment's minimum
   design of 10 draws, two seeds, and 1,600 planned unit identities.
   Both cells and all configurations share the draw/fold training seed as a
   paired common random number.
2. **A two-sided one-sample Student t test is primary.**  The estimand is a mean
   draw effect.  A sign-flip randomization distribution is exact only under a
   sign-exchangeability/symmetry assumption, not under an unrestricted
   mean-zero null.  Complete `2^24` sign-flip enumeration remains frozen as a
   sensitivity analysis, alongside a 100,000-resample draw bootstrap.
3. **N14R is a new one-member corrective family.**  Retrofitting a newly
   generated endpoint into the already observed historical `N-supp` family
   would mix prospective and completed analyses.  N14R is therefore registered
   as its own family of size one.  No additional confirmatory HPO endpoint may
   be added after the preregistration tag.

These decisions were made using only the disclosed historical pilot values and
code/contract review.  The pilot is excluded from all N14R estimates and tests.
The preregistration commit/tag and a server-timestamped GitHub comment must exist
before the first N14R training process starts.
