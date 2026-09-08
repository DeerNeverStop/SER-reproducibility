# N14R frozen execution runbook

All commands are run from the repository root. The sole authorized interpreter
is `E:\科研\SER\ser_gpu\Scripts\python.exe`. No training command may be issued
until steps 1--5, plus the tagged-checkout and PINS-verification portion of
step 6, have completed and the public pre-execution receipt exists.

1. Run the unchanged-v2 and N14R test suites:

   ```powershell
   & 'E:\科研\SER\ser_gpu\Scripts\python.exe' -m pytest v2\tests v2_1\n14r\tests -q
   ```

2. Generate the plan from the pinned raw manifest. This applies the frozen
   H1/H2/H3 hygiene transform and must report 7,442 input rows, 7,435 analysis
   rows, 2,240 unique planned units, and 56 split files:

   ```powershell
   & 'E:\科研\SER\ser_gpu\Scripts\python.exe' -m v2_1.n14r.plan --manifest v2\manifests\cremad_manifest.csv --out v2_1\n14r\plan
   ```

3. Run all tests again, then create and immediately verify `PINS.json` against
   the actual feature directory:

   ```powershell
   & 'E:\科研\SER\ser_gpu\Scripts\python.exe' -m pytest v2\tests v2_1\n14r\tests -q
   & 'E:\科研\SER\ser_gpu\Scripts\python.exe' -m v2_1.n14r.make_pins --repo . --features v2\features --out v2_1\n14r\PINS.json
   & 'E:\科研\SER\ser_gpu\Scripts\python.exe' -m v2_1.n14r.make_pins --repo . --features v2\features --out v2_1\n14r\PINS.json --verify
   ```

4. After PINS creation, do not change any pinned file. Commit the exact tree,
   create annotated tag `SER26-N14R-prereg-1`, push the commit and tag, and
   verify the remote tag resolves to that commit.

5. Before training, post a server-timestamped public receipt on PR #6 naming
   the exact commit, tag, PINS content-manifest hash, plan hashes, 24 primary +
   4 reserve draws, 1,920 primary and 2,240 maximum unique planned units, and
   the final design deviations recorded in `DESIGN_DECISIONS.md`.

6. On the execution checkout, fast-forward to the tagged commit and repeat the
   PINS verification command. Then run the runner; it independently rechecks
   all pins, package versions, GPU identity, feature contents, hygienic
   population, and split coverage before fitting:

   ```powershell
   & 'E:\科研\SER\ser_gpu\Scripts\python.exe' -m v2_1.n14r.run --plan v2_1\n14r\plan --manifests v2\manifests --features v2\features --run v2_1\n14r\runs\n14r --device cuda
   ```

7. Scoring is forbidden until the runner writes `completion.json` and
   `analysis_lock.json`. After complete closure, score once and independently
   verify every raw unit and every inferential value:

   ```powershell
   & 'E:\科研\SER\ser_gpu\Scripts\python.exe' -m v2_1.n14r.score --plan v2_1\n14r\plan --run v2_1\n14r\runs\n14r --manifest v2\manifests\cremad_manifest.csv --output v2_1\n14r\results\n14r_results.json
   & 'E:\科研\SER\ser_gpu\Scripts\python.exe' -m v2_1.n14r.verify --plan v2_1\n14r\plan --run v2_1\n14r\runs\n14r --result v2_1\n14r\results\n14r_results.json --spec v2_1\n14r\spec.json --output v2_1\n14r\results\n14r_verification.json
   ```

The 125 GPU-hour value is a soft planning allocation, not an execution cap.
The fixed limit is two attempts per unique unit and at most 28 activated draws;
fewer than 24 complete draws after reserves yields `not tested (incomplete)`.
