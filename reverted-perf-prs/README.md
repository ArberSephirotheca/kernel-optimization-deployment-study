# Reverted performance PRs of vLLM and SGLang, and why each one was reverted

This folder lists every performance PR that vLLM and SGLang reverted or rolled back in 24 months
(2024-09-29 to 2026-09-28), and records for each one the reason of the revert in detail: what failed,
where, why, how it was found, and how it was resolved.

- [`RECORDS.md`](RECORDS.md) has one section for each incident.
- [`reasons.csv`](reasons.csv) has the same records as a table, one row for each incident.

## Short answer

- **127 incidents: vLLM 53, SGLang 74.** They come from 131 reverts of performance PRs, out of the 467
  reverts and rollbacks in the two repositories. An incident is one optimization: a resubmit that was
  reverted again, or two PRs that one revert undid, count once.
- **81 of the 127 (64%) had a real defect of the change.** The other 46 split as follows:
  - 21 have no stated mechanism;
  - 19 had no defect of the change: a false alarm, a review or design decision, or a planned
    replacement;
  - 6 were not faster, or the gain was a measurement artifact.
- **22 of the 81 defects were silent:** wrong output or lower accuracy, with no error. CUDA graph, compile
  and stream-order defects have the most silent failures (8 of 13). With 4 silent incidents of unknown
  mechanism, the 26 silent incidents take 39% of the days in main; see [Silent failures](#silent-failures).
- **Reverts are fast, but the failures are found after the merge.** The median time in main is 1.0 day.
  CI or a test after the merge found 56 failures, maintainers, reviewers or the author 39, and users
  20. In 6 incidents, a check was already red or cancelled at the merge.
- **71 of the 127 came back,** with a median of 7 days from the revert to the reland.

## Why they were reverted

| Kind of reason | vLLM | SGLang | Both | Silent | Median days in main |
| --- | ---: | ---: | ---: | ---: | ---: |
| a defect of the change | 36 | 45 | 81 | 22 | 1.0 |
| not known | 7 | 14 | 21 | 4 | 1.2 |
| no defect of the change (review or design) | 4 | 6 | 10 | 0 | 2.4 |
| no defect of the change (false alarm or test) | 2 | 6 | 8 | 0 | 0.8 |
| not faster, or a wrong measurement | 3 | 3 | 6 | 0 | 6.5 |
| no defect of the change (planned) | 1 | 0 | 1 | 0 | 38.2 |

| Cause | vLLM | SGLang | Both | Silent | Median days in main |
| --- | ---: | ---: | ---: | ---: | ---: |
| UNK: the mechanism is not stated and not found (the symptom may be stated) | 7 | 14 | 21 | 4 | 1.2 |
| BUILD: build, toolchain or platform, also a platform path that is not guarded | 5 | 10 | 15 | 1 | 0.3 |
| PRE: unenforced precondition (a shape, layout, buffer size, launch limit or supported configuration that the call site or the backend selection does not check) | 7 | 7 | 14 | 5 | 1.5 |
| GRAPH: CUDA graph capture, torch.compile or stream order | 7 | 6 | 13 | 8 | 1.4 |
| PROC: process or design (review, discussion, a better design) | 4 | 6 | 10 | 0 | 2.4 |
| DEP: dependency version or behavior | 4 | 5 | 9 | 0 | 0.9 |
| TEST: false alarm or test-only failure; no defect of the PR was shown | 2 | 6 | 8 | 0 | 0.8 |
| KERNEL: defect inside kernel code (also in a dependency's kernel) | 0 | 7 | 7 | 3 | 1.2 |
| API: caller and callee disagree | 1 | 6 | 7 | 0 | 0.5 |
| PERFREG: slower on another configuration, or a wrong measurement | 3 | 3 | 6 | 0 | 6.5 |
| NUM: numerical precision or semantics | 1 | 3 | 4 | 2 | 2.2 |
| MODEL: model-specific layout or behavior | 3 | 1 | 4 | 2 | 3.4 |
| LOGIC: logic defect in host or model code, not in a kernel | 4 | 0 | 4 | 1 | 0.6 |
| BUDGET: resource budget (memory, shared memory, start-up time) | 3 | 0 | 3 | 0 | 0.9 |
| SKEW: merge skew | 1 | 0 | 1 | 0 | 0.0 |
| NOTFAIL: planned rollback or replacement, not a failure | 1 | 0 | 1 | 0 | 38.2 |

**What the defects look like** (one or two examples each; RECORDS.md has all of them):
- **BUILD (15).** Platform code that runs or compiles on a platform it was not written for. Examples:
  - a 32-bit warp-shuffle mask on HIP's 64-wide wavefronts (vLLM #44173);
  - a gfx950-only builtin compiled for gfx90a (vLLM #52606);
  - a CUDA-only SM90 check run on AMD (SGLang #15712);
  - an AMD MoRI change that added a key to the shared two-batch-overlap state and broke DeepEP
    (SGLang #17953);
  - a missing CUDA arch flag for a new `.cu` file (vLLM #20762).
- **PRE (14).** A launch or a call that breaks a requirement of the kernel or of the backend, which
  nothing checks. Examples:
  - a launch grid y dimension above 65,535 (SGLang #16043);
  - FA3 scheduler metadata larger than its host buffer (vLLM #23465);
  - non-contiguous input to kernels that need contiguous input (vLLM #14384, SGLang #21734);
  - `BLOCK_SIZE_K = 256` with FP8 scale groups of 128 (SGLang #9087);
  - `num_token = 0` (SGLang #21766);
  - a batched embedding split by placeholder counts that some VLMs do not match (SGLang #25910);
  - a backend selection that picked a kernel without support for bidirectional attention (vLLM #48666).
- **GRAPH (13; 8 silent).** Examples:
  - a value of one step frozen into a captured CUDA graph (vLLM #50004);
  - a null pointer captured after a host-side `set_()` (SGLang #29142);
  - outputs without constant addresses across piecewise replays (vLLM #29558);
  - two streams that use the same storage with no event between them (vLLM #48223, #49236);
  - `cudaHostRegister` from a background thread during graph capture (vLLM #45850);
  - torch.compile tracing breaks (SGLang #18905).
- **DEP (9).** Features that the pinned dependency does not have yet (FlashInfer: vLLM #30746, #38050;
  SGLang #18801), Triton cache hints that the released Triton rejects (vLLM #15511), and a PyTorch flag
  that older versions do not have (vLLM #20836).
- **API (7).** Mostly the version skew between the `sglang` package and the separately released
  `sgl-kernel` package: tests and runtime code that call a kernel package that CI did not rebuild
  (SGLang #8364, #32994).
- **KERNEL (7).** Defects inside kernels, for example:
  - removed `tl.debug_barrier()` calls that let loads and stores of one program race (SGLang #11133,
    73 days in main);
  - a sentinel comparison in FlashInfer's MNNVL all-reduce that flush-to-zero breaks (SGLang #12787);
  - NaNs from a FlashMLA update (SGLang #15242).

**No defect of the change (19).**
- **False alarms (8).** CI failures that came from another PR, a flaky test, an unscaled test, a CI
  runner name, or a code-rule test. In three of them, the same code came back unchanged.
- **Review and design decisions (10).** PRs merged before a full review, a better design elsewhere, or a
  maintainer decision.
- **One planned replacement** (vLLM #32805).

**Not known (21).** Mostly SGLang reverts that say only "Reverts #N", with no later record that explains
them.

## Who found them, and what came back

| Found by | vLLM | SGLang | Both |
| --- | ---: | ---: | ---: |
| CI or a test after the merge | 23 | 33 | 56 |
| maintainers, reviewers or the author | 15 | 24 | 39 |
| users | 12 | 8 | 20 |
| unknown | 1 | 5 | 6 |
| a check that was red or cancelled before the merge | 2 | 4 | 6 |

| Symptom | Incidents | Silent |
| --- | ---: | ---: |
| CI failure | 18 | 0 |
| crash | 17 | 0 |
| wrong output | 16 | 16 |
| test failure | 15 | 0 |
| unknown | 14 | 0 |
| none (no failure) | 10 | 0 |
| slower | 8 | 0 |
| lower accuracy | 8 | 8 |
| hang | 7 | 0 |
| illegal memory access | 5 | 0 |
| build failure | 4 | 0 |
| NaN | 2 | 2 |
| more memory, compile failure, OOM | 1 each | 0 |

- **Time in main.** The median is 1.0 day (1.5 days for silent failures, 0.9 for loud ones), and the
  total is 635 days.
- **The silent failures that stayed longest:**

  | Incident | Days in main | Cause |
  | --- | ---: | --- |
  | SGLang #11133 | 73 | KERNEL |
  | vLLM #28271 | 32 | MODEL |
  | SGLang #9087 | 28 | PRE |
  | SGLang #15242 | 22 | KERNEL |
  | vLLM #50004 | 19 | GRAPH |
  | vLLM #49236 | 18 | GRAPH |
- **Relands.** 71 incidents came back:

  | How it came back | Incidents |
  | --- | ---: |
  | with a fix | 37 |
  | changed | 14 |
  | as part of another PR | 7 |
  | the same code | 4 |
  | split into smaller PRs | 3 |
  | the PR was re-applied (changes not checked) | 3 |
  | after a dependency release | 2 |
  | with a smaller test | 1 |

  50 did not come back. In 6 more, only a part was reverted, so no reland was needed.
- **The written reason.** 114 records quote a reason from the revert, the thread, a reland or an issue;
  92 of these quotes come from the revert itself.

## Silent failures

A silent failure gives wrong output, lower accuracy or NaNs, with no error. 26 incidents are silent: the 22
silent defects and 4 of the 21 incidents with no known mechanism (vLLM 16 of 53, SGLang 10 of 74).

- **They cost the most time.** They are 20% of the incidents, but 39% of the days in main (249 of 635).
  The six defects that stayed longest in main are all silent: SGLang #11133 (73 days), vLLM #28271 (32),
  SGLang #9087 (28), SGLang #15242 (22), vLLM #50004 (19) and vLLM #49236 (18). 10 of the 26 stayed more
  than 5 days, against 19 of the other 101 incidents.
- **Users found the long ones, and tests found the short ones.** A CI test found 10 of them, but only after
  the merge, with a median of 1.3 days in main; examples are multi-GPU tests, AMD CI, nightly jobs and
  LM-eval. Users found 8 (median 16.9 days), and the maintainers or the author 7 (median 4.0 days). In
  SGLang #11133, 40 of the 244 unit tests failed, but CI showed a pass, because the exit code of the test
  file did not show the failures.
- **Each one needs a specific configuration.** A test with another configuration passes. The configuration
  is one of these:
  - a model: Qwen2-VL with TP (vLLM #28271); Qwen3-Next, which uses the changed Qwen3.5 code (vLLM
    #34492); Bamba (vLLM #14778);
  - a platform: ROCm (vLLM #51692, SGLang #26235), the HIP path (SGLang #6220), TPU (vLLM #14995);
  - a parallel setting: DP with EP (vLLM #45309, #48223);
  - a batch state: concurrent long contexts (vLLM #50004), padded batches (SGLang #11133), prompts above
    2,048 tokens (vLLM #35216).

  In vLLM #23465, the kernel writes 1 + 4·round_up(B, 4) metadata slots into a host buffer of 4·B + 1
  slots. The two sizes are equal when B is a multiple of 4, so a test with B = 4 or 8 passes.
- **Half of them are capture, stream or precondition defects:** 8 graph capture and stream-order defects
  and 5 unchecked preconditions. 13 of the 26 come from system optimizations, not from kernel changes, so a
  check of kernel code alone is not enough.
- **The fix is small once the mechanism is known:** make the input contiguous (vLLM #14384), correct the
  buffer size (vLLM #23465), launch the shared expert first (SGLang #29142), or multiply by the factor
  again on the non-CUDA paths (SGLang #6220). 14 of the 26 came back, 9 of them with a fix. The difficult
  part is to find the failure before the merge.

| Group | Incidents | Days in main (sum) | Median days | Found by users | Found by a CI test after the merge | What can find it before the merge (assessment) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Graph capture and stream order | 8 | 53.2 | 1.3 | 1 | 5 | a capture-stability and stream-order check (not built yet); the CI tests before the merge |
| A kernel used outside its requirements (PRE) | 5 | 42.2 | 5.7 | 1 | 2 | call-site contracts: buffer size, layout, tile size, supported features |
| A defect inside a kernel | 3 | 96.4 | 22.1 | 1 | 1 | the warpDRF race check; a comparison with the upstream kernel |
| Model layout or platform path | 4 | 33.4 | 0.6 | 3 | 0 | a test of each model and platform path that uses the changed code |
| Lower precision | 2 | 7.0 | 3.5 | 0 | 1 | an accuracy test with an error budget |
| Mechanism not known | 4 | 16.3 | 0.7 | 2 | 1 | not known |
| all | 26 | 248.6 | 1.5 | 8 | 10 | |

| Incident | Optimization | What went wrong | Days | Found by | Before the merge (assessment) |
| --- | --- | --- | ---: | --- | --- |
| **Graph capture and stream order** | | | | | |
| vLLM #50004 | adaptive top-k width (DeepSeek-V4) | a writer outside the graph changed the row stride, but the captured reader kept the old stride | 19.2 | users | capture stability: the writer and the captured reader agree on the layout |
| vLLM #49236 | one scratch pool for all layers | a producer wrote a buffer that a reader on another stream still used | 18.4 | the maintainers | stream order: no event between the two uses |
| vLLM #29558 | piecewise CUDA graphs for DeepEP HT | the outputs did not keep the same addresses across replays | 11.1 | H200 CI, after the merge | capture stability: fixed addresses; the H200 tests |
| SGLang #14044 | one memory pool for two graph runners | accuracy 0; the shared pool or the pool id (not confirmed) | 1.4 | CI, after the merge | the piecewise CUDA graph test |
| vLLM #48223 | dual-stream decode (ROCm) | two streams used the same tensor with no copy | 1.1 | AMD CI, after the merge | stream order: aliased tensors |
| SGLang #19190 | fewer CPU-GPU sync points | pinned non-blocking copies doubled the KL divergence (the mechanism is not stated) | 1.1 | the KL test, after the merge | the KL test |
| SGLang #29142 | new stream order in the DeepSeek MoE | a host-side `set_()` made the captured GEMM read address 0 | 0.6 | CI, after the merge | capture stability: a host change during the capture |
| vLLM #45309 | fewer eager breaks in the capture (DeepSeek-V4) | probably a per-step value that stays fixed in the graph (not confirmed) | 0.3 | a maintainer's bisect | capture stability (if the cause is correct) |
| **A kernel used outside its requirements (PRE)** | | | | | |
| SGLang #9087 | fused-MoE tuning config (B200) | a K tile of 256 covers two FP8 scale groups of 128 (probable) | 27.8 | a user | a launch rule: BLOCK_SIZE_K is not larger than the scale group (128) |
| vLLM #51692 | pre-shuffled FP8 GEMM from AITER (ROCm) | the kernel read row-major scales as column-major; some weights were shuffled twice | 6.1 | AMD engineers | a layout contract at the call site |
| vLLM #48666 | FA4 FP8-KV path (Gemma-4) | the backend selection chose a kernel without bidirectional attention | 5.7 | the nightly performance test | a feature check in the backend selection |
| vLLM #23465 | FA3 swizzle update | the kernel wrote past the metadata buffer when B is not a multiple of 4 | 1.5 | 2xH100 CI, after the merge | a bounds contract with a symbolic B |
| vLLM #14384 | less MLA CPU overhead | strided input to an in-place kernel that needs contiguous input | 1.1 | a maintainer | a contiguity contract at the call site |
| **A defect inside a kernel** | | | | | |
| SGLang #11133 | MTP topk > 1 (Qwen3-Next); it also removed three `tl.debug_barrier()` calls | loads and stores of one program race | 73.3 | a user (CI showed a pass) | the warpDRF race check (finds it with barrier.orders_global); a CI that reports failed tests |
| SGLang #15242 | FP8 sparse update of FlashMLA | NaNs with the NVFP4 checkpoint | 22.1 | the author | a comparison with the upstream FlashMLA output |
| SGLang #26235 | no softmax in the EAGLE draft (topk 1) | on AMD, an argmax kernel gave an index outside the vocabulary (not certain) | 1.0 | the AMD nightly test | the AMD accuracy test |
| **Model layout or platform path** | | | | | |
| vLLM #28271 | no TP gather in Qwen-VL | the Qwen2-VL qkv layout needs the gather under TP | 31.9 | a user | a TP 2 against TP 1 comparison for each model |
| vLLM #34492 | fused GDN projections (Qwen3.5) | Qwen3-Next uses the same code, and its split sizes became wrong | 0.8 | a user | a test of each model that uses the code |
| SGLang #6220 | routed scaling factor fused into a CUDA kernel | the non-CUDA paths lost the factor | 0.4 | unknown | a HIP accuracy test |
| vLLM #14778 | fewer copies in the Mamba2 prefill | `copy_` wrote into a temporary from tensor indexing, not into the cache | 0.3 | users | a Mamba2 accuracy test; a lint rule for this pattern |
| **Lower precision** | | | | | |
| SGLang #28166 | GELU in the cuBLASLt GEMM epilogue (FLUX) | the images moved away from the unfused reference | 4.0 | the maintainer | an image comparison with the unfused path |
| vLLM #34582 | NemotronH router not in fp32 | the accuracy fell by 4% to 5% | 3.0 | LM-eval CI, after the merge | the LM-eval test |
| **Mechanism not known** | | | | | |
| vLLM #37205 | router GEMM kernel (gpt-oss) | rare NaNs | 14.6 | a user's internal test | not known |
| vLLM #14995 | TPU bucket padding | wrong output with a padding gap of 64 | 1.0 | the maintainers | a test that the padding does not change the output |
| vLLM #35216 | compile passes made once at init | wrong logits above about 2,048 tokens at -O3 | 0.4 | a user's bisect | a long-prompt test at -O3 |
| SGLang #33623 | MLA gate fused into the QKV-A GEMM (Kimi-K3) | the output did not stop before 2,048 tokens | 0.3 | CI, after the merge | the CI test |

**What could find them before the merge.** This is an assessment from the records, not a result: no check
ran, except warpDRF on SGLang #11133. [`silent.py`](silent.py) has the assessment of each incident.
1. **Run the CI tests before the merge, on the hardware that they need.** A CI test found 10 of the 26,
   but only after the merge. CI must also fail when a test fails (SGLang #11133).
2. **Add formal checks where a test needs a rare state.** They target 12 of the 26: a capture-stability
   check 4 (one of them probable), a stream-order check 2, call-site contracts 5, and the warpDRF race
   check 1. Examples of a rare state are a batch size (vLLM #23465), stream timing (vLLM #49236, #48223)
   and a value that changes after the capture (vLLM #50004). The capture-stability and stream-order check
   does not exist yet.
3. **Compare with a reference on more configurations.** Items 1 and 2 together cover 17 of the 26. The
   other 9 need a comparison on each model, platform path or input that uses the changed code (vLLM
   #28271, #34492, #14778; SGLang #6220, #15242, #28166), or their mechanism is not known (vLLM #37205,
   #14995, #35216).

## How the list was built

1. **The reverts.** The source is the deep study's B1 census (`deep-study/data/confirmed-reverts.csv`).
   It holds every first-parent commit of the window whose subject uses revert, rollback, back-out, undo,
   reapply or reland language, adjudicated in context. Of its rows, 467 are reverts, partial reverts
   or explicit rollbacks: vLLM 191 and SGLang 276. They name 441 distinct target PRs.
2. **Which targets are performance PRs.** The construct is the deep study's codebook A1: a PR whose
   primary stated purpose is to make the system faster or more resource-efficient. This includes:
   - kernels, fusions and tuning configs;
   - faster backends and defaults;
   - low-precision paths for efficiency;
   - system optimizations: the scheduler, start-up, memory, communication, CUDA graphs;
   - fixes of performance regressions.

   Bug fixes, features, model support, hardware enablement without an efficiency claim, refactors, CI
   and dependency bumps are not performance PRs.

   A1 classified only the PRs created in its own window, and it decided most of them by title and path
   rules. Thus every reverted target that no A1 coder had read was read here, with its title, labels
   and full body: 361 targets. The 46 rollbacks that name no PR were read too; a target was found for
   14 of them, by reading, a commit lookup or a search. The A1 verdicts that a coder made (75) or that
   an explicit performance tag decided (5) were kept, except 5 coder verdicts that the full body
   contradicts. All decisions are in
   [`classification.py`](classification.py), with a verbatim quote for each performance verdict.
3. **The catalog.** [`catalog.py`](catalog.py) joins the census, A1 and the decisions. It stops if any
   target has no decision. The result is 131 reverts of performance PRs in 127 incidents.
4. **The records.** For each incident, [`evidence.py`](evidence.py) builds an evidence sheet from the
   downloaded records. A sheet has:
   - the target PR;
   - each revert, with its human discussion;
   - the comments on the target after its merge;
   - the issues and PRs that cross-reference the target or the revert (failure reports, relands,
     fixes).

   Each record in [`records.py`](records.py) follows the chain of evidence to the fix, the reland or
   the dependency fix. For example, the FlashInfer fix #3304 explains the SGLang #12787 hang, and the
   reland #29463 explains SGLang #29142. When only a diff that the earlier study read supports a
   mechanism, the record says "the earlier study's reading".
5. **The reports.** [`build.py`](build.py) writes RECORDS.md and reasons.csv, and prints the tables
   above. The tables of the silent failures also use [`silent.py`](silent.py): the group of each cause,
   and a short note and an assessment for each silent incident.

## Relation to the earlier study (`../perf-revert-study`)

The earlier study took the census rows whose reverted change the census coded as `kernel_performance`.
This folder uses the A1 definition of a performance PR instead.
- **85 incidents are in both.**
- **42 are new.** Most of them are system optimizations that the census coded "other" or
  "hardware_backend", for example:
  - multi-process tokenizers (SGLang #6555);
  - NPU graphs (SGLang #8027);
  - start-up parallelism (vLLM #40331);
  - garbage-collection overhead (vLLM #26368);
  - KV-cache layouts for PD (vLLM #30207).
- **13 of the earlier incidents are not here:**
  - 12 PRs whose purpose was not performance: 6 dependency updates and migrations, 3 refactors, a
    feature, a default switch with no efficiency claim, and an accuracy fix;
  - one rollback with no performance target (r31502).

The reading of the full evidence changed the code of 6 incidents of the earlier study, and the silent
flag of 3:

| Incident | Earlier | Now | Why |
| --- | --- | --- | --- |
| SGLang #12787 | GRAPH | KERNEL | FlashInfer #3304 fixed a sentinel comparison in the MNNVL all-reduce kernel; piecewise graphs only made the hang more frequent |
| SGLang #30835 | GRAPH | TEST | a CPU unit-test job failed; no product defect was shown |
| SGLang #26235 | NUM | KERNEL (low confidence) | the texts do not support argmax tie-breaking; the ROCm reland cites an AITER greedy_sample bug that returns an index outside the vocabulary |
| SGLang #36752 | GRAPH | UNK | no mechanism is stated |
| vLLM #15511 | BUILD | DEP | the change needed an unreleased Triton |
| vLLM #48641 | NUM | UNK | the analyzer's explanation is a guess, and a maintainer doubted it |
| SGLang #29142 | not silent | silent | the decode graph read a null pointer and gave garbage, not an error |
| SGLang #3613 | silent | not silent | the reports are exceptions in CUDA graph capture |
| SGLang #37376 | silent | not silent | a speed test failed on the acceptance length; the outputs were not wrong |

The earlier study's coding and README now use these corrected codes, and its README notes the change.

## Limits

- **One coder.** One coder, with AI assistance, wrote the records. 73 records have high confidence, 26
  medium and 28 low.
- **Mechanisms.** A record states a mechanism only when the revert, the discussion, a reland, a fix or
  an issue states it. Some discussions took place on Slack, and some CI logs have expired. 21
  mechanisms are unknown.
- **The census window and its words.** A rollback without revert words in its commit subject is not in
  the census.
- **The purpose decides the scope.** A non-performance PR that was reverted for performance reasons is
  out of scope, for example the zero-init correctness fix of vLLM #38148. vLLM #16158 rolls back a V1
  default of `max_num_seqs`, but the PR that set that default was not found, so it is out of the list.

## How to rebuild

```
python3 -I reverted-perf-prs/fetch.py records [--repo vllm|sglang]   # PR records of all targets and reverts
python3 -I reverted-perf-prs/catalog.py --csv                        # the catalog (catalog.csv)
python3 -I reverted-perf-prs/fetch.py detail [--repo vllm|sglang]    # comments, timelines, relands, evidence
python3 -I reverted-perf-prs/evidence.py OUT_DIR                     # evidence sheets for coding
python3 -I reverted-perf-prs/build.py                                # RECORDS.md, reasons.csv and the tables
```

The downloaded records are in `data/` (gitignored); the scripts only parse them as JSON.
