# Reverted kernel-performance PRs in vLLM and SGLang

Why do vLLM and SGLang revert kernel-performance PRs, and how long does each revert
cost? Which pre-merge check would have stopped each one? This directory codes the
root cause of every kernel-performance revert in the deep study's revert census.
It measures merge → revert → reland from GitHub, and it asks how many of these
reverts a kernel verifier such as warpDRF would have prevented.

**Note (2026-10-14).** [`../reverted-perf-prs`](../reverted-perf-prs/README.md) replaces
this list. It uses the codebook A1 definition of a performance PR (127 incidents; 85
of the 98 here are in it), and it reads the full evidence of each incident. That
reading changed the cause of six incidents here (SGLang #12787, #30835, #26235 and
#36752; vLLM #15511 and #48641) and the silent flag of three (SGLang #29142, #3613
and #37376). The tables and the text below use the corrected codes. The other
fields, such as how each failure was found, keep this study's reading.

**Short answer.** The study covers 98 failure incidents: 34 in vLLM and 64 in SGLang.

- **Causes.** The largest groups are these:
  - build and platform breaks: 13;
  - dependency versions: 12;
  - unenforced preconditions: 11;
  - CUDA graph, torch.compile or stream-ordering defects: 9;
  - causes that nobody stated: 15.

  SGLang adds two groups that vLLM does not have. Defects inside kernel code: 8.
  Caller/callee mismatches, often between `sglang` and the separately released
  `sgl-kernel` package: 6.
- **The largest lever is when the existing CI runs.** A CI lane or test that
  already existed found 51 of the 98 (vLLM 18, SGLang 33), but only after the
  merge, or the PR merged with the check red or cancelled. In vLLM these lanes are
  mostly special hardware (H100/H200, B200, AMD). In SGLang they are mostly the
  normal PR CI and unit tests.
- **Preconditions are the most expensive class.** They cost the most rework: 424
  days from revert to reland in total. Silent failures stay in main longest. The
  longest was 73 days: SGLang #11133 removed `tl.debug_barrier()` calls from a
  Triton kernel while the CI passed.
- **warpDRF can address 2 of the 98 directly.** Both are in SGLang and both are
  Triton. warpDRF was run on both (2026-10-13; see "Where warpDRF fits"):
  - #16043, a launch grid above the 65,535 limit: found, but a filter hides it;
  - #11133, the removed barriers: found in one of the two kernels, and only with a
    guarantee that warpDRF withholds by default. The other kernel exceeds the
    checker's limits.

  Seven more are in its class (bounds, launch conditions, host-side
  preconditions), but in CUDA, HIP or library kernels.

## Scope

The B1 revert census (`deep-study/data/confirmed-reverts.csv`, 24 months to October
2026) adjudicates every first-parent commit whose subject uses revert, rollback,
back-out, undo, reapply or reland words. The study takes every revert or rollback
row whose reverted change is coded `kernel_performance`.

| | vLLM | SGLang |
| --- | ---: | ---: |
| Revert and rollback rows in the census | 36 | 66 |
| Reverted PRs that the census names | 36 | 57 |
| Rows that name no PR (target found by reading) | 1 (0) | 9 (7) |
| Coded rows (one per reverted PR or per rollback without one) | 37 | 66 |
| Planned rollbacks (not failures) | 2 (#25508, #35475) | 1 (#3230) |
| Failure incidents | 34 | 64 |

Two vLLM PRs (#12355, #12435) were reverted together, so they are one incident.
One SGLang optimization was reverted twice (#8364, then its resubmit #8770); it is
also one incident.

**Why about 100, not hundreds.** Hundreds is the count of all reverts: vLLM and
SGLang together have 467 reverts and rollbacks in the window. Most performance PRs
are not reverted. Of the vLLM performance PRs created and merged in the 12 months to
September 2026, 23 of 1,277 were reverted (1.8%). In SGLang it was 34 of 1,423
(2.4%) (`deep-study/data/a2-revert-rates.csv`). Most kernel defects are fixed forward
instead. The deep study estimates 1,787 kernel-correctness fixes in the two
repositories in 24 months (B2).

**What the census class misses.** The class misses a few vLLM performance PRs:
- **Two kernel-performance PRs in other classes.** #12591 is coded
  `hardware_backend`. #51692 shares a revert row with a bug fix and is coded
  `kernel_correctness`.
- **Six performance PRs outside kernels.** These are #14778, #20000, #22628,
  #26368, #33641 and #52764.

A performance change that a later PR turns off without a revert word ("disable by
default", "fall back") is not in the census. That undercount is not measured.

## Method

For each row, the record that was read:
- the revert PR body and discussion;
- the original PR's comments after its merge;
- the linked issues;
- the diff between the original and its reland. Where it shows the real fix, this
  is the strongest evidence.

**Rows without a target.** The target is the PR or commit that the revert names
(for example, commit 3451fc328 for #16899), or else the last PR that changed the
rolled-back lines.

**Finding the relands.** Three sources give the candidates:
- the original PR's timeline cross-references;
- the author's PRs in the 150 days after the revert;
- for the incidents that still had none, a title-keyword search of all merged PRs
  after the revert.

Each reland was confirmed by reading it. A rollback whose change stays (a partial
revert of one part) needs no reland.

**Coding.** Each row has these fields: cause, symptom, whether the failure was
silent, how it was found, the CI lane that found it, the fit with warpDRF, and the
confidence in the cause. The codebook is in [`coding.py`](coding.py) (vLLM rows),
which [`coding_sglang.py`](coding_sglang.py) (SGLang rows) shares.

**Files.**
- [`fetch.py`](fetch.py) downloads the GitHub records (read-only) to
  `data/raw/<repo>/` (gitignored).
- [`summarize.py`](summarize.py) joins them with the census and the coding, checks
  that the coding covers every census row, writes [`cases.csv`](cases.csv), and
  prints every table below.

```sh
python3 -I perf-revert-study/fetch.py
python3 -I perf-revert-study/summarize.py --asof 2026-10-13 --cases
```

**Limits.**
- **One coder.** One coder (AI-assisted) coded the rows, not the census's two.
- **CI logs not read.** The Buildkite and GitHub Actions logs were not read.
- **Low confidence.** 26 causes have low confidence: 5 in vLLM and 21 in SGLang.
  SGLang reverts often say only "Reverts #N".
- **The targeted check is a judgment,** not an experiment.
- **The warpDRF fit is a judgment from the code,** except for the two direct cases,
  which were run (see "Where warpDRF fits"); warpDRF was not run on the other
  kernels.
- **"Not relanded" is an upper bound.** A reland can be missed when it neither
  references the original nor matches the author or keyword searches.

## Why they were reverted

| Cause | vLLM | SGLang | Both | Silent | Check that targets the cause |
| --- | ---: | ---: | ---: | ---: | --- |
| UNK | 3 | 12 | 15 | 2 | not known |
| BUILD | 5 | 8 | 13 | 1 | build on every target platform and toolchain |
| DEP | 4 | 8 | 12 | 0 | dependency gate (merge after the version is pinned) |
| PRE | 6 | 5 | 11 | 4 | precondition check at the call site or in the backend selection |
| GRAPH | 5 | 4 | 9 | 7 | capture/replay differential test and a stream-race detector |
| KERNEL | 0 | 8 | 8 | 4 | differential test against the reference on the deployed shapes; a bounds and race checker |
| API | 0 | 6 | 6 | 0 | signature check across caller and callee (and across the sglang and sgl-kernel versions) |
| TEST | 2 | 4 | 6 | 0 | run the test before the merge on its target hardware |
| NUM | 1 | 3 | 4 | 2 | accuracy eval of each changed model variant |
| PERFREG | 0 | 4 | 4 | 0 | benchmark each configuration that the change affects, with the right metric |
| BUDGET | 3 | 0 | 3 | 0 | budget check (shared memory, GPU memory, start-up time) |
| MODEL | 1 | 2 | 3 | 2 | accuracy eval of each changed model variant |
| PROC | 3 | 0 | 3 | 0 | review policy |
| SKEW | 1 | 0 | 1 | 0 | merge queue (test the merge result) |

"Silent" means wrong output or lower accuracy without an error. The causes:
- **PRE:** a precondition (shape, layout, buffer size, launch limit, supported
  feature or configuration) that the call site or the backend selection does not
  enforce.
- **GRAPH:** a CUDA graph capture, torch.compile or stream-ordering invariant.
- **BUILD:** build, toolchain or platform portability, also a platform path that is
  not guarded.
- **DEP:** a dependency version or behavior.
- **API:** a caller and a callee disagree.
- **KERNEL:** a defect inside kernel code.
- **NUM:** numerical precision or semantics.
- **PERFREG:** slower on another configuration, or a wrong measurement.
- **TEST:** the failure is in the test only.
- **SKEW:** a concurrent PR changed an API.
- **UNK:** not stated and not found.

### vLLM

| Cause | Incidents | Silent | Median days to revert | Relanded | Median days revert to reland | Rework days | Check that targets the cause |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| PRE | 6 | 3 | 1.4 | 6 of 6 | 21.4 | 393 | precondition check at the call site or in the backend selection |
| BUILD | 5 | 0 | 0.2 | 4 of 5 | 10.2 | 50 | build on every target platform and toolchain |
| GRAPH | 5 | 5 | 11.1 | 2 of 5 | 11.4 | 23 | capture/replay differential test and a stream-race detector |
| DEP | 4 | 0 | 0.6 | 3 of 4 | 16.0 | 77 | dependency gate (merge after the version is pinned) |
| BUDGET | 3 | 0 | 0.9 | 1 of 3 | 10.2 | 10 | budget check (shared memory, GPU memory, start-up time) |
| PROC | 3 | 0 | 0.7 | 3 of 3 | 10.0 | 26 | review policy |
| UNK | 3 | 1 | 4.0 | 1 of 3 | 41.8 | 42 | not known |
| TEST | 2 | 0 | 1.2 | 2 of 2 | 1.9 | 4 | run the test before the merge on its target hardware |
| MODEL | 1 | 1 | 0.8 | 1 of 1 | 1.3 | 1 | accuracy eval of each changed model variant |
| NUM | 1 | 1 | 3.0 | 1 of 1 | 7.3 | 7 | accuracy eval of each changed model variant |
| SKEW | 1 | 0 | 0.0 | 1 of 1 | 2.2 | 2 | merge queue (test the merge result) |

"Rework days" is the sum, over the relanded incidents, of the days from the revert
to the reland.

- **PRE: an optimization calls a kernel outside a condition that the kernel
  needs (6).** The condition is usually stated only in the kernel's source.
  - FlashAttention needs `head_dim % 32 == 0` (#12355/#12435, llava-onevision has
    72).
  - The CUDA rotary op needs contiguous input (#14384, gsm8k fell to 0.06).
  - The CUTLASS block-FP8 GEMM needs `M % 4 == 0` (#24666).
  - FA3's scheduler-metadata buffer must hold `1 + 4·round_up(B, 4)` slots; the
    host allocated `4·B + 1` (#23465).
  - FA4 FP8-KV has no bidirectional attention (#48666).
  - All-reduce paths need contiguous input (#28841).

  These defects were found fast (median 1.4 days), but they took the longest to
  fix: the precondition had to be found first. #12355 came back after 229 days,
  with a dispatch guard.
- **GRAPH (5): all five are silent and slow to find (median 11 days).**
  - A metadata writer and its captured consumer disagree on a row stride (#50004).
  - A scratch pool is reused across streams without events (#49236).
  - An input tensor is aliased across two streams (#48223).
  - Outputs do not keep a constant address across graph replays (#29558).
  - The captured region was widened across an auxiliary stream (#45309).

  Three of the five never came back.
- **BUILD (5).** Three broke the ROCm build: a 32-bit shuffle mask on wave64, an
  NVIDIA-only PDL attribute, and a gfx950 builtin. The other two are missing CUDA
  arch flags and a module-level import.
- **The remaining vLLM causes.**
  - Dependency (4): DeepGEMM v2, FlashInfer top of tree, FlashInfer 0.6.6, and
    Triton cache hints that needed an unreleased Triton (#15511).
  - Budgets (3): L4 shared memory, a 12.22 GiB autotune workspace, JIT warmup time.
  - Process (3).
  - Test only (2).
  - Precision (1).
  - Model layout (1).
  - Merge skew (1).
  - Unknown (3). For #48641, the CI failure analyzer's explanation of an MTP hang
    is a guess, and a maintainer doubted it.

### SGLang

| Cause | Incidents | Silent | Median days to revert | Relanded | Median days revert to reland | Rework days | Check that targets the cause |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| UNK | 12 | 1 | 1.6 | 3 of 12 | 10.2 | 47 | not known |
| BUILD | 8 | 1 | 0.3 | 5 of 8 | 2.3 | 9 | build on every target platform and toolchain |
| DEP | 8 | 0 | 1.4 | 4 of 8 | 10.2 | 75 | dependency gate (merge after the version is pinned) |
| KERNEL | 8 | 4 | 9.4 | 5 of 8 | 7.3 | 158 | differential test against the reference on the deployed shapes; a bounds and race checker |
| API | 6 | 0 | 0.3 | 6 of 6 | 3.5 | 37 | signature check across caller and callee (and across the sglang and sgl-kernel versions) |
| PRE | 5 | 1 | 4.9 | 3 of 5 | 6.9 | 31 | precondition check at the call site or in the backend selection |
| GRAPH | 4 | 2 | 1.3 | 3 of 4 | 3.0 | 8 | capture/replay differential test and a stream-race detector |
| PERFREG | 4 | 0 | 0.9 | 1 of 4 | 2.7 | 3 | benchmark each configuration that the change affects, with the right metric |
| TEST | 4 | 0 | 0.6 | 2 of 4 | 18.8 | 38 | run the test before the merge on its target hardware |
| NUM | 3 | 1 | 1.4 | 1 of 3 | 46.4 | 46 | accuracy eval of each changed model variant |
| MODEL | 2 | 1 | 0.3 | 1 of 2 | 1.2 | 1 | accuracy eval of each changed model variant |

- **KERNEL: defects inside kernel code (8).** Four of them are silent, and they
  stay in main longest (median 9.4 days).
  - #11133 commented out three `tl.debug_barrier()` calls in the `causal_conv1d`
    Triton kernels. These kernels store a shifted conv state into the buffer that
    the same program loads. 40 of 244 unit tests failed, but the CI passed, because
    the test file did not emit its exit code. A CI fix (#16681) exposed the
    failures 73 days later.
  - #3628 accessed memory out of bounds when `num_groups % 16 != 0` (fixed in
    #4231).
  - #4197 moved `1/scale` to the host, which then read a device pointer.
  - Two FlashMLA updates gave NaNs or lower accuracy on B200 (#15242, #18902).
  - A multi-block `moe_align` kernel raised exceptions only inside CUDA graphs
    (#3613).
  - FlashInfer's MNNVL all-reduce hung (#12787). It compared the polled value with
    its -0.0 sentinel by a floating-point comparison, and flush-to-zero turned a
    negative subnormal input into -0.0 (fixed in FlashInfer #3304). The hang
    appeared with piecewise CUDA graphs, and the reland first disabled them for
    MNNVL.
  - Skipping the full-vocabulary softmax in the EAGLE draft collapsed MTP accuracy
    on AMD (#26235; low confidence). The later ROCm version (#35872) uses a masked
    Triton argmax, because AITER's `greedy_sample` returned an index outside the
    vocabulary. An earlier reading of this case, argmax tie-breaking, has no
    support in the texts.
- **API: a caller and a callee disagree (6).** None of these exists in vLLM.
  - A test passed a keyword that the function did not have, twice (#8364, #8770).
  - After a JIT migration, call sites passed 8 arguments to a 7-argument wrapper
    (#19437).
  - The CI ran an old `sgl-kernel` build (#32994).
  - A new argument went through shared model code that GLM-4's MoE class did not
    accept (#9671).
  - The PR called the wrong attention wrapper (#5476).
  - The PR missed one backend condition (#24884).

  The separate `sgl-kernel` release adds a version skew that vLLM does not have.
- **BUILD (8).** Most are platform paths that are not guarded:
  - non-CUDA paths that lost a scaling factor (#6220, silent);
  - HIP that needed its own scan branch (#7884);
  - a CUDA-only SM90 check that ran on AMD (#15712);
  - two diffusion kernels without an AMD fallback (#16879, #17241);
  - a CPU AMX change that broke NVIDIA FSDP (#28527).

  The other two are a FlashMLA submodule that was not registered (#4449), and GCC
  and ROCm support for the new JIT framework (#13453).
- **DEP (8).**
  - FlashInfer interfaces not yet released (#8687, #18801, #31502).
  - DeepGEMM updates (#8790, #18362).
  - An FA3 bump that broke FA4 (#11444).
  - PyTorch's eager NCCL init (#5728).
  - An internal Triton interface that differs on AMD (#11488).
- **GRAPH (4).** Two of them are silent, and the median time to revert is 1.3
  days:
  - a null pointer captured into the decode graph after `dispose_tensor` (#29142,
    silent: gsm8k fell to 0.005 with no error);
  - pinned non-blocking copies that doubled the KL divergence (#19190, silent);
  - torch.compile breaks (#18619, #18905).

  The MNNVL hang under piecewise graphs (#12787) is coded KERNEL, because its cause
  was in FlashInfer's kernel. The full-graph page-size rule (#30835) is coded TEST:
  a CPU unit-test job failed, and no product defect was shown. The narrower stream
  fence of #36752 is coded UNK: the author reverted it with no reason, and no
  failure is described.
- **PRE (5).**
  - A Triton grid of `(seq_num, max_seq_len)` exceeds the y limit above 65,535
    tokens (#16043).
  - A Triton MoE config with `BLOCK_SIZE_K = 256` against FP8 groups of 128 gave
    wrong answers for 27.8 days (#9087).
  - A strided view went into ROCm RMSNorm, which needs contiguous input (#21734).
  - The `num_token = 0` case of a JIT activation kernel (#21766).
  - Qwen2 has no DP attention (#11381).
- **NUM (3).**
  - A GELU GEMM epilogue changed images (#28166, silent).
  - A fused SiLU-mul-quant kernel lowered the EAGLE acceptance length below the
    limit of a speed test (#37376); the outputs were not wrong.
  - Test tolerance (#20501).
- **PERFREG (4).** A slower all-reduce (#2981), a slower decode plan in other cases
  (#3987), and a slower cuDNN FP4 GEMM (#17309). In one case, a measurement artifact
  (allocated rather than reserved memory) showed the gain (#18611).
- **UNK (12).** These reverts state no reason ("Reverts #N"), and no later record
  explains them.

**How the two repositories differ.**
- **SGLang records explain less.** 12 of 64 SGLang causes are unknown, against 3
  of 34 in vLLM. 21 SGLang codes have low confidence, against 5.
- **SGLang performance PRs change kernel code more often.** This covers
  `sgl-kernel`, JIT and Triton kernels, so in-kernel defects (KERNEL) and package
  skew (API) appear only there.
- **The GRAPH class behaves differently.** In vLLM all five are silent and slow to
  find (median 11.1 days). In SGLang two of four are silent, and the median time
  to revert is 1.3 days.
- **SGLang brings fewer optimizations back.** 34 of 64 came back (53%), against
  25 of 34 in vLLM (74%). When one does come back, it is faster: the median from
  revert to reland is 5.6 days in SGLang and 10.0 days in vLLM.

## How long it took

| | vLLM | SGLang | Both |
| --- | ---: | ---: | ---: |
| Median days from merge to revert | 1.1 | 0.9 | 0.9 |
| Silent failures: incidents, median days to revert (max) | 11, 3.0 (19.2) | 11, 1.1 (73.3) | 22, 2.2 (73.3) |
| Loud failures: incidents, median days to revert (max) | 23, 0.9 (4.6) | 52, 0.9 (17.8) | 75, 0.9 (17.8) |
| Days in main: silent / loud | 76.8 / 26.2 | 149.8 / 138.1 | 226.5 / 164.4 |
| Came back | 25 of 34 | 34 of 64 | 59 of 98 |
| Median days from revert to reland | 10.0 | 5.6 | 6.9 |
| Median days from merge to reland | 10.8 | 9.3 | 10.7 |
| Rework days (sum, revert to reland) | 636 | 452 | 1,088 |
| Not relanded as of 2026-10-13 | 8 | 29 | 37 |

**Who found them.**

| | vLLM | SGLang | Both |
| --- | ---: | ---: | ---: |
| An existing CI lane or test, after the merge | 16 | 30 | 46 |
| A pre-merge check that failed or was cancelled, then merged | 2 | 1 | 3 |
| Maintainers | 10 | 15 | 25 |
| Users | 5 | 9 | 14 |
| Unknown | 1 | 9 | 10 |

In vLLM, 14 of the 18 lanes are special hardware: H100/H200 5, B200 5, AMD 4. In
SGLang, 24 of the 33 lanes are the normal PR or main CI (15) or unit and kernel
tests (9), and 8 are AMD lanes. So many SGLang PRs merged before their own CI
passed:
- "Merged first to unblock" (#13453).
- "Didn't pass AMD CI" (#15712).
- A merge without the `run-ci` label (#30835).
- A GitHub auto-merge setting merged a red PR (vLLM #52606).

Five reverted SGLang PRs carry the `bypass-fastfail` label.

**What did not come back.**

| Repository | Cause | PRs (days since the revert) |
| --- | --- | --- |
| vLLM | GRAPH | #29558, #45309, #49236 |
| vLLM | UNK | #37205, #48641 |
| vLLM | BUILD | #44173 |
| vLLM | DEP | #15511 |
| vLLM | BUDGET | #52989 |
| SGLang | UNK | 9 of its 12 |
| SGLang | PERFREG | 3 of its 4 |
| SGLang | KERNEL | #3613, #4197 |
| SGLang | PRE | #9087 (399 d), #16043 (246 d) |

The other SGLang ones are spread over BUILD, DEP, GRAPH, NUM, MODEL and TEST.

## What would shorten the path to production

The goal is a shorter time until an optimization is safely in production. The data
ranks these levers.

1. **Run the lanes that found the defect before the merge, and require them to
   pass (51 of 98).** No new tool is needed.
   - In SGLang, most of these are the normal CI and unit tests, so the lever is
     "merge only when the PR's own CI passes".
   - In vLLM, most are special hardware lanes, so the lever is a test selection by
     changed files and target hardware. vLLM already runs a CI selector in shadow
     mode.

   The cost is GPU time before the merge.
2. **Check preconditions at call sites and in the backend selection (11
   incidents).** This class costs the most rework (424 days). Each precondition is
   a short predicate over shapes, strides, buffer sizes, launch limits or features:
   - `head_dim % 32`, `M % 4`, a metadata size;
   - a contiguous input;
   - grid y ≤ 65,535, `num_token > 0`;
   - `BLOCK_SIZE_K ≤ group`;
   - no bidirectional attention.

   A contract per kernel, checked at each call site and backend selection, stops
   the merge. A test that happens to use a passing shape does not.
3. **Test CUDA graph capture, torch.compile and stream order (9 incidents).**
   Three checks fit:
   - replay captured graphs at batch sizes and context lengths other than the
     capture ones, and compare with eager;
   - run multi-stream paths under a stream-race detector, for example PyTorch's
     CUDA sanitizer;
   - assert capture-stable strides and addresses for persistent buffers.

   In vLLM these defects were the slowest to find, and most of them did not come
   back.
4. **Smaller levers.**

   | Lever | Incidents |
   | --- | ---: |
   | A build matrix over the target platforms and toolchains (BUILD) | 13 |
   | A dependency gate (DEP) | 12 |
   | Differential kernel tests and a bounds and race checker (KERNEL) | 8 |
   | An accuracy eval of each changed model variant (NUM, MODEL) | 7 |
   | Signature checks across callers, callees and the two SGLang packages (API) | 6 |
   | Benchmarks of each affected configuration (PERFREG) | 4 |
   | Budget checks (BUDGET) | 3 |
   | Review policy (PROC) | 3 |
   | A merge queue (SKEW) | 1 |

   Also: SGLang's reverts should state the reason, because 12 of 64 do not.

## Where warpDRF fits

- **Direct: 2 of 98, both SGLang, both Triton.** warpDRF reads Triton kernels and
  checks bounds, launch conditions and races in them and at their call sites.
  - #11133 removed the barriers between a load and a store that overlap inside one
    program. This is an ordering hazard of the kind that warpDRF's
    `memory_ordering` check targets. It was silent for 73 days.
  - #16043 launched a grid whose y dimension can exceed 65,535. This is a
    grid-limit obligation at the call site.

  **Measured on 2026-10-13** (warpDRF b756719, Faial c8192ca1; record and
  evidence: `warpDRF/results/direct-revert-cases-2026-10-13.json`):
  - **#16043: found.** At the launch in `GetKAndS.triton`, the obligation
    `grid limit: G1 <= 65535` fails for all 16 specializations: the host never
    bounds `max_seq_len`. But it is 1 of 18 failing obligations, the contract is
    insufficient because of a race through loaded offsets (probably false), and
    the driver's reviewer filter drops every grid-limit finding. So a reviewer
    would see it only in the raw certificate.
  - **#11133, update kernel (the third barrier): found with a guarantee.** The
    check used the pair that differs only in the barrier lines (72bacc88 without,
    e75299a1 = #16899 with), with one specialization (width 5, plain decoding,
    with bias):

    | Version | Default guarantees | With `barrier.orders_global` |
    | --- | --- | --- |
    | Without the barrier | unknown (a bounds query timed out) | violation: conv-state load line 688 / store line 744 |
    | With the barrier | violation: the same pair | verified, all five properties |

    By default, warpDRF withholds `barrier.orders_global` (a CTA barrier orders
    global memory) until that guarantee is validated, so it reports the conflict
    with or without the barrier. CUDA documents this ordering for
    `__syncthreads()`. So a real kernel needs this guarantee, which the earlier
    footprint pilot never needed. The full call-site certificates time out
    (256 specializations, stopped after 3,900 s); one derivation takes about 20
    minutes.
  - **#11133, forward kernel (the first two barriers): not checkable today.**
    It has three gaps:
    - the front end typed a cast to `tl.int1` as opaque. Triton defines the
      cast as `!= 0`, and that rule (not committed) removes the gap;
    - the call-site plan probes `KERNEL_WIDTH = 64`, outside the widths 2 to 5
      of the kernel, and makes no specialization;
    - a hand-built specialization needs 32 to 37 assumptions, and Faial's async
      profile allows 16.
- **In its class: 7 more incidents (8 PRs).** These are host-side preconditions or
  bounds defects in CUDA, HIP or library kernels, which the Triton front end does
  not read:
  - vLLM: #12355/#12435, #14384, #24666, #23465;
  - SGLang: #3628, #21766, #21734.
- **Outside warpDRF.** The rest are build, dependency, interface, numerics,
  process, model and graph/stream-ordering defects. Some Triton-kernel defects are
  also outside: #9087 is a config that stays in bounds but applies the wrong
  scale, and #37376 and #20501 are numerics.

For the question "how to deploy performance PRs faster":
- A Triton-kernel verifier addresses about 2% of these reverts.
- With CUDA/HIP front ends or declared contracts for library kernels, the
  precondition-and-bounds idea reaches about 9%.
- The largest gains come from three other levers: pre-merge lane selection with
  green-CI enforcement, precondition contracts at call sites, and graph/stream
  checks.

## Cases

One row per coded row. Days are measured from the merge of the reverted PR.

### vLLM

| PR | Cause | Days to revert | Reland (days after the revert) | What happened |
| --- | --- | ---: | --- | --- |
| #12355 | PRE | 1.2 | #24347 (reimplemented, 229.1 d) | The vLLM FlashAttention build needs head_dim % 32 == 0; the generic ViT layer called it for llava-onevision (head_dim 72). A concurrent PR (#12093) also removed flash_attn_func. |
| #12435 | PRE | 0.29 | #24347 (reimplemented, 229.1 d) | Forward fix of #12355 (flash_attn_varlen_func); the same head_dim % 32 precondition failed. The reland adds a dispatch guard and an upstream-FA fallback. |
| #11589 | BUDGET | 1.15 | not needed | The PR also changed the default FP8 MoE Triton config, which needs more shared memory than the L4 has. Only the default was reverted. |
| #14384 | PRE | 1.1 | #14540 (fixed, 2.5 d) | MLA called the in-place CUDA rotary kernel on strided q_pe/k_pe slices; the kernel needs contiguous input. gsm8k fell to 0.06. The fix makes the input contiguous. |
| #15511 | DEP | 1.14 | none (564 d) | Triton cache hints that needed an unreleased Triton: with the released Triton 3.2, ptxas rejects .evict_last with .cg (issue #15619). |
| #19820 | DEP | 0.29 | #20087 (after dependency, 16.0 d) | Migrated to the DeepGEMM v2 API before DeepGEMM released it. |
| #20062 | PROC | 0.73 | #20628 (modified, 12.5 d) | The new default cuda_graph_sizes doubled the capture time; reviewers wanted a discussion. |
| #20762 | BUILD | 4.62 | #23045 (superseded, 29.4 d) | The new MoE permute .cu file had no CUDA arch flags; Llama-4 Maverick failed with 'PTX compiled with an unsupported toolchain'. An existing 2xH100 model test reproduces it. Build fix #21426; #23045 superseded the reapply. |
| #22521 | UNK | 4.03 | #25135 (superseded, 41.8 d) | ROCm AITER rope; the revert says only 'requires more investigation'. It came back inside #25135 (a ROCm upstreaming PR) 42 days later. |
| #22538 | SKEW | 0.01 | #22951 (fixed, 2.2 d) | A concurrent PR renamed the activation and added a Marlin MoE path; the merged code used the old name (RuntimeError 17 minutes after the merge). |
| #24666 | PRE | 1.54 | #25696 (fixed, 7.5 d) | The refactor dispatched to CUTLASS, which needs M % 4 == 0 on Hopper; the padding inside the compiled graph broke DeepEP (illegal memory access) under CUDA graphs with decode DBO at M = 2. |
| #25508 | NOTFAIL | 46.13 | not needed | A workaround (graph size 992 for FA3) was rolled back after FA3 lifted the restriction. |
| #27512 | BUILD | 0.32 | #28083 (fixed, 1.0 d) | Pre-merge CI failed and was taken as unrelated; the only change in the reland makes a module-level import of the FLA Triton norm lazy. |
| #28841 | PRE | 1.27 | #30647 (superseded, 112.0 d) | The slice moved before the all-reduce, so the all-reduce got a non-contiguous tensor; only one communicator was changed to reshape. |
| #28971 | TEST | 0.86 | #29708 (smaller test, 1.1 d) | The new Triton LoRA MoE test failed in nightly; the reland removed the M = 32 and 64 shapes. A known garbage-output issue under FULL CUDA graphs was open at the merge. |
| #29558 | GRAPH | 11.13 | none (299 d) | DeepEP HT outputs did not keep a constant address across piecewise graph replays (garbage); a shape that crossed a split op also broke torch.compile. |
| #30746 | DEP | 0.29 | #31195 (modified, 49.8 d) | Needed FlashInfer top of tree (#30993); the Blackwell CI broke. Resubmitted as opt-in. |
| #31645 | PROC | 0.71 | #31528 (superseded, 3.9 d) | A parallel PR with a better design was preferred. |
| #32339 | TEST | 1.58 | #32615 (same code, 2.8 d) | False alarm: the MLA backend test used unscaled random weights (unstable LSE merge, #32529); the test was not in the PR's CI. |
| #23465 | PRE | 1.49 | #34043 (fixed, 6.5 d) | FA3 with the swizzle writes 1 + 4*round_up(B, 4) scheduler-metadata slots; the host buffer had 4*B + 1, so the kernel wrote out of bounds. |
| #34492 | MODEL | 0.75 | #34697 (fixed, 1.3 d) | Wrong shard sizes for Qwen3-Next's fused qkvz projection and no TP division of the shard offset; Qwen3-Next (changed, not tested) gave random symbols. |
| #34582 | NUM | 2.98 | #35121 (superseded, 7.3 d) | bf16 router in place of fp32: 4-5% lower accuracy on Nemotron 3 Nano FP8. #35121 keeps fp32 accumulation and output. |
| #37205 | UNK | 14.57 | none (194 d) | Infrequent NaNs with CUDA graphs, prefix caching and chunked prefill (EAGLE3 acceptance to 0). No root cause was found; the wrapper does not check the contiguity and alignment that its TMA descriptors assume (plausible, not confirmed). |
| #38050 | DEP | 0.9 | #38251 (same code, 11.2 d) | The pinned FlashInfer 0.6.6 does not support the CuteDSL MoE path; the B200 kernel test crashed. |
| #44173 | BUILD | 0.15 | none (126 d) | One warp per group with a 32-bit shuffle mask; HIP (wave64) needs a 64-bit mask, so the ROCm build failed. The fix PR is still open. |
| #45309 | GRAPH | 0.32 | none (117 d) | Removing eager breaks put the KV insert and an auxiliary-stream compressor into the captured region; DeepSeek-V4 Flash (DP4 EP) gave garbage. The mechanism is not stated. |
| #42996 | BUILD | 0.1 | #46006 (fixed, 0.3 d) | The PDL launch attribute is NVIDIA-only; the ROCm build failed. The reland adds a guard. |
| #48641 | UNK | 3.31 | none (84 d) | In-place bf16 logits in place of an fp32 copy; the CI failure analyzer linked the PR to one MTP nightly hang (DP2). Its numerics explanation is a guess, and a maintainer doubted that the PR was the cause; the mechanism is not known. |
| #48597 | PROC | 0.77 | #49791+ (split, 10.0 d) | A 2,139-line PR merged without the full review; it came back as focused PRs. |
| #50004 | GRAPH | 19.24 | #52823 (fixed, 5.7 d) | The metadata writer packed rows with a batch-dependent stride, but the captured consumer kept its capture-time stride; later rows read stale data under concurrency. |
| #49236 | GRAPH | 18.35 | none (55 d) | A model-wide scratch pool reused storage across layers and CUDA streams without event tracking; a producer could overwrite a buffer that a consumer still read. |
| #48223 | GRAPH | 1.13 | #52033 (fixed, 17.2 d) | Dual-stream decode: routed and shared experts read the same input tensor in unquantized models, a cross-stream race. The reland turns off the overlap when the inputs alias. |
| #48666 | PRE | 5.71 | #53175 (fixed, 35.4 d) | The FA4 FP8-KV kernel does not support bidirectional (multimodal-prefix) attention, but the backend selection chose it; Gemma-4 emitted special tokens, seen as a perf regression. |
| #52989 | BUDGET | 0.93 | none (53 d) | A larger autotune bucket made the FlashInfer CUTLASS MoE workspace 12.22 GiB; OOM. |
| #52606 | BUILD | 0.19 | #54038 (fixed, 19.5 d) | A gfx950-only builtin compiled for gfx90a; the PR-head AMD build failed, but a GitHub auto-merge setting merged the PR. |
| #56323 | BUDGET | 0.16 | #58092 (modified, 10.2 d) | The explicit Triton JIT warmup increased the engine start-up time; H200 CI jobs timed out. #58092 registers the MRV2 sampler JIT warmups again, for ROCm. |
| — (rollback #35475) | NOTFAIL |  | not needed | Planned removal of the fast_moe_cold_start workaround for torch >= 2.11. |

### SGLang

| PR | Cause | Days to revert | Reland (days after the revert) | What happened |
| --- | --- | ---: | --- | --- |
| #2981 (rollback #3006) | PERFREG | 0.9 | none (631 d) | The switch to the sgl-kernel (TensorRT-LLM) custom all-reduce was slower (440 vs 450 tok/s, Llama-3.1-8B TP8); rolled back to the vLLM all-reduce. |
| — (rollback #3230) | NOTFAIL |  | not needed | Planned removal of the vendored TensorRT-LLM MoE code in favor of the cuBLAS grouped GEMM API. |
| #3535 | UNK | 1.01 | none (603 d) | tl.range() with host num_stages in the shared FP8 block GEMM (a ROCm change) broke MTP/NextN (#3582); merged without the DeepSeek owners. |
| #3665 (rollback #3722) | UNK | 1.91 | none (600 d) | HIP BLOCK and num_warps of the Triton extend attention (#3665) were rolled back; no reason given. |
| #3613 | KERNEL | 1.18 | none (591 d) | Multi-block moe_align CUDA kernel with an HBM buffer for block-wise reduction; it raised exceptions only inside CUDA graphs end to end (an A800 user before the merge; Qwen MoE, DeepSeek-V3). The author suspected its shared-memory and HBM use; the mechanism was not confirmed. |
| #3987 | PERFREG | 0.01 | #4012 (fixed, 2.7 d) | A fast FlashInfer MLA decode plan for graph replay regressed other cases; a revision came back. |
| #4197 | KERNEL | 0.0 | none (584 d) | The PR moved 1/scale from the per-tensor FP8 quant kernel to the host, which then read a device pointer; the unit test segfaulted 4 minutes after the merge. |
| #3628 | KERNEL | 17.68 | #4231 (fixed, 1.1 d) | Warp-shuffle reduction and vectorized I/O in the per-token-group FP8 quant CUDA kernel accessed memory out of bounds when num_groups % 16 != 0 (fix #4231); test_mla failed after 17.7 days. |
| #4449 | BUILD | 0.08 | #4472 (superseded, 0.3 d) | The FlashMLA submodule was not added; the PR worked only with the author's local install. A co-author integrated FlashMLA as a backend the same day (#4472). |
| #5476 | API | 0.65 | #5555 (fixed, 16.7 d) | Skipping the LSE without a prefix called the paged prefill wrapper where the ragged one was needed; the reward-model test failed. The fix uses the ragged wrapper. |
| #5728 | DEP | 0.41 | none (534 d) | Eager NCCL init with device_id used 3.8 GB more memory, broke DeepSeek-V3 with MTP and hung a CI test (PyTorch issue 151858). |
| #6220 | BUILD | 0.41 | #6970 (fixed, 0.8 d) | The routed scaling factor moved into a CUDA-only kernel, but the multiply was removed for all platforms, so non-CUDA paths lost it; the same-day fix keeps it there. |
| #7884 | BUILD | 10.88 | #8461 (fixed, 2.3 d) | A block/warp-scan rewrite of the moe_align CUDA kernel failed many AMD tests; the fix gives HIP its own branch. |
| #8364 | API | 0.46 | #9088 (fixed, 10.6 d) | [1/2] of a fused-gate change: the test passed a new keyword that the Python function did not have (TypeError). |
| #8770 | API | 1.99 | #9088 (fixed, 2.1 d) | The resubmit of #8364 broke the unit tests again (details on Slack); the third try stayed. |
| #8687 | DEP | 1.94 | none (428 d) | New FlashInfer all-reduce one-shot parameters need a FlashInfer interface 'in next release'. |
| #8790 | DEP | 2.29 | #9340 (after dependency, 10.2 d) | A DeepGEMM update for sgl-kernel broke the kernel CI; it depended on #9167. |
| #9671 | API | 0.1 | #10008 (fixed, 1.6 d) | An AMD MXFP4 optimization passed a new argument through the shared DeepSeek shared-expert code; GLM-4.5's MoE class did not take it, so GLM-4.5-Air crashed on B200. |
| #9087 | PRE | 27.78 | none (399 d) | A Triton fused-MoE config ported from vLLM used BLOCK_SIZE_K = 256 with FP8 block size 128, so a K tile spans two scale groups; Qwen3 answers were wrong for 27.8 days. The retuned config uses 64 and 128. |
| #9534 | UNK | 5.61 | #10312 (modified, 30.3 d) | [1/2] quant-kernel optimizations and refactors broke the DeepEP test; the reland puts the new kernel behind enable_v2. |
| #11381 | PRE | 0.21 | #11457 (fixed, 2.6 d) | Symmetric-memory all-reduce for Qwen-VL broke the restriction that Qwen2 has no DP attention; the author cancelled the pending CI test. |
| #11444 | DEP | 0.47 | #11606 (fixed, 5.8 d) | An FA3 bump changed the sgl-kernel bindings and broke FA4. |
| #11488 | DEP | 1.51 | #12176 (fixed, 10.3 d) | csgmv as the default LoRA backend came with a custom Triton kernel cache that uses an internal Triton interface, which differs between Triton versions and backends; AMD LoRA tests failed. |
| #12343 | UNK | 0.96 | #12440 (superseded, 6.9 d) | EP tuning support for the Triton fused-MoE configs; 'the implementation was incorrect'. The same author added EP support in a tuning-tool refactor (#12440). |
| #12888 (rollback #13314) | TEST | 6.6 | #14829 (same code, 25.4 d) | Rolled back while a Marlin MoE crash (#13234, an illegal memory access) was debugged; the crash had another cause, and the same change came back (#14829). |
| #13453 | BUILD | 0.09 | #13764 (fixed, 2.8 d) | The new JIT kernel framework (HiCache kernel) failed CI; the reland fixes GCC 11.4 compatibility and falls back to sgl-kernel on ROCm and NPU. |
| #13600 | TEST | 0.91 | #14277 (same code, 12.1 d) | csgmv on all CUDA GPUs was reverted for a B200 CI failure; the same code came back 12 days later. |
| #13699 | UNK | 5.03 | none (300 d) | Registering mm_fp4 as a custom op (a profile cosmetic) broke an FP4 case that was not tested. |
| #15712 | BUILD | 0.15 | #16723 (fixed, 2.4 d) | SwapAB in the Triton fused-MoE kernel ran a CUDA-only SM90 check on AMD; the rework returns early when not on CUDA. |
| #15242 | KERNEL | 22.13 | #18902 (superseded, 58.3 d) | A FlashMLA update (FP8 sparse MLA) gave NaNs and 0 gsm8k accuracy for an NVFP4 DeepSeek checkpoint at batch size above 1. |
| #11133 (rollback #16899) | KERNEL | 73.28 | not needed | The PR commented out three tl.debug_barrier() calls in the causal_conv1d Triton kernels, which store a shifted conv state into the buffer that the same program loads; 40 of 244 tests failed, but the CI passed. |
| #16879 | BUILD | 0.38 | none (269 d) | Moving the diffusion timestep embedding to a JIT kernel broke the AMD CI. |
| #17241 | BUILD | 0.04 | none (267 d) | Fused q/k norm for Flux2 had no fallback on AMD (disputed by the author). |
| #17309 (rollback #17369) | PERFREG | 12.12 | none (255 d) | fp4-gemm-backend=auto picked the cuDNN FP4 GEMM (cuDNN >= 9.15, CUDA 13), which was slower. |
| #16043 | PRE | 4.86 | none (246 d) | The fused Triton kernel launched grid = (seq_num, max_seq_len); above 65,535 tokens the grid y limit fails the launch. |
| #18362 | DEP | 0.16 | none (245 d) | A DeepGEMM upgrade (no Python stable ABI) had Python-version compatibility issues. |
| #18611 | PERFREG | 0.82 | none (239 d) | A layerwise-offload buffer pool lowered the allocated memory but not the reserved VRAM; the gain was a measurement artifact. |
| #18164 | MODEL | 0.45 | none (238 d) | A unified rotary-embedding dispatch changed the output of some diffusion models. |
| #18905 | GRAPH | 1.45 | #19140 (fixed, 4.1 d) | A JIT per-token-group quant kernel broke torch.compile (Dynamo cannot trace the JIT call, #19118); the re-land adopts that fix. |
| #19190 | GRAPH | 1.07 | #19639 (split, 1.1 d) | Removing CPU-GPU sync points (pinned non-blocking copies) doubled the KL divergence. |
| #18619 | GRAPH | 17.8 | none (224 d) | TP for Qwen-Image broke torch.compile and made generation slower; it also broke Nunchaku. |
| #19437 | API | 0.85 | #20012 (fixed, 1.0 d) | Moving NVFP4 kernels to JIT: two call sites still passed 8 arguments to the 7-argument wrapper; all NVFP4 MoE models broke. |
| #12787 | KERNEL | 0.08 | #23402 (fixed, 90.3 d) | MNNVL all-reduce fusion hung. FlashInfer #3304 fixed the cause in its MNNVL all-reduce kernel: it compared the polled value with the -0.0 sentinel by a floating-point comparison, and flush-to-zero turned a negative subnormal input into -0.0, so the kernel polled forever. Piecewise CUDA graphs only made the hang more frequent. The reland came back about 90 days later with a backend flag, once SGLang used FlashInfer 0.6.12, which has the fix. |
| #18801 | DEP | 1.35 | #23590 (after dependency, 49.1 d) | The CuteDSL FP4 GEMM backend required an unreleased FlashInfer; a CI job failed. |
| #18902 (rollback #21430) | KERNEL | 18.68 | #21922 (fixed, 7.3 d) | A FlashMLA rebase lowered GLM-5-FP8 accuracy (0.919 against 0.95) on B200 with DP and FP8 KV cache (#21291); correct on H200 and with the older FlashMLA. |
| #14162 | UNK | 0.5 | #22316 (fixed, 10.2 d) | FP8 communication in DeepEP low-latency dispatch for W4A8; no reason given. |
| #20501 | NUM | 1.36 | none (193 d) | A fused temperature + softmax Triton sampling kernel failed CI 'mainly due to numerical tolerance'; the reland with new tolerances is not merged. |
| #21766 | PRE | 0.27 | #22094 (fixed, 21.3 d) | A JIT activation CUDA kernel (PDL, vectorized) failed CI; the reland fixes the num_token = 0 case. |
| #22574 | MODEL | 0.14 | #22672 (fixed, 1.2 d) | FLUX.1-dev ModelOpt NVFP4 support failed a CI job; the reland a day later masks the global quantization flags for the second transformer of dual-transformer models (Wan2.2). |
| #21734 | PRE | 5.86 | #23227 (fixed, 6.9 d) | reshape(-1, head_dim) became a view that keeps the QKV-split stride; ROCm RMSNorm kernels assume contiguous input and faulted in 7+ tests. |
| #24884 | API | 0.21 | #25406 (fixed, 1.3 d) | Decoupling Mega MoE from DeepEP missed a Mega MoE condition in the DeepSeek model code; the reland adds it. |
| #25321 | UNK | 0.4 | none (150 d) | Skipping the parent class's workspace buffer in MLA backends; no reason given. |
| #26235 | KERNEL | 1.03 | #26397 (fixed, 0.5 d) | Skipping the full-vocab softmax in the EAGLE draft (topk 1): DeepSeek-V3.2 MTP accuracy on AMD MI35x fell to 0.035 with 96% invalid output; the reland is CUDA-only. The cause is disputed. The later ROCm version (#35872) uses a masked Triton argmax, because AITER's greedy_sample returned an index outside the vocabulary; the texts do not support argmax tie-breaking. |
| #28166 | NUM | 4.03 | #33536 (modified, 46.4 d) | tanh-GELU in the cuBLASLt GEMM epilogue moved FLUX images away from the reference. |
| #26791 (rollback #28347) | UNK | 15.84 | none (110 d) | The Gemma4 NVFP4 MoE default attention backend on SM10X was restored to trtllm_mha. |
| #29142 | GRAPH | 0.55 | #29463 (fixed, 3.0 d) | Routed experts on the main stream: dispose_tensor emptied hidden_states before a later kernel launch, so the decode CUDA graph captured a null pointer; gsm8k fell from about 0.7 to 0.005, with no error. |
| #29702 | UNK | 1.39 | none (101 d) | RMSNorm as an SGLang custom op under piecewise CUDA graphs; CI failed. |
| #28527 | BUILD | 0.95 | none (95 d) | CPU AMX optimizations added a non-idempotent second weight pass and a CPU restore of CUDA parameters; FSDP inference on NVIDIA failed. |
| #30835 | TEST | 0.09 | none (92 d) | A three-line server-argument rule (keep page size 1 for FA4 when the prefill CUDA graph is FULL) merged without CI; a CPU unit-test job (base-a-test-cpu) then failed with exit code 255. The new condition runs only after an SM100 check, so a CPU job reaches it only through a unit test of the rule. The job log has expired, and no product defect was shown. |
| — (rollback #31502) | DEP |  | none (88 d) | A FlashInfer 0.6.15 bump changed the CuteDSL MoE wrapper API (TypeError); the bump itself was reverted the same day (#31625). |
| #32994 | API | 0.06 | #33471 (split, 5.4 d) | The CI ran the old sgl-kernel, whose assert the PR had changed; the kernel and runtime parts came back as separate PRs. |
| #33623 | UNK | 0.34 | none (61 d) | Fusing the MLA gate projection into the QKV-A GEMM for Kimi K3 (tile_m 32 for the new shape); outputs ran to 2,048 tokens; cause not stated. |
| #31323 | TEST | 0.23 | none (57 d) | A persistent top-k buffer with shared-expert columns (AMD GLM-5) read the process-global ServerArgs, which a code-rule test forbids; no product defect was shown. |
| #37376 | NUM | 0.87 | none (35 d) | A fused SiLU-mul-FP8-quant Triton kernel was not numerically equal to the two-kernel path; the EAGLE acceptance length fell from 4.73 to 3.95, below the speed test's 4.0 (323 to 265 tok/s). |
| #36752 | UNK | 6.88 | none (34 d) | Publishing the final multi-layer EAGLE shared-read event (a narrower stream fence); the author reverted it after seven days with no reason. CI runs on the revert failed, but no failure is described. |
| #40445 | UNK | 2.12 | none (18 d) | One NPU scatter call for the K and V cache writes; no reason given. |
