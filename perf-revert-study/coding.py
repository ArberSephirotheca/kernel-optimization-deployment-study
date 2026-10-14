"""Manual coding of the vLLM kernel-performance revert rows of the B1 census, and the codebook that
coding_sglang.py shares.

One row per reverted PR; a census row without a target PR has `rollback` (the census revert PR)
and pr=None. Sources, read for each row: the revert PR body and discussion, the original PR's
comments after its merge, the linked issues, and the diff between the original and its reland.
The PR numbers in `reland` and `detail` are vLLM PRs.

cause:
  PRE     a precondition (shape, layout, buffer size, launch limit, supported feature or
          configuration) that the call site or the backend selection does not enforce
  GRAPH   a CUDA graph capture, torch.compile integration or stream-ordering invariant
  BUILD   build, toolchain or platform portability, also a platform-specific path that is not
          guarded and breaks another platform at run time
  DEP     a dependency version or behavior: not released, not pinned, or not as documented
  SKEW    a concurrent PR changed an API that the PR uses (merge skew)
  API     a caller and a callee disagree: inside the PR, or between sglang and the separately
          released sgl-kernel package
  KERNEL  a defect inside the kernel code (own, vendored or third-party)
  NUM     numerical precision or numerical semantics
  MODEL   model-specific weight layout or model behavior
  BUDGET  shared-memory, GPU-memory or start-up-time budget
  PERFREG the change is slower on another configuration, or a wrong measurement showed the gain
  TEST    the failure is in the test only; no product defect is shown
  PROC    process or design decision; no defect
  UNK     cause not stated and not found
  NOTFAIL planned rollback of a workaround or planned removal (not a failure)
symptom:  build | crash | hang | oom | timeout | test | silent | accuracy | perf | none | unknown
silent:   the run gives wrong output or lower accuracy without an error
detected: precheck_ignored (a pre-merge check failed or was cancelled, the PR merged anyway) |
          postmerge_ci | maintainer | user | unknown
lane:     the existing CI lane or test that found the defect after the merge ("" if none)
warpdrf:  direct (a bounds, race or launch-condition defect in a Triton kernel or at its call site,
          which warpDRF checks; expected from the code, warpDRF was not run) | class (a defect of
          such a kind, or a host-side kernel precondition, but in a CUDA, HIP or library kernel that
          warpDRF's Triton front end does not read) | no | unknown
conf:     confidence in the cause: high | medium | low
reland:   the PRs that brought the optimization back (fixed, superseded, split, ...)
"""

ROWS = [
    dict(pr=12355, incident="vit-fa", cause="PRE", symptom="crash", silent=False, detected="maintainer", lane="",
         warpdrf="class", conf="high", reland=[24347], reland_kind="reimplemented",
         detail="The vLLM FlashAttention build needs head_dim % 32 == 0; the generic ViT layer called it for "
                "llava-onevision (head_dim 72). A concurrent PR (#12093) also removed flash_attn_func."),
    dict(pr=12435, incident="vit-fa", cause="PRE", symptom="crash", silent=False, detected="maintainer", lane="",
         warpdrf="class", conf="high", reland=[24347], reland_kind="reimplemented",
         detail="Forward fix of #12355 (flash_attn_varlen_func); the same head_dim % 32 precondition failed. "
                "The reland adds a dispatch guard and an upstream-FA fallback."),
    dict(pr=11589, incident="11589", cause="BUDGET", symptom="crash", silent=False, detected="maintainer", lane="",
         warpdrf="no", conf="high", reland=[], reland_kind="not needed",
         detail="The PR also changed the default FP8 MoE Triton config, which needs more shared memory than the "
                "L4 has. Only the default was reverted."),
    dict(pr=14384, incident="14384", cause="PRE", symptom="silent", silent=True, detected="maintainer", lane="",
         warpdrf="class", conf="high", reland=[14540], reland_kind="fixed",
         detail="MLA called the in-place CUDA rotary kernel on strided q_pe/k_pe slices; the kernel needs "
                "contiguous input. gsm8k fell to 0.06. The fix makes the input contiguous."),
    dict(pr=15511, incident="15511", cause="DEP", symptom="build", silent=False, detected="postmerge_ci",
         lane="fused MoE tests (later CI)", warpdrf="no", conf="high", reland=[], reland_kind="none",
         detail="Triton cache hints that needed an unreleased Triton: with the released Triton 3.2, ptxas rejects "
                ".evict_last with .cg (issue #15619)."),
    dict(pr=19820, incident="19820", cause="DEP", symptom="crash", silent=False, detected="maintainer", lane="",
         warpdrf="no", conf="high", reland=[20087], reland_kind="after dependency",
         detail="Migrated to the DeepGEMM v2 API before DeepGEMM released it."),
    dict(pr=20062, incident="20062", cause="PROC", symptom="none", silent=False, detected="maintainer", lane="",
         warpdrf="no", conf="high", reland=[20628], reland_kind="modified",
         detail="The new default cuda_graph_sizes doubled the capture time; reviewers wanted a discussion."),
    dict(pr=20762, incident="20762", cause="BUILD", symptom="crash", silent=False, detected="user", lane="",
         warpdrf="no", conf="high", reland=[23045], reland_kind="superseded",
         detail="The new MoE permute .cu file had no CUDA arch flags; Llama-4 Maverick failed with 'PTX "
                "compiled with an unsupported toolchain'. An existing 2xH100 model test reproduces it. Build "
                "fix #21426; #23045 superseded the reapply."),
    dict(pr=22521, incident="22521", cause="UNK", symptom="unknown", silent=False, detected="unknown", lane="",
         warpdrf="unknown", conf="low", reland=[25135], reland_kind="superseded",
         detail="ROCm AITER rope; the revert says only 'requires more investigation'. It came back inside #25135 (a ROCm upstreaming PR) 42 days later."),
    dict(pr=22538, incident="22538", cause="SKEW", symptom="crash", silent=False, detected="maintainer", lane="",
         warpdrf="no", conf="high", reland=[22951], reland_kind="fixed",
         detail="A concurrent PR renamed the activation and added a Marlin MoE path; the merged code used the "
                "old name (RuntimeError 17 minutes after the merge)."),
    dict(pr=24666, incident="24666", cause="PRE", symptom="crash", silent=False, detected="user", lane="",
         warpdrf="class", conf="medium", reland=[25696], reland_kind="fixed",
         detail="The refactor dispatched to CUTLASS, which needs M % 4 == 0 on Hopper; the padding inside the "
                "compiled graph broke DeepEP (illegal memory access) under CUDA graphs with decode DBO at M = 2."),
    dict(pr=25508, incident="25508", cause="NOTFAIL", symptom="none", silent=False, detected="maintainer", lane="",
         warpdrf="no", conf="high", reland=[], reland_kind="not needed",
         detail="A workaround (graph size 992 for FA3) was rolled back after FA3 lifted the restriction."),
    dict(pr=27512, incident="27512", cause="BUILD", symptom="test", silent=False, detected="precheck_ignored",
         lane="PR CI (red before the merge)", warpdrf="no", conf="medium", reland=[28083], reland_kind="fixed",
         detail="Pre-merge CI failed and was taken as unrelated; the only change in the reland makes a "
                "module-level import of the FLA Triton norm lazy."),
    dict(pr=28841, incident="28841", cause="PRE", symptom="test", silent=False, detected="postmerge_ci",
         lane="LoRA TP Test (Distributed)", warpdrf="no", conf="medium", reland=[30647], reland_kind="superseded",
         detail="The slice moved before the all-reduce, so the all-reduce got a non-contiguous tensor; only one "
                "communicator was changed to reshape."),
    dict(pr=28971, incident="28971", cause="TEST", symptom="test", silent=False, detected="postmerge_ci",
         lane="nightly kernel tests", warpdrf="unknown", conf="low", reland=[29708], reland_kind="smaller test",
         detail="The new Triton LoRA MoE test failed in nightly; the reland removed the M = 32 and 64 shapes. "
                "A known garbage-output issue under FULL CUDA graphs was open at the merge."),
    dict(pr=29558, incident="29558", cause="GRAPH", symptom="silent", silent=True, detected="postmerge_ci",
         lane="H200 DP + DeepEP HT test", warpdrf="no", conf="high", reland=[], reland_kind="none",
         detail="DeepEP HT outputs did not keep a constant address across piecewise graph replays (garbage); a "
                "shape that crossed a split op also broke torch.compile."),
    dict(pr=30746, incident="30746", cause="DEP", symptom="test", silent=False, detected="postmerge_ci",
         lane="Blackwell CI", warpdrf="no", conf="high", reland=[31195], reland_kind="modified",
         detail="Needed FlashInfer top of tree (#30993); the Blackwell CI broke. Resubmitted as opt-in."),
    dict(pr=31645, incident="31645", cause="PROC", symptom="none", silent=False, detected="maintainer", lane="",
         warpdrf="no", conf="high", reland=[31528], reland_kind="superseded",
         detail="A parallel PR with a better design was preferred."),
    dict(pr=32339, incident="32339", cause="TEST", symptom="test", silent=False, detected="postmerge_ci",
         lane="V1 attention tests (B200)", warpdrf="no", conf="high", reland=[32615], reland_kind="same code",
         detail="False alarm: the MLA backend test used unscaled random weights (unstable LSE merge, #32529); "
                "the test was not in the PR's CI."),
    dict(pr=23465, incident="23465", cause="PRE", symptom="silent", silent=True, detected="postmerge_ci",
         lane="Distributed Tests (2xH100)", warpdrf="class", conf="high", reland=[34043], reland_kind="fixed",
         detail="FA3 with the swizzle writes 1 + 4*round_up(B, 4) scheduler-metadata slots; the host buffer had "
                "4*B + 1, so the kernel wrote out of bounds."),
    dict(pr=34492, incident="34492", cause="MODEL", symptom="silent", silent=True, detected="user", lane="",
         warpdrf="no", conf="high", reland=[34697], reland_kind="fixed",
         detail="Wrong shard sizes for Qwen3-Next's fused qkvz projection and no TP division of the shard "
                "offset; Qwen3-Next (changed, not tested) gave random symbols."),
    dict(pr=34582, incident="34582", cause="NUM", symptom="accuracy", silent=True, detected="postmerge_ci",
         lane="LM Eval Large Models (4xH100)", warpdrf="no", conf="high", reland=[35121], reland_kind="superseded",
         detail="bf16 router in place of fp32: 4-5% lower accuracy on Nemotron 3 Nano FP8. #35121 keeps fp32 "
                "accumulation and output."),
    dict(pr=37205, incident="37205", cause="UNK", symptom="silent", silent=True, detected="user", lane="",
         warpdrf="unknown", conf="low", reland=[], reland_kind="none",
         detail="Infrequent NaNs with CUDA graphs, prefix caching and chunked prefill (EAGLE3 acceptance to 0). "
                "No root cause was found; the wrapper does not check the contiguity and alignment that its TMA "
                "descriptors assume (plausible, not confirmed)."),
    dict(pr=38050, incident="38050", cause="DEP", symptom="crash", silent=False, detected="postmerge_ci",
         lane="Kernels (B200) nightly", warpdrf="no", conf="high", reland=[38251], reland_kind="same code",
         detail="The pinned FlashInfer 0.6.6 does not support the CuteDSL MoE path; the B200 kernel test crashed."),
    dict(pr=44173, incident="44173", cause="BUILD", symptom="build", silent=False, detected="postmerge_ci",
         lane="AMD build", warpdrf="no", conf="high", reland=[], reland_kind="none",
         detail="One warp per group with a 32-bit shuffle mask; HIP (wave64) needs a 64-bit mask, so the ROCm "
                "build failed. The fix PR is still open."),
    dict(pr=45309, incident="45309", cause="GRAPH", symptom="silent", silent=True, detected="maintainer", lane="",
         warpdrf="no", conf="low", reland=[], reland_kind="none",
         detail="Removing eager breaks put the KV insert and an auxiliary-stream compressor into the captured "
                "region; DeepSeek-V4 Flash (DP4 EP) gave garbage. The mechanism is not stated."),
    dict(pr=42996, incident="42996", cause="BUILD", symptom="build", silent=False, detected="postmerge_ci",
         lane="AMD build", warpdrf="no", conf="high", reland=[46006], reland_kind="fixed",
         detail="The PDL launch attribute is NVIDIA-only; the ROCm build failed. The reland adds a guard."),
    dict(pr=48641, incident="48641", cause="UNK", symptom="hang", silent=False, detected="postmerge_ci",
         lane="LM Eval Qwen3.5 (2xB200) nightly", warpdrf="no", conf="low", reland=[], reland_kind="none",
         detail="In-place bf16 logits in place of an fp32 copy; the CI failure analyzer linked the PR to one MTP "
                "nightly hang (DP2). Its numerics explanation is a guess, and a maintainer doubted that the PR was "
                "the cause; the mechanism is not known."),
    dict(pr=48597, incident="48597", cause="PROC", symptom="none", silent=False, detected="maintainer", lane="",
         warpdrf="no", conf="high", reland=[49791, 49792, 50230, 49793], reland_kind="split",
         detail="A 2,139-line PR merged without the full review; it came back as focused PRs."),
    dict(pr=50004, incident="50004", cause="GRAPH", symptom="silent", silent=True, detected="user", lane="",
         warpdrf="no", conf="high", reland=[52823], reland_kind="fixed",
         detail="The metadata writer packed rows with a batch-dependent stride, but the captured consumer kept "
                "its capture-time stride; later rows read stale data under concurrency."),
    dict(pr=49236, incident="49236", cause="GRAPH", symptom="silent", silent=True, detected="maintainer", lane="",
         warpdrf="no", conf="high", reland=[], reland_kind="none",
         detail="A model-wide scratch pool reused storage across layers and CUDA streams without event "
                "tracking; a producer could overwrite a buffer that a consumer still read."),
    dict(pr=48223, incident="48223", cause="GRAPH", symptom="accuracy", silent=True, detected="postmerge_ci",
         lane="AMD CI (gsm8k, DP2 EP)", warpdrf="no", conf="high", reland=[52033], reland_kind="fixed",
         detail="Dual-stream decode: routed and shared experts read the same input tensor in unquantized "
                "models, a cross-stream race. The reland turns off the overlap when the inputs alias."),
    dict(pr=48666, incident="48666", cause="PRE", symptom="silent", silent=True, detected="postmerge_ci",
         lane="nightly perf (H200)", warpdrf="no", conf="high", reland=[53175], reland_kind="fixed",
         detail="The FA4 FP8-KV kernel does not support bidirectional (multimodal-prefix) attention, but the "
                "backend selection chose it; Gemma-4 emitted special tokens, seen as a perf regression."),
    dict(pr=52989, incident="52989", cause="BUDGET", symptom="oom", silent=False, detected="postmerge_ci",
         lane="LM Eval PCP (B200) nightly", warpdrf="no", conf="high", reland=[], reland_kind="none",
         detail="A larger autotune bucket made the FlashInfer CUTLASS MoE workspace 12.22 GiB; OOM."),
    dict(pr=52606, incident="52606", cause="BUILD", symptom="build", silent=False, detected="precheck_ignored",
         lane="AMD image build (red before the merge)", warpdrf="no", conf="high", reland=[54038], reland_kind="fixed",
         detail="A gfx950-only builtin compiled for gfx90a; the PR-head AMD build failed, but a GitHub "
                "auto-merge setting merged the PR."),
    dict(pr=56323, incident="56323", cause="BUDGET", symptom="timeout", silent=False, detected="postmerge_ci",
         lane="H200 CI time limits", warpdrf="no", conf="high", reland=[58092], reland_kind="modified",
         detail="The explicit Triton JIT warmup increased the engine start-up time; H200 CI jobs timed out. #58092 registers the MRV2 sampler JIT warmups again, for ROCm."),
    dict(pr=None, rollback=35475, incident="r35475", cause="NOTFAIL", symptom="none", silent=False,
         detected="maintainer", lane="", warpdrf="no", conf="high", reland=[], reland_kind="not needed",
         detail="Planned removal of the fast_moe_cold_start workaround for torch >= 2.11."),
]

# The check that targets each cause (what would stop such a PR before the merge on any input,
# not only on the configuration that one test happens to run).
CHECK = {
    "PRE": "precondition check at the call site or in the backend selection",
    "GRAPH": "capture/replay differential test and a stream-race detector",
    "BUILD": "build on every target platform and toolchain",
    "DEP": "dependency gate (merge after the version is pinned)",
    "SKEW": "merge queue (test the merge result)",
    "NUM": "accuracy eval of each changed model variant",
    "MODEL": "accuracy eval of each changed model variant",
    "BUDGET": "budget check (shared memory, GPU memory, start-up time)",
    "API": "signature check across caller and callee (and across the sglang and sgl-kernel versions)",
    "KERNEL": "differential test against the reference on the deployed shapes; a bounds and race checker",
    "PERFREG": "benchmark each configuration that the change affects, with the right metric",
    "TEST": "run the test before the merge on its target hardware",
    "PROC": "review policy",
    "UNK": "not known",
    "NOTFAIL": "",
}

# Evidence read in addition to the 36 PRs, their reverts and their relands.
EVIDENCE_PRS = [12445, 14471, 14476, 21426, 23001, 29483, 29697, 29960, 30910, 30914, 31197, 32484, 32529,
                33841, 34683, 34808, 34302, 38169, 38778, 45055, 45066, 45972, 45999, 49033, 49768, 49790,
                51318, 52024, 52836, 52987, 53186, 53294, 56586, 56649, 56654, 22948, 22956, 20049, 20128,
                21334, 25607, 28080, 31978, 28345, 32244, 24347, 14540]
EVIDENCE_ISSUES = [15619, 21322, 27684, 34640, 34704, 38754, 52448]
