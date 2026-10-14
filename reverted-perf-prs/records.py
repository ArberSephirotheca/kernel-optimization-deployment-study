"""The detailed revert reason of each incident in catalog.py: one record per (repo, incident).

Each record was written from the incident's evidence sheet (evidence.py): the target PR, the revert PR and its
discussion, the comments after the merge, the issues and PRs that cross-reference the target or the revert
(failure reports, relands, fixes, dependency fixes), and, where the earlier study (../perf-revert-study) had the
incident, its coding, which was checked again against the evidence.

Fields:
- what: what the PR changed for performance, and the gain it claimed.
- stated: (source, verbatim quote) of the reason that the revert gives; ("", "") when no reason is written.
- reason: the reason in detail: the symptom, where it occurred (hardware, model, configuration), the mechanism
  as far as the evidence shows it, and how it was resolved. It says "not stated" when the evidence does not
  show the mechanism.
- cause: the cause code (CAUSES).
- symptom: the visible failure (SYMPTOMS).
- silent: True when the failure was wrong output or lower accuracy without an error.
- found_by: who found the failure, and where.
- reland: the PRs that brought the optimization back; reland_kind: how (RELAND_KINDS).
- evidence_prs, evidence_issues: the PRs and issues that support the record, other than the target and the revert.
- conf: high (the mechanism is stated or shown), medium (the failure is stated, the mechanism is inferred),
  low (little is written).
"""

CAUSES = {
    "PRE": "unenforced precondition: a shape, layout, buffer size, launch limit or supported configuration that "
           "the call site or the backend selection does not check",
    "KERNEL": "defect inside kernel code (also in a dependency's kernel)",
    "LOGIC": "logic defect in host or model code, not in a kernel",
    "GRAPH": "CUDA graph capture, torch.compile or stream order",
    "BUILD": "build, toolchain or platform, also a platform path that is not guarded",
    "DEP": "dependency version or behavior",
    "API": "caller and callee disagree",
    "NUM": "numerical precision or semantics",
    "MODEL": "model-specific layout or behavior",
    "BUDGET": "resource budget (memory, shared memory, start-up time)",
    "PERFREG": "slower on another configuration, or a wrong measurement",
    "SKEW": "merge skew",
    "TEST": "false alarm or test-only failure: no defect of the PR was shown",
    "PROC": "process or design (review, discussion, a better design)",
    "UNK": "the mechanism is not stated and not found (the symptom may be stated)",
    "NOTFAIL": "planned rollback or replacement, not a failure",
}
SYMPTOMS = ("wrong output", "lower accuracy", "NaN", "crash", "illegal memory access", "hang", "OOM", "slower",
            "more memory", "CI failure", "test failure", "build failure", "compile failure", "none", "unknown")
RELAND_KINDS = ("fixed", "same code", "modified", "relanded", "split", "smaller", "superseded", "after dependency",
                "not needed", "none")   # "relanded": re-applied; whether the code changed was not checked

RECORDS = {
    # ---------------------------------------------------------------- SGLang
    ("sglang", "sglang-2828"): dict(
        what="adds TurboMind's AWQ GEMM kernels as a quantization method (awq_turbomind); the linked issue calls "
             "TurboMind's AWQ and GPTQ 'among the best-performing open-source implementations'",
        stated=("revert #2866", "it introduces extra dependency and breaks some internal stuff"),
        reason="The PR imported TurboMind unconditionally, so TurboMind became a hard dependency. The maintainer "
               "reverted it the same day and asked for TurboMind to be 'an optional dependency and only import it "
               "when it is used'. The 'internal stuff' that broke is not named. A resubmit (#2900) was closed "
               "without a merge.",
        cause="DEP", symptom="unknown", silent=False, found_by="the maintainer, in internal use, the same day",
        reland=[], reland_kind="none", evidence_prs=[2900], evidence_issues=[2788], conf="medium"),
    ("sglang", "sglang-3535"): dict(
        what="uses tl.range() with num_stages set by the host in the FP8 block GEMM Triton kernels, so that "
             "Triton makes a better software pipeline on ROCm",
        stated=("revert #3632", "this pr #3535 breaks MTP(NextN) #3582 ... Please involve @zhyncs and @ispobock in "
                                "any changes related to DeepSeek. Please do not merge directly in the future"),
        reason="The change targeted ROCm but edited the block GEMM kernels that every platform runs. It broke the "
               "DeepSeek-V3/R1 NextN (MTP) speculative decoding that #3582 had added three days earlier. The revert "
               "does not describe the failure or its mechanism. The maintainers also objected that the PR merged "
               "without the DeepSeek code owners.",
        cause="UNK", symptom="unknown", silent=False, found_by="the maintainers, one day after the merge",
        reland=[], reland_kind="none", evidence_prs=[3582], evidence_issues=[], conf="medium"),
    ("sglang", "sglang-3665"): dict(
        what="changes the BLOCK sizes and num_warps of the Triton extend-attention kernel on HIP, for AMD prefill "
             "speed",
        stated=("", ""),
        reason="The rollback #3722 restored the earlier BLOCK and num_warps values on HIP two days after the merge. "
               "Neither PR gives a reason, and no comment or linked issue explains the rollback.",
        cause="UNK", symptom="unknown", silent=False, found_by="unknown", reland=[], reland_kind="none",
        evidence_prs=[], evidence_issues=[], conf="low"),
    ("sglang", "sglang-3613"): dict(
        what="a multi-block moe_align_block_size CUDA kernel with a block-wise reduction through an HBM buffer and "
             "a cooperative launch; it claims 3x to 6x faster alignment on A100",
        stated=("revert #3982", "Will merge it after further modification from @BBuf"),
        reason="The revert gives no failure. Before the merge, a user on A800 hit an exception during CUDA graph "
               "capture with the new kernel. Three days after the revert, the author wrote that 'the kernel has "
               "problems when used together with cudagraph in end2end mode (verifed in Qwen MoE, DeepSeek V3 MoE)', "
               "that it 'can only be produced when used inside cuda graph mode', and that disabling CUDA graphs "
               "avoids it. The author suspected the kernel's resource use: it uses a lot of shared memory and an HBM "
               "buffer for the reduction, so 'resources bugget may be limited when multiple concurrent kerenels "
               "executed'. The mechanism was not confirmed. The kernel did not come back in this form.",
        cause="KERNEL", symptom="crash", silent=False,
        found_by="users who ran it end to end with CUDA graphs (A800 before the merge, then H800/H200 tests)",
        reland=[], reland_kind="none", evidence_prs=[], evidence_issues=[], conf="low"),
    ("sglang", "sglang-3987"): dict(
        what="a fast FlashInfer MLA decode plan for CUDA graph replay that keeps seq_lens on the CPU, so that replay "
             "does not copy indptr tensors between CPU and GPU (that copy also hung graph replay)",
        stated=("revert #4008", "because it introduces regression for other cases"),
        reason="The revert says only that the change caused a regression in 'other cases'; it names neither the "
               "cases nor the kind of regression (the census reads it as a performance regression). A revision came "
               "back two days later (#4012, '[Revision] Add fast decode plan for flashinfer mla'), and its "
               "description does not say what changed.",
        cause="PERFREG", symptom="unknown", silent=False, found_by="the maintainer, the same day",
        reland=[4012], reland_kind="modified", evidence_prs=[4012], evidence_issues=[], conf="low"),
    ("sglang", "sglang-4197"): dict(
        what="removes a redundant computation of the scale in the per-tensor FP8 quant kernel (with a benchmark "
             "against vLLM's kernel)",
        stated=("comment on #4197", "This PR will fail the unit test: ... Fatal Python error: Segmentation fault"),
        reason="The PR moved the 1/scale computation from the kernel to the host, and the host code then read a "
               "device pointer. The sgl-kernel unit test test_per_tensor_quant_fp8.py segfaulted. A reviewer also "
               "wrote that the change 'would likely broke CUDA Graph'. It was reverted minutes after the merge, and "
               "it did not come back.",
        cause="KERNEL", symptom="crash", silent=False, found_by="a reviewer who ran the unit test after the merge",
        reland=[], reland_kind="none", evidence_prs=[], evidence_issues=[], conf="high"),
    ("sglang", "sglang-3628"): dict(
        what="a warp-shuffle reduction and FlashInfer vectorized loads in the per-token-group FP8 quant CUDA "
             "kernel (faster than the Triton kernel in its benchmark)",
        stated=("revert #4219", "revert pr 3628 to pass test_mla ci"),
        reason="17.7 days after the merge, test_mla failed in CI. The fix #4231 ('fix per_token_group_quant_fp8 "
               "illegal memory when num_groups % 16 != 0') shows the cause: the vectorized path accessed memory out of "
               "bounds when the number of groups is not a multiple of 16. The optimization came back with the fix.",
        cause="KERNEL", symptom="illegal memory access", silent=False,
        found_by="post-merge CI (test_mla), 17.7 days after the merge",
        reland=[4231], reland_kind="fixed", evidence_prs=[4231], evidence_issues=[], conf="high"),
    ("sglang", "sglang-4449"): dict(
        what="adds DeepSeek's FlashMLA decode kernel to sgl-kernel",
        stated=("revert #4470", "missing `git submodule add https://github.com/deepseek-ai/FlashMLA "
                                "sgl-kernel/3rdparty/flashmla` ... the original pr simply does not work"),
        reason="The PR did not register the FlashMLA submodule, so sgl-kernel built only where FlashMLA was already "
               "installed by hand: the unit tests passed on the maintainer's machine because 'I have previously "
               "manually installed flash_mla locally'. It was reverted the same day. A co-author integrated FlashMLA "
               "as an attention backend the same day (#4472).",
        cause="BUILD", symptom="build failure", silent=False, found_by="the maintainer, the same day",
        reland=[4472], reland_kind="superseded", evidence_prs=[4472], evidence_issues=[], conf="high"),
    ("sglang", "sglang-4764"): dict(
        what="routes DeepSeek's scaling rotary embedding to FlashInfer's fused apply_rope_with_cos_sin_cache_inplace "
             "kernel (about 5% gain on 8xH20)",
        stated=("revert #4959", "Our inner developer found some crashes related to the RoPE change especially with "
                                "torch.compile"),
        reason="After the change, DeepSeek-R1 crashed with torch.compile; issue #4964 reports 'Running DeepSeek R1 "
               "error when enable flashinfer-mla and torch-compile'. The revert also removed the follow-up #4813. "
               "The mechanism is not stated. A different CUDA RoPE for DeepSeek came back two weeks later (#5385, "
               "'Apply deepseek cuda rope').",
        cause="GRAPH", symptom="crash", silent=False,
        found_by="internal users, five days after the merge; also user issue #4964",
        reland=[5385], reland_kind="modified", evidence_prs=[4813, 5385], evidence_issues=[4964, 4974], conf="medium"),
    ("sglang", "sglang-5476"): dict(
        what="skips the log-sum-exp in ragged prefill when there is no cached prefix, to save compute",
        stated=("revert #5544", "fix `python3 models/test_reward_models.py`"),
        reason="The reward-model test failed after the merge. The no-prefix path called the paged prefill wrapper "
               "where the ragged one was needed (the earlier study's reading of the fix). The author reopened the "
               "change as #5555, which passed test_reward_models.py, and it merged 17 days later.",
        cause="API", symptom="test failure", silent=False, found_by="post-merge CI (test_reward_models.py)",
        reland=[5555], reland_kind="fixed", evidence_prs=[5555], evidence_issues=[], conf="medium"),
    ("sglang", "sglang-5728"): dict(
        what="passes device_id to the distributed init so that NCCL communicators are created eagerly; it should "
             "cut warmup latency and let sub-groups use ncclCommSplit",
        stated=("revert #5786", "It introduces many problems. See the original thread."),
        reason="The thread lists three failures on the day of the merge: on H20 with TP=8 the change 'consumes 3.8GB "
               "more memory'; DeepSeek-V3 with MTP failed at start-up with a scheduler exception; and the CI test "
               "test_update_weights_from_distributed.py 'will hang forever after your change' (issue #5770 also "
               "reports an NCCL error in CUDA graph capture). A maintainer later confirmed that eager NCCL init with "
               "device_id has problems in PyTorch itself (pytorch/pytorch#151858). It did not come back.",
        cause="DEP", symptom="hang", silent=False,
        found_by="users and the maintainer, the same day (an H20 user, a DeepSeek-V3 MTP user, the CI test)",
        reland=[], reland_kind="none", evidence_prs=[], evidence_issues=[5770], conf="high"),
    ("sglang", "sglang-6220"): dict(
        what="fuses the multiply by DeepSeek's routed scaling factor into the topk_reduce kernel",
        stated=("", ""),
        reason="The PR removed the multiply by routed_scaling_factor from the model for every platform, but fused "
               "it only into the CUDA kernel, so the HIP and other non-CUDA paths lost the factor and computed wrong "
               "MoE outputs. The revert gives no reason. The same-day reland #6970 shows the fix: it multiplies by "
               "the factor again 'if not _is_cuda' and keeps the fused path only under '_is_cuda'.",
        cause="BUILD", symptom="wrong output", silent=True, found_by="unknown (reverted the next day)",
        reland=[6970], reland_kind="fixed", evidence_prs=[6970], evidence_issues=[], conf="medium"),
    ("sglang", "sglang-7623"): dict(
        what="shards the embedding over the attention-TP group also when dp > 1; for DeepSeek-R1 with TP16 and DP4 "
             "the embedding takes 442 MiB per TP group in place of 1767.5 MiB",
        stated=("comment on #7623", "We have to revert this PR because it cannot pass newly added tests in #7872. I "
                                   "note that `use_attn_tp_group` in `VocabParallelEmbedding` was not properly "
                                   "tested before."),
        reason="New tests from #7872 failed with the change: TestHybridDPTP hung in CI. The maintainer noted that "
               "the use_attn_tp_group path of VocabParallelEmbedding, which this one-line change turned on, had not "
               "been tested before. The mechanism of the hang is not stated. It did not come back.",
        cause="UNK", symptom="hang", silent=False,
        found_by="post-merge CI, four days after the merge (new tests from #7872)",
        reland=[], reland_kind="none", evidence_prs=[7872], evidence_issues=[], conf="medium"),
    ("sglang", "sglang-7236"): dict(
        what="replaces the simple transfer of the Mooncake connector with its batch transfer for PD "
             "disaggregation; it claims 8x to 12x faster KV transfer (650 ms to 60 ms)",
        stated=("comment on revert #7968", "we only need to fix batch_transfer_sync with nvlink transport, then I "
                                           "will release a new package and bring this change back. The older "
                                           "transfer_sync is functioning well."),
        reason="The batch transfer API of the Mooncake transfer engine did not work with the NVLink transport. The "
               "fix belonged in Mooncake and needed a new package release, so SGLang went back to the older "
               "transfer_sync. The failure itself is not described. No reland was found.",
        cause="DEP", symptom="unknown", silent=False, found_by="the maintainers, 18 days after the merge",
        reland=[], reland_kind="none", evidence_prs=[], evidence_issues=[], conf="medium"),
    ("sglang", "sglang-7884"): dict(
        what="a block- and warp-scan algorithm in moe_align_block_size_kernel (about 10% faster)",
        stated=("revert #8457", "PR has bug, and it caused a lot of test failture in ci."),
        reason="The new scan code did not work on AMD: '7884 has issue when running in AMD environment', and many "
               "AMD CI tests failed 11 days after the merge. The hot-fix #8461 brought the improvement back with "
               "separate HIP and CUDA branches ('diverge the branches for HIP and CUDA').",
        cause="BUILD", symptom="test failure", silent=False, found_by="post-merge AMD CI, 11 days after the merge",
        reland=[8461], reland_kind="fixed", evidence_prs=[8461], evidence_issues=[], conf="high"),
    ("sglang", "sglang-8364"): dict(
        what="fuses the multiply by routed_scaling_factor into the moe_fused_gate (select_experts) kernel, as "
             "TRT-LLM does; 10.46% faster prefill at batch size 1",
        stated=("comment on #8364", "Please fix the broken unit test for the fused gate. TypeError: "
                                   "biased_grouped_topk_gpu() got an unexpected keyword argument "
                                   "'apply_routed_scaling_factor_on_output'"),
        reason="Two reverts. The first PR (#8364) updated the kernel and its unit test, but the unit test called the "
               "Python function with a keyword that the installed sgl-kernel did not accept, so the test failed with "
               "a TypeError, and the PR was reverted the same day. The resubmit (#8770) 'broke the uts' again two "
               "days later (the details were discussed on Slack). The third try (#9088) stayed: it removed the unit "
               "test 'because it would fail until sgl-kernel is updated'. The cause is the version skew between the "
               "sglang package and the separately released sgl-kernel package.",
        cause="API", symptom="test failure", silent=False, found_by="post-merge unit tests, twice",
        reland=[9088], reland_kind="fixed", evidence_prs=[9088], evidence_issues=[], conf="high"),
    ("sglang", "sglang-6555"): dict(
        what="runs several tokenizer-manager processes to raise tokenizer throughput and cut latency",
        stated=("revert #8960", "It is a great optimization but I need to review this PR more carefully to reduce "
                                "code duplication."),
        reason="No failure. The maintainer reverted the 1,135-line PR the same day to review it more carefully and "
               "to reduce code duplication. It came back as #8964 three weeks later.",
        cause="PROC", symptom="none", silent=False, found_by="the maintainer, at a review after the merge",
        reland=[8964], reland_kind="modified", evidence_prs=[8964], evidence_issues=[], conf="high"),
    ("sglang", "sglang-8027"): dict(
        what="an NPU graph runner for Ascend, like CUDA graphs, to cut the overhead of operator calls; it moves the "
             "shared logic into a base GraphRunner class",
        stated=("revert #9348", "This reverts #8027 due to missing git history of `cuda_graph_runner.py`."),
        reason="No failure. The first revert restored the git history of cuda_graph_runner.py, which the refactor "
               "had moved. The rework (#9350) was reverted the next day with no reason; a comment objects that the "
               "file layout was changed again ('Why do you repeatedly modify the layout of the CUDAGraph file?'). "
               "The feature merged a third time as #9399.",
        cause="PROC", symptom="none", silent=False, found_by="the authors and the maintainers",
        reland=[9399], reland_kind="modified", evidence_prs=[9399], evidence_issues=[], conf="high"),
    ("sglang", "sglang-9671"): dict(
        what="fuses the activation quantization into the activation, layernorm, GEMM and flatten ops for "
             "DeepSeek-V3/R1 MXFP4 on AMD (about 10% more throughput)",
        stated=("revert #9959", "Because #9671 breaks `python3 -m sglang.launch_server --model zai-org/GLM-4.5-Air "
                                "--tp-size 4 --trust-remote-code --mem-fraction-static 0.85 --ep-size 4` on B200"),
        reason="The AMD optimization passed a new argument (gemm_output_zero_allocator) through the shared DeepSeek "
               "MoE code. GLM-4.5 reuses that code, but its MoE class did not take the argument, so GLM-4.5-Air "
               "crashed at start-up on NVIDIA B200. The next-day reland #10008 adds glm4_moe.py and passes the "
               "argument by keyword.",
        cause="API", symptom="crash", silent=False, found_by="the maintainer, the same day (GLM-4.5-Air on B200)",
        reland=[10008], reland_kind="fixed", evidence_prs=[10008], evidence_issues=[], conf="high"),
    ("sglang", "sglang-9087"): dict(
        what="a Triton fused-MoE tuning config for Qwen3-30B-A3B FP8 on B200, ported from vLLM #19455",
        stated=("revert #10185", "With the triton configuration, Qwen3 return incorrect response when using triton "
                                 "moe backend. Let's revert the commit and retune the result."),
        reason="The config sets BLOCK_SIZE_K = 256, but the FP8 weights have block scales over groups of 128, so one "
               "K tile spans two scale groups while the kernel applies one scale per tile (the earlier study's "
               "reading of the config). Qwen3-30B FP8 gave wrong answers with the Triton MoE backend on B200 for "
               "27.8 days. A retuned config (#10537) was closed without a merge.",
        cause="PRE", symptom="wrong output", silent=True, found_by="a user, 27.8 days after the merge",
        reland=[], reland_kind="none", evidence_prs=[10537], evidence_issues=[], conf="medium"),
    ("sglang", "sglang-8238"): dict(
        what="NCCL symmetric memory (window registration) for TP all-reduce, behind --enable-symm-mem; its "
             "DeepSeek-R1 benchmark goes from 6671.94 to 7129.24 tok/s",
        stated=("revert #10210", "Revert the changes of model forward code in #8238. We will do a clean version in "
                                 "#9358"),
        reason="No failure. A partial revert took the changes to the model forward code out, so that a cleaner "
               "version (#9358) could merge as one PR 'for easier review and benchmarking'. The author answered that "
               "#9358 already had the cleanup. The same day, the revert was itself reverted (#10238).",
        cause="PROC", symptom="none", silent=False, found_by="the maintainer, at a review 39 days after the merge",
        reland=[10238], reland_kind="same code", evidence_prs=[9358, 10238], evidence_issues=[], conf="high"),
    ("sglang", "sglang-9534"): dict(
        what="[1/2] optimizations and refactors of the quantization kernels in sgl-kernel (#7601 has the rest)",
        stated=("comment on #9534", "this pr breaks https://github.com/sgl-project/sglang/blob/main/test/srt/ep/"
                                   "test_deepep_small.py"),
        reason="Six days after the merge, the DeepEP test test_deepep_small.py failed in CI (an exception in the "
               "detokenizer process). The PR had planned a two-step check because the sgl-kernel unit tests and "
               "the srt tests depend on each other across a kernel release; issue #10291 ('Run full CI for kernel "
               "changes') was opened the same day. The reland #10312 says that its 'real change is tiny' and that "
               "it fixes 'either error or 0% acc' in that test on Blackwell and Hopper; it does not state the "
               "mechanism.",
        cause="UNK", symptom="test failure", silent=False,
        found_by="post-merge CI (test_deepep_small.py), six days after the merge",
        reland=[10312], reland_kind="fixed", evidence_prs=[7601, 10312], evidence_issues=[10291], conf="low"),
    ("sglang", "sglang-11381"): dict(
        what="an in-place symmetric-memory all-reduce for Qwen2.5-VL and other models with in-place all-reduce: one "
             "all-reduce goes from 446.9 us to 209.1 us, and input throughput rises 2% to 3.9%",
        stated=("comment on revert #11436", "The reason is qwen2 doesn't support dp-attention, but this PR breaks "
                                            "this restriction."),
        reason="The change broke the vLLM dependency test. Before the merge, the author had cancelled that CI job "
               "because it waited about an hour for a runner ('Not realize it happens to break this test'). Qwen2 "
               "does not support DP attention, and the new path did not respect that restriction. It came back "
               "with a fix three days later (#11457).",
        cause="PRE", symptom="test failure", silent=False,
        found_by="the vLLM dependency test, which the author had cancelled before the merge",
        reland=[11457], reland_kind="fixed", evidence_prs=[11457], evidence_issues=[], conf="medium"),
    ("sglang", "sglang-11488"): dict(
        what="makes the csgmv (chunked SGMV) LoRA kernels the default LoRA backend",
        stated=("comment on #11488", "this pr breaks lora unit tests on AMD"),
        reason="The LoRA unit tests failed on AMD the day of the merge. The reland #12176 explains: one optimization "
               "that came with csgmv, a custom Triton kernel cache, 'depends on an internal triton interface that "
               "varies between triton versions and backends'. The reland puts that cache behind a default-off "
               "environment variable and keeps csgmv as the default.",
        cause="DEP", symptom="test failure", silent=False, found_by="post-merge AMD CI (LoRA tests), the same day",
        reland=[12176], reland_kind="fixed", evidence_prs=[12176], evidence_issues=[], conf="high"),
    ("sglang", "sglang-11592"): dict(
        what="the scheduler and context manager for PD multiplexing (prefill and decode share the GPUs and switch "
             "streams), a base for later performance work",
        stated=("revert #12267", "this pr breaks the CIs"),
        reason="CI failed the day after the merge. The reland #12275 shows the fix: it delays the import of "
               "sgl_kernel.spatial to the PD-multiplexing initialization ('lazy import spatial'), so the eager import "
               "of the new kernel module was what broke CI where that module is not available.",
        cause="BUILD", symptom="CI failure", silent=False, found_by="post-merge CI, the next day",
        reland=[12275], reland_kind="fixed", evidence_prs=[12275], evidence_issues=[], conf="medium"),
    ("sglang", "sglang-12343"): dict(
        what="expert-parallel support in the Triton fused-MoE tuning, so that configs are tuned per EP size",
        stated=("comment on revert #12377", "The implementation was incorrect"),
        reason="The maintainer said only that 'The implementation was incorrect'; neither the revert nor the "
               "thread says what was wrong. The same author added EP support a week later in a refactor of the "
               "tuning tools (#12440).",
        cause="UNK", symptom="unknown", silent=False, found_by="the maintainers, the next day",
        reland=[12440], reland_kind="superseded", evidence_prs=[12440], evidence_issues=[], conf="low"),
    ("sglang", "sglang-12888"): dict(
        what="uses the moe_sum_reduce kernel in fused_marlin_moe (4.6 us to 1.5 us at batch size 1)",
        stated=("rollback #13314", "moe_sum_reduce may have launch issue in some case, reported in #13234"),
        reason="A user reported an illegal memory access with Marlin MoE (#13234), and the change was rolled back on "
               "suspicion. The user then reported that the crash remained with the rollback ('Still crashes. I "
               "don't think this is root cause'). The same change came back after 25 days (#14829).",
        cause="TEST", symptom="illegal memory access", silent=False, found_by="a user report (#13234)",
        reland=[14829], reland_kind="same code", evidence_prs=[14829], evidence_issues=[13234], conf="medium"),
    ("sglang", "sglang-13453"): dict(
        what="a JIT kernel framework (tvm-ffi bindings) with a HiCache JIT kernel; JIT kernels can use runtime "
             "information such as the element size and the head dimension for more optimization",
        stated=("revert #13644", "due to CI errors"),
        reason="The maintainer merged it 'first to unblock the kernel dev' and reverted it the same day because of a "
               "CI error. The reland #13764 fixes 'compatibility issue, support gcc >= 11.4.0' and falls back to the "
               "sgl-kernel kernel on ROCm and NPU.",
        cause="BUILD", symptom="CI failure", silent=False, found_by="post-merge CI, the same day",
        reland=[13764], reland_kind="fixed", evidence_prs=[13764], evidence_issues=[], conf="high"),
    ("sglang", "sglang-13600"): dict(
        what="enables the csgmv LoRA backend automatically on CUDA",
        stated=("revert #13707", "since it will break b200 ci"),
        reason="The revert links a B200 CI failure. The same code came back 12 days later unchanged (#14277, a revert "
               "of the revert), so the failure was not shown to be a defect of the PR.",
        cause="TEST", symptom="CI failure", silent=False, found_by="post-merge B200 CI, the next day",
        reland=[14277], reland_kind="same code", evidence_prs=[14277], evidence_issues=[], conf="low"),
    ("sglang", "sglang-14044"): dict(
        what="makes the piecewise CUDA graph runner use the same global graph memory pool as the main CUDA graph "
             "runner, to save GPU memory",
        stated=("revert #14278", "reverts PR #14044 to test if it resolves the `TestQwen3NextPiecewiseCudaGraph` "
                                 "failure on H100 (accuracy=0.0, invalid=1.0)"),
        reason="TestQwen3NextPiecewiseCudaGraph gave accuracy 0.0 on H100. The revert tested two hypotheses: that "
               "sharing one memory pool between the piecewise and the main CUDA graph runners caused it, and that "
               "set_graph_pool_id no longer set the pool id once one was set. The Qwen3 test passed with the revert. "
               "Which of the two was the mechanism was not confirmed. It did not come back.",
        cause="GRAPH", symptom="wrong output", silent=True,
        found_by="post-merge CI (TestQwen3NextPiecewiseCudaGraph on H100), the next day",
        reland=[], reland_kind="none", evidence_prs=[], evidence_issues=[], conf="medium"),
    ("sglang", "sglang-13699"): dict(
        what="registers mm_fp4 directly as a custom op, to remove a large pybind line from the FP4 profile under "
             "piecewise CUDA graphs",
        stated=("revert #15284", "Seems like it broke this, and I only tested normal FP4 CUDA graph in that PR by "
                                 "accident. Anyway, the original issue was cosmetic, so revert it for now."),
        reason="The author found that the change broke an FP4 case that the revert does not name; the PR had been "
               "tested only with the normal FP4 CUDA graph path. Because the original problem was cosmetic (a profile "
               "line), the author reverted it in place of a fix. The failure is not described.",
        cause="UNK", symptom="unknown", silent=False, found_by="the author, five days after the merge",
        reland=[], reland_kind="none", evidence_prs=[], evidence_issues=[], conf="low"),
    ("sglang", "sglang-14644"): dict(
        what="simplifies load_mm_data so that it loads all multimodal data directly ('1 token to 1 data'), to "
             "speed up most VLM models",
        stated=("revert #15911", "Reverts sgl-project/sglang#14644 due to CI error"),
        reason="A Qwen2.5-VL CI case failed after the merge, and the maintainer reverted the PR the same day. The "
               "author then showed that the PR was not the cause: Qwen2.5-VL output was garbage also with the PR "
               "reverted locally, and the regression came from another PR ('[bug fix][pp] fix weight load for "
               "qwen2.5-vl (#15138)', fixed in #15398). The refactor came back as #16152.",
        cause="TEST", symptom="CI failure", silent=False, found_by="post-merge CI, the same day",
        reland=[16152], reland_kind="relanded", evidence_prs=[15138, 15398, 16152], evidence_issues=[], conf="high"),
    ("sglang", "sglang-16330"): dict(
        what="warmup requests at start-up for chosen resolutions in the diffusion server, so that the first real "
             "requests do not pay the warmup cost",
        stated=("revert #16433", "AttributeError: '_TorchMemorySaverAdapterNoop' object has no attribute "
                                 "'log_timing'"),
        reason="CUDA graph capture crashed in CI: cuda_graph_runner.py read memory_saver_adapter.log_timing, which "
               "the no-op memory-saver adapter (used when the memory saver is off) does not have. The author had "
               "merged after the multimodal-gen tests passed on retries, because the tests 'have been quite flaky "
               "recently', and agreed that this should not excuse merging a PR with known issues. It did not come "
               "back in this form.",
        cause="API", symptom="crash", silent=False, found_by="post-merge CI, the same day",
        reland=[], reland_kind="none", evidence_prs=[], evidence_issues=[], conf="high"),
    ("sglang", "sglang-13592"): dict(
        what="a zero-copy transfer of large multimodal tensors from the tokenizer to the scheduler in place of "
             "send_pyobj/recv_pyobj, to cut serialization overhead for VLMs",
        stated=("revert #16386", "The code quality needs improvement. We could adopt an approach that requires fewer "
                                 "changes, avoids modifying key interfaces, and still achieves the same performance."),
        reason="No failure. The PR merged after CI passed but before the scheduler reviewer approved; changes to the "
               "manager and scheduler code 'need to be handled carefully and approved by the reviewer before "
               "merging, even if all CIs have passed'. The reverter proposed a smaller change that keeps the key "
               "interfaces.",
        cause="PROC", symptom="none", silent=False, found_by="reviewers, the next day",
        reland=[], reland_kind="none", evidence_prs=[], evidence_issues=[], conf="high"),
    ("sglang", "sglang-15712"): dict(
        what="SwapAB in the Triton fused-MoE kernel on SM90: for a small M with fp8_w8a8, it transposes A and B to "
             "use WGMMA better",
        stated=("revert #16676", "merged cause AMD CI failed [stage-a-test-1-amd (linux-mi325-gpu-1)]"),
        reason="The PR merged although AMD CI had not passed ('this pr didn't pass amd CI'), and AMD CI on MI325 "
               "failed. The SwapAB condition ran a CUDA-only SM90 capability check on AMD; the rework #16723 returns "
               "early when the platform is not CUDA (the earlier study's reading of the rework). It came back three "
               "days later.",
        cause="BUILD", symptom="CI failure", silent=False, found_by="AMD CI (MI325), red at the merge",
        reland=[16723], reland_kind="fixed", evidence_prs=[16723], evidence_issues=[], conf="high"),
    ("sglang", "sglang-15242"): dict(
        what="updates FlashMLA to a branch with SM100 optimizations of the FP8 sparse MLA kernel (DeepSeek-V3.2)",
        stated=("revert #16678", "This update is breaking the nvfp4 Deepseek checkpoint, leading to 0 accuracy with "
                                 "gsm8k 8 shots. It works fine with the fp8 checkpoint, however."),
        reason="With the NVFP4 DeepSeek checkpoint, gsm8k accuracy fell to 0 unless the batch size was limited to 1; "
               "the FP8 checkpoint worked. The author compared the sgl-kernel FlashMLA output with the upstream "
               "FlashMLA kernel and found NaNs in the new output (for example 'NaNs in o_test: 263168'). The FlashMLA "
               "fork also reverted its 'SM100 optimizations for fp8 sparse_mla kernel' (sgl-project/FlashMLA #4). A "
               "later FlashMLA rebase (#18902) replaced the update.",
        cause="KERNEL", symptom="NaN", silent=True,
        found_by="the author, with the NVFP4 checkpoint, 22 days after the merge",
        reland=[18902], reland_kind="superseded", evidence_prs=[18902], evidence_issues=[], conf="high"),
    ("sglang", "sglang-13947"): dict(
        what="compiles the grammar on one TP rank and syncs the result, to cut the constrained-decoding overhead "
             "under TP (issue #13809)",
        stated=("revert #16845", "DP Attention is broken."),
        reason="The change broke DP attention; the revert gives no more detail. A later issue (#35826, 'Grammar token "
               "sync initializes a singleton NCCL group under DP attention') reports a related problem with the "
               "grammar token sync under DP attention. Redo attempts (#16874, #38392, #39053) had not merged by "
               "the end of the census.",
        cause="PRE", symptom="unknown", silent=False, found_by="the maintainer, the next day",
        reland=[], reland_kind="none", evidence_prs=[16874, 38392, 39053], evidence_issues=[13809, 35826], conf="low"),
    ("sglang", "sglang-11133"): dict(
        what="MTP speculative decoding with topk > 1 for Qwen3-Next and the FLA kernels ('Up to 6% faster'); it also "
             "commented out three tl.debug_barrier() calls in the causal_conv1d Triton kernels",
        stated=("rollback #16899", "The bug this pr fixes was introduced in 3451fc328, which commented out these "
                                   "debug barriers."),
        reason="The original kernels, adapted from vLLM, had the barriers 'due to bug in Triton compiler'. The kernels "
               "store a shifted conv state into the buffer that the same program loads, so without the barriers the "
               "loads and stores of one program race; issue #16714 reports a 'causal_conv1d_update bug with padded "
               "batches'. 40 of the 244 causal_conv1d unit tests failed, but CI passed, because the test file did "
               "not emit its exit code. 73 days later, #16899 restored the barriers; the rest of the PR stayed. "
               "warpDRF finds this race in the update kernel when the guarantee barrier.orders_global is granted "
               "(run of 2026-10-13).",
        cause="KERNEL", symptom="wrong output", silent=True,
        found_by="a user report (#16714), 73 days after the merge; the unit tests had failed, but CI did not report it",
        reland=[], reland_kind="not needed", evidence_prs=[16899], evidence_issues=[16714], conf="high"),
    ("sglang", "sglang-16879"): dict(
        what="moves the diffusion timestep embedding to a JIT kernel (with a performance comparison on Wan2.1)",
        stated=("revert #17257", "It caused AMD CI error"),
        reason="AMD CI failed after the merge (a later red run came from a commit before the revert). The failure "
               "and its mechanism are not described beyond the AMD CI error. It did not come back in this form; the "
               "old sgl-kernel time-embedding kernel was later deleted (#17278).",
        cause="BUILD", symptom="CI failure", silent=False, found_by="post-merge AMD CI, the same day",
        reland=[], reland_kind="none", evidence_prs=[17278], evidence_issues=[], conf="medium"),
    ("sglang", "sglang-17241"): dict(
        what="fuses the q and k norms in Flux2Attention (to cut the denoising time per step)",
        stated=("revert #17332", "First, not implementing a fallback when applying qk_norm will break the AMD diffusion "
                                 "CI. Second, the decision to use qknorm should be based on whether "
                                 "`can_use_fused_inplace_qknorm` returns true, rather than being applied directly. "
                                 "Third, converting non-contiguous q and k tensors directly to contiguous here is a "
                                 "performance-sensitive operation and should not be done."),
        reason="The maintainer gave four reasons: the fused qk norm had no fallback, which broke the AMD diffusion CI; "
               "the fused path should be chosen by can_use_fused_inplace_qknorm; a .contiguous() on non-contiguous q "
               "and k is a costly copy; and the benchmark had no warmup. The author answered that apply_qk_norm "
               "already checks can_use_fused_inplace_qknorm and has a fallback, and that the change had passed CI. "
               "It did not come back; the maintainer planned a refactor of the qk-norm logic.",
        cause="BUILD", symptom="CI failure", silent=False, found_by="post-merge AMD diffusion CI, the same day",
        reland=[], reland_kind="none", evidence_prs=[], evidence_issues=[], conf="low"),
    ("sglang", "sglang-17309"): dict(
        what="sets the FP4 GEMM backend to auto on SM100 so that FlashInfer picks it (and renames the flag with a "
             "flashinfer_ prefix); it cites performance data for CUDA 13",
        stated=("rollback #17369", "After manually installing nvidia-cudnn-cu13==9.18.0.77, it still seems to be worse "
                                   "(mm_fp4 picks this when cudnn >= 9.15.xx and CUDA 13, so hold off on this change "
                                   "for now (otherwise API will pick incorrect backend)"),
        reason="With auto, FlashInfer's mm_fp4 picks the cuDNN FP4 GEMM when cuDNN is 9.15 or newer and CUDA is 13, "
               "and that backend was slower, also with cuDNN 9.18 (FlashInfer issue #2375). The author rolled back to "
               "the CUTLASS backend 12 days after the merge. No reland was found.",
        cause="PERFREG", symptom="slower", silent=False, found_by="the author, 12 days after the merge",
        reland=[], reland_kind="none", evidence_prs=[], evidence_issues=[], conf="high"),
    ("sglang", "sglang-16043"): dict(
        what="fuses the gather of k_fp8 and k_scale into one Triton kernel in get_topk_ragged (the DeepSeek-V3.2 "
             "indexer), to cut concatenations",
        stated=("comment on #16043", "after this PR was merged, DeepSeek-v3.2 encounters the following error when "
                                    "running with a long context (> 65,536 tokens): Triton Error [HIP]: Code: 1, "
                                    "Messsage: invalid argument. This occurs because the Triton kernel grid is "
                                    "configured as grid = (seq_num, max_seq_len)."),
        reason="The fused kernel launches grid = (seq_num, max_seq_len). The y dimension of a launch grid is limited "
               "to 65,535, so the launch fails for contexts of more than 65,535 tokens; a user hit it on AMD "
               "('invalid argument'), and the maintainer also found errors with very long contexts. It was reverted "
               "five days after the merge and did not come back. warpDRF finds this launch-limit violation at the "
               "call site (run of 2026-10-13), but its reviewer filter drops grid limits.",
        cause="PRE", symptom="crash", silent=False,
        found_by="a user on AMD with a long context, and the maintainer, five days after the merge",
        reland=[], reland_kind="none", evidence_prs=[], evidence_issues=[], conf="high"),
    ("sglang", "sglang-18611"): dict(
        what="a GPU buffer pool for the layerwise offload of DiT weights, to reuse allocations across layers and "
             "steps; it reported a lower peak GPU memory (23.10 GB to 21.79 GB on Wan2.2-T2V-A14B)",
        stated=("revert #18866", "While the original PR reported decreased 'Peak GPU memory' metrics, this reduction "
                                 "is misleading as it only reflects PyTorch's allocated memory tracking, not actual "
                                 "VRAM consumption."),
        reason="The pool keeps up to 2 x prefetch_size buffers per tensor size on the GPU. "
               "torch.cuda.max_memory_allocated() went down, but torch.cuda.max_memory_reserved(), the physical VRAM "
               "that the process holds, went up. Thus the reported gain was a measurement artifact. The author "
               "agreed after a new test: the peak VRAM stays the same, and only the allocated memory decreases.",
        cause="PERFREG", symptom="more memory", silent=False, found_by="the maintainer, the next day",
        reland=[], reland_kind="none", evidence_prs=[], evidence_issues=[], conf="high"),
    ("sglang", "sglang-18905"): dict(
        what="a JIT version of the per_token_group_quant_8bit kernel (faster than the AOT kernel in its benchmark)",
        stated=("revert #19131", "Reverts sgl-project/sglang#18905 due to consistent failure"),
        reason="The piecewise CUDA graph test failed in CI after the merge; the author first thought that main had "
               "failed before the PR too. The fix #19118 ('Fix Dynamo tracing in per_token_group_quant_8bit') shows "
               "the cause: Dynamo could not trace the JIT kernel call under torch.compile. The re-land #19140 merged "
               "with that fix five days later.",
        cause="GRAPH", symptom="test failure", silent=False,
        found_by="post-merge CI (piecewise CUDA graph test), the next day",
        reland=[19140], reland_kind="fixed", evidence_prs=[19118, 19140], evidence_issues=[], conf="medium"),
    ("sglang", "sglang-17953"): dict(
        what="two-batch overlap (TBO) for MoRI expert-parallel communication on AMD: an async API for low latency, "
             "and several HIP streams that overlap communication and computation",
        stated=("comment on #17953", "I need to revert this PR in #19161 since it will break current CI"),
        reason="The DeepEP CI failed after the merge. The fix commit in the reland #19216 ('fix deepep ut regression') "
               "shows the cause: the change set state.num_tokens in the shared two-batch-overlap state for every "
               "all-to-all backend, so the DeepEP path ended with an extra key, and the state check raised "
               "'Unexpected keys when clearing'. The fix sets it only when the backend is MoRI. The reland merged "
               "three days later.",
        cause="BUILD", symptom="test failure", silent=False, found_by="post-merge CI (DeepEP tests), two days after",
        reland=[19216], reland_kind="fixed", evidence_prs=[19216], evidence_issues=[], conf="high"),
    ("sglang", "sglang-19190"): dict(
        what="removes CPU-GPU sync points (pinned non-blocking copies, output_size in repeat_interleave, GPU-side "
             "index builds) and adds the prefill CUDA graph plumbing for DP attention",
        stated=("revert #19581", "The sync patch removes GPU synchronization points in logits_processor.py "
                                 "(pin_memory + non_blocking transfers, output_size in repeat_interleave). This "
                                 "roughly doubles the baseline KL divergence from ~0.001 to ~0.002+, pushing it past "
                                 "the test threshold of 0.002."),
        reason="test_swa_radix_cache_kl became flaky on H200. A bisect showed that the PR doubled the KL divergence "
               "(decode 0.00086 to 0.00272), and with the revert 10 of 10 local runs passed. The split reland #19639 "
               "brought back everything except the logits_processor.py changes, which it names as the cause. How the "
               "pinned non-blocking copies changed the results is not stated. Two attempts to bring the "
               "logits-processor part back (#19640, and #19646 with a median KL in the test) were closed.",
        cause="GRAPH", symptom="lower accuracy", silent=True,
        found_by="post-merge CI (a flaky KL test), then a bisect, one day after the merge",
        reland=[19639], reland_kind="split", evidence_prs=[19639, 19640, 19646], evidence_issues=[], conf="medium"),
    ("sglang", "sglang-18619"): dict(
        what="tensor parallelism for Qwen-Image-Edit-2511, which 'improves inference speed and saves memory'",
        stated=("revert #19510", "It break torch compile"),
        reason="With torch.compile on, generation became slower after the PR; the revert shows the denoising time "
               "per step before and after. Other testers saw the opposite on H100 (0.2995 to 0.2543 s per step). "
               "Later, the PR was also found to break Qwen-Image with Nunchaku (fixed in #21415). The mechanism of "
               "the torch.compile problem is not stated.",
        cause="GRAPH", symptom="slower", silent=False, found_by="the maintainer, 14 days after the merge",
        reland=[], reland_kind="none", evidence_prs=[21415], evidence_issues=[], conf="low"),
    ("sglang", "sglang-12787"): dict(
        what="a TRT-LLM MNNVL fused all-reduce + residual + RMSNorm backend through FlashInfer's unified fusion API, "
             "to cut multi-GPU decode latency on NVL systems",
        stated=("revert #20792", "Reverts sgl-project/sglang#12787, as requested by @Fridge003"),
        reason="MNNVL all-reduce fusion hung; the reland #23402 says that the hangs correlated with piecewise CUDA "
               "graph capture and replay. FlashInfer #3304 fixed the cause in its MNNVL all-reduce kernel: the kernel "
               "compared the polled value with the -0.0 sentinel by a floating-point comparison, and flush-to-zero "
               "turned a negative subnormal input into -0.0, so the kernel polled forever. The reland first disabled "
               "piecewise CUDA graphs for MNNVL, and merged about 90 days later, once SGLang used FlashInfer 0.6.12, "
               "which has the fix.",
        cause="KERNEL", symptom="hang", silent=False, found_by="the maintainers, the same day",
        reland=[23402], reland_kind="fixed", evidence_prs=[23402], evidence_issues=[], conf="high"),
    ("sglang", "sglang-18801"): dict(
        what="a CuteDSL FP4 dense GEMM backend for mm_fp4, 'quite faster than the original CUTLASS backend' with "
             "autotune",
        stated=("", ""),
        reason="The PR needed an unreleased FlashInfer ('requires unreleased Flashinfer') but merged, because the "
               "backend is not selected by default. Two days later a CI job failed, and the revert links only that "
               "job. It came back as #23590 ('Reland Cute-DSL FP4 dense GEMM') after the FlashInfer release.",
        cause="DEP", symptom="CI failure", silent=False, found_by="post-merge CI, two days after the merge",
        reland=[23590], reland_kind="after dependency", evidence_prs=[23590], evidence_issues=[], conf="medium"),
    ("sglang", "sglang-14162"): dict(
        what="FP8 in place of BF16 for the DeepEP low-latency dispatch of DeepSeek-R1-0528 W4A8 (the BF16 dispatch "
             "doubled the communication latency), with a per-token to per-tensor quant kernel",
        stated=("", ""),
        reason="The revert gives no reason, and no comment or linked issue explains it. The reland #22316 merged 10 "
               "days later with the same description, so what changed is not stated.",
        cause="UNK", symptom="unknown", silent=False, found_by="unknown",
        reland=[22316], reland_kind="relanded", evidence_prs=[22316], evidence_issues=[], conf="low"),
    ("sglang", "sglang-18941"): dict(
        what="TTL-based prefix pinning in HiRadixCache for agentic workloads: pinned prefixes survive LRU eviction "
             "(backed up to host memory) and expire after a TTL that each cache hit refreshes",
        stated=("rollback #21884", "Will approach this a different way in follow up PRs"),
        reason="No failure. The author removed the hard pin a month after the merge to take another approach; a later "
               "'soft retention' PR (#27970, priority decay through retention_seconds) follows that direction and is "
               "open.",
        cause="PROC", symptom="none", silent=False, found_by="the author, 30 days after the merge",
        reland=[], reland_kind="none", evidence_prs=[27970], evidence_issues=[], conf="high"),
    ("sglang", "sglang-20501"): dict(
        what="a Triton kernel that fuses the temperature division and the softmax in sampling, to save a kernel "
             "launch and global-memory passes in each decode step",
        stated=("", ""),
        reason="The revert gives no reason. The reland #22166 says that the PR 'failed in CI mainly due to numerical "
               "tolerance': it splits the test references into a strict fp32 one and a looser native one, relaxes the "
               "row-sum tolerance, and removes unstable cases. That reland was closed without a merge. A later PR "
               "(#41050) fuses the two steps in the FlashInfer sampler path, and it is open.",
        cause="NUM", symptom="test failure", silent=False, found_by="post-merge CI, the next day",
        reland=[], reland_kind="none", evidence_prs=[22166, 41050], evidence_issues=[], conf="medium"),
    ("sglang", "sglang-21766"): dict(
        what="JIT activation kernels; the gain comes 'mostly from PDL and vectorization'",
        stated=("revert #22078", "Reverts sgl-project/sglang#21766 for this failure: <a CI job link>"),
        reason="A CI job failed the day of the merge. The reland #22094 says what was fixed: the 'num_token = 0 "
               "case'; the kernel did not handle an empty input. It came back 22 days later.",
        cause="PRE", symptom="CI failure", silent=False, found_by="post-merge CI, the same day",
        reland=[22094], reland_kind="fixed", evidence_prs=[22094], evidence_issues=[], conf="high"),
    ("sglang", "sglang-22574"): dict(
        what="ModelOpt NVFP4 support for FLUX.1-dev in SGLang diffusion (22.95% faster denoising than BF16)",
        stated=("revert #22649", "#22574 failed ci: <a CI job link>"),
        reason="A CI job failed the day after the merge; the revert links only the job. The reland #22672 merged a "
               "day later; the earlier study read in it a fix that masks the global quantization flags for the "
               "second transformer of dual-transformer models (Wan2.2). The description of the reland does not state "
               "the fix.",
        cause="MODEL", symptom="CI failure", silent=False, found_by="post-merge CI, the next day",
        reland=[22672], reland_kind="fixed", evidence_prs=[22672], evidence_issues=[], conf="low"),
    ("sglang", "sglang-21734"): dict(
        what="cuts GPU kernel overhead under piecewise CUDA graphs with the inductor compiler for FP8 models: a "
             "stride-preserving view in apply_qk_norm, static FP8 quantization in plain PyTorch ops, and more",
        stated=("revert #23159", "#21734 introduces a non-recoverable `Memory access fault by GPU` on AMD CI that "
                                 "takes down 7+ tests across 14 model files."),
        reason="apply_qk_norm replaced q.reshape(-1, head_dim) with q.view(*q.shape[:-1], -1, head_dim). Under "
               "piecewise CUDA graph capture, q and k are often non-contiguous views from the QKV split, so the view "
               "keeps a stride that the q_norm and k_norm kernels on ROCm do not expect, and the kernels wrote out of "
               "bounds. AMD CI had failed on the PR before the merge too. A redo (#23227) merged a week after the "
               "revert.",
        cause="PRE", symptom="illegal memory access", silent=False,
        found_by="AMD CI, red on the PR before the merge and on main after it",
        reland=[23227], reland_kind="fixed", evidence_prs=[23039, 23227], evidence_issues=[], conf="high"),
    ("sglang", "sglang-25321"): dict(
        what="stops the trtllm_mla attention backend from allocating the workspace buffer of its parent class "
             "(flashinfer_mla), to save GPU memory",
        stated=("", ""),
        reason="The author reverted it the same day with no reason. The CI-monitor reports of that day "
               "cross-reference the revert, and a later issue about non-finite logits on Blackwell (#27185) links "
               "it, but no failure is described.",
        cause="UNK", symptom="unknown", silent=False, found_by="the author, the same day",
        reland=[], reland_kind="none", evidence_prs=[], evidence_issues=[27185], conf="low"),
    ("sglang", "sglang-26235"): dict(
        what="skips the full-vocabulary softmax in the EAGLE draft when topk = 1, because argmax(logits) equals "
             "argmax(softmax(logits)); the softmax cost about 43 us per call",
        stated=("revert #26358", "it caused DeepSeek-V3.2 + MTP GSM8K accuracy to collapse to ~0.04 with ~96% invalid "
                                 "output on the very next nightly run."),
        reason="On AMD MI35x with ROCm 7.2, the DeepSeek-V3.2 MTP nightly scored 0.035 with 96% invalid output; a "
               "manual run of the PR also failed, and the revert branch scored 0.975. The same day, the reland #26397 "
               "brought the change back with a gate that keeps it off on HIP. The cause is disputed: #26633 (closed) "
               "argued that the ROCm failure fired once and passed in isolation. The ROCm version that merged in "
               "September (#35872) uses a masked Triton argmax, because AITER's greedy_sample 'has an unmasked "
               "vector-tail correctness issue' that returned an index outside the vocabulary (vocabulary size "
               "151,666), which fits the invalid outputs. The earlier study's reading (argmax tie-breaking on ROCm) "
               "is not supported by these texts.",
        cause="KERNEL", symptom="lower accuracy", silent=True,
        found_by="the AMD nightly accuracy test (DeepSeek-V3.2 MTP, MI35x), the next day",
        reland=[26397], reland_kind="modified", evidence_prs=[26397, 26633, 35872], evidence_issues=[], conf="low"),
    ("sglang", "sglang-25910"): dict(
        what="combines the ViT calls for the images of different requests in one batch",
        stated=("revert #26442", "it has been breaking AMD CI on main continuously since merge, with three distinct "
                                 "VLM symptoms all rooted in the same line it introduced. The pre-merge AMD CI on "
                                 "#25910 had already flagged both the crash and the perf regression"),
        reason="The new batched path split the batched image embedding by the placeholder token counts. This assumes "
               "that the embedding has as many rows as the placeholders, but several VLMs return more rows when "
               "items are batched across requests (MiniCPM-V: 1024 against 640; Qwen2.5-VL: 768 against 384). The "
               "result was three failures on AMD CI: a MiniCPM-V-2_6 crash in torch.split ('split_with_sizes expects "
               "split_sizes to sum exactly to 1024'), a MiniCPM-V-2_6 accuracy drop (0.4422 to 0.2522), and a 3x "
               "lower Qwen2.5-VL-3B throughput (about 2700 to 870 tok/s). The pre-merge AMD CI had shown the crash "
               "and the regression. It did not come back.",
        cause="PRE", symptom="crash", silent=False,
        found_by="AMD CI, red before the merge and on main for a week after it",
        reland=[], reland_kind="none", evidence_prs=[], evidence_issues=[], conf="high"),
    ("sglang", "sglang-27659"): dict(
        what="reuses one maximum-size output buffer across the capture sizes of the breakable CUDA graph runner, "
             "with sliced views for the smaller sizes, to save GPU memory",
        stated=("", ""),
        reason="It was reverted the same day with no reason; the CI states of the revert show a red base PR test. A "
               "new version merged the next day with a typed ShapeKey (#27857, '[BCG] Share output buffers across "
               "capture sizes + typed ShapeKey'). What failed is not stated.",
        cause="UNK", symptom="CI failure", silent=False, found_by="CI, the same day",
        reland=[27857], reland_kind="modified", evidence_prs=[27857], evidence_issues=[], conf="low"),
    ("sglang", "sglang-28166"): dict(
        what="fuses FLUX's tanh-GELU into the up-projection GEMM through the cuBLASLt GELU epilogue, to remove a GELU "
             "kernel and an HBM round trip",
        stated=("revert #28708", "follow-up checks on diffusion GELU epilogue fusion showed that the fused path can "
                                 "move generated images away from the original model behavior. For now we want the "
                                 "runtime to stay closer to the unfused/reference model semantics."),
        reason="No CI failure ('there is no acc issue in CI'). Checks after the merge showed that the cuBLASLt GELU "
               "epilogue changes the generated images relative to the unfused reference. The fusion came back in "
               "August behind a quality setting (#33536, #33819).",
        cause="NUM", symptom="wrong output", silent=True, found_by="the maintainer, at checks four days after the merge",
        reland=[33536], reland_kind="modified", evidence_prs=[33536, 33819], evidence_issues=[], conf="high"),
    ("sglang", "sglang-29142"): dict(
        what="swaps the dual-stream order in DeepSeek's MoE so that the routed experts run on the main stream and "
             "the shared experts on the alternate stream, the order that motivated the optimization (about 1 us per "
             "layer, 1% in target verify)",
        stated=("reland #29463", "The hazard is a host-side mutation of the input tensor's metadata between two CUDA "
                                 "kernel launches inside the captured decode graph."),
        reason="The revert (#29452) gives no reason. The reland #29463 says that the PR dropped the gsm8k score of "
               "test_moe_ep_extra.py from about 0.7 to 0.005, and explains why. The PR put the routed call before the "
               "shared-expert launch in Python order. Inside the routed call, dispose_tensor() calls "
               ".set_(torch.empty(0)) on the original hidden_states, so the shared-expert GEMM that launched next "
               "captured data_ptr() == 0 into the decode CUDA graph, and at each replay it read from a null address. "
               "The reland keeps the new stream assignment and launches the shared expert first; 'source-order "
               "alone is sufficient'.",
        cause="GRAPH", symptom="lower accuracy", silent=True,
        found_by="post-merge CI (test_moe_ep_extra.py), the same day",
        reland=[29463], reland_kind="fixed", evidence_prs=[29463], evidence_issues=[], conf="high"),
    ("sglang", "sglang-29702"): dict(
        what="registers RMSNorm as an SGLang custom op under piecewise CUDA graphs in place of the old C++ FlashInfer "
             "kernel; the operator is 20% faster for 16k tokens and an 8k hidden size",
        stated=("", ""),
        reason="It was reverted two days after the merge with no reason; the CI-monitor report of that day "
               "cross-references the revert. The failure is not described. It did not come back.",
        cause="UNK", symptom="unknown", silent=False, found_by="unknown (two days after the merge)",
        reland=[], reland_kind="none", evidence_prs=[], evidence_issues=[], conf="low"),
    ("sglang", "sglang-30310"): dict(
        what="does not allocate indexer KV memory on the layers that skip top-k (indexShare), so the KV cache pool "
             "grows by 15% and each rank saves 18 GB (GLM-5.2 NVFP4)",
        stated=("revert #30472", "This causes an IMA, and reverting it fixes the crash"),
        reason="The change caused an illegal memory access, and the revert fixed it the same day; the mechanism is "
               "not stated. A reapply (#30528) was closed. A narrower change merged in August (#30531, '[DSA] Skip "
               "indexer KV cache for skip-topk layers').",
        cause="UNK", symptom="illegal memory access", silent=False, found_by="a user, the same day",
        reland=[30531], reland_kind="modified", evidence_prs=[30528, 30531], evidence_issues=[], conf="medium"),
    ("sglang", "sglang-28527"): dict(
        what="AMX optimizations for diffusion on CPU: an AMX attention backend, AMX weight-packed linear layers, and "
             "channels-last 3D for the VAE (2x to 10x faster stages on Wan2.2)",
        stated=("revert #30716", "That change adds a second unconditional process_weights_after_loading() pass in the "
                                 "diffusion FSDP loader. The hook is not idempotent for quantized weights, and the "
                                 "added device_loading_context also breaks FSDP inference by attempting to restore "
                                 "CUDA-backed parameters to CPU storage."),
        reason="The CPU change edited the shared diffusion FSDP loader: it added a second, unconditional "
               "process_weights_after_loading() pass, which is not idempotent for quantized weights, and a "
               "device_loading_context that tried to restore CUDA parameters to CPU storage. FSDP inference on "
               "NVIDIA failed in the multimodal-gen 2-GPU test with a CPU/CUDA storage mismatch. The maintainer asked "
               "that all NVIDIA multimodal-gen tests pass before such a merge. A new version (#30719) is open.",
        cause="BUILD", symptom="crash", silent=False, found_by="post-merge CI (multimodal-gen 2-GPU test), the next day",
        reland=[], reland_kind="none", evidence_prs=[30719], evidence_issues=[], conf="high"),
    ("sglang", "sglang-30835"): dict(
        what="keeps page size 1 for FA4 when the full prefill CUDA graph is on, in place of the forced page size "
             "128, so that the full prefill graph can run",
        stated=("revert #31062", "It breaks CI"),
        reason="The three-line server-argument rule merged without CI; a CPU unit-test job (base-a-test-cpu) of a "
               "later PR then failed with exit code 255. The new condition runs only after an SM100 check, so a CPU "
               "job reaches it only through a unit test of the rule. The job log has expired, and no product defect "
               "was shown. It did not come back.",
        cause="TEST", symptom="CI failure", silent=False, found_by="the CI of a later PR (#29981), the same day",
        reland=[], reland_kind="none", evidence_prs=[29981], evidence_issues=[], conf="low"),
    ("sglang", "sglang-33026"): dict(
        what="fixes two TCP-layer TTFT stalls in the Rust server, for example about 200 ms on large request bodies, "
             "with a 16 MiB SO_RCVBUF on the listener",
        stated=("reland #33269", "tokio::net::TcpSocket::listen() needs a reactor, so #33026 moved listen() from the "
                                 "synchronous runtime::start() onto the async api runtime, and tests connecting right "
                                 "after start() returned raced it (ECONNREFUSED)."),
        reason="The revert gives no reason; the reland explains it. To set SO_RCVBUF, the PR moved listen() onto the "
               "async runtime, so runtime::start() could return before the socket listened. The tests in "
               "runtime::tests connect right after start() returns, and they failed with ECONNREFUSED in the "
               "test_cargo_workspace CI job. The reland builds the listener with socket2, which sets SO_RCVBUF before "
               "bind and stays synchronous, so start() binds and listens before it returns.",
        cause="API", symptom="test failure", silent=False, found_by="post-merge CI (test_cargo_workspace), the next day",
        reland=[33269], reland_kind="fixed", evidence_prs=[33269], evidence_issues=[], conf="high"),
    ("sglang", "sglang-32994"): dict(
        what="uses FlashInfer's fused RMSNorm + quant kernels on SM90, SM100 and SM120 (1.5% to 2.5% faster on "
             "H100, 4% to 5% on B200)",
        stated=("comment on revert #33455", "this CI run didn't build the latest sgl-kernel, the assert that was hit "
                                            "should not even trigger on the latest kernel as I had modified it"),
        reason="CI failed on an assert in sgl-kernel. The PR had changed that assert in the kernel package, but the "
               "runtime tests ran against the released sgl-kernel, so the old assert fired. The change came back in "
               "two parts: the kernel first (#33469), then the runtime (#33471).",
        cause="API", symptom="CI failure", silent=False, found_by="post-merge CI, the same day",
        reland=[33471], reland_kind="split", evidence_prs=[33469, 33471], evidence_issues=[], conf="high"),
    ("sglang", "sglang-33623"): dict(
        what="fuses the MLA gate projection into the QKV-A GEMM for Kimi-K3 (one merged projection; tile_m 32 for the "
             "merged shape)",
        stated=("revert #34642", "Breaks CI ... and seems like a real regression given full 2048 tokens"),
        reason="CI failed the day after the merge, and the outputs ran to the full 2,048 tokens, which points to "
               "degenerate output that does not stop. The cause is not stated. It did not come back.",
        cause="UNK", symptom="wrong output", silent=True, found_by="post-merge CI, the next day",
        reland=[], reland_kind="none", evidence_prs=[], evidence_issues=[], conf="low"),
    ("sglang", "sglang-31323"): dict(
        what="pre-populates a persistent top-k buffer with the shared-expert columns, so that the aiter grouped-topk "
             "kernel writes only the routed columns and one append kernel per layer goes away (GLM-5.2 on AMD; "
             "bit-identical results)",
        stated=("comment on revert #35105", "Revert this to unblock the CI"),
        reason="The new code read the process-global ServerArgs to size the buffer, and the code-rule test "
               "test_global_config_read_ratchet.py forbids such reads, so it failed on main and blocked other PRs. "
               "The fix #35075 ('Read aiter topk sizing from schedule config') and a reland (#35113) were closed. No "
               "product defect was shown.",
        cause="TEST", symptom="test failure", silent=False, found_by="post-merge CI, the same day",
        reland=[], reland_kind="none", evidence_prs=[35075, 35113], evidence_issues=[], conf="high"),
    ("sglang", "sglang-29525"): dict(
        what="a DeepEP v2 (ElasticBuffer) all-to-all backend with a fixed per-rank capacity, so that the MoE decode "
             "path can be captured in CUDA graphs on every topology, also multi-node",
        stated=("reland #35634", "reverted by #35568 because its e2e test used an unregistered CI runner name"),
        reason="The revert gives no reason; the reland explains it: the PR's end-to-end test named a CI runner that "
               "was not registered. The reland fixed the registration and merged eight days later. No product "
               "defect was shown.",
        cause="TEST", symptom="CI failure", silent=False, found_by="CI, the same day",
        reland=[35634], reland_kind="fixed", evidence_prs=[35634], evidence_issues=[], conf="high"),
    ("sglang", "sglang-37376"): dict(
        what="a Triton kernel that fuses silu_and_mul and per_token_group_quant_fp8 in the block-wise FP8 MoE path, "
             "to cut launch and memory-traffic overhead",
        stated=("revert #38381", "It is not numerically equivalent to the two-kernel path it replaces, and the "
                                 "difference is large enough to cost speculative-decoding acceptance on GLM-5.2-FP8 "
                                 "EAGLE MTP."),
        reason="The test test_dsa_glm52_tp_mtp.py::test_bs_1_speed asserts an acceptance length above 4.0. A bisect "
               "on 8-GPU H200 runs showed the acceptance length fall from 4.727 at the parent to 3.949 with the PR, and "
               "the speed from 323 to 265 tokens/s. The fused kernel is not numerically equal to the two kernels that "
               "it replaces. It did not come back.",
        cause="NUM", symptom="slower", silent=False, found_by="post-merge CI (the GLM-5.2 MTP speed test), the next day",
        reland=[], reland_kind="none", evidence_prs=[], evidence_issues=[], conf="high"),
    ("sglang", "sglang-36752"): dict(
        what="publishes the final draft graph's shared-read completion event in multi-layer EAGLE, so that the "
             "overlap scheduler can wait on it in place of a broader forward-stream fence",
        stated=("", ""),
        reason="The author reverted it seven days after the merge with no reason; the CI states on the revert show "
               "failed runs, but no failure is described. The change narrows a stream fence, so a read that the new "
               "event does not cover would race with the next writes, but nothing in the record shows that this "
               "happened.",
        cause="UNK", symptom="unknown", silent=False, found_by="the author, seven days after the merge",
        reland=[], reland_kind="none", evidence_prs=[], evidence_issues=[], conf="low"),
    ("sglang", "sglang-28417"): dict(
        what="enables piecewise CUDA graphs (capture and replay driven by torch.compile) on NPU behind an opt-in "
             "variable, with traceable custom ops for the NPU helpers",
        stated=("revert #40895", "NPU will support BCG only, PCG support will be dropped"),
        reason="No failure. The NPU maintainers decided to support only breakable CUDA graphs (BCG) and to drop PCG, "
               "so the PR was reverted the day it merged.",
        cause="PROC", symptom="none", silent=False, found_by="the NPU maintainers, the same day",
        reland=[], reland_kind="none", evidence_prs=[], evidence_issues=[], conf="high"),
    ("sglang", "sglang-40445"): dict(
        what="writes K and V into the paged KV cache with one npu_scatter_pa_kv_cache call on the NPU FIA path, in "
             "place of two scatter kernels (about 11.1 us per layer), behind a default-off variable",
        stated=("", ""),
        reason="The NPU maintainer reverted it two days after the merge with no reason, and no comment or issue "
               "explains it.",
        cause="UNK", symptom="unknown", silent=False, found_by="unknown", reland=[], reland_kind="none",
        evidence_prs=[], evidence_issues=[], conf="low"),
    # ---------------------------------------------------------------- vLLM
    ("vllm", "vllm-12355"): dict(
        what="FlashAttention in the generic ViT multi-head attention layer (FA2 in #12355; FA3 through "
             "flash_attn_varlen_func in #12435)",
        stated=("revert #12445", "Unfortunately FA3 support on ViT isn't at a reliable state to be used so revert the "
                                 "related PRs to always use xformers or SDPA."),
        reason="Two problems on the day after the merge. A concurrent FA3 PR (#12093) removed flash_attn_func, which "
               "#12355 called, so CI broke; #12435 switched to flash_attn_varlen_func. Then llava-onevision crashed in "
               "its SigLIP attention: vLLM's FlashAttention build needs a head size that is a multiple of 32, and that "
               "vision tower has 72 (the PR itself notes that Qwen2-VL's head size of 80 does not work). Both PRs were "
               "reverted to xformers or SDPA. FA3 in the ViT came back in September 2025 (#24347) with a dispatch "
               "guard.",
        cause="PRE", symptom="crash", silent=False, found_by="CI and the maintainers, the day after the merge",
        reland=[24347], reland_kind="modified", evidence_prs=[12093, 24347], evidence_issues=[], conf="high"),
    ("vllm", "vllm-11589"): dict(
        what="Triton fused-MoE configs for block-quantized FP8 (DeepSeek-V3), adapted from SGLang",
        stated=("revert #12629", "previous PR for pulling in block configs also changed defaults for FP8 ... this "
                                 "broke L4 MoE since there was not enough SHM for the default configuration"),
        reason="Next to the block configs, the PR changed the default config of the non-block FP8 MoE kernel. That "
               "default needs more shared memory than an L4 GPU has, so MoE on L4 broke. The next day, only the "
               "default was reverted; the block configs stayed.",
        cause="BUDGET", symptom="crash", silent=False, found_by="the maintainers, the next day (L4)",
        reland=[], reland_kind="not needed", evidence_prs=[], evidence_issues=[], conf="high"),
    ("vllm", "vllm-14384"): dict(
        what="temporary changes that cut the CPU overhead of the rotary embedding in MLA on V1",
        stated=("revert #14471", "On current main we see: gsm8k ... exact_match 0.06"),
        reason="DeepSeek-Coder-V2-Lite on V1 scored 0.06 on gsm8k the day after the merge, against 0.77 with the "
               "revert. The fix #14540 states the cause: 'the cuda kernel needs the input to be continuous'; MLA called "
               "the in-place CUDA rotary kernel on strided q_pe and k_pe slices. The fix makes the input contiguous, "
               "and the optimization came back two days later.",
        cause="PRE", symptom="lower accuracy", silent=True, found_by="a maintainer, the day after the merge",
        reland=[14540], reland_kind="fixed", evidence_prs=[14540], evidence_issues=[], conf="high"),
    ("vllm", "vllm-12325"): dict(
        what="a QKVCrossParallelLinear layer that computes the cross-attention q and kv projections without oversized "
             "matrices (a drop-in module for Mllama, BART and Whisper)",
        stated=("rollback #14498", "Unfortunately, QKVCrossParallelLinear doesn't support quantization that needs "
                                   "extra weight and weight_attrs registration"),
        reason="The new layer mapped only the weight and the bias to the q and kv projections, so quantization methods "
               "that register more parameters (for example weight_scale) could not load Mllama with BNB "
               "quantization. No test showed the problem ('This didn't show up in any of the tests'). The rollback "
               "put Mllama back on a model-level implementation; a refactor that supports BNB 4-bit (#14545) followed "
               "three days later.",
        cause="API", symptom="crash", silent=False, found_by="a maintainer, two days after the merge",
        reland=[14545], reland_kind="fixed", evidence_prs=[14545], evidence_issues=[], conf="high"),
    ("vllm", "vllm-14778"): dict(
        what="removes and fuses memory copies in the Mamba2 layer during prefill (layer latency from 5 ms to 3 ms on "
             "Bamba-9B)",
        stated=("comment on #14778", "this breaks mamba2 based models, unfortunately"),
        reason="Bamba-9B scored 0 on gsm8k after the merge, against 0.22 with the revert. The author found the cause: "
               "the old loop did ssm_state[idx].copy_(...) with a scalar idx, which copies in place, but the new code "
               "indexes with an integer tensor, and 'the indexing semantics is different', so the in-place copy did "
               "not reach the cache. The re-attempt #14857 builds views with state_indices_tensor and merged six days "
               "later.",
        cause="LOGIC", symptom="lower accuracy", silent=True, found_by="users of Bamba, the day after the merge",
        reland=[14857], reland_kind="fixed", evidence_prs=[14857], evidence_issues=[], conf="high"),
    ("vllm", "vllm-14995"): dict(
        what="bucket padding on TPU: above a padding gap, the token count is padded in steps of the gap, not to the "
             "next power of 2 (300 tokens pad to 320, not 512)",
        stated=("rollback #15565", "We have found a correctness issue with padding_gap == 64. As a temporary fix, this "
                                   "PR ensures that the default padding for now is exponential (as it was before)."),
        reason="A correctness issue appeared with the default padding gap of 64 the day after the merge; the details "
               "were discussed on Slack ('See the slack, the issue is being discussed'). The rollback made exponential "
               "padding the default again, and the gap option stayed. The mechanism is not stated.",
        cause="UNK", symptom="wrong output", silent=True, found_by="the maintainers, the day after the merge",
        reland=[], reland_kind="not needed", evidence_prs=[], evidence_issues=[], conf="low"),
    ("vllm", "vllm-15511"): dict(
        what="cache hints (a cache modifier and an eviction policy) on the loads of the fused_moe Triton kernel, for a "
             "Triton change that was not yet released (triton-lang/triton#6278)",
        stated=("revert #15645", "Internal Triton PTX codegen error caught in later CI and other issues mentioned in "
                                 "issue thread."),
        reason="The hints needed an unreleased Triton ('The latest stable version of triton dont contains' the "
               "change). With the released Triton, the kernel failed to compile (issue #15619, 'Triton JIT Compile "
               "Regression from PR 15511'). The author explained that older Triton versions do not allow a cache "
               "modifier and an eviction policy on the same load, a check that Triton's main branch had removed. It "
               "did not come back.",
        cause="DEP", symptom="compile failure", silent=False,
        found_by="later CI and a user issue (#15619), the day after the merge",
        reland=[], reland_kind="none", evidence_prs=[], evidence_issues=[15619], conf="high"),
    ("vllm", "vllm-12591"): dict(
        what="a new Triton FlashAttention-2 kernel for ROCm with FP8 and variable-length sequences, 'slightly faster "
             "than the current one'",
        stated=("rollback #18226", "The kernel brought in in #12591 turned out to have performance issues, and broken "
                                   "support for FP8 quantized models."),
        reason="About three weeks after the merge, ROCm users saw prefill take several seconds for a very short "
               "sequence, and FP8 models broke. #15734 and #12591 'were supposed to be merged in the opposite order', "
               "so the Triton FA path of the ROCm backend was broken for a few commits (#17235). The rollback put back "
               "the performant kernel from the ROCm fork, which also supports FP8.",
        cause="PERFREG", symptom="slower", silent=False, found_by="ROCm users, about three weeks after the merge",
        reland=[], reland_kind="none", evidence_prs=[15734, 17235], evidence_issues=[], conf="medium"),
    ("vllm", "vllm-20062"): dict(
        what="makes the default cuda_graph_sizes follow max_num_seqs on V1, so that CUDA graphs cover every batch "
             "size",
        stated=("revert #20128", "Reverts #20062 as we would like to engage in further discussion on the \"right\" "
                                 "default behavior"),
        reason="No failure. After the merge, reviewers noted that the change doubled the CUDA graph capture time "
               "(from 512 to 1024 sizes) and so the start-up time, that CUDA graphs matter less at large batch sizes, "
               "and that the captured shape is the token count, not the sequence count. They reverted it to discuss "
               "the default. A new default, min(max_num_seqs * 2, 512), merged two weeks later (#20628).",
        cause="PROC", symptom="slower", silent=False, found_by="reviewers, the same day",
        reland=[20628], reland_kind="modified", evidence_prs=[20628], evidence_issues=[], conf="high"),
    ("vllm", "vllm-20762"): dict(
        what="faster non-blockwise FP8 CUTLASS MoE: the ab_strides and c_strides are precomputed again, and new "
             "kernels shuffle the hidden states, the input scales and the outputs",
        stated=("revert #21334", "This reverts commit 9fb2d22 to fix #21322"),
        reason="Llama-4 Maverick failed to start (issue #21322). The first error came from a stale build; after a "
               "recompile, the new shuffle kernel failed with 'the provided PTX was compiled with an unsupported "
               "toolchain', and the reporter found that the new .cu file was compiled without the CUDA arch flags "
               "(fix #21426). The revert unblocked Meta's internal work. The reapply (#23001) was closed, and #23045 "
               "later integrated the CUDA MoE permute and unpermute.",
        cause="BUILD", symptom="crash", silent=False, found_by="users (internal work at Meta), four days after the merge",
        reland=[23045], reland_kind="superseded", evidence_prs=[21426, 23001, 23045], evidence_issues=[21322],
        conf="high"),
    ("vllm", "vllm-20836"): dict(
        what="skips the C++ compilation of the symbolic-shape guards of torch.compile, which vLLM drops anyway: "
             "build_guards from about 5 s to 0.5 s",
        stated=("comment on #20836", "AttributeError: torch._dynamo.config.enable_cpp_symbolic_shape_guards does not "
                                     "exist"),
        reason="The flag torch._dynamo.config.enable_cpp_symbolic_shape_guards exists only in recent PyTorch, so with "
               "an older PyTorch (the vLLM CPU benchmark job of pytorch-integration-testing) setting it raised an "
               "AttributeError. It was reverted the next day; a version that works with older PyTorch merged two "
               "weeks later (#23048).",
        cause="DEP", symptom="crash", silent=False,
        found_by="PyTorch's integration testing (the vLLM CPU benchmark job), the next day",
        reland=[23048], reland_kind="fixed", evidence_prs=[23048], evidence_issues=[], conf="high"),
    ("vllm", "vllm-22538"): dict(
        what="a CUDA kernel for the gpt-oss activation (swiglu_oai)",
        stated=("revert #22948", "RuntimeError: Worker failed with error 'Unsupported activation: swigluoai. Only silu "
                                 "and swiglu_oai activations are supported.'"),
        reason="A PR that merged at about the same time renamed the activation and added a Marlin MoE path, and the "
               "merged code still used the old name, so gpt-oss failed at start-up 17 minutes after the merge ('the "
               "merge order case'). The fixed version (#22951) merged two days later.",
        cause="SKEW", symptom="crash", silent=False, found_by="a maintainer, 17 minutes after the merge",
        reland=[22951], reland_kind="fixed", evidence_prs=[22951], evidence_issues=[], conf="high"),
    ("vllm", "vllm-22521"): dict(
        what="uses AITER's rope ops in the RotaryEmbedding module on ROCm (lower TTFT and TPOT for Llama-3-8B)",
        stated=("revert #22956", "Requires more investigation."),
        reason="The revert gives no failure, only 'Requires more investigation'. The AITER rope came back 42 days "
               "later inside a ROCm upstreaming PR (#25135), which adds rocm_aiter_rope_ops.py.",
        cause="UNK", symptom="unknown", silent=False, found_by="unknown (four days after the merge)",
        reland=[25135], reland_kind="superseded", evidence_prs=[25135], evidence_issues=[], conf="low"),
    ("vllm", "vllm-20000"): dict(
        what="avoids a useless list-to-list conversion (to_py_obj) in tokenizer.decode, to speed up detokenization",
        stated=("revert #23396", "this optimization is not necessary anymore"),
        reason="No failure. The PR merged by accident ('Oops accidentally merged this PR'). Three weeks later, the "
               "author measured that decode_token itself was 28% faster, but that after optimizations in "
               "transformers the end-to-end gain was only about 0.2%, so the maintainers reverted it.",
        cause="PROC", symptom="none", silent=False, found_by="the maintainers, three weeks after the merge",
        reland=[], reland_kind="none", evidence_prs=[], evidence_issues=[], conf="high"),
    ("vllm", "vllm-22628"): dict(
        what="a YOCO-style prefill optimization for Gemma3n behind --kv-sharing-fast-prefill: the cross-decoder layers "
             "that reuse the shared KV cache skip prefill",
        stated=("partial revert #23897", "Multi-modal processor tests for H2OVLChatModel and "
                                         "KimiVLForConditionalGeneration are broken on trunk after merging #22628."),
        reason="After the merge, the multi-modal processor tests of other models failed when they ran after the "
               "Gemma3n test in one session: a TypeError in _LazyConfigMapping for H2OVL, and an OOM for Kimi-VL. A "
               "maintainer also found 'a bad interaction with PyTorch 2.8.0' that broke another CI test, and a LoRA "
               "user hit a torch.compile failure for Gemma3n (#23970). The author reverted the Gemma3n model changes "
               "the next day; they came back four weeks later in #24357, which 'addresses the CI test failures' but "
               "does not say how. The mechanism is not stated.",
        cause="UNK", symptom="test failure", silent=False, found_by="post-merge CI and maintainers, the next day",
        reland=[24357], reland_kind="fixed", evidence_prs=[24357], evidence_issues=[23970], conf="medium"),
    ("vllm", "vllm-24666"): dict(
        what="replaces the apply_w8a8_block_fp8_linear custom op with a class that Fp8LinearMethod runs (part of "
             "#24629)",
        stated=("revert #25607", "That commit is causing an illegal memory access when torch.compile is used with "
                                 "decode DBO"),
        reason="With torch.compile and decode DBO, DeepSeek-R1 hit an illegal memory access in DeepEP "
               "(internode_ll.cu). The refactor made Hopper use the CUTLASS kernel in place of Triton, and CUTLASS "
               "needs M to be a multiple of 4, so the compiled graph padded the input. The failure came 'when "
               "capturing the first cudagraph with a token count that is not a multiple of 4 (in this case 2)' "
               "(issue #25623, 'torch.compile padding causes IMA on Hopper + DBO'; also #25527, 'm must be divisible "
               "by 4'). The fixed version (#25696) merged a week later.",
        cause="PRE", symptom="illegal memory access", silent=False,
        found_by="a user deployment (llm-d wide EP), the next day; also issue #25527",
        reland=[25696], reland_kind="fixed", evidence_prs=[25696], evidence_issues=[25527, 25623], conf="high"),
    ("vllm", "vllm-27512"): dict(
        what="takes the input and output projections of the GDN (gated delta net) attention out of its custom op, so "
             "that torch.compile covers them, and adds a torch-native gated RMSNorm",
        stated=("", ""),
        reason="The author reverted it the same day with no reason. The earlier study found that CI had failed before "
               "the merge and was taken as unrelated, and that the only change in 'Attempt 2' (#28083, merged two days "
               "later) makes a module-level import of the FLA Triton norm lazy; the description of #28083 does not "
               "say so.",
        cause="BUILD", symptom="CI failure", silent=False, found_by="the author, the same day (CI red before the merge)",
        reland=[28083], reland_kind="fixed", evidence_prs=[28083], evidence_issues=[], conf="low"),
    ("vllm", "vllm-28012"): dict(
        what="busy-waits on the shm message queue from the main thread while it blocks on Future.get_result(), in "
             "place of a separate spinning thread that #26866 had added (a fix of a large performance regression)",
        stated=("revert #28289", "This PR #28012 is breaking PD deployments with TP>1."),
        reason="With PD disaggregation through the NixlConnector and TP 2, requests got stuck. The resubmit #28319 "
               "names the cause: 'a bug related to aggregating kv connector outputs which broke for example Nixl P/D "
               "for TP > 1'; its second commit fixes it, and it merged the same day as the revert.",
        cause="LOGIC", symptom="hang", silent=False, found_by="a maintainer, with a PD deployment, three days after the merge",
        reland=[28319], reland_kind="fixed", evidence_prs=[26866, 28319], evidence_issues=[], conf="high"),
    ("vllm", "vllm-26368"): dict(
        what="stores the output tokens as list[np.ndarray] in place of list[list[int]], so that fewer Python objects "
             "trigger the garbage collector at large batch sizes",
        stated=("revert #29121", "the original PR introduced throughput regression ... Although replacing list[int] to "
                                 "np.ndarray could avoid bumping gc allocation count, but the conversion overhead is "
                                 "way too big and would regress the throughput e2e."),
        reason="Two reverts. The first PR broke the 'V1 Test others' CI job on the day it merged and was reverted; the "
               "redo (#28771) passed the tests and merged the same day. Six days later, AMD measured a throughput "
               "regression (average latency 222.8 s against 192.9 s without the change), which the author confirmed; "
               "a fix (#29033) did not help, because the NumPy conversion costs more than the garbage collection "
               "saves. The change also broke out-of-tree backends (vllm-ascend, vllm-gaudi, tpu-inference) that read "
               "the sampled token ids. It did not come back.",
        cause="PERFREG", symptom="slower", silent=False,
        found_by="CI the same day; then an AMD benchmark six days after the redo",
        reland=[], reland_kind="none", evidence_prs=[28771, 29033], evidence_issues=[], conf="high"),
    ("vllm", "vllm-28841"): dict(
        what="moves the slice op before the all-reduce in GPT-OSS, so that the all-reduce + RMSNorm fusion matches "
             "again on B200",
        stated=("revert #29483", "The original PR #28841 breaks LoRA TP Test (Distributed)."),
        reason="The LoRA TP distributed test failed the day after the merge. With the slice before the all-reduce, the "
               "all-reduce got a non-contiguous tensor, and only one communicator had been changed to reshape it (the "
               "earlier study's reading). A later PR (#30647) removed the padding and slicing for GPT-OSS with the "
               "FlashInfer MXFP4 MoE.",
        cause="PRE", symptom="test failure", silent=False, found_by="post-merge CI (LoRA TP Test), the next day",
        reland=[30647], reland_kind="superseded", evidence_prs=[30647], evidence_issues=[], conf="medium"),
    ("vllm", "vllm-28971"): dict(
        what="FusedMoE LoRA through the Triton kernels for MXFP4 models (gpt-oss): the activation and the moe_sum are "
             "unfused so that the LoRA modules can be injected",
        stated=("revert #29697", "it is breaking the following tests from nightly"),
        reason="The new Triton LoRA MoE test failed in the nightly kernel tests on the day of the merge. The reland "
               "(#29708) merged two days later with the M = 32 and 64 shapes removed from the test (the earlier "
               "study's reading). A garbage-output issue under FULL CUDA graphs was known at the merge.",
        cause="TEST", symptom="test failure", silent=False, found_by="the nightly kernel tests, the same day",
        reland=[29708], reland_kind="smaller", evidence_prs=[29708], evidence_issues=[], conf="low"),
    ("vllm", "vllm-28271"): dict(
        what="removes the TP all-gather and re-slicing in split_qkv of the Qwen-VL vision attention, because "
             "head-parallel attention already keeps full q, k and v per rank (redundant synchronization)",
        stated=("partial revert #30542", "Fixes #30250, the model generates incorrect output when TP is used."),
        reason="Qwen2-VL with TP generated wrong output (issue #30250) a month after the merge. Qwen2-VL's ViT qkv_proj "
               "returns [s, b, 3 * head * head_dim], not [s, b, head * 3 * head_dim], so TP cannot shard the hidden "
               "states along the head dimension, and the removed gather was needed there; Qwen2.5-VL and Qwen3-VL are "
               "not affected. Only the Qwen2-VL part was reverted.",
        cause="MODEL", symptom="wrong output", silent=True, found_by="a user report (#30250), a month after the merge",
        reland=[], reland_kind="not needed", evidence_prs=[], evidence_issues=[30250], conf="high"),
    ("vllm", "vllm-29558"): dict(
        what="enables piecewise CUDA graphs by default for DeepEP high-throughput mode, with the MoE ops as split "
             "points (5.3% more throughput, 4.4% lower TTFT)",
        stated=("partial revert #30910", "There seems to be multiple issues here ... 1) torch.compile does not support "
                                         "Size as output ... 2) the outputs do not seem to have consistent addresses "
                                         "leading to garbage outputs"),
        reason="The H200 distributed tests broke (issue #30889). There were two problems. torch.compile does not "
               "support a Size value as a graph output, so the common MoE pattern 'orig_shape = hidden_states.shape' "
               "before the splitting op and '.view(orig_shape)' after it broke compilation. And the outputs did not "
               "keep constant addresses across the piecewise graph replays, which gave garbage. The PIECEWISE support "
               "for DeepEP HT was reverted 11 days after the merge; a fix for wrong output with MoE ops as piecewise "
               "split points (#61179) is open.",
        cause="GRAPH", symptom="wrong output", silent=True, found_by="post-merge H200 CI, 11 days after the merge",
        reland=[], reland_kind="none", evidence_prs=[30914, 61179], evidence_issues=[30889], conf="high"),
    ("vllm", "vllm-30746"): dict(
        what="runs the MLA prefill in FP8 on SM100 through FlashInfer's CUTLASS FMHA and TRT-LLM ragged kernels when "
             "the KV cache is FP8",
        stated=("revert #31197", "Revert #30746 untill #30993 is merged."),
        reason="The PR needed a newer FlashInfer ('This PR needs #30993 - Please use flashinfer top of tree for FP8 "
               "Prefill'), but it merged before that bump and broke the Blackwell CI tests. It was reverted the next "
               "day until the FlashInfer bump (#30993, v0.6.1) merged; it came back as an opt-in in #31195.",
        cause="DEP", symptom="CI failure", silent=False, found_by="post-merge Blackwell CI, the next day",
        reland=[31195], reland_kind="after dependency", evidence_prs=[30993, 31195], evidence_issues=[], conf="high"),
    ("vllm", "vllm-32339"): dict(
        what="makes FLASHINFER_MLA the default MLA decode backend on Blackwell, and TRT-LLM the default MLA prefill "
             "backend",
        stated=("revert #32484", "The FI prefill backend has correctness issues (#27684) which caused CI failure. Will "
                                 "make a separate PR to improve CI coverage as this should have been detected in the "
                                 "original PR."),
        reason="test_mla_backends.py failed in CI after the merge, and the revert blamed known correctness issues of "
               "the FlashInfer prefill backend (#27684). The root cause was in the test: it used unscaled random "
               "projection weights, which made the LSE merge numerically unstable. #32529 scaled the weights, and the "
               "same change came back two days later (#32615).",
        cause="TEST", symptom="CI failure", silent=False, found_by="post-merge CI (attention tests on B200), the next day",
        reland=[32615], reland_kind="same code", evidence_prs=[32529, 32615], evidence_issues=[27684], conf="high"),
    ("vllm", "vllm-30207"): dict(
        what="makes the NIXL connector use the contiguous cross-layer KV cache layout (RFC #27742), which cuts the "
             "fragmentation of transfer buffers (more than 2x tok/s and TTFT)",
        stated=("revert #33241", "Fully reverts #30207 and its #33052 follow-up."),
        reason="The revert gives no reason beyond the two PRs. Issue #33074 ('PD report DeepseekV32 AssertionError: "
               "num_kv_heads == 1') and the follow-up #33052 ('Fix Can't instantiate abstract class "
               "DeepseekV32IndexerBackend') show that the new layout broke DeepSeek-V3.2 in PD deployments. It came "
               "back a week later as #33339, and it is now off by default ('this time it is off by default').",
        cause="MODEL", symptom="crash", silent=False, found_by="users of PD with DeepSeek-V3.2, four days after the merge",
        reland=[33339], reland_kind="modified", evidence_prs=[33052, 33339], evidence_issues=[33074], conf="medium"),
    ("vllm", "vllm-33641"): dict(
        what="stores one compiled artifact per unique subgraph hash in place of one per subgraph index, to speed up "
             "the cold start of vLLM's torch.compile",
        stated=("", ""),
        reason="The author reverted it the next day with no reason. The revert cross-references two CI-failure issues "
               "of that day: #33816 (the Quantized Models test, test_gptq_marlin.py) and #33802 (the distributed 2xH100 "
               "tests, which the revert of the FA3 swizzle also names). The mechanism is not stated. A version that "
               "'stops compiling identical artifacts' merged three days later (#34003).",
        cause="UNK", symptom="CI failure", silent=False, found_by="post-merge CI, the next day",
        reland=[34003], reland_kind="modified", evidence_prs=[34003], evidence_issues=[33802, 33816], conf="low"),
    ("vllm", "vllm-23465"): dict(
        what="updates FA3 to include a new swizzle optimization of its scheduler (the vLLM side of "
             "vllm-project/flash-attention#82)",
        stated=("revert #33841", "As described in #33802, #23465 broke the Distributed Tests 2 GPUs (H100)."),
        reason="test_async_tp.py::test_async_pass_correctness failed in the distributed 2xH100 tests the day after the "
               "merge, and a first fix attempt (#33854) did not fix it. The reapply #34043 came back six days later "
               "'with correct metadata sizes': FA3 with the swizzle writes 1 + 4 * round_up(B, 4) scheduler-metadata "
               "slots, but the host buffer had 4 * B + 1, so the kernel wrote past the buffer.",
        cause="PRE", symptom="wrong output", silent=True,
        found_by="post-merge CI (distributed 2xH100 tests), the day after the merge",
        reland=[34043], reland_kind="fixed", evidence_prs=[33854, 34043], evidence_issues=[33802], conf="high"),
    ("vllm", "vllm-34492"): dict(
        what="fuses the qkvz and ba projections of the GDN layers of Qwen3.5, for speed",
        stated=("revert #34683", "It makes the output of qwen3 next as random symbols"),
        reason="Qwen3-Next, which shares the GDN code but was not tested, gave random symbols (gsm8k 0) the day after "
               "the merge (#34640). The redo #34697 states the root issue: Qwen3-Next's qkvz_proj output_sizes "
               "should be one fused size [sum(key_dim, key_dim, value_dim, value_dim)], not "
               "[sum(key_dim, key_dim, value_dim), value_dim]. The redo merged the next day.",
        cause="MODEL", symptom="wrong output", silent=True, found_by="a user report (#34640), the day after the merge",
        reland=[34697], reland_kind="fixed", evidence_prs=[34697], evidence_issues=[34640], conf="high"),
    ("vllm", "vllm-34582"): dict(
        what="stops forcing the NemotronH MoE router to fp32; the router took about 40% of the forward pass under "
             "normal workloads",
        stated=("revert #34808", "#34582 introduced an accuracy degradation."),
        reason="The LM-eval test for large models (#34704) caught a 4% to 5% accuracy drop for Nemotron 3 Nano FP8 "
               "(about 2% on a standalone GSM8K); the BF16 model also lost accuracy, but within the test tolerance. "
               "The router GEMM came back in #35121 as a bf16 GEMM with fp32 accumulation and output, with fp32 "
               "compute kept for NemotronH.",
        cause="NUM", symptom="lower accuracy", silent=True, found_by="post-merge CI (LM Eval Large Models), two days after",
        reland=[35121], reland_kind="superseded", evidence_prs=[34302, 34704, 35121], evidence_issues=[], conf="high"),
    ("vllm", "vllm-32805"): dict(
        what="keeps the MoE layer-name strings out of the torch.compile graph (they are popped from a list in the "
             "forward context), to cut the cold-start time",
        stated=("rollback #35475", "This supercedes the fast_moe_cold_start workaround for torch>=2.11"),
        reason="A planned replacement, not a failure. The workaround assumed that the custom MoE ops always run in "
               "order. It gave garbled output for ERNIE-4.5-VL (two experts, one of them used in decode), fixed by "
               "turning it off for that model (#35587), and it was turned off with speculative decoding (#33624). "
               "With torch 2.11, #35475 lifts the layer names as graph inputs (register_opaque_type with hoist=True) "
               "and drops the workaround; #38123 later allows strings in custom ops without slower compilation.",
        cause="NOTFAIL", symptom="none", silent=False, found_by="the author (planned)",
        reland=[], reland_kind="not needed", evidence_prs=[33624, 35587, 38123], evidence_issues=[29992], conf="high"),
    ("vllm", "vllm-35216"): dict(
        what="constructs the compilation passes once at VllmBackend init in place of at each call (one pass took more "
             "than 1 s to construct, issue #34635)",
        stated=("revert #37733", "Reverts #35216 see #37732"),
        reason="Issue #37732 found by a bisect that the PR 'causes garbage logits for prompts exceeding ~2048 tokens "
               "when using -O3 compilation level' (Kimi-K2.5 NVFP4 on GB200 with the qk-norm-rope fusion). The PR was "
               "reverted the same day. The mechanism is not stated. A later PR that pre-creates the FlashInfer "
               "AR+RMSNorm workspaces at backend init (#61268) is open.",
        cause="UNK", symptom="wrong output", silent=True, found_by="a user's bisect (#37732), the day after the merge",
        reland=[], reland_kind="none", evidence_prs=[61268], evidence_issues=[34635, 37732], conf="medium"),
    ("vllm", "vllm-38050"): dict(
        what="integrates FlashInfer's NVFP4 CuteDSL MoE kernels as a MoE backend",
        stated=("comment on #38050", "this causes flashinfer ci to fail since this isn't actually supported in the "
                                    "0.6.6 version we use. Please be more careful next time with testing"),
        reason="The pinned FlashInfer 0.6.6 does not support this CuteDSL MoE path, so the Kernels (B200) nightly test "
               "test_flashinfer_cutedsl_moe_masked crashed with a CUDA coredump in break_fp4_bytes / "
               "dequantize_nvfp4_to_dtype. The CI failure analyzer generated the revert. A CuteDSL batched-experts "
               "backend for NVFP4 MoE (#38251) merged 11 days later.",
        cause="DEP", symptom="crash", silent=False, found_by="the nightly CI (Kernels, B200), the next day",
        reland=[38251], reland_kind="modified", evidence_prs=[38251], evidence_issues=[], conf="high"),
    ("vllm", "vllm-37205"): dict(
        what="an optimized router GEMM kernel for gpt-oss (1% to 2% more output throughput at batch size 1)",
        stated=("revert #38778", "which is reported to cause accuracy issue for gpt-oss-120b"),
        reason="Two weeks after the merge, issue #38754 traced by manual bisection 'infrequent NaNs' to this kernel: "
               "in an internal multi-turn test with CUDA graphs, prefix caching and chunked prefill, the NaNs "
               "poisoned the EAGLE3 draft model, whose acceptance rate fell to zero. No root cause was found; the "
               "earlier study noted that the wrapper does not check the contiguity and alignment that its TMA "
               "descriptors assume (plausible, not confirmed). It did not come back.",
        cause="UNK", symptom="NaN", silent=True, found_by="a user's internal test (#38754), two weeks after the merge",
        reland=[], reland_kind="none", evidence_prs=[], evidence_issues=[38754], conf="low"),
    ("vllm", "vllm-40331"): dict(
        what="overlaps serial start-up work: background imports of torch and transformers, a weight prefetch, and a "
             "forkserver prewarm (cold start 11.5% to 20% faster)",
        stated=("revert #40438", "I think the following code is causing the recent main failures. ... After removing "
                                 "this code, the local test passed. with contextlib.suppress(Exception): import "
                                 "transformers"),
        reason="The background thread imported transformers while the main thread imported it too, so the two imports "
               "raced: 'ImportError: cannot import name GenerationConfig from transformers' (#40442, #40463), and "
               "test_truncation.py failed on main. Without the background import, the test passed. It was reverted "
               "the same day. Smaller parts were proposed later (#40439, #40440, #45483, #45485, #45501) and closed.",
        cause="LOGIC", symptom="crash", silent=False, found_by="users and post-merge CI, the same day",
        reland=[], reland_kind="none", evidence_prs=[40439, 40440, 45501], evidence_issues=[40442, 40463], conf="high"),
    ("vllm", "vllm-39917"): dict(
        what="replaces the shared-memory routing replay (routed-experts capture for RL) with a device cache and an "
             "async D2H pipeline that works with CUDA graphs, multi-node TP and DP",
        stated=("revert #42434", "the author from nemo RL has some urgent releases so might not be able to support "
                                 "prefix caching and other features very soon, so we all agree to use #39568 and "
                                 "revert #39917 for now."),
        reason="A user saw nearly 10% lower throughput on Qwen3.5-35B-A3B-FP8 (2x RTX 5090) after the merge, which "
               "#42148 fixed by skipping the routed-experts hot path when the feature is off. The revert a week later "
               "was a design decision: the maintainers chose the other RFC (#39568), because this design could not "
               "soon support prefix caching and other features.",
        cause="PROC", symptom="none", silent=False, found_by="the maintainers, a week after the merge",
        reland=[], reland_kind="none", evidence_prs=[39568, 42148], evidence_issues=[], conf="high"),
    ("vllm", "vllm-42686"): dict(
        what="backports pytorch/pytorch#176994 to PyTorch 2.11, to fix the performance problems of Inductor graph "
             "partition (issue #27828)",
        stated=("revert #42913", "AsyncTP Correctness Tests (2xH100) — async TP compiled output differs from baseline "
                                 "(seeded sampling mismatch); Sequence Parallel Correctness Tests (2xH100) — sequence "
                                 "parallel compiled output differs from baseline"),
        reason="The patch changes the behavior of torch.compile through env_override.py, and in the next nightly build "
               "the compiled outputs of the async-TP and sequence-parallel correctness tests differed from the "
               "baseline. The CI failure analyzer generated the revert. A reapply (#42980) was closed.",
        cause="GRAPH", symptom="test failure", silent=False, found_by="the nightly CI, the next day",
        reland=[], reland_kind="none", evidence_prs=[42980], evidence_issues=[27828], conf="medium"),
    ("vllm", "vllm-44173"): dict(
        what="rewrites silu_and_mul_per_block_quant with one warp per group, a warp-shuffle abs-max and vectorized "
             "I/O (the block-FP8 MoE path of DeepSeek)",
        stated=("revert #45066", "#44173 broke ROCm build, so let's revert it to unblock CI while we work on a proper "
                                 "fix in #45055"),
        reason="The kernel uses a 32-bit shuffle mask, but HIP has 64-wide wavefronts and needs a 64-bit mask (#45055: "
               "'shuffle mask dtype error'), so the ROCm build failed the day of the merge. The fix #45055 is still "
               "open, and a portable version (#56586) was closed.",
        cause="BUILD", symptom="build failure", silent=False, found_by="the AMD build, the same day",
        reland=[], reland_kind="none", evidence_prs=[45055, 45062, 56586], evidence_issues=[], conf="high"),
    ("vllm", "vllm-45309"): dict(
        what="reduces the eager breaks during CUDA graph capture in DeepSeek-V4 attention, so that the q "
             "up-projection with the KV insert, and the compressor on an auxiliary stream, run inside the captured "
             "graph (26.8% to 27.9% better E2E TTFT)",
        stated=("revert #45972", "Reverts #45309, which causes DeepSeek-V4 Flash to emit garbage output (a correctness "
                                 "regression)."),
        reason="A single-commit bisect showed that DeepSeek-V4 Flash with DP 4 and EP answered 'the the the the …' in "
               "place of 'Paris'. The revert does not state the mechanism. The removed code comment calls the moved "
               "ops 'metadata-dependent', and a later revert (#52836) says that the same narrow eager region "
               "corrupted output on Model Runner V1 (#52401 restored the wide region there). Thus a value of one step "
               "frozen in the graph is probable, but not confirmed.",
        cause="GRAPH", symptom="wrong output", silent=True, found_by="a maintainer's bisect, the next day",
        reland=[], reland_kind="none", evidence_prs=[51430, 52401, 52836], evidence_issues=[], conf="low"),
    ("vllm", "vllm-42996"): dict(
        what="adds programmatic dependent launch (PDL) to the DeepGEMM kernel launches",
        stated=("revert #45999", "This broke AMD build"),
        reason="The PDL launch attribute exists only on NVIDIA, so the ROCm build failed. The change came back the same "
               "day (#46006); the earlier study reads a ROCm guard in it, and its description is empty.",
        cause="BUILD", symptom="build failure", silent=False, found_by="the AMD build, the same day",
        reland=[46006], reland_kind="fixed", evidence_prs=[46006], evidence_issues=[], conf="medium"),
    ("vllm", "vllm-45850"): dict(
        what="pins the CPU tensors and the mmap region of the CPU KV offload in a background thread, for a faster "
             "start-up",
        stated=("comment on issue #46933", "I believe there is a race between _pin_cpu_tensors and CUDA graph captures."),
        reason="With CPU KV offloading, CUDA graph capture hung (issue #46933). The background thread called "
               "cudaHostRegister while the main thread captured CUDA graphs, and the call returned "
               "cudaErrorStreamCaptureUnsupported ('The operation is not permitted when the stream is capturing'). "
               "Joining the thread before the capture fixed it; the maintainers preferred the revert, five days after "
               "the merge. A pinning that waits on events per block was proposed as the proper solution.",
        cause="GRAPH", symptom="hang", silent=False, found_by="a user report (#46933), four days after the merge",
        reland=[], reland_kind="none", evidence_prs=[46955], evidence_issues=[46933], conf="high"),
    ("vllm", "vllm-44109"): dict(
        what="templates the RMSNorm CUDA and CPU kernels on HasWeight, so that weightless RMSNorm (FlashNorm-folded "
             "checkpoints) skips the wasted load and multiply of an all-ones weight",
        stated=("partial revert #46987", "On XPU, weightless RMSNorm (weight=None) ... falls back to the native eager "
                                         "implementation, which decomposes into many small aten ops (pow, mean, "
                                         "rsqrt, mul, add). At small batch sizes ... this fallback significantly "
                                         "increases the number of CPU-dispatched kernels and hurts TPOT."),
        reason="On XPU, with weight=None the layer fell back to the native eager implementation, which splits into "
               "many small aten ops; at small batch sizes the model is bound by host dispatch, so TPOT got worse. The "
               "XPU part was reverted 12 days after the merge; the XPU kernels support weightless RMSNorm since "
               "(vllm-xpu-kernels #441).",
        cause="PERFREG", symptom="slower", silent=False, found_by="the XPU maintainers, 12 days after the merge",
        reland=[], reland_kind="not needed", evidence_prs=[], evidence_issues=[], conf="high"),
    ("vllm", "vllm-48641"): dict(
        what="drops the fp32 copy of the logits in apply_sampling_params; with speculative decoding the copy reached "
             "several GiB (an OOM at warmup for Qwen3-8B with a dflash speculator)",
        stated=("revert #49033", "The MTP variant of Qwen3.5-397B-A17B-NVFP4 hung in sample_tokens (fatal TimeoutError: "
                                 "RPC call to sample_tokens timed out) while the identical non-MTP variant of the "
                                 "same model passed."),
        reason="The CI failure analyzer linked the PR to one nightly failure: the MTP variant of Qwen3.5-397B NVFP4 "
               "hung in sample_tokens with DP 2, while the non-MTP variant passed. Its explanation, that the in-place "
               "bf16 change 'plausibly corrupts spec-decode verification numerics and desyncs the DP=2 ranks', is a "
               "guess, and a maintainer at first doubted that the PR was the cause. It did not come back.",
        cause="UNK", symptom="hang", silent=False, found_by="the nightly CI (an automatic revert), three days after",
        reland=[], reland_kind="none", evidence_prs=[53630], evidence_issues=[], conf="low"),
    ("vllm", "vllm-48597"): dict(
        what="a 2,139-line set of Blackwell decode optimizations for GLM-5.2 (router GEMMs, CuTeDSL kernels, MTP "
             "fusion, PDL)",
        stated=("comment on #48597", "I've reverted this PR since it didn't go through the proper review process, and "
                                    "it includes some questionable design choices."),
        reason="No failure. The PR merged without the proper review, and the reviewer objected to some design "
               "choices. It came back as focused PRs (#49791, #49792, #49793, #50230).",
        cause="PROC", symptom="none", silent=False, found_by="a maintainer, the same day",
        reland=[49791, 49792, 49793, 50230], reland_kind="split", evidence_prs=[], evidence_issues=[], conf="high"),
    ("vllm", "vllm-48223"): dict(
        what="dual-stream decode on ROCm (the shared experts run on an auxiliary stream beside the routed experts), "
             "made compatible with HIP graphs; 3% to 4% lower TPOT with DP attention",
        stated=("reland #52033", "Qwen3.5 uses the same tensor for hidden state and shared expert inputs, and never "
                                 "creates any copies of either tensor throughout the MoE. This causes a race condition "
                                 "during multi-stream. Specifically, it uses the same tensors because it's not "
                                 "quantizing the routed experts' input."),
        reason="The AMD CI gsm8k test failed for Qwen3.5 with DP 2 and EP. The reland explains: when the routed "
               "experts are not quantized, the shared-expert input on the auxiliary stream is the same tensor as the "
               "hidden state of the routed path on the main stream, with no copy, so the two streams race; quantized "
               "routed experts copy their input first, so they do not alias. The reland turns the overlap off on ROCm "
               "when the routed input is not quantized, and it merged 17 days later. The texts do not name the "
               "kernel that writes the shared tensor.",
        cause="GRAPH", symptom="lower accuracy", silent=True,
        found_by="AMD CI (gsm8k, Qwen3.5 with DP 2 and EP), the same day",
        reland=[52033], reland_kind="fixed", evidence_prs=[52010, 52033], evidence_issues=[], conf="high"),
    ("vllm", "vllm-50004"): dict(
        what="sizes the C128A top-k metadata rows of DeepSeek-V4 by the current context (active_topk_width) in place "
             "of the maximum, so that the sparse decode kernel loops fewer times (1.0% E2E throughput)",
        stated=("revert #51318", "The C128A metadata builder runs before FULL CUDA graph replay, but the sparse decode "
                                 "consumer is captured and reuses its capture-time row layout. ... Row 0 still started "
                                 "at offset 0, but later rows could be read from stale offsets."),
        reason="Under concurrent requests at long contexts, DeepSeek-V4 output was corrupted (#52448; unbalanced "
               "reasoning tags, #48089). The metadata builder runs outside the CUDA graph and wrote rows with a stride "
               "that depends on the batch, while the captured consumer kept the stride from capture time, so the rows "
               "after the first were read at stale offsets. It was reverted 19 days after the merge, and it came back "
               "in #52823, which keeps capacity-strided slices, asserts that the strides match, and adds a CUDA graph "
               "test.",
        cause="GRAPH", symptom="wrong output", silent=True,
        found_by="users under concurrent load (#52448, #48089), 19 days after the merge",
        reland=[52823], reland_kind="fixed", evidence_prs=[52823], evidence_issues=[48089, 52448], conf="high"),
    ("vllm", "vllm-49236"): dict(
        what="a model-wide scratch pool that reuses workspace buffers across the layers in DeepSeek-V4's eager break, "
             "to cut the allocation overhead (a 3.9% better TTFT, per the revert)",
        stated=("revert #52836", "The model-wide scratch pool can reuse storage across layers and CUDA streams without "
                                 "the caching allocator's stream/event lifetime tracking. That allows a producer to "
                                 "overwrite shared workspace while a previous consumer may still be using it."),
        reason="The pool gave the same storage to different layers and CUDA streams, but nothing recorded which stream "
               "still used a buffer, which the caching allocator does with stream and event lifetimes. Thus a producer "
               "could overwrite a workspace that an earlier consumer on another stream still read. The revert, 19 days "
               "after the merge, restored allocator-backed temporary buffers 'until safe cross-stream synchronization "
               "is implemented'. It also notes that the narrow eager region corrupted Model Runner V1 output (#52401). "
               "It did not come back.",
        cause="GRAPH", symptom="wrong output", silent=True, found_by="the maintainers, 19 days after the merge",
        reland=[], reland_kind="none", evidence_prs=[51430, 52401], evidence_issues=[], conf="high"),
    ("vllm", "vllm-48666"): dict(
        what="wires the FA4 FP8-KV dequant path into vLLM, so that Gemma-4 full attention (head size 512) uses FA4 with "
             "an FP8 KV cache on SM90",
        stated=("comment on #48666", "This regresses Gemma4 on Hopper nightly ... compared to using Triton"),
        reason="The nightly H200 performance test showed a regression for Gemma-4 31B FP8. The author found the cause: "
               "the FA4 FP8-KV kernel does not support the bidirectional attention of the multimodal prefix, but the "
               "backend selection chose it also when 'use_bidirectional_attention' is 'vision' with --language-only, "
               "so Gemma-4 emitted special tokens, which the benchmark counted as slow output. The resubmit #53175 "
               "fixes the backend selection, and it merged a month later.",
        cause="PRE", symptom="wrong output", silent=True, found_by="the nightly performance test (H200), five days after",
        reland=[53175], reland_kind="fixed", evidence_prs=[42175, 53175], evidence_issues=[], conf="high"),
    ("vllm", "vllm-52989"): dict(
        what="passes vLLM's maximum token bucket to the autotuning of FlashInfer's CUTLASS MoE (in place of its default "
             "of 8,192), so that large prefill batches use tactics tuned for their size",
        stated=("revert #53186", "every one of the four workers raises: MemoryError: CUDA out of memory. Tried to "
                                 "allocate 12.22 GiB."),
        reason="With the larger tuning bucket, the workspace of FlashInfer's CUTLASS MoE grew to 12.22 GiB, and the "
               "nightly B200 LM Eval PCP job (GLM-5.2 NVFP4 with TP2, PCP2 and EP) ran out of memory in "
               "getWorkspaceInfo. The CI failure analyzer generated the revert the next day ('Let's just revert for "
               "now to avoid causing OOMs for users').",
        cause="BUDGET", symptom="OOM", silent=False, found_by="the nightly CI (B200 LM Eval PCP), the next day",
        reland=[], reland_kind="none", evidence_prs=[53209], evidence_issues=[], conf="high"),
    ("vllm", "vllm-52606"): dict(
        what="fuses the nine Triton kernels of Kimi-K3's KDA chunked prefill on ROCm into a prologue kernel and a "
             "chunk-recurrence kernel",
        stated=("revert #53294", "it deterministically breaks the AMD CI image build on every architecture set that "
                                 "includes gfx90a: error: '__builtin_amdgcn_mfma_f32_16x16x32_bf16' needs target "
                                 "feature gfx950-insts"),
        reason="The kernel uses a gfx950-only builtin, so the AMD image build failed for every architecture set that "
               "includes gfx90a. The AMD build of the PR head had failed on all three attempts, but a GitHub auto-merge "
               "setting merged the PR anyway ('there was a GitHub setting that did this .. it's fixed now'). The "
               "reland (#54038) merged three weeks later.",
        cause="BUILD", symptom="build failure", silent=False, found_by="the AMD image build, red before the merge",
        reland=[54038], reland_kind="fixed", evidence_prs=[53731, 54038], evidence_issues=[], conf="high"),
    ("vllm", "vllm-52764"): dict(
        what="runs the multimodal renderer warmup (20 s or more) at the same time as the engine-core initialization, "
             "to hide it",
        stated=("revert #54023", "a live Python background thread is doing multimodal processor work while engine-core "
                                 "initialization forks."),
        reason="Main CI hung at start-up in the multimodal Whisper jobs on H200 and MI300 until their timeouts (75 and "
               "25 minutes). The captured stack shows the warmup thread inside Whisper spectrogram preprocessing while "
               "the main thread forked the engine-core processes; a fork while another thread holds locks can deadlock "
               "the child. A maintainer noted that the same CI step had passed on the PR. A new version merged a week "
               "later (#54557).",
        cause="LOGIC", symptom="hang", silent=False, found_by="post-merge CI (H200 and MI300), the same day",
        reland=[54557], reland_kind="relanded", evidence_prs=[54557], evidence_issues=[], conf="high"),
    ("vllm", "vllm-56323"): dict(
        what="moves the sampling and DFlash JIT kernels of DeepSeek-V4 to the explicit Triton JIT warmup (part 6 of a "
             "series), so that the kernels compile before serving",
        stated=("", ""),
        reason="The revert (#56654) gives no reason. #56649 shows the effect: with the explicit warmups, H200 CI lanes "
               "exceeded their time budgets (for example 32 JIT warmups that took 415.65 s in one job), and after the "
               "revert the affected lanes passed within their original budgets. #58092 later registered the MRV2 "
               "sampler JIT warmups again for ROCm. Issues also report CPU Triton fallbacks that the PR left unused "
               "(#58214, #55005).",
        cause="BUDGET", symptom="slower", silent=False, found_by="post-merge CI (H200 time limits), the same day",
        reland=[58092], reland_kind="modified", evidence_prs=[56649, 58092], evidence_issues=[55005, 58214], conf="medium"),
    ("vllm", "vllm-51692"): dict(
        what="an AITER B-preshuffled block-scaled FP8 GEMM on ROCm, used when the shapes allow it and configs are tuned "
             "(4% to 8% more QPS for DeepSeek-V3 TP8 with DP attention)",
        stated=("revert #57132", "When testing gsm8k for DeepSeek-V4 on AMD GPUs using latest vLLM main, the acc value "
                                 "becomes 0"),
        reason="The next day, a ROCm engineer saw garbage output with DeepSeek-V3 TP8 and a failed start with "
               "DeepSeek-V4 TP8. The fix #56433 found four problems in the new kernel's use, for example that it read "
               "the activation scale as column-major while the callers emit row-major scales, and that models such as "
               "DeepSeek-V4 shuffle some weights again after the kernel B-preshuffles them. DeepSeek-V4 gsm8k still "
               "fell to 0, so #57132 reverted both PRs six days after the merge.",
        cause="PRE", symptom="lower accuracy", silent=True, found_by="AMD engineers, the day after the merge",
        reland=[], reland_kind="none", evidence_prs=[56433, 56502], evidence_issues=[], conf="high"),
}
