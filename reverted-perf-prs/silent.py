"""The silent failures: wrong output, lower accuracy or NaNs, with no error. build.py prints the tables of the
README section "Silent failures" from this file and checks that NOTES has one entry for each silent incident.

GROUPS puts each cause in a group. NOTES gives for each silent incident a short form of its record in records.py,
and an assessment of what could have found it before the merge. The assessments are the coder's judgment, not
results: no check ran, except warpDRF on SGLang #11133 (run of 2026-10-13, with barrier.orders_global granted).
The capture-stability and stream-order check does not exist yet.

    NOTES[(repo, incident)] = (optimization, what went wrong, found by, before the merge, formal)

formal is the formal check that targets the mechanism: "capture" (the captured graph keeps no value, address or
layout that changes after the capture), "stream" (no two streams use the same storage without an event between
the uses), "contract" (a call-site contract of the kernel or the backend), "race" (the warpDRF race check), or "".
A trailing "?" marks a probable mechanism.
"""

GROUPS = {  # cause -> group, in the order of the tables
    "GRAPH": "Graph capture and stream order",
    "PRE": "A kernel used outside its requirements (PRE)",
    "KERNEL": "A defect inside a kernel",
    "MODEL": "Model layout or platform path", "LOGIC": "Model layout or platform path", "BUILD": "Model layout or platform path",
    "NUM": "Lower precision",
    "UNK": "Mechanism not known",
}

CHECKS = {  # group -> what can find it before the merge
    "Graph capture and stream order": "a capture-stability and stream-order check (not built yet); the CI tests before the merge",
    "A kernel used outside its requirements (PRE)": "call-site contracts: buffer size, layout, tile size, supported features",
    "A defect inside a kernel": "the warpDRF race check; a comparison with the upstream kernel",
    "Model layout or platform path": "a test of each model and platform path that uses the changed code",
    "Lower precision": "an accuracy test with an error budget",
    "Mechanism not known": "not known",
}

NOTES = {
    # Graph capture and stream order
    ("vllm", "vllm-50004"): ("adaptive top-k width (DeepSeek-V4)",
                             "a writer outside the graph changed the row stride, but the captured reader kept the old stride",
                             "users", "capture stability: the writer and the captured reader agree on the layout", "capture"),
    ("vllm", "vllm-49236"): ("one scratch pool for all layers",
                             "a producer wrote a buffer that a reader on another stream still used",
                             "the maintainers", "stream order: no event between the two uses", "stream"),
    ("vllm", "vllm-29558"): ("piecewise CUDA graphs for DeepEP HT",
                             "the outputs did not keep the same addresses across replays",
                             "H200 CI, after the merge", "capture stability: fixed addresses; the H200 tests", "capture"),
    ("sglang", "sglang-14044"): ("one memory pool for two graph runners",
                                 "accuracy 0; the shared pool or the pool id (not confirmed)",
                                 "CI, after the merge", "the piecewise CUDA graph test", ""),
    ("vllm", "vllm-48223"): ("dual-stream decode (ROCm)",
                             "two streams used the same tensor with no copy",
                             "AMD CI, after the merge", "stream order: aliased tensors", "stream"),
    ("sglang", "sglang-19190"): ("fewer CPU-GPU sync points",
                                 "pinned non-blocking copies doubled the KL divergence (the mechanism is not stated)",
                                 "the KL test, after the merge", "the KL test", ""),
    ("sglang", "sglang-29142"): ("new stream order in the DeepSeek MoE",
                                 "a host-side `set_()` made the captured GEMM read address 0",
                                 "CI, after the merge", "capture stability: a host change during the capture", "capture"),
    ("vllm", "vllm-45309"): ("fewer eager breaks in the capture (DeepSeek-V4)",
                             "probably a per-step value that stays fixed in the graph (not confirmed)",
                             "a maintainer's bisect", "capture stability (if the cause is correct)", "capture?"),
    # A kernel used outside its requirements
    ("sglang", "sglang-9087"): ("fused-MoE tuning config (B200)",
                                "a K tile of 256 covers two FP8 scale groups of 128 (probable)",
                                "a user", "a launch rule: BLOCK_SIZE_K is not larger than the scale group (128)", "contract"),
    ("vllm", "vllm-51692"): ("pre-shuffled FP8 GEMM from AITER (ROCm)",
                             "the kernel read row-major scales as column-major; some weights were shuffled twice",
                             "AMD engineers", "a layout contract at the call site", "contract"),
    ("vllm", "vllm-48666"): ("FA4 FP8-KV path (Gemma-4)",
                             "the backend selection chose a kernel without bidirectional attention",
                             "the nightly performance test", "a feature check in the backend selection", "contract"),
    ("vllm", "vllm-23465"): ("FA3 swizzle update",
                             "the kernel wrote past the metadata buffer when B is not a multiple of 4",
                             "2xH100 CI, after the merge", "a bounds contract with a symbolic B", "contract"),
    ("vllm", "vllm-14384"): ("less MLA CPU overhead",
                             "strided input to an in-place kernel that needs contiguous input",
                             "a maintainer", "a contiguity contract at the call site", "contract"),
    # A defect inside a kernel
    ("sglang", "sglang-11133"): ("MTP topk > 1 (Qwen3-Next); it also removed three `tl.debug_barrier()` calls",
                                 "loads and stores of one program race",
                                 "a user (CI showed a pass)",
                                 "the warpDRF race check (finds it with barrier.orders_global); a CI that reports failed tests",
                                 "race"),
    ("sglang", "sglang-15242"): ("FP8 sparse update of FlashMLA",
                                 "NaNs with the NVFP4 checkpoint",
                                 "the author", "a comparison with the upstream FlashMLA output", ""),
    ("sglang", "sglang-26235"): ("no softmax in the EAGLE draft (topk 1)",
                                 "on AMD, an argmax kernel gave an index outside the vocabulary (not certain)",
                                 "the AMD nightly test", "the AMD accuracy test", ""),
    # Model layout or platform path
    ("vllm", "vllm-28271"): ("no TP gather in Qwen-VL",
                             "the Qwen2-VL qkv layout needs the gather under TP",
                             "a user", "a TP 2 against TP 1 comparison for each model", ""),
    ("vllm", "vllm-34492"): ("fused GDN projections (Qwen3.5)",
                             "Qwen3-Next uses the same code, and its split sizes became wrong",
                             "a user", "a test of each model that uses the code", ""),
    ("vllm", "vllm-14778"): ("fewer copies in the Mamba2 prefill",
                             "`copy_` wrote into a temporary from tensor indexing, not into the cache",
                             "users", "a Mamba2 accuracy test; a lint rule for this pattern", ""),
    ("sglang", "sglang-6220"): ("routed scaling factor fused into a CUDA kernel",
                                "the non-CUDA paths lost the factor",
                                "unknown", "a HIP accuracy test", ""),
    # Lower precision
    ("sglang", "sglang-28166"): ("GELU in the cuBLASLt GEMM epilogue (FLUX)",
                                 "the images moved away from the unfused reference",
                                 "the maintainer", "an image comparison with the unfused path", ""),
    ("vllm", "vllm-34582"): ("NemotronH router not in fp32",
                             "the accuracy fell by 4% to 5%",
                             "LM-eval CI, after the merge", "the LM-eval test", ""),
    # Mechanism not known
    ("vllm", "vllm-37205"): ("router GEMM kernel (gpt-oss)", "rare NaNs", "a user's internal test", "not known", ""),
    ("vllm", "vllm-14995"): ("TPU bucket padding", "wrong output with a padding gap of 64",
                             "the maintainers", "a test that the padding does not change the output", ""),
    ("vllm", "vllm-35216"): ("compile passes made once at init", "wrong logits above about 2,048 tokens at -O3",
                             "a user's bisect", "a long-prompt test at -O3", ""),
    ("sglang", "sglang-33623"): ("MLA gate fused into the QKV-A GEMM (Kimi-K3)",
                                 "the output did not stop before 2,048 tokens",
                                 "CI, after the merge", "the CI test", ""),
}
