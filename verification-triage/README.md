# Verification triage of escaped kernel bugs

The deep study's counterfactual detectors are judgments, not experiments: no
technique was run on the bugs (see `deep-study/SYNTHESIS.md`, open
limitations). This directory triages which escaped bugs could be measured
instead, by running tests, sanitizers or static/symbolic analysis on the
pre-fix code, with the one GPU available (NVIDIA Thor, `sm_110`).

[`triage.py`](triage.py) selects the confirmed kernel-correctness cases in
`deep-study/data/kernel-correctness-cases.csv` whose primary or secondary
failure class is one a verifier or sanitizer could catch before merge: shape or
alignment edge, out-of-bounds memory access, race or nondeterminism, warp mask.
For each it reads the fix commit in a partial clone (`data/repos/`, gitignored;
`git clone --filter=blob:none`), classifies the changed files, sizes the
kernel-side change, infers the hardware from the coded fields and paths, looks
for a TIRx port in TIRx-kernels, and assigns a tier. Output:
[`triage.csv`](triage.csv), one row per case.

| Tier | Meaning | Cases (primary class) |
| --- | --- | ---: |
| A | Kernel source (CUDA C++ or Triton) in the repo, NVIDIA generic or Blackwell | 29 (24) |
| B | Kernel source in the repo, but needs Hopper (14), AMD (8), CPU (4), NPU (2) or XPU (1), or most of the fix is outside kernels (11) | 40 (30) |
| C | The fix is in Python glue or integration code, not kernel source | 62 (47) |
| X | Not measurable (dependency bump only, missing commit) | 0 |

131 cases: 101 with one of the four classes as primary (62 shape/alignment,
28 out-of-bounds, 10 race/nondeterminism, 1 warp mask) and 30 as secondary.

## What the triage shows

**Half of these bugs are not in kernel source.** 62 of 131 fixes change only
Python: page-table and CUDA-graph buffer sizing, dispatcher conditions,
metadata construction, assertions, wrapper defaults. The kernel is often fine;
its caller violates a precondition (an extent, an alignment, a supported
shape). Verifying kernels alone misses these. Verifying the kernel–caller
contract would not: state each kernel's preconditions, then check that every
call site meets them.

**GPU-kernel races are six cases.** The race/nondeterminism class mixes kernel
races with host concurrency (cross-process shared memory, sockets, JIT
builds). The kernel-level synchronization bugs are vLLM Marlin (#11493, CUDA)
and `ep_scatter` (#34991, Triton), SGLang's overlap-mode metadata kernels
(#2051, Triton), the fused KDA verify kernel (#39219, #39524, Triton) and an
undefined-behaviour warp sync (#9620, CUDA). They are the cases a
synchronization checker such as warpDRF or Racecheck addresses.

**Bounds and shape bugs inside kernels are the larger verifiable set.** Tier A
holds 7 out-of-bounds kernels (int32 offset overflow in Triton top-k renorm and
CUTLASS MoE pre-reorder, int32 truncation in GGUF dequantize, `cache_kernels.cu`,
DSA + MTP, sparse MLA workspace, RoPE cache) and 11 shape/alignment kernels
(hidden size not divisible by 16, RMSNorm for hidden size 64/128/256, Mamba2
varlen, invalid MoE slots, zero-length chunked prefill). Each fails at shapes
the PR's tests did not run, which symbolic bounds checking over the admitted
shape domain targets by construction.

**Hardware limits what one GPU can measure.** 35 cases need non-NVIDIA
hardware and 20 Hopper; Tier A is what Thor can run, 29 cases (17 Triton,
12 CUDA C++).

**Only two Tier A cases have a TIRx counterpart**, both RMSNorm: SGLang's JIT
RMSNorm for hidden size 64/128/256 (#20661) and vLLM's RMSNorm batch invariance
(#48391). The TIRx tools (warpDRF's extractor, Synccheck, Racecheck) reach
almost none of the production bugs directly; production kernels are CUDA C++
and Triton.

## Tools available

| Tool | Status |
| --- | --- |
| `compute-sanitizer` (memcheck, racecheck) | on Thor (CUDA 13.3.1) |
| Triton 3.8.0, including `TRITON_INTERPRET=1` CPU execution | on Thor (`~/warpdrf-env/venv`) |
| Faial for CUDA C++ kernels | `faial-drf` with its Clang-based `cu-to-json` front end, built locally (outside this repository); checks data races at block and grid level, not bounds |
| warpDRF extractor, TIRx Synccheck/Racecheck | TIRx kernels only |
| Randomized shape tests | to write per kernel |

## Shortlist for the measured study

Twelve Tier A cases spanning the classes, chosen for a localized kernel fix and
a clear trigger. Each needs a harness that runs the pre-fix kernel standalone on
Thor (Triton kernels by importing the kernel function; CUDA kernels compiled as
a torch extension with the repository's headers).

| Class | Cases |
| --- | --- |
| Race / sync | vLLM `5dba257506` Marlin (#11493, CUDA); vLLM `55d8073d06` `ep_scatter` (#34991, Triton); SGLang `2fd835b9c1` KDA verify (#39219, Triton); SGLang `dbb1235d58` warp sync UB in per-token-group quantization (#9620, CUDA) |
| Out of bounds | SGLang `9234e40aed` top-k renorm int32 overflow (#35571, Triton); SGLang `a1f9508dd4` MoE pre-reorder int32 overflow (#35188, Triton); vLLM `f219788f91` GGUF dequantize int32 truncation (#44971, CUDA); vLLM `fb8851f254` `cache_kernels.cu` OOB (#28760, CUDA) |
| Shape / alignment | SGLang `db7343c992` per-token kernel, hidden dim not divisible by 16 (#8543, CUDA); SGLang `777edb6ef7` JIT RMSNorm, hidden size 64/128/256 (#20661, CUDA, TIRx counterpart); vLLM `d7ab9be775` invalid MoE slots (#46408, Triton) |
| Determinism | vLLM `b6cbba8bc8` RMSNorm batch invariance (#48391, CUDA, TIRx counterpart) |

For each case, run on the pre-fix commit (the fix's first parent) and on the
fix: the repository's tests for that kernel; a randomized test over the
shapes the dispatcher admits; `compute-sanitizer` memcheck or racecheck; the
Triton interpreter for Triton kernels; and, where available, static or
symbolic analysis. Record which detector catches the bug, at what effort, and
with what false alarms on the fixed version.

## Caveats

- Classes, hardware and counterfactual detectors come from the deep study's
  LLM coding (model–model agreement, not human). The triage's file and
  hardware categories are path heuristics; spot checks found integration fixes
  that touch a few kernel lines in Tier A (for example SGLang #26255 and
  #21987) and weak name-based TIRx matches (`topk`, `gemm`, `moe`).
- The pre-fix state is the fix commit's parent; 85% of cases do not name an
  introducing PR, and a fix commit can bundle unrelated changes.
- Tier A means runnable in principle on Thor, not reproduced.

```sh
git clone --filter=blob:none --no-checkout https://github.com/vllm-project/vllm.git data/repos/vllm
git clone --filter=blob:none --no-checkout https://github.com/sgl-project/sglang.git data/repos/sglang
python3 verification-triage/triage.py
```
