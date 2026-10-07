#!/usr/bin/env python3
"""Triage escaped kernel-correctness bugs for a measured verification case study.

  triage.py [--out verification-triage/triage.csv]

Reads the deep study's confirmed kernel-correctness cases
(deep-study/data/kernel-correctness-cases.csv) whose primary or secondary
failure class is a shape/alignment edge, out-of-bounds memory access, race or
nondeterminism, or warp mask: the classes a verifier or sanitizer could catch
before merge. For each case it inspects the fix commit in a local partial clone
(data/repos/{vllm,sglang}, see SUBMODULE.md / config/study.json):

- the files the fix changed, classified as CUDA/C++ kernel source, Triton
  kernel source, other Python, tests, build/dependency files, or other;
- the kernel-side lines changed (tests and docs excluded);
- the hardware the bug needs, from the coded affected_hardware field and
  path hints;
- whether the affected kernel has a TIRx port in TIRx-kernels (by name);

and assigns a feasibility tier for measuring, on the pre-fix commit (the
fix's first parent), whether tests, sanitizers or static/symbolic analysis
catch the bug on the one available GPU (NVIDIA Thor, sm_110):

  A  kernel source in the repo (CUDA C++ or Triton), runnable on Thor
  B  kernel source in the repo, but needs other hardware (Hopper-only, AMD,
     NPU, CPU) or the fix is mostly outside kernels
  C  the bug is in Python glue or integration logic, not kernel source
  X  not measurable here: the fix bumps an external library, or the commit
     is missing

The tiers are a first pass from metadata and paths; each Tier A case still
needs a reproduction harness before it counts.
"""
import argparse
import collections
import csv
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "deep-study/data/kernel-correctness-cases.csv"
REPOS = {"vllm": ROOT / "data/repos/vllm", "sglang": ROOT / "data/repos/sglang"}
CLASSES = {"shape_alignment_edge", "memory_safety_oob", "nondeterminism_race_sync", "warp_subgroup_mask"}

NON_NVIDIA = [("rocm", "AMD"), ("hip", "AMD"), ("amd", "AMD"), ("aiter", "AMD"), ("mi300", "AMD"), ("mi355", "AMD"),
              ("npu", "NPU"), ("ascend", "NPU"), ("xpu", "Intel XPU"), ("tpu", "TPU"), ("cpu", "CPU"),
              ("x86", "CPU"), ("habana", "Gaudi"), ("hpu", "Gaudi"), ("musa", "MUSA")]
HOPPER_ONLY = ["hopper", "sm90", "sm_90", "h100", "h200", "h20", "fa3", "flash_attn_3", "flashattention-3", "wgmma",
               "fp8_blockwise_sm90", "cutlass_sm90"]
BLACKWELL = ["blackwell", "sm100", "sm_100", "b200", "gb200", "gb300", "sm120", "sm_120", "rtx 50", "5090",
             "tcgen05", "trtllm-gen", "dgx spark", "sm121"]
EXTERNAL = ["flashinfer", "deep_gemm", "deepgemm", "flash_mla", "cutlass", "triton_kernels", "aiter", "fa3", "xformers"]
# TIRx-kernels names (sm_110a-runnable marked) matched by keywords in the kernel description or paths.
ZOO = [("rmsnorm", "flashinfer_rmsnorm (sm_110a)"), ("rms_norm", "flashinfer_rmsnorm (sm_110a)"),
       ("layernorm", "flashinfer_layernorm (sm_110a)"), ("layer_norm", "flashinfer_layernorm (sm_110a)"),
       ("merge_state", "merge_state (sm_110a)"), ("merge_attn", "merge_state (sm_110a)"),
       ("top_k", "filtered_topk / fast_topk_clusters (sm_110a)"), ("topk", "filtered_topk / fast_topk_clusters (sm_110a)"),
       ("act_and_mul", "act_and_mul (sm_110a)"), ("silu_and_mul", "act_and_mul (sm_110a)"),
       ("selective_state", "selective_state_update_* (sm_110a)"), ("mamba", "selective_state_update_* (sm_110a)"),
       ("mxfp4", "mxfp4_quantize (sm_110a)"), ("kda", "kda / recurrent_kda_* "), ("gdn", "gdn_* / cudnn_sm100_gdn_*"),
       ("mla", "mla / flash_mla ports"), ("moe", "moe ports"), ("gemm", "gemm ports"),
       ("attention", "attention ports"), ("flash_attn", "flash_attention4 (sm_110a)")]


def git(repo, *args):
    p = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    return p.stdout if p.returncode == 0 else None


def classify(path, content):
    p = path.lower()
    name = p.rsplit("/", 1)[-1]
    if "/tests/" in "/" + p or name.startswith("test_") or "/test/" in "/" + p or p.startswith("test"):
        return "test"
    if p.startswith(("docs/", "examples/")) or name.endswith((".md", ".rst")):
        return "docs"
    if "benchmark" in p:
        return "benchmark"
    if name in ("requirements.txt", "setup.py", "pyproject.toml", "cmakelists.txt", "dockerfile") or \
            p.startswith("requirements") or name.endswith((".cmake", ".txt", ".toml", ".lock", ".yaml", ".yml")) or \
            "docker" in p:
        return "build_deps"
    if name.endswith((".cu", ".cuh")):
        return "cuda"
    if name.endswith((".cpp", ".cc", ".h", ".hpp", ".c")):
        return "cpp"
    if name.endswith(".py"):
        if content and ("@triton.jit" in content or "@triton.autotune" in content or "import triton" in content
                        and "tl." in content):
            return "triton"
        return "python"
    return "other"


def hardware(row, paths):
    text = " ".join([row["affected_hardware"], row["affected_kernel"], row["subject"], *paths]).lower()
    for key, label in NON_NVIDIA:
        if re.search(rf"(?<![a-z]){re.escape(key)}(?![a-z])", text):
            return f"non-NVIDIA ({label})"
    if any(k in text for k in HOPPER_ONLY):
        return "NVIDIA Hopper"
    if any(k in text for k in BLACKWELL):
        return "NVIDIA Blackwell"
    if row["hardware_specific"] == "yes":
        return "NVIDIA, specific (unstated)"
    return "NVIDIA, generic"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "verification-triage/triage.csv")
    args = ap.parse_args()
    rows = [r for r in csv.DictReader(open(CASES)) if r["case_status"] == "confirmed_kernel_correctness"]
    out = []
    for r in rows:
        secondary = [c for c in (r["secondary_classes"] or "").split(";") if c]
        if r["failure_class"] not in CLASSES and not set(secondary) & CLASSES:
            continue
        repo = REPOS[r["repo"]]
        sha = r["fix_sha"]
        rec = {k: r[k] for k in ("case_id", "repo", "fix_pr", "fix_sha", "fix_date", "url", "subject",
                                 "failure_class", "secondary_classes", "symptom", "affected_kernel",
                                 "affected_hardware", "hardware_specific", "introducing_ref", "detected_by",
                                 "escaped_ci", "regression_test_added", "counterfactual_primary",
                                 "counterfactual_all")}
        rec["class_role"] = "primary" if r["failure_class"] in CLASSES else "secondary"
        if git(repo, "cat-file", "-t", sha) is None:
            rec.update(tier="X", tier_reason="fix commit not in the clone")
            out.append(rec)
            continue
        parent = (git(repo, "rev-parse", f"{sha}^1") or "").strip()
        numstat = git(repo, "show", "--numstat", "--format=", "--no-renames", sha) or ""
        files = []
        for line in numstat.splitlines():
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            add, rem, path = parts
            content = git(repo, "show", f"{sha}:{path}") if path.endswith(".py") else None
            if content is None and path.endswith(".py") and parent:
                content = git(repo, "show", f"{parent}:{path}")
            kind = classify(path, content)
            files.append((path, kind, int(add) if add.isdigit() else 0, int(rem) if rem.isdigit() else 0))
        kinds = collections.Counter()
        for _, kind, a, d in files:
            kinds[kind] += a + d
        kernel_lines = kinds["cuda"] + kinds["cpp"] + kinds["triton"]
        glue_lines = kinds["python"]
        paths = [f for f, _, _, _ in files]
        hw = hardware(r, paths)
        text = (r["affected_kernel"] + " " + " ".join(paths)).lower()
        zoo = next((z for k, z in ZOO if k in text), "")
        external = [e for e in EXTERNAL if e in text]
        lang = ("CUDA/C++" if kinds["cuda"] + kinds["cpp"] >= max(kinds["triton"], 1) and kinds["cuda"] + kinds["cpp"] else
                "Triton" if kinds["triton"] else "Python" if glue_lines else "build/deps" if kinds["build_deps"] else "other")
        rec.update(parent_sha=parent, files_changed=len(files), kernel_lines=kernel_lines, python_lines=glue_lines,
                   test_lines=kinds["test"], build_lines=kinds["build_deps"], fix_language=lang, hardware=hw,
                   zoo_counterpart=zoo, external_library_mentions=";".join(external),
                   kernel_files=";".join(f for f, k, _, _ in files if k in ("cuda", "cpp", "triton"))[:400],
                   python_files=";".join(f for f, k, _, _ in files if k == "python")[:300])
        on_thor = hw in ("NVIDIA, generic", "NVIDIA, specific (unstated)", "NVIDIA Blackwell")
        if lang == "build/deps" or (kernel_lines == 0 and glue_lines == 0):
            tier, why = "X", "fix changes only build/dependency or non-code files (external library)"
        elif kernel_lines == 0:
            tier, why = "C", "fix is in Python glue or integration code, not kernel source"
        elif not on_thor:
            tier, why = "B", f"kernel source in the repo, but needs {hw}"
        elif kernel_lines < glue_lines:
            tier, why = "B", "kernel source changed, but most of the fix is outside kernels"
        else:
            tier, why = "A", f"{lang} kernel source in the repo; {hw}"
        rec.update(tier=tier, tier_reason=why)
        out.append(rec)
    fields = list(dict.fromkeys(k for rec in out for k in rec))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(out)
    print(f"{len(out)} cases -> {args.out}")
    print("tiers:", dict(collections.Counter(r["tier"] for r in out)))


if __name__ == "__main__":
    main()
