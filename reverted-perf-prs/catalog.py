"""The reverted performance PRs of vLLM and SGLang: every revert or rollback of the B1 census (24 months
to 2026-09-28) whose target is a performance PR (classification.py), grouped into incidents.

An incident is one optimization that was reverted. A resubmit that was reverted again, or two PRs that one
revert undid together, are one incident (INCIDENT_GROUPS).

    python3 -I reverted-perf-prs/catalog.py            # prints the counts
    python3 -I reverted-perf-prs/catalog.py --csv      # also writes catalog.csv (one row per revert of a performance PR)
"""
import argparse, csv, re, runpy
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
CENSUS = HERE.parent / "deep-study" / "data" / "confirmed-reverts.csv"
REVERTS = ("confirmed_revert", "partial_revert", "explicit_rollback")

# (repo, PR) -> incident id, for PRs that belong to one incident
INCIDENT_GROUPS = {
    ("vllm", 12355): "vllm-12355", ("vllm", 12435): "vllm-12355",      # reverted together by #12445
    ("vllm", 30207): "vllm-30207", ("vllm", 33052): "vllm-30207",      # reverted together by #33241
    ("vllm", 26368): "vllm-26368", ("vllm", 28771): "vllm-26368",      # #28771 redoes #26368
    ("sglang", 8364): "sglang-8364", ("sglang", 8770): "sglang-8364",  # #8770 resubmits #8364
    ("sglang", 8027): "sglang-8027", ("sglang", 9350): "sglang-8027",  # #9350 reworks #8027
}


def decisions():
    K = runpy.run_path(str(HERE / "classification.py"))
    C = runpy.run_path(str(HERE / "candidates.py"))
    return K, C


def rows():
    """One row per (census revert, target) whose target is a performance PR; asserts full coverage."""
    K, C = decisions()
    D, R = K["DECISIONS"], K["ROLLBACKS"]
    out, undecided = [], []
    for r in C["rows"]():
        repo, n = r["repo"], r["target"]
        if n == "":
            rb = R.get((repo, int(r["revert_pr"])) if r["revert_pr"] else None)
            if rb is None:
                undecided.append((repo, "revert", r["revert_pr"] or r["revert_sha"]))
                continue
            target, verdict, kind, ev = rb
            how = "read here (rollback with no PR named)"
            if target is not None:
                pr = C["pull"](repo, target) or {}
                r = dict(r, target=target, title=pr.get("title", r["title"]), merged=(pr.get("merged_at") or "")[:10])
        elif (repo, n) in D:
            verdict, kind, ev = D[(repo, n)]
            how = "read here"
        elif r["a1_source"].startswith("adjudicated") or r["a1_source"] == "rule_confirmed":
            verdict = {"confirmed_performance": "perf", "not_performance": "not"}.get(r["a1_class"], "unclear")
            kind, ev = (r["a1_type"] or "-"), r["a1_evidence"]
            how = "A1 coder" if r["a1_source"].startswith("adjudicated") else "A1 rule (explicit performance tag)"
        else:
            undecided.append((repo, "target", n))
            continue
        if verdict != "perf":
            continue
        tgt = r["target"] if r["target"] != "" else None
        out.append(dict(r, perf_type=kind, evidence=ev, decided_by=how,
                        incident=INCIDENT_GROUPS.get((repo, tgt), f"{repo}-{tgt}" if tgt else f"{repo}-r{r['revert_pr']}")))
    assert not undecided, f"{len(undecided)} undecided: {undecided[:10]}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="store_true")
    rs = rows()
    inc = defaultdict(list)
    for r in rs:
        inc[r["incident"]].append(r)
    print(f"{len(rs)} reverts of performance PRs; {len(inc)} incidents "
          f"(vLLM {sum(k.startswith('vllm') for k in inc)}, SGLang {sum(k.startswith('sglang') for k in inc)})")
    print("by decision source:", dict(Counter(r["decided_by"] for r in rs)))
    print("by perf type:", dict(Counter(r["perf_type"] for r in rs)))
    print("by revert status:", dict(Counter(r["revert_status"] for r in rs)))
    print("by census class:", dict(Counter(r["census_class"] for r in rs)))
    if ap.parse_args().csv:
        keys = ["incident", "repo", "target", "title", "merged", "perf_type", "evidence", "decided_by", "revert_pr",
                "revert_sha", "revert_date", "revert_status", "subject", "census_class", "census_reason",
                "census_evidence"]
        with open(HERE / "catalog.csv", "w", newline="") as f:
            w = csv.DictWriter(f, keys, extrasaction="ignore", lineterminator="\n")
            w.writeheader()
            w.writerows(sorted(rs, key=lambda r: (r["repo"], r["revert_date"])))
        print("wrote catalog.csv")


if __name__ == "__main__":
    main()
