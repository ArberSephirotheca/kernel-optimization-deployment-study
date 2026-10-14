"""Join the manual coding (coding.py for vLLM, coding_sglang.py for SGLang) with the B1 census and the
GitHub records (data/raw/<repo>/, see fetch.py); write cases.csv and print the tables of the README.

    python3 -I perf-revert-study/summarize.py [--asof 2026-10-13] [--cases]
"""
import argparse, csv, json, re, runpy, statistics
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
CENSUS = HERE.parent / "deep-study" / "data" / "confirmed-reverts.csv"
REVERTS = ("confirmed_revert", "partial_revert", "explicit_rollback")
REPOS = {"vllm": "coding.py", "sglang": "coding_sglang.py"}
NAMES = {"vllm": "vLLM", "sglang": "SGLang"}
CHECK = runpy.run_path(str(HERE / "coding.py"))["CHECK"]


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None


def merged_at(repo, n):
    return ts(json.loads((HERE / "data" / "raw" / repo / f"pr-{n}.json").read_text())["pr"].get("merged_at"))


def median(xs):
    return round(statistics.median(xs), 1) if xs else None


def build(repo, asof):
    census = [r for r in csv.DictReader(open(CENSUS))
              if r["repo"] == repo and r["reverted_class"] == "kernel_performance" and r["revert_status"] in REVERTS]
    first, targetless = {}, {}
    for r in census:
        ps = [int(p) for p in re.findall(r"\d+", r["reverted_prs"] or "")]
        for p in ps:
            if p not in first or r["revert_date"] < first[p]["revert_date"]:
                first[p] = r
        if not ps:
            targetless[int(r["revert_pr"])] = r
    coded = runpy.run_path(str(HERE / REPOS[repo]))["ROWS"]
    with_target = {c["pr"] for c in coded if not c.get("rollback")}
    rollbacks = {c["rollback"] for c in coded if c.get("rollback")}
    assert with_target == set(first), f"{repo}: coding and census targets differ: {sorted(with_target ^ set(first))}"
    assert rollbacks == set(targetless), f"{repo}: coding and census rollbacks differ: {sorted(rollbacks ^ set(targetless))}"
    out = []
    for c in coded:
        cr = targetless[c["rollback"]] if c.get("rollback") else first[c["pr"]]
        m, rv = (merged_at(repo, c["pr"]) if c["pr"] else None), ts(cr["revert_date"])
        rel = [x for x in (merged_at(repo, n) for n in c["reland"]) if x]
        rl = min(rel) if rel else None
        out.append(dict(
            c, repo=repo, rollback=c.get("rollback", ""), check=CHECK[c["cause"]],
            merged=m.date().isoformat() if m else "", revert_pr=int(cr["revert_pr"]), revert_status=cr["revert_status"],
            revert_date=rv.date().isoformat(), days_to_revert=round((rv - m).total_seconds() / 86400, 2) if m else None,
            reland_date=rl.date().isoformat() if rl else "",
            days_revert_to_reland=round((rl - rv).total_seconds() / 86400, 1) if rl else None,
            days_merge_to_reland=round((rl - m).total_seconds() / 86400, 1) if rl and m else None,
            days_not_relanded=(asof - rv.date()).days if c["reland_kind"] == "none" else None))
    return out


def incidents(rows):
    """The first row of each failure incident (rows of one incident were reverted together or are one
    optimization reverted twice)."""
    seen = {}
    for r in rows:
        if r["cause"] != "NOTFAIL":
            seen.setdefault((r["repo"], r["incident"]), r)
    return list(seen.values())


def table(head, body, align=None):
    align = align or "l" + "r" * (len(head) - 1)
    print("| " + " | ".join(head) + " |")
    print("|" + "|".join(" --- " if a == "l" else " ---: " for a in align) + "|")
    for b in body:
        print("| " + " | ".join("" if x is None else str(x) for x in b) + " |")
    print()


def cause_table(inc):
    by = defaultdict(list)
    for r in inc:
        by[r["cause"]].append(r)
    body = []
    for c, rs in sorted(by.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        rel = [r["days_revert_to_reland"] for r in rs if r["days_revert_to_reland"] is not None]
        dtr = [r["days_to_revert"] for r in rs if r["days_to_revert"] is not None]
        body.append([c, len(rs), sum(r["silent"] for r in rs), median(dtr), f"{len(rel)} of {len(rs)}",
                     median(rel), round(sum(rel)) if rel else None, CHECK[c]])
    table(["Cause", "Incidents", "Silent", "Median days to revert", "Relanded", "Median days revert to reland",
           "Rework days", "Check that targets the cause"], body, "lrrrrrrl")


def report(name, rows, inc):
    dtr = [r["days_to_revert"] for r in inc if r["days_to_revert"] is not None]
    print(f"### {name}: {len(rows)} coded rows, {len({r['pr'] for r in rows if r['pr']})} PRs, "
          f"{len(inc)} failure incidents, {sum(r['cause'] == 'NOTFAIL' for r in rows)} planned rollbacks\n")
    cause_table(inc)
    for s in (True, False):
        xs = [r["days_to_revert"] for r in inc if r["silent"] is s and r["days_to_revert"] is not None]
        print(f"silent={s}: {len(xs)} incidents, median {median(xs)} d, max {max(xs) if xs else None} d, "
              f"sum {round(sum(xs), 1)} d in main")
    print(f"median days to revert: {median(dtr)}; detected by: {dict(Counter(r['detected'] for r in inc))}")
    lanes = [r for r in inc if r["lane"]]
    print(f"found by an existing lane after the merge, or red/cancelled before it: {len(lanes)}")
    print("warpDRF:", dict(Counter(r["warpdrf"] for r in inc)),
          {k: [r["pr"] for r in rows if r["warpdrf"] == k] for k in ("direct", "class")})
    rel = [r for r in inc if r["days_revert_to_reland"] is not None]
    print(f"relanded {len(rel)} of {len(inc)}: median revert to reland {median([r['days_revert_to_reland'] for r in rel])} d, "
          f"median merge to reland {median([r['days_merge_to_reland'] for r in rel if r['days_merge_to_reland'] is not None])} d, "
          f"sum revert to reland {round(sum(r['days_revert_to_reland'] for r in rel))} d")
    nr = [r for r in inc if r["reland_kind"] == "none"]
    print(f"not relanded {len(nr)}:", ", ".join(f"#{r['pr'] or r['rollback']} {r['cause']} {r['days_not_relanded']} d" for r in nr))
    print(f"confidence: {dict(Counter(r['conf'] for r in inc))}\n")


def cases(rows):
    body = []
    for r in rows:
        ident = f"#{r['pr']}" if r["pr"] else "—"
        if r["rollback"]:
            ident += f" (rollback #{r['rollback']})"
        rl = (f"#{r['reland'][0]}" + ("+" if len(r["reland"]) > 1 else "") + f" ({r['reland_kind']}, "
              f"{r['days_revert_to_reland']} d)") if r["reland_date"] else (
              f"none ({r['days_not_relanded']} d)" if r["reland_kind"] == "none" else r["reland_kind"])
        body.append([ident, r["cause"], r["days_to_revert"], rl, r["detail"]])
    table(["PR", "Cause", "Days to revert", "Reland (days after the revert)", "What happened"], body, "llrll")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", default="2026-10-13", help="date for rows that have no reland yet")
    ap.add_argument("--cases", action="store_true", help="also print the case tables")
    a = ap.parse_args()
    asof = date.fromisoformat(a.asof)
    rows = {repo: build(repo, asof) for repo in REPOS}
    keys = ["repo", "pr", "rollback", "cause", "symptom", "silent", "detected", "lane", "check", "warpdrf", "conf",
            "merged", "revert_pr", "revert_status", "revert_date", "days_to_revert", "reland", "reland_kind",
            "reland_date", "days_revert_to_reland", "days_merge_to_reland", "days_not_relanded", "detail"]
    with open(HERE / "cases.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for repo in REPOS:
            for r in rows[repo]:
                w.writerow(dict(r, reland=" ".join(f"#{n}" for n in r["reland"])))
    inc = {repo: incidents(rows[repo]) for repo in REPOS}
    for repo in REPOS:
        report(NAMES[repo], rows[repo], inc[repo])
    both = inc["vllm"] + inc["sglang"]
    print("### Both repositories\n")
    by = defaultdict(lambda: Counter())
    for r in both:
        by[r["cause"]][r["repo"]] += 1
    body = [[c, n["vllm"], n["sglang"], n["vllm"] + n["sglang"],
             sum(r["silent"] for r in both if r["cause"] == c), CHECK[c]]
            for c, n in sorted(by.items(), key=lambda kv: (-sum(kv[1].values()), kv[0]))]
    table(["Cause", "vLLM", "SGLang", "Both", "Silent", "Check that targets the cause"], body, "lrrrrl")
    report("Both", rows["vllm"] + rows["sglang"], both)
    if a.cases:
        for repo in REPOS:
            print(f"### Cases: {NAMES[repo]}\n")
            cases(rows[repo])


if __name__ == "__main__":
    main()
