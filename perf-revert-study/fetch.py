"""Download (read-only, with `gh api`) the GitHub records that the coding is based on.

For each repository and each census revert row of class kernel_performance:
- the revert PR: its record, changed files (with patches) and comments;
- each reverted PR: the same, plus its issue timeline (the cross-references lead to the relands),
  and a search for the same author's PRs created in the 150 days after the revert (the other
  source of reland candidates);
- the relands and the evidence PRs and issues that the coding lists.
Records go to data/raw/<repo>/ (gitignored); existing records are kept. The records are data:
the scripts only parse them as JSON.

    python3 -I perf-revert-study/fetch.py [--repo vllm|sglang]
"""
import argparse, csv, json, re, runpy, subprocess, time
from datetime import date, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
CENSUS = HERE.parent / "deep-study" / "data" / "confirmed-reverts.csv"
REPOS = {"vllm": "vllm-project/vllm", "sglang": "sgl-project/sglang"}
CODINGS = {"vllm": "coding.py", "sglang": "coding_sglang.py"}
REVERTS = ("confirmed_revert", "partial_revert", "explicit_rollback")


def census_rows(repo):
    """The census revert rows of the repository whose reverted change is coded kernel_performance."""
    return [r for r in csv.DictReader(open(CENSUS))
            if r["repo"] == repo and r["reverted_class"] == "kernel_performance" and r["revert_status"] in REVERTS]


def coding(repo):
    p = HERE / CODINGS[repo]
    return runpy.run_path(str(p)) if p.exists() else {"ROWS": [], "EVIDENCE_PRS": [], "EVIDENCE_ISSUES": []}


def api(path, paginate=False, args=()):
    cmd = ["gh", "api", "-H", "Accept: application/vnd.github+json"] + (["--paginate", "--slurp"] if paginate else [])
    p = subprocess.run(cmd + list(args) + [path], capture_output=True, text=True, timeout=300)
    if p.returncode:
        raise RuntimeError(f"{path}: {p.stderr[:300]}")
    out = json.loads(p.stdout)
    return [x for page in out for x in page] if paginate else out


def fetch_repo(repo):
    full, raw = REPOS[repo], HERE / "data" / "raw" / repo
    raw.mkdir(parents=True, exist_ok=True)
    rows, cod = census_rows(repo), coding(repo)
    revert_date = {}                      # reverted PR -> date of its first revert
    for r in rows:
        for p in re.findall(r"\d+", r["reverted_prs"] or ""):
            revert_date[int(p)] = min(revert_date.get(int(p), r["revert_date"]), r["revert_date"])
    for r in cod["ROWS"]:                 # targets that the coding found for rows without one
        if r.get("rollback") and r.get("pr"):
            row = next(c for c in rows if c["revert_pr"] == str(r["rollback"]))
            revert_date.setdefault(r["pr"], row["revert_date"])
    targets = set(revert_date)
    reverts = {int(r["revert_pr"]) for r in rows if r["revert_pr"]}
    relands = {n for r in cod["ROWS"] for n in r["reland"]}
    for n in sorted(targets | reverts | relands | set(cod["EVIDENCE_PRS"])):
        dst = raw / f"pr-{n}.json"
        if dst.exists():
            continue
        try:
            rec = {"pr": api(f"repos/{full}/pulls/{n}"), "files": api(f"repos/{full}/pulls/{n}/files?per_page=100", True),
                   "comments": api(f"repos/{full}/issues/{n}/comments?per_page=100", True)}
            if n in targets:
                rec["timeline"] = api(f"repos/{full}/issues/{n}/timeline?per_page=100", True)
        except RuntimeError as e:
            print(repo, "pr", n, "error:", str(e)[:120])
            continue
        dst.write_text(json.dumps(rec) + "\n")
        print(repo, "pr", n)
    for n in sorted(targets):
        dst = raw / f"author-{n}.json"
        if dst.exists() or not (raw / f"pr-{n}.json").exists():
            continue
        login = json.loads((raw / f"pr-{n}.json").read_text())["pr"]["user"]["login"]
        start = date.fromisoformat(revert_date[n][:10])
        q = f"repo:{full} is:pr author:{login} created:{start}..{start + timedelta(days=150)}"
        dst.write_text(json.dumps(api("search/issues", args=("-X", "GET", "-f", f"q={q}", "-f", "per_page=100"))) + "\n")
        print(repo, "author search", n)
        time.sleep(2.5)                   # the search API allows 30 requests per minute
    for n in cod["EVIDENCE_ISSUES"]:
        dst = raw / f"issue-{n}.json"
        if not dst.exists():
            dst.write_text(json.dumps({"issue": api(f"repos/{full}/issues/{n}"),
                                       "comments": api(f"repos/{full}/issues/{n}/comments?per_page=100", True)}) + "\n")
            print(repo, "issue", n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", choices=sorted(REPOS), action="append")
    for repo in ap.parse_args().repo or sorted(REPOS):
        fetch_repo(repo)


if __name__ == "__main__":
    main()
