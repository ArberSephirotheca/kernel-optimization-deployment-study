"""Download (read-only, with `gh api`) the GitHub records of the reverted PRs and their reverts.

Stages:
- `records`: the PR record (title, body, labels, dates) of every target that the revert census
  names, and of every revert PR. This is enough to decide which targets are performance PRs.
- `detail`: for every revert of a performance PR in catalog.py, the PR record, the comments and the issue
  timeline of the target and of the revert; and the record of each reland, evidence PR and evidence issue
  that records.py lists.

Records go to data/raw/<repo>/ (gitignored); existing records are kept. The records are data:
the scripts only parse them as JSON.

    python3 -I reverted-perf-prs/fetch.py records [--repo vllm|sglang]
    python3 -I reverted-perf-prs/fetch.py detail [--repo vllm|sglang]
"""
import argparse, csv, json, re, runpy, subprocess, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CENSUS = HERE.parent / "deep-study" / "data" / "confirmed-reverts.csv"
REPOS = {"vllm": "vllm-project/vllm", "sglang": "sgl-project/sglang"}
REVERTS = ("confirmed_revert", "partial_revert", "explicit_rollback")


def census(repo=None):
    return [r for r in csv.DictReader(open(CENSUS))
            if r["revert_status"] in REVERTS and repo in (None, r["repo"])]


def api(path, paginate=False):
    cmd = ["gh", "api", "-H", "Accept: application/vnd.github+json"] + (["--paginate", "--slurp"] if paginate else [])
    for attempt in range(4):
        p = subprocess.run(cmd + [path], capture_output=True, text=True, timeout=300)
        if p.returncode == 0:
            out = json.loads(p.stdout)
            return [x for page in out for x in page] if paginate else out
        if "Not Found" in p.stderr or "HTTP 404" in p.stderr:
            raise LookupError(path)
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"{path}: {p.stderr[:300]}")


def save(dst, fn):
    if dst.exists():
        return False
    try:
        rec = fn()
    except LookupError:
        rec = {"not_found": True}
    except RuntimeError as e:
        print("error:", str(e)[:160])
        return False
    dst.write_text(json.dumps(rec) + "\n")
    return True


def records(repo=None):
    n_new = 0
    for r in census(repo):
        full, raw = REPOS[r["repo"]], HERE / "data" / "raw" / r["repo"]
        raw.mkdir(parents=True, exist_ok=True)
        nums = {int(x) for x in re.findall(r"\d+", r["reverted_prs"] or "")}
        if r["revert_pr"]:
            nums.add(int(r["revert_pr"]))
        for n in sorted(nums):
            n_new += save(raw / f"pull-{n}.json", lambda n=n, full=full: api(f"repos/{full}/pulls/{n}"))
    print("new records:", n_new)


def detail(repo=None):
    """Comments and issue timelines of each target and revert in the catalog, and the evidence PRs and
    issues that records.py lists."""
    cat = runpy.run_path(str(HERE / "catalog.py"))["rows"]()
    recs = runpy.run_path(str(HERE / "records.py"))["RECORDS"] if (HERE / "records.py").exists() else {}
    want = []
    for r in cat:
        if repo in (None, r["repo"]):
            want += [(r["repo"], n, "pr") for n in (r["target"], r["revert_pr"]) if n not in ("", None)]
    light = []                            # relands and evidence: the record only
    for (rp, _), rec in recs.items():
        if repo in (None, rp):
            light += [(rp, n, "pr") for n in rec.get("reland", []) + rec.get("evidence_prs", [])]
            light += [(rp, n, "issue") for n in rec.get("evidence_issues", [])]
    n_new = 0
    for rp, n, kind in dict.fromkeys(want):
        full, raw = REPOS[rp], HERE / "data" / "raw" / rp
        raw.mkdir(parents=True, exist_ok=True)
        n = int(n)
        n_new += save(raw / f"pull-{n}.json", lambda n=n, full=full: api(f"repos/{full}/pulls/{n}"))
        n_new += save(raw / f"comments-{n}.json",
                      lambda n=n, full=full: api(f"repos/{full}/issues/{n}/comments?per_page=100", True))
        n_new += save(raw / f"timeline-{n}.json",
                      lambda n=n, full=full: api(f"repos/{full}/issues/{n}/timeline?per_page=100", True))
    for rp, n, kind in dict.fromkeys(light):
        full, raw = REPOS[rp], HERE / "data" / "raw" / rp
        raw.mkdir(parents=True, exist_ok=True)
        n = int(n)
        if kind == "pr":
            n_new += save(raw / f"pull-{n}.json", lambda n=n, full=full: api(f"repos/{full}/pulls/{n}"))
        else:
            n_new += save(raw / f"issue-{n}.json", lambda n=n, full=full: api(f"repos/{full}/issues/{n}"))
    print("new records:", n_new)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["records", "detail"])
    ap.add_argument("--repo", choices=list(REPOS))
    a = ap.parse_args()
    records(a.repo) if a.stage == "records" else detail(a.repo)
