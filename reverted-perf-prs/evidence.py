"""Evidence sheets for coding the revert reason of each incident in catalog.py.

A sheet shows, for one incident: the target PR (title, dates, body excerpt); each revert (body and the human
comments); the human comments on the target after its merge; the cross-references to the target and the
revert (issues that report the failure, relands, fixes); the census reason; and the earlier coding of
../perf-revert-study when it has the incident. Bots are left out.

    python3 -I reverted-perf-prs/evidence.py OUT_DIR [--only INCIDENT ...] [--per-sheet 12]
"""
import argparse, json, re, runpy
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent / "perf-revert-study"
BOT = re.compile(r"(\[bot\]$|^(mergify|github-actions|coderabbitai|gemini-code-assist|copilot|sourcery-ai|vllm-bot|"
                 r"sglang-bot|dosubot|codecov|netlify|claude)$)", re.I)


def load(repo, kind, n):
    p = HERE / "data" / "raw" / repo / f"{kind}-{n}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return None if isinstance(d, dict) and d.get("not_found") else d


def clean(text, n):
    text = re.sub(r"<!--.*?-->", "", text or "", flags=re.S)
    text = re.sub(r"(?s)<details>.*?</details>", "", text)
    text = re.sub(r"(?im)^\s*- \[[ x]\].*$", "", text)
    text = re.sub(r"(?s)### CI States.*", "", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)|<img[^>]*>", "[image]", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:n] + (" …" if len(text) > n else "")


def human_comments(repo, n, after="", limit=8, width=320):
    out = []
    for c in load(repo, "comments", n) or []:
        who = (c.get("user") or {}).get("login", "?")
        if BOT.search(who) or (c.get("created_at") or "") < after:
            continue
        body = clean(c.get("body"), width)
        if body and not body.startswith("/"):
            out.append(f"[{c['created_at'][:10]} {who}] {body}")
    return out[:limit]


def cross_refs(repo, n, after="", limit=14):
    out = []
    for ev in load(repo, "timeline", n) or []:
        if ev.get("event") != "cross-referenced":
            continue
        src = (ev.get("source") or {}).get("issue") or {}
        if not src or (ev.get("created_at") or "") < after:
            continue
        kind = "PR" if src.get("pull_request") else "issue"
        merged = ((src.get("pull_request") or {}).get("merged_at") or "")[:10]
        repo_url = src.get("repository_url", "")
        other = "" if repo_url.endswith(("vllm-project/vllm", "sgl-project/sglang")) else f" ({repo_url.split('/repos/')[-1]})"
        out.append(f"{ev['created_at'][:10]} {kind} #{src.get('number')}{other} [{src.get('state')}"
                   f"{', merged ' + merged if merged else ''}] {src.get('title', '')[:110]}")
    return out[:limit]


def earlier_coding():
    rows = {}
    for repo, fn in (("vllm", "coding.py"), ("sglang", "coding_sglang.py")):
        if not (STUDY / fn).exists():  # the earlier study is not in every checkout; the sheets then leave it out
            continue
        for r in runpy.run_path(str(STUDY / fn))["ROWS"]:
            if r.get("pr"):
                rows[(repo, r["pr"])] = r
    return rows


def sheets(only=None):
    cat = runpy.run_path(str(HERE / "catalog.py"))["rows"]()
    old = earlier_coding()
    inc = defaultdict(list)
    for r in cat:
        inc[r["incident"]].append(r)
    order = sorted(inc, key=lambda k: (k.split("-")[0], min(r["revert_date"] for r in inc[k])))
    out = []
    for k in order:
        if only and k not in only:
            continue
        rs, lines = inc[k], [f"######## {k}"]
        for t in sorted({(r["repo"], r["target"]) for r in rs}, key=lambda x: int(x[1])):
            repo, n = t
            pr = load(repo, "pull", n) or {}
            merged = (pr.get("merged_at") or "")
            lines.append(f"TARGET #{n}: {pr.get('title', '?')} | by {(pr.get('user') or {}).get('login', '?')} | "
                         f"merged {merged[:10]} | +{pr.get('additions', '?')}/-{pr.get('deletions', '?')} | "
                         f"labels {[l['name'] for l in pr.get('labels', [])]}")
            lines.append(f"  body: {clean(pr.get('body'), 700)}")
            for c in human_comments(repo, n, after=merged, limit=8):
                lines.append(f"  after-merge comment {c}")
            for x in cross_refs(repo, n, after=merged):
                lines.append(f"  xref {x}")
            if (repo, n) in old:
                o = old[(repo, n)]
                lines.append(f"  EARLIER CODING: cause={o['cause']} symptom={o.get('symptom')} silent={o.get('silent')} "
                             f"detected={o.get('detected')} lane={o.get('lane')!r} reland={o.get('reland')} "
                             f"{o.get('reland_kind')} conf={o.get('conf')} | {o.get('detail')}")
        for r in sorted(rs, key=lambda r: r["revert_date"]):
            rv = load(r["repo"], "pull", r["revert_pr"]) if r["revert_pr"] else None
            lines.append(f"REVERT #{r['revert_pr'] or r['revert_sha']} ({r['revert_status']}, {r['revert_date']}): "
                         f"{(rv or {}).get('title', r['subject'])} | by {((rv or {}).get('user') or {}).get('login', '?')}")
            lines.append(f"  census: {r['census_reason']} | {r['census_evidence'][:200]!r}")
            if rv:
                lines.append(f"  body: {clean(rv.get('body'), 900)}")
                for c in human_comments(r["repo"], r["revert_pr"], limit=6):
                    lines.append(f"  comment {c}")
                for x in cross_refs(r["repo"], r["revert_pr"], limit=8):
                    lines.append(f"  xref {x}")
        out.append((k, "\n".join(lines)))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--per-sheet", type=int, default=12)
    a = ap.parse_args()
    sh = sheets(set(a.only) if a.only else None)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(sh), a.per_sheet):
        (out / f"evidence-{i // a.per_sheet:02d}.txt").write_text("\n\n".join(s for _, s in sh[i:i + a.per_sheet]) + "\n")
    print(len(sh), "incidents in", (len(sh) + a.per_sheet - 1) // a.per_sheet, "sheets")
