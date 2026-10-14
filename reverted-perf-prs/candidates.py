"""The candidate set: every target of every revert or rollback in the B1 census, with what is needed
to decide whether the target is a performance PR (codebook A1 of the deep study).

For each (repo, revert, target) it joins:
- the census row (revert status, census class, census reason and its quote);
- the A1 population verdict, and whether a coder adjudicated it or a rule decided it;
- the target's PR record from data/raw (title, labels, full body), else the census input excerpt.

    python3 -I reverted-perf-prs/candidates.py --csv            # writes candidates.csv
    python3 -I reverted-perf-prs/candidates.py --dump FILE      # a reading sheet for classification
"""
import argparse, csv, json, re
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "deep-study" / "data"
BATCHES = HERE.parent / "deep-study" / "coding" / "batches" / "b1_reverts"
REVERTS = ("confirmed_revert", "partial_revert", "explicit_rollback")
PERF_WORDS = re.compile(r"\b(perf|performance|optimi[sz]|speed[- ]?up|faster|accelerat|latency|throughput|overhead|"
                        r"tpot|ttft|itl|tok/s|tokens?/s|e2e|fus(e|ed|ion)|tun(e|ed|ing)|cuda ?graph|overlap|"
                        r"reduce[sd]? (the )?(memory|time|cost|launch)|x faster|\d+(\.\d+)?\s?%|\d+(\.\d+)?x\b)", re.I)


def census_inputs():
    recs = {}
    for f in sorted(BATCHES.glob("batch-*-input.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                recs[d["id"]] = d
    return recs


def pull(repo, n):
    p = HERE / "data" / "raw" / repo / f"pull-{n}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return None if d.get("not_found") else d


def strip(body):
    body = re.sub(r"<!--.*?-->", "", body or "", flags=re.S)
    return re.sub(r"\n\s*\n+", "\n", body).strip()


def rows():
    pop = {(r["repo"], int(r["number"])): r for r in csv.DictReader(open(DATA / "performance-pr-population.csv"))}
    inputs = census_inputs()
    out = []
    for c in csv.DictReader(open(DATA / "confirmed-reverts.csv")):
        if c["revert_status"] not in REVERTS:
            continue
        rid = f"{c['repo']}:{c['revert_sha'][:10]}"
        inp = inputs.get(rid, {})
        excerpts = {t["pr"]: t for t in inp.get("targets", []) if t.get("pr")}
        nums = [int(x) for x in re.findall(r"\d+", c["reverted_prs"] or "")] or [None]
        for n in nums:
            a1 = pop.get((c["repo"], n)) if n else None
            pr = pull(c["repo"], n) if n else None
            ex = excerpts.get(n, {})
            out.append({
                "repo": c["repo"], "revert_sha": c["revert_sha"][:10], "revert_pr": c["revert_pr"],
                "revert_date": c["revert_date"][:10], "revert_status": c["revert_status"],
                "subject": c["subject"], "target": n or "",
                "title": (pr or {}).get("title") or ex.get("title") or c["reverted_title"],
                "merged": ((pr or {}).get("merged_at") or ex.get("merged_at") or c["reverted_merged_at"] or "")[:10],
                "labels": " ".join(l["name"] for l in (pr or {}).get("labels", [])) or " ".join(ex.get("labels", [])),
                "body": strip((pr or {}).get("body") if pr else ex.get("body")),
                "body_source": "pull" if pr else ("census input" if ex else "none"),
                "census_class": c["reverted_class"], "census_reason": c["revert_reason"],
                "census_evidence": c["reason_evidence"], "census_rationale": c["rationale"],
                "a1_class": a1["classification"] if a1 else "", "a1_type": a1["perf_type"] if a1 else "",
                "a1_source": a1["classification_source"] if a1 else "", "a1_evidence": a1["evidence"] if a1 else "",
            })
    return out


def dump(rs, path):
    with open(path, "w") as f:
        for i, r in enumerate(rs):
            hits = [l.strip()[:200] for l in r["body"].splitlines() if PERF_WORDS.search(l)][:6]
            f.write(f"### [{i}] {r['repo']} #{r['target'] or '?'} (revert {r['revert_pr'] or r['revert_sha']}, "
                    f"{r['revert_status']}, {r['revert_date']})\n")
            f.write(f"title: {r['title']}\nlabels: {r['labels'] or '-'} | census: {r['census_class']} / "
                    f"{r['census_reason']} | A1: {r['a1_class'] or '-'} {r['a1_type']} ({r['a1_source'] or '-'})\n")
            if r["a1_evidence"] and r["a1_source"].startswith("adjudicated"):
                f.write(f"A1 evidence: {r['a1_evidence'][:200]}\n")
            f.write(f"body[{r['body_source']}]: {r['body'][:420].replace(chr(10), ' / ')}\n")
            if hits:
                f.write("perf lines: " + " || ".join(hits) + "\n")
            f.write("\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="store_true")
    ap.add_argument("--dump")
    a = ap.parse_args()
    rs = rows()
    print(len(rs), "candidate rows")
    if a.csv:
        keys = [k for k in rs[0] if k != "body"]
        with open(HERE / "candidates.csv", "w", newline="") as f:
            w = csv.DictWriter(f, keys, extrasaction="ignore", lineterminator="\n")
            w.writeheader()
            w.writerows(rs)
    if a.dump:
        dump(rs, a.dump)
