"""MATRIX.md and ISSUE-MAP.md from results (a list of results.jsonl records)."""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from bench.instruments.reliability.cells import ISSUES, registry  # noqa: E402

BOUNDARY = ("Claim class: advisory / code_green_local. A PASS proves this lcm-x tree, on this host sha, under this "
            "scripted in-process scenario, meets the bars; not behaviour under real models, the real ACP/gateway "
            "processes, real transports or customer boxes.")


def cell_word(r: dict) -> str:
    if r["verdict"] == "FAIL":
        return "FAIL " + "/".join(sorted(r.get("failed_bars", {})))
    return r["verdict"]


def issue_status(issue: int, rows: list[dict], capability: str, bars: tuple) -> str:
    live = [r for r in rows if r["verdict"] in ("PASS", "FAIL")]
    if not live:
        why = "; ".join(sorted({str(r.get("reason"))[:90] for r in rows})) if rows else capability
        return f"NOT COVERED ({why or 'no targeting cell'})"
    hits = [r["cell"] for r in live if set(r.get("failed_bars", {})) & set(bars)]
    if hits:
        return "REPRODUCES (" + ", ".join(sorted(set(hits))) + ")"
    return "DOES NOT REPRODUCE" + (" (some targeting cells ERROR/UNSUPPORTED)" if len(live) < len(rows) else "")


def write(out: Path, results: list[dict], wall: float) -> None:
    hosts = sorted({r["host"] for r in results})
    refs = sorted({(r["plugin_ref"], r["plugin_sha"][:12]) for r in results})
    order = [c["id"] for c in registry()]
    lines = ["# Reliability matrix (R1)", "", BOUNDARY, "", f"Wall clock: {wall:.0f} s for {len(results)} cells.", ""]
    for ref, sha in refs:
        rs = [r for r in results if r["plugin_sha"][:12] == sha]
        by = {(r["cell"], r["host"]): r for r in rs}
        lines += [f"## lcm-x `{ref}` ({sha})", "", "| cell | " + " | ".join(hosts) + " |", "|---|" + "---|" * len(hosts)]
        for cid in [c for c in order if any((c, h) in by for h in hosts)]:
            lines.append(f"| `{cid}` | " + " | ".join(cell_word(by[(cid, h)]) if (cid, h) in by else "-" for h in hosts) + " |")
        lines += ["", "| host | host sha | PASS | FAIL | ERROR | UNSUPPORTED |", "|---|---|---|---|---|---|"]
        for h in hosts:
            n = Counter(r["verdict"] for r in rs if r["host"] == h)
            sha_h = next((r["host_sha"][:10] for r in rs if r["host"] == h), "")
            lines.append(f"| {h} | {sha_h} | {n['PASS']} | {n['FAIL']} | {n['ERROR']} | {n['UNSUPPORTED']} |")
        lines += ["", "### Failed bars, errors and unsupported cells", ""]
        for r in sorted(rs, key=lambda r: (r["cell"], r["host"])):
            if r["verdict"] == "FAIL":
                detail = {b: {k: v for k, v in d.items() if not isinstance(v, (list, dict)) or k in ("failed_turns",)}
                          if isinstance(d, dict) else d for b, d in r["failed_bars"].items()}
                lines.append(f"- `{r['cell']}` on {r['host']}: `{json.dumps(detail, default=str)[:600]}`")
            elif r["verdict"] in ("ERROR", "UNSUPPORTED"):
                lines.append(f"- {r['verdict']} `{r['cell']}` on {r['host']}: {str(r.get('reason'))[:300]}")
        lines.append("")
    (out / "MATRIX.md").write_text("\n".join(lines) + "\n")

    targeting = defaultdict(list)
    for c in registry():
        for t in c["targets"]:
            targeting[t].append(c["id"])
    im = ["# Issue map (R1)", "", BOUNDARY, "",
          "REPRODUCES = a targeting cell FAILs on that issue's bar. Per host at the evaluated ref.", ""]
    for ref, sha in refs:
        rs = [r for r in results if r["plugin_sha"][:12] == sha]
        im += [f"## lcm-x `{ref}` ({sha})", "", "| issue | bar | targeting cells | " + " | ".join(hosts) + " |",
               "|---|---|---|" + "---|" * len(hosts)]
        for issue, (issue_bars, capability) in ISSUES.items():
            cells_for = targeting.get(issue, [])
            row = [str(issue), "/".join(issue_bars), ", ".join(f"`{c}`" for c in cells_for) or "-"]
            for h in hosts:
                hr = [r for r in rs if r["host"] == h and r["cell"] in cells_for]
                if not hr and cells_for:
                    row.append("not run")
                else:
                    row.append(issue_status(issue, hr, capability, issue_bars))
            im.append("| " + " | ".join(row) + " |")
        im.append("")
    (out / "ISSUE-MAP.md").write_text("\n".join(im) + "\n")


if __name__ == "__main__":  # re-render from an existing results.jsonl: python report.py <out>
    target = Path(sys.argv[1])
    recs = [json.loads(x) for x in (target / "results.jsonl").read_text().splitlines() if x.strip()]
    write(target, recs, sum(r.get("wall_s", 0) for r in recs))
