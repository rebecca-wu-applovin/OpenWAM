"""Build VERIFIER_LOGIC.md: the verifier's logic with clickable code pointers.

Pointers are written as {{file::regex}} in the template below and resolved to the first matching line, so the
document stays correct when the code moves: re-run after every code change.
  python benchmarks/sharpa_webxr/_build_logic_doc.py [--sens sensitivity.json] [--calib calib.tsv]
Links are relative to this directory (they open in VS Code / GitHub markdown preview).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "VERIFIER_LOGIC.template.md"
OUT = HERE / "VERIFIER_LOGIC.md"


def resolve(m: re.Match) -> str:
    rel, pat = m.group(1).strip(), m.group(2).strip()
    path = HERE / rel
    lines = path.read_text().splitlines()
    rx = re.compile(pat)
    for k, line in enumerate(lines, 1):
        if rx.search(line):
            sym = re.match(r"\s*(?:def|class)\s+(\w+)|\s*([A-Z_]+)\s*=", line)
            name = (sym.group(1) + ("()" if "def " in line else "")) if sym and sym.group(1) else (sym.group(2) if sym else None)
            label = f"{name} — {rel}:{k}" if name else f"{rel}:{k}"
            return f"[`{label}`]({rel}#L{k})"
    raise SystemExit(f"pointer not found: {rel} :: {pat}")


def task_appendix() -> str:
    """One row per task: the stage chain with every parameter the spec sets (the effective thresholds are these plus
    the defaults in the check function), and a pointer to the line in _build_task_specs.py."""
    specs = json.loads((HERE / "task_specs.json").read_text())
    lines = (HERE / "_build_task_specs.py").read_text().splitlines()
    out = ["| task | what is asked | stage chain (check: parameters set in the spec) | spec line |", "|---|---|---|---|"]
    for k, sp in sorted(specs.items()):
        key = k[:5]
        ln = next((i for i, l in enumerate(lines, 1) if l.strip().startswith(f'"{key}"')), None)
        ptr = f"[`_build_task_specs.py:{ln}`](_build_task_specs.py#L{ln})" if ln else ""
        if sp.get("skip"):
            out.append(f"| {key} | {sp.get('task', '')} | **skipped**: {sp['skip']} | {ptr} |")
            continue
        chain = []
        for st in sp["stages"]:
            params = {x: y for x, y in st.items() if x not in ("name", "check", "obj", "target")}
            ps = ", ".join(f"{x}={json.dumps(y)}" for x, y in params.items())
            chain.append(f"{st['check']}" + (f"({ps})" if ps else ""))
        out.append(f"| {key} | {sp.get('task', '')} | " + " → ".join(chain).replace("|", "/") + f" | {ptr} |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sens", type=Path, default=None, help="sensitivity table json (from the threshold sweep)")
    ap.add_argument("--calib", type=Path, default=None, help="calibrate.py --tsv output")
    a = ap.parse_args()
    text = TEMPLATE.read_text()
    text = re.sub(r"\{\{([^:{}]+)::(.+?)\}\}", resolve, text)
    if a.sens and a.sens.exists():
        text = text.replace("<!-- SENSITIVITY -->", a.sens.read_text())
    if a.calib and a.calib.exists():
        rows = [l.split("\t") for l in a.calib.read_text().splitlines()]
        head, body = rows[0], rows[1:]
        tab = ["| task | category | success / recover / failure | recall | specificity |", "|---|---|---|---|---|"]
        for r in body:
            d = dict(zip(head, r))
            tab.append(f"| {d['scene'][:5]} | {d['category']} | {d['n_success']} / {d['n_recover']} / {d['n_failure']} | {d['recall']} | {d['specificity']} |")
        text = text.replace("<!-- CALIBRATION -->", "\n".join(tab))
    text = text.replace("<!-- TASKS -->", task_appendix())
    OUT.write_text(text)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
