"""Convert analysis.py (percent-format cells) to <Account>.ipynb and (optionally) a zip.  Usage: python make_nb.py Account"""
import re, sys, zipfile
from pathlib import Path
import nbformat as nbf

acct = sys.argv[1] if len(sys.argv) > 1 else "RealDevRay"
src = Path("analysis.py").read_text(encoding="utf8")
cells, cur, kind = [], [], "code"


def flush():
    global cur
    text = "\n".join(cur).strip("\n")
    if text.strip():
        if kind == "markdown":
            text = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in text.splitlines())
            cells.append(nbf.v4.new_markdown_cell(text))
        else:
            cells.append(nbf.v4.new_code_cell(text))
    cur = []


for line in src.splitlines():
    m = re.match(r"# %%(\s*\[markdown\])?\s*$", line)
    if m:
        flush(); kind = "markdown" if m.group(1) else "code"
    else:
        cur.append(line)
flush()
nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
out = Path(f"{acct}.ipynb"); nbf.write(nb, out); print("wrote", out, len(cells), "cells")
