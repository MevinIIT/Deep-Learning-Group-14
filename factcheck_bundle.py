# factcheck_bundle.py -- bundles the helper files into ONE self-contained factcheck.py.
# The coursework only accepts factcheck.py, so the helpers are inlined.
# Run:  python factcheck_bundle.py      -> writes bundle/factcheck.py
import os

src = open("factcheck.py").read()
for module, line in [("preprocessing", "from preprocessing import tokenize"),
                     ("overlap_metrics", "from overlap_metrics import METRICS")]:
    assert line in src, "import line not found: " + line
    body = open(module + ".py").read()
    src = src.replace(line, "# ---- inlined from " + module + ".py ----\n" + body + "\n# ---- end " + module + ".py ----", 1)

os.makedirs("bundle", exist_ok=True)
open("bundle/factcheck.py", "w").write(src)
print("wrote bundle/factcheck.py -> copy it next to factchecking_main.py in a clean folder and run both --mode commands")