"""Inject results/summary.json and results/simulation.json into report/template.html."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(ROOT, "results", "summary.json")) as f:
    data = json.load(f)
with open(os.path.join(ROOT, "results", "simulation.json")) as f:
    sim = json.load(f)
with open(os.path.join(ROOT, "report", "template.html")) as f:
    page = f.read()
page = page.replace("__DATA__", json.dumps(data)).replace("__SIM__", json.dumps(sim))
with open(os.path.join(ROOT, "report", "index.html"), "w") as f:
    f.write(page)
print("wrote report/index.html")
