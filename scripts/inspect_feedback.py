"""Per-question diff of the feedback experiment (critic on vs off)."""

import json

d = json.load(open("results/eval.results.json", encoding="utf-8"))
on = d["feedback_experiment"]["critic_on"]["per_question"]
off = d["feedback_experiment"]["critic_off"]["per_question"]

for a, b in zip(on, off):
    if a["coverage"] != b["coverage"]:
        print(f"ON={a['coverage']:.2f} OFF={b['coverage']:.2f} rev={a['revisions']}")
        print(f"  Q: {a['question']}")
        for fb in a["feedback"]:
            print(f"  feedback: {fb[:110]}")
