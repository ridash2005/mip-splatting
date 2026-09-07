RUNS   := results/runs.csv
FIGDIR := figures

.PHONY: selftest table1 table2 figures figures-published report clean-figures

# The reporting chain, verified against a fixture with known answers. Needs no
# GPU, so it runs on the dev machine as well as inside the Kaggle kernel.
selftest:
	python tools/selftest_pipeline.py

# ITERS selects which runs to average: 30000 is the R1/R2 target, 7000 the R0
# smoke rows. Mixing them would produce a number describing neither run.
ITERS  ?= 30000

table1:
	python tools/make_table.py --table 1 --runs $(RUNS) --iterations $(ITERS) --md results/table1.md

table2:
	python tools/make_table.py --table 2 --runs $(RUNS) --iterations $(ITERS) --md results/table2.md

figures:
	python tools/make_figures.py --mode measured --runs $(RUNS) --iterations $(ITERS) --out $(FIGDIR)

# The §13 report for one rung, rendered from the summary.json its kernel wrote.
#   make report SUMMARY=results/kaggle_runs/r0_smoke_v3/summary.json OUT=results/R0-report.md
SUMMARY ?= results/kaggle_runs/r0_smoke_v3/summary.json
OUT     ?= results/R0-report.md

report:
	python tools/report_rung.py $(SUMMARY) --md $(OUT)

# Target-only reference figures (published numbers, never our measurements).
# Kept separate so `make figures` can never be satisfied by them.
figures-published:
	python tools/make_figures.py --mode published --out $(FIGDIR)/published

clean-figures:
	rm -rf $(FIGDIR)
