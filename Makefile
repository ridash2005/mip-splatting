RUNS   := results/runs.csv
FIGDIR := figures

.PHONY: table1 table2 figures figures-published clean-figures

table1:
	python tools/make_table.py --table 1 --runs $(RUNS) --md results/table1.md

table2:
	python tools/make_table.py --table 2 --runs $(RUNS) --md results/table2.md

figures:
	python tools/make_figures.py --mode measured --runs $(RUNS) --out $(FIGDIR)

# Target-only reference figures (published numbers, never our measurements).
# Kept separate so `make figures` can never be satisfied by them.
figures-published:
	python tools/make_figures.py --mode published --out $(FIGDIR)/published

clean-figures:
	rm -rf $(FIGDIR)
