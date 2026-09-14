# Submission

What is handed to the panel. Both files are build output — `make deliver`
writes them here from `thesis/main.pdf` and `slides/BTP-Panel.pptx`, and both
regenerate from `results/runs.csv` by `make all`. They are committed so that
the exact files submitted are recoverable at the tag that submitted them.

| file | what it is |
|---|---|
| `BTP-Thesis.pdf` | the thesis, 97 pages |
| `BTP-Panel-Presentation.pptx` | the panel deck, 16 slides |
| `BTP-Panel-Presentation.pdf` | the same deck exported to PDF |

The PDF export is the one step `make deliver` does not do: PowerPoint has to
produce it. Re-export it whenever the deck changes, or it will quietly show an
older set of slides than the `.pptx` beside it.

The state these were built from is tagged `thesis-autumn-2026`.
