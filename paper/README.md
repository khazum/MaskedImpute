# Genome Biology manuscript workspace

The prior NeurIPS draft, generated tables, figures, and associated results are
preserved under `historical/v26_neurips/neurips_paper/` as development evidence.
They are not evidence for the new submission.

The Genome Biology manuscript is structured as a Methodology article in
`manuscript.tex`. Numerical conclusions remain visibly marked until they can be
rendered from sealed, machine-readable publication assets after method freeze
and the one-use final evaluation. Author metadata and declarations remain
absent until supplied and approved by the authors.

The vendored class and Vancouver bibliography style come from the official
Springer Nature LaTeX template, version 3.1 (December 2024). The source archive
checksum is recorded in `submission_checklist.md`. From this directory, use:

```bash
latexmk -pdf -halt-on-error manuscript.tex
```

A successful draft build does not imply submission readiness. Every red
`PENDING` marker and every unchecked scientific or author/legal item must be
resolved first.
