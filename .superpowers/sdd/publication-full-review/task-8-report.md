# Task 8 Genome Biology package review report

## Status

Implementation and self-verification are complete. Independent acceptance is
not claimed. The package is a compilable, fail-closed Methodology draft; it is
not submission-ready and does not establish empirical competitiveness.

## Current venue authority

The review used the Genome Biology general and Methodology instructions
verified on 23 July 2026:

- <https://link.springer.com/journal/13059/submission-guidelines>
- <https://link.springer.com/journal/13059/submission-guidelines/methodology>

The Methodology criteria still require evidence of an outstanding novel method,
a side-by-side advance over current state-of-the-art methods, same-dataset
computational comparison, known-truth or synthetic benchmarking where possible,
and real-data utility. None of those empirical criteria is marked complete
without frozen results.

The draft now follows the required front-matter and section contract: an
unstructured 62-word placeholder-bearing abstract, six keywords, Background,
Results, Discussion, Conclusions, Methods, Abbreviations, all seven required
declaration headings, and References. The `referee` and `lineno` class options
provide double spacing and line numbering, and page numbering is explicit.

## Findings and corrections

| ID | Severity | Finding | Disposition |
|---|---|---|---|
| F-103 | Important | The title page silently omitted author names, institutional addresses, email addresses, and the corresponding-author designation. Although the README described those fields as absent, the rendered draft did not show the blocking omission. | Closed without inventing people or institutions. The title page now renders explicit red author-input blockers for the approved author list and order, affiliation mapping, all author emails, and corresponding author. |
| F-104 | Important | The abstract said all analyses were already generated, and the external-reference Results prose said development evidence was reported, although no scientific workload or frozen analysis exists. | Closed. The 62-word draft abstract describes the prespecified benchmark, requests the sealed result and scope-qualified conclusion, and explicitly states that the draft makes no empirical advance claim. External-reference evidence is future tense. |
| F-105 | Important | The compact checklist marked the state-of-the-art advancement exercise and final abstract constraint complete even though results and final abstract wording are unavailable. This could be read as satisfying the Methodology acceptance criteria prematurely. | Closed. Structural facts remain checked, while clear-advance, same-dataset, known-truth, real-data, and final-abstract evidence remain explicit unchecked blockers. |
| F-106 | Minor | The manuscript lacked the Abbreviations section named by the project venue contract and used several abbreviations without expansion at first use. | Partially closed in the initial correction: an Abbreviations section was added and the identified prose terms were expanded, but the first rendered `IQR` remained unexpanded and absent from the section. The residual defect is tracked as F-111. |
| F-107 | Minor | Both checklists retained excluded template-integrity validation gates, and the paper README pointed readers to the compact instance. | Closed as explicitly required by Task 8. The two active gates and the stale README reference were removed. Ordinary template provenance, differing upstream notices, source-license separation, and publisher-redistribution caveats remain. No replacement integrity mechanism was added. |
| F-108 | Minor | The checklists and manuscript phrased a static DOI-issuing deposition as mandatory, while the current Methodology instructions recommend it. | Partially closed in the initial correction: the recommendation and author decision were stated, but unconditional archive-identifier inputs remained in the declaration and checklists. The residual defect is tracked as F-110. |
| F-109 | Minor | The DCA bibliography record used a malformed TeX accent escape for Gökcen Eraslan. | Closed with the conventional BibTeX escape. All twelve cited DOI, title, year, and venue records matched registry metadata, and all citations resolve in the built manuscript. |
| F-110 | Minor | The manuscript declaration, compact author-input checklist, and full availability checklist still required a static software archive or its identifier regardless of the authors' deposition decision. | Closed test-first. Public source and data access and an OSI-compliant license remain unconditional. Static deposition is conditional on the author decision; if an archive is created, its persistent identifier, citation, and consistency with the manuscript release remain fail-closed. |
| F-111 | Minor | The first rendered use of `IQR` occurred in a pending Results marker without expansion, while the Abbreviations section omitted it and the compact checklist claimed first-use consistency. | Closed test-first. The marker now expands interquartile range (IQR) at first use, and the alphabetized Abbreviations section includes its canonical expansion. |

The paper README now documents the exact
`pdflatex`/`bibtex`/`pdflatex`/`pdflatex` build sequence used in review.
Publisher class and bibliography-style notices remain unmodified.

## Test-first evidence

The focused venue-contract tests initially reported:

```text
4 failed, 53 deselected in 0.11s
```

The failures reproduced the missing fail-closed front matter, missing
Abbreviations/order boundary, unexecuted-analysis completion language, and
evidence-dependent checked checklist items.

After the bounded corrections, the same focused set reported:

```text
4 passed, 53 deselected in 0.06s
```

After formatting, the new checks together with the existing manuscript
contracts reported:

```text
7 passed, 175 deselected in 2.01s
```

Follow-up review exposed F-110 and F-111. Their two focused regression tests
first reported:

```text
2 failed in 0.09s
```

After the bounded corrections, the same two tests reported:

```text
2 passed in 0.03s
```

The complete repository-hygiene owner reported:

```text
59 passed in 26.11s
```

The existing manuscript/comparator documentation subset reported:

```text
3 passed in 2.16s
```

The active obsolete-term checks reported:

```text
2 passed in 0.80s
```

## Build and static evidence

A clean output state was built with:

```text
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
bibtex manuscript
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
```

Every command exited zero. The final PDF contained 14 A4 pages. Log scans found
no unresolved citation, unresolved reference, fatal error, emergency stop, or
overfull box. Extracted text showed the visible title-page blockers, abstract,
keywords, ordered main sections, the first-use interquartile range (IQR)
expansion, the canonical IQR abbreviation entry, Declarations, the conditional
static-archive input, all seven declaration subheadings, and References.

Ruff 0.14.4 reported all 164 Python files formatted and all checks passing.
Scoped Python compilation and `git diff --check` exited zero. Generated
manuscript products and Python/test/linter caches were removed after
inspection.

## Unresolved submission blockers

Scientific blockers remain:

- the real development, final, trajectory, downstream, and scaling workloads
  have not been run;
- no sealed result, figure, table, interval, ranking, safety conclusion, or
  superiority statement is available;
- a clear side-by-side advance over current state-of-the-art methods on common
  datasets, known-truth performance, and real-data utility remain unevaluated;
- every red `PENDING SEALED EVIDENCE` marker must be replaced only from the
  final permitted publication evidence.

Human, legal, and external blockers remain:

- author list, order, affiliations, email addresses, correspondence, ORCIDs,
  approval, and contributions;
- ethics and consent determinations, competing interests, funding,
  acknowledgements, and the author-approved LLM disclosure;
- an OSI-compliant project license, live public source location, public
  conclusion-supporting and third-party data, accessions and reviewer links;
- the author decision on recommended static deposition and any resulting
  archive identifier;
- figure, table, supplement, data/code statement, Minimum Standards, preprint,
  cover-letter, redistribution, and final submission authorization checks.

No human metadata, institutional determination, license choice, repository or
archive identifier, scientific result, or submission authorization was
invented. No real scientific workload ran, and no estimand, population,
selection rule, method configuration, seed policy, or claim permission changed.
