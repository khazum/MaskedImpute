# Genome Biology Methodology Submission Checklist

**Status:** Blocking human-author checklist

**Guidance verified:** 23 July 2026

This checklist deliberately records requirements, not answers. An unchecked item blocks submission. The manuscript is currently a sealed-evidence draft, not submission-ready. The authors may mark an item not applicable only after making and recording the underlying determination; no contributor or automated system may infer names, affiliations, ethics decisions, consent, funding, contributions, conflicts, repository identifiers, preprint details, or disclosure wording.

## Official sources

- [Genome Biology Methodology instructions](https://link.springer.com/journal/13059/submission-guidelines/methodology)
- [Genome Biology general submission guidelines](https://link.springer.com/journal/13059/submission-guidelines)
- [BMC editorial policies](https://link.springer.com/brands/bmc/editorial-policies)
- [BioMed Central Minimum Standards of Reporting Checklist](https://resource-cms.springernature.com/springer-cms/rest/v1/content/7117202/data/v2)
- [Springer Nature LaTeX author support](https://www.springernature.com/gp/authors/campaigns/latex-author-support)
- [Official Springer Nature LaTeX package, version 3.1 (December 2024)](https://cms-resources.apps.public.k8s.springernature.io/springer-cms/rest/v1/content/18782940/data/v12)

Recheck every official page immediately before submission and record that the manuscript still conforms to the then-current instructions.

## 1. Methodology article fit

- [ ] The authors confirm that MaskImpute is a substantive novel method, not only a faster implementation of an existing method or a pipeline combining existing methods.
- [ ] The manuscript explains why the method should be useful to a wide genomics or post-genomics audience.
- [ ] Current state-of-the-art methods are compared side by side on the same datasets, with known-ground-truth and real-data demonstrations where possible.
- [ ] The frozen evidence supports the manuscript's exact advance claim. If it supports competitiveness but not a clear advance, the authors have completed a presubmission enquiry or selected a more suitable article type or venue.
- [ ] Every superiority statement is limited to the named endpoint and comparison supported by the prespecified multiplicity-aware analysis.

### Prespecified fair-comparison evidence

- [ ] The comparator smoke receipt records all 34 smoke completions before the complete development competition starts.
- [ ] The development execution-status table contains exactly 2,896 scheduled rows: 16 observed-count rows, 48 capacity-matched autoencoder rows, 1,200 MaskImpute rows, and 1,632 comparator rows.
- [ ] Readiness includes both required controls, all five established comparators, and at least three modern comparators.
- [ ] BiAEImpute has an explicit eligible, failed, timeout, resource-exceeded, or unavailable status; its status is not inferred from omission.
- [ ] Every eligible comparator has exactly one complete selected payload and complete method bindings, and every ineligible comparator retains complete reason-coded failure counts.
- [ ] The selected payloads and method identities used for candidate assessment are unchanged in final, trajectory, and scaling execution.
- [ ] The final execution-status table contains exactly 1,760 scheduled rows, and the trajectory execution-status table contains exactly 44 trajectory rows.
- [ ] Every scheduled method appears in the complete execution-status table, including failures and unavailable methods.
- [ ] No performance, ranking, or superiority claim is made against unavailable methods or for an estimand lacking an eligible selected configuration.
- [ ] Comparator selection is development-only and cannot access MaskImpute performance, downstream endpoints, or final data.

## 2. Author identity, authorship, and approval

- [ ] The title page contains the full name, institutional address, and email address of every author.
- [ ] One or more corresponding authors are identified, and their contact details are correct.
- [ ] The author order has been supplied and approved by every author.
- [ ] Every listed author made a qualifying contribution, approved the submitted version, accepts accountability for their own contribution, and agrees that integrity questions about any part of the work will be investigated and resolved.
- [ ] `Authors' contributions` uses author initials and describes each individual's actual contribution.
- [ ] Every contributor who does not meet the authorship criteria is handled in `Acknowledgements`, and each named person has permitted the acknowledgement.
- [ ] All authors have read and approved the manuscript, figures, tables, supplement, data statement, code statement, and submission.
- [ ] An author, not a third party or automated system, will submit the manuscript.
- [ ] The submission files retain author names and affiliations because Genome Biology uses single-anonymous, not double-anonymous, review.

## 3. LLM and AI disclosure

- [ ] No LLM or AI system is listed as an author.
- [ ] The authors have approved an accurate Methods statement describing substantive LLM use in study design, software development, analysis, or manuscript preparation. The statement distinguishes such use from copyediting alone and identifies the human validation and accountability applied to the output.
- [ ] The final AI-use disclosure remains in Methods and contains no unresolved placeholder.
- [ ] Human authors have reviewed every AI-assisted factual statement, citation, analysis, code path, figure, and conclusion against primary evidence or executable results.
- [ ] No prohibited generative-AI image is included. Any policy exception or non-generative image manipulation has been disclosed as required by the current BMC policy.

## 4. Required Declarations

Keep all seven required headings. When a heading is not relevant, retain it and use the journal-appropriate `Not applicable` statement only after author review. `Authors' information` may be omitted because it is optional.

### Ethics approval and consent to participate

- [ ] The authors have assessed every human-data accession, the original studies' approval and consent information, applicable data-use terms, and whether this secondary analysis requires local ethics review, exemption, or waiver.
- [ ] The final statement accurately gives the approving or exempting committee and reference number where appropriate, including a waiver when applicable; otherwise the authors have explicitly justified the journal-appropriate not-applicable statement.
- [ ] Any animal-data use has likewise been assessed and reported under the applicable approval and consent rules.

### Consent for publication

- [ ] The authors have determined whether the manuscript contains any identifiable individual's details, images, video, or other personal data.
- [ ] Required written consent for publication has been obtained and stated, or the authors have supplied the journal-appropriate not-applicable statement.

### Availability of data and materials

- [ ] The statement identifies where the minimal data needed to interpret, reproduce, and build on every reported result can be found.
- [ ] Every repository name, persistent HTTPS link, accession, DOI, restriction, and access condition in the statement is live and exact.

### Competing interests

- [ ] Every author has declared all financial and non-financial competing interests, or all authors have approved the journal's no-competing-interests statement.
- [ ] Authors are referred to by initials where individual disclosures are needed.

### Funding

- [ ] Every funding source is declared.
- [ ] Each funder's role in conceptualization, design, data collection, analysis, the decision to publish, and manuscript preparation is stated, or the authors have supplied the appropriate no-funding statement.

### Authors' contributions

- [ ] The contribution statement is complete, uses the approved initials, and ends with confirmation that all authors read and approved the final manuscript when appropriate.

### Acknowledgements

- [ ] All non-author contributions, including writing or technical assistance and supplied materials, are acknowledged with permission, or the heading contains `Not applicable`.

## 5. Public data, code, and software record

- [ ] Every dataset supporting a conclusion is available in a recognized public repository, the article, or an additional file, except for a restriction that is ethically or legally necessary and fully described.
- [ ] Every third-party or previously published dataset remains accessible under terms compatible with publication and reviewer evaluation.
- [ ] Private reviewer-access links are supplied where public release is not yet possible, without requiring reviewers to reveal their identity.
- [ ] Every public dataset is cited in the reference list with a persistent accession or DOI, and generated datasets/results have machine-readable deposits.
- [ ] The public MaskImpute source repository URL is live; no placeholder remains.
- [ ] The repository contains the exact source and in-house analysis scripts used for the manuscript, or any permitted supplementary-script location is stated explicitly.
- [ ] MaskImpute has an identified OSI-compliant source-code license, and the repository and manuscript state the same license.
- [ ] The authors have acted on the journal's recommendation to create a static archived release, or have recorded a decision not to do so; if created, the release has a DOI or other unique identifier, is cited, and is bound to the reported tag or commit.
- [ ] The availability statement gives the project name, project home page, archived release, supported operating systems, programming language, other requirements, license, and any use restriction.
- [ ] Public source, archive, environment locks, container, commands, accession ledger, raw long-form results, machine-readable tables, and smoke workflow identify one internally consistent release.
- [ ] Supporting data are not hosted only on a personal or departmental website.

## 6. Minimum Standards of Reporting

- [ ] The Methods section states the exact sample size as a number for every experimental group and condition.
- [ ] The manuscript explains the considerations that determined each sample size and reports a power analysis if one was performed or was appropriate.
- [ ] Individual biological-draw values are shown in addition to summary statistics wherever `n < 6`, including the five-draw within-mechanism comparisons in this design.
- [ ] Biological replicates, technical views, model seeds, and matrix entries are clearly distinguished; none is allowed to inflate the biological sample size.
- [ ] Sample or dataset collection and all inclusion and exclusion criteria are reported, including failed runs, timeouts, and unavailable combinations.
- [ ] Allocation to groups or conditions and all relevant randomization procedures are reported; any non-random allocation is stated.
- [ ] Blinding of human assessment to group assignment or outcomes is described where relevant; a reason is given where blinding is not relevant or feasible.
- [ ] The number of times each experiment was replicated and the variation across repetitions are reported.
- [ ] Statistical methods identify one- or two-sided tests, multiple-comparison adjustment, means or medians, interval construction, and the meaning of every error bar.
- [ ] The manuscript justifies each inferential procedure, addresses its assumptions, and reports variation within the compared groups.
- [ ] Figure and table legends contain information essential to interpreting the displayed data.
- [ ] Every software tool, database, service, dataset, and other material is identified by provider and version or persistent identifier; the authors have considered the journal's RRID recommendation and included applicable identifiers.

## 7. Manuscript structure and content

- [ ] The manuscript order is title page; Abstract; Keywords; Background; Results; Discussion; Conclusions; Methods; Abbreviations; Declarations; References; followed by the project's additional-file listing when files are supplied.
- [ ] The unstructured Abstract is at most 100 words, has no citations, and minimizes abbreviations.
- [ ] There are 3–10 keywords representing the article's content.
- [ ] Background states the study aims, relevant literature, and why the study was necessary.
- [ ] Results reports the findings and statistical analysis in text, tables, or figures without unsupported claims.
- [ ] Discussion places the findings in context, states limitations, and addresses practical or operational issues for using the method.
- [ ] Conclusions states the main conclusions and their importance without exceeding the frozen evidence.
- [ ] Methods follows Conclusions and reports aim, design, setting, materials, processes, comparisons, statistics, sample-size rationale, and software requirements.
- [ ] Every abbreviation is defined at first use, and the Abbreviations section is consistent with the text.
- [ ] References follow the journal's Vancouver style, and cited web resources and public datasets have complete reference entries.

## 8. Formatting and source-package checks

- [ ] The manuscript uses double-line spacing and includes line and page numbering.
- [ ] SI units are used, special characters are embedded correctly, and no manual page breaks are present.
- [ ] All manuscript, bibliography, figure, table, and supplement sources are editable and included in the submission package.
- [ ] The source compiles without errors under pdfLaTeX and a TeX Live 2021-compatible environment; the resulting PDF has no unresolved citations, references, or missing assets.
- [ ] Every figure and table title is at most 15 words, and every legend is at most 300 words.
- [ ] Every figure file is at most 10 MB, is cited in order, uses an accepted format, and remains legible at the journal's production dimensions.
- [ ] Tables are editable, are cited in order, and do not use color or shading to encode meaning.

### Springer Nature template provenance

- [ ] The paper is based on the project-chosen Springer Nature authoring template version 3.1, December 2024.
- [ ] A Vancouver-compatible bibliography style is selected and the journal-level rules override generic examples in the template.
- [ ] Publisher assets retain all upstream notices and remain separate from MaskImpute's OSI-licensed source.
- [ ] The complete template package is not described as having one package-wide LPPL or OSI license: the ZIP has no package-wide license file, its manual states `Copyright Springer Nature`, `sn-jnl.cls` contains an LPPL 1.3c-or-later notice plus a restrictive distribution sentence, and bibliography files contain differing notices.
- [ ] Publisher clarification has been obtained before independently redistributing the complete template package; otherwise only use and submit the files as provided for their intended authoring purpose.

## 9. Additional files and supplementary information

- [ ] Every additional file is at most 20 MB and is named and cited sequentially in the manuscript.
- [ ] The manuscript's separate additional-file listing gives each file's name, extension/format, title, and description.
- [ ] Results that would otherwise be described as `data not shown` are included in an additional file or recognized repository.
- [ ] Additional files contain no consent forms, language-editing certificates, tracked-change manuscripts, or individual participant details.
- [ ] Long supplementary material has an accessible organization, with a contents page and headings mirroring the main text where useful.
- [ ] Supplementary tables/data are machine-readable; workflows, negative results, and supplemental methods needed to understand the paper are included.
- [ ] Code details in supplemental Methods do not replace the required public source repository for the described tool.

## 10. Preprint and prior-publication decision

- [ ] The authors have recorded whether a preprint exists or will be posted. No automated process has made this decision.
- [ ] If applicable, the preprint DOI and license are disclosed to the journal and any overlapping content is transparent; otherwise the authors have marked this item not applicable.
- [ ] The authors understand that an eligible preprint is not prior publication, but the manuscript is not simultaneously under consideration at another journal and has no undisclosed overlapping publication.
- [ ] If a preprint is published, responsibility for updating its record with the final article DOI and URL has been assigned to an author.

## 11. Cover letter and final authorization

- [ ] The cover letter explains why the manuscript belongs in Genome Biology and specifically satisfies the Methodology criteria.
- [ ] The cover letter explains any issue relating to journal policies and declares potential competing interests.
- [ ] The cover letter confirms that all authors approved submission and that the manuscript is not published or under consideration elsewhere, while transparently identifying any permitted preprint.
- [ ] A collection or special-issue name is included if applicable; otherwise the authors have marked this item not applicable.
- [ ] Any suggested reviewers have verifiable identities and institutional contact information or persistent researcher identifiers, and no conflicted reviewer is suggested.
- [ ] The corresponding author has rechecked the current official instructions, reviewed every item above, and authorized submission only after every applicable box is checked.
