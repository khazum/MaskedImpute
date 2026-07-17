# External-reference development implementation plan

1. Add contract tests for fixed CLI locators, no overwrite, successful and
   unavailable adapter receipts, and a conditionally runnable real-adapter
   integration path.
2. Add validator tests that reject hand-written checkpoints, empty metrics,
   input/reference/output/log tampering, and runtime/source/locator drift.
3. Implement the focused producer with fixed Tung preparation, matched-bulk
   reference construction, pre/post runtime and source validation, canonical
   artifact publication, and terminal failure receipts.
4. Implement the production loader that reopens all bytes and semantically
   recomputes references, snapshots, conversions, and endpoint units.
5. Replace publication freeze's self-signed parser with the production loader,
   retain strict terminal-reason syntax, and bind the external denominator to
   the canonical Tung source.
6. Bind the exact ignored operational roots needed by the publication worktree
   into the study freeze and revalidate their closed tree receipts throughout
   the final-round lifecycle.
7. Run targeted tests first, then publication-freeze and related suites, static
   checks, and a real adapter attempt only if the exact pinned runtime lock and
   locators are present.
