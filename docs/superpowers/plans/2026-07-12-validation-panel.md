# Four-Mechanism Validation Panel Implementation Plan

> **Execution rule:** implement with subagent-driven development, one reviewed task at a time. The legacy Splatter suite is never read by final-generation code.

**Goal:** Generate deterministic, provenance-complete development and sealed-final AnnData matrices from SymSim, SERGIO, SPARSim, and a real-data-fitted semisynthetic mechanism.

**Architecture:** A tracked source ledger pins upstream commits and licenses. Fetching places pristine upstream code under ignored `artifacts/external/`. Thin adapters invoke upstream software without modifying it and translate native outputs into the strict `maskimpute_benchmark.schema` contract. A dataset registry separates biological seeds from paired technical-view seeds and never exposes final seeds before the state controller materializes them.

**Execution environments:** Python 3.11 with NumPy/SciPy/Pandas/AnnData; PyTorch is not required for generation. SymSim and SPARSim run in a pinned R 4.4/Bioconductor environment. SERGIO runs from its pinned Python source. Every adapter records the actual upstream commit, environment digest, parameters, source checksum, biological seed, and measurement seed.

## Binding panel

| Mechanism | Pinned source | Biological truth | Paired technical views |
|---|---|---|---|
| SymSim | `YosefLab/SymSim@76a674b407ce44bf2690a9161cf28b905598d0a5` | discrete molecule counts | UMI capture efficiency, moderate/severe |
| SERGIO | `PayamDiba/SERGIO@a6190b74425112834c8fa9b4b6157d9cb3d1ab88` | clean continuous GRN/SDE expression | independently seeded dropout/UMI noise, moderate/severe |
| SPARSim | `sysbiobig/sparsim@4e7712fb236a92ce7c173da169c8a29cc2a9f0ef` | continuous `gene_matrix` | independently seeded count generation at two library-depth regimes |
| Semisynthetic | pinned public pancreatic UMI source | high-depth Gamma-Poisson reference proxy | independent thinning/count splitting, moderate/severe |

Development uses two biological draws per mechanism at 900 cells and 500 genes. Final uses five new biological draws per mechanism at 2,700 cells and 1,200 genes, with the same biological truth reused across paired technical views. Final generator seeds come only from the sealed round manifest.

---

### Task 1: Pinned source and data ledger

**Files:**

- Create: `study/sources.json`
- Create: `maskimpute_benchmark/sources.py`
- Create: `scripts/fetch_study_sources.py`
- Test: `tests/test_sources.py`

**Requirements:**

- Validate URL, exact 40-character commit, source type, license, citation DOI, and expected archive/tree checksum for all four mechanisms and orthogonal datasets.
- Fetch only into a caller-supplied ignored root; reject an existing checkout at the wrong commit or with local changes.
- Clone with no credential persistence, detach the exact commit, and write a canonical fetch receipt. Never auto-update a pin.
- Unit tests use local temporary Git repositories; a network integration test is explicitly marked.

**Verification:** `python -m pytest tests/test_sources.py -q`

**Commit:** `feat: pin publication study sources`

---

### Task 2: Simulator adapter contract and native-output sealing

**Files:**

- Create: `maskimpute_benchmark/simulators/__init__.py`
- Create: `maskimpute_benchmark/simulators/base.py`
- Create: `maskimpute_benchmark/simulators/native.py`
- Test: `tests/test_simulator_contract.py`

**Interfaces:**

- `SimulationRequest(mechanism, namespace, biological_id, biological_seed, measurement_seed, technical_view, cells, genes, output_path)`
- `SimulationArtifact(adata, native_manifest, dataset_sha256)`
- `validate_simulation_request(request, protocol, final_manifest=None)`
- `seal_native_outputs(files, metadata) -> native_manifest`

**Requirements:**

- Development/final namespaces are disjoint; final requests require a materialized manifest seed and exact protocol dimensions.
- Biological and measurement seeds are distinct fields. Technical views may share biological seed/truth but never claim independent-draw status.
- Native files are hashed before translation. Output AnnData passes strict schema validation and embeds the native-manifest hash.
- Deterministic IDs are derived from namespace/mechanism/biological draw/view, not from labels.

**Verification:** `python -m pytest tests/test_simulator_contract.py -q`

**Commit:** `feat: define simulator adapter contract`

---

### Task 3: SymSim adapter

**Files:**

- Create: `scripts/simulators/run_symsim.R`
- Create: `maskimpute_benchmark/simulators/symsim.py`
- Test: `tests/test_symsim_adapter.py`

**Requirements:**

- Use upstream `SimulateTrueCounts` once per biological draw with five discrete populations and a prespecified 5% rare population.
- Use the same `true_counts` for both calls to `True2ObservedCounts`; reset only the measurement seed and capture/depth parameters.
- Store observed UMI counts in `X`, exact molecule counts in `pre_capture_counts`, group/marker truth only in evaluator metadata, and `truth_kind=exact_pre_capture`.
- Smoke fixture is at most 30 cells × 20 genes; deterministic reruns have identical dataset hashes and different measurement seeds preserve identical truth.

**Verification:** `python -m pytest tests/test_symsim_adapter.py -q -m 'not network'`

**Commit:** `feat: add pinned SymSim validation adapter`

---

### Task 4: SERGIO adapter

**Files:**

- Create: `scripts/simulators/run_sergio.py`
- Create: `maskimpute_benchmark/simulators/sergio.py`
- Test: `tests/test_sergio_adapter.py`

**Requirements:**

- Import the pristine pinned checkout by explicit path and use the upstream 1,200-gene/9-cell-type GRN inputs for final draws.
- Seed clean SDE generation independently from technical modules. Preserve clean expression and pre-dropout technical expression; generate outlier/library/dropout/UMI stages with view-specific seeds.
- Store UMI counts in `X`, clean or pre-dropout expression in the declared continuous primary layer, and `truth_kind=exact_continuous`.
- Record any compatibility shim in the adapter receipt; never silently edit upstream files.

**Verification:** `python -m pytest tests/test_sergio_adapter.py -q -m 'not network'`

**Commit:** `feat: add pinned SERGIO validation adapter`

---

### Task 5: SPARSim adapter

**Files:**

- Create: `scripts/simulators/run_sparsim.R`
- Create: `maskimpute_benchmark/simulators/sparsim.py`
- Test: `tests/test_sparsim_adapter.py`

**Requirements:**

- Build a three-group panel from prespecified Chu C1/C3/C6 parameters, sampling library sizes to the requested cell count without changing biological group proportions between views.
- Separate `gene_expr_simulation_seed` from `count_data_simulation_seed`. Reuse `gene_matrix` across technical views and vary only the count-generation seed/depth regime.
- Store measured counts in `X`, `gene_matrix` in `latent_expression`, and `truth_kind=exact_continuous`.
- The adapter must detect upstream output orientation and reject silent cell/gene transposition.

**Verification:** `python -m pytest tests/test_sparsim_adapter.py -q -m 'not network'`

**Commit:** `feat: add pinned SPARSim validation adapter`

---

### Task 6: Semisynthetic high-depth adapter

**Files:**

- Create: `maskimpute_benchmark/simulators/semisynthetic.py`
- Create: `scripts/prepare_semisynthetic_source.py`
- Test: `tests/test_semisynthetic_adapter.py`

**Requirements:**

- Fit Gamma-Poisson cell-type/gene parameters from a development-only source partition and a disjoint final source partition.
- Generate one high-depth reference per biological seed, then derive paired observed views through independent binomial thinning. Provide optional binomial A/B count splits for replicate-concordance endpoints.
- Store thinned counts in `X`, high-depth proxy in `reference_counts`, held-out split in `heldout_counts` where present, and `truth_kind=proxy_high_depth`.
- Explicitly label proxy truth; exact pre-capture-zero score metrics remain undefined.

**Verification:** `python -m pytest tests/test_semisynthetic_adapter.py -q`

**Commit:** `feat: add semisynthetic count-thinning adapter`

---

### Task 7: Dataset registry and generation orchestrator

**Files:**

- Create: `maskimpute_benchmark/datasets.py`
- Create: `scripts/generate_study_datasets.py`
- Create: `study/development_panel.json`
- Test: `tests/test_dataset_registry.py`

**Requirements:**

- Expand biological seeds into exactly two development draws or five final draws per mechanism and exactly two paired technical views.
- Emit one registry row per dataset view with `biological_id`, `technical_view`, dataset hash, truth hash, source receipt, generator seed, measurement seed, and status.
- Prove paired views share truth hashes; prove all final biological seeds are absent from development records; reject duplicate IDs/checksums and dimension drift.
- `--namespace final` requires the state controller's one-use execution claim and writes only beneath that round.
- Failed adapters remain as status rows with logs; they are never silently dropped.

**Verification:** `python -m pytest tests/test_dataset_registry.py -q`

**Commit:** `feat: orchestrate four-mechanism study datasets`

---

### Task 8: External integration smoke and provenance report

**Files:**

- Create: `scripts/verify_validation_panel.py`
- Create: `docs/validation-panel-reproduction.md`
- Test: `tests/test_validation_panel_smoke.py`

**Requirements:**

- Fetch exact pins into a fresh ignored directory, run a tiny draw from every adapter twice, and compare deterministic hashes.
- Validate truth isolation by executing a dummy method only on `make_inference_view` and proving no evaluator field is reachable.
- Generate a machine-readable provenance/status table and record R/Python session information.
- The smoke command must fit within 20 minutes and 4 GiB RAM; full dimensions are exercised separately before freeze.

**Verification:** `python scripts/verify_validation_panel.py --smoke`

**Commit:** `test: verify four-mechanism validation panel`

## Completion gate

Run:

```bash
python -m pytest tests/test_sources.py tests/test_simulator_contract.py \
  tests/test_symsim_adapter.py tests/test_sergio_adapter.py \
  tests/test_sparsim_adapter.py tests/test_semisynthetic_adapter.py \
  tests/test_dataset_registry.py tests/test_validation_panel_smoke.py -q
```

Do not generate the final-size panel until every adapter is reviewed, the method and competitors are frozen, the repository is recursively clean, and `verify-final` has atomically claimed the round.
