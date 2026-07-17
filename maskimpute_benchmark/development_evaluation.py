"""Truth-isolated safety evaluation and selection-artifact assembly.

The method runner finishes before any function in this module receives group,
marker, spike-in, protein, replicate, or bulk-reference information.  This
module is therefore evaluator-owned: it consumes immutable method outputs and
keeps all validation targets on the evaluator side of that boundary.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import shutil
import stat
import tarfile
import tempfile
from types import MappingProxyType
from typing import Mapping, Sequence
import zlib

import numpy as np


BOOTSTRAP_SEED = 20_260_712
NULL_DE_ALPHA = 0.05
NULL_DE_MIN_GENES = 100
CITE_METHOD_GENE_COUNT = 500
_ORTHOGONAL_OUTPUT_ENCODING = "zlib_raw_f64_v1"
_ORTHOGONAL_OUTPUT_COMPRESSION_LEVEL = 6
_ORTHOGONAL_MAX_MATRIX_UNCOMPRESSED_NBYTES = 256 * 1024**2
_ORTHOGONAL_RECORD_OVERHEAD_BYTES = 1024**2
_ORTHOGONAL_STORAGE_RESERVE_BYTES = 1024**3


class DevelopmentEvaluationError(RuntimeError):
    """Raised when selection evidence is incomplete, inconsistent, or altered."""


@dataclass(frozen=True, slots=True)
class RawArtifactBinding:
    """One immutable reconstruction file bound by its verified raw-byte hash."""

    run_id: str
    kind: str
    path: str
    file_sha256: str


@dataclass(frozen=True, slots=True)
class ReconstructionEvidence:
    """Validated completed runner checkpoint and all of its raw files."""

    checkpoint_path: str
    checkpoint_file_sha256: str
    checkpoint_sha256: str
    plan_sha256: str
    input_hashes: Mapping[str, str]
    records: tuple[Mapping[str, object], ...]
    raw_artifacts: tuple[RawArtifactBinding, ...]


@dataclass(frozen=True, slots=True)
class ReconstructionSelectionBundle:
    """Selection-schema rows plus auditable evaluator-only null-DE details."""

    records: tuple[dict[str, object], ...]
    null_de_audits: tuple[dict[str, object], ...]


@dataclass(frozen=True, slots=True)
class EndpointUnit:
    """One endpoint measurement nested in a biological unit."""

    unit_id: str
    biological_id: str
    technical_id: str
    value: float

    def __post_init__(self) -> None:
        for name in ("unit_id", "biological_id", "technical_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"endpoint {name} must be nonempty")
        if isinstance(self.value, bool) or not math.isfinite(float(self.value)):
            raise ValueError("endpoint value must be finite")


@dataclass(frozen=True, slots=True)
class EndpointInterval:
    """Paired candidate-minus-observed hierarchical endpoint interval."""

    configuration: str
    endpoint: str
    comparison: str
    estimate: float | None
    ci_lower: float | None
    ci_upper: float | None
    status: str
    reason: str | None
    n_biological_units: int
    n_technical_units: int
    n_boot: int
    bootstrap_sha256: str

    def selection_row(self) -> dict[str, object]:
        return {
            "configuration": self.configuration,
            "endpoint": self.endpoint,
            "comparison": self.comparison,
            "estimate": self.estimate,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class SourceReceiptBinding:
    source_id: str
    path: str
    file_sha256: str


@dataclass(frozen=True, slots=True)
class SourceArtifactBinding:
    source_id: str
    path: str
    file_sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class RealSourceEvidence:
    ledger_path: str
    ledger_file_sha256: str
    ledger_sha256: str
    receipts: tuple[SourceReceiptBinding, ...]
    artifacts: tuple[SourceArtifactBinding, ...]


@dataclass(frozen=True, slots=True)
class CiteSeqSource:
    cell_ids: tuple[str, ...]
    gene_ids: tuple[str, ...]
    endpoint_gene_ids: tuple[str, ...]
    rna_counts: np.ndarray
    protein_ids: tuple[str, ...]
    protein_counts: np.ndarray
    rna_file_sha256: str
    protein_file_sha256: str


@dataclass(frozen=True, slots=True)
class TungSource:
    cell_ids: tuple[str, ...]
    sample_ids: tuple[str, ...]
    individual_ids: tuple[str, ...]
    replicate_ids: tuple[str, ...]
    gene_ids: tuple[str, ...]
    counts: np.ndarray
    ercc_mask: np.ndarray
    bulk_profiles: Mapping[str, np.ndarray]
    lane_profiles: Mapping[str, np.ndarray]
    single_sample_file_sha256: str
    bulk_sample_file_sha256: str
    single_lane_file_sha256: str


@dataclass(frozen=True, slots=True)
class BaronSource:
    member_names: tuple[str, ...]
    gene_counts: tuple[int, ...]
    cell_counts: tuple[int, ...]
    archive_file_sha256: str

    @property
    def human_gene_count(self) -> int:
        return self.gene_counts[0]

    @property
    def mouse_gene_count(self) -> int:
        return self.gene_counts[4]


@dataclass(frozen=True, slots=True)
class OrthogonalInput:
    source_id: str
    method_input: object

    def __post_init__(self) -> None:
        from .methods import MethodInput

        if not re.fullmatch(r"[a-z][a-z0-9-]*", self.source_id):
            raise ValueError("orthogonal source_id is invalid")
        if not isinstance(self.method_input, MethodInput):
            raise TypeError("orthogonal method_input must be a MethodInput")


@dataclass(frozen=True, slots=True)
class OrthogonalConfiguration:
    configuration_id: str
    configuration_sha256: str
    payload: Mapping[str, object]

    def __post_init__(self) -> None:
        from .protocol import canonical_sha256

        if not re.fullmatch(r"[a-z][a-z0-9-]*", self.configuration_id):
            raise ValueError("orthogonal configuration_id is invalid")
        if not isinstance(self.payload, Mapping):
            raise TypeError("orthogonal configuration payload must be a mapping")
        payload = json.loads(
            json.dumps(dict(self.payload), allow_nan=False, sort_keys=True)
        )
        if canonical_sha256(payload) != self.configuration_sha256:
            raise ValueError("orthogonal configuration checksum mismatch")
        object.__setattr__(self, "payload", MappingProxyType(payload))


@dataclass(frozen=True, slots=True)
class OrthogonalExecutionRequest:
    source_id: str
    configuration: OrthogonalConfiguration
    model_seed: int
    method_input: object


@dataclass(frozen=True, slots=True)
class OrthogonalOutputEvidence:
    output_directory: Path
    manifest_path: Path
    manifest_file_sha256: str
    manifest_sha256: str
    records: tuple[Mapping[str, object], ...]


@dataclass(frozen=True, slots=True)
class OrthogonalSelectionBundle:
    intervals: tuple[dict[str, object], ...]
    audits: tuple[dict[str, object], ...]


@dataclass(frozen=True, slots=True)
class PreparedRealOrthogonalPanel:
    source_evidence: RealSourceEvidence
    baron: BaronSource
    cite: CiteSeqSource
    tung: TungSource
    method_inputs: tuple[OrthogonalInput, ...]


@dataclass(frozen=True, slots=True)
class PublicMaskImputeOrthogonalExecutor:
    """Truth-free external-data executor using the public production API."""

    count_model_config: object
    calibration_artifact: object
    device: str = "cuda"

    def __call__(self, request: OrthogonalExecutionRequest) -> np.ndarray:
        from maskimpute import MaskImputeConfig, fit_p_pre_zero_count_model
        from maskimpute.impute import impute_counts

        if not isinstance(request, OrthogonalExecutionRequest):
            raise TypeError("request must be OrthogonalExecutionRequest")
        payload = dict(request.configuration.payload)
        hyperparameters = payload.get("hyperparameters")
        if not isinstance(hyperparameters, dict):
            raise DevelopmentEvaluationError(
                "orthogonal MaskImpute configuration lacks hyperparameters"
            )
        config = MaskImputeConfig(**hyperparameters, seed=request.model_seed)
        score = fit_p_pre_zero_count_model(
            request.method_input.counts,
            request.method_input.obs_ids,
            self.count_model_config,
        )
        score_policy = payload.get("score_policy")
        if score_policy == "direct_cross_fitted_count_score":
            calibration = None
        elif score_policy in {
            "retained_calibrator",
            "retained_calibrated_count_score",
            "retained_development_calibrator",
        }:
            calibration = self.calibration_artifact
        else:
            raise DevelopmentEvaluationError(
                "orthogonal MaskImpute score policy is not authorized"
            )
        result = impute_counts(
            request.method_input.counts,
            score,
            config,
            self.device,
            cell_ids=request.method_input.obs_ids,
            calibration_artifact=calibration,
        )
        return np.asarray(result.selective_counts, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class NullDEResult:
    """One prespecified null differential-expression safety endpoint."""

    status: str
    fpr: float | None
    nominal_alpha: float
    n_tested_genes: int
    split_sha256: str
    gene_mask_sha256: str
    reason: str | None = None


def _read_stable_bytes(
    path: Path, name: str, *, max_bytes: int | None = None
) -> tuple[bytes, str]:
    """Read one regular file once from an O_NOFOLLOW descriptor and recheck it."""

    if max_bytes is not None and (
        isinstance(max_bytes, bool) or type(max_bytes) is not int or max_bytes < 0
    ):
        raise ValueError("max_bytes must be a nonnegative integer or None")

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise DevelopmentEvaluationError(f"cannot open {name}: {error}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise DevelopmentEvaluationError(f"{name} is not a regular file")
        if max_bytes is not None and before.st_size > max_bytes:
            raise DevelopmentEvaluationError(f"{name} exceeds its byte bound")
        chunks: list[bytes] = []
        observed_nbytes = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            observed_nbytes += len(chunk)
            if max_bytes is not None and observed_nbytes > max_bytes:
                raise DevelopmentEvaluationError(f"{name} exceeds its byte bound")
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    raw = b"".join(chunks)
    if identity_before != identity_after or len(raw) != before.st_size:
        raise DevelopmentEvaluationError(f"{name} changed while it was read")
    return raw, hashlib.sha256(raw).hexdigest()


def _file_sha256(path: Path) -> str:
    return _read_stable_bytes(path, path.name)[1]


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise DevelopmentEvaluationError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise DevelopmentEvaluationError(f"nonfinite JSON constant: {value}")


def _strict_json(path: Path, name: str) -> tuple[dict[str, object], bytes]:
    try:
        raw, _digest = _read_stable_bytes(path, name)
        parsed = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_json_object,
        )
    except DevelopmentEvaluationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DevelopmentEvaluationError(f"cannot load {name}: {error}") from error
    if not isinstance(parsed, dict):
        raise DevelopmentEvaluationError(f"{name} must be a JSON object")
    return parsed, raw


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        ensure_ascii=False,
    ).encode("utf-8")


def _require_regular_file(path: Path, name: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise DevelopmentEvaluationError(f"{name} is missing") from error
    import stat

    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise DevelopmentEvaluationError(f"{name} must be a regular non-symlink file")


def validate_real_source_artifacts(repository: Path) -> RealSourceEvidence:
    """Revalidate the fixed Baron, CITE-seq, and Tung source bytes and receipts."""

    from .sources import SourceLedgerError, load_source_ledger

    if not isinstance(repository, Path):
        raise TypeError("repository must be a pathlib.Path")
    root = repository.absolute()
    ledger_path = root / "study/sources.json"
    ledger_raw, ledger_file_sha256 = _read_stable_bytes(ledger_path, "source ledger")
    descriptor, temporary_name = tempfile.mkstemp(suffix=".json")
    temporary_ledger = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(ledger_raw)
            stream.flush()
            os.fsync(stream.fileno())
        ledger = load_source_ledger(temporary_ledger)
    except SourceLedgerError as error:
        raise DevelopmentEvaluationError(
            f"source ledger is invalid: {error}"
        ) from error
    finally:
        temporary_ledger.unlink(missing_ok=True)
    required_roles = {
        "baron-pancreas-umi": "semisynthetic_source",
        "cite-seq-cbmc-rna-protein": "orthogonal_validation",
        "tung-ipsc-ercc-bulk-replicates": "orthogonal_validation",
    }
    sources = {source.id: source for source in ledger.sources}
    if not set(required_roles) <= set(sources):
        raise DevelopmentEvaluationError("required real sources are absent from ledger")
    receipt_bindings: list[SourceReceiptBinding] = []
    artifact_bindings: list[SourceArtifactBinding] = []
    receipt_keys = {
        "schema_version",
        "source_id",
        "role",
        "source_type",
        "source_url",
        "revision",
        "resolved_revision",
        "license",
        "citation_doi",
        "verified_checksum",
        "ledger_sha256",
        "artifacts",
    }
    for source_id in required_roles:
        source = sources[source_id]
        if (
            source.role != required_roles[source_id]
            or source.source_type != "data"
            or source.eligibility != "eligible"
            or not source.artifacts
        ):
            raise DevelopmentEvaluationError(
                f"real source {source_id} is not an eligible pinned data source"
            )
        receipt_relative = f"artifacts/external/receipts/{source_id}.json"
        receipt_path = root.joinpath(*PurePosixPath(receipt_relative).parts)
        _require_regular_file(receipt_path, f"{source_id} receipt")
        receipt, raw_receipt = _strict_json(receipt_path, f"{source_id} receipt")
        canonical_receipt = (
            json.dumps(
                receipt,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
                ensure_ascii=False,
            ).encode("utf-8")
            + b"\n"
        )
        if raw_receipt != canonical_receipt or set(receipt) != receipt_keys:
            raise DevelopmentEvaluationError(
                f"{source_id} receipt is not canonical or has wrong fields"
            )
        expected_header = {
            "schema_version": 1,
            "source_id": source.id,
            "role": source.role,
            "source_type": source.source_type,
            "source_url": source.url,
            "revision": source.revision,
            "resolved_revision": source.revision,
            "license": source.license,
            "citation_doi": source.citation_doi,
            "verified_checksum": None,
            "ledger_sha256": ledger.sha256,
        }
        if any(receipt.get(key) != value for key, value in expected_header.items()):
            raise DevelopmentEvaluationError(
                f"{source_id} receipt mismatches tracked source ledger"
            )
        receipt_artifacts = receipt.get("artifacts")
        if not isinstance(receipt_artifacts, list) or len(receipt_artifacts) != len(
            source.artifacts
        ):
            raise DevelopmentEvaluationError(
                f"{source_id} receipt artifact denominator is incomplete"
            )
        for pinned, received in zip(source.artifacts, receipt_artifacts, strict=True):
            if not isinstance(received, dict) or set(received) != {
                "name",
                "sha256",
                "size_bytes",
            }:
                raise DevelopmentEvaluationError(
                    f"{source_id} receipt artifact schema is invalid"
                )
            relative = f"artifacts/external/data/{source_id}/{pinned.name}"
            path = root.joinpath(*PurePosixPath(relative).parts)
            artifact_raw, actual_sha = _read_stable_bytes(
                path, f"{source_id} artifact {pinned.name}"
            )
            actual_size = len(artifact_raw)
            if (
                received.get("name") != pinned.name
                or received.get("sha256") != pinned.expected_checksum.value
                or received.get("size_bytes") != actual_size
                or actual_sha != pinned.expected_checksum.value
            ):
                raise DevelopmentEvaluationError(
                    f"{source_id} artifact checksum or size mismatch"
                )
            artifact_bindings.append(
                SourceArtifactBinding(source_id, relative, actual_sha, actual_size)
            )
        receipt_bindings.append(
            SourceReceiptBinding(
                source_id,
                receipt_relative,
                hashlib.sha256(raw_receipt).hexdigest(),
            )
        )
    return RealSourceEvidence(
        ledger_path="study/sources.json",
        ledger_file_sha256=ledger_file_sha256,
        ledger_sha256=ledger.sha256,
        receipts=tuple(receipt_bindings),
        artifacts=tuple(artifact_bindings),
    )


def _readonly_counts(value: object, name: str) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.ndim != 2
        or array.dtype.kind not in "iuf"
        or not np.isfinite(array).all()
        or bool((array < 0).any())
        or not np.equal(array, np.floor(array)).all()
    ):
        raise DevelopmentEvaluationError(
            f"{name} must be a finite nonnegative integer matrix"
        )
    maximum = float(array.max(initial=0))
    dtype = np.int32 if maximum <= np.iinfo(np.int32).max else np.int64
    result = np.asarray(array, dtype=dtype, order="C").copy()
    result.setflags(write=False)
    return result


def _gzip_text(raw: bytes) -> io.TextIOWrapper:
    compressed = gzip.GzipFile(fileobj=io.BytesIO(raw), mode="rb")
    return io.TextIOWrapper(compressed, encoding="utf-8", newline="")


def _cite_selected_rows(
    path: Path,
    raw: bytes,
    *,
    identifier_transform: object,
    retained_ids: set[str],
) -> tuple[list[str], dict[str, tuple[str, list[str]]]]:
    """Stream a wide CITE-seq CSV and retain only endpoint rows."""

    if not callable(identifier_transform):
        raise TypeError("identifier_transform must be callable")
    selected: dict[str, tuple[str, list[str]]] = {}
    try:
        with _gzip_text(raw) as stream:
            reader = csv.reader(stream)
            header = next(reader)
            if not header:
                raise DevelopmentEvaluationError(f"{path.name} header is empty")
            for row in reader:
                if len(row) != len(header):
                    raise DevelopmentEvaluationError(
                        f"{path.name} has inconsistent rows"
                    )
                identifier = str(identifier_transform(row[0]))
                if identifier not in retained_ids:
                    continue
                if identifier in selected:
                    raise DevelopmentEvaluationError(
                        f"{path.name} endpoint identifier is duplicated"
                    )
                selected[identifier] = (row[0], row[1:])
    except DevelopmentEvaluationError:
        raise
    except (OSError, UnicodeError, csv.Error, StopIteration) as error:
        raise DevelopmentEvaluationError(
            f"cannot parse {path.name}: {error}"
        ) from error
    return header, selected


def _cite_method_rna_panel(
    path: Path,
    raw: bytes,
) -> tuple[list[str], tuple[str, ...], dict[str, list[str]]]:
    """Select a fixed count-only 500-gene panel while forcing endpoint genes."""

    totals: dict[str, int] = {}
    try:
        with _gzip_text(raw) as stream:
            reader = csv.reader(stream)
            header = next(reader)
            if not header:
                raise DevelopmentEvaluationError(f"{path.name} header is empty")
            for row in reader:
                if len(row) != len(header):
                    raise DevelopmentEvaluationError(
                        f"{path.name} has inconsistent rows"
                    )
                raw_id = row[0]
                if not raw_id.startswith("HUMAN_"):
                    continue
                symbol = raw_id.removeprefix("HUMAN_")
                if not symbol or symbol in totals:
                    raise DevelopmentEvaluationError(
                        "CITE-seq human RNA gene symbols are duplicated"
                    )
                try:
                    total = 0
                    for raw_count in row[1:]:
                        count = int(raw_count)
                        if count < 0:
                            raise DevelopmentEvaluationError(
                                "CITE-seq RNA counts must be nonnegative"
                            )
                        total += count
                except (TypeError, ValueError, OverflowError) as error:
                    raise DevelopmentEvaluationError(
                        "CITE-seq RNA counts are not integers"
                    ) from error
                totals[symbol] = total
    except DevelopmentEvaluationError:
        raise
    except (OSError, UnicodeError, csv.Error, StopIteration) as error:
        raise DevelopmentEvaluationError(
            f"cannot parse {path.name}: {error}"
        ) from error
    endpoint_lookup: dict[str, str] = {}
    endpoint_symbols = set(_PROTEIN_TO_RNA.values())
    for symbol in totals:
        canonical = symbol.upper()
        if canonical not in endpoint_symbols:
            continue
        if canonical in endpoint_lookup:
            raise DevelopmentEvaluationError(
                "CITE-seq endpoint RNA gene is duplicated after case folding"
            )
        endpoint_lookup[canonical] = symbol
    endpoint_order = tuple(
        endpoint_lookup[gene]
        for gene in dict.fromkeys(_PROTEIN_TO_RNA.values())
        if gene in endpoint_lookup
    )
    remaining = sorted(
        (gene for gene in totals if gene not in set(endpoint_order)),
        key=lambda gene: (-totals[gene], gene),
    )
    selected_order = tuple(
        (
            *endpoint_order,
            *remaining[: max(0, CITE_METHOD_GENE_COUNT - len(endpoint_order))],
        )
    )
    selected_set = set(selected_order)
    selected_rows: dict[str, list[str]] = {}
    try:
        with _gzip_text(raw) as stream:
            reader = csv.reader(stream)
            repeated_header = next(reader)
            if repeated_header != header:
                raise DevelopmentEvaluationError(
                    "CITE-seq RNA header changed between streaming passes"
                )
            for row in reader:
                symbol = row[0].removeprefix("HUMAN_")
                if symbol in selected_set:
                    selected_rows[symbol] = row[1:]
    except DevelopmentEvaluationError:
        raise
    except (OSError, UnicodeError, csv.Error, StopIteration) as error:
        raise DevelopmentEvaluationError(
            f"cannot re-read {path.name}: {error}"
        ) from error
    if set(selected_rows) != selected_set:
        raise DevelopmentEvaluationError("CITE-seq RNA panel changed between passes")
    return header, selected_order, selected_rows


def prepare_cite_seq_source(rna_path: Path, protein_path: Path) -> CiteSeqSource:
    """Prepare only the fixed antibody-matched RNA features and ADT endpoints."""

    for path, name in ((rna_path, "rna_path"), (protein_path, "protein_path")):
        if not isinstance(path, Path):
            raise TypeError(f"{name} must be a pathlib.Path")
    rna_raw, rna_sha256 = _read_stable_bytes(rna_path, "CITE-seq RNA source")
    protein_raw, protein_sha256 = _read_stable_bytes(
        protein_path, "CITE-seq ADT source"
    )
    rna_header, method_genes, rna_by_symbol = _cite_method_rna_panel(rna_path, rna_raw)
    protein_header, selected_protein = _cite_selected_rows(
        protein_path,
        protein_raw,
        identifier_transform=lambda value: value.upper(),
        retained_ids=set(_PROTEIN_TO_RNA),
    )
    rna_cells = tuple(rna_header[1:])
    protein_cells = tuple(protein_header[1:])
    if (
        not rna_cells
        or rna_cells != protein_cells
        or len(set(rna_cells)) != len(rna_cells)
    ):
        raise DevelopmentEvaluationError("CITE-seq RNA and ADT cell IDs do not align")
    protein_by_id = {key: value[1] for key, value in selected_protein.items()}
    display_by_id = {key: value[0] for key, value in selected_protein.items()}
    method_by_canonical = {gene.upper(): gene for gene in method_genes}
    retained = [
        (protein, method_by_canonical[gene])
        for protein, gene in _PROTEIN_TO_RNA.items()
        if protein in protein_by_id and gene in method_by_canonical
    ]
    if len(retained) < 2:
        raise DevelopmentEvaluationError(
            "CITE-seq source has fewer than two variable-capable matched analytes"
        )
    try:
        rna = _readonly_counts(
            np.asarray(
                [
                    [int(value) for value in rna_by_symbol[gene]]
                    for gene in method_genes
                ],
                dtype=np.int64,
            ).T,
            "CITE-seq RNA endpoint counts",
        )
        proteins = _readonly_counts(
            np.asarray(
                [
                    [int(value) for value in protein_by_id[protein]]
                    for protein, _ in retained
                ],
                dtype=np.int64,
            ).T,
            "CITE-seq ADT endpoint counts",
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise DevelopmentEvaluationError("CITE-seq counts are not integers") from error
    return CiteSeqSource(
        cell_ids=rna_cells,
        gene_ids=method_genes,
        endpoint_gene_ids=tuple(gene for _, gene in retained),
        rna_counts=rna,
        protein_ids=tuple(display_by_id[protein] for protein, _ in retained),
        protein_counts=proteins,
        rna_file_sha256=rna_sha256,
        protein_file_sha256=protein_sha256,
    )


def _read_tung_table(path: Path, raw: bytes, metadata_columns: Sequence[str]):
    import pandas as pd

    try:
        frame = pd.read_csv(
            io.BytesIO(raw), sep="\t", compression="gzip", low_memory=False
        )
    except Exception as error:
        raise DevelopmentEvaluationError(f"cannot parse {path.name}") from error
    metadata = tuple(metadata_columns)
    if tuple(frame.columns[: len(metadata)]) != metadata:
        raise DevelopmentEvaluationError(
            f"{path.name} metadata columns do not match the source contract"
        )
    genes = tuple(str(value) for value in frame.columns[len(metadata) :])
    if not genes or len(set(genes)) != len(genes):
        raise DevelopmentEvaluationError(f"{path.name} gene IDs are invalid")
    meta = frame.loc[:, list(metadata)].astype(str)
    counts = _readonly_counts(
        frame.iloc[:, len(metadata) :].to_numpy(copy=True), path.name
    )
    return genes, meta, counts


def prepare_tung_source(
    single_sample_path: Path,
    bulk_sample_path: Path,
    single_lane_path: Path,
) -> TungSource:
    """Prepare Tung cell counts and evaluator-private bulk/lane reference profiles."""

    if any(
        not isinstance(path, Path)
        for path in (single_sample_path, bulk_sample_path, single_lane_path)
    ):
        raise TypeError("Tung source paths must be pathlib.Path values")
    single_raw, single_sha256 = _read_stable_bytes(
        single_sample_path, "Tung single-cell sample source"
    )
    bulk_raw, bulk_sha256 = _read_stable_bytes(
        bulk_sample_path, "Tung bulk sample source"
    )
    lane_raw, lane_sha256 = _read_stable_bytes(single_lane_path, "Tung lane source")
    genes, cell_meta, counts = _read_tung_table(
        single_sample_path, single_raw, ("individual", "replicate", "well")
    )
    bulk_genes, bulk_meta, bulk_counts = _read_tung_table(
        bulk_sample_path, bulk_raw, ("individual", "replicate", "well")
    )
    lane_genes, lane_meta, lane_counts = _read_tung_table(
        single_lane_path,
        lane_raw,
        ("individual", "replicate", "well", "index", "lane", "flow_cell"),
    )
    if bulk_genes != genes or lane_genes != genes:
        raise DevelopmentEvaluationError("Tung source gene orders do not align")
    cell_ids = tuple(
        f"{row.individual}:{row.replicate}:{row.well}"
        for row in cell_meta.itertuples(index=False)
    )
    if len(set(cell_ids)) != len(cell_ids):
        raise DevelopmentEvaluationError("Tung cell identifiers are duplicated")
    sample_ids = tuple(
        f"{row.individual}:{row.replicate}" for row in cell_meta.itertuples(index=False)
    )
    bulk_profiles: dict[str, np.ndarray] = {}
    for index, row in enumerate(bulk_meta.itertuples(index=False)):
        key = f"{row.individual}:{row.replicate}"
        if key in bulk_profiles:
            raise DevelopmentEvaluationError("Tung bulk sample key is duplicated")
        profile = np.asarray(bulk_counts[index], dtype=np.float64).copy()
        profile.setflags(write=False)
        bulk_profiles[key] = profile
    if set(bulk_profiles) != set(sample_ids):
        raise DevelopmentEvaluationError(
            "Tung single-cell and bulk samples do not align"
        )
    lane_sums: dict[str, np.ndarray] = {}
    for index, row in enumerate(lane_meta.itertuples(index=False)):
        key = f"{row.individual}:{row.replicate}:{row.flow_cell}:{row.lane}"
        if key not in lane_sums:
            lane_sums[key] = np.zeros(len(genes), dtype=np.float64)
        lane_sums[key] += lane_counts[index]
    for value in lane_sums.values():
        value.setflags(write=False)
    if any(":".join(key.split(":")[:2]) not in set(sample_ids) for key in lane_sums):
        raise DevelopmentEvaluationError("Tung lane sample is absent from cell data")
    ercc = np.asarray([gene.upper().startswith("ERCC-") for gene in genes])
    ercc.setflags(write=False)
    if not ercc.any() or bool(ercc.all()):
        raise DevelopmentEvaluationError(
            "Tung source must contain both endogenous and ERCC features"
        )
    return TungSource(
        cell_ids=cell_ids,
        sample_ids=sample_ids,
        individual_ids=tuple(cell_meta["individual"].astype(str)),
        replicate_ids=tuple(cell_meta["replicate"].astype(str)),
        gene_ids=genes,
        counts=counts,
        ercc_mask=ercc,
        bulk_profiles=MappingProxyType(bulk_profiles),
        lane_profiles=MappingProxyType(lane_sums),
        single_sample_file_sha256=single_sha256,
        bulk_sample_file_sha256=bulk_sha256,
        single_lane_file_sha256=lane_sha256,
    )


_BARON_MEMBERS = (
    "GSM2230757_human1_umifm_counts.csv.gz",
    "GSM2230758_human2_umifm_counts.csv.gz",
    "GSM2230759_human3_umifm_counts.csv.gz",
    "GSM2230760_human4_umifm_counts.csv.gz",
    "GSM2230761_mouse1_umifm_counts.csv.gz",
    "GSM2230762_mouse2_umifm_counts.csv.gz",
)


def prepare_baron_source(archive_path: Path) -> BaronSource:
    """Validate the six-donor Baron archive without materializing its count matrix."""

    if not isinstance(archive_path, Path):
        raise TypeError("archive_path must be a pathlib.Path")
    archive_raw, archive_sha256 = _read_stable_bytes(archive_path, "Baron archive")
    cell_counts: list[int] = []
    gene_counts: list[int] = []
    try:
        with tarfile.open(fileobj=io.BytesIO(archive_raw), mode="r:*") as archive:
            members = archive.getmembers()
            names = tuple(member.name for member in members)
            if names != _BARON_MEMBERS or any(
                not member.isfile()
                or PurePosixPath(member.name).is_absolute()
                or ".." in PurePosixPath(member.name).parts
                for member in members
            ):
                raise DevelopmentEvaluationError(
                    "Baron archive member denominator or paths are invalid"
                )
            for member in members:
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise DevelopmentEvaluationError(
                        "Baron archive member is unreadable"
                    )
                with gzip.open(extracted, "rt", encoding="utf-8", newline="") as stream:
                    reader = csv.reader(stream)
                    header = next(reader)
                    if len(header) < 4 or header[1:3] != [
                        "barcode",
                        "assigned_cluster",
                    ]:
                        raise DevelopmentEvaluationError(
                            "Baron archive count-table header is invalid"
                        )
                    count = 0
                    for row in reader:
                        if len(row) != len(header):
                            raise DevelopmentEvaluationError(
                                "Baron archive count-table row width is invalid"
                            )
                        count += 1
                    if count == 0:
                        raise DevelopmentEvaluationError(
                            "Baron archive count table is empty"
                        )
                    cell_counts.append(count)
                    gene_counts.append(len(header) - 3)
    except DevelopmentEvaluationError:
        raise
    except (OSError, tarfile.TarError, UnicodeError, csv.Error, StopIteration) as error:
        raise DevelopmentEvaluationError(
            f"cannot parse Baron archive: {error}"
        ) from error
    if len(set(gene_counts[:4])) != 1 or len(set(gene_counts[4:])) != 1:
        raise DevelopmentEvaluationError(
            "Baron donor gene counts do not align within species"
        )
    return BaronSource(
        member_names=_BARON_MEMBERS,
        gene_counts=tuple(gene_counts),
        cell_counts=tuple(cell_counts),
        archive_file_sha256=archive_sha256,
    )


def _publish_bound_file(path: Path, data: bytes) -> str:
    digest = hashlib.sha256(data).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            _require_regular_file(path, "existing orthogonal output")
            existing, _digest = _read_stable_bytes(path, "existing orthogonal output")
            if existing != data:
                raise DevelopmentEvaluationError(
                    f"existing orthogonal output conflicts with {path.name}"
                )
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return digest


def _replace_checkpoint_file(path: Path, data: bytes) -> str:
    """Atomically replace a mutable checkpoint while rejecting symlink targets."""

    digest = hashlib.sha256(data).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and (path.is_symlink() or not path.is_file()):
        raise DevelopmentEvaluationError("orthogonal checkpoint path is unsafe")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return digest


def _zlib_compress_bound(uncompressed_nbytes: int) -> int:
    """Return zlib's documented single-call upper bound."""

    return (
        uncompressed_nbytes
        + (uncompressed_nbytes >> 12)
        + (uncompressed_nbytes >> 14)
        + (uncompressed_nbytes >> 25)
        + 13
    )


def _preflight_orthogonal_output_storage(
    output_directory: Path,
    *,
    remaining_shapes: Sequence[tuple[int, int]],
) -> dict[str, int | str]:
    """Fail closed unless every remaining output fits its compressed upper bound."""

    shapes = tuple(remaining_shapes)
    uncompressed_sizes: list[int] = []
    for shape in shapes:
        if (
            not isinstance(shape, tuple)
            or len(shape) != 2
            or any(type(value) is not int or value <= 0 for value in shape)
        ):
            raise DevelopmentEvaluationError(
                "orthogonal storage preflight shape is invalid"
            )
        uncompressed_nbytes = shape[0] * shape[1] * 8
        if uncompressed_nbytes > _ORTHOGONAL_MAX_MATRIX_UNCOMPRESSED_NBYTES:
            raise DevelopmentEvaluationError(
                "orthogonal output exceeds the fixed matrix byte bound"
            )
        uncompressed_sizes.append(uncompressed_nbytes)
    required = (
        sum(_zlib_compress_bound(value) for value in uncompressed_sizes)
        + len(shapes) * _ORTHOGONAL_RECORD_OVERHEAD_BYTES
        + _ORTHOGONAL_STORAGE_RESERVE_BYTES
    )
    try:
        free = shutil.disk_usage(output_directory).free
    except OSError as error:
        raise DevelopmentEvaluationError(
            "orthogonal free storage cannot be measured"
        ) from error
    if free < required:
        raise DevelopmentEvaluationError(
            "orthogonal free storage is below the fail-closed compressed-output bound"
        )
    return {
        "schema": "maskimpute-orthogonal-storage-preflight-v1",
        "remaining_output_count": len(shapes),
        "required_free_bytes": required,
        "observed_free_bytes": free,
    }


def _decode_orthogonal_output(
    output_path: Path,
    record: Mapping[str, object],
) -> bytes:
    """Revalidate one compressed output and decompress it within receipt bounds."""

    shape = record.get("output_shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(type(value) is not int or value <= 0 for value in shape)
    ):
        raise DevelopmentEvaluationError("orthogonal output shape is invalid")
    expected_nbytes = shape[0] * shape[1] * 8
    if expected_nbytes > _ORTHOGONAL_MAX_MATRIX_UNCOMPRESSED_NBYTES:
        raise DevelopmentEvaluationError(
            "orthogonal output exceeds the fixed matrix byte bound"
        )
    if (
        record.get("output_encoding") != _ORTHOGONAL_OUTPUT_ENCODING
        or record.get("output_dtype") != "<f8"
        or record.get("output_scale") != "log2_cp10k_plus_1"
        or record.get("output_uncompressed_nbytes") != expected_nbytes
    ):
        raise DevelopmentEvaluationError(
            "orthogonal compressed output encoding or size differs"
        )
    compressed_nbytes = record.get("output_compressed_nbytes")
    compressed_sha256 = record.get("output_file_sha256")
    raw_sha256 = record.get("output_uncompressed_sha256")
    maximum_compressed = _zlib_compress_bound(expected_nbytes)
    if (
        type(compressed_nbytes) is not int
        or not 0 < compressed_nbytes <= maximum_compressed
        or not isinstance(compressed_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", compressed_sha256) is None
        or not isinstance(raw_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", raw_sha256) is None
    ):
        raise DevelopmentEvaluationError(
            "orthogonal compressed output receipt is invalid"
        )
    compressed, actual_sha256 = _read_stable_bytes(
        output_path,
        "orthogonal compressed method output",
        max_bytes=maximum_compressed,
    )
    if (
        len(compressed) != compressed_nbytes
        or actual_sha256 != compressed_sha256
    ):
        raise DevelopmentEvaluationError(
            "orthogonal compressed output checksum or size mismatch"
        )
    try:
        decompressor = zlib.decompressobj()
        raw = decompressor.decompress(compressed, expected_nbytes + 1)
        raw += decompressor.flush(max(1, expected_nbytes + 1 - len(raw)))
    except zlib.error as error:
        raise DevelopmentEvaluationError(
            "orthogonal compressed output cannot be decompressed"
        ) from error
    if (
        len(raw) != expected_nbytes
        or not decompressor.eof
        or decompressor.unconsumed_tail
        or decompressor.unused_data
        or hashlib.sha256(raw).hexdigest() != raw_sha256
    ):
        raise DevelopmentEvaluationError(
            "orthogonal compressed output differs from its receipt"
        )
    values = np.frombuffer(raw, dtype="<f8")
    if values.size != shape[0] * shape[1] or not np.isfinite(values).all():
        raise DevelopmentEvaluationError(
            "orthogonal compressed output contains invalid values"
        )
    return raw


def _write_orthogonal_checkpoint(
    manifest_path: Path,
    authority: Mapping[str, object],
    records: Sequence[Mapping[str, object]],
    planned_record_count: int,
) -> None:
    from .protocol import canonical_sha256

    status = "completed" if len(records) == planned_record_count else "running"
    core = {
        "schema_version": 2,
        "artifact_type": "maskimpute_orthogonal_method_outputs",
        "authority": dict(authority),
        "status": status,
        "planned_record_count": planned_record_count,
        "records": [dict(value) for value in records],
    }
    payload = {**core, "manifest_sha256": canonical_sha256(core)}
    _replace_checkpoint_file(manifest_path, _canonical_json_bytes(payload) + b"\n")


def _orthogonal_authority_core(
    inputs: Sequence[OrthogonalInput],
    configurations: Sequence[OrthogonalConfiguration],
    model_seeds: Sequence[int],
    artifact_bindings: Mapping[str, object],
) -> dict[str, object]:
    from .runner import method_input_sha256

    if set(artifact_bindings) != {
        "count_model_config_sha256",
        "retained_calibration_artifact_sha256",
        "score_fit_policy",
    }:
        raise DevelopmentEvaluationError(
            "orthogonal artifact bindings have wrong fields"
        )
    for name in (
        "count_model_config_sha256",
        "retained_calibration_artifact_sha256",
    ):
        if not isinstance(artifact_bindings[name], str) or not re.fullmatch(
            r"[0-9a-f]{64}", str(artifact_bindings[name])
        ):
            raise DevelopmentEvaluationError(
                f"orthogonal artifact binding {name} is invalid"
            )
    if (
        artifact_bindings["score_fit_policy"]
        != "refit_cross_fitted_count_score_from_truth_free_input"
    ):
        raise DevelopmentEvaluationError(
            "orthogonal score fit policy is not authorized"
        )
    return {
        "inputs": [
            {
                "source_id": value.source_id,
                "source_dataset_sha256": value.method_input.source_dataset_sha256,
                "method_input_sha256": method_input_sha256(value.method_input),
                "shape": list(value.method_input.shape),
                "cell_ids_sha256": hashlib.sha256(
                    _canonical_json_bytes(list(value.method_input.obs_ids))
                ).hexdigest(),
                "gene_ids_sha256": hashlib.sha256(
                    _canonical_json_bytes(list(value.method_input.var_ids))
                ).hexdigest(),
            }
            for value in inputs
        ],
        "configurations": [
            {
                "configuration_id": value.configuration_id,
                "configuration_sha256": value.configuration_sha256,
                "payload": dict(value.payload),
            }
            for value in configurations
        ],
        "model_seeds": list(model_seeds),
        "artifact_bindings": dict(artifact_bindings),
    }


def _load_orthogonal_evidence(
    output_directory: Path,
    expected_authority: Mapping[str, object],
    *,
    allow_running: bool = False,
) -> OrthogonalOutputEvidence:
    from .protocol import canonical_sha256

    manifest_path = output_directory / "orthogonal_outputs.json"
    _require_regular_file(manifest_path, "orthogonal output manifest")
    payload, raw = _strict_json(manifest_path, "orthogonal output manifest")
    canonical = _canonical_json_bytes(payload) + b"\n"
    if raw != canonical or set(payload) != {
        "schema_version",
        "artifact_type",
        "authority",
        "status",
        "planned_record_count",
        "records",
        "manifest_sha256",
    }:
        raise DevelopmentEvaluationError(
            "orthogonal output manifest is noncanonical or has wrong fields"
        )
    core = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    if (
        payload.get("schema_version") != 2
        or payload.get("artifact_type") != "maskimpute_orthogonal_method_outputs"
        or payload.get("authority") != dict(expected_authority)
        or payload.get("manifest_sha256") != canonical_sha256(core)
    ):
        raise DevelopmentEvaluationError(
            "orthogonal output manifest authority mismatch"
        )
    records = payload.get("records")
    if not isinstance(records, list):
        raise DevelopmentEvaluationError("orthogonal output records are invalid")
    inputs = expected_authority.get("inputs")
    configurations = expected_authority.get("configurations")
    seeds = expected_authority.get("model_seeds")
    if (
        not isinstance(inputs, list)
        or not isinstance(configurations, list)
        or not isinstance(seeds, list)
    ):
        raise DevelopmentEvaluationError("orthogonal expected authority is malformed")
    expected_rows: list[tuple[dict[str, object], str, str, int | None]] = []
    for input_row in inputs:
        if not isinstance(input_row, dict):
            raise DevelopmentEvaluationError("orthogonal input authority is malformed")
        expected_rows.append((input_row, "observed", "0" * 64, None))
        for configuration in configurations:
            if not isinstance(configuration, dict):
                raise DevelopmentEvaluationError(
                    "orthogonal configuration authority is malformed"
                )
            for seed in seeds:
                expected_rows.append(
                    (
                        input_row,
                        str(configuration.get("configuration_id")),
                        str(configuration.get("configuration_sha256")),
                        int(seed),
                    )
                )
    status = payload.get("status")
    if (
        status not in {"running", "completed"}
        or payload.get("planned_record_count") != len(expected_rows)
        or len(records) > len(expected_rows)
        or (status == "completed" and len(records) != len(expected_rows))
        or (status == "running" and len(records) >= len(expected_rows))
        or (status == "running" and not allow_running)
    ):
        raise DevelopmentEvaluationError(
            "orthogonal output checkpoint status or denominator is invalid"
        )
    record_fields = {
        "source_id",
        "configuration",
        "configuration_sha256",
        "model_seed",
        "method_input_sha256",
        "status",
        "reason",
        "output_path",
        "output_file_sha256",
        "output_compressed_nbytes",
        "output_encoding",
        "output_uncompressed_nbytes",
        "output_uncompressed_sha256",
        "output_shape",
        "output_dtype",
        "output_scale",
    }
    seen_output_paths: set[str] = set()
    for index, (record, expected_row) in enumerate(zip(records, expected_rows)):
        if not isinstance(record, dict) or set(record) != record_fields:
            raise DevelopmentEvaluationError(
                f"orthogonal output record {index} has wrong fields"
            )
        input_row, configuration_id, configuration_sha, seed = expected_row
        if (
            record.get("source_id") != input_row.get("source_id")
            or record.get("configuration") != configuration_id
            or record.get("configuration_sha256") != configuration_sha
            or record.get("model_seed") != seed
            or record.get("method_input_sha256") != input_row.get("method_input_sha256")
            or record.get("status") not in {"completed", "failed"}
        ):
            raise DevelopmentEvaluationError(
                f"orthogonal output record {index} mismatches authority"
            )
        path = record.get("output_path")
        digest = record.get("output_file_sha256")
        seed_token = "deterministic" if seed is None else f"seed-{seed}"
        expected_path = (
            f"outputs/{input_row.get('source_id')}--{configuration_id}--"
            f"{seed_token}.log2-cp10k-f64.zlib"
        )
        if record.get("status") == "completed":
            if (
                path != expected_path
                or not isinstance(digest, str)
                or record.get("reason") is not None
                or record.get("output_shape") != input_row.get("shape")
            ):
                raise DevelopmentEvaluationError(
                    "completed orthogonal output binding is incomplete"
                )
            if path in seen_output_paths:
                raise DevelopmentEvaluationError(
                    "orthogonal output paths are not unique"
                )
            seen_output_paths.add(path)
            relative = PurePosixPath(path)
            if relative.is_absolute() or ".." in relative.parts:
                raise DevelopmentEvaluationError("orthogonal output path is unsafe")
            output_path = output_directory.joinpath(*relative.parts)
            _require_regular_file(output_path, "orthogonal method output")
            _decode_orthogonal_output(output_path, record)
        elif (
            not isinstance(record.get("reason"), str)
            or not record.get("reason")
            or any(
                record.get(field) is not None
                for field in (
                    "output_path",
                    "output_file_sha256",
                    "output_compressed_nbytes",
                    "output_encoding",
                    "output_uncompressed_nbytes",
                    "output_uncompressed_sha256",
                    "output_shape",
                    "output_dtype",
                    "output_scale",
                )
            )
        ):
            raise DevelopmentEvaluationError(
                "failed orthogonal output binding is malformed"
            )
    return OrthogonalOutputEvidence(
        output_directory=output_directory,
        manifest_path=manifest_path,
        manifest_file_sha256=hashlib.sha256(raw).hexdigest(),
        manifest_sha256=str(payload["manifest_sha256"]),
        records=tuple(records),
    )


def produce_orthogonal_outputs(
    output_directory: Path,
    *,
    inputs: Sequence[OrthogonalInput],
    configurations: Sequence[OrthogonalConfiguration],
    model_seeds: Sequence[int],
    artifact_bindings: Mapping[str, object],
    executor: object,
) -> OrthogonalOutputEvidence:
    """Run a fixed truth-free orthogonal panel and seal every common-scale output."""

    from .methods import count_equivalent_to_log2_cp10k

    if not isinstance(output_directory, Path):
        raise TypeError("output_directory must be a pathlib.Path")
    input_values = tuple(inputs)
    configuration_values = tuple(configurations)
    seeds = tuple(model_seeds)
    if (
        not input_values
        or any(not isinstance(value, OrthogonalInput) for value in input_values)
        or len({value.source_id for value in input_values}) != len(input_values)
    ):
        raise ValueError("orthogonal inputs must be nonempty and source-unique")
    if (
        not configuration_values
        or any(
            not isinstance(value, OrthogonalConfiguration)
            for value in configuration_values
        )
        or len({value.configuration_id for value in configuration_values})
        != len(configuration_values)
    ):
        raise ValueError("orthogonal configurations must be nonempty and unique")
    if (
        not seeds
        or any(type(value) is not int or value not in {42, 43, 44} for value in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("orthogonal model seeds must be unique members of 42/43/44")
    if not callable(executor):
        raise TypeError("orthogonal executor must be callable")
    root = output_directory.absolute()
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise DevelopmentEvaluationError(
            "orthogonal output directory must not be a symlink"
        )
    authority = _orthogonal_authority_core(
        input_values, configuration_values, seeds, artifact_bindings
    )
    manifest_path = root / "orthogonal_outputs.json"
    planned_record_count = len(input_values) * (
        1 + len(configuration_values) * len(seeds)
    )
    if manifest_path.exists():
        checkpoint = _load_orthogonal_evidence(root, authority, allow_running=True)
        if len(checkpoint.records) == planned_record_count:
            return _load_orthogonal_evidence(root, authority)
        records = [dict(value) for value in checkpoint.records]
    else:
        records = []
    tasks: list[
        tuple[
            OrthogonalInput,
            str,
            str,
            int | None,
            OrthogonalConfiguration | None,
        ]
    ] = []
    for method_input in input_values:
        tasks.append((method_input, "observed", "0" * 64, None, None))
        for configuration in configuration_values:
            for seed in seeds:
                tasks.append(
                    (
                        method_input,
                        configuration.configuration_id,
                        configuration.configuration_sha256,
                        seed,
                        configuration,
                    )
                )
    if len(tasks) != planned_record_count:
        raise DevelopmentEvaluationError("orthogonal task denominator is inconsistent")
    _preflight_orthogonal_output_storage(
        root,
        remaining_shapes=tuple(
            value.method_input.shape for value, *_rest in tasks[len(records) :]
        ),
    )
    input_sha = {
        str(row["source_id"]): str(row["method_input_sha256"])
        for row in authority["inputs"]
    }
    for method_input, configuration_id, configuration_sha, seed, configuration in tasks[
        len(records) :
    ]:
        reason: str | None = None
        output: np.ndarray | None
        if configuration is None:
            output = count_equivalent_to_log2_cp10k(method_input.method_input.counts)
        else:
            try:
                raw_output = executor(
                    OrthogonalExecutionRequest(
                        source_id=method_input.source_id,
                        configuration=configuration,
                        model_seed=int(seed),
                        method_input=method_input.method_input,
                    )
                )
                raw = np.asarray(raw_output, dtype=np.float64)
                if (
                    raw.shape != method_input.method_input.shape
                    or not np.isfinite(raw).all()
                    or bool((raw < 0.0).any())
                ):
                    raise ValueError("executor returned an invalid raw-count output")
                output = count_equivalent_to_log2_cp10k(raw)
            except Exception as error:
                output = None
                reason = f"executor_error:{type(error).__name__}"
        seed_token = "deterministic" if seed is None else f"seed-{seed}"
        relative = (
            f"outputs/{method_input.source_id}--{configuration_id}--"
            f"{seed_token}.log2-cp10k-f64.zlib"
        )
        if output is None:
            output_path = output_digest = output_shape = None
            output_compressed_nbytes = None
            output_encoding = None
            output_uncompressed_nbytes = None
            output_uncompressed_sha256 = None
            status = "failed"
        else:
            encoded = np.asarray(output, dtype="<f8", order="C").tobytes(order="C")
            compressed = zlib.compress(
                encoded, level=_ORTHOGONAL_OUTPUT_COMPRESSION_LEVEL
            )
            output_digest = _publish_bound_file(root / relative, compressed)
            output_path = relative
            output_compressed_nbytes = len(compressed)
            output_encoding = _ORTHOGONAL_OUTPUT_ENCODING
            output_uncompressed_nbytes = len(encoded)
            output_uncompressed_sha256 = hashlib.sha256(encoded).hexdigest()
            output_shape = list(output.shape)
            status = "completed"
        records.append(
            {
                "source_id": method_input.source_id,
                "configuration": configuration_id,
                "configuration_sha256": configuration_sha,
                "model_seed": seed,
                "method_input_sha256": input_sha[method_input.source_id],
                "status": status,
                "reason": reason,
                "output_path": output_path,
                "output_file_sha256": output_digest,
                "output_compressed_nbytes": output_compressed_nbytes,
                "output_encoding": output_encoding,
                "output_uncompressed_nbytes": output_uncompressed_nbytes,
                "output_uncompressed_sha256": output_uncompressed_sha256,
                "output_shape": output_shape,
                "output_dtype": None if output is None else "<f8",
                "output_scale": None if output is None else "log2_cp10k_plus_1",
            }
        )
        _write_orthogonal_checkpoint(
            manifest_path, authority, records, planned_record_count
        )
    return _load_orthogonal_evidence(root, authority)


def _repository_file(
    repository: Path,
    relative_value: str,
    name: str,
) -> Path:
    relative = PurePosixPath(relative_value)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise DevelopmentEvaluationError(f"{name} path is unsafe")
    path = repository.joinpath(*relative.parts)
    current = path
    while current != repository.parent:
        if current.exists() and current.is_symlink():
            raise DevelopmentEvaluationError(f"{name} path contains a symlink")
        if current == repository:
            break
        current = current.parent
    return path


def _verify_bound_repository_file(
    repository: Path,
    relative: str,
    expected_sha256: str,
    name: str,
    *,
    expected_size: int | None = None,
) -> None:
    path = _repository_file(repository, relative, name)
    raw, actual_sha256 = _read_stable_bytes(path, name)
    if (
        not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)
        or actual_sha256 != expected_sha256
        or (expected_size is not None and len(raw) != expected_size)
    ):
        raise DevelopmentEvaluationError(f"{name} checksum or size mismatch")


def write_development_selection_artifacts(
    repository: Path,
    *,
    dataset_manifest_sha256: str,
    count_score_manifest_sha256: str,
    retained_calibration_artifact_sha256: str,
    reconstruction: ReconstructionEvidence,
    reconstruction_relative_directory: str,
    orthogonal: OrthogonalOutputEvidence,
    orthogonal_relative_directory: str,
    records: Sequence[Mapping[str, object]],
    intervals: Sequence[Mapping[str, object]],
    null_de_audits: Sequence[Mapping[str, object]],
    orthogonal_audits: Sequence[Mapping[str, object]],
    sources: RealSourceEvidence,
) -> tuple[Path, Path]:
    """Write canonical schema 2 plus its acyclic, byte-bound evidence manifest."""

    from .protocol import canonical_sha256

    if not isinstance(repository, Path):
        raise TypeError("repository must be a pathlib.Path")
    root = repository.absolute()
    if not isinstance(reconstruction, ReconstructionEvidence):
        raise TypeError("reconstruction must be ReconstructionEvidence")
    if not isinstance(orthogonal, OrthogonalOutputEvidence):
        raise TypeError("orthogonal must be OrthogonalOutputEvidence")
    if not isinstance(sources, RealSourceEvidence):
        raise TypeError("sources must be RealSourceEvidence")
    for name, value in (
        ("dataset_manifest_sha256", dataset_manifest_sha256),
        ("count_score_manifest_sha256", count_score_manifest_sha256),
        (
            "retained_calibration_artifact_sha256",
            retained_calibration_artifact_sha256,
        ),
    ):
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise DevelopmentEvaluationError(f"{name} is invalid")
    _verify_bound_repository_file(
        root,
        "artifacts/study/development/count_scores/manifest.json",
        count_score_manifest_sha256,
        "count-score manifest",
    )
    _verify_bound_repository_file(
        root,
        "artifacts/study/development/calibration/retained_calibration.json",
        retained_calibration_artifact_sha256,
        "retained calibration artifact",
    )
    _repository_file(
        root, reconstruction_relative_directory, "reconstruction directory"
    )
    checkpoint_relative = str(
        PurePosixPath(reconstruction_relative_directory)
        / reconstruction.checkpoint_path
    )
    _verify_bound_repository_file(
        root,
        checkpoint_relative,
        reconstruction.checkpoint_file_sha256,
        "reconstruction checkpoint",
    )
    for binding in reconstruction.raw_artifacts:
        relative = str(PurePosixPath(reconstruction_relative_directory) / binding.path)
        _verify_bound_repository_file(
            root, relative, binding.file_sha256, f"reconstruction {binding.kind}"
        )
    orthogonal_manifest_relative = str(
        PurePosixPath(orthogonal_relative_directory) / orthogonal.manifest_path.name
    )
    _verify_bound_repository_file(
        root,
        orthogonal_manifest_relative,
        orthogonal.manifest_file_sha256,
        "orthogonal output manifest",
    )
    for index, record in enumerate(orthogonal.records):
        if record.get("status") != "completed":
            continue
        path = record.get("output_path")
        digest = record.get("output_file_sha256")
        if not isinstance(path, str) or not isinstance(digest, str):
            raise DevelopmentEvaluationError(
                f"orthogonal output record {index} binding is incomplete"
            )
        relative = str(PurePosixPath(orthogonal_relative_directory) / path)
        _verify_bound_repository_file(
            root, relative, digest, f"orthogonal output record {index}"
        )
    _verify_bound_repository_file(
        root,
        sources.ledger_path,
        sources.ledger_file_sha256,
        "source ledger",
    )
    required_source_ids = {
        "baron-pancreas-umi",
        "cite-seq-cbmc-rna-protein",
        "tung-ipsc-ercc-bulk-replicates",
    }
    if {
        value.source_id for value in sources.receipts
    } != required_source_ids or not required_source_ids <= {
        value.source_id for value in sources.artifacts
    }:
        raise DevelopmentEvaluationError(
            "evaluation source evidence is missing a required real source"
        )
    for binding in sources.receipts:
        _verify_bound_repository_file(
            root, binding.path, binding.file_sha256, f"{binding.source_id} receipt"
        )
    for binding in sources.artifacts:
        _verify_bound_repository_file(
            root,
            binding.path,
            binding.file_sha256,
            f"{binding.source_id} source artifact",
            expected_size=binding.size_bytes,
        )
    result_records = [dict(value) for value in records]
    result_intervals = [dict(value) for value in intervals]
    if not result_records or not result_intervals:
        raise DevelopmentEvaluationError(
            "selection records and orthogonal intervals must be nonempty"
        )
    evidence_core = {
        "schema_version": 2,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "count_score_manifest_sha256": count_score_manifest_sha256,
        "retained_calibration_artifact_sha256": (retained_calibration_artifact_sha256),
        "records": result_records,
        "orthogonal_intervals": result_intervals,
    }
    evaluation_core = {
        "schema_version": 1,
        "artifact_type": "maskimpute_development_selection_evaluation_manifest",
        "selection_evidence_sha256": canonical_sha256(evidence_core),
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "count_score_manifest": {
            "path": "artifacts/study/development/count_scores/manifest.json",
            "file_sha256": count_score_manifest_sha256,
        },
        "retained_calibration_artifact": {
            "path": "artifacts/study/development/calibration/retained_calibration.json",
            "file_sha256": retained_calibration_artifact_sha256,
        },
        "reconstruction": {
            "checkpoint_path": checkpoint_relative,
            "checkpoint_file_sha256": reconstruction.checkpoint_file_sha256,
            "checkpoint_sha256": reconstruction.checkpoint_sha256,
            "plan_sha256": reconstruction.plan_sha256,
            "input_hashes": dict(reconstruction.input_hashes),
            "raw_artifacts": [
                {
                    **asdict(value),
                    "path": str(
                        PurePosixPath(reconstruction_relative_directory) / value.path
                    ),
                }
                for value in reconstruction.raw_artifacts
            ],
        },
        "orthogonal": {
            "manifest_path": orthogonal_manifest_relative,
            "manifest_file_sha256": orthogonal.manifest_file_sha256,
            "manifest_sha256": orthogonal.manifest_sha256,
            "records": [dict(value) for value in orthogonal.records],
        },
        "sources": {
            "ledger_path": sources.ledger_path,
            "ledger_file_sha256": sources.ledger_file_sha256,
            "ledger_sha256": sources.ledger_sha256,
            "receipts": [asdict(value) for value in sources.receipts],
            "artifacts": [asdict(value) for value in sources.artifacts],
        },
        "null_de_audits": [dict(value) for value in null_de_audits],
        "orthogonal_audits": [dict(value) for value in orthogonal_audits],
        "combined_score": None,
    }
    evaluation_payload = {
        **evaluation_core,
        "manifest_sha256": canonical_sha256(evaluation_core),
    }
    evaluation_relative = (
        "artifacts/study/development/evaluation/evaluation_manifest.json"
    )
    evaluation_path = _repository_file(root, evaluation_relative, "evaluation manifest")
    evaluation_file_sha = _publish_bound_file(
        evaluation_path, _canonical_json_bytes(evaluation_payload) + b"\n"
    )
    result_core = {
        **evidence_core,
        "evaluation_manifest_sha256": evaluation_file_sha,
    }
    result_payload = {**result_core, "result_sha256": canonical_sha256(result_core)}
    result_path = _repository_file(
        root,
        "artifacts/study/development/evaluation/development_selection_input.json",
        "development selection input",
    )
    _publish_bound_file(result_path, _canonical_json_bytes(result_payload) + b"\n")
    return result_path, evaluation_path


def _orthogonal_output_matrix(
    evidence: OrthogonalOutputEvidence,
    *,
    source_id: str,
    configuration: str,
    model_seed: int | None,
) -> np.ndarray | None:
    matches = [
        record
        for record in evidence.records
        if record.get("source_id") == source_id
        and record.get("configuration") == configuration
        and record.get("model_seed") == model_seed
    ]
    if len(matches) != 1:
        return None
    record = matches[0]
    if record.get("status") != "completed":
        return None
    path = record.get("output_path")
    shape = record.get("output_shape")
    if (
        not isinstance(path, str)
        or not isinstance(shape, list)
        or len(shape) != 2
    ):
        raise DevelopmentEvaluationError("orthogonal output binding is malformed")
    output_path = evidence.output_directory.joinpath(*PurePosixPath(path).parts)
    _require_regular_file(output_path, "orthogonal evaluator output")
    raw = _decode_orthogonal_output(output_path, record)
    values = np.frombuffer(raw, dtype="<f8")
    if values.size != int(shape[0]) * int(shape[1]):
        raise DevelopmentEvaluationError("orthogonal evaluator output shape mismatch")
    return values.reshape((int(shape[0]), int(shape[1]))).copy()


def _average_endpoint_seed_units(
    seed_units: Sequence[Sequence[EndpointUnit]],
) -> tuple[EndpointUnit, ...] | None:
    values = tuple(tuple(seed) for seed in seed_units)
    if not values:
        return None
    lookups = [
        {
            (unit.unit_id, unit.biological_id, unit.technical_id): unit.value
            for unit in seed
        }
        for seed in values
    ]
    identities = set(lookups[0])
    if not identities or any(set(lookup) != identities for lookup in lookups[1:]):
        return None
    return tuple(
        EndpointUnit(
            unit_id=identity[0],
            biological_id=identity[1],
            technical_id=identity[2],
            value=float(np.mean([lookup[identity] for lookup in lookups])),
        )
        for identity in sorted(identities)
    )


def _unavailable_endpoint_interval(
    configuration: str,
    endpoint: str,
    reason: str,
    n_boot: int,
) -> EndpointInterval:
    return EndpointInterval(
        configuration=configuration,
        endpoint=endpoint,
        comparison="observed",
        estimate=None,
        ci_lower=None,
        ci_upper=None,
        status="unavailable",
        reason=reason,
        n_biological_units=0,
        n_technical_units=0,
        n_boot=n_boot,
        bootstrap_sha256=_bootstrap_sha256(()),
    )


def evaluate_cite_orthogonal_interval(
    evidence: OrthogonalOutputEvidence,
    source: CiteSeqSource,
    configuration: str,
    *,
    n_boot: int = 10_000,
) -> EndpointInterval:
    """Evaluate a cell-bootstrap endpoint conditional on the single CBMC specimen."""

    if not isinstance(evidence, OrthogonalOutputEvidence):
        raise TypeError("evidence must be OrthogonalOutputEvidence")
    if not isinstance(source, CiteSeqSource):
        raise TypeError("source must be CiteSeqSource")
    source_id = "cite-seq-cbmc-rna-protein"
    observed_output = _orthogonal_output_matrix(
        evidence,
        source_id=source_id,
        configuration="observed",
        model_seed=None,
    )
    if observed_output is None:
        return _unavailable_endpoint_interval(
            configuration,
            "rna_protein_concordance",
            "observed_cite_output_unavailable",
            n_boot,
        )
    if observed_output.shape != source.rna_counts.shape:
        raise DevelopmentEvaluationError("CITE-seq output shape mismatches source")
    candidate_seed_outputs: list[np.ndarray] = []
    for seed in (42, 43, 44):
        output = _orthogonal_output_matrix(
            evidence,
            source_id=source_id,
            configuration=configuration,
            model_seed=seed,
        )
        if output is None:
            return _unavailable_endpoint_interval(
                configuration,
                "rna_protein_concordance",
                f"candidate_cite_output_unavailable_seed_{seed}",
                n_boot,
            )
        if output.shape != observed_output.shape:
            raise DevelopmentEvaluationError("CITE-seq candidate output shape mismatch")
        candidate_seed_outputs.append(output)
    candidate_output = np.mean(np.stack(candidate_seed_outputs, axis=0), axis=0)
    return _cite_conditional_cell_bootstrap_interval(
        configuration=configuration,
        candidate_output=candidate_output,
        observed_output=observed_output,
        source=source,
        n_boot=n_boot,
    )


def _cite_conditional_cell_bootstrap_interval(
    *,
    configuration: str,
    candidate_output: np.ndarray,
    observed_output: np.ndarray,
    source: CiteSeqSource,
    n_boot: int,
) -> EndpointInterval:
    if type(n_boot) is not int or n_boot <= 0:
        raise ValueError("n_boot must be a positive integer")
    matched = _matched_rna_protein_indices(source.gene_ids, source.protein_ids)
    if len(matched) < 3:
        return _unavailable_endpoint_interval(
            configuration,
            "rna_protein_concordance",
            "fewer_than_three_matched_cite_markers",
            n_boot,
        )
    rows = np.arange(observed_output.shape[0])
    observed = _rna_protein_marker_correlations(
        observed_output, source.protein_counts, matched, rows
    )
    fixed = np.isfinite(observed)
    if int(fixed.sum()) < 3:
        return _unavailable_endpoint_interval(
            configuration,
            "rna_protein_concordance",
            "fewer_than_three_variable_observed_cite_markers",
            n_boot,
        )
    candidate = _rna_protein_marker_correlations(
        candidate_output, source.protein_counts, matched, rows
    )
    if not np.isfinite(candidate[fixed]).all():
        return _unavailable_endpoint_interval(
            configuration,
            "rna_protein_concordance",
            "candidate_non_testable_on_fixed_cite_marker_denominator",
            n_boot,
        )
    estimate = float(np.median(candidate[fixed] - observed[fixed]))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    distribution: list[float] = []
    attempts = 0
    maximum_attempts = max(n_boot * 20, n_boot + 100)
    while len(distribution) < n_boot and attempts < maximum_attempts:
        attempts += 1
        sampled = rng.integers(0, observed_output.shape[0], observed_output.shape[0])
        observed_sample = _rna_protein_marker_correlations(
            observed_output, source.protein_counts, matched, sampled
        )
        candidate_sample = _rna_protein_marker_correlations(
            candidate_output, source.protein_counts, matched, sampled
        )
        if not (
            np.isfinite(observed_sample[fixed]).all()
            and np.isfinite(candidate_sample[fixed]).all()
        ):
            continue
        distribution.append(
            float(np.median(candidate_sample[fixed] - observed_sample[fixed]))
        )
    if len(distribution) != n_boot:
        return _unavailable_endpoint_interval(
            configuration,
            "rna_protein_concordance",
            "cite_cell_bootstrap_denominator_unstable",
            n_boot,
        )
    lower, upper = np.quantile(np.asarray(distribution), (0.025, 0.975))
    return EndpointInterval(
        configuration=configuration,
        endpoint="rna_protein_concordance",
        comparison="observed",
        estimate=estimate,
        ci_lower=float(lower),
        ci_upper=float(upper),
        status="completed",
        reason=None,
        n_biological_units=1,
        n_technical_units=int(fixed.sum()),
        n_boot=n_boot,
        bootstrap_sha256=_bootstrap_sha256(distribution),
    )


def evaluate_tung_orthogonal_intervals(
    evidence: OrthogonalOutputEvidence,
    source: TungSource,
    configuration: str,
    *,
    n_boot: int = 10_000,
) -> tuple[EndpointInterval, ...]:
    """Evaluate three Tung endpoints with replicates nested in individuals."""

    if not isinstance(evidence, OrthogonalOutputEvidence):
        raise TypeError("evidence must be OrthogonalOutputEvidence")
    if not isinstance(source, TungSource):
        raise TypeError("source must be TungSource")
    source_id = "tung-ipsc-ercc-bulk-replicates"
    endpoints = (
        "ercc_recovery",
        "technical_replicate_concordance",
        "bulk_pseudobulk_concordance",
    )
    observed_output = _orthogonal_output_matrix(
        evidence,
        source_id=source_id,
        configuration="observed",
        model_seed=None,
    )
    if observed_output is None:
        return tuple(
            _unavailable_endpoint_interval(
                configuration, endpoint, "observed_tung_output_unavailable", n_boot
            )
            for endpoint in endpoints
        )
    observed = tung_concordance_units(observed_output, source)
    by_seed: list[Mapping[str, tuple[EndpointUnit, ...]]] = []
    for seed in (42, 43, 44):
        output = _orthogonal_output_matrix(
            evidence,
            source_id=source_id,
            configuration=configuration,
            model_seed=seed,
        )
        if output is None:
            return tuple(
                _unavailable_endpoint_interval(
                    configuration,
                    endpoint,
                    f"candidate_tung_output_unavailable_seed_{seed}",
                    n_boot,
                )
                for endpoint in endpoints
            )
        by_seed.append(tung_concordance_units(output, source))
    intervals: list[EndpointInterval] = []
    for endpoint in endpoints:
        candidate = _average_endpoint_seed_units([value[endpoint] for value in by_seed])
        if candidate is None:
            intervals.append(
                _unavailable_endpoint_interval(
                    configuration,
                    endpoint,
                    "candidate_tung_endpoint_units_incomplete",
                    n_boot,
                )
            )
            continue
        intervals.append(
            hierarchical_endpoint_interval(
                configuration=configuration,
                endpoint=endpoint,
                candidate_units=candidate,
                observed_units=observed[endpoint],
                n_boot=n_boot,
            )
        )
    return tuple(intervals)


def evaluate_real_orthogonal_intervals(
    evidence: OrthogonalOutputEvidence,
    cite_source: CiteSeqSource,
    tung_source: TungSource,
    configurations: Sequence[str],
    *,
    n_boot: int = 10_000,
) -> OrthogonalSelectionBundle:
    """Emit all four exact selection intervals and detailed unavailable reasons."""

    configuration_values = tuple(configurations)
    if (
        not configuration_values
        or any(
            not isinstance(value, str) or not re.fullmatch(r"[a-z][a-z0-9-]*", value)
            for value in configuration_values
        )
        or len(set(configuration_values)) != len(configuration_values)
    ):
        raise ValueError("orthogonal configurations must be nonempty and unique")
    values: list[EndpointInterval] = []
    for configuration in configuration_values:
        values.append(
            evaluate_cite_orthogonal_interval(
                evidence, cite_source, configuration, n_boot=n_boot
            )
        )
        values.extend(
            evaluate_tung_orthogonal_intervals(
                evidence, tung_source, configuration, n_boot=n_boot
            )
        )
    intervals = tuple(value.selection_row() for value in values)
    audits = tuple(
        {
            **value.selection_row(),
            "reason": value.reason,
            "n_biological_units": value.n_biological_units,
            "n_technical_units": value.n_technical_units,
            "n_boot": value.n_boot,
            "bootstrap_sha256": value.bootstrap_sha256,
            "aggregation": (
                (
                    "average_model_seed_outputs_then_paired_cell_bootstrap_of_"
                    "median_matched_marker_correlations"
                )
                if value.endpoint == "rna_protein_concordance"
                else (
                    "average_model_seeds_then_average_technical_units_within_"
                    "biological_unit_then_hierarchical_bootstrap"
                )
            ),
            "inference_scope": (
                "cell_level_conditional_on_one_cbmc_specimen_not_cross_specimen"
                if value.endpoint == "rna_protein_concordance"
                else "biological_individuals_with_nested_technical_units"
            ),
            "profile_scale": (
                "matched_marker_rank_correlation_across_cells"
                if value.endpoint == "rna_protein_concordance"
                else (
                    "fixed_observed_library_size_weighted_count_equivalent_pseudobulk"
                )
            ),
        }
        for value in values
    )
    return OrthogonalSelectionBundle(intervals=intervals, audits=audits)


def prepare_real_orthogonal_panel(repository: Path) -> PreparedRealOrthogonalPanel:
    """Byte-validate and prepare the three fixed real sources for evaluation."""

    import anndata as ad
    import pandas as pd

    from .methods import prepare_method_input
    from .protocol import canonical_sha256

    evidence = validate_real_source_artifacts(repository)
    data_root = repository.absolute() / "artifacts/external/data"
    baron = prepare_baron_source(data_root / "baron-pancreas-umi/GSE84133_RAW.tar")
    cite = prepare_cite_seq_source(
        data_root
        / "cite-seq-cbmc-rna-protein/GSE100866_CBMC_8K_13AB_10X-RNA_umi.csv.gz",
        data_root
        / "cite-seq-cbmc-rna-protein/GSE100866_CBMC_8K_13AB_10X-ADT_umi.csv.gz",
    )
    tung = prepare_tung_source(
        data_root
        / "tung-ipsc-ercc-bulk-replicates/GSE77288_molecules-raw-single-per-sample.txt.gz",
        data_root
        / "tung-ipsc-ercc-bulk-replicates/GSE77288_reads-raw-bulk-per-sample.txt.gz",
        data_root
        / "tung-ipsc-ercc-bulk-replicates/GSE77288_molecules-raw-single-per-lane.txt.gz",
    )

    expected_source_sha = {
        (value.source_id, PurePosixPath(value.path).name): value.file_sha256
        for value in evidence.artifacts
    }
    prepared_source_sha = {
        ("baron-pancreas-umi", "GSE84133_RAW.tar"): baron.archive_file_sha256,
        (
            "cite-seq-cbmc-rna-protein",
            "GSE100866_CBMC_8K_13AB_10X-RNA_umi.csv.gz",
        ): cite.rna_file_sha256,
        (
            "cite-seq-cbmc-rna-protein",
            "GSE100866_CBMC_8K_13AB_10X-ADT_umi.csv.gz",
        ): cite.protein_file_sha256,
        (
            "tung-ipsc-ercc-bulk-replicates",
            "GSE77288_molecules-raw-single-per-sample.txt.gz",
        ): tung.single_sample_file_sha256,
        (
            "tung-ipsc-ercc-bulk-replicates",
            "GSE77288_reads-raw-bulk-per-sample.txt.gz",
        ): tung.bulk_sample_file_sha256,
        (
            "tung-ipsc-ercc-bulk-replicates",
            "GSE77288_molecules-raw-single-per-lane.txt.gz",
        ): tung.single_lane_file_sha256,
    }
    if any(
        expected_source_sha.get(identity) != digest
        for identity, digest in prepared_source_sha.items()
    ):
        raise DevelopmentEvaluationError(
            "prepared source bytes differ from the validated source evidence"
        )

    def make_input(
        source_id: str,
        counts: np.ndarray,
        cell_ids: Sequence[str],
        gene_ids: Sequence[str],
        source_hashes: Sequence[str],
        panel_rule: str,
    ) -> OrthogonalInput:
        source_sha = canonical_sha256(
            {
                "source_id": source_id,
                "source_file_sha256": list(source_hashes),
                "panel_rule": panel_rule,
                "cell_ids": list(cell_ids),
                "gene_ids": list(gene_ids),
                "shape": list(counts.shape),
            }
        )
        dataset = ad.AnnData(
            X=counts,
            obs=pd.DataFrame(index=list(cell_ids)),
            var=pd.DataFrame(index=list(gene_ids)),
        )
        dataset.uns["source_dataset_sha256"] = source_sha
        dataset.uns["allowed_covariates"] = {"obs": [], "var": []}
        return OrthogonalInput(source_id, prepare_method_input(dataset))

    method_inputs = (
        make_input(
            "cite-seq-cbmc-rna-protein",
            cite.rna_counts,
            cite.cell_ids,
            cite.gene_ids,
            (cite.rna_file_sha256,),
            (
                "top_500_human_genes_by_total_observed_rna_count_with_fixed_"
                "antibody_matched_genes_forced;adt_values_withheld"
            ),
        ),
        make_input(
            "tung-ipsc-ercc-bulk-replicates",
            tung.counts,
            tung.cell_ids,
            tung.gene_ids,
            (tung.single_sample_file_sha256,),
            "all_single_cell_genes;bulk_and_lane_profiles_withheld",
        ),
    )
    return PreparedRealOrthogonalPanel(evidence, baron, cite, tung, method_inputs)


def load_orthogonal_output_evidence(
    output_directory: Path,
    *,
    expected_authority: Mapping[str, object],
) -> OrthogonalOutputEvidence:
    """Revalidate an existing manifest against freshly derived authority."""

    if not isinstance(expected_authority, Mapping):
        raise TypeError("expected_authority must be a mapping")
    return _load_orthogonal_evidence(output_directory.absolute(), expected_authority)


def _thaw_authority_value(value: object) -> object:
    if isinstance(value, tuple):
        if all(
            isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)
            for item in value
        ):
            return {str(item[0]): _thaw_authority_value(item[1]) for item in value}
        return [_thaw_authority_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _thaw_authority_value(item) for key, item in value.items()}
    return value


def run_real_orthogonal_outputs(
    repository: Path,
    output_directory: Path,
    *,
    prepared_panel: PreparedRealOrthogonalPanel | None = None,
) -> OrthogonalOutputEvidence:
    """Run all 20 candidate configurations on fixed real inputs with truth withheld."""

    from maskimpute import PreZeroCountModelConfig
    from maskimpute.calibration import load_calibration_artifact

    from .runner import load_runner_authority

    root = repository.resolve(strict=True)
    if root != Path(__file__).resolve().parents[1]:
        raise DevelopmentEvaluationError(
            "orthogonal publication execution must use the active repository"
        )
    panel = (
        prepare_real_orthogonal_panel(root)
        if prepared_panel is None
        else prepared_panel
    )
    if not isinstance(panel, PreparedRealOrthogonalPanel):
        raise TypeError("prepared_panel must be PreparedRealOrthogonalPanel")
    authority = load_runner_authority()
    configurations = tuple(
        OrthogonalConfiguration(
            configuration_id=value.configuration_id,
            configuration_sha256=value.configuration_sha256,
            payload=dict(value.payload),
        )
        for value in authority.configurations
        if value.kind == "candidate_search"
    )
    if len(configurations) != 20:
        raise DevelopmentEvaluationError(
            "orthogonal runner requires the exact 20 candidate configurations"
        )
    calibration_sha = authority.retained_calibration_sha256
    if calibration_sha is None:
        raise DevelopmentEvaluationError(
            "retained calibration authority is pending for orthogonal execution"
        )
    calibration_path = root / authority.retained_calibration_path
    calibration_raw, calibration_actual_sha = _read_stable_bytes(
        calibration_path, "retained calibration artifact"
    )
    if calibration_actual_sha != calibration_sha:
        raise DevelopmentEvaluationError(
            "retained calibration checksum failed before orthogonal execution"
        )
    count_config_payload = _thaw_authority_value(authority.count_model_config)
    if not isinstance(count_config_payload, dict):
        raise DevelopmentEvaluationError("count-model authority is malformed")
    descriptor, temporary_name = tempfile.mkstemp(suffix=".json")
    temporary_calibration = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(calibration_raw)
            stream.flush()
            os.fsync(stream.fileno())
        calibration_artifact = load_calibration_artifact(temporary_calibration)
    finally:
        temporary_calibration.unlink(missing_ok=True)
    executor = PublicMaskImputeOrthogonalExecutor(
        count_model_config=PreZeroCountModelConfig(**count_config_payload),
        calibration_artifact=calibration_artifact,
    )
    return produce_orthogonal_outputs(
        output_directory,
        inputs=panel.method_inputs,
        configurations=configurations,
        model_seeds=(42, 43, 44),
        artifact_bindings={
            "count_model_config_sha256": authority.count_model_config_sha256,
            "retained_calibration_artifact_sha256": calibration_sha,
            "score_fit_policy": (
                "refit_cross_fitted_count_score_from_truth_free_input"
            ),
        },
        executor=executor,
    )


def build_development_selection_input(
    repository: Path,
    *,
    reconstruction_relative_directory: str = (
        "artifacts/study/development/competition-reconstruction"
    ),
    orthogonal_relative_directory: str = (
        "artifacts/study/development/evaluation/orthogonal"
    ),
) -> tuple[Path, Path]:
    """Build the fixed schema-2 selection input from completed immutable evidence."""

    from .methods import load_method_registry
    from .runner import (
        build_competition_plan,
        load_prepared_development_panel,
        load_runner_authority,
    )
    from .selection import load_publication_execution_authority

    root = repository.resolve(strict=True)
    module_root = Path(__file__).resolve().parents[1]
    if root != module_root:
        raise DevelopmentEvaluationError(
            "publication selection input must be built in the active repository"
        )
    selection = load_publication_execution_authority()
    runner_authority = load_runner_authority()
    bindings, prepared = load_prepared_development_panel(runner_authority)
    checkpoint_directory = _repository_file(
        root, reconstruction_relative_directory, "reconstruction directory"
    )
    checkpoint_payload, _ = _strict_json(
        checkpoint_directory / "checkpoint.json", "reconstruction checkpoint"
    )
    checkpoint_inputs = checkpoint_payload.get("input_hashes")
    if not isinstance(checkpoint_inputs, dict):
        raise DevelopmentEvaluationError("checkpoint input hashes are invalid")
    environment_sha = checkpoint_inputs.get("execution_environment_sha256")
    if not isinstance(environment_sha, str):
        raise DevelopmentEvaluationError(
            "checkpoint execution environment binding is absent"
        )
    registry = load_method_registry(root / "study/methods.json")
    plan = build_competition_plan(
        registry,
        bindings,
        runner_authority,
        execution_environment_sha256=environment_sha,
    )
    reconstruction = load_completed_reconstruction_checkpoint(
        checkpoint_directory, plan
    )
    reconstruction_bundle = build_reconstruction_selection_records(
        reconstruction,
        checkpoint_directory=checkpoint_directory,
        prepared_datasets=prepared,
        declarations=selection.declarations,
        method_bindings=selection.method_bindings,
    )
    real_panel = prepare_real_orthogonal_panel(root)
    orthogonal_directory = _repository_file(
        root, orthogonal_relative_directory, "orthogonal directory"
    )
    count_score_sha = selection.count_score_manifest.sha256
    calibration_sha = selection.retained_calibration.sha256
    if count_score_sha is None or calibration_sha is None:
        raise DevelopmentEvaluationError(
            "score/calibration authority is pending and blocks selection input"
        )
    orthogonal_configurations = tuple(
        OrthogonalConfiguration(
            configuration_id=value.configuration_id,
            configuration_sha256=value.configuration_sha256,
            payload=dict(value.payload),
        )
        for value in runner_authority.configurations
        if value.kind == "candidate_search"
    )
    orthogonal_artifact_bindings = {
        "count_model_config_sha256": runner_authority.count_model_config_sha256,
        "retained_calibration_artifact_sha256": calibration_sha,
        "score_fit_policy": ("refit_cross_fitted_count_score_from_truth_free_input"),
    }
    expected_orthogonal_authority = _orthogonal_authority_core(
        real_panel.method_inputs,
        orthogonal_configurations,
        (42, 43, 44),
        orthogonal_artifact_bindings,
    )
    orthogonal = (
        load_orthogonal_output_evidence(
            orthogonal_directory,
            expected_authority=expected_orthogonal_authority,
        )
        if (orthogonal_directory / "orthogonal_outputs.json").is_file()
        else run_real_orthogonal_outputs(
            root, orthogonal_directory, prepared_panel=real_panel
        )
    )
    configuration_ids = tuple(
        attempt.configuration_id for attempt in selection.attempts
    )
    endpoint_bundle = evaluate_real_orthogonal_intervals(
        orthogonal,
        real_panel.cite,
        real_panel.tung,
        configuration_ids,
    )
    return write_development_selection_artifacts(
        root,
        dataset_manifest_sha256=str(
            reconstruction.input_hashes["dataset_manifest_sha256"]
        ),
        count_score_manifest_sha256=count_score_sha,
        retained_calibration_artifact_sha256=calibration_sha,
        reconstruction=reconstruction,
        reconstruction_relative_directory=reconstruction_relative_directory,
        orthogonal=orthogonal,
        orthogonal_relative_directory=orthogonal_relative_directory,
        records=reconstruction_bundle.records,
        intervals=endpoint_bundle.intervals,
        null_de_audits=reconstruction_bundle.null_de_audits,
        orthogonal_audits=endpoint_bundle.audits,
        sources=real_panel.source_evidence,
    )


def load_completed_reconstruction_checkpoint(
    checkpoint_directory: Path,
    plan: object,
) -> ReconstructionEvidence:
    """Load an exact completed runner checkpoint and enumerate its bound files."""

    from .runner import (
        CheckpointStore,
        CompetitionPlan,
        RunnerContractError,
    )
    from .protocol import canonical_sha256

    if not isinstance(checkpoint_directory, Path):
        raise TypeError("checkpoint_directory must be a pathlib.Path")
    if not isinstance(plan, CompetitionPlan):
        raise TypeError("plan must be a CompetitionPlan")
    store = CheckpointStore(checkpoint_directory)
    try:
        report = store.load(plan)
        if (
            report.status != "completed"
            or len(report.records) != report.planned_run_count
        ):
            raise DevelopmentEvaluationError(
                "reconstruction checkpoint is not complete"
            )
        checkpoint_path = store.checkpoint_path
        checkpoint_raw, checkpoint_file_sha256 = _read_stable_bytes(
            checkpoint_path, "reconstruction checkpoint"
        )
        checkpoint_payload = json.loads(
            checkpoint_raw.decode("utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_json_object,
        )
        if (
            not isinstance(checkpoint_payload, dict)
            or set(checkpoint_payload)
            != {
                "schema_version",
                "plan_sha256",
                "input_hashes",
                "planned_run_count",
                "status",
                "evaluation_scope",
                "selection_complete",
                "selection_blockers",
                "records",
                "budget",
                "checkpoint_sha256",
            }
            or checkpoint_raw != _canonical_json_bytes(checkpoint_payload) + b"\n"
            or checkpoint_payload.get("checkpoint_sha256")
            != canonical_sha256(
                {
                    key: value
                    for key, value in checkpoint_payload.items()
                    if key != "checkpoint_sha256"
                }
            )
            or checkpoint_payload.get("checkpoint_sha256") != report.checkpoint_sha256
            or checkpoint_payload.get("schema_version") != report.schema_version
            or checkpoint_payload.get("plan_sha256") != report.plan_sha256
            or checkpoint_payload.get("input_hashes") != dict(report.input_hashes)
            or checkpoint_payload.get("planned_run_count") != report.planned_run_count
            or checkpoint_payload.get("status") != report.status
            or checkpoint_payload.get("evaluation_scope") != report.evaluation_scope
            or checkpoint_payload.get("selection_complete") != report.selection_complete
            or checkpoint_payload.get("selection_blockers")
            != list(report.selection_blockers)
            or checkpoint_payload.get("records") != list(report.records)
            or checkpoint_payload.get("budget") != dict(report.budget)
        ):
            raise DevelopmentEvaluationError(
                "reconstruction checkpoint changed after runner validation"
            )
        raw_artifacts: list[RawArtifactBinding] = []
        for index, record in enumerate(report.records):
            run = record.get("run")
            if not isinstance(run, Mapping):
                raise DevelopmentEvaluationError(
                    f"checkpoint record {index} run is invalid"
                )
            run_id = run.get("run_id")
            if not isinstance(run_id, str) or not run_id:
                raise DevelopmentEvaluationError(
                    f"checkpoint record {index} run_id is invalid"
                )
            for kind in ("stdout", "stderr", "native_output", "evaluator_output"):
                path = run.get(f"{kind}_path")
                if path is None:
                    if kind == "evaluator_output" and run.get("status") == "completed":
                        raise DevelopmentEvaluationError(
                            f"completed run {run_id} lacks evaluator output"
                        )
                    continue
                file_digest = run.get(f"{kind}_file_sha256")
                if not isinstance(file_digest, str) or not re.fullmatch(
                    r"[0-9a-f]{64}", file_digest
                ):
                    raise DevelopmentEvaluationError(
                        f"checkpoint {kind} binding is incomplete"
                    )
                relative = PurePosixPath(str(path))
                if relative.is_absolute() or ".." in relative.parts:
                    raise DevelopmentEvaluationError(
                        f"checkpoint {kind} path is unsafe"
                    )
                artifact_path = checkpoint_directory.joinpath(*relative.parts)
                for parent in (artifact_path, *artifact_path.parents):
                    if parent == checkpoint_directory.parent:
                        break
                    if parent.is_symlink():
                        raise DevelopmentEvaluationError(
                            f"checkpoint {kind} path contains a symlink"
                        )
                _raw, actual_file_digest = _read_stable_bytes(
                    artifact_path, f"reconstruction {kind} artifact"
                )
                if actual_file_digest != file_digest:
                    raise DevelopmentEvaluationError(
                        f"checkpoint {kind} raw-byte checksum changed"
                    )
                if (
                    kind in {"stdout", "stderr"}
                    and run.get(f"{kind}_sha256") != actual_file_digest
                ):
                    raise DevelopmentEvaluationError(
                        f"checkpoint {kind} content checksum changed"
                    )
                raw_artifacts.append(
                    RawArtifactBinding(
                        run_id=run_id,
                        kind=kind,
                        path=str(path),
                        file_sha256=str(file_digest),
                    )
                )
            score_evidence = record.get("p_pre_zero_evidence")
            if not isinstance(score_evidence, Mapping):
                raise DevelopmentEvaluationError(
                    f"checkpoint record {index} p_pre_zero evidence is invalid"
                )
            score_storage = score_evidence.get("storage")
            if not isinstance(score_storage, Mapping):
                raise DevelopmentEvaluationError(
                    f"checkpoint record {index} p_pre_zero storage is invalid"
                )
            score_path = score_storage.get("path")
            if score_path is not None:
                score_digest = score_storage.get("compressed_sha256")
                if not isinstance(score_path, str) or not isinstance(
                    score_digest, str
                ):
                    raise DevelopmentEvaluationError(
                        "checkpoint p_pre_zero binding is incomplete"
                    )
                relative = PurePosixPath(score_path)
                if relative.is_absolute() or ".." in relative.parts:
                    raise DevelopmentEvaluationError(
                        "checkpoint p_pre_zero path is unsafe"
                    )
                score_artifact_path = checkpoint_directory.joinpath(*relative.parts)
                _score_raw, actual_score_digest = _read_stable_bytes(
                    score_artifact_path, "reconstruction p_pre_zero artifact"
                )
                if actual_score_digest != score_digest:
                    raise DevelopmentEvaluationError(
                        "checkpoint p_pre_zero raw-byte checksum changed"
                    )
                raw_artifacts.append(
                    RawArtifactBinding(
                        run_id=run_id,
                        kind="p_pre_zero",
                        path=score_path,
                        file_sha256=score_digest,
                    )
                )
    except DevelopmentEvaluationError:
        raise
    except (RunnerContractError, OSError, TypeError, ValueError) as error:
        raise DevelopmentEvaluationError(
            f"reconstruction checkpoint validation failed: {error}"
        ) from error
    return ReconstructionEvidence(
        checkpoint_path="checkpoint.json",
        checkpoint_file_sha256=checkpoint_file_sha256,
        checkpoint_sha256=report.checkpoint_sha256,
        plan_sha256=report.plan_sha256,
        input_hashes=dict(report.input_hashes),
        records=report.records,
        raw_artifacts=tuple(raw_artifacts),
    )


_SELECTION_RECONSTRUCTION_METRICS = (
    "mse",
    "mse_dropout",
    "gnrmse",
    "mse_pre_dropout_zero",
    "corr_err",
    "mse_non_dropout_nonzero",
)
_SELECTION_FAILED_STATUS = {
    "failed": "failed",
    "timeout": "timeout",
    "unavailable": "unavailable",
    "resource_exceeded": "resource_exceeded",
    "infrastructure_error": "failed",
    "blocked_authority": "failed",
    "budget_exhausted": "failed",
}


def _selection_method(run: Mapping[str, object], declared: set[str]) -> str | None:
    configuration_id = run.get("configuration_id")
    if (
        run.get("configuration_kind") == "candidate_search"
        and isinstance(configuration_id, str)
        and configuration_id in declared
    ):
        return configuration_id
    method_id = run.get("method_id")
    return method_id if isinstance(method_id, str) and method_id in declared else None


def _selection_status(status: object) -> str:
    if status == "completed":
        return "completed"
    if isinstance(status, str) and status in _SELECTION_FAILED_STATUS:
        return _SELECTION_FAILED_STATUS[status]
    raise DevelopmentEvaluationError(f"runner status {status!r} is not recognized")


def _read_evaluator_output(
    checkpoint_directory: Path,
    run: Mapping[str, object],
) -> np.ndarray:
    relative = run.get("evaluator_output_path")
    shape = run.get("evaluator_output_shape")
    expected = run.get("evaluator_output_file_sha256")
    if (
        not isinstance(relative, str)
        or not isinstance(shape, list)
        or len(shape) != 2
        or any(type(value) is not int or value <= 0 for value in shape)
        or not isinstance(expected, str)
    ):
        raise DevelopmentEvaluationError("evaluator output binding is incomplete")
    from pathlib import PurePosixPath

    safe = PurePosixPath(relative)
    if safe.is_absolute() or ".." in safe.parts:
        raise DevelopmentEvaluationError("evaluator output path is unsafe")
    path = checkpoint_directory.joinpath(*safe.parts)
    raw, actual_digest = _read_stable_bytes(path, "evaluator output")
    if actual_digest != expected:
        raise DevelopmentEvaluationError("evaluator output file checksum mismatch")
    values = np.frombuffer(raw, dtype="<f8")
    if values.size != shape[0] * shape[1]:
        raise DevelopmentEvaluationError("evaluator output byte size is invalid")
    output = values.reshape((shape[0], shape[1])).copy()
    if not np.isfinite(output).all():
        raise DevelopmentEvaluationError("evaluator output is nonfinite")
    return output


def _null_de_entropy_sha256(
    evidence: ReconstructionEvidence,
    run: Mapping[str, object],
) -> str:
    """Derive assignment entropy only after the complete checkpoint is sealed."""

    mechanism = run.get("mechanism")
    biological_id = run.get("biological_id")
    if not isinstance(mechanism, str) or not isinstance(biological_id, str):
        raise DevelopmentEvaluationError("null-DE run identity is incomplete")
    digest = hashlib.sha256()
    digest.update(b"maskimpute-null-de-post-execution-entropy-v1\0")
    digest.update(evidence.checkpoint_sha256.encode("ascii"))
    digest.update(b"\0")
    digest.update(mechanism.encode("utf-8"))
    digest.update(b"\0")
    digest.update(biological_id.encode("utf-8"))
    return digest.hexdigest()


def _null_de_sentinel(kind: str, entropy_sha256: str, reason: str) -> str:
    return hashlib.sha256(
        f"maskimpute-null-de-{kind}-v1\0{entropy_sha256}\0{reason}".encode()
    ).hexdigest()


def _null_de_unavailable_reason(error: ValueError) -> str:
    message = str(error)
    if "at least four" in message:
        return "insufficient_cells_per_stratum"
    if "fewer than" in message:
        return "insufficient_fixed_gene_denominator"
    if "residual degrees" in message:
        return "insufficient_residual_degrees_of_freedom"
    return f"null_de_design_invalid:{type(error).__name__}"


def build_reconstruction_selection_records(
    evidence: ReconstructionEvidence,
    *,
    checkpoint_directory: Path,
    prepared_datasets: Mapping[str, object],
    declarations: Sequence[object],
    method_bindings: Mapping[str, str],
) -> ReconstructionSelectionBundle:
    """Bridge runner rows to selection rows and append evaluator-only null-DE."""

    from .methods import count_equivalent_to_log2_cp10k
    from .runner import PreparedDataset, method_input_sha256
    from .selection import MethodDeclaration

    if not isinstance(evidence, ReconstructionEvidence):
        raise TypeError("evidence must be ReconstructionEvidence")
    if not isinstance(checkpoint_directory, Path):
        raise TypeError("checkpoint_directory must be a pathlib.Path")
    declaration_values = tuple(declarations)
    if any(not isinstance(value, MethodDeclaration) for value in declaration_values):
        raise TypeError("declarations must contain MethodDeclaration values")
    declared = {value.id for value in declaration_values}
    if set(method_bindings) < declared:
        raise DevelopmentEvaluationError("selection method bindings are incomplete")
    records: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    null_designs: dict[str, tuple[str, np.ndarray | None, str, str, str | None]] = {}
    for record_index, stored in enumerate(evidence.records):
        run = stored.get("run")
        metric_rows = stored.get("metrics")
        if not isinstance(run, Mapping) or not isinstance(metric_rows, list):
            raise DevelopmentEvaluationError(
                f"checkpoint record {record_index} is malformed"
            )
        method = _selection_method(run, declared)
        if method is None:
            continue
        dataset_id = run.get("dataset_id")
        prepared = (
            prepared_datasets.get(dataset_id) if isinstance(dataset_id, str) else None
        )
        if not isinstance(prepared, PreparedDataset):
            raise DevelopmentEvaluationError(
                f"prepared evaluator dataset is missing for {dataset_id!r}"
            )
        if (
            run.get("source_dataset_sha256") != prepared.binding.dataset_sha256
            or run.get("method_input_sha256")
            != method_input_sha256(prepared.method_input)
            or run.get("retained_cell_ids_sha256")
            != prepared.audit.retained_cell_ids_sha256
        ):
            raise DevelopmentEvaluationError(
                f"checkpoint dataset/QC binding mismatches {dataset_id}"
            )
        metric_lookup: dict[str, Mapping[str, object]] = {}
        for metric in metric_rows:
            if not isinstance(metric, Mapping) or not isinstance(
                metric.get("metric"), str
            ):
                raise DevelopmentEvaluationError("runner metric row is malformed")
            name = str(metric["metric"])
            if name in metric_lookup:
                raise DevelopmentEvaluationError("runner metric identity is duplicated")
            metric_lookup[name] = metric
        common = {
            "mechanism": run.get("mechanism"),
            "biological_id": run.get("biological_id"),
            "technical_view": run.get("technical_view"),
            "dataset_id": dataset_id,
            "dataset_sha256": run.get("source_dataset_sha256"),
            "method": method,
            "method_sha256": method_bindings[method],
            "model_seed": run.get("model_seed"),
        }
        for name in _SELECTION_RECONSTRUCTION_METRICS:
            if name == "mse_pre_dropout_zero" and run.get("mechanism") != "symsim":
                continue
            source = metric_lookup.get(name)
            if source is None:
                raise DevelopmentEvaluationError(
                    f"runner metric denominator lacks {name}"
                )
            status = _selection_status(source.get("status"))
            value = source.get("value") if status == "completed" else None
            records.append({**common, "metric": name, "value": value, "status": status})

        run_status = _selection_status(run.get("status"))
        entropy_sha256 = _null_de_entropy_sha256(evidence, run)
        design = null_designs.get(dataset_id)
        if design is None:
            evaluator = prepared.evaluator_dataset
            if "group" not in evaluator.obs:
                design_reason = "evaluator_group_labels_unavailable"
                fixed_mask = None
                split_sha256 = _null_de_sentinel(
                    "split-unavailable", entropy_sha256, design_reason
                )
                gene_mask_sha256 = _null_de_sentinel(
                    "gene-mask-unavailable", entropy_sha256, design_reason
                )
            else:
                strata = tuple(evaluator.obs["group"].astype(str))
                try:
                    observed = count_equivalent_to_log2_cp10k(
                        prepared.method_input.counts
                    )
                    fixed_mask, gene_mask_sha256 = fixed_null_de_gene_mask(
                        observed,
                        prepared.method_input.obs_ids,
                        strata,
                        entropy_sha256=entropy_sha256,
                    )
                    _assignment, split_sha256 = balanced_null_split(
                        prepared.method_input.obs_ids,
                        strata,
                        entropy_sha256=entropy_sha256,
                    )
                    design_reason = None
                except ValueError as error:
                    design_reason = _null_de_unavailable_reason(error)
                    fixed_mask = None
                    split_sha256 = _null_de_sentinel(
                        "split-unavailable", entropy_sha256, design_reason
                    )
                    gene_mask_sha256 = _null_de_sentinel(
                        "gene-mask-unavailable", entropy_sha256, design_reason
                    )
            design = (
                entropy_sha256,
                fixed_mask,
                split_sha256,
                gene_mask_sha256,
                design_reason,
            )
            null_designs[dataset_id] = design
        elif design[0] != entropy_sha256:
            raise DevelopmentEvaluationError(
                "one dataset maps to inconsistent null-DE biological authority"
            )
        (
            entropy_sha256,
            fixed_mask,
            split_sha256,
            gene_mask_sha256,
            design_reason,
        ) = design
        null_result: NullDEResult
        if run_status != "completed":
            null_result = NullDEResult(
                status=run_status,
                fpr=None,
                nominal_alpha=NULL_DE_ALPHA,
                n_tested_genes=(0 if fixed_mask is None else int(fixed_mask.sum())),
                split_sha256=split_sha256,
                gene_mask_sha256=gene_mask_sha256,
                reason=str(run.get("reason") or "method_run_not_completed"),
            )
        elif design_reason is not None or fixed_mask is None:
            null_result = NullDEResult(
                status="unavailable",
                fpr=None,
                nominal_alpha=NULL_DE_ALPHA,
                n_tested_genes=0,
                split_sha256=split_sha256,
                gene_mask_sha256=gene_mask_sha256,
                reason=design_reason or "null_de_design_unavailable",
            )
        else:
            output = _read_evaluator_output(checkpoint_directory, run)
            if output.shape != prepared.method_input.shape:
                raise DevelopmentEvaluationError(
                    f"evaluator output shape mismatches {dataset_id}"
                )
            null_result = evaluate_null_de_fpr(
                output,
                prepared.method_input.obs_ids,
                tuple(prepared.evaluator_dataset.obs["group"].astype(str)),
                fixed_gene_mask=fixed_mask,
                entropy_sha256=entropy_sha256,
            )
        records.append(
            {
                **common,
                "metric": "null_de_fpr",
                "value": null_result.fpr,
                "status": null_result.status,
            }
        )
        audits.append(
            {
                "run_id": run.get("run_id"),
                "dataset_id": dataset_id,
                "method": method,
                "model_seed": run.get("model_seed"),
                "status": null_result.status,
                "value": null_result.fpr,
                "nominal_alpha": null_result.nominal_alpha,
                "n_tested_genes": null_result.n_tested_genes,
                "fixed_gene_count": (
                    0 if fixed_mask is None else int(fixed_mask.sum())
                ),
                "split_entropy_sha256": entropy_sha256,
                "split_entropy_derivation": (
                    "sha256(completed_checkpoint_sha256,mechanism,biological_id)"
                ),
                "split_sha256": null_result.split_sha256,
                "gene_mask_sha256": null_result.gene_mask_sha256,
                "reason": null_result.reason,
                "evaluator_output_file_sha256": run.get("evaluator_output_file_sha256"),
            }
        )
    records.sort(
        key=lambda row: (
            str(row["method"]),
            str(row["metric"]),
            str(row["mechanism"]),
            str(row["biological_id"]),
            str(row["technical_view"]),
            -1 if row["model_seed"] is None else int(row["model_seed"]),
        )
    )
    return ReconstructionSelectionBundle(tuple(records), tuple(audits))


def _canonical_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must contain nonempty strings")
    return value


_PROTEIN_TO_RNA = {
    "CD3": "CD3D",
    "CD4": "CD4",
    "CD8": "CD8A",
    "CD45RA": "PTPRC",
    "CD56": "NCAM1",
    "CD16": "FCGR3A",
    "CD10": "MME",
    "CD11C": "ITGAX",
    "CD14": "CD14",
    "CD19": "CD19",
    "CD34": "CD34",
    "CCR5": "CCR5",
    "CCR7": "CCR7",
}


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    from scipy.stats import spearmanr

    if left.size != right.size or left.size < 3:
        return None
    if np.ptp(left) == 0.0 or np.ptp(right) == 0.0:
        return None
    value = float(spearmanr(left, right).statistic)
    return value if math.isfinite(value) else None


def rna_protein_concordance_units(
    rna_output: object,
    gene_ids: Sequence[str],
    protein_counts: object,
    protein_ids: Sequence[str],
    *,
    cell_ids: Sequence[str] | None = None,
) -> tuple[EndpointUnit, ...]:
    """Compute descriptive matched-marker correlations across cells in one specimen."""

    rna = np.asarray(rna_output, dtype=np.float64)
    protein = np.asarray(protein_counts, dtype=np.float64)
    genes = tuple(_canonical_text(value, "gene_ids") for value in gene_ids)
    proteins = tuple(_canonical_text(value, "protein_ids") for value in protein_ids)
    if (
        rna.ndim != 2
        or protein.ndim != 2
        or rna.shape[0] != protein.shape[0]
        or rna.shape[1] != len(genes)
        or protein.shape[1] != len(proteins)
        or not np.isfinite(rna).all()
        or not np.isfinite(protein).all()
    ):
        raise ValueError("RNA/protein matrices and identifiers are inconsistent")
    cells = (
        tuple(f"cell-{index:08d}" for index in range(rna.shape[0]))
        if cell_ids is None
        else tuple(_canonical_text(value, "cell_ids") for value in cell_ids)
    )
    if len(cells) != rna.shape[0] or len(set(cells)) != len(cells):
        raise ValueError("RNA/protein cell IDs must be complete and unique")
    matched = _matched_rna_protein_indices(genes, proteins)
    correlations = _rna_protein_marker_correlations(
        rna, protein, matched, np.arange(rna.shape[0])
    )
    units: list[EndpointUnit] = []
    for match, value in zip(matched, correlations, strict=True):
        if not math.isfinite(float(value)):
            continue
        gene_index, protein_index = match
        identity = f"{genes[gene_index]}:{proteins[protein_index]}"
        units.append(
            EndpointUnit(
                unit_id=identity,
                biological_id="cbmc-single-specimen",
                technical_id=identity,
                value=float(value),
            )
        )
    return tuple(units)


def _matched_rna_protein_indices(
    gene_ids: Sequence[str], protein_ids: Sequence[str]
) -> tuple[tuple[int, int], ...]:
    canonical_genes: dict[str, int] = {}
    for index, gene in enumerate(gene_ids):
        symbol = gene.removeprefix("HUMAN_").upper()
        if symbol in _PROTEIN_TO_RNA.values() and symbol in canonical_genes:
            raise ValueError("RNA gene symbols are duplicated after prefix removal")
        if symbol in _PROTEIN_TO_RNA.values():
            canonical_genes[symbol] = index
    matched: list[tuple[int, int]] = []
    for protein_index, raw_protein in enumerate(protein_ids):
        protein_id = raw_protein.upper()
        gene_symbol = _PROTEIN_TO_RNA.get(protein_id)
        gene_index = canonical_genes.get(gene_symbol or "")
        if gene_symbol is None or gene_index is None:
            continue
        matched.append((gene_index, protein_index))
    return tuple(matched)


def _rna_protein_marker_correlations(
    rna: np.ndarray,
    protein: np.ndarray,
    matched: Sequence[tuple[int, int]],
    rows: np.ndarray,
) -> np.ndarray:
    values = np.full(len(matched), np.nan, dtype=np.float64)
    for index, (gene_index, protein_index) in enumerate(matched):
        value = _spearman(rna[rows, gene_index], protein[rows, protein_index])
        if value is not None:
            values[index] = value
    return values


def _bootstrap_sha256(values: Sequence[float]) -> str:
    return hashlib.sha256(
        np.asarray(tuple(values), dtype="<f8").tobytes(order="C")
    ).hexdigest()


def hierarchical_endpoint_interval(
    *,
    configuration: str,
    endpoint: str,
    candidate_units: Sequence[EndpointUnit],
    observed_units: Sequence[EndpointUnit],
    n_boot: int = 10_000,
    seed: int = BOOTSTRAP_SEED,
) -> EndpointInterval:
    """Bootstrap paired technical units within resampled biological units."""

    configuration = _canonical_text(configuration, "configuration")
    endpoint = _canonical_text(endpoint, "endpoint")
    if type(n_boot) is not int or n_boot <= 0:
        raise ValueError("n_boot must be a positive integer")
    if type(seed) is not int or isinstance(seed, bool) or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    candidate = tuple(candidate_units)
    observed = tuple(observed_units)
    if any(not isinstance(value, EndpointUnit) for value in (*candidate, *observed)):
        raise TypeError("endpoint inputs must contain EndpointUnit values")
    candidate_lookup = {
        (value.unit_id, value.biological_id, value.technical_id): value.value
        for value in candidate
    }
    observed_lookup = {
        (value.unit_id, value.biological_id, value.technical_id): value.value
        for value in observed
    }
    if len(candidate_lookup) != len(candidate) or len(observed_lookup) != len(observed):
        raise ValueError("endpoint unit identities must be unique")
    if set(candidate_lookup) != set(observed_lookup):
        return EndpointInterval(
            configuration,
            endpoint,
            "observed",
            None,
            None,
            None,
            "unavailable",
            "candidate_observed_units_not_paired",
            0,
            0,
            n_boot,
            _bootstrap_sha256(()),
        )
    by_biological: dict[str, list[float]] = {}
    for identity in sorted(candidate_lookup):
        difference = candidate_lookup[identity] - observed_lookup[identity]
        by_biological.setdefault(identity[1], []).append(difference)
    if len(by_biological) < 2:
        return EndpointInterval(
            configuration,
            endpoint,
            "observed",
            None,
            None,
            None,
            "unavailable",
            "fewer_than_two_biological_units",
            len(by_biological),
            len(candidate_lookup),
            n_boot,
            _bootstrap_sha256(()),
        )
    biological_ids = sorted(by_biological)
    collapsed = [float(np.mean(by_biological[value])) for value in biological_ids]
    estimate = float(np.median(collapsed))
    rng = np.random.default_rng(seed)
    distribution: list[float] = []
    for _ in range(n_boot):
        sampled_effects: list[float] = []
        for biological_index in rng.integers(
            0, len(biological_ids), size=len(biological_ids)
        ):
            technical = by_biological[biological_ids[int(biological_index)]]
            sampled = rng.integers(0, len(technical), size=len(technical))
            sampled_effects.append(
                float(np.mean([technical[int(index)] for index in sampled]))
            )
        distribution.append(float(np.median(sampled_effects)))
    lower, upper = np.quantile(np.asarray(distribution), (0.025, 0.975))
    return EndpointInterval(
        configuration=configuration,
        endpoint=endpoint,
        comparison="observed",
        estimate=estimate,
        ci_lower=float(lower),
        ci_upper=float(upper),
        status="completed",
        reason=None,
        n_biological_units=len(by_biological),
        n_technical_units=len(candidate_lookup),
        n_boot=n_boot,
        bootstrap_sha256=_bootstrap_sha256(distribution),
    )


def _profile_concordance(
    candidate: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
) -> float | None:
    # The evaluator-owned reference alone fixes the denominator.  A method
    # cannot improve its score by zeroing or otherwise removing features.
    selected = mask & np.isfinite(reference) & (reference > 0.0)
    selected &= np.isfinite(candidate)
    return _spearman(candidate[selected], reference[selected])


def tung_concordance_units(
    output_log2_cp10k: object,
    source: TungSource,
) -> Mapping[str, tuple[EndpointUnit, ...]]:
    """Compute concordance from fixed-library-weighted count-equivalent profiles."""

    if not isinstance(source, TungSource):
        raise TypeError("source must be a TungSource")
    output = np.asarray(output_log2_cp10k, dtype=np.float64)
    if (
        output.shape != source.counts.shape
        or output.ndim != 2
        or not np.isfinite(output).all()
        or bool((output < 0.0).any())
    ):
        raise ValueError("Tung method output must match the prepared source")
    linear = np.exp2(output) - 1.0
    if not np.isfinite(linear).all():
        raise ValueError("Tung method output cannot be converted from log2 CP10k")
    observed_library_sizes = np.asarray(source.counts.sum(axis=1), dtype=np.float64)
    if bool((observed_library_sizes <= 0.0).any()):
        raise DevelopmentEvaluationError(
            "Tung observed cells must have positive fixed library sizes"
        )
    count_equivalent = linear / 10_000.0 * observed_library_sizes[:, None]
    ercc_units: list[EndpointUnit] = []
    bulk_units: list[EndpointUnit] = []
    lane_units: list[EndpointUnit] = []
    sample_array = np.asarray(source.sample_ids)
    individual_by_sample: dict[str, str] = {}
    for sample, individual in zip(
        source.sample_ids, source.individual_ids, strict=True
    ):
        previous = individual_by_sample.setdefault(sample, individual)
        if previous != individual:
            raise DevelopmentEvaluationError(
                "Tung sample maps to multiple biological individuals"
            )
    endogenous = ~np.asarray(source.ercc_mask, dtype=bool)
    ercc = np.asarray(source.ercc_mask, dtype=bool)
    for sample in sorted(set(source.sample_ids)):
        rows = sample_array == sample
        # Fixed observed library weights preserve aggregate count scale while
        # preventing a candidate from changing its own pseudobulk weights.
        candidate_profile = np.sum(count_equivalent[rows], axis=0)
        bulk_profile = np.asarray(source.bulk_profiles[sample], dtype=np.float64)
        if bulk_profile.shape != candidate_profile.shape:
            raise DevelopmentEvaluationError(
                "Tung bulk profile shape mismatches output"
            )
        individual = individual_by_sample[sample]
        ercc_value = _profile_concordance(candidate_profile, bulk_profile, ercc)
        if ercc_value is not None:
            ercc_units.append(EndpointUnit(sample, individual, sample, ercc_value))
        bulk_value = _profile_concordance(candidate_profile, bulk_profile, endogenous)
        if bulk_value is not None:
            bulk_units.append(EndpointUnit(sample, individual, sample, bulk_value))
        prefix = f"{sample}:"
        for lane_key in sorted(
            key for key in source.lane_profiles if key.startswith(prefix)
        ):
            lane_profile = np.asarray(source.lane_profiles[lane_key], dtype=np.float64)
            if lane_profile.shape != candidate_profile.shape:
                raise DevelopmentEvaluationError(
                    "Tung lane profile shape mismatches output"
                )
            lane_value = _profile_concordance(
                candidate_profile, lane_profile, endogenous
            )
            if lane_value is not None:
                lane_units.append(
                    EndpointUnit(lane_key, individual, lane_key, lane_value)
                )
    return MappingProxyType(
        {
            "ercc_recovery": tuple(ercc_units),
            "technical_replicate_concordance": tuple(lane_units),
            "bulk_pseudobulk_concordance": tuple(bulk_units),
        }
    )


def balanced_null_split(
    cell_ids: Sequence[str],
    strata: Sequence[str],
    *,
    entropy_sha256: str,
) -> tuple[np.ndarray, str]:
    """Create a balanced split from evaluator-private, post-execution entropy."""

    identifiers = tuple(_canonical_text(value, "cell_ids") for value in cell_ids)
    groups = tuple(_canonical_text(value, "strata") for value in strata)
    if not identifiers or len(identifiers) != len(groups):
        raise ValueError("cell_ids and strata must have the same nonzero length")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("cell_ids must be unique")
    if not isinstance(entropy_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", entropy_sha256
    ):
        raise ValueError("entropy_sha256 must be a lowercase SHA-256 digest")
    by_stratum: dict[str, list[int]] = {}
    for index, stratum in enumerate(groups):
        by_stratum.setdefault(stratum, []).append(index)
    if any(len(indices) < 4 for indices in by_stratum.values()):
        raise ValueError("every null-DE stratum requires at least four cells")

    selected = np.zeros(len(identifiers), dtype=np.bool_)
    for stratum in sorted(by_stratum):
        ordered = sorted(
            by_stratum[stratum],
            key=lambda index: (
                hashlib.sha256(
                    (
                        "maskimpute-null-de-private-assignment-v2\0"
                        f"{entropy_sha256}\0{identifiers[index]}"
                    ).encode()
                ).digest(),
                identifiers[index],
            ),
        )
        for rank, index in enumerate(ordered):
            selected[index] = rank % 2 == 0
    digest = hashlib.sha256()
    digest.update(b"maskimpute-null-de-balanced-split-v2\0")
    digest.update(entropy_sha256.encode("ascii"))
    for cell_id, stratum, assignment in zip(
        identifiers, groups, selected.tolist(), strict=True
    ):
        digest.update(b"\0")
        digest.update(cell_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(stratum.encode("utf-8"))
        digest.update(b"\0")
        digest.update(b"1" if assignment else b"0")
    return selected, digest.hexdigest()


def _null_de_design(
    n_rows: int,
    cell_ids: Sequence[str],
    strata: Sequence[str],
    *,
    entropy_sha256: str,
) -> tuple[np.ndarray, str, np.ndarray, np.ndarray, int, float]:
    assignment, split_sha256 = balanced_null_split(
        cell_ids, strata, entropy_sha256=entropy_sha256
    )
    if n_rows != len(cell_ids):
        raise ValueError("method output rows must match cell_ids")
    stratum_values = tuple(strata)
    levels = sorted(set(stratum_values))
    design_columns = [np.ones(n_rows), assignment.astype(np.float64)]
    design_columns.extend(
        np.asarray([value == level for value in stratum_values], dtype=np.float64)
        for level in levels[1:]
    )
    design = np.column_stack(design_columns)
    rank = int(np.linalg.matrix_rank(design))
    degrees_of_freedom = n_rows - rank
    inverse = np.linalg.pinv(design.T @ design)
    coefficient_variance_factor = float(inverse[1, 1])
    return (
        assignment,
        split_sha256,
        design,
        inverse,
        degrees_of_freedom,
        coefficient_variance_factor,
    )


def _null_de_standard_errors(
    output: np.ndarray,
    design: np.ndarray,
    inverse: np.ndarray,
    degrees_of_freedom: int,
    coefficient_variance_factor: float,
) -> tuple[np.ndarray, np.ndarray]:
    coefficients = inverse @ design.T @ output
    residual = output - design @ coefficients
    residual_variance = np.sum(residual * residual, axis=0) / degrees_of_freedom
    standard_error = np.sqrt(residual_variance * coefficient_variance_factor)
    return coefficients, standard_error


def fixed_null_de_gene_mask(
    observed_log2_cp10k: object,
    cell_ids: Sequence[str],
    strata: Sequence[str],
    *,
    entropy_sha256: str,
) -> tuple[np.ndarray, str]:
    """Freeze the null-DE denominator from observed counts, never method output."""

    observed = np.asarray(observed_log2_cp10k, dtype=np.float64)
    if observed.ndim != 2 or not np.isfinite(observed).all():
        raise ValueError("observed output must be a finite two-dimensional matrix")
    (
        _assignment,
        _split_sha256,
        design,
        inverse,
        degrees_of_freedom,
        coefficient_variance_factor,
    ) = _null_de_design(
        observed.shape[0],
        cell_ids,
        strata,
        entropy_sha256=entropy_sha256,
    )
    if degrees_of_freedom <= 0:
        raise ValueError("null-DE design has no residual degrees of freedom")
    _coefficients, standard_error = _null_de_standard_errors(
        observed,
        design,
        inverse,
        degrees_of_freedom,
        coefficient_variance_factor,
    )
    mask = np.isfinite(standard_error) & (standard_error > 0.0)
    if int(mask.sum()) < NULL_DE_MIN_GENES:
        raise ValueError(
            f"null-DE observed denominator has fewer than {NULL_DE_MIN_GENES} genes"
        )
    mask.setflags(write=False)
    digest = hashlib.sha256()
    digest.update(b"maskimpute-null-de-fixed-gene-mask-v1\0")
    digest.update(entropy_sha256.encode("ascii"))
    digest.update(b"\0")
    digest.update(str(observed.shape[1]).encode("ascii"))
    digest.update(b"\0")
    digest.update(np.packbits(mask, bitorder="little").tobytes())
    return mask, digest.hexdigest()


def evaluate_null_de_fpr(
    method_output_log2_cp10k: object,
    cell_ids: Sequence[str],
    strata: Sequence[str],
    *,
    fixed_gene_mask: object,
    entropy_sha256: str,
    nominal_alpha: float = NULL_DE_ALPHA,
) -> NullDEResult:
    """Estimate gene-level null-DE FPR with stratum-adjusted two-sided OLS tests."""

    from scipy.stats import t as student_t

    output = np.asarray(method_output_log2_cp10k, dtype=np.float64)
    if output.ndim != 2 or not np.isfinite(output).all():
        raise ValueError("method output must be a finite two-dimensional matrix")
    mask = np.asarray(fixed_gene_mask)
    if (
        mask.dtype != np.bool_
        or mask.ndim != 1
        or mask.shape[0] != output.shape[1]
        or int(mask.sum()) < NULL_DE_MIN_GENES
    ):
        raise ValueError(
            "fixed_gene_mask must be boolean, gene-aligned, and meet the fixed minimum"
        )
    if (
        isinstance(nominal_alpha, bool)
        or not isinstance(nominal_alpha, (int, float))
        or not math.isfinite(float(nominal_alpha))
        or not 0.0 < float(nominal_alpha) < 1.0
    ):
        raise ValueError("nominal_alpha must lie strictly between zero and one")
    (
        _assignment,
        split_sha256,
        design,
        inverse,
        degrees_of_freedom,
        coefficient_variance_factor,
    ) = _null_de_design(
        output.shape[0],
        cell_ids,
        strata,
        entropy_sha256=entropy_sha256,
    )
    mask_digest = hashlib.sha256()
    mask_digest.update(b"maskimpute-null-de-fixed-gene-mask-v1\0")
    mask_digest.update(entropy_sha256.encode("ascii"))
    mask_digest.update(b"\0")
    mask_digest.update(str(output.shape[1]).encode("ascii"))
    mask_digest.update(b"\0")
    mask_digest.update(np.packbits(mask, bitorder="little").tobytes())
    gene_mask_sha256 = mask_digest.hexdigest()
    if degrees_of_freedom <= 0:
        return NullDEResult(
            status="unavailable",
            fpr=None,
            nominal_alpha=float(nominal_alpha),
            n_tested_genes=0,
            split_sha256=split_sha256,
            gene_mask_sha256=gene_mask_sha256,
            reason="insufficient_residual_degrees_of_freedom",
        )
    coefficients, standard_error = _null_de_standard_errors(
        output,
        design,
        inverse,
        degrees_of_freedom,
        coefficient_variance_factor,
    )
    n_tested = int(mask.sum())
    if not bool(
        (np.isfinite(standard_error[mask]) & (standard_error[mask] > 0.0)).all()
    ):
        return NullDEResult(
            status="unavailable",
            fpr=None,
            nominal_alpha=float(nominal_alpha),
            n_tested_genes=n_tested,
            split_sha256=split_sha256,
            gene_mask_sha256=gene_mask_sha256,
            reason="method_non_testable_on_fixed_gene_denominator",
        )
    statistic = np.abs(coefficients[1, mask] / standard_error[mask])
    p_values = 2.0 * student_t.sf(statistic, degrees_of_freedom)
    fpr = float(np.mean(p_values < float(nominal_alpha)))
    return NullDEResult(
        status="completed",
        fpr=fpr,
        nominal_alpha=float(nominal_alpha),
        n_tested_genes=n_tested,
        split_sha256=split_sha256,
        gene_mask_sha256=gene_mask_sha256,
    )


__all__ = [
    "BaronSource",
    "BOOTSTRAP_SEED",
    "CITE_METHOD_GENE_COUNT",
    "CiteSeqSource",
    "DevelopmentEvaluationError",
    "EndpointInterval",
    "EndpointUnit",
    "NULL_DE_ALPHA",
    "NULL_DE_MIN_GENES",
    "NullDEResult",
    "OrthogonalConfiguration",
    "OrthogonalExecutionRequest",
    "OrthogonalInput",
    "OrthogonalOutputEvidence",
    "OrthogonalSelectionBundle",
    "PreparedRealOrthogonalPanel",
    "PublicMaskImputeOrthogonalExecutor",
    "RawArtifactBinding",
    "RealSourceEvidence",
    "ReconstructionEvidence",
    "ReconstructionSelectionBundle",
    "SourceArtifactBinding",
    "SourceReceiptBinding",
    "TungSource",
    "balanced_null_split",
    "build_development_selection_input",
    "build_reconstruction_selection_records",
    "evaluate_null_de_fpr",
    "evaluate_cite_orthogonal_interval",
    "evaluate_real_orthogonal_intervals",
    "evaluate_tung_orthogonal_intervals",
    "hierarchical_endpoint_interval",
    "fixed_null_de_gene_mask",
    "load_completed_reconstruction_checkpoint",
    "load_orthogonal_output_evidence",
    "prepare_baron_source",
    "prepare_cite_seq_source",
    "prepare_tung_source",
    "prepare_real_orthogonal_panel",
    "produce_orthogonal_outputs",
    "rna_protein_concordance_units",
    "run_real_orthogonal_outputs",
    "tung_concordance_units",
    "validate_real_source_artifacts",
    "write_development_selection_artifacts",
]
