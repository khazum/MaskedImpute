"""Receipt-bound frozen-final null differential-expression safety evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from types import MappingProxyType

import anndata as ad
import numpy as np
from scipy import sparse

from .development_evaluation import (
    NULL_DE_ALPHA,
    balanced_null_split,
    evaluate_null_de_fpr,
    fixed_null_de_gene_mask,
)
from .downstream_evaluation import MethodOutput
from .downstream_evidence import (
    DatasetEvidenceBinding,
    DownstreamEvidencePlan,
    DownstreamPlanEntry,
)
from .methods import count_equivalent_to_log2_cp10k
from .protocol import canonical_sha256


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
FINAL_NULL_DE_ALGORITHM = "final-null-de-v1"
_STAGING_FILE = re.compile(
    r"\.(?:plan\.json|final_null_de_manifest\.json|[0-9]{8}\.json)\."
    r"[A-Za-z0-9_-]{6,}\.tmp\Z"
)


class FinalNullDEError(ValueError):
    """Raised when final null-DE evidence is incomplete or changed."""


class _FinalNullDEUnavailable(ValueError):
    """Expected dataset-level mathematical unavailability."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class FinalNullDEPlan:
    """Complete evaluated-round source and downstream-evidence binding."""

    source_plan: DownstreamEvidencePlan
    downstream_directory: str
    downstream_manifest_file_sha256: str
    downstream_manifest_payload_sha256: str
    evaluator_source_sha256: str
    plan_sha256: str

    def body(self) -> dict[str, object]:
        evaluated = self.source_plan.evaluated_round_binding
        if evaluated is None:
            raise FinalNullDEError("evaluated final receipt binding is absent")
        return {
            "schema_version": 1,
            "algorithm": FINAL_NULL_DE_ALGORITHM,
            "repository_root": evaluated.repository_root,
            "round_root": evaluated.round_root,
            "downstream_directory": self.downstream_directory,
            "downstream_manifest_file_sha256": (self.downstream_manifest_file_sha256),
            "downstream_manifest_payload_sha256": (
                self.downstream_manifest_payload_sha256
            ),
            "evaluator_source_sha256": self.evaluator_source_sha256,
            "source_plan": self.source_plan.to_dict(),
            "planned_denominator_count": len(self.source_plan.entries),
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.body(), "plan_sha256": self.plan_sha256}


@dataclass(frozen=True, slots=True)
class FinalNullDEManifest:
    """Fully replayed final null-DE archive."""

    plan_sha256: str
    manifest_sha256: str
    manifest_file_sha256: str
    planned_denominator_count: int
    records: tuple[Mapping[str, object], ...]
    payload: Mapping[str, object]


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise FinalNullDEError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FinalNullDEError(f"{name} must be a nonempty string")
    return value


def final_null_de_entropy_sha256(
    *,
    receipt_payload_sha256: str,
    dataset_id: str,
    dataset_sha256: str,
    mechanism: str,
    biological_id: str,
    technical_view: str,
    retained_cell_ids: Sequence[str],
) -> str:
    """Derive one method-independent split entropy per frozen final dataset."""

    if isinstance(retained_cell_ids, (str, bytes)) or not isinstance(
        retained_cell_ids, Sequence
    ):
        raise TypeError("retained_cell_ids must be a sequence of strings")
    cells = tuple(retained_cell_ids)
    if (
        not cells
        or len(cells) != len(set(cells))
        or any(not isinstance(value, str) or not value for value in cells)
    ):
        raise FinalNullDEError("retained_cell_ids must be unique nonempty strings")
    stable_cell_set_sha256 = canonical_sha256(sorted(cells))

    return canonical_sha256(
        {
            "algorithm": FINAL_NULL_DE_ALGORITHM,
            "evaluation_receipt_payload_sha256": _digest(
                receipt_payload_sha256, "evaluation receipt payload"
            ),
            "dataset_id": _text(dataset_id, "dataset_id"),
            "dataset_sha256": _digest(dataset_sha256, "dataset semantic checksum"),
            "mechanism": _text(mechanism, "mechanism"),
            "biological_id": _text(biological_id, "biological_id"),
            "technical_view": _text(technical_view, "technical_view"),
            "retained_cell_count": len(cells),
            "retained_cell_set_sha256": stable_cell_set_sha256,
        }
    )


def _load_bound_dataset(binding: DatasetEvidenceBinding) -> ad.AnnData:
    """Use the downstream evaluator's byte and semantic source validator."""

    from .downstream_evidence import _read_bound_dataset

    return _read_bound_dataset(binding)


def _decode_bound_output(
    plan: DownstreamEvidencePlan,
    entry: DownstreamPlanEntry,
    binding: DatasetEvidenceBinding,
) -> MethodOutput:
    """Use the downstream evaluator's bounded final-output decoder."""

    from .downstream_evidence import _decode_output

    return _decode_output(plan, entry, binding)


def _aligned_dataset_values(
    dataset: ad.AnnData, binding: DatasetEvidenceBinding
) -> tuple[np.ndarray, tuple[str, ...]]:
    if not isinstance(dataset, ad.AnnData):
        raise FinalNullDEError("bound evaluator dataset is not AnnData")
    observed_ids = tuple(dataset.obs_names.astype(str))
    positions = {value: index for index, value in enumerate(observed_ids)}
    if any(value not in positions for value in binding.retained_cell_ids):
        raise FinalNullDEError("retained final cell is absent from evaluator dataset")
    indices = np.asarray(
        [positions[value] for value in binding.retained_cell_ids], dtype=np.int64
    )
    if "group" not in dataset.obs:
        raise _FinalNullDEUnavailable("evaluator_group_labels_unavailable")
    raw_groups = tuple(dataset.obs["group"].iloc[indices].tolist())
    if any(not isinstance(value, str) or not value.strip() for value in raw_groups):
        raise _FinalNullDEUnavailable("evaluator_group_labels_unavailable")
    groups = tuple(raw_groups)
    values = dataset.X[indices, :]
    if sparse.issparse(values):
        values = values.toarray()
    counts = np.asarray(values)
    if counts.shape != (len(binding.retained_cell_ids), len(binding.gene_ids)):
        raise FinalNullDEError("observed final matrix shape differs")
    return count_equivalent_to_log2_cp10k(counts), groups


def _dataset_context(
    dataset: ad.AnnData,
    binding: DatasetEvidenceBinding,
    receipt_payload_sha256: str,
) -> Mapping[str, object]:
    entropy_sha256 = final_null_de_entropy_sha256(
        receipt_payload_sha256=receipt_payload_sha256,
        dataset_id=binding.dataset_id,
        dataset_sha256=binding.dataset_sha256,
        mechanism=binding.mechanism,
        biological_id=binding.biological_id,
        technical_view=binding.technical_view,
        retained_cell_ids=binding.retained_cell_ids,
    )
    try:
        observed, groups = _aligned_dataset_values(dataset, binding)
    except _FinalNullDEUnavailable as error:
        return MappingProxyType(
            {
                "entropy_sha256": entropy_sha256,
                "split_sha256": None,
                "gene_mask": None,
                "gene_mask_sha256": None,
                "groups": None,
                "availability_reason": error.reason,
            }
        )
    try:
        _assignment, split_sha256 = balanced_null_split(
            binding.retained_cell_ids,
            groups,
            entropy_sha256=entropy_sha256,
        )
    except (TypeError, ValueError):
        return MappingProxyType(
            {
                "entropy_sha256": entropy_sha256,
                "split_sha256": None,
                "gene_mask": None,
                "gene_mask_sha256": None,
                "groups": groups,
                "availability_reason": "balanced_split_unavailable",
            }
        )
    try:
        gene_mask, gene_mask_sha256 = fixed_null_de_gene_mask(
            observed,
            binding.retained_cell_ids,
            groups,
            entropy_sha256=entropy_sha256,
        )
    except (TypeError, ValueError):
        return MappingProxyType(
            {
                "entropy_sha256": entropy_sha256,
                "split_sha256": split_sha256,
                "gene_mask": None,
                "gene_mask_sha256": None,
                "groups": groups,
                "availability_reason": ("fixed_observed_gene_denominator_unavailable"),
            }
        )
    return MappingProxyType(
        {
            "entropy_sha256": entropy_sha256,
            "split_sha256": split_sha256,
            "gene_mask": gene_mask,
            "gene_mask_sha256": gene_mask_sha256,
            "groups": groups,
            "availability_reason": None,
        }
    )


def _record_common(
    plan: DownstreamEvidencePlan,
    entry: DownstreamPlanEntry,
    binding: DatasetEvidenceBinding,
    context: Mapping[str, object],
) -> dict[str, object]:
    evaluated = plan.evaluated_round_binding
    if evaluated is None:
        raise FinalNullDEError("evaluated final receipt binding is absent")
    return {
        "schema_version": 1,
        "algorithm": FINAL_NULL_DE_ALGORITHM,
        "ordinal": entry.ordinal,
        "source_plan_sha256": plan.plan_sha256,
        "evaluated_round_binding_sha256": evaluated.binding_sha256,
        "evaluation_receipt_payload_sha256": (
            evaluated.evaluation_receipt_payload_sha256
        ),
        "source_record_path": entry.source_record_path,
        "source_record_sha256": entry.source_record_sha256,
        "run_id": entry.run_id,
        "method_id": entry.method_id,
        "dataset_id": entry.dataset_id,
        "dataset_sha256": entry.source_dataset_sha256,
        "mechanism": entry.mechanism,
        "biological_id": entry.biological_id,
        "technical_view": entry.technical_view,
        "model_seed": entry.model_seed,
        "configuration_id": entry.configuration_id,
        "configuration_sha256": entry.configuration_sha256,
        "method_artifact_sha256": entry.method_artifact_sha256,
        "method_input_sha256": entry.method_input_sha256,
        "retained_cell_ids_sha256": binding.retained_cell_ids_sha256,
        "upstream_status": entry.status,
        "upstream_reason": entry.reason,
        "entropy_sha256": context["entropy_sha256"],
        "split_sha256": context["split_sha256"],
        "gene_mask_sha256": context["gene_mask_sha256"],
        "nominal_alpha": NULL_DE_ALPHA,
    }


def _seal_record(body: Mapping[str, object]) -> dict[str, object]:
    result = dict(body)
    return {**result, "record_sha256": canonical_sha256(result)}


def evaluate_final_null_de_records(
    plan: DownstreamEvidencePlan,
) -> tuple[Mapping[str, object], ...]:
    """Re-evaluate one complete null-DE row per primary final source run.

    The accepted plan is the evaluator-only, receipt-bound final downstream
    source plan. Method outputs and evaluator datasets are loaded only through
    its byte/semantic validators; no caller-supplied matrices are accepted.
    """

    if not isinstance(plan, DownstreamEvidencePlan):
        raise TypeError("plan must be a DownstreamEvidencePlan")
    if (
        plan.source_kind != "final"
        or plan.evidence_scope != "all"
        or plan.evaluated_round_binding is None
    ):
        raise FinalNullDEError("null-DE requires the complete evaluated final source")
    bindings = {value.dataset_id: value for value in plan.datasets}
    if len(bindings) != len(plan.datasets) or not bindings:
        raise FinalNullDEError("final dataset denominator is invalid")
    if not plan.entries or any(
        entry.dataset_id not in bindings for entry in plan.entries
    ):
        raise FinalNullDEError("final run denominator is invalid")

    contexts: dict[str, Mapping[str, object]] = {}
    records: list[Mapping[str, object]] = []
    receipt_sha256 = plan.evaluated_round_binding.evaluation_receipt_payload_sha256
    for expected_ordinal, entry in enumerate(plan.entries, start=1):
        if type(entry.ordinal) is not int or entry.ordinal != expected_ordinal:
            raise FinalNullDEError("final run denominator is not ordered")
        binding = bindings[entry.dataset_id]
        context = contexts.get(entry.dataset_id)
        if context is None:
            dataset = _load_bound_dataset(binding)
            context = _dataset_context(dataset, binding, receipt_sha256)
            contexts[entry.dataset_id] = context
        common = _record_common(plan, entry, binding, context)
        if entry.status != "completed":
            records.append(
                MappingProxyType(
                    _seal_record(
                        {
                            **common,
                            "status": entry.status,
                            "reason_code": "upstream_run_not_completed",
                            "fpr": None,
                            "n_tested_genes": 0,
                        }
                    )
                )
            )
            continue

        availability_reason = context["availability_reason"]
        if availability_reason is not None:
            records.append(
                MappingProxyType(
                    _seal_record(
                        {
                            **common,
                            "status": "unavailable",
                            "reason_code": availability_reason,
                            "fpr": None,
                            "n_tested_genes": 0,
                        }
                    )
                )
            )
            continue

        output = _decode_bound_output(plan, entry, binding)
        if (
            output.cell_ids != binding.retained_cell_ids
            or output.gene_ids != binding.gene_ids
        ):
            raise FinalNullDEError("decoded final output identities differ")
        result = evaluate_null_de_fpr(
            output.values,
            binding.retained_cell_ids,
            context["groups"],
            fixed_gene_mask=context["gene_mask"],
            entropy_sha256=str(context["entropy_sha256"]),
            nominal_alpha=NULL_DE_ALPHA,
        )
        if (
            result.split_sha256 != context["split_sha256"]
            or result.gene_mask_sha256 != context["gene_mask_sha256"]
        ):
            raise FinalNullDEError("null-DE split or gene denominator changed")
        records.append(
            MappingProxyType(
                _seal_record(
                    {
                        **common,
                        "status": result.status,
                        "reason_code": result.reason,
                        "fpr": result.fpr,
                        "n_tested_genes": result.n_tested_genes,
                    }
                )
            )
        )
    return tuple(records)


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise FinalNullDEError("final null-DE value is not canonical JSON") from error


def _evaluator_source_sha256() -> str:
    """Bind this stage and every reused statistical/source-decoding routine."""

    from .downstream_evidence import DownstreamEvidenceError, _stable_file_bytes

    package = Path(__file__).absolute().parent
    digest = hashlib.sha256(b"maskimpute-final-null-de-evaluator-source-v1\0")
    for filename in (
        "final_null_de.py",
        "development_evaluation.py",
        "downstream_evidence.py",
        "downstream_evaluation.py",
        "methods/observed.py",
    ):
        path = package / filename
        try:
            raw, _file_sha256 = _stable_file_bytes(
                path,
                f"final null-DE evaluator source {filename}",
                max_bytes=32 * 1024 * 1024,
            )
        except (DownstreamEvidenceError, OSError) as error:
            raise FinalNullDEError(
                f"final null-DE evaluator source {filename} is unavailable"
            ) from error
        encoded = filename.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


def _create_final_null_de_plan(
    source_plan: DownstreamEvidencePlan,
    *,
    downstream_directory: str,
    downstream_manifest_file_sha256: str,
    downstream_manifest_payload_sha256: str,
) -> FinalNullDEPlan:
    if not isinstance(source_plan, DownstreamEvidencePlan):
        raise TypeError("source_plan must be a DownstreamEvidencePlan")
    if (
        source_plan.source_kind != "final"
        or source_plan.evidence_scope != "all"
        or source_plan.evaluated_round_binding is None
        or not source_plan.entries
    ):
        raise FinalNullDEError(
            "null-DE source is not the complete evaluated final plan"
        )
    directory = Path(_text(downstream_directory, "downstream directory")).absolute()
    provisional = FinalNullDEPlan(
        source_plan=source_plan,
        downstream_directory=str(directory),
        downstream_manifest_file_sha256=_digest(
            downstream_manifest_file_sha256,
            "downstream manifest file checksum",
        ),
        downstream_manifest_payload_sha256=_digest(
            downstream_manifest_payload_sha256,
            "downstream manifest payload checksum",
        ),
        evaluator_source_sha256=_evaluator_source_sha256(),
        plan_sha256="0" * 64,
    )
    return FinalNullDEPlan(
        source_plan=provisional.source_plan,
        downstream_directory=provisional.downstream_directory,
        downstream_manifest_file_sha256=(provisional.downstream_manifest_file_sha256),
        downstream_manifest_payload_sha256=(
            provisional.downstream_manifest_payload_sha256
        ),
        evaluator_source_sha256=provisional.evaluator_source_sha256,
        plan_sha256=canonical_sha256(provisional.body()),
    )


def _expected_downstream_directory(source_plan: DownstreamEvidencePlan) -> Path:
    from .downstream_evidence import (
        DownstreamEvidenceError,
        expected_final_downstream_output_directory,
    )

    try:
        return expected_final_downstream_output_directory(source_plan)
    except (DownstreamEvidenceError, TypeError, ValueError) as error:
        raise FinalNullDEError("evaluated final receipt binding is absent") from error


def expected_final_null_de_output_directory(plan: FinalNullDEPlan) -> Path:
    """Return the sole receipt-namespaced production archive location."""

    if not isinstance(plan, FinalNullDEPlan):
        raise TypeError("plan must be a FinalNullDEPlan")
    evaluated = plan.source_plan.evaluated_round_binding
    if evaluated is None:
        raise FinalNullDEError("evaluated final receipt binding is absent")
    repository = Path(evaluated.repository_root)
    return (
        repository.parent
        / f"{repository.name}-final-analysis"
        / "null-de"
        / evaluated.round_id
        / evaluated.evaluation_receipt_payload_sha256
    ).absolute()


def build_final_null_de_plan(
    repository: str | Path,
    round_directory: str | Path,
) -> FinalNullDEPlan:
    """Build the production plan from one evaluated round and its downstream archive."""

    from .downstream_evidence import (
        DownstreamEvidenceError,
        build_final_downstream_evidence_plan,
        load_downstream_evidence_manifest,
        load_downstream_evidence_plan,
    )

    try:
        source_plan = build_final_downstream_evidence_plan(repository, round_directory)
        downstream_directory = _expected_downstream_directory(source_plan)
        persisted_source_plan = load_downstream_evidence_plan(downstream_directory)
        downstream_manifest = load_downstream_evidence_manifest(downstream_directory)
        if persisted_source_plan.to_dict() != source_plan.to_dict():
            raise FinalNullDEError(
                "final downstream plan differs from the evaluated source"
            )
        if (
            downstream_manifest.plan_sha256 != source_plan.plan_sha256
            or type(downstream_manifest.planned_denominator_count) is not int
            or downstream_manifest.planned_denominator_count != len(source_plan.entries)
        ):
            raise FinalNullDEError(
                "final downstream denominator differs from the evaluated source"
            )
        _raw, manifest_file_sha256 = _unique_file_bytes(
            downstream_directory / "downstream_manifest.json",
            "final downstream manifest",
        )
    except FinalNullDEError:
        raise
    except (DownstreamEvidenceError, OSError, TypeError, ValueError) as error:
        raise FinalNullDEError(
            "validated final downstream evidence is unavailable"
        ) from error
    return _create_final_null_de_plan(
        source_plan,
        downstream_directory=str(downstream_directory),
        downstream_manifest_file_sha256=manifest_file_sha256,
        downstream_manifest_payload_sha256=downstream_manifest.manifest_sha256,
    )


def _rebuild_plan(value: FinalNullDEPlan | Mapping[str, object]) -> FinalNullDEPlan:
    if isinstance(value, FinalNullDEPlan):
        evaluated = value.source_plan.evaluated_round_binding
        if evaluated is None:
            raise FinalNullDEError("evaluated final receipt binding is absent")
        repository = evaluated.repository_root
        round_root = evaluated.round_root
    elif isinstance(value, Mapping):
        source_plan = value.get("source_plan")
        evaluated_payload = (
            source_plan.get("evaluated_round_binding")
            if isinstance(source_plan, Mapping)
            else None
        )
        if not isinstance(evaluated_payload, Mapping):
            raise FinalNullDEError("persisted evaluated final binding is absent")
        repository = _text(
            evaluated_payload.get("repository_root"), "persisted repository root"
        )
        round_root = _text(evaluated_payload.get("round_root"), "persisted round root")
    else:
        raise TypeError("plan value is invalid")
    return build_final_null_de_plan(repository, round_root)


def _revalidate_plan(plan: FinalNullDEPlan) -> None:
    if not isinstance(plan, FinalNullDEPlan):
        raise TypeError("plan must be a FinalNullDEPlan")
    if (
        plan.plan_sha256 != canonical_sha256(plan.body())
        or plan.evaluator_source_sha256 != _evaluator_source_sha256()
    ):
        raise FinalNullDEError("final null-DE plan checksum differs")
    rebuilt = _rebuild_plan(plan)
    if rebuilt.to_dict() != plan.to_dict():
        raise FinalNullDEError("final null-DE plan sources changed")


def _wrap_downstream_io(action, *args):
    from .downstream_evidence import DownstreamEvidenceError

    try:
        return action(*args)
    except DownstreamEvidenceError as error:
        raise FinalNullDEError(str(error)) from error


def _validate_output_location(plan: FinalNullDEPlan, output_root: Path) -> None:
    from .downstream_evidence import _reject_symlink_chain

    _wrap_downstream_io(_reject_symlink_chain, output_root, "final null-DE output")
    evaluated = plan.source_plan.evaluated_round_binding
    if evaluated is None:
        raise FinalNullDEError("evaluated final receipt binding is absent")
    repository = Path(evaluated.repository_root).absolute()
    try:
        output_root.relative_to(repository)
    except ValueError:
        pass
    else:
        raise FinalNullDEError(
            "final null-DE output must remain outside the frozen repository"
        )
    if output_root != expected_final_null_de_output_directory(plan):
        raise FinalNullDEError(
            "final null-DE output differs from its evaluated-receipt namespace"
        )


@contextmanager
def _archive_lock(output_root: Path, *, exclusive: bool):
    """Hold one stable directory-inode lock across archive inspection/publication."""

    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    try:
        descriptor = os.open(output_root, flags)
        opened_before = os.fstat(descriptor)
        if not stat.S_ISDIR(opened_before.st_mode):
            raise FinalNullDEError("final null-DE output lock is not a directory")
        fcntl.flock(
            descriptor,
            fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH,
        )
        opened_after = os.fstat(descriptor)
        named = output_root.lstat()
        if (
            not stat.S_ISDIR(named.st_mode)
            or stat.S_ISLNK(named.st_mode)
            or (opened_before.st_dev, opened_before.st_ino)
            != (opened_after.st_dev, opened_after.st_ino)
            or (opened_before.st_dev, opened_before.st_ino)
            != (named.st_dev, named.st_ino)
        ):
            raise FinalNullDEError(
                "final null-DE output directory changed while locking"
            )
        yield
        opened_final = os.fstat(descriptor)
        named_final = output_root.lstat()
        if (
            (opened_before.st_dev, opened_before.st_ino)
            != (opened_final.st_dev, opened_final.st_ino)
            or (opened_before.st_dev, opened_before.st_ino)
            != (named_final.st_dev, named_final.st_ino)
            or not stat.S_ISDIR(named_final.st_mode)
            or stat.S_ISLNK(named_final.st_mode)
        ):
            raise FinalNullDEError(
                "final null-DE output directory changed while locked"
            )
    except FinalNullDEError:
        raise
    except OSError as error:
        raise FinalNullDEError("final null-DE output lock failed") from error
    finally:
        if descriptor >= 0:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)


def _recover_interrupted_staging(output_root: Path) -> None:
    """Remove only recognized uncommitted hard-link publication temporaries."""

    for directory in (output_root, output_root / "records"):
        if not os.path.lexists(directory):
            continue
        try:
            metadata = directory.lstat()
            children = tuple(directory.iterdir())
        except OSError as error:
            raise FinalNullDEError(
                "final null-DE staging directory is unavailable"
            ) from error
        if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            raise FinalNullDEError("final null-DE staging directory is invalid")
        changed = False
        for path in children:
            if _STAGING_FILE.fullmatch(path.name) is None:
                continue
            try:
                child = path.lstat()
            except OSError as error:
                raise FinalNullDEError(
                    "final null-DE staging file is unavailable"
                ) from error
            if not stat.S_ISREG(child.st_mode) or stat.S_ISLNK(child.st_mode):
                raise FinalNullDEError("final null-DE staging file is invalid")
            try:
                path.unlink()
            except OSError as error:
                raise FinalNullDEError(
                    "final null-DE interrupted staging cannot be recovered"
                ) from error
            changed = True
        if changed:
            descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)


def _validate_archive_layout(output_root: Path) -> None:
    try:
        names = {path.name for path in output_root.iterdir()}
    except OSError as error:
        raise FinalNullDEError("final null-DE archive layout is unavailable") from error
    allowed = {"plan.json", "records", "final_null_de_manifest.json"}
    if not names.issubset(allowed):
        raise FinalNullDEError("final null-DE archive layout contains extra files")


def _record_names(output_root: Path, planned: int) -> tuple[str, ...]:
    from .downstream_evidence import _reject_symlink_chain

    records_root = output_root / "records"
    if not os.path.lexists(records_root):
        return ()
    _wrap_downstream_io(_reject_symlink_chain, records_root, "final null-DE records")
    if not records_root.is_dir():
        raise FinalNullDEError("final null-DE record directory is invalid")
    names = tuple(sorted(path.name for path in records_root.iterdir()))
    expected = tuple(f"{index:08d}.json" for index in range(1, len(names) + 1))
    if names != expected or len(names) > planned:
        raise FinalNullDEError("final null-DE records are not a canonical prefix")
    return names


def _file_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _unique_file_bytes(path: Path, name: str) -> tuple[bytes, str]:
    from .downstream_evidence import _stable_file_bytes

    try:
        before = path.lstat()
    except OSError as error:
        raise FinalNullDEError(f"{name} is unavailable") from error
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or before.st_nlink != 1
    ):
        raise FinalNullDEError(f"{name} must be a unique regular file")
    raw, file_sha256 = _wrap_downstream_io(_stable_file_bytes, path, name)
    try:
        after = path.lstat()
    except OSError as error:
        raise FinalNullDEError(f"{name} changed while being read") from error
    if _file_identity(before) != _file_identity(after):
        raise FinalNullDEError(f"{name} changed while being read")
    return raw, file_sha256


def _strict_json(path: Path, name: str) -> tuple[dict[str, object], bytes, str]:
    from .downstream_evidence import _strict_json as strict_json

    try:
        before = path.lstat()
    except OSError as error:
        raise FinalNullDEError(f"{name} is unavailable") from error
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or before.st_nlink != 1
    ):
        raise FinalNullDEError(f"{name} must be a unique regular file")
    value, raw, file_sha256 = _wrap_downstream_io(strict_json, path, name)
    try:
        after = path.lstat()
    except OSError as error:
        raise FinalNullDEError(f"{name} changed while being read") from error
    if _file_identity(before) != _file_identity(after):
        raise FinalNullDEError(f"{name} changed while being read")
    return value, raw, file_sha256


def _publish_immutable(path: Path, raw: bytes, name: str) -> str:
    from .downstream_evidence import _publish_immutable as publish

    return _wrap_downstream_io(publish, path, raw, name)


def _load_persisted_plan(output_root: Path) -> FinalNullDEPlan:
    payload, _raw, _file_sha256 = _strict_json(
        output_root / "plan.json", "final null-DE plan"
    )
    rebuilt = _rebuild_plan(payload)
    if rebuilt.to_dict() != payload:
        raise FinalNullDEError("persisted final null-DE plan differs")
    return rebuilt


def _expected_records(plan: FinalNullDEPlan) -> tuple[Mapping[str, object], ...]:
    return evaluate_final_null_de_records(plan.source_plan)


def _load_record_prefix(
    output_root: Path,
    plan: FinalNullDEPlan,
    expected_records: tuple[Mapping[str, object], ...],
) -> tuple[Mapping[str, object], ...]:
    names = _record_names(output_root, len(expected_records))
    records: list[Mapping[str, object]] = []
    for index, name in enumerate(names):
        record, raw, _file_sha = _strict_json(
            output_root / "records" / name,
            "final null-DE record",
        )
        if raw != _canonical_bytes(dict(expected_records[index])) + b"\n":
            raise FinalNullDEError("final null-DE record re-evaluation differs")
        records.append(MappingProxyType(record))
    return tuple(records)


def _manifest_payload(
    output_root: Path,
    plan: FinalNullDEPlan,
    records: tuple[Mapping[str, object], ...],
) -> dict[str, object]:
    if len(records) != len(plan.source_plan.entries):
        raise FinalNullDEError("final null-DE manifest denominator is incomplete")
    references: list[dict[str, object]] = []
    for ordinal, record in enumerate(records, start=1):
        path = output_root / "records" / f"{ordinal:08d}.json"
        raw, file_sha256 = _unique_file_bytes(path, "final null-DE record")
        if raw != _canonical_bytes(dict(record)) + b"\n":
            raise FinalNullDEError("final null-DE record re-evaluation differs")
        references.append(
            {
                "ordinal": ordinal,
                "run_id": record["run_id"],
                "path": f"records/{ordinal:08d}.json",
                "sha256": file_sha256,
                "record_sha256": record["record_sha256"],
            }
        )
    _plan_raw, plan_file_sha256 = _unique_file_bytes(
        output_root / "plan.json", "final null-DE plan"
    )
    body: dict[str, object] = {
        "schema_version": 1,
        "algorithm": FINAL_NULL_DE_ALGORITHM,
        "status": "completed",
        "plan_sha256": plan.plan_sha256,
        "plan_file_sha256": plan_file_sha256,
        "source_plan_sha256": plan.source_plan.plan_sha256,
        "evaluated_round_binding_sha256": (
            plan.source_plan.evaluated_round_binding.binding_sha256
            if plan.source_plan.evaluated_round_binding is not None
            else None
        ),
        "downstream_manifest_file_sha256": (plan.downstream_manifest_file_sha256),
        "downstream_manifest_payload_sha256": (plan.downstream_manifest_payload_sha256),
        "evaluator_source_sha256": plan.evaluator_source_sha256,
        "planned_denominator_count": len(plan.source_plan.entries),
        "recorded_denominator_count": len(records),
        "records": references,
    }
    return {**body, "manifest_sha256": canonical_sha256(body)}


def _run_final_null_de_evidence_locked(
    plan: FinalNullDEPlan,
    output_root: Path,
    max_denominators: int | None,
) -> dict[str, object]:
    _revalidate_plan(plan)
    _recover_interrupted_staging(output_root)
    _validate_archive_layout(output_root)
    manifest_path = output_root / "final_null_de_manifest.json"
    if os.path.lexists(manifest_path):
        persisted = _load_persisted_plan(output_root)
        if persisted.to_dict() != plan.to_dict():
            raise FinalNullDEError("completed final null-DE plan differs")
        return dict(_load_final_null_de_manifest_locked(output_root).payload)

    _publish_immutable(
        output_root / "plan.json",
        _canonical_bytes(plan.to_dict()) + b"\n",
        "final null-DE plan",
    )
    existing_names = _record_names(output_root, len(plan.source_plan.entries))
    if max_denominators == 0 and not existing_names:
        return {
            "schema_version": 1,
            "status": "running",
            "plan_sha256": plan.plan_sha256,
            "planned_denominator_count": len(plan.source_plan.entries),
            "recorded_denominator_count": 0,
        }
    expected = _expected_records(plan)
    records = list(_load_record_prefix(output_root, plan, expected))
    remaining = len(expected) - len(records)
    count = remaining if max_denominators is None else min(remaining, max_denominators)
    for record in expected[len(records) : len(records) + count]:
        ordinal = len(records) + 1
        _publish_immutable(
            output_root / "records" / f"{ordinal:08d}.json",
            _canonical_bytes(dict(record)) + b"\n",
            "final null-DE record",
        )
        records.append(record)
    if len(records) != len(expected):
        return {
            "schema_version": 1,
            "status": "running",
            "plan_sha256": plan.plan_sha256,
            "planned_denominator_count": len(expected),
            "recorded_denominator_count": len(records),
        }
    _revalidate_plan(plan)
    manifest = _manifest_payload(output_root, plan, tuple(records))
    _publish_immutable(
        manifest_path,
        _canonical_bytes(manifest) + b"\n",
        "final null-DE manifest",
    )
    return manifest


def run_final_null_de_evidence(
    plan: FinalNullDEPlan,
    output_directory: str | Path,
    *,
    max_denominators: int | None = None,
) -> dict[str, object]:
    """Resume an immutable prefix and publish a complete replayable archive."""

    if max_denominators is not None and (
        isinstance(max_denominators, bool)
        or type(max_denominators) is not int
        or max_denominators < 0
    ):
        raise ValueError("max_denominators must be a nonnegative integer or null")
    _revalidate_plan(plan)
    output_root = Path(output_directory).absolute()
    _validate_output_location(plan, output_root)
    from .downstream_evidence import _ensure_directory

    _wrap_downstream_io(
        _ensure_directory, output_root, "final null-DE output directory"
    )
    with _archive_lock(output_root, exclusive=True):
        return _run_final_null_de_evidence_locked(plan, output_root, max_denominators)


def _load_final_null_de_manifest_locked(
    output_root: Path,
) -> FinalNullDEManifest:
    _validate_archive_layout(output_root)
    plan = _load_persisted_plan(output_root)
    _validate_output_location(plan, output_root)
    expected = _expected_records(plan)
    records = _load_record_prefix(output_root, plan, expected)
    if len(records) != len(expected):
        raise FinalNullDEError("final null-DE manifest denominator is incomplete")
    manifest, _raw, manifest_file_sha256 = _strict_json(
        output_root / "final_null_de_manifest.json",
        "final null-DE manifest",
    )
    expected_manifest = _manifest_payload(output_root, plan, records)
    if manifest != expected_manifest:
        raise FinalNullDEError("final null-DE manifest completeness differs")
    return FinalNullDEManifest(
        plan_sha256=plan.plan_sha256,
        manifest_sha256=_digest(
            manifest.get("manifest_sha256"), "final null-DE manifest checksum"
        ),
        manifest_file_sha256=manifest_file_sha256,
        planned_denominator_count=len(expected),
        records=records,
        payload=MappingProxyType(manifest),
    )


def load_final_null_de_manifest(
    output_directory: str | Path,
) -> FinalNullDEManifest:
    """Rebuild all sources and recompute every completed final null-DE row."""

    output_root = Path(output_directory).absolute()
    from .downstream_evidence import _existing_directory

    _wrap_downstream_io(
        _existing_directory, output_root, "final null-DE output directory"
    )
    with _archive_lock(output_root, exclusive=False):
        return _load_final_null_de_manifest_locked(output_root)


__all__ = [
    "FINAL_NULL_DE_ALGORITHM",
    "FinalNullDEError",
    "FinalNullDEManifest",
    "FinalNullDEPlan",
    "build_final_null_de_plan",
    "evaluate_final_null_de_records",
    "expected_final_null_de_output_directory",
    "final_null_de_entropy_sha256",
    "load_final_null_de_manifest",
    "run_final_null_de_evidence",
]
