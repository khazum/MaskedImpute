from pathlib import Path
import json

import pytest

from maskimpute_benchmark.protocol import canonical_sha256, load_protocol


def test_canonical_hash_ignores_mapping_order():
    assert canonical_sha256({"b": 2, "a": 1}) == canonical_sha256({"a": 1, "b": 2})


def test_protocol_declares_four_non_splatter_mechanisms():
    protocol = load_protocol(Path("study/protocol.json"))
    assert protocol.mechanisms == ("symsim", "sergio", "sparsim", "semisynthetic")
    assert protocol.final_draws_per_condition == 5
    assert protocol.final_model_seeds == 3


def test_protocol_rejects_splatter_as_final(tmp_path):
    path = tmp_path / "protocol.json"
    path.write_text(
        json.dumps({"schema_version": 1, "final": {"mechanisms": ["splatter"]}})
    )
    with pytest.raises(ValueError, match="Splatter is development-only"):
        load_protocol(path)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_protocol_rejects_nonfinite_resource_limits(tmp_path, invalid):
    protocol = json.loads(Path("study/protocol.json").read_text(encoding="utf-8"))
    protocol["max_rss_gib"] = invalid
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")

    with pytest.raises(ValueError, match="finite"):
        load_protocol(path)


def test_protocol_rejects_nonfinite_constants_even_in_unknown_fields(tmp_path):
    text = Path("study/protocol.json").read_text(encoding="utf-8")
    text = text[:-2] + ',\n  "unvalidated_metadata": NaN\n}\n'
    path = tmp_path / "protocol.json"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="non-finite JSON constant"):
        load_protocol(path)


def test_protocol_rejects_duplicate_object_keys(tmp_path):
    path = tmp_path / "protocol.json"
    path.write_text(
        Path("study/protocol.json")
        .read_text(encoding="utf-8")
        .replace('"schema_version": 1,', '"schema_version": 1, "schema_version": 1,'),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_protocol(path)


@pytest.mark.parametrize("section", [None, "development", "final"])
def test_protocol_rejects_unknown_fields(tmp_path: Path, section: str | None) -> None:
    protocol = json.loads(Path("study/protocol.json").read_text(encoding="utf-8"))
    target = protocol if section is None else protocol[section]
    target["unexpected"] = "not-authoritative"
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")

    with pytest.raises(ValueError, match="fields"):
        load_protocol(path)


@pytest.mark.parametrize(
    ("section", "namespace"),
    [("development", "development"), ("final", "publication-final")],
)
def test_protocol_namespaces_match_the_execution_contract(
    tmp_path: Path, section: str, namespace: str
) -> None:
    protocol = json.loads(Path("study/protocol.json").read_text(encoding="utf-8"))
    protocol[section]["namespace"] = namespace
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")

    with pytest.raises(ValueError, match="namespace"):
        load_protocol(path)


def test_protocol_primary_metrics_are_unique(tmp_path: Path) -> None:
    protocol = json.loads(Path("study/protocol.json").read_text(encoding="utf-8"))
    protocol["primary_metrics"].append(protocol["primary_metrics"][0])
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")

    with pytest.raises(ValueError, match="unique"):
        load_protocol(path)
