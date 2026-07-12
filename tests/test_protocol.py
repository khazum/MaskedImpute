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
    path.write_text(json.dumps({"schema_version": 1, "final": {"mechanisms": ["splatter"]}}))
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
