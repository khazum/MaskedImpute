#!/usr/bin/env python3
"""Run one paired SERGIO simulation from a pristine pinned checkout."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import stat
import sys
from typing import Any

import networkx as nx
import numpy as np
import scipy


_SERGIO_COMMIT = "a6190b74425112834c8fa9b4b6157d9cb3d1ab88"
_SERGIO_TREE = "15558fe60f62683c6fa46bcde01d9f3d3382e34a"
_COMPATIBILITY_SHIM = {
    "numpy_removed_aliases": {
        "np.float": "builtins.float",
        "np.int": "builtins.int",
    }
}
_VIEW_PARAMETERS: dict[str, dict[str, int | float]] = {
    "moderate": {
        "outlier_prob": 0.01,
        "outlier_mean": 0.8,
        "outlier_scale": 1.0,
        "library_log_mean": 5.2,
        "library_log_sd": 0.3,
        "dropout_shape": 6.5,
        "dropout_percentile": 65,
    },
    "severe": {
        "outlier_prob": 0.01,
        "outlier_mean": 0.8,
        "outlier_scale": 1.0,
        "library_log_mean": 4.6,
        "library_log_sd": 0.4,
        "dropout_shape": 6.5,
        "dropout_percentile": 82,
    },
}
_PROFILES = {
    "De-noised_100G_9T_300cPerT_4_DS1": {
        "simulated_genes": 100,
        "maximum_requested_genes": 100,
        "interaction_path": (
            "data_sets/De-noised_100G_9T_300cPerT_4_DS1/Interaction_cID_4.txt"
        ),
        "regulator_path": ("data_sets/De-noised_100G_9T_300cPerT_4_DS1/Regs_cID_4.txt"),
    },
    "De-noised_1200G_9T_300cPerT_6_DS3": {
        "simulated_genes": 1200,
        "maximum_requested_genes": 1200,
        "interaction_path": (
            "data_sets/De-noised_1200G_9T_300cPerT_6_DS3/Interaction_cID_6.txt"
        ),
        "regulator_path": (
            "data_sets/De-noised_1200G_9T_300cPerT_6_DS3/Regs_cID_6.txt"
        ),
    },
}
_ARRAY_NAMES = (
    "clean.npy",
    "dropout_indicator_moderate.npy",
    "dropout_indicator_severe.npy",
    "observed_moderate.npy",
    "observed_severe.npy",
    "pre_dropout_moderate.npy",
    "pre_dropout_severe.npy",
)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _load_config(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("SERGIO config is not strict UTF-8 JSON") from error
    if not isinstance(value, dict) or _canonical_json_bytes(value) != data:
        raise ValueError("SERGIO config is not canonical JSON")
    return value


def _plain_int(value: object, name: str, *, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} is outside its integer domain")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_path(path: Path, root: Path, name: str) -> Path:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ValueError(f"{name} cannot be inspected") from error
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
    ):
        raise ValueError(f"{name} must be a unique non-symlink regular file")
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{name} escapes the explicit SERGIO checkout") from error
    if resolved != path.absolute():
        raise ValueError(f"{name} must use a canonical path without symlinks")
    return resolved


def _validate_config(
    config: dict[str, object], checkout: Path, output_dir: Path
) -> dict[str, object]:
    if (
        set(config)
        != {
            "adapter",
            "profile",
            "schema_version",
            "seeds",
            "simulation",
            "source",
            "views",
        }
        or config.get("schema_version") != 1
    ):
        raise ValueError("SERGIO config has the wrong closed schema")
    adapter = config.get("adapter")
    profile = config.get("profile")
    seeds = config.get("seeds")
    simulation = config.get("simulation")
    source = config.get("source")
    views = config.get("views")
    if not all(
        isinstance(value, dict)
        for value in (adapter, profile, seeds, simulation, source)
    ) or not isinstance(views, list):
        raise ValueError("SERGIO config sections have invalid types")
    assert isinstance(adapter, dict)
    assert isinstance(profile, dict)
    assert isinstance(seeds, dict)
    assert isinstance(simulation, dict)
    assert isinstance(source, dict)

    if (
        set(adapter)
        != {
            "compatibility_shim",
            "python_adapter_sha256",
            "python_runner_sha256",
        }
        or adapter.get("compatibility_shim") != _COMPATIBILITY_SHIM
    ):
        raise ValueError("SERGIO adapter or compatibility-shim binding is invalid")
    for key in ("python_adapter_sha256", "python_runner_sha256"):
        value = adapter.get(key)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"SERGIO {key} is invalid")
    if adapter["python_runner_sha256"] != _sha256_file(Path(__file__)):
        raise ValueError("SERGIO runner bytes do not match the adapter config")

    checkout = checkout.resolve(strict=True)
    logical_module_path = "SERGIO/sergio.py"
    module_path = checkout / logical_module_path
    expected_source = {
        "commit": _SERGIO_COMMIT,
        "module_path": logical_module_path,
        "tree": _SERGIO_TREE,
    }
    if source != expected_source:
        raise ValueError("SERGIO config does not bind the explicit pinned module path")
    _regular_path(module_path, checkout, "SERGIO module")

    profile_name = profile.get("name")
    if not isinstance(profile_name, str) or profile_name not in _PROFILES:
        raise ValueError("SERGIO config names an unsupported input profile")
    profile_spec = _PROFILES[profile_name]
    expected_profile = {
        "name": profile_name,
        "interaction_path": profile_spec["interaction_path"],
        "regulator_path": profile_spec["regulator_path"],
    }
    if profile != expected_profile:
        raise ValueError("SERGIO config changes the pinned input profile")
    interaction = _regular_path(
        checkout / str(profile_spec["interaction_path"]),
        checkout,
        "SERGIO interaction input",
    )
    regulators = _regular_path(
        checkout / str(profile_spec["regulator_path"]),
        checkout,
        "SERGIO regulator input",
    )

    cells = _plain_int(simulation.get("cells"), "cells", minimum=18, maximum=2**31 - 1)
    requested_genes = _plain_int(
        simulation.get("requested_genes"),
        "requested_genes",
        minimum=1,
        maximum=1200,
    )
    simulated_genes = _plain_int(
        simulation.get("simulated_genes"),
        "simulated_genes",
        minimum=100,
        maximum=1200,
    )
    if cells % 9 != 0:
        raise ValueError("SERGIO cells must be divisible by nine")
    expected_simulation = {
        "cells": cells,
        "cell_types": 9,
        "cells_per_type": cells // 9,
        "decays": 0.8,
        "noise_params": 1.0,
        "noise_type": "dpd",
        "requested_genes": requested_genes,
        "sampling_state": 15,
        "shared_coop_state": 2.0,
        "simulated_genes": simulated_genes,
    }
    if simulation != expected_simulation:
        raise ValueError("SERGIO simulation parameters are not the fixed study design")
    if (
        simulated_genes != profile_spec["simulated_genes"]
        or requested_genes > int(profile_spec["maximum_requested_genes"])
        or (requested_genes <= 100) != (simulated_genes == 100)
    ):
        raise ValueError("SERGIO profile does not match the requested gene count")

    if set(seeds) != {"biological"} or not isinstance(seeds.get("biological"), dict):
        raise ValueError("SERGIO biological seed config is invalid")
    biological = seeds["biological"]
    assert isinstance(biological, dict)
    if set(biological) != {"original", "mapped_numpy"}:
        raise ValueError("SERGIO biological seed binding is invalid")
    _plain_int(
        biological.get("original"),
        "original biological seed",
        minimum=0,
        maximum=2**63 - 1,
    )
    biological_mapped = _plain_int(
        biological.get("mapped_numpy"),
        "mapped biological seed",
        minimum=1,
        maximum=2**32 - 1,
    )

    if len(views) != 2:
        raise ValueError("SERGIO config requires exactly two views")
    observed_views: dict[str, dict[str, object]] = {}
    mapped_seeds = {biological_mapped}
    for value in views:
        if not isinstance(value, dict):
            raise ValueError("SERGIO view config must be an object")
        name = value.get("technical_view")
        if (
            not isinstance(name, str)
            or name not in _VIEW_PARAMETERS
            or name in observed_views
        ):
            raise ValueError("SERGIO view names must be moderate then severe")
        expected_keys = {
            "technical_view",
            "measurement_seed_original",
            "measurement_seed_numpy",
            *_VIEW_PARAMETERS[name],
        }
        if set(value) != expected_keys or any(
            value.get(key) != expected
            for key, expected in _VIEW_PARAMETERS[name].items()
        ):
            raise ValueError("SERGIO technical parameters are not fixed")
        _plain_int(
            value.get("measurement_seed_original"),
            f"{name} original measurement seed",
            minimum=0,
            maximum=2**63 - 1,
        )
        mapped = _plain_int(
            value.get("measurement_seed_numpy"),
            f"{name} mapped measurement seed",
            minimum=1,
            maximum=2**32 - 1,
        )
        if mapped in mapped_seeds:
            raise ValueError("SERGIO mapped RNG seeds must be distinct")
        mapped_seeds.add(mapped)
        observed_views[name] = value
    if [value.get("technical_view") for value in views] != ["moderate", "severe"]:
        raise ValueError("SERGIO views must use canonical order")

    if output_dir.is_symlink() or not output_dir.is_dir():
        raise ValueError("SERGIO output directory must be a regular directory")
    if {path.name for path in output_dir.iterdir()} != {"config.json"}:
        raise ValueError("SERGIO output directory was not empty apart from config")
    return {
        "biological_seed": biological_mapped,
        "cells": cells,
        "cells_per_type": cells // 9,
        "checkout": checkout,
        "interaction": interaction,
        "logical_module_path": logical_module_path,
        "module_path": module_path,
        "output_dir": output_dir,
        "regulators": regulators,
        "requested_genes": requested_genes,
        "simulated_genes": simulated_genes,
        "views": observed_views,
    }


def _apply_numpy_compatibility_shim() -> None:
    if "int" not in np.__dict__:
        setattr(np, "int", int)
    if "float" not in np.__dict__:
        setattr(np, "float", float)


def _import_sergio(checkout: Path, module_path: Path) -> type[Any]:
    sys.dont_write_bytecode = True
    _apply_numpy_compatibility_shim()
    if any(name == "SERGIO" or name.startswith("SERGIO.") for name in sys.modules):
        raise RuntimeError("SERGIO was imported before explicit-path verification")
    sys.path.insert(0, checkout.as_posix())
    package = importlib.import_module("SERGIO")
    module = importlib.import_module("SERGIO.sergio")
    observed_module = Path(module.__file__).resolve(strict=True)
    package_file = Path(package.__file__).resolve(strict=True)
    if (
        observed_module != module_path
        or package_file != checkout / "SERGIO/__init__.py"
    ):
        raise RuntimeError(
            "imported SERGIO module does not match the explicit checkout"
        )
    for name, imported in tuple(sys.modules.items()):
        if name != "SERGIO" and not name.startswith("SERGIO."):
            continue
        imported_file = getattr(imported, "__file__", None)
        if not isinstance(imported_file, str):
            raise RuntimeError("imported SERGIO namespace module has no source path")
        try:
            _regular_path(
                Path(imported_file), checkout, f"imported SERGIO module {name}"
            ).relative_to(checkout)
        except ValueError as error:
            raise RuntimeError(
                "imported SERGIO module escaped the explicit checkout"
            ) from error
    simulator = getattr(module, "sergio", None)
    if not isinstance(simulator, type):
        raise RuntimeError("explicit SERGIO module does not expose its simulator class")
    return simulator


def _validate_expression(
    value: object, shape: tuple[int, int, int], name: str
) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.shape != shape
        or array.dtype.kind not in {"f", "i", "u"}
        or not np.isfinite(array).all()
        or bool((array < 0).any())
    ):
        raise RuntimeError(f"SERGIO {name} has invalid shape, dtype, or range")
    return np.ascontiguousarray(array, dtype="<f8")


def _gene_cell(array: np.ndarray, requested_genes: int) -> np.ndarray:
    selected = array[:, :requested_genes, :]
    return np.ascontiguousarray(np.concatenate(selected, axis=1), dtype="<f8")


def _write_npy(path: Path, values: np.ndarray) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o644,
    )
    try:
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as output:
                np.save(output, values, allow_pickle=False)
                output.flush()
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def _write_json(path: Path, value: object) -> None:
    data = _canonical_json_bytes(value)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o644,
    )
    try:
        try:
            remaining = memoryview(data)
            while remaining:
                written = os.write(descriptor, remaining)
                remaining = remaining[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def _run(config: dict[str, object], binding: dict[str, object]) -> None:
    checkout = binding["checkout"]
    module_path = binding["module_path"]
    logical_module_path = binding["logical_module_path"]
    output_dir = binding["output_dir"]
    if not all(
        isinstance(value, Path) for value in (checkout, module_path, output_dir)
    ) or not isinstance(logical_module_path, str):
        raise RuntimeError("SERGIO runner path binding is invalid")
    assert isinstance(checkout, Path)
    assert isinstance(module_path, Path)
    assert isinstance(output_dir, Path)
    simulator_class = _import_sergio(checkout, module_path)
    simulated_genes = int(binding["simulated_genes"])
    requested_genes = int(binding["requested_genes"])
    cells = int(binding["cells"])
    cells_per_type = int(binding["cells_per_type"])
    biological_seed = int(binding["biological_seed"])
    interaction = binding["interaction"]
    regulators = binding["regulators"]
    views = binding["views"]
    if (
        not isinstance(interaction, Path)
        or not isinstance(regulators, Path)
        or not isinstance(views, dict)
    ):
        raise RuntimeError("SERGIO runner config binding is invalid")

    created: list[Path] = []
    try:
        np.random.seed(biological_seed)
        simulator = simulator_class(
            number_genes=simulated_genes,
            number_bins=9,
            number_sc=cells_per_type,
            noise_params=1.0,
            noise_type="dpd",
            decays=0.8,
            dynamics=False,
            sampling_state=15,
        )
        simulator.build_graph(
            interaction.as_posix(),
            regulators.as_posix(),
            shared_coop_state=2.0,
        )
        simulator.simulate()
        full_shape = (9, simulated_genes, cells_per_type)
        clean_full = _validate_expression(
            simulator.getExpressions(), full_shape, "clean expression"
        )
        clean = _gene_cell(clean_full, requested_genes)
        pre_dropout: dict[str, np.ndarray] = {}
        indicators: dict[str, np.ndarray] = {}
        observed: dict[str, np.ndarray] = {}
        for name in ("moderate", "severe"):
            view = views[name]
            if not isinstance(view, dict):
                raise RuntimeError("SERGIO technical view binding is invalid")
            np.random.seed(int(view["measurement_seed_numpy"]))
            outliers = simulator.outlier_effect(
                clean_full.copy(),
                outlier_prob=float(view["outlier_prob"]),
                mean=float(view["outlier_mean"]),
                scale=float(view["outlier_scale"]),
            )
            _library_factors, before_dropout = simulator.lib_size_effect(
                outliers,
                mean=float(view["library_log_mean"]),
                scale=float(view["library_log_sd"]),
            )
            before_dropout_full = _validate_expression(
                before_dropout, full_shape, f"{name} pre-dropout expression"
            )
            indicator_full = np.asarray(
                simulator.dropout_indicator(
                    before_dropout_full,
                    shape=float(view["dropout_shape"]),
                    percentile=int(view["dropout_percentile"]),
                )
            )
            if (
                indicator_full.shape != full_shape
                or indicator_full.dtype.kind not in {"b", "i", "u", "f"}
                or not np.isfinite(indicator_full).all()
                or not bool(np.isin(indicator_full, (0, 1)).all())
            ):
                raise RuntimeError(f"SERGIO {name} dropout indicator is invalid")
            indicator_full = np.ascontiguousarray(indicator_full, dtype=np.uint8)
            counts_full = np.asarray(
                simulator.convert_to_UMIcounts(
                    np.multiply(indicator_full, before_dropout_full)
                )
            )
            if (
                counts_full.shape != full_shape
                or counts_full.dtype.kind not in {"i", "u"}
                or bool((counts_full < 0).any())
                or bool((counts_full > np.iinfo(np.int64).max).any())
            ):
                raise RuntimeError(f"SERGIO {name} UMI counts are invalid")
            pre_dropout[name] = _gene_cell(before_dropout_full, requested_genes)
            indicators[name] = np.ascontiguousarray(
                np.concatenate(indicator_full[:, :requested_genes, :], axis=1),
                dtype=np.uint8,
            )
            observed[name] = np.ascontiguousarray(
                np.concatenate(counts_full[:, :requested_genes, :], axis=1),
                dtype="<i8",
            )
        arrays = {
            "clean.npy": np.ascontiguousarray(clean, dtype="<f8"),
            "pre_dropout_moderate.npy": np.ascontiguousarray(
                pre_dropout["moderate"], dtype="<f8"
            ),
            "pre_dropout_severe.npy": np.ascontiguousarray(
                pre_dropout["severe"], dtype="<f8"
            ),
            "dropout_indicator_moderate.npy": indicators["moderate"],
            "dropout_indicator_severe.npy": indicators["severe"],
            "observed_moderate.npy": observed["moderate"],
            "observed_severe.npy": observed["severe"],
        }
        expected_shape = (requested_genes, cells)
        if any(array.shape != expected_shape for array in arrays.values()):
            raise RuntimeError("SERGIO selected arrays have the wrong orientation")
        for name in _ARRAY_NAMES:
            path = output_dir / name
            _write_npy(path, arrays[name])
            created.append(path)

        config_views = config["views"]
        assert isinstance(config_views, list)
        call_per_view = {
            name: {
                "outlier_effect": 1,
                "lib_size_effect": 1,
                "dropout_indicator": 1,
                "convert_to_umi_counts": 1,
            }
            for name in ("moderate", "severe")
        }
        metadata = {
            "schema_version": 1,
            "array_sha256": {
                name: _sha256_file(output_dir / name) for name in _ARRAY_NAMES
            },
            "biological_seed_numpy": biological_seed,
            "call_counts": {
                "sergio_constructor": 1,
                "build_graph": 1,
                "simulate": 1,
                "get_expressions": 1,
                "outlier_effect": 2,
                "lib_size_effect": 2,
                "dropout_indicator": 2,
                "convert_to_umi_counts": 2,
                "per_view": call_per_view,
            },
            "cells": cells,
            "cell_types": 9,
            "compatibility_shim": _COMPATIBILITY_SHIM,
            "measurement_seeds_numpy": {
                name: int(views[name]["measurement_seed_numpy"])
                for name in ("moderate", "severe")
            },
            "module_path": logical_module_path,
            "requested_genes": requested_genes,
            "simulated_genes": simulated_genes,
            "versions": {
                "networkx": nx.__version__,
                "numpy": np.__version__,
                "python": platform.python_version(),
                "scipy": scipy.__version__,
                "sergio": "1.0.0",
            },
            "views": [view["technical_view"] for view in config_views],
        }
        metadata_path = output_dir / "run_metadata.json"
        _write_json(metadata_path, metadata)
        created.append(metadata_path)
        directory_fd = os.open(output_dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        for path in reversed(created):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise


def main(arguments: list[str]) -> int:
    if len(arguments) != 3:
        raise SystemExit(
            "usage: run_sergio.py CONFIG_JSON EXPLICIT_CHECKOUT OUTPUT_DIRECTORY"
        )
    config_path = Path(arguments[0]).absolute()
    checkout = Path(arguments[1]).absolute()
    output_dir = Path(arguments[2]).absolute()
    if (
        config_path.is_symlink()
        or not config_path.is_file()
        or config_path.parent != output_dir
    ):
        raise ValueError("SERGIO config must be the regular config.json in output")
    config_metadata = config_path.lstat()
    if not stat.S_ISREG(config_metadata.st_mode) or config_metadata.st_nlink != 1:
        raise ValueError("SERGIO config must be a unique regular file")
    config = _load_config(config_path)
    binding = _validate_config(config, checkout, output_dir)
    _run(config, binding)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
