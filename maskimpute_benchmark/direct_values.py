"""Lossless typed values shared by every direct-evidence boundary."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
import math


class FrozenDirectList(tuple[object, ...]):
    """A frozen JSON array, distinct from an object encoded as pairs."""


class FrozenDirectObject(tuple[tuple[str, object], ...]):
    """A frozen nested JSON object."""


def freeze_direct_value(value: object) -> object:
    """Freeze one canonical JSON value without losing container identity."""

    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("direct value keys must be strings")
        return FrozenDirectObject(
            (key, freeze_direct_value(nested)) for key, nested in sorted(value.items())
        )
    if isinstance(value, (list, tuple)):
        return FrozenDirectList(freeze_direct_value(nested) for nested in value)
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise ValueError("direct value is not canonical JSON")


def freeze_direct_mapping(
    value: Mapping[str, object],
) -> tuple[tuple[str, object], ...]:
    """Freeze a top-level payload mapping in canonical key order."""

    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError("direct value must be a string-keyed mapping")
    return tuple(
        (key, freeze_direct_value(nested)) for key, nested in sorted(value.items())
    )


def _thaw_direct_value(value: object) -> object:
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("direct value keys must be strings")
        return {key: _thaw_direct_value(nested) for key, nested in value.items()}
    if isinstance(value, FrozenDirectObject):
        return {item[0]: _thaw_direct_value(item[1]) for item in value}
    if isinstance(value, FrozenDirectList):
        return [_thaw_direct_value(item) for item in value]
    if isinstance(value, tuple):
        if all(
            isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)
            for item in value
        ):
            return {item[0]: _thaw_direct_value(item[1]) for item in value}
        return [_thaw_direct_value(item) for item in value]
    return value


def direct_json_value(value: object, *, payload: bool = False) -> object:
    """Encode one direct dataclass/value without losing JSON container types."""

    if payload:
        return _thaw_direct_value(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: direct_json_value(
                getattr(value, item.name),
                payload=item.name in {"payload", "configuration_payload"},
            )
            for item in fields(value)
        }
    if isinstance(value, tuple):
        return [direct_json_value(item) for item in value]
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("direct value keys must be strings")
        return {key: direct_json_value(nested) for key, nested in value.items()}
    return value


def direct_equal(left: object, right: object) -> bool:
    """Compare direct values recursively without primitive type coercion."""

    if is_dataclass(left) or is_dataclass(right):
        if (
            not is_dataclass(left)
            or not is_dataclass(right)
            or isinstance(left, type)
            or isinstance(right, type)
            or type(left) is not type(right)
        ):
            return False
        return all(
            direct_equal(getattr(left, item.name), getattr(right, item.name))
            for item in fields(left)
        )
    if isinstance(left, FrozenDirectObject) or isinstance(right, FrozenDirectObject):
        left_object = dict(left) if isinstance(left, FrozenDirectObject) else left
        right_object = dict(right) if isinstance(right, FrozenDirectObject) else right
        if not isinstance(left_object, Mapping) or not isinstance(
            right_object, Mapping
        ):
            return False
        return set(left_object) == set(right_object) and all(
            direct_equal(left_object[key], right_object[key]) for key in left_object
        )
    if isinstance(left, FrozenDirectList) or isinstance(right, FrozenDirectList):
        if not (isinstance(left, FrozenDirectList) or type(left) is list) or not (
            isinstance(right, FrozenDirectList) or type(right) is list
        ):
            return False
        return len(left) == len(right) and all(
            direct_equal(first, second)
            for first, second in zip(left, right, strict=True)
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return False
        return set(left) == set(right) and all(
            direct_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            return False
        return all(
            direct_equal(first, second)
            for first, second in zip(left, right, strict=True)
        )
    if type(left) is float and type(right) is float:
        return left.hex() == right.hex()
    return type(left) is type(right) and left == right


__all__ = [
    "FrozenDirectList",
    "FrozenDirectObject",
    "direct_equal",
    "direct_json_value",
    "freeze_direct_mapping",
    "freeze_direct_value",
]
