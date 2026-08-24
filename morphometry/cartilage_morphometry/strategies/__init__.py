"""Strategy registry for the cartilage pipeline.

Each strategy kind (cleanup / anchor / subch / smoothing) has a registry dict
mapping a string name → callable. Library-provided strategies are registered
on import of the corresponding sub-module. Study code can register new ones
via the `register_*` decorators without modifying the library.

Lookup is via `get_*(name)`; KeyError on unknown names lists available
choices for fast debugging.
"""
from __future__ import annotations

from typing import Callable

_REGISTRY: dict[str, dict[str, Callable]] = {
    "cleanup": {},
    "anchor": {},
    "subch": {},
    "smoothing": {},
    "thickness": {},
}


def _register(kind: str, name: str):
    def deco(fn: Callable) -> Callable:
        if name in _REGISTRY[kind]:
            raise ValueError(
                f"{kind} strategy '{name}' already registered as {_REGISTRY[kind][name]}"
            )
        _REGISTRY[kind][name] = fn
        return fn
    return deco


def register_cleanup(name: str):
    return _register("cleanup", name)


def register_anchor(name: str):
    return _register("anchor", name)


def register_subch(name: str):
    return _register("subch", name)


def register_smoothing(name: str):
    return _register("smoothing", name)


def register_thickness(name: str):
    return _register("thickness", name)


def _get(kind: str, name: str) -> Callable:
    try:
        return _REGISTRY[kind][name]
    except KeyError:
        avail = sorted(_REGISTRY[kind].keys())
        raise KeyError(
            f"Unknown {kind} strategy {name!r}. Available: {avail}"
        ) from None


def get_cleanup(name: str) -> Callable:
    return _get("cleanup", name)


def get_anchor(name: str) -> Callable:
    return _get("anchor", name)


def get_subch(name: str) -> Callable:
    return _get("subch", name)


def get_smoothing(name: str) -> Callable:
    return _get("smoothing", name)


def get_thickness(name: str) -> Callable:
    return _get("thickness", name)


def list_strategies() -> dict[str, list[str]]:
    return {k: sorted(v.keys()) for k, v in _REGISTRY.items()}


# Trigger library-strategy registration on package import.
from . import cleanup, anchor, subch, smoothing, thickness   # noqa: E402, F401
