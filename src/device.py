#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Torch device utilities shared by ver1/ver2 entry points."""

from __future__ import annotations

from typing import Optional, Union

try:
    import torch
except ImportError:  # pragma: no cover - torch is a runtime dependency
    torch = None  # type: ignore


def _canonical_device_name(name: str) -> str:
    return name.strip().lower()


def get_torch_device(preferred: Optional[str] = None) -> "torch.device":
    """Pick the best available torch device.

    Preference order: explicit ``preferred`` > MPS > CUDA > CPU.
    ``preferred`` may be ``None``/``auto``/``cpu``/``mps``/``cuda[:index]``.
    """

    if torch is None:  # fallback when torch is not installed
        raise RuntimeError("PyTorch is required to resolve the compute device.")

    def make_device(device_name: str) -> "torch.device":
        return torch.device(device_name)

    def mps_available() -> bool:
        return bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()

    if preferred:
        pref = _canonical_device_name(preferred)
        if pref in {"auto", ""}:
            preferred = None
        elif pref == "cpu":
            return make_device("cpu")
        elif pref.startswith("cuda"):
            if torch.cuda.is_available():
                return make_device(pref)
        elif pref == "mps":
            if mps_available():
                return make_device("mps")

    if mps_available():
        return make_device("mps")
    if torch.cuda.is_available():
        return make_device("cuda")
    return make_device("cpu")


def device_as_str(device: Union[str, "torch.device"]) -> str:
    if isinstance(device, str):
        return device
    index = getattr(device, "index", None)
    return device.type if index is None else f"{device.type}:{index}"


__all__ = ["get_torch_device", "device_as_str"]

