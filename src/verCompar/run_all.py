#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Unified runner for verCompar experiments."""

from __future__ import annotations

import argparse
import importlib
import traceback
from pathlib import Path
from typing import Dict, Iterable, Optional

from . import config


def _load_callable(module_path: str, entrypoint: str):
    module = importlib.import_module(module_path)
    try:
        fn = getattr(module, entrypoint)
    except AttributeError as exc:  # pragma: no cover - defensive
        raise AttributeError(f"Entrypoint '{entrypoint}' not found in module '{module_path}'") from exc
    return fn


def run_models(model_names: Optional[Iterable[str]] = None,
               stop_on_error: bool = False) -> Dict[str, Dict[str, object]]:
    model_order = list(model_names) if model_names else list(config.MODEL_CONFIGS.keys())
    results: Dict[str, Dict[str, object]] = {}

    for name in model_order:
        if name not in config.MODEL_CONFIGS:
            print(f"[Runner] Skip unknown model '{name}'")
            continue

        cfg = config.MODEL_CONFIGS[name]
        entry = _load_callable(cfg["module"], cfg.get("entrypoint", "run_experiment"))

        overrides = dict(cfg.get("overrides", {}))
        out_subdir = cfg.get("output_subdir", name)

        global_settings = dict(config.GLOBAL_SETTINGS)
        global_settings["preferred_device"] = config.PREFERRED_DEVICE

        if name == "tft":
            global_settings["out_dir"] = Path(config.OUT_DIR) / out_subdir
        else:
            global_settings["out_dir"] = Path(config.OUT_DIR)
            overrides.setdefault("output_subdir", out_subdir)

        print(f"[Runner] >>> Running {name} ...")
        try:
            result = entry(overrides=overrides, global_settings=global_settings)
            results[name] = result
            print(f"[Runner] <<< {name} completed. Output: {result.get('run_dir')}")
        except Exception as exc:  # pragma: no cover - runtime safeguard
            tb = traceback.format_exc()
            results[name] = {"error": str(exc), "traceback": tb}
            print(f"[Runner] !!! {name} failed: {exc}")
            if stop_on_error:
                raise

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multiple verCompar model experiments")
    parser.add_argument("models", nargs="*", help="Models to run (default: all)")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop execution on the first error")
    args = parser.parse_args()

    models = args.models if args.models else None
    run_models(models, stop_on_error=args.stop_on_error)


if __name__ == "__main__":
    main()

