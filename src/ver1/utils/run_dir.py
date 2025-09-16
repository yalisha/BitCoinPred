#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path


def create_next_run_dir(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    max_id = 0
    for p in base.iterdir():
        if p.is_dir() and p.name.isdigit():
            try:
                max_id = max(max_id, int(p.name))
            except ValueError:
                continue
    run_id = max_id + 1
    run_dir = base / str(run_id)
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir

