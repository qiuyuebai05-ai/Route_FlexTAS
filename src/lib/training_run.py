"""Human-readable training run ids and per-run output directories."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any

from definitions import OUT_DIR

RUN_DIR_ENV = "ROUTE_FLEXTAS_RUN_DIR"


def current_run_dir() -> str | None:
    return os.environ.get(RUN_DIR_ENV) or None


def _format_steps(time_steps: int) -> str:
    if time_steps >= 1_000_000:
        return f"{time_steps / 1_000_000:g}M"
    if time_steps >= 1_000:
        return f"{time_steps / 1_000:g}k"
    return str(time_steps)


def _jitter_tag(jitters: Any) -> str:
    import numpy as np

    arr = np.asarray(jitters).reshape(-1)
    parts = [f"{float(j):g}".replace(".", "p") for j in arr[:3]]
    return "j" + "_".join(parts)


def _clock_error_tag(clock_error_bound: float) -> str:
    return f"ce{float(clock_error_bound):g}us".replace(".", "p")


def build_run_id(
    *,
    topo: str,
    alg: str,
    time_steps: int,
    num_envs: int,
    num_flows: int,
    link_rate: int,
    jitters: Any,
    clock_error_bound: float = 0.0,
    started_at: datetime | None = None,
    extra: str | None = None,
) -> str:
    started_at = started_at or datetime.now()
    stamp = started_at.strftime("%Y%m%d-%H%M%S")
    alg_short = re.sub(r"[^A-Za-z0-9]", "", alg)[:12] or "DRL"
    core = (
        f"{stamp}_{topo}_{alg_short}_{_format_steps(time_steps)}step"
        f"_{num_envs}env_{num_flows}flow_lr{link_rate}_{_jitter_tag(jitters)}"
        f"_{_clock_error_tag(clock_error_bound)}"
    )
    if extra:
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", extra.strip())[:32]
        core = f"{core}_{safe}"
    return core


def allocate_run_dir(run_id: str, parent: str | None = None) -> str:
    """Return a new directory path; append _2, _3 if name exists."""
    parent = parent or os.path.join(OUT_DIR, "runs")
    os.makedirs(parent, exist_ok=True)
    candidate = os.path.join(parent, run_id)
    if not os.path.exists(candidate):
        os.makedirs(candidate)
        return candidate
    n = 2
    while True:
        candidate = os.path.join(parent, f"{run_id}_{n}")
        if not os.path.exists(candidate):
            os.makedirs(candidate)
            return candidate
        n += 1


def write_run_meta(run_dir: str, meta: dict[str, Any]) -> str:
    path = os.path.join(run_dir, "run_meta.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)
    return path


def link_latest_run(run_dir: str) -> None:
    """Point out/latest and out/runs/latest at this run directory."""
    for link_path in (
        os.path.join(OUT_DIR, "latest"),
        os.path.join(OUT_DIR, "runs", "latest"),
    ):
        parent = os.path.dirname(link_path)
        os.makedirs(parent, exist_ok=True)
        try:
            if os.path.islink(link_path) or os.path.exists(link_path):
                os.remove(link_path)
            os.symlink(run_dir, link_path, target_is_directory=True)
        except OSError:
            pass
