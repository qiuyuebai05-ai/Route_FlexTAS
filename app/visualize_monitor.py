"""Visualize training Monitor CSVs under out/monitor and EvalCallback evaluations.npz."""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.results_plotter import load_results, ts2xy

from definitions import OUT_DIR


def _default_latest_run_dir() -> str | None:
    link = os.path.join(OUT_DIR, "latest")
    if os.path.islink(link) or os.path.isdir(link):
        return os.path.realpath(link)
    return None


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < window:
        return values
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")


def _smooth_xy(x: np.ndarray, y: np.ndarray, window: int = 50):
    y_s = moving_average(y, window)
    x_s = x[len(x) - len(y_s) :]
    return x_s, y_s


def plot_training_curve(monitor_dir: str, window: int = 50) -> tuple[np.ndarray, np.ndarray]:
    x, y = ts2xy(load_results(monitor_dir), "timesteps")
    if len(y) == 0:
        return x, y
    return _smooth_xy(x, y, window=window)


def save_run_dashboard(
    run_dir: str,
    eval_npz: str | None,
    window: int = 50,
) -> str:
    """Single run: training curve + optional EvalCallback curve."""
    run_name = os.path.basename(run_dir.rstrip(os.sep))
    n_train = len(glob.glob(os.path.join(run_dir, "train_*.monitor.csv")))
    fig, axes = plt.subplots(2 if eval_npz and os.path.isfile(eval_npz) else 1, 1, figsize=(10, 8), squeeze=False)
    ax_train = axes[0, 0]

    try:
        x, y = plot_training_curve(run_dir, window=window)
    except Exception:
        x, y = np.array([]), np.array([])

    if len(y) == 0:
        ax_train.text(0.5, 0.5, "No training episodes in monitor logs", ha="center", va="center")
    else:
        ax_train.plot(x, y, color="#2563eb", lw=1.5)
        ax_train.set_xlabel("Environment timesteps")
        ax_train.set_ylabel("Episode reward (MA-{0})".format(window))
        ax_train.set_title(f"Training run {run_name} ({n_train} parallel env log(s))")
        ax_train.grid(True, alpha=0.3)

    if len(axes) > 1:
        ax_eval = axes[1, 0]
        data = np.load(eval_npz)
        timesteps = data["timesteps"]
        results = data["results"]
        mean_r = results.mean(axis=1)
        std_r = results.std(axis=1)
        ax_eval.plot(timesteps, mean_r, color="#059669", lw=1.5, label="Eval mean reward")
        ax_eval.fill_between(timesteps, mean_r - std_r, mean_r + std_r, color="#059669", alpha=0.2)
        ax_eval.set_xlabel("Training timesteps (EvalCallback)")
        ax_eval.set_ylabel("Reward")
        ax_eval.set_title("Periodic evaluation (best model selection)")
        ax_eval.legend(loc="lower right")
        ax_eval.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(run_dir, "training_dashboard.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def save_runs_overview(monitor_root: str, window: int = 50) -> str:
    """Overlay smoothed training curves for every numbered run subdirectory."""
    run_dirs = sorted(
        (p for p in Path(monitor_root).iterdir() if p.is_dir() and p.name.isdigit()),
        key=lambda p: int(p.name),
    )
    fig, ax = plt.subplots(figsize=(11, 6))
    for run_dir in run_dirs:
        csvs = list(run_dir.glob("train_*.monitor.csv"))
        if not csvs:
            continue
        try:
            x, y = plot_training_curve(str(run_dir), window=window)
        except Exception:
            continue
        if len(y) == 0:
            continue
        ax.plot(x, y, lw=1.2, label=f"run {run_dir.name} ({len(csvs)} env)")

    ax.set_xlabel("Environment timesteps")
    ax.set_ylabel("Episode reward (smoothed)")
    ax.set_title("All training runs under out/monitor")
    ax.grid(True, alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    out_path = os.path.join(monitor_root, "training_overview.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def summarize_runs(monitor_root: str) -> list[dict]:
    rows = []
    for run_dir in sorted(Path(monitor_root).iterdir(), key=lambda p: p.name):
        if not run_dir.is_dir() or not run_dir.name.isdigit():
            continue
        episodes = 0
        timesteps = 0
        rewards: list[float] = []
        for csv_path in run_dir.glob("train_*.monitor.csv"):
            with open(csv_path, encoding="utf-8") as f:
                lines = [ln for ln in f if not ln.startswith("#") and ln.strip() != "r,l,t"]
            episodes += len(lines)
            for ln in lines:
                parts = ln.strip().split(",")
                if len(parts) >= 2:
                    rewards.append(float(parts[0]))
                    timesteps += int(parts[1])
        if episodes == 0:
            continue
        rows.append(
            {
                "run": run_dir.name,
                "envs": len(list(run_dir.glob("train_*.monitor.csv"))),
                "episodes": episodes,
                "timesteps": timesteps,
                "mean_reward": float(np.mean(rewards)),
                "last_100_mean": float(np.mean(rewards[-100:])),
            }
        )
    return rows


def main():
    latest = _default_latest_run_dir()
    default_monitor = (
        os.path.join(latest, "monitor")
        if latest and os.path.isdir(os.path.join(latest, "monitor"))
        else os.path.join(OUT_DIR, "monitor")
    )
    default_eval = (
        os.path.join(latest, "evaluations.npz")
        if latest and os.path.isfile(os.path.join(latest, "evaluations.npz"))
        else os.path.join(OUT_DIR, "evaluations.npz")
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--monitor_root",
        default=default_monitor,
        help="Monitor CSV folder (default: out/latest/monitor if present)",
    )
    parser.add_argument(
        "--run",
        type=int,
        nargs="*",
        default=None,
        help="Run id(s) to plot (default: all non-empty runs)",
    )
    parser.add_argument(
        "--eval_npz",
        default=default_eval,
        help="EvalCallback npz (default: out/latest/evaluations.npz if present)",
    )
    parser.add_argument("--smooth", type=int, default=50, help="Moving-average window")
    args = parser.parse_args()

    plt.switch_backend("Agg")

    if list(Path(args.monitor_root).glob("train_*.monitor.csv")):
        out = save_run_dashboard(args.monitor_root, args.eval_npz, window=args.smooth)
        print("Saved:", out)
        return

    run_ids = args.run
    if run_ids is None:
        run_ids = [int(r["run"]) for r in summarize_runs(args.monitor_root)]

    stats_list = summarize_runs(args.monitor_root)
    stats = {int(r["run"]): r for r in stats_list}
    longest = max(stats, key=lambda k: stats[k]["timesteps"]) if stats else None

    paths = []
    for run_id in sorted(run_ids):
        run_dir = os.path.join(args.monitor_root, str(run_id))
        if not os.path.isdir(run_dir):
            continue
        eval_npz = args.eval_npz if run_id == longest else None
        paths.append(save_run_dashboard(run_dir, eval_npz, window=args.smooth))

    overview = save_runs_overview(args.monitor_root, window=args.smooth)

    print("=== Monitor summary ===")
    for row in summarize_runs(args.monitor_root):
        print(
            f"run {row['run']}: envs={row['envs']}, episodes={row['episodes']}, "
            f"timesteps={row['timesteps']}, mean_r={row['mean_reward']:.2f}, "
            f"last100={row['last_100_mean']:.2f}"
        )
    print("\nSaved:")
    for p in paths:
        print(" ", p)
    print(" ", overview)


if __name__ == "__main__":
    main()
