"""Evaluate one trained policy on reproducible, independently generated flow sets."""

import argparse
import csv
import json
import os
import random
import time
from collections import Counter

import numpy as np
import torch

from src.app.drl_scheduler import DrlScheduler
from src.env.env import NetEnv
from src.network.net import Network, generate_flows, generate_graph


def _parse_modes(values):
    valid = {"deterministic", "stochastic"}
    unknown = set(values) - valid
    if unknown:
        raise ValueError(f"Unknown evaluation modes: {sorted(unknown)}")
    return values


def _resolve_model_path(path):
    if os.path.isfile(path):
        return path
    if not path.endswith(".zip") and os.path.isfile(f"{path}.zip"):
        return f"{path}.zip"
    raise FileNotFoundError(f"Cannot find model: {path}")


def _run_episode(model, *, topo, link_rate, num_flows, jitter,
                 instance_seed, deterministic, policy_seed, alg):
    random.seed(instance_seed)
    graph = generate_graph(topo, link_rate)
    flows = generate_flows(graph, num_flows, seed=instance_seed, jitters=jitter)
    env = NetEnv(Network(graph, flows))

    # Keep the flow set and flow order identical across evaluation modes/repeats.
    random.seed(instance_seed + 1)
    obs, _ = env.reset(seed=instance_seed)
    np.random.seed(policy_seed)
    torch.manual_seed(policy_seed)

    terminated = False
    truncated = False
    episode_reward = 0.0
    steps = 0
    route_choices = []
    selected_hops = []
    path_stretches = []
    info = {}
    started = time.perf_counter()

    while not (terminated or truncated):
        selecting_route = not env.route_selected
        flow = env.current_flow()
        candidate_paths = env.candidate_paths[flow] if selecting_route else None
        predict_kwargs = {"deterministic": deterministic}
        if alg == "MaskablePPO":
            predict_kwargs["action_masks"] = env.action_masks()
        action, _ = model.predict(obs, **predict_kwargs)
        action = int(np.asarray(action).item())

        if selecting_route:
            shortest_hops = len(candidate_paths[0])
            route_choices.append(action)
            selected_hops.append(len(candidate_paths[action]))
            path_stretches.append(len(candidate_paths[action]) / shortest_hops)

        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += float(reward)
        steps += 1

    elapsed = time.perf_counter() - started
    success = bool(info.get("success", False))
    result = {
        "instance_seed": instance_seed,
        "deterministic": deterministic,
        "policy_seed": policy_seed,
        "success": success,
        "episode_reward": episode_reward,
        "scheduled_flows": env.flow_index,
        "scheduled_fraction": env.flow_index / num_flows,
        "steps": steps,
        "runtime_seconds": elapsed,
        "failure_message": "" if success else info.get("msg", ""),
        "routes_selected": len(route_choices),
        "alternate_routes": sum(choice != 0 for choice in route_choices),
        "alternate_route_fraction": (
            sum(choice != 0 for choice in route_choices) / len(route_choices)
            if route_choices else 0.0
        ),
        "mean_selected_hops": float(np.mean(selected_hops)) if selected_hops else 0.0,
        "mean_path_stretch": float(np.mean(path_stretches)) if path_stretches else 0.0,
        "route_choices": json.dumps(route_choices),
    }
    env.close()
    return result


def _summarize(rows, num_instances):
    summary = {}
    for deterministic in (True, False):
        mode_rows = [row for row in rows if row["deterministic"] == deterministic]
        if not mode_rows:
            continue
        mode = "deterministic" if deterministic else "stochastic"
        successful_instances = {
            row["instance_seed"] for row in mode_rows if row["success"]
        }
        route_counts = Counter()
        for row in mode_rows:
            route_counts.update(json.loads(row["route_choices"]))
        summary[mode] = {
            "attempts": len(mode_rows),
            "num_instances": num_instances,
            "attempt_success_rate": float(np.mean([row["success"] for row in mode_rows])),
            "instance_any_success_rate": len(successful_instances) / num_instances,
            "mean_reward": float(np.mean([row["episode_reward"] for row in mode_rows])),
            "mean_scheduled_fraction": float(np.mean([
                row["scheduled_fraction"] for row in mode_rows
            ])),
            "mean_runtime_seconds": float(np.mean([
                row["runtime_seconds"] for row in mode_rows
            ])),
            "alternate_route_fraction": (
                sum(row["alternate_routes"] for row in mode_rows)
                / sum(row["routes_selected"] for row in mode_rows)
            ),
            "mean_path_stretch": float(np.mean([
                row["mean_path_stretch"] for row in mode_rows
            ])),
            "route_choice_counts": dict(sorted(route_counts.items())),
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--alg", default="MaskablePPO", choices=DrlScheduler.SUPPORTING_ALG)
    parser.add_argument("--topo", default="CEV")
    parser.add_argument("--link_rate", type=int, default=100)
    parser.add_argument("--num_flows", type=int, default=50)
    parser.add_argument("--jitter", type=float, default=0.1)
    parser.add_argument("--num_instances", type=int, default=100)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument(
        "--modes", nargs="+", default=["deterministic", "stochastic"]
    )
    parser.add_argument("--stochastic_repeats", type=int, default=5)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    args.modes = _parse_modes(args.modes)

    model_path = _resolve_model_path(args.model)
    print(f"Loading model from {model_path} ...", flush=True)
    model = DrlScheduler.SUPPORTING_ALG[args.alg].load(model_path, device="cpu")
    print("Model loaded. Starting evaluation ...", flush=True)

    planned = args.num_instances * (
        ("deterministic" in args.modes)
        + ("stochastic" in args.modes) * args.stochastic_repeats
    )
    print(
        f"Plan: {args.num_instances} flow instances, {planned} episodes total "
        f"(modes={args.modes}, stochastic_repeats={args.stochastic_repeats}).",
        flush=True,
    )
    rows = []
    done_eps = 0
    eval_started = time.perf_counter()

    for offset in range(args.num_instances):
        instance_seed = args.seed + offset
        if "deterministic" in args.modes:
            rows.append(_run_episode(
                model,
                topo=args.topo,
                link_rate=args.link_rate,
                num_flows=args.num_flows,
                jitter=args.jitter,
                instance_seed=instance_seed,
                deterministic=True,
                policy_seed=instance_seed,
                alg=args.alg,
            ))
            done_eps += 1
        if "stochastic" in args.modes:
            for repeat in range(args.stochastic_repeats):
                rows.append(_run_episode(
                    model,
                    topo=args.topo,
                    link_rate=args.link_rate,
                    num_flows=args.num_flows,
                    jitter=args.jitter,
                    instance_seed=instance_seed,
                    deterministic=False,
                    policy_seed=instance_seed * 1000 + repeat,
                    alg=args.alg,
                ))
                done_eps += 1

        elapsed = time.perf_counter() - eval_started
        inst_rows = [r for r in rows if r["instance_seed"] == instance_seed]
        inst_ok = any(r["success"] for r in inst_rows)
        print(
            f"[{offset + 1}/{args.num_instances}] seed={instance_seed} "
            f"any_success={inst_ok} | episodes {done_eps}/{planned} "
            f"| elapsed {elapsed:.0f}s",
            flush=True,
        )

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "episodes.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "model": os.path.abspath(model_path),
        "settings": vars(args),
        "summary": _summarize(rows, args.num_instances),
    }
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        json.dump(report, summary_file, indent=2, ensure_ascii=False)

    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    print(f"Episode data: {csv_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
