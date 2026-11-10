import argparse
from datetime import datetime
import logging
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import random
import os
import time

from app.test import test
from definitions import OUT_DIR
from src.agent.encoder import FeaturesExtractor
from src.app.drl_scheduler import DrlScheduler
from src.env.env import NetEnv, TrainingNetEnv
from src.lib.log_config import log_config
from src.lib.timing_decorator import format_duration_hms, timing_decorator
from src.lib.training_run import (
    RUN_DIR_ENV,
    allocate_run_dir,
    build_run_id,
    link_latest_run,
    write_run_meta,
)
from src.network.net import FlowGenerator, generate_graph, Network

from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch

TOPO = 'CEV'


def _sb3_device() -> str:
    """Use CPU when GPU is missing or PyTorch lacks kernels (e.g. sm_120)."""
    if not torch.cuda.is_available():
        return "cpu"
    try:
        torch.ones(1, device="cuda")
        return "cuda"
    except Exception as error:
        logging.warning("CUDA probe failed (%s); training on CPU.", error)
        return "cpu"
NUM_ENVS = multiprocessing.cpu_count()
NUM_FLOWS = 50

DRL_ALG = 'MaskablePPO'

MONITOR_ROOT_DIR = os.path.join(OUT_DIR, "monitor")

# Set in __main__ before train(): per-run artifact root (out/runs/<run_id>/).
RUN_DIR: str | None = None


def get_best_model_path():
    base = RUN_DIR if RUN_DIR else OUT_DIR
    return os.path.join(base, f"best_model_{TOPO}_{DRL_ALG}")


def make_env(num_flows, rank: int, topo: str, list_jitters, training: bool = True,
             link_rate: int = 100, seed: int = 0):
    def _init():
        worker_seed = seed + rank
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

        graph = generate_graph(topo, link_rate)

        assert list_jitters.ndim == 2
        list_flow_generators = []
        for jitters in list_jitters:
            flow_generator = FlowGenerator(graph, jitters=jitters)
            list_flow_generators.append(flow_generator)

        if training:
            env = TrainingNetEnv(graph, random.choice(list_flow_generators), num_flows, 0.2, 0.1)
        else:
            flows = random.choice(list_flow_generators)(num_flows)
            env = NetEnv(Network(graph, flows))

        # Wrap the environment with Monitor
        env = Monitor(
            env,
            os.path.join(MONITOR_DIR, f'{"train" if training else "eval"}_{rank}'),
            info_keywords=("is_success",),
        )

        return env

    return _init


@timing_decorator(logging.info)
def train(topo: str, num_time_steps, jitters, num_flows=NUM_FLOWS,
          pre_trained_model=None, link_rate=100, seed=0,
          eval_episodes=20, eval_freq=10000, checkpoint_freq=100000):
    os.makedirs(OUT_DIR, exist_ok=True)
    set_random_seed(seed)

    n_envs = NUM_ENVS  # Number of environments to create
    env = SubprocVecEnv([
        make_env(num_flows, i, topo, jitters, link_rate=link_rate, seed=seed)
        for i in range(n_envs)
        ])

    device = _sb3_device()
    logging.info("SB3 policy device: %s", device)

    if pre_trained_model is not None:
        model = DrlScheduler.SUPPORTING_ALG[DRL_ALG].load(
            pre_trained_model, env, device=device,
        )
    else:
        policy_kwargs = dict(
            features_extractor_class=FeaturesExtractor,
        )

        if DRL_ALG == 'DQN':
            # limit the buffer size.
            model = DrlScheduler.SUPPORTING_ALG[DRL_ALG](
                "MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1,
                buffer_size=500_000, device=device, seed=seed,
            )
        else:
            model = DrlScheduler.SUPPORTING_ALG[DRL_ALG](
                "MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1,
                device=device, seed=seed,
            )
    model.set_random_seed(seed)

    eval_env = SubprocVecEnv([
        make_env(
            num_flows, i, topo, jitters, training=False,
            link_rate=link_rate, seed=seed + 10000,
        )
        for i in range(n_envs)
        ])
    eval_log_path = RUN_DIR if RUN_DIR else OUT_DIR
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=get_best_model_path(),
        log_path=eval_log_path,
        eval_freq=max(eval_freq // n_envs, 1),
        n_eval_episodes=eval_episodes,
    )
    checkpoint_dir = os.path.join(RUN_DIR if RUN_DIR else OUT_DIR, "checkpoints")
    checkpoint_callback = CheckpointCallback(
        save_freq=max(checkpoint_freq // n_envs, 1),
        save_path=checkpoint_dir,
        name_prefix="model",
    )
    callback = CallbackList([eval_callback, checkpoint_callback])

    # logging.debug(model.policy)

    # Train the agent
    learn_started = time.perf_counter()
    model.learn(total_timesteps=num_time_steps, callback=callback)
    learn_seconds = time.perf_counter() - learn_started
    learn_hms = format_duration_hms(learn_seconds)
    final_model_path = os.path.join(RUN_DIR if RUN_DIR else OUT_DIR, "final_model")
    model.save(final_model_path)
    logging.info("Final model saved to %s.zip", final_model_path)

    logging.info("------Finish learning------")
    logging.info(
        "PPO training wall time: %s (%.1f seconds)",
        learn_hms,
        learn_seconds,
    )
    print(f"PPO training finished, total wall time: {learn_hms}")
    eval_env.close()
    env.close()


def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")


def plot_results(log_folder, title="Learning Curve"):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), "timesteps")
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel("Number of Timesteps")
    plt.ylabel("Rewards")
    plt.title(title + " Smoothed")
    plt.savefig(os.path.join(log_folder, "reward.png"))
    plt.show()


if __name__ == "__main__":
    # specify an existing model to train.
    parser = argparse.ArgumentParser()
    parser.add_argument('--time_steps', type=int, required=True)
    parser.add_argument('--num_flows', type=int, nargs='?', default=NUM_FLOWS)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--num_envs', type=int, default=NUM_ENVS)
    parser.add_argument('--alg', type=str, default=None)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--topo', type=str, default="CEV")
    parser.add_argument('--link_rate', type=int, default=100)
    parser.add_argument('--jitters', type=str, nargs='+', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--eval_episodes', type=int, default=20)
    parser.add_argument('--eval_freq', type=int, default=10000)
    parser.add_argument('--checkpoint_freq', type=int, default=100000)
    parser.add_argument(
        '--run_name',
        type=str,
        default=None,
        help='Optional suffix for this run folder name (experiment tag)',
    )
    args = parser.parse_args()

    if args.alg is not None:
        assert args.alg in DrlScheduler.SUPPORTING_ALG, ValueError(f"Unknown alg {args.alg}")
        DRL_ALG = args.alg

    if args.link_rate is not None:
        support_link_rates = [100, 1000]
        assert args.link_rate in support_link_rates, \
            f"Unknown link rate {args.link_rate}, which is not in supported link rates {support_link_rates}"

    list_jitters = []
    for jitters in args.jitters:
        jitters = [float(j) for j in jitters.split(",")]
        assert all(0 <= j <= 1 for j in jitters), ValueError(f"Jitters should be in range [0, 1]")
        list_jitters.append(jitters)
    list_jitters = np.array(list_jitters)

    TOPO = args.topo

    NUM_ENVS = args.num_envs

    run_started = datetime.now()
    run_id = build_run_id(
        topo=args.topo,
        alg=DRL_ALG,
        time_steps=args.time_steps,
        num_envs=NUM_ENVS,
        num_flows=args.num_flows,
        link_rate=args.link_rate,
        jitters=list_jitters,
        started_at=run_started,
        extra=args.run_name,
    )
    RUN_DIR = allocate_run_dir(run_id)
    os.environ[RUN_DIR_ENV] = RUN_DIR
    os.makedirs(os.path.join(RUN_DIR, "env_logs"), exist_ok=True)
    MONITOR_DIR = os.path.join(RUN_DIR, "monitor")
    os.makedirs(MONITOR_DIR, exist_ok=True)
    link_latest_run(RUN_DIR)
    # Legacy flat monitor tree (numbered) — keep a symlink for old scripts.
    legacy_link = os.path.join(MONITOR_ROOT_DIR, "latest")
    try:
        if os.path.islink(legacy_link) or os.path.exists(legacy_link):
            os.remove(legacy_link)
        os.symlink(MONITOR_DIR, legacy_link, target_is_directory=True)
    except OSError:
        logging.debug("Could not update monitor/latest symlink", exc_info=True)

    log_config(os.path.join(RUN_DIR, "train.log"), logging.DEBUG)
    write_run_meta(
        RUN_DIR,
        {
            "run_id": os.path.basename(RUN_DIR),
            "started_at": run_started.isoformat(),
            "args": vars(args),
            "drl_alg": DRL_ALG,
            "paths": {
                "run_dir": RUN_DIR,
                "monitor_dir": MONITOR_DIR,
                "best_model_dir": get_best_model_path(),
                "final_model_zip": os.path.join(RUN_DIR, "final_model.zip"),
                "checkpoints_dir": os.path.join(RUN_DIR, "checkpoints"),
            },
        },
    )

    logging.info("Training run directory: %s", RUN_DIR)
    logging.info(args)

    logging.info("start training...")
    script_started = time.perf_counter()
    train(args.topo, args.time_steps,
          list_jitters,
          num_flows=args.num_flows,
          pre_trained_model=args.model,
          link_rate=args.link_rate,
          seed=args.seed,
          eval_episodes=args.eval_episodes,
          eval_freq=args.eval_freq,
          checkpoint_freq=args.checkpoint_freq)

    # MONITOR_DIR = os.path.join(MONITOR_ROOT_DIR, str(1))
    plot_results(MONITOR_DIR)

    test(args.topo, args.num_flows, NUM_ENVS,
         os.path.join(get_best_model_path(), "best_model"), DRL_ALG, args.link_rate)

    script_seconds = time.perf_counter() - script_started
    script_hms = format_duration_hms(script_seconds)
    logging.info(
        "Total train.py wall time (learn + plot + test): %s (%.1f seconds)",
        script_hms,
        script_seconds,
    )
    print(f"train.py finished, total wall time: {script_hms}")
    write_run_meta(
        RUN_DIR,
        {
            "run_id": os.path.basename(RUN_DIR),
            "started_at": run_started.isoformat(),
            "finished_at": datetime.now().isoformat(),
            "args": vars(args),
            "drl_alg": DRL_ALG,
            "wall_time_seconds": script_seconds,
            "wall_time_hms": script_hms,
            "paths": {
                "run_dir": RUN_DIR,
                "monitor_dir": MONITOR_DIR,
                "best_model_dir": get_best_model_path(),
                "best_model_zip": os.path.join(get_best_model_path(), "best_model.zip"),
                "final_model_zip": os.path.join(RUN_DIR, "final_model.zip"),
                "checkpoints_dir": os.path.join(RUN_DIR, "checkpoints"),
                "evaluations_npz": os.path.join(RUN_DIR, "evaluations.npz"),
                "train_log": os.path.join(RUN_DIR, "train.log"),
                "env_logs_dir": os.path.join(RUN_DIR, "env_logs"),
                "latest_symlink": os.path.join(OUT_DIR, "latest"),
            },
        },
    )
    link_latest_run(RUN_DIR)
    print(f"Artifacts saved under: {RUN_DIR}")
    print(f"Latest run symlink: {os.path.join(OUT_DIR, 'latest')}")
