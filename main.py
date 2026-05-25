import argparse
import multiprocessing
import os
import sys
import torch


def main():
    parser = argparse.ArgumentParser(description="RL Autonomous Driving Simulator")
    parser.add_argument(
        "--mode", choices=["train", "watch", "demo"], required=True,
        help=(
            "train : headless fast training (SubprocVecEnv) | "
            "watch : training with live 2D visualisation (DummyVecEnv) | "
            "demo  : visualise a trained model"
        ),
    )
    parser.add_argument("--model", default="models/best/best_model.zip",
                        help="Trained model path (demo / watch-resume mode)")
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--envs", type=int, default=8,
                        help="Parallel env count (train=8 recommended, watch uses 8 always)")
    parser.add_argument("--cars", type=int, default=8,
                        help="Number of cars shown in demo / watch mode")
    args = parser.parse_args()

    # ── train: headless, maximum speed ────────────────────────────────────────
    if args.mode == "train":
        from src.training.train import train
        train(n_envs=args.envs, total_timesteps=args.timesteps)

    # ── watch: train + live 2D visualisation ──────────────────────────────────
    elif args.mode == "watch":
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
        from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
        from src.environment.car_env import CarEnv
        from src.rendering.simulator_2d import Simulator2D, WatchCallback

        n_cars = args.cars
        os.makedirs("models/best", exist_ok=True)
        os.makedirs("tensorboard_logs", exist_ok=True)

        # Create envs explicitly so we can pass them to the renderer
        raw_envs = [CarEnv() for _ in range(n_cars)]
        for i, env in enumerate(raw_envs):
            env.reset(seed=i)

        vec_env = DummyVecEnv([lambda e=env: e for env in raw_envs])
        eval_env = VecMonitor(DummyVecEnv([lambda: CarEnv()]))

        device = "mps" if torch.backends.mps.is_available() else "cpu"

        # Try resuming from existing model
        if os.path.exists(args.model):
            print(f"Resuming from {args.model}")
            model = PPO.load(args.model, env=vec_env, device=device)
        else:
            model = PPO(
                "MlpPolicy", vec_env,
                learning_rate=3e-4, n_steps=512, batch_size=64,
                n_epochs=5, gamma=0.99, gae_lambda=0.95,
                clip_range=0.2, ent_coef=0.01,
                tensorboard_log="./tensorboard_logs/",
                device=device, verbose=0,
                policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
            )

        renderer = Simulator2D(envs=raw_envs, model=None)

        checkpoint_cb = CheckpointCallback(
            save_freq=max(50_000 // n_cars, 1),
            save_path="./models/",
            name_prefix="ppo_car",
        )
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path="./models/best/",
            eval_freq=max(25_000 // n_cars, 1),
            n_eval_episodes=5,
            deterministic=True,
            verbose=0,
        )
        watch_cb = WatchCallback(renderer, render_fps=30)

        print(f"Watch mode: {n_cars} cars | device={device}")
        print("Close the window or press ESC to stop.")
        model.learn(
            total_timesteps=args.timesteps,
            callback=[checkpoint_cb, eval_cb, watch_cb],
        )
        model.save("models/ppo_car_final")
        print("Saved: models/ppo_car_final.zip")

    # ── demo: load model, show N cars ─────────────────────────────────────────
    elif args.mode == "demo":
        from stable_baselines3 import PPO
        from src.environment.car_env import CarEnv
        from src.rendering.simulator_2d import Simulator2D

        n_cars = args.cars
        envs   = [CarEnv() for _ in range(n_cars)]

        model = None
        if os.path.exists(args.model):
            model = PPO.load(args.model, device="cpu")
            print(f"Model loaded: {args.model}")
        else:
            print(f"Model not found at {args.model} — using random actions.")

        renderer = Simulator2D(envs=envs, model=model)
        renderer.run_demo()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
