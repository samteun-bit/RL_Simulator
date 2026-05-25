import argparse
import multiprocessing
import sys


def main():
    parser = argparse.ArgumentParser(description="RL Autonomous Driving Simulator")
    parser.add_argument(
        "--mode", choices=["train", "demo"], required=True,
        help="'train' to train the RL agent, 'demo' to visualize a trained model",
    )
    parser.add_argument(
        "--model", type=str, default="models/best/best_model.zip",
        help="Path to trained model .zip (demo mode only)",
    )
    parser.add_argument(
        "--timesteps", type=int, default=2_000_000,
        help="Total training timesteps (default: 2,000,000)",
    )
    parser.add_argument(
        "--envs", type=int, default=8,
        help="Number of parallel training environments (default: 8)",
    )
    args = parser.parse_args()

    if args.mode == "train":
        # Guard required for SubprocVecEnv on macOS (spawn start method)
        from src.training.train import train
        train(n_envs=args.envs, total_timesteps=args.timesteps)

    elif args.mode == "demo":
        # Import only here — ShowBase opens a window on import
        from src.rendering.simulator_3d import Simulator3D
        app = Simulator3D(model_path=args.model)
        app.run()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
