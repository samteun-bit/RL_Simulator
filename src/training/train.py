import os
import multiprocessing
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from src.environment.car_env import CarEnv


def make_env(rank, seed=0):
    def _init():
        env = CarEnv()
        env.reset(seed=seed + rank)
        return env
    return _init


def train(n_envs: int = 8, total_timesteps: int = 2_000_000):
    os.makedirs("models/best", exist_ok=True)
    os.makedirs("tensorboard_logs", exist_ok=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Training on device: {device}")
    print(f"Parallel environments: {n_envs}")
    print(f"Total timesteps: {total_timesteps:,}")

    env = VecMonitor(SubprocVecEnv([make_env(i) for i in range(n_envs)]))
    eval_env = VecMonitor(SubprocVecEnv([make_env(99, seed=42)]))

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log="./tensorboard_logs/",
        device=device,
        verbose=1,
        policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(50_000 // n_envs, 1),
        save_path="./models/",
        name_prefix="ppo_car",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="./models/best/",
        eval_freq=max(25_000 // n_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
        verbose=1,
    )

    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_cb, eval_cb])
    model.save("models/ppo_car_final")
    print("Training complete. Model saved to models/ppo_car_final.zip")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    train()
