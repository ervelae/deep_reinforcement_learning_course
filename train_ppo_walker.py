from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
import os
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3.common.utils import TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.noise import NormalActionNoise, VectorizedActionNoise
import torch
import torch.optim as optim
import signal
import sys

if __name__ == "__main__":

    model_name = f"ppo_walker_v3.zip"
    model_path = os.path.join("models", model_name)
    vecnormalize_path = os.path.join("environments", f"ppo_walker_v3.pkl")
    log_dir = "./logs/ppo_walker/"

    # Create the environment
    env = make_vec_env(
        "BipedalWalker-v3", n_envs=16, seed=2024, vec_env_cls=SubprocVecEnv
    )
    # Load normalization statistics if they exist
    if os.path.exists(vecnormalize_path):
        env = VecNormalize.load(vecnormalize_path, env)
        env.training = True  # Make sure to set to training mode
        env.norm_reward = True
        print("Loaded VecNormalize statistics.")
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=True)
        env.training = True  # Make sure to set to training mode

    # Check if a saved model exists
    if os.path.exists(model_path):
        # Load the saved model
        model = PPO.load(model_path, env=env)
        print("Loaded existing model.")

    else:
        # hyperparametes
        PPO_args = {
            "learning_rate": 1.9048549687349276e-05,
            "n_steps": 4096,
            "batch_size": 256,
            "n_epochs": 9,
            "gamma": 0.9164871331861589,
            "gae_lambda": 0.8320587769258511,
            "clip_range": 0.10959994031871292,
            "vf_coef": 0.7343081359148522,
            "ent_coef": 0.00010431365509533178,
            "max_grad_norm": 0.7425455184418255,
        }
        PPO_args_v2 = {
            "batch_size": 64,
            "clip_range": 0.18,
            "ent_coef": 0.0,
            "gae_lambda": 0.95,
            "gamma": 0.999,
            "learning_rate": 0.0003,
            # "n_envs": 32,
            "n_epochs": 10,
            "n_steps": 2048,
            # "n_timesteps": 5000000.0,
            # "normalize": True),
            # "policy": "MlpPolicy",
        }
        PPO_args_v3 = {
            "learning_rate": 0.00031016290222399755,
            "n_steps": 1024,
            "batch_size": 128,
            "n_epochs": 8,
            "gae_lambda": 0.8610756390880515,
            "clip_range": 0.2641177750386453,
            "vf_coef": 0.9182477231951106,
            "ent_coef": 4.9293649594261e-08,
            "max_grad_norm": 0.6337416874273284,
        }

        # This is our on-policy learner.
        model = PPO(
            "MlpPolicy",
            env,
            # use_sde=True,
            stats_window_size=100,
            tensorboard_log=log_dir,
            seed=2024,
            **PPO_args_v3,
        )

        print("Created a new model.")

    ## Handle if the training is interrupted by the user
    def signal_handler(signal, frame):
        print("Interrupted! Saving model...")
        model.save(model_path)
        env.save(vecnormalize_path)
        env.close()
        print("Saved the current model. Exiting...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        model.learn(
            total_timesteps=30e6,
            progress_bar=True,
            reset_num_timesteps=False,
            tb_log_name=model_name,
        )
    except KeyboardInterrupt:
        pass

    # model.save(model_path[:-4]+f"_checkpoint_seed{seed}.zip")  #checkpoints
    # env.save(vecnormalize_path[:-4]+f"_checkpoint_seed{seed}.pkl")

    model.save(model_path)
    env.save(vecnormalize_path)
    env.close()
    print(f"Finished training.")
