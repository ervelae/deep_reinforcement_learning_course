from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import SAC
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
    
    
    model_name = f"sac_walker_v2.zip"
    model_path = os.path.join("models", model_name)
    vecnormalize_path = os.path.join("environments", f"sac_walker_v2.pkl")
    
    ## Make environment

    env = make_vec_env(
        "BipedalWalker-v3",
        n_envs=16,  
        seed=2024,
        vec_env_cls=SubprocVecEnv,
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
        model = SAC.load(model_path, env=env)
        print("Loaded existing model.")
    else:
        SAC_args = {
            "learning_rate": 7.613357745707285e-05,
            "buffer_size": 952203,
            "batch_size": 256,
            "tau": 0.014397519335421863,
            "gamma": 0.9660841833980763,
            "train_freq": 1,
            "gradient_steps": 13,
            "ent_coef": 0.001,
        }

        # This is our off-policy learner.
        model = SAC(
            "MlpPolicy",
            env,
            learning_starts=int(1e4),
            use_sde=True,
            tensorboard_log=f"./logs/sac_walker/",
            verbose=0,
            seed=2024,
            **SAC_args,
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
            total_timesteps=3e6,
            progress_bar=True,
            reset_num_timesteps=False,
            tb_log_name=model_name,
        )
    except KeyboardInterrupt:
        pass


    # Save the models
    model.save(model_path)
    env.save(vecnormalize_path)
    env.close()
    print(f"Finished training for {model_name}.")

"""
2:21:00
490 000 iterations
"""