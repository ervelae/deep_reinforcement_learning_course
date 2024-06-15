from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import TD3
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
from stable_baselines3.common.callbacks import BaseCallback


# Custom wrapper to change the mass of the torso.
class ChangeMassWrapper(gym.Wrapper):
    def __init__(self, env, torso_mass):
        super().__init__(env)
        self.torso_mass = torso_mass
        self.env.unwrapped.model.body_mass[1] = self.torso_mass


class SaveOnBestTrainingRewardCallback(BaseCallback):
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Retrieve training reward
            x, y = self.model.env.get_attr("rewards")
            mean_reward = np.mean(y[-100:])
            if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps}")
                print(
                    f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}"
                )

            # New best model, save the model
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                if self.verbose > 0:
                    print(f"Saving new best model to {self.save_path}")
                self.model.save(self.save_path)

        return True


if __name__ == "__main__":

    for mass in [9,6,3]:
        for seed in [10, 11]:

            model_name = f"td3_hopper_mass{mass}_v3.zip"
            model_path = os.path.join("models", model_name)
            vecnormalize_path = os.path.join("environments", f"td3_hopper_mass{mass}_v3.pkl")

            log_dir = "./logs/td3_hopper/"
            #callback = SaveOnBestTrainingRewardCallback(
            #    check_freq=10000, log_dir=log_dir + model_name
            #)

            n_envs = 32
            env = make_vec_env(
                "Hopper-v4",
                n_envs=n_envs,
                seed=seed,
                vec_env_cls=SubprocVecEnv,
                wrapper_class=ChangeMassWrapper,
                wrapper_kwargs=dict(torso_mass=mass),
                env_kwargs={"healthy_reward": 0.5},
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

            # Load model if it exists
            if os.path.exists(model_path):
                # Load the saved model
                model = TD3.load(model_path, env=env)
                print("Loaded existing model.")

                # Changes to the model
                # Define the action noise
                n_actions = env.action_space.shape[-1]
                base_noise = NormalActionNoise(
                    mean=np.zeros(n_actions), sigma=0.07 * np.ones(n_actions)
                )
                vectorized_noise = VectorizedActionNoise(base_noise, n_envs=n_envs)
                model.policy.action_noise = vectorized_noise
                model.action_noise = vectorized_noise

            else:
                # Define a new model

                # Define the action noise
                n_actions = env.action_space.shape[-1]
                base_noise = NormalActionNoise(
                    mean=np.zeros(n_actions), sigma=0.3 * np.ones(n_actions)
                )
                vectorized_noise = VectorizedActionNoise(base_noise, n_envs=n_envs)
                
                model = TD3(
                    "MlpPolicy",
                    env,
                    learning_rate=0.001,
                    buffer_size=int(1e6),
                    learning_starts=10000,
                    batch_size=256,
                    tau=0.005,
                    gamma=0.99,
                    #train_freq=(32, "step"),
                    #gradient_steps=32,
                    action_noise=vectorized_noise,
                    replay_buffer_class=None,
                    replay_buffer_kwargs=None,
                    optimize_memory_usage=False,
                    policy_delay=2,
                    target_policy_noise=0.2,
                    target_noise_clip=0.5,
                    stats_window_size=100,
                    tensorboard_log=log_dir,
                    policy_kwargs=None,
                    verbose=0,
                    seed=None,
                    device="auto",
                    _init_setup_model=True,
                )
                print("Created a new model.")

            # Save the model when the training is interrupted with signal_handler
            def signal_handler(signal, frame):
                print("Interrupted! Saving model...")
                model.save(model_path)
                env.save(vecnormalize_path)
                env.close()
                print("Saved the current model and environment. Exiting...")
                sys.exit(0)
            signal.signal(signal.SIGINT, signal_handler)

            # Train the model
            try:
                model.learn(
                    total_timesteps=4e6,
                    progress_bar=True,
                    reset_num_timesteps=False,
                    tb_log_name=model_name,
                    # callback=callback
                )
            except KeyboardInterrupt:
                pass

            # Save the models
            model.save(model_path)
            model.save(model_path[:-4]+f"_checkpoint_seed{seed}.zip")  #checkpoints
            env.save(vecnormalize_path)
            env.save(vecnormalize_path[:-4]+f"_checkpoint_seed{seed}.pkl")
            env.close()
            print(f"Finished training for mass {mass} and seed {seed}.")
