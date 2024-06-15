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



# Custom wrapper to change the mass of the torso.
class ChangeMassWrapper(gym.Wrapper):
    def __init__(self, env, torso_mass):
        super().__init__(env)
        self.torso_mass = torso_mass
        self.env.unwrapped.model.body_mass[1] = self.torso_mass


if __name__ == "__main__":
    env = make_vec_env(
        "Hopper-v4",
        n_envs=4,
        seed=1,
        vec_env_cls=SubprocVecEnv,
        wrapper_class=ChangeMassWrapper,
        wrapper_kwargs=dict(torso_mass=6),
    )
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    # Check if a saved model exists
    model_path = "hopper_mass_6.zip"
    if os.path.exists(model_path):
        # Load the saved model
        model = TD3.load(model_path, env=env)
        print("Loaded existing model.")

        # Change noise parameter
        n_actions = env.action_space.shape[-1]
        base_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        vectorized_noise = VectorizedActionNoise(base_noise, n_envs=4)
        model.action_noise = vectorized_noise

        # Change learning_starts
        model.learning_starts = 1e3

        # Change train_freq        
        model.train_freq = TrainFreq(1, TrainFrequencyUnit.STEP)

        # Learning rate
        = 0.0003
    else:
        # Define the action noise
        n_actions = env.action_space.shape[-1]
        base_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        # Create vectorized action noise
        vectorized_noise = VectorizedActionNoise(base_noise, n_envs=4)


        # Create a new model
        model = TD3(
            "MlpPolicy",
            env,
            learning_rate=0.001, # How big the updates are to the actor. 
            buffer_size=1e6, # How many samples are stored in the replay buffer.
            learning_starts=1e3, # How many steps of the model to collect data before learning starts.
            batch_size=256, # How many samples are used for each gradient update.
            tau=0.005, # How much we update the target network with the weights of the main network.
            gamma=0.99, # How much we value future rewards.
            train_freq=(5, "step"), 
            gradient_steps=1, 
            action_noise=vectorized_noise, 
            replay_buffer_class=None,
            replay_buffer_kwargs=None,
            optimize_memory_usage=False,
            policy_delay=2, # How many critic updates for every actor update.
            target_policy_noise=0.2,
            target_noise_clip=0.5,
            stats_window_size=10000,
            tensorboard_log="./logs/hopper_mass_6/",
            policy_kwargs=None,
            verbose=0,
            seed=None,
            device="auto",
            _init_setup_model=True,
        )
        print("Created a new model.")

    model.learn(
        total_timesteps=1e6, progress_bar=True, reset_num_timesteps=False,
    )  # Adjust the number of timesteps as needed

    model.save("hopper_mass_6")
