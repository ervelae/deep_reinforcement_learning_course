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
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Retrieve training reward
            x, y = self.model.env.get_attr('rewards')
            mean_reward = np.mean(y[-100:])
            if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

            # New best model, save the model
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                if self.verbose > 0:
                    print(f"Saving new best model to {self.save_path}")
                self.model.save(self.save_path)

        return True

if __name__ == "__main__":

    for mass in [6]:
        for seed in [27]:
            
            model_name = f"sac_hopper_mass{mass}_v7.zip"
            model_path = os.path.join("models", model_name)

            log_dir = "./logs/sac_hopper/"
            callback = SaveOnBestTrainingRewardCallback(check_freq=10000, log_dir=log_dir+model_name)

            n_envs = 32
            env = make_vec_env(
                "Hopper-v4",
                n_envs=n_envs,
                seed=seed,
                vec_env_cls=SubprocVecEnv,
                wrapper_class=ChangeMassWrapper,
                wrapper_kwargs=dict(torso_mass=mass),
                env_kwargs = {"healthy_reward": 0},
            )
            env = VecNormalize(env, norm_obs=True, norm_reward=True)

            if os.path.exists(model_path):
                # Load the saved model
                model = SAC.load(model_path, env=env)
                print("Loaded existing model.")

                # Changes to the model:
                # More action noise
                n_actions = env.action_space.shape[-1]
                base_noise = NormalActionNoise(
                    mean=np.zeros(n_actions), sigma = 0.3 * np.ones(n_actions)
                )
                #vectorized_noise = VectorizedActionNoise(base_noise, n_envs=n_envs)
                #model.action_noise = vectorized_noise

                # More gamma
                #model.gamma = 0.98

                # More learning rate
                #model.learning_rate = 0.00073

                # More frequen model training
                #model.train_freq = TrainFreq(64, TrainFrequencyUnit.STEP)
                
                # More batch size
                #model.batch_size = 256*3

                # Buffer size
                #model.buffer_size = int(3e5)
                
                #model.learning_starts = 1e4
                
                #model.gradient_steps = 32
                #model.target_update_interval = 64
                #model.ent_coef = 0.001
                #model.sde_sample_freq = -1
                #model.buffer_size = int(1e6)
                #gSDE (generalized State-Dependent Exploration)


            else:
                # Define a new model

                # Define the action noise
                n_actions = env.action_space.shape[-1]
                base_noise = NormalActionNoise(
                    mean=np.zeros(n_actions), sigma=0.6 * np.ones(n_actions)
                )
                # Create vectorized action noise
                vectorized_noise = VectorizedActionNoise(base_noise, n_envs=n_envs)

                # Define the policy kwargs to use AMSGrad
                policy_kwargs = dict(
                    optimizer_class=optim.Adam, #optimizer_kwargs=dict(amsgrad=True),
                    #net_arch=[256, 256],  # Define the network architecture
                    use_sde=True,  # Enable gSDE
                    log_std_init=-3,  # Initial value for log standard deviation
                )

                model = SAC(
                    "MlpPolicy",
                    env,
                    learning_rate=0.00073,  # Slightly higher so doesnt get stuck in local minima
                    buffer_size=int(1e6),
                    learning_starts=10000,
                    batch_size=256,
                    tau=0.02,
                    gamma=0.98,
                    train_freq=(256, "step"),  # To make it more computationally efficient
                    gradient_steps=64,
                    action_noise=vectorized_noise,
                    replay_buffer_class=None,
                    replay_buffer_kwargs=None,
                    optimize_memory_usage=False,
                    ent_coef='auto',
                    target_update_interval=100, # This higher
                    target_entropy="auto",
                    use_sde=True,  # Turn to true
                    sde_sample_freq=8, # from -1 to 8
                    use_sde_at_warmup=False,
                    stats_window_size=100,
                    tensorboard_log=log_dir,
                    #policy_kwargs=policy_kwargs,
                    verbose=0,
                    # seed=1,
                    device="auto",
                    _init_setup_model=True,
                )
                print("Created a new model.")


            def signal_handler(signal, frame):
                print("Interrupted! Saving model...")
                model.save(model_path)
                env.save(f"models/Hopper-v4_vecnormalize_mass{mass}_cp{seed}.pkl")
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
                    #callback=callback
                )
            except KeyboardInterrupt:
                pass

            model.save(model_path)
            model.save(f"models/Hopper-v4_vecnormalize_mass{mass}_cp{seed}.pkl")
            env.close()
            print(f"Finished training for mass {mass} and seed {seed}.")
        env.save(f"environments/Hopper-v4_vecnormalize_mass{mass}_final.pkl")
