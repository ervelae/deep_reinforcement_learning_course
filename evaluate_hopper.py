from stable_baselines3 import SAC
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import VecNormalize
import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import numpy as np

# Custom wrapper to change the mass of the torso
class ChangeMassWrapper(gym.Wrapper):
    def __init__(self, env, torso_mass):
        super().__init__(env)
        self.torso_mass = torso_mass
        self.env.unwrapped.model.body_mass[1] = self.torso_mass
        print(f"Changed torso mass to {self.torso_mass}")

if __name__ == '__main__':
    for train_mass in [6]:
        for mass in [6]:
            env = make_vec_env(
                "Hopper-v4", 
                n_envs=1, 
                #seed=9, 
                vec_env_cls=SubprocVecEnv, 
                wrapper_class=ChangeMassWrapper, 
                wrapper_kwargs=dict(torso_mass=mass),
                #env_kwargs = {"terminate_when_unhealthy": False},
            )
            #env = VecNormalize(env, norm_obs=True, norm_reward=True)
            env = VecNormalize.load(f"environments/td3_hopper_mass{train_mass}_v3.pkl", env)
            env.training = False
            env.norm_reward = False

            # Load the trained model
            model = TD3.load(f"models/td3_hopper_mass{train_mass}_v3.zip", env=env)

            n_episodes = 2  # Number of episodes to evaluate
            total_rewards = []

            for episode in range(n_episodes):
                obs = env.reset()
                done = False
                episode_reward = 0

                while not done:
                    action, _states = model.predict(obs, deterministic=True)  # No action noise
                    obs, reward, done, info = env.step(action)
                    episode_reward += reward
                    env.render(mode='human')

                total_rewards.append(episode_reward)
                print(f"Episode {episode + 1}: Total reward: {episode_reward}")


            #np.save(f"assigment/total_rewards_mass{train_mass}_{mass}.npy", total_rewards)
            mean_reward = np.mean(total_rewards)
            std_reward = np.std(total_rewards)


            print(f"Mass {mass}. Total mean reward over {n_episodes} episodes: {mean_reward} ± {std_reward}")
            env.reset()
            env.close()