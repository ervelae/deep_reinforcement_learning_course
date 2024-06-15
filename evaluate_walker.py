from stable_baselines3 import SAC
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import os



if __name__ == '__main__':

    model_name = f"sac_walker_v2.zip"
    model_path = os.path.join("models", model_name)
    vecnormalize_path = os.path.join("environments", f"sac_walker_v2.pkl")

    env = make_vec_env(
        "BipedalWalker-v3", 
        n_envs=1, 
        #seed=1, 
        vec_env_cls=SubprocVecEnv, 
    )
    #env = VecNormalize(env, norm_obs=True, norm_reward=True)
    env = VecNormalize.load(vecnormalize_path, env)
    env.training = False
    env.norm_reward = False

    # Load the trained model
    model = SAC.load(model_path, env=env)

    # Evaluate the model
    obs = env.reset()
    done = False
    total_reward = 0

    n_episodes = 0

    while not done:
    #for i in range(500):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        #if done: n_episodes += 1
        total_reward += reward
        env.render(mode='human')

    print(f"Total mean reward: {total_reward}")

    env.close()