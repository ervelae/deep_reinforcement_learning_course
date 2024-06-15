import optuna
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import SAC
import os
import numpy as np
import gymnasium as gym
from stable_baselines3.common.noise import NormalActionNoise, VectorizedActionNoise


def objective(trial):
    # Define the hyperparameter search space
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    buffer_size = trial.suggest_int("buffer_size", 100000, 1000000)    
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    tau = trial.suggest_float("tau", 0.004, 0.02)
    #gamma = trial.suggest_float("gamma", 0.9, 0.999)
    train_freq = trial.suggest_categorical("train_freq", [1, 8, 16])
    gradient_steps = trial.suggest_int("gradient_steps", 1, 16)
    ent_coef = trial.suggest_categorical("ent_coef", ['auto', 0.1, 0.01, 0.001])
    #vf_coef = trial.suggest_float("vf_coef", 0.3, 1.0)
    #max_grad_norm  = trial.suggest_float("max_grad_norm ", 0.3, 1.0)

    #learning_starts = trial.suggest_int("learning_starts", 1000, 10000)

    model_name = f"sac_walker_trial_{trial.number}.zip"
    model_path = os.path.join("models", model_name)
    vecnormalize_path = os.path.join("environments", f"sac_walker_trial{trial.number}_v1.pkl")
    log_dir = "./logs/sac_walker/"

    # Create the environment
    n_envs = 16
    env = make_vec_env(
        "BipedalWalker-v3", 
        n_envs=n_envs, 
        seed=2024, 
        vec_env_cls=SubprocVecEnv
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

    # Define the action noise
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions))

    # Initialize the SAC model with the suggested hyperparameters
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        tau=tau,
        gamma=gamma,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        ent_coef=ent_coef,
        learning_starts=int(1e4),
        verbose=0,
        tensorboard_log=log_dir,
        use_sde=True, 
        sde_sample_freq=-1, 
        seed=2024
    )

    model.learn(total_timesteps=6e5, progress_bar=False, tb_log_name=model_name)

    # Save the normalization statistics
    env.save(vecnormalize_path)
    env.close()

    # Create a new environment for evaluation
    eval_env = make_vec_env("BipedalWalker-v3", n_envs=1, seed=2024, vec_env_cls=SubprocVecEnv)
    eval_env = VecNormalize.load(vecnormalize_path, eval_env)
    eval_env.training = False  # Set to evaluation mode
    eval_env.norm_reward = False

    # Evaluate the model
    total_reward = 0.0
    n_episodes = 10
    for _ in range(n_episodes):
        obs = eval_env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, info = eval_env.step(action)
            total_reward += reward

    avg_reward = total_reward / n_episodes
    eval_env.close

    return avg_reward

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=1000, show_progress_bar=True, n_jobs=-1, gc_after_trial=True, timeout=43200)   
    
    # Print the best hyperparameters
    print("="*80)
    print(f"Best hyperparameters: {study.best_params}")
    print(f"Best value: {study.best_value}")
    print(f"Best trial number: {study.best_trial.number}")

    # Save the study
    study.trials_dataframe().to_csv("sac_walker_study.csv")

    """
    Trial 28 finished with value: 292.19811625498033 and parameters: {'learning_rate': 0.000989437903491443, 'buffer_size': 330729, 'batch_size': 256, 'tau': 0.017436953222665016, 'gamma': 0.9323973846666475, 'train_freq': 1, 'gradient_steps': 1, 'ent_coef': 'auto'}. Best is trial 28 with value: 292.19811625498033.
    Trial 29 finished with value: 284.1877076136626 and parameters: {'learning_rate': 0.0004024195553420781, 'buffer_size': 323633, 'batch_size': 256, 'tau': 0.01803798984170235, 'gamma': 0.9344511007686334, 'train_freq': 1, 'gradient_steps': 1, 'ent_coef': 'auto'}. Best is trial 28 with value: 292.19811625498033.
    Trial 39 finished with value: 239.3368082100999 and parameters: {'learning_rate': 0.0006217874405073876, 'buffer_size': 219758, 'batch_size': 256, 'tau': 0.0165685456099668, 'gamma': 0.9313076714120362, 'train_freq': 1, 'gradient_steps': 2, 'ent_coef': 'auto'}. Best is trial 28 with value: 292.19811625498033.
    Trial 38 finished with value: 284.4026509932643 and parameters: {'learning_rate': 0.0006129386160183833, 'buffer_size': 215097, 'batch_size': 256, 'tau': 0.01667043562506057, 'gamma': 0.9275519127092472, 'train_freq': 1, 'gradient_steps': 2, 'ent_coef': 'auto'}. Best is trial 28 with value: 292.19811625498033.
    Trial 20 finished with value: 318.51089287815773 and parameters: {'learning_rate': 7.613357745707285e-05, 'buffer_size': 952203, 'batch_size': 256, 'tau': 0.014397519335421863, 'gamma': 0.9660841833980763, 'train_freq': 1, 'gradient_steps': 13, 'ent_coef': 0.001}. Best is trial 20 with value: 318.51089287815773.
    """