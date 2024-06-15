import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import optuna
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
import numpy as np
import gymnasium as gym
from stable_baselines3.common.noise import NormalActionNoise, VectorizedActionNoise
import torch
import torch.optim as optim

import os


def objective(trial):
    # Define the hyperparameter search space
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_categorical("n_steps", [1024, 2048, 4096])  #[1024, 1536, 2048, 2560, 3072, 3584, 4096]
    batch_size = trial.suggest_categorical("batch_size", [7, 11, 13])
    n_epochs = trial.suggest_int("n_epochs", 3, 12)
    #gamma = trial.suggest_float("gamma", 0.9, 0.999)
    gae_lambda = trial.suggest_float("gae_lambda", 0.8, 0.99)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.3)
    vf_coef = trial.suggest_float("vf_coef", 0.5, 1.0)
    ent_coef = trial.suggest_float("ent_coef", 1e-8, 0.1, log=True)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.5, 1.0)

    model_name = f"ppo_walker_trial_{trial.number}_v3.zip"
    model_path = os.path.join("models/hpo_ppo_walker_v3", model_name)
    vecnormalize_path = os.path.join("environments", f"ppo_walker_trial{trial.number}_v3.pkl")
    log_dir = "./logs/hpo_ppo_walker_v2/"

    # Create the environment
    n_envs = 17
    env = make_vec_env(
        "BipedalWalker-v3", 
        n_envs=n_envs, 
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


    # This is our on-policy learner.
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=0.999,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        vf_coef=vf_coef,
        ent_coef=ent_coef,
        max_grad_norm=max_grad_norm,
        verbose=0,
        tensorboard_log=log_dir,
        #use_sde=True,
        seed=2024
    )
    

    model.learn(
        total_timesteps=6e5,
        progress_bar=False,
        tb_log_name=model_name,
        #reset_num_timesteps=True,
    )

    model.save(model_path)
    env.save(vecnormalize_path)
    env.close()

    env = make_vec_env(
        "BipedalWalker-v3", 
        n_envs=1, 
        seed=2024, 
        vec_env_cls=SubprocVecEnv,
        )
    env = VecNormalize.load(vecnormalize_path, env)
    env.training = False  
    env.norm_reward = False

    # Evaluate the model
    total_reward = 0.0
    n_episodes = 10
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward

    avg_reward = total_reward / n_episodes

    return avg_reward


if __name__ == "__main__":

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=1, show_progress_bar=True, n_jobs=3, gc_after_trial=True, timeout=25200)   
    
    # Print the best hyperparameters
    print("="*80)
    print(f"Best hyperparameters: {study.best_params}")
    print(f"Best value: {study.best_value}")
    print(f"Best trial number: {study.best_trial.number}")

    # Save the study
    #study.trials_dataframe().to_csv("ppo_walker_study.csv")


    """
    [I 2024-06-04 09:24:46,628] Trial 88 finished with value: -102.59769883095233 and parameters: {'learning_rate': 1.9048549687349276e-05, 'n_steps': 4096, 'batch_size': 256, 'n_epochs': 9, 'gamma': 0.9164871331861589, 'gae_lambda': 0.8320587769258511, 'clip_range': 0.10959994031871292, 'vf_coef': 0.7343081359148522, 'ent_coef': 0.00010431365509533178, 'max_grad_norm': 0.7425455184418255}. Best is trial 88 with value: -102.59769883095233.
    [I 2024-06-04 11:44:09,240] Trial 112 finished with value: -112.09042315468552 and parameters: {'learning_rate': 1.2397505081060392e-05, 'n_steps': 4096, 'batch_size': 256, 'n_epochs': 8, 'gamma': 0.9003271913433495, 'gae_lambda': 0.8859557426027301, 'clip_range': 0.11473440247065311, 'vf_coef': 0.7555433762792627, 'ent_coef': 9.177555818741966e-07, 'max_grad_norm': 0.8630837665464939}. Best is trial 88 with value: -
    [I 2024-06-04 11:30:11,731] Trial 107 finished with value: -106.09429892434396 and parameters: {'learning_rate': 3.0189220837930454e-05, 'n_steps': 4096, 'batch_size': 256, 'n_epochs': 8, 'gamma': 0.912820600148753, 'gae_lambda': 0.8156868852408922, 'clip_range': 0.12323393152146955, 'vf_coef': 0.7028172756422498, 'ent_coef': 3.944723371129888e-07, 'max_grad_norm': 0.6979539654540983}. Best is trial 88 with value: -102.59769883095233.
    [I 2024-06-04 10:08:35,330] Trial 95 finished with value: -109.6988338892933 and parameters: {'learning_rate': 1.7336731879304704e-05, 'n_steps': 4096, 'batch_size': 256, 'n_epochs': 9, 'gamma': 0.9227805282793814, 'gae_lambda': 0.8199854647718949, 'clip_range': 0.11108753557120031, 'vf_coef': 0.7052001969320962, 'ent_coef': 5.343708112329292e-07, 'max_grad_norm': 0.7098671415624066}. Best is trial 88 with value: -102.59769883095233.
    [I 2024-06-04 08:46:06,487] Trial 82 finished with value: -105.42694432960279 and parameters: {'learning_rate': 1.847516683837326e-05, 'n_steps': 4096, 'batch_size': 256, 'n_epochs': 3, 'gamma': 0.9459653766565994, 'gae_lambda': 0.8345917276812699, 'clip_range': 0.28204684350206466, 'vf_coef': 0.7485809661065143, 'ent_coef': 5.433765088971615e-05, 'max_grad_norm': 0.7079643828943224}. Best is trial 17 with value: -103.31657590495668.
    [I 2024-06-04 07:25:33,307] Trial 61 finished with value: -108.48232663665922 and parameters: {'learning_rate': 1.3249605251802883e-05, 'n_steps': 4096, 'batch_size': 64, 'n_epochs': 7, 'gamma': 0.9196865991738459, 'gae_lambda': 0.8662928675350118, 'clip_range': 0.21767514052614273, 'vf_coef': 0.8361261835229874, 'ent_coef': 0.00018884003139478994, 'max_grad_norm': 0.6644142276723306}. Best is trial 17 with value: -103.31657590495668.
    [I 2024-06-04 06:02:16,285] Trial 52 finished with value: -105.66645505782279 and parameters: {'learning_rate': 7.023144817833303e-05, 'n_steps': 4096, 'batch_size': 128, 'n_epochs': 8, 'gamma': 0.9072483966529401, 'gae_lambda': 0.8751489385334541, 'clip_range': 0.25095792767562863, 'vf_coef': 0.9949993006100347, 'ent_coef': 1.5598480660533988e-06, 'max_grad_norm': 0.5805144887489354}. Best is trial 17 with value: -103.31657590495668.
    [I 2024-06-04 02:38:21,039] Trial 17 finished with value: -103.31657590495668 and parameters: {'learning_rate': 3.351955079836216e-05, 'n_steps': 4096, 'batch_size': 256, 'n_epochs': 10, 'gamma': 0.9020859759979845, 'gae_lambda': 0.8397477382977857, 'clip_range': 0.2944263820613385, 'vf_coef': 0.9026984629440811, 'ent_coef': 0.0006907840024932988, 'max_grad_norm': 0.8378685397997767}. Best is trial 17 with value: -103.31657590495668.
    [I 2024-06-04 01:59:29,393] Trial 13 finished with value: -109.27822489097484 and parameters: {'learning_rate': 1.6749734028299486e-05, 'n_steps': 2048, 'batch_size': 256, 'n_epochs': 4, 'gamma': 0.9183352647195325, 'gae_lambda': 0.8375794648566864, 'clip_range': 0.16070807622789818, 'vf_coef': 0.9575356092065446, 'ent_coef': 8.457206245250578e-05, 'max_grad_norm': 0.7133905603562662}. Best is trial 13 with value: -109.27822489097484.
    [I 2024-06-04 11:30:11,731] Trial 107 finished with value: -106.09429892434396 and parameters: {'learning_rate': 3.0189220837930454e-05, 'n_steps': 4096, 'batch_size': 256, 'n_epochs': 8, 'gamma': 0.912820600148753, 'gae_lambda': 0.8156868852408922, 'clip_range': 0.12323393152146955, 'vf_coef': 0.7028172756422498, 'ent_coef': 3.944723371129888e-07, 'max_grad_norm': 0.6979539654540983}. Best is trial 88 with value: -102.59769883095233.
    """

    """
    [I 2024-06-07 02:04:28,931] Trial 60 finished with value: 251.86264743091812 and parameters: {'learning_rate': 0.00020621210484816964, 'n_steps': 1024, 'batch_size': 128, 'n_epochs': 7, 'gae_lambda': 0.9226888766712824, 'clip_range': 0.24770514302308205, 'vf_coef': 0.9814857185069275, 'ent_coef': 0.004772026711424567, 'max_grad_norm': 0.7913081085260443}. Best is trial 25 with value: 275.4551026342516.
    [I 2024-06-07 02:03:41,004] Trial 58 finished with value: 244.71290984626958 and parameters: {'learning_rate': 0.0002077473846005787, 'n_steps': 1024, 'batch_size': 128, 'n_epochs': 10, 'gae_lambda': 0.9254717286300449, 'clip_range': 0.24460411722507105, 'vf_coef': 0.9764162809905504, 'ent_coef': 2.065990681422602e-07, 'max_grad_norm': 0.839426226463243}. Best is trial 25 with value: 275.4551026342516.
    [I 2024-06-07 02:02:05,340] Trial 59 finished with value: 226.32644068363533 and parameters: {'learning_rate': 0.0001988737518706357, 'n_steps': 1024, 'batch_size': 128, 'n_epochs': 10, 'gae_lambda': 0.924887238622581, 'clip_range': 0.25042314363194634, 'vf_coef': 0.945303586557029, 'ent_coef': 1.92675403819723e-08, 'max_grad_norm': 0.8567517188093834}. Best is trial 25 with value: 275.4551026342516.
    [I 2024-06-07 01:49:45,478] Trial 57 finished with value: 218.83161236858254 and parameters: {'learning_rate': 0.0003302931437281527, 'n_steps': 1024, 'batch_size': 128, 'n_epochs': 8, 'gae_lambda': 0.9211814435130776, 'clip_range': 0.231718679106133, 'vf_coef': 0.9446316056918721, 'ent_coef': 5.151611550439731e-08, 'max_grad_norm': 0.8596227049319631}. Best is trial 25 with value: 275.4551026342516.
    [I 2024-06-07 00:53:29,826] Trial 49 finished with value: 258.2353354187943 and parameters: {'learning_rate': 0.0002815951689212672, 'n_steps': 1024, 'batch_size': 128, 'n_epochs': 9, 'gae_lambda': 0.9381630945094658, 'clip_range': 0.23983764970254579, 'vf_coef': 0.9181516036545054, 'ent_coef': 9.410505732399547e-08, 'max_grad_norm': 0.581215226384458}. Best is trial 25 with value: 275.4551026342516.
    [I 2024-06-07 00:29:07,427] Trial 43 finished with value: 263.00471904921466 and parameters: {'learning_rate': 0.0003224523296216894, 'n_steps': 1024, 'batch_size': 128, 'n_epochs': 9, 'gae_lambda': 0.9646310204704821, 'clip_range': 0.25733105209638113, 'vf_coef': 0.9244399056075016, 'ent_coef': 2.8944656252382256e-08, 'max_grad_norm': 0.5815280755503478}. Best is trial 25 with value: 275.4551026342516.
    [I 2024-06-06 23:11:14,032] Trial 35 finished with value: 215.6400225312603 and parameters: {'learning_rate': 0.00022341984083271615, 'n_steps': 1024, 'batch_size': 128, 'n_epochs': 11, 'gae_lambda': 0.8980198945363849, 'clip_range': 0.2635096441234777, 'vf_coef': 0.9578071082140054, 'ent_coef': 7.360022626865931e-08, 'max_grad_norm': 0.7407092086646897}. Best is trial 25 with value: 275.4551026342516.
    [I 2024-06-06 23:05:33,577] Trial 34 finished with value: 256.9663331893791 and parameters: {'learning_rate': 0.00023780193032013017, 'n_steps': 1024, 'batch_size': 128, 'n_epochs': 8, 'gae_lambda': 0.9308873086445467, 'clip_range': 0.2633438124704594, 'vf_coef': 0.9019861270722463, 'ent_coef': 8.058000867623879e-08, 'max_grad_norm': 0.7340245222007533}. Best is trial 25 with value: 275.4551026342516.
    [I 2024-06-06 22:57:15,663] Trial 33 finished with value: 243.74370539310553 and parameters: {'learning_rate': 0.0002467258275229326, 'n_steps': 1024, 'batch_size': 128, 'n_epochs': 8, 'gae_lambda': 0.8993107661828209, 'clip_range': 0.26146151647602073, 'vf_coef': 0.9009920997706611, 'ent_coef': 3.641654658907957e-08, 'max_grad_norm': 0.8042846504146127}. Best is trial 25 with value: 275.4551026342516.
    [I 2024-06-06 22:39:25,860] Trial 32 finished with value: 241.41345262098025 and parameters: {'learning_rate': 0.00038666955275555023, 'n_steps': 1024, 'batch_size': 128, 'n_epochs': 5, 'gae_lambda': 0.8708633947101831, 'clip_range': 0.23067527620092662, 'vf_coef': 0.95380843224898, 'ent_coef': 9.525937744467367e-07, 'max_grad_norm': 0.9007438742282277}. Best is trial 25 with value: 275.4551026342516.
    [I 2024-06-06 21:20:05,925] Trial 18 finished with value: 244.69321664548687 and parameters: {'learning_rate': 0.00012914758487126934, 'n_steps': 1024, 'batch_size': 128, 'n_epochs': 10, 'gae_lambda': 0.9476652306005665, 'clip_range': 0.25081066940984914, 'vf_coef': 0.9988241546576087, 'ent_coef': 1.6191157720727573e-08, 'max_grad_norm': 0.707289269263472}. Best is trial 18 with value: 244.69321664548687.
    [I 2024-06-06 22:25:02,810] Trial 27 finished with value: 239.55408140806966 and parameters: {'learning_rate': 0.0007201987868350441, 'n_steps': 1024, 'batch_size': 128, 'n_epochs': 8, 'gae_lambda': 0.915388533159873, 'clip_range': 0.2721560387052964, 'vf_coef': 0.8598622164014148, 'ent_coef': 6.051274441858893e-08, 'max_grad_norm': 0.6535318792279359}. Best is trial 25 with value: 275.4551026342516.
    [I 2024-06-06 22:06:27,816] Trial 25 finished with value: 275.4551026342516 and parameters: {'learning_rate': 0.00031016290222399755, 'n_steps': 1024, 'batch_size': 128, 'n_epochs': 8, 'gae_lambda': 0.8610756390880515, 'clip_range': 0.2641177750386453, 'vf_coef': 0.9182477231951106, 'ent_coef': 4.9293649594261e-08, 'max_grad_norm': 0.6337416874273284}. Best is trial 25 with value: 275.4551026342516.
    """