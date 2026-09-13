import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize
import supersuit as ss
from src.envs.QuantumRepeaterMARL import QuantumRepeaterParallelEnv

TOTAL_STEPS = 9000000   # 9 Million steps (Total of Phase 1 + Phase 2 to make it a fair baseline comparison)
NUM_CORES = 6

if __name__ == "__main__":
    log_dir = "./logs/logs_marl/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs("models/marl_comm_agent/noisy_baseline", exist_ok=True)

    print("======================================================")
    print(f"NOISY BASELINE TRAINING: TRAINING ON p_swap=0.5 FROM SCRATCH")
    print(f"Total Timesteps: {TOTAL_STEPS}")
    print(f"Running on {NUM_CORES} CPU Cores!")
    print("======================================================")

    env = QuantumRepeaterParallelEnv(n_segments=4, tau_c=10.0, p_swap=0.5)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=NUM_CORES, num_cpus=NUM_CORES, base_class='stable_baselines3')
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=4e-4,
        gamma=0.999,
        n_steps=1000, 
        batch_size=250,
        policy_kwargs=dict(net_arch=dict(pi=[32, 32], vf=[32, 32])),
        tensorboard_log=log_dir
    )

    with torch.no_grad():
        model.policy.action_net.bias.fill_(-2.0)

    model.learn(total_timesteps=TOTAL_STEPS, tb_log_name="PPO_MARL_Noisy_From_Scratch")

    model.save("models/marl_comm_agent/noisy_baseline/quantum_repeater_marl_agent")
    env.save("models/marl_comm_agent/noisy_baseline/vec_normalize_noisy_baseline.pkl")
    
    print("\nTraining Complete! The Noisy From Scratch baseline agent has been saved.")
    env.close()
