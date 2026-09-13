import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize
import supersuit as ss
from src.envs.QuantumRepeaterMARL import QuantumRepeaterParallelEnv

# ==========================================
# CURRICULUM LEARNING HYPERPARAMETERS
# ==========================================
PHASE_1_STEPS = 3000000   # Fast Trial (3 Million)
PHASE_2_STEPS = 6000000   # Fast Trial (6 Million)
NUM_CORES = 6             # Sweet spot (leaves 2 cores for macOS)

# VERY IMPORTANT: When using multiprocessing on macOS, everything must be inside this block!
if __name__ == "__main__":
    log_dir = "./logs/logs_marl/"
    os.makedirs(log_dir, exist_ok=True)

    print("======================================================")
    print(f"PHASE 1: Base (p_swap = 1.0)")
    print(f"Running on {NUM_CORES} CPU Cores!")
    print("======================================================")

    # 1. Instantiate Phase 1 Environment
    env_phase1 = QuantumRepeaterParallelEnv(n_segments=4, tau_c=10.0, p_swap=1.0)
    env_phase1 = ss.pettingzoo_env_to_vec_env_v1(env_phase1)
    env_phase1 = ss.concat_vec_envs_v1(env_phase1, num_vec_envs=NUM_CORES, num_cpus=NUM_CORES, base_class='stable_baselines3')
    env_phase1 = VecMonitor(env_phase1)
    # VecNormalize dynamically scales the +0.1 breadcrumbs and the +500 jackpot so the Neural Network doesn't crash!
    env_phase1 = VecNormalize(env_phase1, norm_obs=False, norm_reward=True, clip_reward=10.0)

    # Initialize the shared brain
    # NOTE: The true rollout buffer size is n_steps * (n_agents * NUM_CORES). 1000 * 18 = 18,000 experiences per update.
    model = PPO(
        "MlpPolicy",
        env_phase1,
        verbose=1,
        learning_rate=4e-4,
        gamma=0.999,
        n_steps=1000, 
        batch_size=250,
        policy_kwargs=dict(net_arch=dict(pi=[32, 32], vf=[32, 32])),
        tensorboard_log=log_dir
    )

    # --------------------------------------------------------
    # THE ACTION SPACE CHAOS FIX:
    # Bias the final layer so the agents default to outputting '0' (doing nothing) instead of randomly destroying qubits!
    import torch
    with torch.no_grad():
        model.policy.action_net.bias.fill_(-2.0)
    # --------------------------------------------------------

    # Train Phase 1 (Commented out to save you 18 minutes! We will just load the saved model)
    # model.learn(total_timesteps=PHASE_1_STEPS, tb_log_name="PPO_MARL_Phase1_Perfect_Hardware")
    # model.save("models/marl_comm_agent/phase1/quantum_repeater_marl_agent_phase1")
    # env_phase1.save("models/marl_comm_agent/phase1/vec_normalize_phase1.pkl")
    
    print("Loading the successfully completed Phase 1 agent from disk...")
    model = PPO.load("models/marl_comm_agent/phase1/quantum_repeater_marl_agent_phase1", env=env_phase1)
    env_phase1 = VecNormalize.load("models/marl_comm_agent/phase1/vec_normalize_phase1.pkl", env_phase1)

    print("\n======================================================")
    print("PHASE 2: THE REAL WORLD (p_swap = 0.5)")
    print("Injecting 50% hardware failure rate to test adaptability.")
    print("======================================================")

    # 2. Instantiate Phase 2 Environment
    env_phase2 = QuantumRepeaterParallelEnv(n_segments=4, tau_c=10.0, p_swap=0.5)
    env_phase2 = ss.pettingzoo_env_to_vec_env_v1(env_phase2)
    env_phase2 = ss.concat_vec_envs_v1(env_phase2, num_vec_envs=NUM_CORES, num_cpus=NUM_CORES, base_class='stable_baselines3')
    env_phase2 = VecMonitor(env_phase2)
    
    # Load the running reward averages from Phase 1 into Phase 2
    env_phase2 = VecNormalize(env_phase2, norm_obs=False, norm_reward=True, clip_reward=10.0)
    # Sync the running reward stats from phase 1 to phase 2 (Skipping obs_rms since norm_obs=False)
    env_phase2.ret_rms = env_phase1.ret_rms

    # Hot-swap the new noisy environment into the already-trained brain
    model.set_env(env_phase2)

    # Train Phase 2
    model.learn(total_timesteps=PHASE_2_STEPS, tb_log_name="PPO_MARL_Phase2_Noisy_Hardware", reset_num_timesteps=False)

    # Save the final Master agent
    model.save("models/marl_comm_agent/phase2/quantum_repeater_marl_agent")
    env_phase2.save("models/marl_comm_agent/phase2/vec_normalize_phase2.pkl")
    print("\nTraining Complete! The master curriculum agent has been saved.")
    
    # CLEANUP: Kill the background CPU workers so the terminal doesn't hang!
    env_phase1.close()
    env_phase2.close()
