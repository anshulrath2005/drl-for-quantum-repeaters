import os
from stable_baselines3 import PPO
import supersuit as ss
from QuantumRepeaterMARL import QuantumRepeaterParallelEnv

log_dir = "./logs_marl/"
os.makedirs(log_dir, exist_ok=True)

# 1. Instantiate the PettingZoo Parallel Environment
env = QuantumRepeaterParallelEnv(n_segments=4, tau_c=10.0)

# 2. The Parameter Sharing Trick!
# This wrapper takes the dictionary outputs of PettingZoo and flattens them into a Vector
env = ss.pettingzoo_env_to_vec_env_v1(env)

# This wrapper tells Stable-Baselines3 to treat the 3 flattened agents as 3 parallel single-player games
env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=1, base_class='stable_baselines3')

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=4e-4,
    gamma=1.0,
    n_steps=8000,
    batch_size=250,
    policy_kwargs=dict(net_arch=dict(pi=[32, 32], vf=[32, 32])),
    tensorboard_log=log_dir
)

print("Decentralized MARL Training started... You can monitor progress via TensorBoard.")
model.learn(total_timesteps=30000000, tb_log_name="PPO_MARL_Quantum_Repeater")
model.save("quantum_repeater_marl_agent")
