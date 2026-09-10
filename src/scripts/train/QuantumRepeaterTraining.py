import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from src.envs.QuantumRepeater import QuantumRepeaterEnv

log_dir = "./logs/logs_single/"
os.makedirs(log_dir, exist_ok=True)

env = make_vec_env(lambda: Monitor(QuantumRepeaterEnv(n_segments=4, tau_c=10.0), log_dir))

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=4e-4,     # alpha_pi
    gamma=1.0,              # No discounting
    n_steps=8000,           # Horizon T
    batch_size=250,         # Standard optimization batch
    policy_kwargs=dict(net_arch=dict(pi=[32, 32], vf=[32, 32])), # Network architecture
    tensorboard_log=log_dir
)

print("Training started... You can monitor progress via TensorBoard.")
model.learn(total_timesteps=30000000, tb_log_name="PPO_Quantum_Repeater")
model.save("models/single_agent/quantum_repeater_agent")