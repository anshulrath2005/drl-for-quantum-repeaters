import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from QuantumRepeater import QuantumRepeaterEnv

log_dir = "./logs/"
os.makedirs(log_dir, exist_ok=True)

env = make_vec_env(lambda: Monitor(QuantumRepeaterEnv(n_segments=4, tau_c=10.0), log_dir))

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

model.learn(total_timesteps=8000000, tb_log_name="PPO_Quantum_Repeater")
model.save("quantum_repeater_agent")