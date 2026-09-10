import numpy as np
from stable_baselines3 import PPO
from src.envs.QuantumRepeater import QuantumRepeaterEnv

model = PPO.load("models/single_agent/quantum_repeater_agent")

def PPO_policy(obs):
    action, _ = model.predict(obs, deterministic=True)
    return action

episodes = 100

env = QuantumRepeaterEnv(
    n_segments=4,
    tau_c=10.0,
    normalized=True
)

episode_rewards = []
success_counts = []
link_ages = []

for seed in range(episodes):
    print(f"\rEvaluating run {seed + 1}/{episodes}...", end="", flush=True)
    obs, _ = env.reset(seed=seed)
    total_reward = 0
    successes = 0
    steps = 0

    while steps < env.max_steps:
        action = PPO_policy(obs)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        if "Link Age" in info:
            successes += 1
            link_ages.append(info["Link Age"])

        if done or truncated:
            obs, _ = env.reset()

    episode_rewards.append(total_reward)
    success_counts.append(successes)

print()
print("PPO Agent")
print(f"Mean reward per 10^5 steps: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
print(f"Mean successes per 10^5 steps: {np.mean(success_counts):.2f}")
if link_ages:
    print(f"Mean end-to-end link age: {np.mean(link_ages):.2f} ± {np.std(link_ages):.2f}")
else:
    print("No successful end-to-end entanglements observed.")