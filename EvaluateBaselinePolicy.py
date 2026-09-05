from MemoryCutoffPolicy import baseline_policy
import numpy as np
from stable_baselines3 import PPO
from QuantumRepeater import QuantumRepeaterEnv

episodes = 50

env = QuantumRepeaterEnv(
    n_segments=4,
    tau_c=10.0,
    normalized=False
)

episode_rewards = []
success_counts = []
link_ages = []

for seed in range(episodes):
    obs, _ = env.reset(seed=seed)
    total_reward = 0
    successes = 0

    for _ in range(env.max_steps):
        action = baseline_policy(obs)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

        if "Link Age" in info:
            successes += 1
            link_ages.append(info["Link Age"])

        if done or truncated:
            break

    episode_rewards.append(total_reward)
    success_counts.append(successes)

print("Memory-Cutoff Baseline")
print(f"Mean episode reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
print(f"Mean successes per episode: {np.mean(success_counts):.2f}")
if link_ages:
    print(f"Mean end-to-end link age: {np.mean(link_ages):.2f}")
else:
    print("No successful end-to-end entanglements observed.")