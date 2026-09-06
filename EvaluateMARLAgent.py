import numpy as np
from stable_baselines3 import PPO
from QuantumRepeaterMARL import QuantumRepeaterParallelEnv

model = PPO.load("quantum_repeater_marl_agent")

episodes = 50
env = QuantumRepeaterParallelEnv(n_segments=4, tau_c=10.0, normalized=True)

episode_rewards = []
success_counts = []
link_ages = []

for seed in range(episodes):
    print(f"\rEvaluating MARL run {seed + 1}/{episodes}...", end="", flush=True)
    obs_dict, _ = env.reset(seed=seed)
    total_reward = 0
    successes = 0
    steps = 0

    while steps < env.max_steps:
        action_dict = {}
        
        # Each agent looks through its own eyes (obs_dict[agent]) and uses the shared brain to pick an action
        for agent in env.agents:
            obs_array = obs_dict[agent]
            action, _ = model.predict(obs_array, deterministic=True)
            action_dict[agent] = action
            
        obs_dict, reward_dict, term_dict, trunc_dict, info_dict = env.step(action_dict)
        
        # Because the reward is broadcast to everyone, we can just log Agent 1's reward as the team score
        if len(reward_dict) > 0:
            total_reward += reward_dict["agent_1"]
        
        steps += 1

        if len(info_dict) > 0 and "Link Age" in info_dict.get("agent_1", {}):
            successes += 1
            link_ages.append(info_dict["agent_1"]["Link Age"])

        # PettingZoo ParallelEnv sets agents list to empty when terminated/truncated
        if not env.agents:
            obs_dict, _ = env.reset()

    episode_rewards.append(total_reward)
    success_counts.append(successes)

print()
print("Decentralized MARL PPO Agent")
print(f"Mean reward per 10^5 steps: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
print(f"Mean successes per 10^5 steps: {np.mean(success_counts):.2f}")
if link_ages:
    print(f"Mean end-to-end link age: {np.mean(link_ages):.2f} ± {np.std(link_ages):.2f}")
else:
    print("No successful end-to-end entanglements observed.")
