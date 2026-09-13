import json
import numpy as np
from stable_baselines3 import PPO
from src.envs.QuantumRepeaterMARL_NoComm import QuantumRepeaterParallelEnv

import multiprocessing

def run_single_episode(args):
    model_path, p_swap_val, seed = args
    model = PPO.load(model_path)
    env = QuantumRepeaterParallelEnv(n_segments=4, tau_c=10.0, p_swap=p_swap_val, normalized=True)
    obs_dict, _ = env.reset(seed=seed)
    total_reward = 0
    successes = 0
    steps = 0
    link_ages = []

    while steps < env.max_steps:
        action_dict = {}
        for agent in env.agents:
            # deterministic=True ensures the AI uses its best strategy, not random exploration
            action, _ = model.predict(obs_dict[agent], deterministic=True)
            action_dict[agent] = action
            
        obs_dict, reward_dict, term_dict, trunc_dict, info_dict = env.step(action_dict)
        
        if len(reward_dict) > 0:
            total_reward += reward_dict.get("agent_1", 0.0)
        
        steps += 1
        if len(info_dict) > 0 and "Link Age" in info_dict.get("agent_1", {}):
            successes += 1
            link_ages.append(info_dict["agent_1"]["Link Age"])

        # PettingZoo ParallelEnv sets agents list to empty when terminated/truncated
        if not env.agents:
            obs_dict, _ = env.reset()

    return total_reward, successes, link_ages

def evaluate(model_path, p_swap_val, json_filename, episodes=30, num_cores=6):
    try:
        _ = PPO.load(model_path)
    except Exception as e:
        print(f"\n[!] Could not load {model_path}. Make sure training for this phase is finished.")
        print("========================================================\n")
        return

    print(f"Evaluating {model_path} (p_swap={p_swap_val}) across {num_cores} cores...")
    
    args_list = [(model_path, p_swap_val, seed) for seed in range(episodes)]
    
    episode_rewards = []
    success_counts = []
    all_link_ages = []
    
    # Spawn background processes to run episodes in parallel!
    with multiprocessing.Pool(num_cores) as pool:
        # imap_unordered allows us to process results as soon as each background worker finishes!
        results = []
        for i, res in enumerate(pool.imap_unordered(run_single_episode, args_list), 1):
            results.append(res)
            # Print a live updating progress bar
            print(f"\rCompleted {i}/{episodes} episodes...", end="", flush=True)
            
    print() # Move to the next line when the progress bar finishes
        
    for total_reward, successes, link_ages in results:
        episode_rewards.append(total_reward)
        success_counts.append(successes)
        all_link_ages.extend(link_ages)

    print(f"--- Results for {model_path} ---")
    print(f"Mean reward per 10^5 steps: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Mean successes per 10^5 steps: {np.mean(success_counts):.2f}")
    if all_link_ages:
        print(f"Mean end-to-end link age: {np.mean(all_link_ages):.2f} ± {np.std(all_link_ages):.2f}")
        # Convert numpy floats to standard Python floats
        output_data = {
            "ages": [float(x) for x in all_link_ages],
            "rewards": [float(x) for x in episode_rewards],
            "successes": [float(x) for x in success_counts]
        }
        with open(json_filename, 'w') as f:
            json.dump(output_data, f)
        print(f"Saved comprehensive metrics (Ages, Rewards, Successes) to {json_filename}")
    else:
        print("No successful end-to-end entanglements observed.")
    print("========================================================\n")

if __name__ == "__main__":
    print("\n========================================================")
    print("TEST 1: The 'Base' Agent in a Perfect Environment")
    print("Testing if it learned to use communication perfectly when hardware is 100% reliable.")
    print("========================================================")
    evaluate("models/marl_nocomm_agent/phase1/nocomm_agent_phase1", 1.0, "data/nocomm_test1_ages.json")
    
    print("========================================================")
    print("TEST 2: The 'Base' Agent in the Imperfect Environment")
    print("Testing if an AI trained in a perfect simulation breaks when faced with 50% swap failures.")
    print("========================================================")
    evaluate("models/marl_nocomm_agent/phase1/nocomm_agent_phase1", 0.5, "data/nocomm_test2_ages.json")

    print("========================================================")
    print("TEST 3: The Curriculum Agent in the Imperfect Environment")
    print("Testing the final agent that was forced to adapt to the 50% swap failures.")
    print("========================================================")
    evaluate("models/marl_nocomm_agent/phase2/nocomm_agent", 0.5, "data/nocomm_test3_ages.json")
