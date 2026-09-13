import json
import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu

def load_data(filename):
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def test_metric(metric_name, p_scratch_arr, p_curr_arr, lower_is_better=False):
    if not p_scratch_arr or not p_curr_arr:
        print(f"  [{metric_name}] Not enough data to test.")
        return
        
    print(f"\n--- Metric: {metric_name.upper()} ---")
    print(f"From-Scratch Agent (Test 4): Mean = {np.mean(p_scratch_arr):.2f} ± {np.std(p_scratch_arr):.2f} (N={len(p_scratch_arr)})")
    print(f"Curriculum Agent   (Test 3): Mean = {np.mean(p_curr_arr):.2f} ± {np.std(p_curr_arr):.2f} (N={len(p_curr_arr)})")
    
    t_stat, p_val_t = ttest_ind(p_curr_arr, p_scratch_arr, equal_var=False)
    
    try:
        u_stat, p_val_u = mannwhitneyu(p_curr_arr, p_scratch_arr, alternative='two-sided')
    except Exception as e:
        u_stat, p_val_u = 0.0, 1.0 
    
    print(f"Welch's T-Test:       p-value = {p_val_t:.4e}")
    print(f"Mann-Whitney U Test:  p-value = {p_val_u:.4e}")
    
    alpha = 0.05
    if p_val_u < alpha:
        p_curr_mean = np.mean(p_curr_arr)
        p_scratch_mean = np.mean(p_scratch_arr)
        
        better = (p_curr_mean < p_scratch_mean) if lower_is_better else (p_curr_mean > p_scratch_mean)
        if better:
            print("Result: CURRICULUM is STATISTICALLY SIGNIFICANTLY BETTER.")
        else:
            print("Result: CURRICULUM is STATISTICALLY SIGNIFICANTLY WORSE.")
    else:
        print("Result: NOT SIGNIFICANT (No statistical difference detected).")

def run_curriculum_test():
    print(f"\n========================================================")
    print(f"CURRICULUM LEARNING HYPOTHESIS TEST")
    print(f"Comparing Comm Agent: Curriculum (Test 3) vs From-Scratch (Test 4) on Noisy Hardware")
    print(f"========================================================")
    
    p_scratch_file = "data/comm_test4_ages.json"
    p_curr_file = "data/comm_test3_ages.json"
    
    p_scratch_data = load_data(p_scratch_file)
    p_curr_data = load_data(p_curr_file)
    
    if p_scratch_data is None or p_curr_data is None:
        print(f"Data files not found. Ensure evaluation data exists in data/ directory.")
        return
        
    test_metric("Total Reward (Secret Key Rate)", p_scratch_data.get("rewards", []), p_curr_data.get("rewards", []), lower_is_better=False)
    test_metric("Throughput (Successes per Episode)", p_scratch_data.get("successes", []), p_curr_data.get("successes", []), lower_is_better=False)
    test_metric("Fidelity Preservation (Link Age)", p_scratch_data.get("ages", []), p_curr_data.get("ages", []), lower_is_better=True)
    print("\n")

if __name__ == "__main__":
    run_curriculum_test()
