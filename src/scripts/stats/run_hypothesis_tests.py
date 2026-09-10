import json
import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu

def load_data(filename):
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def test_metric(metric_name, comm_arr, nocomm_arr):
    if not comm_arr or not nocomm_arr:
        print(f"  [{metric_name}] Not enough data to test.")
        return
        
    print(f"\n--- Metric: {metric_name.upper()} ---")
    print(f"Comm Agent:   Mean = {np.mean(comm_arr):.2f} ± {np.std(comm_arr):.2f} (N={len(comm_arr)})")
    print(f"NoComm Agent: Mean = {np.mean(nocomm_arr):.2f} ± {np.std(nocomm_arr):.2f} (N={len(nocomm_arr)})")
    
    # Welch's T-Test
    t_stat, p_val_t = ttest_ind(comm_arr, nocomm_arr, equal_var=False)
    # Mann-Whitney U Test
    try:
        u_stat, p_val_u = mannwhitneyu(comm_arr, nocomm_arr, alternative='two-sided')
    except Exception as e:
        u_stat, p_val_u = 0.0, 1.0 # Handle identical arrays or extreme cases gracefully
    
    print(f"Welch's T-Test:       p-value = {p_val_t:.4e}")
    print(f"Mann-Whitney U Test:  p-value = {p_val_u:.4e}")
    
    alpha = 0.05
    if p_val_u < alpha:
        print(f"Result: STATISTICALLY SIGNIFICANT")
    else:
        print(f"Result: NOT SIGNIFICANT")

def run_test(case_name, comm_file, nocomm_file):
    print(f"\n========================================================")
    print(f"CASE: {case_name}")
    print(f"========================================================")
    
    comm_data = load_data(comm_file)
    nocomm_data = load_data(nocomm_file)
    
    if comm_data is None or nocomm_data is None:
        print(f"Data files not found for {case_name}. Ensure evaluation finished.")
        return
        
    # Check if old format (list) or new format (dict)
    if isinstance(comm_data, list):
        print("\n[Old JSON format detected (list of ages only)]")
        test_metric("Link Age", comm_data, nocomm_data)
    else:
        test_metric("Total Reward", comm_data.get("rewards", []), nocomm_data.get("rewards", []))
        test_metric("Throughput (Successes)", comm_data.get("successes", []), nocomm_data.get("successes", []))
        test_metric("Link Age (Fidelity)", comm_data.get("ages", []), nocomm_data.get("ages", []))
    
if __name__ == "__main__":
    run_test("Test 1: Phase 1 Agent (Perfect Hardware)", "data/comm_test1_ages.json", "data/nocomm_test1_ages.json")
    run_test("Test 2: Phase 1 Agent (Noisy Hardware)", "data/comm_test2_ages.json", "data/nocomm_test2_ages.json")
    run_test("Test 3: Phase 2 Agent (Noisy Hardware)", "data/comm_test3_ages.json", "data/nocomm_test3_ages.json")
    print("\n")
