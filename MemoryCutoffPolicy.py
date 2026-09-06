import numpy as np

def memory_cutoff_policy(observation, n_segments, cutoff):
    num_pairs = int((n_segments * (n_segments + 1)) / 2)
    mat_as = np.zeros(n_segments - 1, dtype=int)
    mat_ad0 = np.zeros(num_pairs, dtype=int)
    mat_ad1 = np.zeros(num_pairs, dtype=int)
    mat_ap = np.zeros(num_pairs, dtype=int)
    
    # Check slots for discard and purification
    index = 0
    for i in range(n_segments):
        for j in range(i + 1, n_segments + 1):
            age0 = observation[0][i][j]
            age1 = observation[1][i][j]
            
            # Simple Purify Heuristic: If both slots are full, ALWAYS purify
            if age0 >= 0 and age1 >= 0:
                mat_ap[index] = 1
            else:
                # If not purifying, check cutoff
                if age0 >= 0 and age0 >= cutoff:
                    mat_ad0[index] = 1
                if age1 >= 0 and age1 >= cutoff:
                    mat_ad1[index] = 1
                    
            index += 1
    
    # Check for swapping (Strictly slot 0)
    for node in range(1, n_segments):
        left = node - 1
        right = node + 1
        while left >= 0 and observation[0][left][node] == -1:
            left -= 1
        while right <= n_segments and observation[0][node][right] == -1:
            right += 1
        if left >= 0 and right <= n_segments:
            mat_as[node - 1] = 1  # Perform entanglement swapping
            
    return np.concatenate([mat_as, mat_ad0, mat_ad1, mat_ap])

def baseline_policy(observation):
    return memory_cutoff_policy(
        observation,
        n_segments=4,
        cutoff=10.0
    )