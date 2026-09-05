import numpy as np

def memory_cutoff_policy(observation, n_segments, cutoff):
    mat_as = np.zeros(n_segments - 1, dtype=int)
    mat_ad = np.zeros(int((n_segments * (n_segments + 1)) / 2), dtype=int)
    
    index = 0
    for i in range(n_segments):
        for j in range(i + 1, n_segments + 1):
            age = observation[i][j]
            if age >= 0 and age >= cutoff:
                mat_ad[index] = 1  # Discard entanglement
            index += 1
    
    for node in range(1, n_segments):
        left = node - 1
        right = node + 1
        while left >= 0 and observation[left][node] == -1:
            left -= 1
        while right <= n_segments and observation[node][right] == -1:
            right += 1
        if left >= 0 and right <= n_segments:
            mat_as[node - 1] = 1  # Perform entanglement swapping
    return np.concatenate([mat_as, mat_ad])

def baseline_policy(observation):
    return memory_cutoff_policy(
        observation,
        n_segments=4,
        cutoff=10.0
    )