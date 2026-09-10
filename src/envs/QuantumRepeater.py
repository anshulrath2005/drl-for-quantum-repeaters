import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register, registry
from gymnasium.utils.env_checker import check_env
import numpy as np

if "QuantumRepeater-v0" not in registry:
    register(
        id="QuantumRepeater-v0",
        entry_point="QuantumRepeater:QuantumRepeaterEnv",
    )

class QuantumRepeaterEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 1}
    
    def __init__(self, n_segments=4, tau_c=10.0, p_gen=0.1, p_swap=0.5, render_mode=None, normalized=True):
        super(QuantumRepeaterEnv, self).__init__()
        
        self.n = n_segments
        self.tau_c = tau_c
        self.p_gen = p_gen
        self.p_swap = p_swap
        self.render_mode = render_mode
        self.normalized = normalized
        self.normalization_factor = self.tau_c * 5
        
        self.num_pairs = int((self.n * (self.n + 1)) / 2)
        
        # Action Space: (n-1) swaps + discard0 + discard1 + purify
        self.action_space = spaces.MultiBinary(self.n - 1 + 3 * self.num_pairs)
        
        # Observations: 3D matrix (2 slots, n+1, n+1)
        self.observation_space = spaces.Box(low=-1, high=1e5, shape=(2, self.n + 1, self.n + 1), dtype=np.float32)
        
        # Initialize state
        self.state = np.full(shape=(2, self.n + 1, self.n + 1), fill_value=-1, dtype=np.float32)
        
        self.max_steps = 100000
        self.current_step = 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.full(shape=(2, self.n + 1, self.n + 1), fill_value=-1, dtype=np.float32)
        self.current_step = 0
        info = {}
        if self.render_mode == 'human':
            self.render()
        obs = self.state.copy().astype(np.float32)
        if self.normalized:
            obs = np.clip(obs / self.normalization_factor, -1, 1)
        return obs, info
    
    def step(self, action):
        action = np.array(action)
        idx = 0
        
        mat_as = action[idx : idx + self.n - 1]
        idx += self.n - 1
        mat_ad0 = action[idx : idx + self.num_pairs]
        idx += self.num_pairs
        mat_ad1 = action[idx : idx + self.num_pairs]
        idx += self.num_pairs
        mat_ap = action[idx : idx + self.num_pairs]
        
        self.current_step += 1
        
        # 1. Attempt to generate entanglement
        for i in range(self.n):
            # Attempt slot 0
            if self.state[0][i][i+1] == -1:
                if self.np_random.random() < self.p_gen:
                    self.state[0][i][i+1] = 0
            # Attempt slot 1
            if self.state[1][i][i+1] == -1:
                if self.np_random.random() < self.p_gen:
                    self.state[1][i][i+1] = 0
        
        # 2. Update ages of existing entanglements
        for slot in range(2):
            for i in range(self.n):
                for j in range(i + 1, self.n + 1):
                    if self.state[slot][i][j] >= 0:
                        self.state[slot][i][j] += 1
                        
        # 3. Purification Logic
        pair_idx = 0
        for i in range(self.n):
            for j in range(i + 1, self.n + 1):
                if mat_ap[pair_idx] == 1 and self.state[0][i][j] >= 0 and self.state[1][i][j] >= 0:
                    t1 = self.state[0][i][j]
                    t2 = self.state[1][i][j]
                    
                    e1 = 0.5 * (1 - np.exp(-t1 / self.tau_c))
                    e2 = 0.5 * (1 - np.exp(-t2 / self.tau_c))
                    p_succ = (1 - e1) * (1 - e2) + e1 * e2
                    
                    if self.np_random.random() < p_succ:
                        if p_succ > 0:
                            e_new = (e1 * e2) / p_succ
                        else:
                            e_new = 0.0
                        
                        e_new = min(e_new, 0.499999) # Prevent math domain errors
                        t_new = -self.tau_c * np.log(1 - 2 * e_new)
                        
                        self.state[0][i][j] = t_new
                        self.state[1][i][j] = -1
                    else:
                        self.state[0][i][j] = -1
                        self.state[1][i][j] = -1
                pair_idx += 1
                
        # 4. Discard Logic
        pair_idx = 0
        for i in range(self.n):
            for j in range(i + 1, self.n + 1):
                if mat_ad0[pair_idx] == 1:
                    self.state[0][i][j] = -1
                if mat_ad1[pair_idx] == 1:
                    self.state[1][i][j] = -1
                pair_idx += 1
        
        # 5. Entanglement Swapping (Strictly Slot 0)
        for swap_node in range(1, self.n):
            if mat_as[swap_node - 1] == 1:
                left = swap_node - 1
                right = swap_node + 1
                
                while left >= 0 and self.state[0][left][swap_node] == -1:
                    left -= 1
                while right <= self.n and self.state[0][swap_node][right] == -1:
                    right += 1
                
                if left >= 0 and right <= self.n:
                    if self.np_random.random() < self.p_swap:
                        new_age = self.state[0][left][swap_node] + self.state[0][swap_node][right]
                        self.state[0][left][right] = new_age
                    
                    self.state[0][left][swap_node] = -1
                    self.state[0][swap_node][right] = -1

        # 6. Auto-Shift Routine
        for i in range(self.n):
            for j in range(i + 1, self.n + 1):
                if self.state[0][i][j] == -1 and self.state[1][i][j] >= 0:
                    self.state[0][i][j] = self.state[1][i][j]
                    self.state[1][i][j] = -1
                    
        info = {}
        reward = 0.0
        final_link_age = self.state[0][0][self.n]
        
        truncated = self.current_step >= self.max_steps
        terminated = False
        
        if final_link_age >= 0:
            reward = self.calculate_skr(final_link_age) * 100
            info['Link Age'] = final_link_age
            info['SKR'] = self.calculate_skr(final_link_age)
            info['Time'] = self.current_step
            self.state[0][0][self.n] = -1 
            terminated = True 
            
        obs = self.state.copy().astype(np.float32)
        if self.normalized:
            obs = np.clip(obs / self.normalization_factor, -1, 1)
            
        return obs, reward, terminated, truncated, info
    
    def render(self):
        print("Current State (Slot 0):")
        print(self.state[0])
        print("Current State (Slot 1):")
        print(self.state[1])
        
    def calculate_skr(self, t):
        def h(p):
            if p <= 0 or p >= 1: return 0
            return -p * np.log2(p) - (1 - p) * np.log2(1 - p)
        
        nu = 0.5 * (1 - np.exp(-t / self.tau_c))
        return max(0, 1 - h(nu))

if __name__ == "__main__":
    env = gym.make("QuantumRepeater-v0", render_mode='human')
    print("Checking the environment...")
    check_env(env.unwrapped)
    print("Environment check passed.")