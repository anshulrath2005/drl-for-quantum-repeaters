import numpy as np
from pettingzoo.utils.env import ParallelEnv
from gymnasium import spaces

class QuantumRepeaterParallelEnv(ParallelEnv):
    metadata = {'render_modes': ['human'], "name": "quantum_repeater_marl_nocomm_v0"}

    def __init__(self, n_segments=4, tau_c=10.0, p_gen=0.1, p_swap=0.5, normalized=True):
        super().__init__()
        self.n = n_segments
        self.tau_c = tau_c
        self.p_gen = p_gen
        self.p_swap = p_swap
        self.normalized = normalized
        self.normalization_factor = self.tau_c * 5
        self.render_mode = None
        
        # Define our 3 local agents (Node 1, Node 2, Node 3)
        self.possible_agents = [f"agent_{i}" for i in range(1, self.n)]
        self.agents = self.possible_agents[:]
        
        self.num_pairs = int((self.n * (self.n + 1)) / 2)
        
        # Action space: 33 physical actions + 2 communication bits. We use MultiDiscrete to force True/False outputs.
        self.action_spaces = {agent: spaces.MultiDiscrete([2] * (self.n - 1 + 3 * self.num_pairs + 2)) for agent in self.possible_agents}
        
        # Observation space: 3D matrix. Layer 0 & 1 are memory slots. Layer 2 is the Communication Plane.
        self.observation_spaces = {agent: spaces.Box(low=-1, high=1e5, shape=(2, self.n + 1, self.n + 1), dtype=np.float32) for agent in self.possible_agents}
        
        self.state = np.full(shape=(2, self.n + 1, self.n + 1), fill_value=-1, dtype=np.float32)
        # Array to store the classical messages sent on the current turn to be delivered next turn
                
        self.current_step = 0
        self.max_steps = 100000

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.state = np.full(shape=(2, self.n + 1, self.n + 1), fill_value=-1, dtype=np.float32)
        self.current_step = 0
        return {agent: self._get_obs(agent) for agent in self.agents}, {agent: {} for agent in self.agents}

    def _get_obs(self, agent_name):
        """
        PARTIAL OBSERVABILITY (POMDP) with CLASSICAL MESSAGING:
        """
        node_id = int(agent_name.split("_")[1])
        obs = np.full(shape=(2, self.n + 1, self.n + 1), fill_value=-1, dtype=np.float32)
        
        # 1. Unmask the physical links touching this node
        for slot in range(2):
            for i in range(self.n):
                for j in range(i + 1, self.n + 1):
                    if i == node_id or j == node_id:
                        obs[slot][i][j] = self.state[slot][i][j]
                        
        # 2. Normalize the physical ages
        if self.normalized:
            obs[0] = np.clip(obs[0] / self.normalization_factor, -1, 1)
            obs[1] = np.clip(obs[1] / self.normalization_factor, -1, 1)
            
        return obs

    def step(self, actions):
        self.current_step += 1
        
        global_mat_as = np.zeros(self.n - 1, dtype=int)
        global_mat_ad0 = np.zeros(self.num_pairs, dtype=int)
        global_mat_ad1 = np.zeros(self.num_pairs, dtype=int)
        global_mat_ap = np.zeros(self.num_pairs, dtype=int)
        
        # Create a fresh message board for this turn
                
        for agent_name, action in actions.items():
            node_id = int(agent_name.split("_")[1])
            action = np.array(action, dtype=int)
            idx = 0
            
            mat_as = action[idx : idx + self.n - 1]
            idx += self.n - 1
            mat_ad0 = action[idx : idx + self.num_pairs]
            idx += self.num_pairs
            mat_ad1 = action[idx : idx + self.num_pairs]
            idx += self.num_pairs
            mat_ap = action[idx : idx + self.num_pairs]
            idx += self.num_pairs
            
            if mat_as[node_id - 1] == 1:
                global_mat_as[node_id - 1] = 1
            
            pair_idx = 0
            for i in range(self.n):
                for j in range(i + 1, self.n + 1):
                    if i == node_id or j == node_id:
                        if mat_ad0[pair_idx] == 1: global_mat_ad0[pair_idx] = 1
                        if mat_ad1[pair_idx] == 1: global_mat_ad1[pair_idx] = 1
                        if mat_ap[pair_idx] == 1: global_mat_ap[pair_idx] = 1
                    pair_idx += 1

        # =========================================================================
        # THE PHYSICS ENGINE
        # =========================================================================
        
        for i in range(self.n):
            if self.state[0][i][i+1] == -1:
                if np.random.random() < self.p_gen: self.state[0][i][i+1] = 0
            if self.state[1][i][i+1] == -1:
                if np.random.random() < self.p_gen: self.state[1][i][i+1] = 0
        
        for slot in range(2):
            for i in range(self.n):
                for j in range(i + 1, self.n + 1):
                    if self.state[slot][i][j] >= 0:
                        self.state[slot][i][j] += 1
                        
        pair_idx = 0
        for i in range(self.n):
            for j in range(i + 1, self.n + 1):
                if global_mat_ap[pair_idx] == 1 and self.state[0][i][j] >= 0 and self.state[1][i][j] >= 0:
                    t1, t2 = self.state[0][i][j], self.state[1][i][j]
                    e1 = 0.5 * (1 - np.exp(-t1 / self.tau_c))
                    e2 = 0.5 * (1 - np.exp(-t2 / self.tau_c))
                    p_succ = (1 - e1) * (1 - e2) + e1 * e2
                    
                    if np.random.random() < p_succ:
                        e_new = min((e1 * e2) / max(p_succ, 1e-9), 0.499999) 
                        self.state[0][i][j] = -self.tau_c * np.log(1 - 2 * e_new)
                        self.state[1][i][j] = -1
                    else:
                        self.state[0][i][j] = -1
                        self.state[1][i][j] = -1
                pair_idx += 1
                
        pair_idx = 0
        for i in range(self.n):
            for j in range(i + 1, self.n + 1):
                if global_mat_ad0[pair_idx] == 1: self.state[0][i][j] = -1
                if global_mat_ad1[pair_idx] == 1: self.state[1][i][j] = -1
                pair_idx += 1
        
        # =========================================================================
        # REWARD BROADCAST & BREADCRUMBS
        # =========================================================================
        reward = 0.0
        
        for swap_node in range(1, self.n):
            if global_mat_as[swap_node - 1] == 1:
                left = swap_node - 1
                right = swap_node + 1
                
                while left >= 0 and self.state[0][left][swap_node] == -1: left -= 1
                while right <= self.n and self.state[0][swap_node][right] == -1: right += 1
                
                if left >= 0 and right <= self.n:
                    if np.random.random() < self.p_swap:
                        # BREADCRUMB FIX: Only reward +0.1 if the agent is NOT overwriting an existing link!
                        # This stops the infinite money glitch of repeatedly destroying/overwriting links.
                        if self.state[0][left][right] == -1:
                            reward += 0.1
                            
                        self.state[0][left][right] = self.state[0][left][swap_node] + self.state[0][swap_node][right]
                        
                    self.state[0][left][swap_node] = -1
                    self.state[0][swap_node][right] = -1

        for i in range(self.n):
            for j in range(i + 1, self.n + 1):
                if self.state[0][i][j] == -1 and self.state[1][i][j] >= 0:
                    self.state[0][i][j] = self.state[1][i][j]
                    self.state[1][i][j] = -1
                    
        # (Message board removed)
        final_link_age = self.state[0][0][self.n]
        truncated = self.current_step >= self.max_steps
        terminated = False
        
        if final_link_age >= 0:
            base_skr = self.calculate_skr(final_link_age)
            
            # REWARD SHAPING: Smooth exponential curve to prevent zero-gradient cliffs
            reward += base_skr * (100.0 + 400.0 * np.exp(-final_link_age / 3.0))
                
            self.state[0][0][self.n] = -1 
            terminated = True 
            
        rewards = {agent: reward for agent in self.agents}
        terminations = {agent: terminated for agent in self.agents}
        truncations = {agent: truncated for agent in self.agents}
        infos = {agent: {'Link Age': final_link_age} if terminated else {} for agent in self.agents}
        observations = {agent: self._get_obs(agent) for agent in self.agents}
        
        if terminated or truncated:
            self.agents = [] 
            
        return observations, rewards, terminations, truncations, infos

    def calculate_skr(self, t):
        def h(p):
            if p <= 0 or p >= 1: return 0
            return -p * np.log2(p) - (1 - p) * np.log2(1 - p)
        nu = 0.5 * (1 - np.exp(-t / self.tau_c))
        return max(0, 1 - h(nu))
