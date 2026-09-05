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
        
        # Action Space
        self.action_space = spaces.MultiBinary(self.n - 1 + int((self.n * (self.n + 1))/2))
        
        # Observations: State of each node (-1 - no entanglement, age >=0)
        self.observation_space = spaces.Box(low=-1, high=1e5, shape=(self.n + 1, self.n + 1), dtype=np.float32)
        
        # Initialize state
        self.state = np.full(shape=(self.n + 1, self.n + 1), fill_value=-1, dtype=np.float32)
        
        self.last_entanglement_time = 0
        
        self.max_steps = 100000  # Define a horizon for resets
        self.current_step = 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.full(shape=(self.n + 1, self.n + 1), fill_value=-1, dtype=np.float32)
        self.current_step = 0
        info = {}
        if (self.render_mode == 'human'):
            self.render()
        obs = self.state.copy().astype(np.float32)
        if self.normalized:
            obs = np.clip(obs / self.normalization_factor, -1, 1)
        return obs, info
    
    def step(self, action):
        """Take an action in the environment."""
        mat_as = np.array(action)[:self.n - 1]
        mat_ad = np.array(action)[self.n - 1:]
        
        self.current_step += 1
        
        # Each step, attempt to generate entanglement in the segment where entanglement is absent
        for i in range(self.n):
            if self.state[i][i+1] == -1:
                if self.np_random.random() < self.p_gen:
                    self.state[i][i+1] = 0
        
        # Update ages of existing entanglements
        for i in range(self.n):
            for j in range(i + 1, self.n + 1):
                if self.state[i][j] >= 0:
                    self.state[i][j] += 1
                    
        # Discard entanglements that exceed coherence time
        index = 0
        for i in range(self.n):
            for j in range(i + 1, self.n + 1):
                if mat_ad[index] == 1:
                    self.state[i][j] = -1  # Discard entanglement
                index += 1
        
        # Perform entanglement swapping based on actions
        for swap_node in range(1, self.n):
            if mat_as[swap_node - 1] == 1:
                left = swap_node - 1
                right = swap_node + 1
                
                # Find nearest left entangled node
                while left >= 0 and self.state[left][swap_node] == -1:
                    left -= 1
                
                # Find nearest right entangled node
                while right <= self.n and self.state[swap_node][right] == -1:
                    right += 1
                
                if left >= 0 and right <= self.n:
                    if self.np_random.random() < self.p_swap:
                        new_age = self.state[left][swap_node] + self.state[swap_node][right]
                        self.state[left][right] = new_age
                    self.state[left][swap_node] = -1
                    self.state[swap_node][right] = -1
                    
        info = {}
        
        reward = 0.0  # Remove step penalty for trajectory-dependent reward
        final_link_age = self.state[0][self.n]
        
        truncated = self.current_step >= self.max_steps
        terminated = False
        
        if final_link_age >= 0:
            reward = self.calculate_skr(final_link_age) * 100
            info['Link Age'] = final_link_age
            info['SKR'] = self.calculate_skr(final_link_age)
            info['Time'] = self.current_step
            self.state[0][self.n] = -1  # Reset final link after reward
            terminated = True # End episode to backpropagate this exact reward to the trajectory
            
        obs = self.state.copy().astype(np.float32)
        if self.normalized:
            obs = np.clip(obs / self.normalization_factor, -1, 1)
            
        return obs, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment matrix"""
        print("Current State of the Quantum Repeater:")
        print(self.state)
        
    def calculate_skr(self, t):
        def h(p):
            if p <= 0 or p >= 1: return 0
            return -p * np.log2(p) - (1 - p) * np.log2(1 - p)
        
        nu = 0.5 * (1 - np.exp(-t / self.tau_c))
        return max(0, 1 - h(nu))

# Test the environment
if __name__ == "__main__":
    env = gym.make("QuantumRepeater-v0", render_mode='human')
    print("Checking the environment...")
    check_env(env.unwrapped)
    print("Environment check passed.")
    
    obs, info = env.reset()
    
    for _ in range(10):
        action = env.action_space.sample()
        print(f"Action taken: {action[:3]} (swaps), {action[3:]} (discards)")
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        print(f"Reward: {reward}\n")