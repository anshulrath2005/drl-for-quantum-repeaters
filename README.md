# DRL for Quantum Repeaters

This repository implements a memory-based quantum repeater as a Gymnasium environment formulated as a Markov decision process. A deep reinforcement learning agent based on the Proximal Policy Optimization (PPO) algorithm is trained to dynamically control memory discard and entanglement swapping decisions. The learned policies are evaluated based on their ability to optimize the secret key rate in quantum key distribution tasks and are compared against static memory cutoff baseline strategies.

This work is inspired by the reinforcement learning agents developed in the repository: https://github.com/SimonReiss/Master-Thesis and the method outlined in the paper: https://doi.org/10.1103/PhysRevA.108.012406

