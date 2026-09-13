# DRL for Quantum Repeaters

This repository implements a memory-based quantum repeater chain as a Gymnasium and PettingZoo environment, formulated as a multi-agent Partially Observable Markov Decision Process (POMDP). A Multi-Agent Reinforcement Learning (MARL) system based on the Proximal Policy Optimization (PPO) algorithm is trained to dynamically distribute control over memory discard, entanglement swapping, and entanglement purification decisions.

The agents are trained using a curriculum learning strategy to adaptively handle noisy quantum hardware. The learned policies are statistically evaluated on their ability to optimize the BB84 secret key rate in quantum key distribution tasks---prioritizing high-fidelity entanglement generation---and are rigorously benchmarked against static memory-cutoff baseline strategies.

This work is inspired by the reinforcement learning agents developed in the repository: https://github.com/SimonReiss/Master-Thesis and the method outlined in the paper: https://doi.org/10.1103/PhysRevA.108.012406
