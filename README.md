# Q-Bot Warehouse Navigator: 3D Reinforcement Learning Simulator

## Project Overview

This project implements a fundamental Reinforcement Learning (RL) algorithm, **Q-Learning**, to train an autonomous mobile robot (AGV) to navigate a simulated 3D warehouse environment.

The solution is designed as a core portfolio piece, showcasing expertise in:

- **Machine Learning:** Implementing and visualizing a core model (Q-Learning) from first principles.  
- **Robotics / AI:** Demonstrating optimal path planning and state management in a grid world.  
- **Data Visualization:** Using 3D rendering to clearly communicate complex algorithmic results.

## Core RL Process

The Q-Learning algorithm iteratively updates a **Q-table** using the Bellman Equation, balancing exploration (trying new paths) and exploitation (following the best-known path). The Q-table eventually converges to an optimal policy that selects the best action for every state in the warehouse.

## Key Features

- **Q-Learning Implementation:** The Bellman Equation update rule is implemented from scratch using Python/NumPy.  
- **Modular Environment Design (Gym-like):** The `WarehouseEnvironment` class uses the familiar `reset()` and `step()` interface.  
- **3D Policy Visualization:** The learned policy is visualized in an interactive 3D voxel environment using Matplotlib.  
- **Performance Metrics:** The training script plots total reward per episode to show convergence.

## Setup and Requirements

pip install -r requirements.txt
