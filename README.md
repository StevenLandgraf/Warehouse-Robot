🤖 Q-Bot Warehouse Navigator: 3D Reinforcement Learning Simulator

Project Overview

This project implements a fundamental Reinforcement Learning (RL) algorithm, Q-Learning, to train an autonomous mobile robot (AGV) to navigate a simulated warehouse environment. The goal is to find the quickest path from a starting point (S) to a goal (G) while avoiding obstacles (Walls) and negative areas (Hazards).

🚀 Key Features Demonstrated

Q-Learning Implementation: The core Bellman Equation update rule is implemented from scratch.

Environment Design (Gym-like): The WarehouseEnvironment class is structured using the standard reset() and step() methods.

Policy Visualization: The agent's learned policy (the optimal path) is visualized in a real-time, interactive 3D voxel environment using Matplotlib. This allows for immediate verification of the training outcome - even though it's not very sophisticated.

Performance Metrics: A learning curve is plotted to track the total reward per episode, illustrating the agent's convergence and performance stabilization over time.

🛠️ Requirements

The project is written in Python and requires the following libraries:

pip install -r requirements.txt


⚙️ How to Run

Install Dependencies:

pip install -r requirements.txt


Execute the Script:

python warehouse_robot.py


Expected Output

The script will perform the following steps:

Training: It will first print console updates as the agent runs 1000 training episodes.

Learning Curve: A 2D plot will open, showing the reward history across episodes, confirming that the agent's performance has converged.

3D Simulation: A separate 3D interactive window will open, showcasing the robot (a cyan block) navigating the optimal, learned path through the warehouse grid (walls are salmon, the goal is a green pillar, hazards are orange).

🗺️ Warehouse Map Legend

Icon/Color

Type

Reward

Description

Salmon Block

WALL

-5

Impassable Obstacle. Penalized for collision.

Cyan Block

Agent

-1

Living penalty (encourages shortest path).

Green Pillar

GOAL

+100

The delivery package. Terminal state (Success).

Orange Patch

HAZARD

-100

Danger/Spilled Oil. Terminal state (Failure).

Light Gray

EMPTY/START

-1

Traversable empty floor.