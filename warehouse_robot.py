import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Required for 3D plotting
import random
import time

"""
Warehouse Q-Learning Robot (3D Voxel Edition)
---------------------------------------------
A class-based implementation of Q-Learning for a Grid World.
Now includes a real-time 3D visualization using Matplotlib Voxels.

Dependencies: numpy, matplotlib
"""

# --- Configuration ---
GRID_SIZE = 10
ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']
ACTION_MAP = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT'}

# Cell Types
EMPTY = 0
WALL = 1
START = 2
GOAL = 3
HAZARD = 4

# The Map
GRID_LAYOUT = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 2, 0, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 1, 1, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 4, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 4, 0, 0, 0, 0, 1],
    [1, 1, 1, 0, 1, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 4, 0, 1],
    [1, 0, 1, 1, 0, 1, 1, 0, 3, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]

class WarehouseEnvironment:
    def __init__(self):
        self.grid = np.array(GRID_LAYOUT)
        self.rows, self.cols = self.grid.shape
        self.agent_pos = None
        self.start_pos = tuple(np.argwhere(self.grid == START)[0])
        self.reset()

    def reset(self):
        """Resets the agent to the start position."""
        self.agent_pos = self.start_pos
        return self.agent_pos

    def step(self, action_idx):
        """
        Takes a step in the environment.
        Returns: next_state, reward, done
        """
        r, c = self.agent_pos
        action = ACTION_MAP[action_idx]

        # Proposed new position
        nr, nc = r, c
        if action == 'UP': nr -= 1
        elif action == 'DOWN': nr += 1
        elif action == 'LEFT': nc -= 1
        elif action == 'RIGHT': nc += 1

        # Check boundaries and walls
        if (nr < 0 or nr >= self.rows or 
            nc < 0 or nc >= self.cols or 
            self.grid[nr, nc] == WALL):
            # Hit a wall: Stay in place, negative reward
            return (r, c), -5, False
        
        # Valid move
        self.agent_pos = (nr, nc)
        cell_type = self.grid[nr, nc]
        
        if cell_type == GOAL:
            return (nr, nc), 100, True # Done (Win)
        elif cell_type == HAZARD:
            return (nr, nc), -100, True # Done (Fail)
        else:
            return (nr, nc), -1, False # Living penalty

    def render_console(self):
        """ASCII render (Legacy)."""
        display_map = {WALL: '#', EMPTY: '.', START: 'S', GOAL: 'G', HAZARD: 'X'}
        print("\n" + "-" * 20)
        for r in range(self.rows):
            line = ""
            for c in range(self.cols):
                if (r, c) == self.agent_pos:
                    line += "🤖"
                else:
                    line += display_map[self.grid[r, c]] + " "
            print(line)
        print("-" * 20)

class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.env = env
        self.q_table = np.zeros((env.rows, env.cols, len(ACTIONS)))
        self.alpha = alpha      # Learning Rate
        self.gamma = gamma      # Discount Factor
        self.epsilon = epsilon  # Exploration Rate
        
    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, len(ACTIONS) - 1)
        else:
            r, c = state
            return np.argmax(self.q_table[r, c])

    def train(self, episodes=500):
        rewards_history = []
        print(f"🚀 Starting training for {episodes} episodes...")
        
        for ep in range(episodes):
            state = self.env.reset()
            total_reward = 0
            steps = 0
            done = False
            
            while not done and steps < 100:
                action_idx = self.choose_action(state)
                next_state, reward, done = self.env.step(action_idx)
                
                # Bellman Update
                r, c = state
                nr, nc = next_state
                old_val = self.q_table[r, c, action_idx]
                next_max = np.max(self.q_table[nr, nc])
                new_val = old_val + self.alpha * (reward + self.gamma * next_max - old_val)
                self.q_table[r, c, action_idx] = new_val
                
                state = next_state
                total_reward += reward
                steps += 1
            
            if self.epsilon > 0.01: self.epsilon *= 0.995
            rewards_history.append(total_reward)
            
            if (ep + 1) % 100 == 0:
                print(f"Episode {ep+1}: Steps={steps}, Reward={total_reward}")

        return rewards_history

    def visualize_3d_run(self):
        """
        Runs the optimal path and renders it in a 3D Matplotlib Window.
        
        Fix applied: All masks passed to ax.voxels must be 3-dimensional, 
        resolving the ValueError.
        """
        print("\n✨ Preparing 3D Visualization...")
        
        # --- 1. Setup the 3D Plot ---
        plt.ion() # Turn on interactive mode
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("Warehouse Robot 3D Replay")
        ax.set_box_aspect([1, 1, 0.4]) # Adjusted aspect ratio for height
        
        # Set a clear viewing angle
        ax.view_init(elev=90, azim=0) 

        # Define height constants for 3D rendering
        MAX_HEIGHT = 4
        WALL_HEIGHT = 3
        GOAL_HEIGHT = 4
        AGENT_HEIGHT = 1
        FLOOR_DEPTH = 1 
        
        # Set axis limits
        ax.set_xlim(0, self.env.cols)
        ax.set_ylim(0, self.env.rows)
        ax.set_zlim(0, MAX_HEIGHT)
        
        # Prepare 2D masks for static elements
        walls_mask_2d = self.env.grid == WALL
        goals_mask_2d = self.env.grid == GOAL
        hazards_mask_2d = self.env.grid == HAZARD
        
        # Floor mask includes all traversable and goal/hazard cells
        floor_mask_2d = (self.env.grid == EMPTY) | (self.env.grid == START) | goals_mask_2d | hazards_mask_2d

        # Create 3D voxel masks for static elements
        
        # Walls: R x C x WALL_HEIGHT
        walls_3d = np.zeros((self.env.rows, self.env.cols, WALL_HEIGHT), dtype=bool)
        for d in range(WALL_HEIGHT):
            walls_3d[:, :, d] = walls_mask_2d
            
        # Goal: R x C x GOAL_HEIGHT
        goals_3d = np.zeros((self.env.rows, self.env.cols, GOAL_HEIGHT), dtype=bool)
        for d in range(GOAL_HEIGHT):
            goals_3d[:, :, d] = goals_mask_2d
            
        # Floor/Start: R x C x FLOOR_DEPTH (only bottom layer, z=0)
        floor_3d = np.zeros((self.env.rows, self.env.cols, FLOOR_DEPTH), dtype=bool)
        floor_3d[:, :, 0] = floor_mask_2d
        
        # Hazards: Separate 3D mask for coloring on the floor layer
        hazards_3d = np.zeros((self.env.rows, self.env.cols, FLOOR_DEPTH), dtype=bool)
        hazards_3d[:, :, 0] = hazards_mask_2d
        

        # --- 2. Simulation Loop ---
        state = self.env.reset()
        done = False
        step_counter = 0
        
        try:
            while not done:
                ax.clear() # clear previous frame
                
                # Re-draw Static Environment
                
                # 1. Floor (Base for everything else)
                ax.voxels(floor_3d, facecolors='lightgray', edgecolors='black', alpha=0.1)

                # 2. Hazards (Orange/Red floor patches - must be drawn on top of the floor)
                ax.voxels(hazards_3d, facecolors='orange', edgecolors='red', alpha=0.7)
                
                # 3. Walls (Tall Salmon blocks)
                ax.voxels(walls_3d, facecolors='salmon', edgecolors='gray', alpha=0.8)
                
                # 4. Goal (Green Pillar - drawn high)
                ax.voxels(goals_3d, facecolors='lime', edgecolors='green', alpha=0.9)


                # 5. Agent (Blue Block, at z=0)
                agent_voxel = np.zeros((self.env.rows, self.env.cols, AGENT_HEIGHT), dtype=bool)
                r, c = state
                # The agent block is 1x1x1 at the current R, C position in the lowest Z layer
                agent_voxel[r, c, 0] = True 
                
                # Plot the agent. Using different color and higher alpha.
                ax.voxels(agent_voxel, facecolors='cyan', edgecolors='blue', alpha=1.0)
                
                # Formatting and View
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_zticks([])
                
                # Invert Y-axis so (0,0) is top-left like the grid
                ax.invert_yaxis()
                
                # Add text overlay
                ax.text2D(0.05, 0.95, f"Step: {step_counter}", transform=ax.transAxes, color='white')
                
                plt.draw()
                plt.pause(0.3) # Animation speed
                
                # Move Agent
                r, c = state
                action_idx = np.argmax(self.q_table[r, c])
                state, reward, done = self.env.step(action_idx)
                step_counter += 1
                
                if done:
                    # Show final frame
                    if reward > 0: 
                        ax.text2D(0.4, 0.5, "SUCCESS!", transform=ax.transAxes, color='green', fontsize=20, weight='bold')
                    else:
                        ax.text2D(0.4, 0.5, "FAILED", transform=ax.transAxes, color='red', fontsize=20, weight='bold')
                    plt.draw()
                    plt.pause(2.0)
                    
        except KeyboardInterrupt:
            print("Visualization stopped.")
        
        plt.ioff()
        plt.show()

# --- Main Execution ---

if __name__ == "__main__":
    # 1. Setup
    env = WarehouseEnvironment()
    agent = QLearningAgent(env, alpha=0.1, gamma=0.9, epsilon=1.0)
    
    # 2. Train (Fast, no rendering)
    rewards = agent.train(episodes=1000)
    
    # 3. Plot Learning Curve
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title('Agent Training Performance')
    plt.show(block=False) # Don't block execution here
    plt.pause(1) 

    # 4. Run 3D Visualization
    agent.visualize_3d_run()