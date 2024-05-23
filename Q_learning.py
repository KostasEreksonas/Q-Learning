#!/usr/bin/env python3

import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

"""
Set the environment to learn
"""
R = np.array([[0, 0, 0, -1, 0, 0, -1, 0, 0, 0],
              [0, -1, -1, -1, 0, 0, 0, 0, -1, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, -1, 0],
              [0, -1, 0, 0, -1, 0, 0, 0, -1, 0],
              [0, -1, 0, 0, -1, -1, -1, -1, -1, 0],
              [0, -1, 0, 0, 0, 0, 0, 0, -1, 0],
              [0, -1, -1, -1, 0, 0, -1, 0, -1, 0],
              [0, 0, 0, -1, 0, 0, -1, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, -1, 0, 0, 0],
              [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])

"""
Set parameters:
alpha (α) - learning rate
gamma (γ) - discount rate for future rewards
epsilon (ε) - exploration probability where the action is selected randomly instead of Q-value
max epsilon - max value for epsilon
min epsilon - min value for epsilon
decay rate - decay rate for epsilon, decrease probability of random actions over the course of iterations
epochs - number of epochs to run
actions - define possible actions
    up - agent goes up
    down - agent goes down
    left - agent goes left
    right - agent goes right
    stay - agent stays at the same place
x - number of rows
y - number of columns
states - number of possible states in Q table
Q - define Q table
max_steps - max steps in a single epoch
goal_state - goal state when the maze is solved
"""
alpha = 0.7
gamma = 0.95
max_epsilon = 1.0
min_epsilon = 0.05
decay_rate = 0.0005
epochs = 10000
actions = ['up', 'down', 'left', 'right', 'stay']
x = len(R)
y = len(R[0])
states = x*y
Q = np.zeros((states, 5))
max_steps = 2000
goal_state = 49

def find_walls(x, y, state=0, walls=[]):
    """
    Find walls in the maze
    """
    for i in range(x):
        for j in range(y):
            if R[i][j] == -1:
                walls.append(state)
            state += 1
    return walls

def initialize_table(x, y, state=0):
    """
    Initializes the Q table with possible actions in every state
    """
    # Find wall states in a maze and set all their actions as unavailable
    walls = find_walls(x, y)
    if state in walls:
        for a in range(len(actions)):
            Q[state][a] = -1
    # Loop through states and determine possible actions for each state
    for i in range(x):
        for j in range(y):
            # Agent can't move up if it is on the first row or there is a wall above
            if i == 0 or state - 10 in walls:
                Q[state][actions.index('up')] = -1
            # Agent can't move down if it is on the last row or there is a wall below
            if i == x - 1 or state + 10 in walls:
                Q[state][actions.index('down')] = -1
            # Agent can't move left if it is on the first column or there is a wall on the left
            if j == 0 or state - 1 in walls:
                Q[state][actions.index('left')] = -1
            # Agent can't move right if it is on the last column or there is a wall on the right
            if j == y - 1 or state + 1 in walls:
                Q[state][actions.index('right')] = -1
            state += 1

def get_epsilon(epoch):
    """
    Reduce exploration probability (epsilon) with each epoch.
    With every epoch it increases the probability of choosing next state based on Q value instead of random state picking
    """
    return min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * epoch)

def take_action(random_state, epsilon):
    """
    Make a decision on the next action (up, down, left, right).
    If a random number between 0 and 1 is higher than exploration coefficient, choose next action based on a Q value,
    else choose next action randomly.
    """
    if np.random.random() > epsilon:
        if np.max(Q[random_state]) == -1:
            return None # If the best available action is hitting the wall, terminate the agent
        else:
            return actions[np.argmax(Q[random_state])] # Choose action with max Q value
    else:
        indexes = [index for index,value in enumerate(Q[random_state]) if value != -1]
        possible_actions = [value for index,value in enumerate(actions) if index in indexes]
        return random.choice(possible_actions) # Choose random action

def update_table(state, action, reward, next_state):
    """
    Update Q table using temporal difference update rule
    """
    Q[state][actions.index(action)] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state][actions.index(action)])

def select_state(state, action):
    """
    Select the nex state for a given action
    """
    if action == 'up':
        next_state = state - 10
    elif action == 'down':
        next_state = state + 10
    elif action == 'left':
        next_state = state - 1
    elif action == 'right':
        next_state = state + 1
    elif action == 'stay':
        next_state = state
    return next_state

def learn_table(x, y):
    """
    Explore the environment iteratively and refine the Q table
    """
    initialize_table(x, y)
    walls = find_walls(x, y)
    possible_states = [x for x in range(len(Q)) if x not in walls]
    for epoch in range(epochs):
        random_state = random.choice(possible_states)
        epsilon = get_epsilon(epoch)
        for steps in range(max_steps):
            action = take_action(random_state, epsilon)
            next_state = select_state(random_state, action)
            reward = 100 if random_state == goal_state else 0 # Reward the agent only if it finds the goal state
            update_table(random_state, action, reward, next_state)
            random_state = next_state
            if reward == 100: break # Stop the epoch when goal is reached

def visualize(path, random_state):
    """
    Visualize the solved maze environment
    """
    cmap = colors.ListedColormap(['blue', 'white', 'green'])
    norm = colors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    R_copy = R.copy()
    R_copy = R_copy.ravel()
    for x in range(len(R_copy)):
        if x in path:
            R_copy[x] = 1
    R_copy = R_copy.reshape(10,10)
    fig, ax = plt.subplots()
    ax.imshow(R_copy, cmap=cmap, norm=norm)
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
    ax.set_title("Solved maze")
    ax.set_xticks(np.arange(-.5, 10, 1))
    ax.set_yticks(np.arange(-.5, 10, 1))
    fig.savefig(f"images/state-{random_state}-path.png")
    print(f"Figure saved at images/state-{random_state}-path.png")

def find_path(x, y, path=[]):
    """
    Find a path to the goal state from given random state
    """
    learn_table(x, y)
    walls = find_walls(x, y)
    possible_states = [x for x in range(len(Q)) if x not in walls]
    random_state = random.choice(possible_states)
    state = random_state
    while state != goal_state:
        path.append(state)
        action = actions[np.argmax(Q[state])]
        next_state = select_state(state, action)
        state = next_state
        if state == goal_state:
            path.append(state)
            break
    visualize(path, random_state)
    print(f"Path: {path}, length: {len(path)}")

def main():
    find_path(x, y)

if __name__ == "__main__":
    main()
