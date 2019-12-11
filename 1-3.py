import numpy as np
import random
import os
import math
import matplotlib.pyplot as plt


class Maze:
    # Actions
    STAY       = 0
    MOVE_LEFT  = 1
    MOVE_RIGHT = 2
    MOVE_UP    = 3
    MOVE_DOWN  = 4

    # cur_state = 0

    # Give names to actions
    actions_names = {
        STAY: "stay",
        MOVE_LEFT: "move left",
        MOVE_RIGHT: "move right",
        MOVE_UP: "move up",
        MOVE_DOWN: "move down"
    }


    # Reward values
    BANK_REWARD = 1
    CAUGHT_BY_POLICE = -10

    def __init__(self, maze, bank_position, start):
        self.maze                     = maze
        self.bank_pos                 = bank_position
        self.states, self.map         = self.__states()
        self.actions                  = self.__actions(still=True)
        self.actions_p                = self.__actions()
        self.n_actions                = len(self.actions)
        self.n_actions_p              = len(self.actions_p)
        self.n_states                 = len(self.states)
        self.cur_state                = self.map[start]

    def __actions(self, still=False):
        actions = dict()
        if still:
            actions[self.STAY]       = (0, 0)
        actions[self.MOVE_LEFT]  = (0,-1)
        actions[self.MOVE_RIGHT] = (0, 1)
        actions[self.MOVE_UP]    = (-1,0)
        actions[self.MOVE_DOWN]  = (1,0)
        return actions
 

    def __states(self):
        states = dict()
        map = dict()
        end = False
        s = 0
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                for m in range(self.maze.shape[0]):
                    for n in range(self.maze.shape[1]):
                        states[s] = (i,j,m,n)
                        map[(i,j,m,n)] = s
                        s += 1
        return states, map

    def __move(self, state, action, action_p):
        
        # Compute the future position given current (state, action)
        row = min(max(self.states[state][0] + self.actions[action][0], 0), self.maze.shape[0]-1)
        col = min(max(self.states[state][1] + self.actions[action][1], 0), self.maze.shape[1]-1)
        
        row_p = min(max(self.states[state][2] + self.actions_p[action_p][0], 0), self.maze.shape[0]-1)
        col_p = min(max(self.states[state][3] + self.actions_p[action_p][1], 0), self.maze.shape[1]-1)

        return self.map[(row, col, row_p, col_p)] 

    def cal_reward(self, action):

        # Randomly sample the police's action
        police_action = random.randint(1,4)
        next_s = self.__move(self.cur_state ,action, police_action)

        # Reward for caught by police
        if [self.states[next_s][0], self.states[next_s][1]] == [self.states[next_s][2], self.states[next_s][3]]:
            self.cur_state = next_s  
            return  next_s, self.CAUGHT_BY_POLICE
        # Reward for reaching the bank
        elif [self.states[next_s][0], self.states[next_s][1]] == self.bank_pos:
            self.cur_state = next_s  
            return next_s, self.BANK_REWARD
        # Reward for taking a step to an empty cell that is not the exit
        else:
            self.cur_state = next_s  
            return next_s, 0
  


def Q_learning(env, dis_factor, iter):
    # Initialization 
    Q = np.zeros((env.n_states,env.n_actions))
    n = np.zeros((env.n_states,env.n_actions))

    record  = []
    record1 = []
    record2 = []

    # Begin iteration
    for i in range(iter):
        # Randomly sample robber's action
        action = random.randint(0, env.n_actions-1)
        # Save Current state
        cur_state = env.cur_state
        # Calculate the new state and reward
        new_state, reward = env.cal_reward(action)
        # Calculate step size
        alpha = 1 / pow(n[cur_state, action] + 1, 2/3)
        # Update the value
        Q[cur_state, action] += alpha * (reward + dis_factor * max(Q[new_state]) - Q[cur_state, action])
        n[cur_state, action] += 1

        if i % 100000 == 0:
            print("Iteration: " + str(i) + "  Value for Initial state: " + str(np.max(Q[env.map[(0,0,3,3)]])))
        record.append(np.max(Q[env.map[(0,0,3,3)]]))
        record1.append(np.max(Q[env.map[(3,3,2,2)]]))
        record2.append(np.max(Q[env.map[(2,3,2,3)]]))
     
    return Q, record, record1, record2


def epsilon_greedy(env, state, epsilon, Q):
    '''
    return the probability of each actions
    '''
    # Initialization
    other_prob = epsilon / (env.n_actions -1)
    action_prob = np.zeros(env.n_actions) + other_prob
    # Choose the best action according to Q table
    best_action = np.argmax(Q[state])
    # Set the best action probability
    action_prob[best_action] = 1 - epsilon
    return action_prob



def Sarsa(env, dis_factor, iter):

    epsilon =  0.1
    # Initialization 
    Q = np.zeros((env.n_states,env.n_actions))
    n = np.zeros((env.n_states,env.n_actions))

    record  = []

    # For the first action, we use the epsilon-greedy method to decide 
    action_prob = epsilon_greedy(env, env.cur_state, epsilon, Q)
    best_action = np.random.choice(np.arange(env.n_actions), 1, p = action_prob)[0]

    # Begin iteration
    for i in range(iter):
        # Save Current state and action
        cur_state = env.cur_state
        cur_action = best_action

        # Take action a and observe r and s'
        new_state, reward = env.cal_reward(cur_action)

        # Choose a' from s' using epsilon-greedy method
        action_prob = epsilon_greedy(env, new_state, epsilon, Q)
        best_action = np.random.choice(np.arange(env.n_actions), 1, p = action_prob)[0]

        # Calculate step size
        alpha = 1 / pow(n[cur_state, cur_action] + 1, 2/3)

        # Update the value
        Q[cur_state, cur_action] += alpha * (reward + dis_factor * Q[new_state, best_action] - Q[cur_state, cur_action])
        n[cur_state, cur_action] += 1

        if i % 100000 == 0:
            print("Iteration: " + str(i) + "  Value for Initial state: " + str(np.max(Q[env.map[(0,0,3,3)]])))
        record.append(np.max(Q[env.map[(0,0,3,3)]]))
    
    return Q, record




maze_map = [[0 for i in range(4)] for j in range(4)]

maze = Maze(np.array(maze_map), [1,1], (0,0,3,3))
iter_ = 10000000 

Q, record, record1, record2 = Q_learning(maze, 0.8, iter_)

print(Q)

plt.plot(np.linspace(0,len(record)-1,len(record)),record)
plt.plot(np.linspace(0,len(record1)-1,len(record1)),record1)
plt.plot(np.linspace(0,len(record2)-1,len(record2)),record2)
plt.legend()
plt.show()
