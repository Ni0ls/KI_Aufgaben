# -*- coding: utf-8 -*-
"""
Task 3) Cliff Walk

Created on Mon May 30 10:03:28 2022

@author: ni0ls
"""

import numpy as np
from collections import deque 

#Cliff Environment
#s in range of 0-47 -> s = 36 start, s = 47 goal, 37 - 46 cliff respwan to sart

def envMove(s,a):
    r = 0 #game rule is: every move rewards 0, going out of bounds: -1
    done = False

    if s == 0:  #links oben
        if a == 0:  
            s_strich = s
            r = -1
        elif a == 1:  
            s_strich = s
            r = -1
        elif a == 2:  
            s_strich = s+1
        elif a == 3:  
            s_strich = s+12
    elif s < 11: #oben flanke
        if a == 0:  
            s_strich = s-1
        elif a == 1:  
            s_strich = s
            r = -1
        elif a == 2:  
            s_strich = s+1
        elif a == 3:  
            s_strich = s+12
    elif s == 11:       #rechts oben
        if a == 0:  
            s_strich = s-1
        elif a == 1:  
            s_strich = s
            r = -1
        elif a == 2:  
            s_strich = s
            r = -1
        elif a == 3:  
            s_strich = s+12
    elif s == 12 or s == 24:  #links flanke
        if a == 0:  
            s_strich = s
            r = -1
        elif a == 1:  
            s_strich = s-12
        elif a == 2:  
            s_strich = s+1
        elif a == 3:  
            s_strich = s+12
    elif s == 23:  #rechts flanke
        if a == 0:  
            s_strich = s-1
        elif a == 1:  
            s_strich = s-12
        elif a == 2:  
            s_strich = s
            r = -1
        elif a == 3:  
            s_strich = s+12
    elif s == 35:  #rechts flanke eins vor ziel
        if a == 0:  
            s_strich = s-1
        elif a == 1:  
            s_strich = s-12
        elif a == 2:  
            s_strich = s
            r = -1
        elif a == 3:  
            s_strich = s+12 #Ziel erreicht
            r = 0
            done = True
    elif s < 23:  #mitte 
        if a == 0:  
            s_strich = s-1
        elif a == 1:  
            s_strich = s-12
        elif a == 2:  
            s_strich = s+1
        elif a == 3:  
            s_strich = s+12 
    elif s == 36:  #links unten (start)
        if a == 0:  
            s_strich = s
            r = -1
        elif a == 1:  
            s_strich = s-12
        elif a == 2:  
            s_strich = s # ins Cliff
            r = -100
        elif a == 3:  
            s_strich = s
            r = -1
    elif s < 35 and s > 24: #unten flanke (am Cliff)
        if a == 0:  
            s_strich = s-1
        elif a == 1:  
            s_strich = s-12
        elif a == 2:  
            s_strich = s+1
        elif a == 3:  
            s_strich = 36
            r = -100
    elif s == 47:  #rechts unten
        if a == 0:  
            s_strich = s-1
        elif a == 1:  
            s_strich = s-12
        elif a == 2:  
            s_strich = s
            r = -1
        elif a == 3:  
            s_strich = s 
            r = -1
   
    return s_strich, r, done #for given state and action
    
def epsilon_greedy(Q,obs,epsilon = 0.01):
    """
    Implementation of epsilon greedy policy
  
    Parameters:
    Q nxm array (float): Q Table of n states and m actions
    obs         (int):  current state
    epsilon     (float): epsilon used in epsilon greedy algorithm
  
    Returns:
    int: action 
    """ 
    #***YOUR CODE HERE***
    if epsilon > np.random.rand() or sum(Q[obs][:]) == 0: 
        return np.random.randint(0,4) #with probability of epsilon or no Q information gained yet, return a random action = [0-3]
    else:
    
        return np.argmax(Q[obs][:])   #otherwise choose the greedy option (argmax of Q, best known option)
    
    
#%% Q learning
Q = np.random.rand(48,4)/10 # init Q table 
alpha = 0.05
gamma = 0.99
e = 0.1
debug_ = False
pi = np.zeros(48) 

nr_episodes = 500 

scores_window = deque(maxlen=1000)  # last <maxlen> scores
scores = []

for episode in range(nr_episodes):
    obs = 36
    done = False
    
    while not(done):
        #***YOUR CODE HERE***
        # sample a epsilon-greedy action from the action space
        action = epsilon_greedy(Q,obs,epsilon = e)
        # take the action and get the new observation space
        new_obs, reward, done = envMove(obs, action)
        # Q-learning
        
        Q[obs][action] = Q[obs][action] + alpha*(reward + gamma*max(Q[new_obs][:]) - Q[obs][action])
        obs = new_obs 
        
        pi[obs] = np.argmax(Q[obs][:])
        
        if done:
            scores_window.append(reward)       # save last reward received
            if debug_:
              print(reward)
            #print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_window)), end="")
            if episode % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_window)))
                scores.append( np.mean(scores_window) )  
                
print(pi.reshape(4,12))

#%% SARSA Algortihm
Q = np.random.rand(48,4)/10 # init Q table 
alpha = 0.1
gamma = 1
e = 0.1
debug_ = False
pi = np.zeros(48) 

nr_episodes = 500

scores_window = deque(maxlen=1000)  # last <maxlen> scores
scores = []

for episode in range(nr_episodes):
    obs = 36
    done = False
    # sample a epsilon-greedy action from the action space
    action = epsilon_greedy(Q,obs,epsilon = e)
    
    while not(done):
        #***YOUR CODE HERE***
        # take the action and get the new observation space
        new_obs, reward, done = envMove(obs, action)
        new_action = epsilon_greedy(Q,obs,epsilon = e)
        # SARSA Algortim
        Q[obs][action] = Q[obs][action] + alpha*(reward + gamma*Q[new_obs][new_action] - Q[obs][action])
        obs = new_obs 
        action = new_action

        pi[obs] = np.argmax(Q[obs][:])
        
        if done:
            scores_window.append(reward)       # save last reward received
            if debug_:
              print(reward)
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_window)), end="")
            if episode % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_window)))
                scores.append( np.mean(scores_window) )  
                
print(pi.reshape(4,12))
                