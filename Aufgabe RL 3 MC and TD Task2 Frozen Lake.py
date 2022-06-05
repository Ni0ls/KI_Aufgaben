# -*- coding: utf-8 -*-
"""
Exercise for AIM course - Reinforcement Learning

Example Code for
- Monte Carlo Method
- Q Learning 

Using Open AI Gym's Frozen Lake Environment    
https://www.gymlibrary.ml/environments/toy_text/frozen_lake/
> pip install gym     # to install gym lib
> pip install pygame  # to install pygame lib (needed for visualization) 

Suggested further reading:
https://blog.paperspace.com/getting-started-with-openai-gym/
https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/

@author: A. Hanuschkin 22

Edited on Sat May 28 23:34:40 2022

by

@author: N. Benecke
"""


import gym 
import pygame    # used for delaying pygame visualization 
import numpy as np
from collections import deque 

# use matplotlib for plotting & animation 
# and increase default font size of figures 
import matplotlib.pyplot as plt  
import matplotlib.animation as ma
font_size = 16 
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : font_size}
plt.rc('font', **font)
plt.rcParams['svg.fonttype'] = 'none'


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



def plot_training(x,y,fout='Q_Q-Learning.jpg'):    
    """
    Plot of Training Progress Saved to File
  
    Parameters:
    x array (float): x values
    y array (float): y values
    fout    (str): save plot to filename <fout>
  
    Returns:
    """     
    plt.figure()
    plt.plot(x,y,scores,'X', color='black')
    plt.plot(x,y,scores,'--', color='black')
    plt.xlabel('episode')
    plt.ylabel(r'$<reward_{final}>$')
    plt.savefig(fout,dpi=300)



def animate(i,Q,ax):
    """
    Animation function called by matplotlib FuncAnimation
  
    Parameters:
    Q nxm array (float): Q Table of n states and m actions
    ax <AxesSubplot:>: current axis to draw
  
    Returns:
    """ 
    global obs
    global walk_str
    action = epsilon_greedy(Q,obs,epsilon = 0)
    new_obs, reward, done, info = env.step(action)
    obs = new_obs
    print('state',obs,'action:',action_str[action], 'reward',reward)
    walk_str = walk_str + '-' + action_str[action][0]
    if done:
        walk_str = walk_str + '-T'
        if obs != 15:
           print('Broken into the ice...')
        else: 
           print('Done!!')
        
    env_screen = env.render(mode = 'rgb_array')
    ax.clear()
    plt.title(walk_str)
    ax.imshow(env_screen)
    plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
    if done:
        obs = env.reset()
        walk_str = 'S'

def animation_result(Q):
    """
    Animation of agent for given Q table
  
    Parameters:
    Q nxm array (float): Q Table of n states and m actions

    Returns:
    animation object # store the created Animation in a variable that lives as long as the animation should run.
    """ 
    global obs
    global walk_str
    plt.figure()
    fig, ax = plt.subplots()
    obs = env.reset()
    walk_str = 'S'

    ani = ma.FuncAnimation(fig, animate, fargs = [Q,ax],frames=20, interval=1000, repeat=False)
    #ani.save('test.mp4', fps=10)
    #ani.save('test.gif',writer='imagemagick')

    return ani 
    
   # plt.show()
   # pygame.time.wait(5000)


#%% Setting up OPEN AI GYM environment & demo it's usage

env = gym.make('FrozenLake-v1', desc=None,map_name="4x4", is_slippery=False) ## is_slippery=True

# observation and action space 
obs_space = env.observation_space
action_space = env.action_space
print("The observation space: {}".format(obs_space))
print("The action space: {}".format(action_space))

# reset the environment and see the initial observation
obs = env.reset()
print("The initial observation is {}".format(obs))

# sample a random action from the entire action space
random_action = env.action_space.sample()
print('Perform random action',random_action)

# take the action and get the new observation space
new_obs, reward, done, info = env.step(random_action)
print("The new observation is {}".format(new_obs))

action_str = ['LEFT','DOWN','RIGHT','UP']

'''
for jj in range(10):  # visialize 10 random steps 
    env.render(mode = "human")
    random_action = env.action_space.sample()
    new_obs, reward, done, info = env.step(random_action)
    print('action:',action_str[random_action])
    # making the script wait for 50 ms
    pygame.time.wait(50)
'''  



#%% Implement MC Method 
Q = np.zeros([16,4]) # init Q table 
Returns = np.zeros([16,4]) # init Q table 
N = np.zeros([16,4]) # init Q table 
pi = np.zeros(16) # init Q table  # init Q table 
e = 0.1

# alternative way, not using a tabular but more felxible dictonary for Q
#from collections import defaultdict
#Q = defaultdict(lambda: np.zeros(4))

nr_episodes = 20000
for episode in range(nr_episodes):
    # generate sequence of MC moves:     
    # reset the environment and see the initial observation
    obs = env.reset()
    done = False
    seq = []
    
    while not(done):
        #***YOUR CODE HERE***
        # sample a epsilon-greedy action from the action space
        action = epsilon_greedy(Q,obs,epsilon = e)
        # take the action and get the new observation space
        new_obs, reward, done, info = env.step(action)
        seq.append((obs, action, reward))
        obs = new_obs         
        #if done:
            #print(reward)
    
    ##print(seq)
    # MC learning based on sampled seq 
    steps = len(seq) 
    G = 0 
    gamma = 0.99
    for jj in range(steps):
        pos = steps-jj-1
        G = gamma*G + seq[pos][2] 
        # check if first visit....
        first_visit = True
        for ii in range(steps-jj-1):  # check all states visited in this run 
            if seq[ii][0] == seq[pos][0] and seq[ii][1] == seq[pos][1]:
                first_visit = False
        if first_visit:    
            #***YOUR CODE HERE***
            s = seq[pos][0]
            a = seq[pos][1]
            N[s][a] += 1
            Returns[s][a] += 1/N[s][a]*G #TODO what does append mean in algorithm ? question
            Q[s][a] = Q[s][a] + 1/N[s][a]*(Returns[s][a]-Q[s][a])
            pi[s] = np.argmax(Q[s][:])
            
print("\nMC Method ->\nPolicy after", nr_episodes, "episodes and epsion-greedy (",e,") sequence generation:")
print(pi.reshape(4,4))
#np.savez('Q_MC.npz',Q=Q) # uncomment to save your result


#%% Q learning
Q = np.random.rand(16,4)/10 # init Q table 
alpha = 0.05
gamma = 0.99
e = 0.1
debug_ = False
pi = np.zeros(16) 

nr_episodes = 20000

scores_window = deque(maxlen=1000)  # last <maxlen> scores
scores = []

for episode in range(nr_episodes):
    obs = env.reset()
    done = False
    
    while not(done):
        #***YOUR CODE HERE***
        # sample a epsilon-greedy action from the action space
        action = epsilon_greedy(Q,obs,epsilon = e)
        # take the action and get the new observation space
        new_obs, reward, done, info = env.step(action)
        # Q-learning
        
        Q[obs][action] = Q[obs][action] + alpha*(reward + gamma*max(Q[new_obs][:]) - Q[obs][action])
        obs = new_obs 
        
        pi[obs] = np.argmax(Q[obs][:])
        
        if done:
            scores_window.append(reward)       # save last reward received
            if debug_:
              print(reward)
            #print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_window)), end="")
            if episode % 1000 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_window)))
                scores.append( np.mean(scores_window) )  
    
#np.savez('Q_Q-Learning.npz',Q=Q)   # uncomment to save your result
#data = np.load('Q_Q-Learning.npz') # uncomment to load data
#Q = data['Q']                      # uncomment to load data

plot_training(np.arange(0,nr_episodes,1000),scores,fout='Q_Q-Learning.jpg')    
    
ani = animation_result(Q)
print(pi.reshape(4,4))
#%% 
#env.close()
#plt.close()