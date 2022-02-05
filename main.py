import numpy as np

import tensorflow as tf



import cv2
import slitherioenv as env
import warnings

env = env.env()


import time
from keras.layers.core import Dense, Activation, Flatten

import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Conv2D,Flatten






state_size = (1,4,88,192)
action_size =1
scores = []

from agent_ddpg import DDPG
from collections import deque
ACTOR_LR = 1e-4
CRITIC_LR = 1e-3
RANDOM_SEED = 42
MU = 0.0
THETA = 0.15
SIGMA = 0.3
BUFFER_SIZE = 1e6
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 1e-3
EPSILON_START = 1.0
EPSILON_MIN = 0.1
EPSILON_DECAY = 1e-6
N_TIME_STEPS = 1
N_LEARN_UPDATES = 1

if tf.test.is_gpu_available():
    DEVICE = "/GPU:0"
else:
    DEVICE = "/device:CPU:0"
    agent = DDPG(state_size, action_size, ACTOR_LR, CRITIC_LR,
             RANDOM_SEED, MU, THETA, SIGMA, BUFFER_SIZE, BATCH_SIZE,
             EPSILON_START, EPSILON_MIN, EPSILON_DECAY,
             GAMMA, TAU, N_TIME_STEPS, N_LEARN_UPDATES, DEVICE)
print_every=100
n_episodes=100000

scores_deque = deque(maxlen=print_every)
states_deque=deque(maxlen=4)
scores = []
def resizeİmg(img):
    scale_percent = 10
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    width = int(gray.shape[1] * scale_percent / 100)
    height = int(gray.shape[0] * scale_percent / 100)
    dim = (width, height)
    print(gray.shape[1], gray.shape[0])
      
    # resize image
    resized = cv2.resize(gray, dim, interpolation = cv2.INTER_AREA)
    state=resized.reshape(88,192)
    
    return state
for i_episode in range(1, n_episodes+1):
    time.sleep(4)
    agent.reset()
    states_deque=deque(maxlen=4)
    state = env.reset()
    
    

    state=resizeİmg(state)
    for i in range(4):
        
        states_deque.append(state)
    next_states_deque=states_deque
    score = 0
    t = 0
    
 

    while(True):
        t += 1
        action = agent.act(np.array(states_deque).reshape(1,4,88,192))
        print(action)




        next_state, reward, done = env.step(action.reshape(1,1))
      
        next_state=resizeİmg(next_state)
        next_states_deque.append(next_state)
        

        
        agent.step(t, np.array(states_deque).reshape(1,4,88,192), action, reward, np.array(next_states_deque).reshape(1,4,88,192), done)
        state = next_state
        
        states_deque=next_states_deque
       
        if done:
             print("bitti step:",t)
             break 
   
    print('\rEpisode {}'.format(i_episode), end="")
    if i_episode%100==0:
        agent.actor_local.model.save('checkpoint_actorcnn')
        agent.critic_local.model.save('checkpoint_criticcnn')
        print("kaydedildi")

            
        


"""

from tensorflow.keras.models import load_model

trained_model = load_model('checkpoint_actor')
for i_episode in range(5):
    state = env.reset()
    state=state[200:600,200:600]
    cv2.imshow("state",state)
    cv2.waitKey(0)
    state=state.reshape(1,3,400,400)
    state=model.predict(state)
    score = 0.0
    
    t = 0
    while(True):
        t += 1
        env.render()
        
        state  = np.expand_dims(state, axis=0)
        action = trained_model(state)
        action = action.numpy()[0]
        
        action = action.clip(-1, 1)
        
        
        state, reward, done= env.step(action)
        state=state[200:600,200:600]
        cv2.imshow("state",state)
        cv2.waitKey(0)

        state=state.reshape(1,3,400,400)
        state=model.predict(state)
        
        score += reward
        if done:
            break
    
    print("Episode {0} finished after {1} timesteps. Total score: {2}".format(i_episode+1, t+1, score))
"""
env.close()
