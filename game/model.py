# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 11:46:20 2020

@author: Edouard
"""

import random, math
import numpy as np
from keras import backend as K
import keras
import keras.optimizers as Kopt
#from keras.layers import Convolution2D
from keras.layers import Dense, Input #,Dropout, Flatten
from keras.models import Model

#a changer
ModelsPath = "models"

#Parametres pour le jeu et l'entrainement

LoadWeithsAndTest = False  #Validate model, no training
LoadWeithsAndTrain = False  #Load model and saved agent and train further
render = True      #Diplay game while training
LEARNING_RATE = 0.01    
MEMORY_CAPACITY = int(1e4) 
BATCH_SIZE = 32            

#hyperparametre huber loss
HUBER_LOSS_DELTA = 1.0

#actions :left, right, throttle, brake, ebrake 

action_buffer = np.array([#definir toutes les actions ici
        [0.,1.,1.,0.,0.], #Action 1 : droite
        [1.,0.,1.,0.,0.],
        [0.,1.,0.,1.,0.],
        [1.,0.,0.,1.,0.],
        [0.,0.,1.5,0.,0.],
        [0.,0.,0.,1.,0.]] )   

NumberOfDiscActions = action_buffer.shape[0]



def SelectAction(Act):
    return action_buffer[Act]

def SelectArgAction(Act):
    for i in range(NumberOfDiscActions):
        if np.all(Act == action_buffer[i]):
            return i
    raise ValueError('SelectArgAction: Act not in action_buffer')


def huber_loss(y_true, y_pred):
    err = y_true - y_pred

    cond = K.abs(err) <= HUBER_LOSS_DELTA
    if cond == True:
        loss = 0.5 * K.square(err)

    else:
        loss = 0.5 * HUBER_LOSS_DELTA**2 + HUBER_LOSS_DELTA * (K.abs(err) - HUBER_LOSS_DELTA)

    return K.mean(loss)


class Brain:

    def __init__(self, state_Input_shape , action_Shape):
        
        self.state_Input_shape = state_Input_shape 
        self.action_Shape = action_Shape
        
        self.model = self._createModel() # behavior network
        self.model_ = self._createModel()  # target network ( pour calculer la target )

        self.ModelsPath_cp = ModelsPath + "\DDQN_model_cp.h5"
        self.ModelsPath_cp_per = ModelsPath+"\DDQN_model_cp_per.h5"
        
        
        #save les modeles dans des fichiers (le best + periodique)
        save_best = keras.callbacks.ModelCheckpoint(self.ModelsPath_cp,
                                                monitor='loss',
                                                verbose=1,
                                                save_best_only=True,
                                                mode='min',
                                                period=20)
        
        save_per = keras.callbacks.ModelCheckpoint(self.ModelsPath_cp_per,
                                                monitor='loss',
                                                verbose=1,
                                                save_best_only=False,
                                                mode='min',
                                                period=400)
        
#        early_stop = keras.callbacks.EarlyStopping(monitor='loss',
#                                           min_delta=0.001,   
#                                           patience=0,
#                                           verbose=1,
#                                           mode='auto')
    
        self.callbacks_list = [save_best, save_per]#, early_stop]
  
    #modele avec les CNN
      
    def _createModel(self):
        
        brain_in = Input(shape=(self.state_Input_shape,), name='brain_in')
        
        x = brain_in
        #x = Dense(100, activation='Selu')(x)
        x = Dense(50, activation='selu', kernel_initializer = keras.initializers.lecun_uniform() )(x)
        x = Dense(20, activation='selu', kernel_initializer = keras.initializers.lecun_uniform())(x)
        y = Dense(self.action_Shape, activation="linear", kernel_initializer = keras.initializers.lecun_uniform())(x)
        
        model = Model(inputs=brain_in, outputs=y)
        
        self.opt = Kopt.RMSprop(lr=LEARNING_RATE)
        #self.opt = Kopt.RMSprop()
        
        #'mean_squared_error'
        model.compile(loss=huber_loss, optimizer=self.opt)
        model.summary()
        #plot_model(model, to_file='brain_model.png', show_shapes = True)
    
        return model
 
    
    def train(self, x, y, epochs=1, verbose=0):
        self.model.fit(x, y, batch_size=(BATCH_SIZE), epochs=epochs, verbose=verbose, callbacks=self.callbacks_list)
        
   
    #prédit la sortie (les Q values) d'un state à l'aide d'un des réseaux
    def predict(self, s, target=False):
        if target:
            return self.model_.predict(s)
        else:
            return self.model.predict(s)
    
    
    def predictOne(self, s, target=False):
        
        s = np.array([s])
        return self.predict(s, target)
    
    
    def updateTargetModel(self):
        self.model_.set_weights(self.model.get_weights())
        
        
        
from collections import deque
        

class Memory:   # stocké comme ( s, a, r, s_ ) 
        
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        
        #assert(len(self.memory)>batch_size)
        
        samples = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)
        return states, actions, rewards, next_states, dones

        
ACTION_REPEAT = 8
MAX_NB_EPISODES = int(10e3) 
MAX_NB_STEP = ACTION_REPEAT * 100
GAMMA = 0.99   
UPDATE_TARGET_FREQUENCY = int(200)  
EXPLORATION_STOP = int(MAX_NB_STEP*10)    
LAMBDA = - math.log(0.001) / EXPLORATION_STOP   # speed of decay fn of episodes of learning agent
MAX_EPSILON = 0.99
MIN_EPSILON = 0.01
TRAIN_EVERY = 5

class Agent:
    steps = 0
    epsilon = MAX_EPSILON
    memory = Memory(MEMORY_CAPACITY)
    
    def __init__(self, state_Input_len , action_len):
        
        self.state_Input_shape = state_Input_len
        self.action_Shape = action_len
        
        self.state = None
        self.action = None
        self.reward = None
        self.next_state = None
        self.nb_training = 0
        
        self.rewards = []
        
        self.brain = Brain(self.state_Input_shape, self.action_Shape)
        
        self.no_state = np.zeros(state_Input_len)
        self.no_action = np.zeros(action_len)
        
        #entrée du réseau sur un batch
        self.x = np.zeros([BATCH_SIZE,state_Input_len])
        #les targets de ce même batch
        self.y = np.zeros([BATCH_SIZE,action_len])     
        
        self.errors = np.zeros(BATCH_SIZE)
        self.rand = False
        
        self.agentType = 'Learning'
        self.maxEpsilone = MAX_EPSILON
        
        
        #take action with epsilon policy
        #retourne l'action ainsi que son argument qui appartient à (0,..,len(actions)-1)
    def act(self, s):
        
        if  random.random()  < self.epsilon:
            arg_act = np.random.randint(self.action_Shape)
            self.rand=True
            return SelectAction(arg_act), arg_act
        else:
            #x = s[np.newaxis,:,:,:]
            prediction = self.brain.predictOne(s)
            arg_act = np.argmax(prediction[0])
            self.rand=False
            return SelectAction(arg_act), arg_act
        
        
    def observe(self):  # in (s, a, r, s_) format

        if self.steps % UPDATE_TARGET_FREQUENCY == 0:
            self.brain.updateTargetModel()
            print ("Target network update")
        # slowly decrease Epsilon based on our eperience
        self.steps += 1
        self.epsilon = MIN_EPSILON + (self.maxEpsilone - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)


    def _getTargets(self, batch):
        #les states de départ 
        states = np.array(batch[0][:])
        #les next_states 
        states_ = np.array(np.array(agent.no_state) if batch[3][:] is None else np.array(batch[3][:]))
        
        states = states.reshape((BATCH_SIZE,self.state_Input_shape))
        states_ = states_.reshape((BATCH_SIZE,self.state_Input_shape))
        
        predictions = self.brain.predict(states)
        
        pTarget_ = self.brain.predict(states_, target=True)
        predictions_ = self.brain.predict(states_, target=False)
        
        
        for i in range(len(batch[0])): #batch[0] car batch = [tous les x, toutes les actions, tous les x']
            
            initial_state = batch[0][i]
            
            action = batch[1][i]; reward = batch[2][i]; next_state = batch[3][i] 
            arg_action = SelectArgAction(action)
            
            original_Qvalue = predictions[i]
            
            #calcul des targets
            if next_state is None:
                original_Qvalue[arg_action] = reward
            else:
                original_Qvalue[arg_action] = reward + GAMMA * pTarget_[i][ np.argmax(predictions_[i]) ]  # double DQN
            
            self.x[i] = initial_state
            self.y[i] = original_Qvalue
            
            #sec_best = t[np.argsort(t)[::-1][1]]
            if self.steps % 20 == 0 and i == len(batch)-1:
                print('Qvalue',original_Qvalue[arg_action], 'reward: %.4f' % reward,'mean Qvalue',np.mean(original_Qvalue))
                
        return (self.x, self.y)


    def replay(self):    
        
        batch = self.memory.sample(BATCH_SIZE)     
        x, y = self._getTargets(batch)
        self.nb_training +=1
        print("entrainement :", self.nb_training)
        self.brain.train(x, y)
        
        
    def step(self, state, reward, running):
        
        done = running
        if self.steps % ACTION_REPEAT ==0:
            if self.state is None :
                self.state = state
                action, arg_action = self.act(state)
                self.action = action
                is_active = True
                self.observe()
                return action, is_active
                
            elif self.next_state is None :
                self.next_state = state
                self.reward = reward
                action, arg_action = self.act(state)
                self.action = action
                is_active = True
                self.observe()
                return action, is_active
            else :
                self.state = self.next_state
                self.reward = reward
                self.next_state = state
                self.memory.add(self.state,self.action,reward,self.next_state,done)
                if self.steps % TRAIN_EVERY ==0 and len(self.memory.memory) > BATCH_SIZE:
                    self.replay()
                action, arg_action = self.act(state)
                self.action = action
                is_active = True
                self.observe()
                if done : 
                    self.state = None
                    self.next_state = None 
                    self.reward = 0
                return action, is_active
            
        else :
            is_active = False
            action = self.no_action
            self.steps +=1
            return  action, is_active
        
        
import math


if __name__ == '__main__':
    
    #test de Brain : 
    x= np.array([[1,2,3,4,5,6,7]])
    x_= np.array([[1,2,3,4,5,6,7],[1,2,3,4,5,6,7],[1,2,3,4,5,6,7],[1,2,3,4,5,6,7],[1,2,3,4,5,6,7]])
    
    brain = Brain(7, 5)
    
    #predictions
    print(brain.predictOne(x))
    
    
    print(brain.predictOne(x, target = True))
    
    print(brain.predict(x_))
    print(brain.predict(x_, target = True))
    
    
    
    #test de Agent :
    agent = Agent(7,6)
    
    agent.brain.model.predict(x_)
    
    action, arg_action = agent.act(x)
    
    print("action decidee par l'agent :",action)
    print("arg action decidee par l'agent :",arg_action)
   
    target = np.array([1,2,3,4,5])
    #arbitrary values
    reward = 12
    
    for i in range(BATCH_SIZE):
        agent.memory.add(x,action,reward,x,False)
    
    sample = agent.memory.sample(BATCH_SIZE)
    print(sample)
    agent.observe()
    
    for i in range(1500):
        action, active = agent.step(x,12)
        print("step :", agent.steps)
        print("is active:", active)
    #( s, a, r, s_ ) in sumTree
    #get a sample
    
    