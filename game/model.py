import random, math
import numpy as np
from keras import backend as K
from keras import regularizers
import keras
import keras.optimizers as Kopt
from keras.layers import Dense, Input 
from keras.models import Model
import os
from SumTree import SumTree

ModelsPath = "models"
sep = os.path.sep
#Parametres pour le jeu et l'entrainement
ACTION_REPEAT = 8 #la voiture décide une action toutes les ACTION_REPEAT frames
GAMMA = 0.95 #paramétre pondération des récompenses dans le temps
UPDATE_TARGET_FREQUENCY = int(200) #frequence mise a jour du réseau target
EXPLORATION_STOP = int(5000) #arret de l'exploration (epsilon minimal)
LAMBDA = - math.log(0.001) / EXPLORATION_STOP #speed of decay fn of episodes of learning agent
MAX_EPSILON = 0.99
MIN_EPSILON = 0.01
TRAIN_EVERY = 1 #frequence d'entrainement par rapport aux actions prises
INPUT_LEN = 12
LEARNING_RATE = 0.001
MEMORY_CAPACITY = int(3e4) #taille mémoire
BATCH_SIZE = 32            
#hyperparametre huber loss
HUBER_LOSS_DELTA = 1.5

#actions :left, right, throttle, brake, ebrake 

action_buffer = np.array([#definir toutes les actions ici
        [0.,0.6,0.,0.,0.], #Action 1 : droite
        [0.6,0.,0.,0.,0.],
        [0.,0.,1.5,0.,0.],
        [0.,0.,0.,1.5,0.]] ) 

NumberOfDiscActions = action_buffer.shape[0]


"""
retourne le nombre de tentative que l'agent à effectué
"""
def get_nb_tentative(path):
    f = open(path+"nb_tentative.txt", "r")
    var = f.read()
    var = var.split(" ")
    print("exploration stop :", var[1])
    print("nb_tentative :", var[0])
    nb_tentative = int(var[0])
    f.close()
    return nb_tentative


def SelectAction(Act):
    return action_buffer[Act]


def SelectArgAction(Act):

    for i in range(NumberOfDiscActions):
        if np.all(Act == action_buffer[i]):
            return i
    raise ValueError('SelectArgAction: Act not in action_buffer')

"""
huber loss utilisée pour le deep-Q-learning.
https://en.wikipedia.org/wiki/Huber_loss
"""
def huber_loss(y_true, y_pred):
    err = y_true - y_pred

    cond = K.abs(err) <= HUBER_LOSS_DELTA
    if cond == True:
        loss = 0.5 * K.square(err)

    else:
        loss = 0.5 * HUBER_LOSS_DELTA**2 + HUBER_LOSS_DELTA * (K.abs(err) - HUBER_LOSS_DELTA)
    return K.mean(loss)


"""
fonction pour recharger le modéle avec parametres epsilon et les poids du réseau
/!\ les poids doivent être les mêmes
"""
def reload_agent_renforcement(path,reload_model,reload_nb_tentative):
    nb_tentative = 0
    if reload_nb_tentative:
        nb_tentative = get_nb_tentative(path)
        agent = Agent(INPUT_LEN,NumberOfDiscActions,nb_tentative)
        agent.load_model(path + "DDQN_model_cp.h5")
    else : 
        agent = Agent(INPUT_LEN,NumberOfDiscActions)
    return agent


"""
classe Brain, contient les deux réseaux de neurones 
ainsi que les paramétres liés à ceux-ci.

entrée : 
    - state_input_shape : taille de l'espace d'état de l'agent
    - action_shape : nombre d'action que peut effectuer l'agent
"""
class Brain:

    def __init__(self, state_Input_shape , action_shape):
        
        self.state_Input_shape = state_Input_shape 
        self.action_shape = action_shape
        
        self.model = self._createModel() # behavior network
        self.model_ = self._createModel()  # target network ( pour calculer la target )

        self.ModelsPath_cp = ModelsPath + "/DDQN_model_cp.h5"
        self.ModelsPath_cp_per = ModelsPath + "/DDQN_model_cp_per.h5"
        
        
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
        x = Dense(40, activation='selu', kernel_initializer = keras.initializers.lecun_uniform(),kernel_regularizer=regularizers.l2(0.0001))(x)
        x = Dense(20, activation='selu', kernel_initializer = keras.initializers.lecun_uniform(),kernel_regularizer=regularizers.l2(0.0001))(x)
        y = Dense(self.action_shape, activation="linear", kernel_initializer = keras.initializers.lecun_uniform())(x)
        
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
        

"""
mémoire qui contient les experiences passés de l'agent.
Lors d'un entrainement nous piochons dans cette mémoire afin 
d'apprendre grâce aux expériences passées.

entrée : capacité de la mémoire
"""

class Memory:   # stored as ( s, a, r, s_, done ) in SumTree
    e = 0.01
    a = 0.6

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def _getPriority(self, error):
        return (error + self.e) ** self.a

    def add(self, error, sample):
        p = self._getPriority(error)
        self.tree.add(p, sample) 

    def sample(self, n):
        batch = []
        segment = self.tree.total() / n

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append( (idx, data) )

        return batch
    
    #juste ça a voir
    def update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)
        
        
"""
classe Agent dotée d'une Memory et d'une Brain.
permet à l'agent d'apprendre à partir de ces deux composants.

"""
class Agent:
    steps = 0
    epsilon = MAX_EPSILON
    memory = Memory(MEMORY_CAPACITY)
    
    def __init__(self, state_Input_len , action_len, nb_tentative = 0):
        
        self.state_Input_shape = state_Input_len
        self.action_shape = action_len
        self.state = None
        self.action = None
        self.reward = None
        self.next_state = None
        self.nb_training = 0
        self.nb_tentative = nb_tentative
        self.total_score = 0
        self.brain = Brain(self.state_Input_shape, self.action_shape)
        
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
            arg_act = np.random.randint(self.action_shape)
            self.rand=True
            return SelectAction(arg_act), arg_act
        else:
            #x = s[np.newaxis,:,:,:]
            prediction = self.brain.predictOne(s)
            arg_act = np.argmax(prediction[0])
            self.rand=False
            return SelectAction(arg_act), arg_act
            

    def observe(self, sample):  # in (s, a, r, s_) format
        x, y, errors = self._getTargets([(0, sample)], False)
        self.memory.add(errors[0], sample)

        if self.nb_training+1 % UPDATE_TARGET_FREQUENCY == 0:
            self.brain.updateTargetModel()
            print ("Target network update")
            #print ("Target network update")
        # slowly decrease Epsilon based on our eperience
        self.steps += 1
        self.epsilon = MIN_EPSILON + (self.maxEpsilone - MIN_EPSILON) * math.exp(-LAMBDA * self.nb_tentative)

    def observe_no_action(self):
        self.steps += 1
        self.epsilon = MIN_EPSILON + (self.maxEpsilone - MIN_EPSILON) * math.exp(-LAMBDA * self.nb_tentative)
        

    def _getTargets(self, batch, train=True):
        #les states de départ 
        states = np.array([ o[1][0] for o in batch ])
        states_ = np.array([ (self.no_state if o[1][3] is None else o[1][3]) for o in batch ])
        
        #states = np.array(batch[0][:])
        #states_ = np.array(np.array(self.no_state) if batch[3][:] is None else np.array(batch[3][:]))
        if train :
            states = states.reshape((BATCH_SIZE,self.state_Input_shape))
            states_ = states_.reshape((BATCH_SIZE,self.state_Input_shape))
        
        predictions = self.brain.predict(states)
        
        pTarget_ = self.brain.predict(states_, target=True)
        predictions_ = self.brain.predict(states_, target=False)
        
        
        for i in range(len(batch)): #batch[0] car batch = [tous les x, toutes les actions, tous les x']
            o = batch[i][1]
            initial_state = o[0]; action = o[1]; reward = o[2]
            #next_state = batch[3][i] 
            
            arg_action = SelectArgAction(action)
            
            original_Qvalue = predictions[i]
            new_value = original_Qvalue
            #calcul des targets
            if o[4]: #if done
                new_value[arg_action] = reward
            else:
                new_value[arg_action] = reward + GAMMA * pTarget_[i][ np.argmax(predictions_[i]) ]  # double DQN
            
            self.x[i] = initial_state
            self.y[i] = new_value
            
            #sec_best = t[np.argsort(t)[::-1][1]]
            if self.steps % 20 == 0 and i == len(batch)-1:
                print('Qvalue',new_value[arg_action], 'reward: %.4f' % reward,'mean Qvalue',np.mean(new_value))
                
            self.errors[i] = abs(original_Qvalue[arg_action] - new_value[arg_action])
        return (self.x, self.y, self.errors)
    

    def replay(self):    
        batch = self.memory.sample(BATCH_SIZE)
        x, y, errors = self._getTargets(batch,True)

        #update errors
        for i in range(len(batch)):
            idx = batch[i][0]
            self.memory.update(idx, errors[i])
            
        self.nb_training +=1
        print("entrainement :", self.nb_training)
        self.brain.train(x, y)
        
    
    def step(self, state, reward, running):
        print("epsilon=",self.epsilon)
        done = not running
        if self.steps % ACTION_REPEAT == 0:
            if self.state is None :
                self.state = state
                action, arg_action = self.act(state)
                self.action = action
                self.total_score += reward
                is_active = True
                self.observe_no_action()
                return action, is_active
                
            elif self.next_state is None :
                self.next_state = state
                self.reward = reward
                self.total_score += reward
                action, arg_action = self.act(state)
                self.action = action
                is_active = True
                self.observe_no_action()
                return action, is_active
            else :
                self.state = self.next_state
                self.reward = reward
                self.total_score += reward
                self.next_state = state
                #error = self.get_error(self.state,self.action,reward,self.next_state,done)
                self.observe((self.state,self.action,reward,self.next_state,done))
                if self.steps > BATCH_SIZE*(ACTION_REPEAT+2):
                    self.replay()
                action, arg_action = self.act(state)
                self.action = action
                is_active = True
                if done :
                    path = "models/"
                    self.save_score(path)
                    self.save_nb_tentative(path) 
                    self.total_score= 0
                    self.state = None
                    self.next_state = None 
                    self.reward = 0
                    self.nb_tentative +=1
                    
                return action, is_active
            
        else :
            if done :
                self.state = self.next_state
                self.reward = reward
                self.total_score += reward
                self.next_state = state
                #error = self.get_error(self.state,self.action,reward,self.next_state,done)
                self.observe((self.state,self.action,reward,self.next_state,done))
                if self.steps > BATCH_SIZE*(ACTION_REPEAT+2):
                    self.replay()
                path = "models/"
                self.save_nb_tentative(path) 
                self.save_score(path)
                self.total_score= 0
                self.state = None
                self.next_state = None 
                self.reward = 0
                self.nb_tentative +=1
                
            is_active = False
            action = self.no_action
            self.steps +=1
            return  action, is_active
        
        
    def load_model(self, model_path):
        print("recuperation du modele ..")
        self.brain.model.load_weights(model_path)
        self.brain.model_.load_weights(model_path)
        
        
        
    def save_nb_tentative(self, path):
        f = open(path +"nb_tentative.txt", "w") # tout écrasé !
        f.write(str(self.nb_tentative))
        f.write(" ")
        f.write(str(EXPLORATION_STOP))
        f.close()
        
        
    def save_score(self, path):
        f2 = open(path +"score.txt", "a") # a la fin
        f2.write(str(self.total_score))
        f2.write(" ")
        f2.close()
        
if __name__ == "__main__" :
    
    action_buffer = np.array([#definir toutes les actions ici
        [0.,0.6,0.,0.,0.], #Action 1 : droite
        [0.6,0.,0.,0.,0.],
        [0.,0.,1.5,0.,0.],
        [0.,0.,0.,1.5,0.]] ) 
    
    agent = Agent(2,6)
    state = [1,2]
    
    action1 = [0.,1.,1.5,0.,0.]
    action2 = [0.,1.,1.5,0.,0.]
    
    reward = 12
    done = False
    agent.observe((state, action1, reward, state, done))
    
    running = True
    for i in range(200):
        agent.step(state, reward, True)
    agent.step(state, reward, False)