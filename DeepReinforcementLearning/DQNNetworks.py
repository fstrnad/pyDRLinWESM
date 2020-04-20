"""
Implementation of possible different DQN networks for the Learner. Often also called brain.
All networks are based on the same basic DQN algorithm. It can be enhanced via:
 - Fixed target networks (FDQN)
 - Dueling Network Architectures (Dueling DQN)
 - Distributional DQN (not in the paper!)

@author: Felix Strnad

"""


import numpy as np

import tensorflow as tf
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Flatten, Lambda
from tensorflow.keras.layers import Input, GaussianNoise, Activation
from tensorflow.keras.layers import Add
from tensorflow.python.keras import backend as K
from _pylief import NONE


##### old keras imports:
# import tensorflow as tf
# from tensorflow import keras
# from keras import backend as K
# from keras.optimizers import Adam
# from keras.models import Model, Sequential
# from keras.layers.core import Dense,  Lambda
# from keras.layers import Input, GaussianNoise, Activation
# from keras.layers.merge import Add

HUBER_LOSS_DELTA = 2.0

""" Alternative Loss function """
def huber_loss(y_true, y_pred):
    err = y_true - y_pred

    cond = K.abs(err) < HUBER_LOSS_DELTA
    L2 = 0.5 * K.square(err)
    L1 = HUBER_LOSS_DELTA * (K.abs(err) - 0.5 * HUBER_LOSS_DELTA)

    loss = tf.compat.v1.where(cond, L2, L1)     

    return K.mean(loss)

""""
Basic implementation of DQN Networks. See e.g. Mnih et al. 2015 for reference.
"""
class DQNetwork:
    def __init__(self, state_size, action_size, learning_rate, batch_size=64, noisy_net=False):
        self.state_size=state_size
        self.action_size=action_size
        self.batch_size=batch_size
        self.network_learning_rate=learning_rate
        self.noisy_net=noisy_net

        self.model=self._createModel()
        self.initial_weights=self.model.get_weights()
        self.episode_weights=[]

    def _createModel(self):
        model=Sequential()
        # Still need to think about an appropriate upset of the network. 
        # We know that initializer should not be zero, to enable back-propagation
        # see https://cs231n.github.io/neural-networks-2/#init , try e.g. glorot_normal
        model.add(Dense(units=256, activation='relu', input_dim=self.state_size, kernel_initializer='RandomUniform'))
        #Further hidden layer
        model.add(Dense(units=256, activation='relu',  kernel_initializer='RandomUniform'))
        model.add(Dense(units=256, activation='relu',  kernel_initializer='RandomUniform'))
        # The final layer will consist of only action_size neurons, one for each available action. 
        # Their activation function will be linear. Remember that we are trying to approximate the Q function, 
        # which in essence can be of any real value.  
        # Therefore we can’t restrict the output from the network and the linear activation works well.
        if self.noisy_net:
            # Add Gaussian noise to output layer!
            model.add(Dense(self.action_size))
            model.add(GaussianNoise(1))  #TODO Check if this stdev=0.1 of the weights is an appropriate choice!
            model.add(Activation('linear'))
        else:
            model.add(Dense(units=self.action_size, activation='linear', kernel_initializer='RandomUniform'))
        
        # Currently the Adam optimizer is the standard optimizer due to better performance than SGD
        opt = Adam(lr=self.network_learning_rate)
        #model.compile(loss='mse', optimizer=opt)
        model.compile(loss=huber_loss, optimizer=opt)    # Use Own defined Huber Loss Function
        return model
    
    # x: Input array , y: Output array
    def train(self, x, y, weights=None, epoch=1, verbose=0):       
        #print(x.shape, weights.shape, self.batch_size)
        # TODO fix the sample_weight mode for DRL with IS!
        loss = self.model.fit(x, y, batch_size=self.batch_size, epochs=epoch, verbose=verbose, sample_weight=None)
        # print("Loss", loss.history['loss'])
        return loss.history['loss']
        
    def predict(self, states):
        return self.model.predict(states)
    
    def predictOne(self, s):
        #print("Prediction for state", s)
        return self.predict(s.reshape(1, self.state_size)).flatten()
    
    def reset_weights_init(self):
        self.model.load_weights('model_DQN.h5')

    def reset_weights(self):
        session = K.get_session()
        for layer in self.model.layers:
            if isinstance(layer, Dense):
                old = layer.get_weights()
                layer.kernel.initializer.run(session=session)
                layer.bias.initializer.run(session=session)
                
                if np.array_equal(old, layer.get_weights()):
                    print(" after initializer run")
                    print(old, layer.get_weights)
            else:
                print(layer, "not reinitialized")
    
    def store_weights(self):
        self.episode_weights.append(self.model.get_weights())
    def delete_stored_weights(self):
        del self.episode_weights[:]
    def set_best_weights(self, index):
        self.model.set_weights(self.episode_weights[index])
        
        
"""
This class allows to implement two separate networks, where the values of the target network
are kept fixed in time. 
The updateTargetModel allows to equal the weights of the model network with the target values. 
"""
class Fixed_targetDQNetwork(DQNetwork):
    def __init__(self, state_size, action_size, learning_rate,batch_size, noisy_net=False):
        DQNetwork.__init__(self, state_size, action_size, learning_rate, batch_size, noisy_net)
        
        self.target_model=self._createModel()
        self.target_initial_weights=self.target_model.get_weights()
        
    def predict(self, s, target=False):
        if target:
            return self.target_model.predict(s)
        else:
            return self.model.predict(s)

    def predictOne(self, s, target=False):
        return self.predict(s.reshape(1, self.state_size), target=target).flatten()

    def updateTargetModel(self):
        self.target_model.set_weights(self.model.get_weights())
        
    def reset_weights(self):
#         print("Full DQN Network reset!")
        session = K.get_session()
#         for model in [self.model, self.target_model]:
#             for layer in model.layers:
#                 if isinstance(layer, Dense):
#                     old = layer.get_weights()
#                     print("layer: ", layer)
#                     if hasattr(layer, 'kernel_initializer'):
#                         layer.kernel.initializer.run(session=session)
#                         layer.bias.initializer.run(session=session)
#                     
#                     if np.array_equal(old, layer.get_weights()):
#                         print(" after initializer run")
#                         print(old, layer.get_weights)
#                 else:
#                     print(layer, "not reinitialized")   
        for model in [self.model, self.target_model]:
            for layer in model.layers:
                if isinstance(layer, tf.keras.Model): #if you're using a model as a layer
                    self.reset_weights(layer) #apply function recursively
                    continue
        
                #where are the initializers?
                if hasattr(layer, 'cell'):
                    init_container = layer.cell
                else:
                    init_container = layer
        
                for key, initializer in init_container.__dict__.items():
                    if "initializer" not in key: #is this item an initializer?
                        continue #if no, skip it
        
                    # find the corresponding variable, like the kernel or the bias
                    if key == 'recurrent_initializer': #special case check
                        var = getattr(init_container, 'recurrent_kernel')
                    else:
                        var = getattr(init_container, key.replace("_initializer", ""))
        
                    var.assign(initializer(var.shape, var.dtype))
                    #use the initializer

        
"""
This is the implementation of Dueling Network architectures. 
For reference see Wang et al. 2015.
"""
class DuellingDDQNetwork(Fixed_targetDQNetwork):
    def __init__(self, state_size, action_size, learning_rate, batch_size, noisy_net=False):
        Fixed_targetDQNetwork.__init__(self, state_size, action_size, learning_rate, batch_size, noisy_net)
        
        self.model=self._createModel()
        self.initial_weights=self.model.get_weights()
        self.target_model=self._createModel()
        self.target_initial_weights=self.target_model.get_weights()
        print("Created Dueling Networks!")

    def _createModel(self):
        
        input=Input(shape=(self.state_size,))
        advt= Dense(units=256, activation='relu')(input)
        if self.noisy_net:
            # Add Gaussian noise to output layer!
            advt=Dense(self.action_size)(advt)
            advt= GaussianNoise(1)(advt)
            #advt= Activation('linear') # TODO Check if this is correct
        else:
            advt= Dense(units=self.action_size)(advt)
        
        #Further layer to estimate state values V(s)
        value=Dense(units=256, activation='relu')(input)
        value=Dense(1)(value)

        advt= Lambda(lambda advt: advt - tf.reduce_mean(input_tensor=advt, axis=-1, keepdims=True))(advt)
        
        final=Add()([value, advt])
        model=Model( inputs=input, outputs=final)
        
        opt = Adam(lr=self.network_learning_rate)

        #model.compile(loss='mse', optimizer=opt)
        model.compile(loss=huber_loss, optimizer=opt)    # Use Own defined Huber Loss Function

        #model.save_weights('model_DQN.h5')
        return model  
    

class NoisyDQNetwork(Fixed_targetDQNetwork):
    def __init__(self, state_size, action_size, learning_rate, batch_size):

        Fixed_targetDQNetwork.__init__(self, state_size, action_size, learning_rate, batch_size)
        
        self.model=self._createModel()
        self.initial_weights=self.model.get_weights()
        self.target_model=self._createModel()
        self.target_initial_weights=self.target_model.get_weights()
        print("Created Dueling Networks!")


    def _createModel(self):
        model=Sequential()

        model.add(Dense(units=256, activation='relu', input_dim=self.state_size, kernel_initializer='RandomUniform'))
        #Further hidden layer
        model.add(Dense(units=256, activation='relu',  kernel_initializer='RandomUniform'))
        
        # Add Gaussian noise to output layer!
        model.add(Dense(self.action_size))
        model.add(GaussianNoise(1))
        model.add(Activation('linear'))
        
        #model.add(Dense(units=self.action_size, activation='linear', kernel_initializer='RandomUniform'))
        
        # Currently the Adam optimizer is the standard optimizer due to better performance than SGD
        opt = Adam(lr=self.network_learning_rate)
        #model.compile(loss='mse', optimizer=opt)
        model.compile(loss=huber_loss, optimizer=opt)    # Use Own defined Huber Loss Function
        return model
    

"""
This is the implementation of the distributional DQN Algorithm.
It is based on Bellmare et al. 2017. However, this version was not investigated
in the current version of the paper.
""" 
class CategoricalDQN(Fixed_targetDQNetwork):
    def __init__(self, state_size, action_size, learning_rate, batch_size=64, num_atoms=51, noisy_net=False):
        Fixed_targetDQNetwork.__init__(self,state_size, action_size, learning_rate, batch_size, noisy_net)

        self.num_atoms=num_atoms
    
        self.model=self._createC51Model()
        self.initial_weights=self.model.get_weights()
        self.target_model=self._createC51Model()


        
    def _createC51Model(self):
        
        #print(self.state_size)
        state_input = Input(shape=(self.state_size,)) 
        
        if self.noisy_net==True:
            distr= Dense(512)(state_input)
            distr= Dense(self.action_size)(state_input)
            distr= GaussianNoise(1)(distr)
            distr= Activation('relu')(distr)
        else:
            distr= Dense(512, activation='relu')(state_input)
        
        
        distribution_list = []
        for i in range(self.action_size):
            distribution_list.append(Dense(self.num_atoms, activation='softmax')(distr))

        model = Model(inputs=state_input, outputs=distribution_list)   # Here we use the new api of keras, the input and output are replaced with inputs and outputs

        opt = Adam(lr=self.network_learning_rate)
        model.compile(loss='categorical_crossentropy',optimizer=opt)

        return model


    def predictOne(self, s, target=False):
        return self.predict(s.reshape(1, self.state_size), target=target)
    
        # x: Input array , y: Output array
    def train(self, x, y, weights=None, epoch=1, verbose=0):
        loss = self.model.fit(x, y, batch_size=self.batch_size, epochs=epoch, verbose=verbose, sample_weight=weights)
        return loss.history['loss']

    
class CategoricalDuelingDQN(CategoricalDQN):
    
    def __init__(self, state_size, action_size, learning_rate, batch_size=64, num_atoms=51, noisy_net=False):
        CategoricalDQN.__init__(self,state_size, action_size, learning_rate, batch_size, num_atoms, noisy_net=False)

        self.model=self._createCategoricalDuelingModel()
        self.initial_weights=self.model.get_weights()
        self.target_model=self._createCategoricalDuelingModel()
    
    def _createCategoricalDuelingModel(self):
        input=Input(shape=(self.state_size,))
        advt= Dense(units=256, activation='relu')(input)
        if self.noisy_net:
            # Add Gaussian noise to output layer!
            advt=Dense(self.action_size)(advt)
            advt= GaussianNoise(1)(advt)
            #advt= Activation('linear') # TODO Check if this is correct
        else:
            advt= Dense(units=self.action_size)(advt)
        
        #Further layer to estimate state values V(s)
        value=Dense(units=256, activation='relu')(input)
        value=Dense(1)(value)

        advt= Lambda(lambda advt: advt - tf.reduce_mean(input_tensor=advt, axis=-1, keepdims=True))(advt)
        
        final=Add()([value, advt])
        
#         model=Model( inputs=input, outputs=final)
#         
#         opt = Adam(lr=self.network_learning_rate)
# 
#         model.compile(loss=huber_loss, optimizer=opt)    # Use Own defined Huber Loss Function
#         
#                 #print(self.state_size)
#         state_input = Input(shape=(self.state_size,)) 
#         
#         distr= Dense(512, activation='relu')(state_input)
        
        
        distribution_list = []
        for i in range(self.action_size):
            distribution_list.append(Dense(self.num_atoms, activation='softmax')(final))

        
        model = Model(inputs=input, outputs=distribution_list)   # Here we use the new api of keras, the input and output are replaced with inputs and outputs

        opt = Adam(lr=self.network_learning_rate)
        model.compile(loss='categorical_crossentropy',optimizer=opt)

        return model