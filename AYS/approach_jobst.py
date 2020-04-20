
import numpy as np
from scipy.integrate import odeint

from gym import Env
from gym.spaces import Box, Discrete



class ODESurvivalEnv(Env):
    
    reward_range = (0, 1)
    
    # static parameters:
    actions = None
    """list of possible actions"""
    default_action = None
    rhs = None
    """function (state, t, action) providing RHS of state ODE"""
    lives = None
    """function (state, t) returning True if system still lives in state"""
    observe = None
    """function (state, t) returning observation of same dimension as state"""
    t0 = None
    """initial time"""
    dim = None
    """state space dimension"""
    state0 = None
    """initial state"""
    dt = None
    """time step"""

    # state information:
    t = None
    """last time"""
    state = None
    """last state"""
    
    
    def __init__(self, *, actions, rhs, lives, state0, observe = lambda state, t: state, t0=0, dt=1):
        # TODO: validate args
        self.actions = actions
        self.rhs = rhs
        self.lives = lives
        self.t0 = t0
        self.state0 = state0 = np.array(state0)
        d = self.dim = state0.size
        self.observe = observe
        self.dt = dt
        self.action_space = Discrete(len(actions))
        self.observation_space = Box(low=np.repeat(-np.inf, d), high=np.repeat(np.inf, d))
        
        
    def step(self, action_id):
        """
        Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object): an action provided by the environment
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """

        nextt = self.t + self.dt
        s = self.state = odeint(self.rhs, self.state, [self.t, nextt], (self.actions[action_id],))[-1, :]
        t = self.t = nextt
        observation = self.observe(s, t)
        lives = self.lives(s, t)
        reward = lives * 1
        done = not lives
        info = {}
        return observation, reward, done, info
    
    def reset(self):
        """Resets the state of the environment and returns an initial observation.
        Returns: observation (object): the initial observation of the
            space.
        """
        s = self.state = self.state0
        t = self.t = self.t0
        return self.observe(s, t)

# AYS example from Kittel et al. 2017:

tauA = 50
tauS = 50
beta = 0.03
beta_LG = 0.015
eps = 147
theta = 8.57e-5
rho = 2
sigma = 4e12
sigma_ET = 2.83e12
phi = 4.7e10

AYS0 = [240, 7e13, 5e11]

APB = 345
YSF = 4e13 

def rhs(AYS, t, a):
    """a[0]=True for low growth, a[1]=True for energy transition"""
    A,Y,S = AYS
    U = Y/eps
    Gamma = 1/(1+(S/(sigma_ET if a[1] else sigma))**rho)
    F = Gamma*U
    E = F/phi
    R = U-F
    return [E-A/tauA, (beta_LG if a[0] else beta)*Y-theta*A*Y, R-S/tauS]

env = ODESurvivalEnv(actions=[(False, False), (False, True), (True, False), (True, True)], 
                     rhs=rhs, lives = lambda AYS, t: AYS[0]<APB and AYS[1]>YSF,
                     observe = lambda AYS, t: AYS / (AYS + AYS0),
                     state0=AYS0)

env.reset()

while True:
    observation, reward, done, info = env.step(3)
    print(env.t, env.state, observation, reward)
    if done: break


# learn:

import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt

env.reset()

tf.reset_default_graph()


# Set learning parameters
alpha=0.05 # 0.1
y = .99
e = e0 = 0.3 # 0.1
ityp = 1e100 # 50
jmax = 1000
num_episodes = 2000 #2000

#These lines establish the feed-forward part of the network used to choose actions
inputs1 = tf.placeholder(shape=[1,3],dtype=tf.float32)
W = tf.Variable(tf.random_uniform([3,4],0,.01)) #.01
Qout = tf.matmul(inputs1,W)
predict = tf.argmax(Qout,1)

#Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
nextQ = tf.placeholder(shape=[1,4],dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate=alpha)
updateModel = trainer.minimize(loss)

init = tf.global_variables_initializer() #tf.initialize_all_variables()

#create lists to contain total rewards and steps per episode
jList = []
rList = []
with tf.Session() as sess:
    sess.run(init)
    for i in range(num_episodes):
        #Reset environment and get first new observation
        s = env.reset()
        rAll = 0
        d = False
        j = 0
        #The Q-Network
        freqs = np.zeros(4)
        ahist = np.zeros(jmax)
        dhist = np.zeros((jmax,3))
        while j < jmax:
            j+=1
            #Choose an action by greedily (with e chance of random action) from the Q-network
            a,allQ = sess.run([predict,Qout],feed_dict={inputs1:s.reshape((1,-1))})
            if np.random.rand(1) < e:
                a[0] = env.action_space.sample()
            freqs[a[0]] += 1
            #Get new state and reward from environment
            s1,r,d,_ = env.step(a[0])
            ahist[j-1] = a[0]
            dhist[j-1,:] = s1
            #Obtain the Q' values by feeding the new state through our network
            Q1 = sess.run(Qout,feed_dict={inputs1:s1.reshape((1,-1))})
            #Obtain maxQ' and set our target value for chosen action.
            maxQ1 = np.max(Q1)
            targetQ = allQ
            targetQ[0,a[0]] = r + y*maxQ1
            #Train our network using target and predicted Q values
            _,W1 = sess.run([updateModel,W],feed_dict={inputs1:s.reshape((1,-1)),nextQ:targetQ})
            rAll += r
            s = s1
            if d == True:
                #Reduce chance of random action as we train the model.
                e = e0 / ((i/ityp) + 1)
                break
        jList.append(j)
        rList.append(rAll)
        print("!",i,rAll,freqs)
        if rAll == jmax:
            plt.plot(dhist[:,0],"b")
            plt.plot(dhist[:,0]*0 + APB/(APB+AYS0[0]),"b:")
            plt.plot(dhist[:,1],"k")
            plt.plot(dhist[:,1]*0 + YSF/(YSF+AYS0[1]),"k:")
            plt.plot(dhist[:,2],"y")
            plt.plot(ahist+1,".")
            plt.savefig('./images/sucess_jobst_tensorflow_AYS.pdf')
            plt.show()
print("Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%")



