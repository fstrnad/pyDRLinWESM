"""
This is the implementation of the Agent class. 

The agent requires a memory taken from the memory class and a Network taken from the DQNNetwork class.
The basic reinforcement learning algorithm is the same for all possible improvements of the learning agent.

@author: Felix Strnad
@version: 1.0

You can use the following reference:
Felix M. Strnad, Wolfram Barfuss, Jonathan F. Donges and Jobst Heitzig, 
Deep reinforcement learning in World-Earth system models to discover sustainable management strategies,
Chaos, 2019

"""


import numpy as np
import sys
import matplotlib.pyplot as plt

from DeepReinforcementLearning.memories import PERISMemory, PERMemory, Memory
import tensorflow as tf
from packaging import version

if version.parse(tf.__version__)>=version.parse('2.0'): 
    from DeepReinforcementLearning.DQNNetworks_tf2 import DuellingDDQNetwork, Fixed_targetDQNetwork
else:    
    from DeepReinforcementLearning.DQNNetworks import DuellingDDQNetwork, Fixed_targetDQNetwork

import random

from timeit import default_timer as timer
import os

import AYS.AYS_Environment as ays_env
test_sp=[(0.3, 0.8, 0.5)  ]

from inspect import currentframe, getframeinfo

def print_debug_info():
    frameinfo = getframeinfo(currentframe())
    print ("Error in: ", frameinfo.filename, frameinfo.lineno)


# We need this random Agent to initialize the SumTree with values
# We need this random Agent to initialize the SumTree with values
class randomAgent(object):

    def __init__(self, memory, action_size, prior_exp_replay ):
        self.memory=memory
        self.action_size=action_size
        self.experiences=0
        self.prior_exp_replay=prior_exp_replay
    
    def act(self):
        return random.randint(0, self.action_size-1)
    
    def observe(self, sample):
        # We use the reward as an initializer for a good estimate for first guess
        error=abs(sample[2])   # add reward as first guess for priority
        if self.prior_exp_replay:
            self.memory.add(error,sample)
        else:
            self.memory.add(sample)
        self.experiences+=1
    
    def replay(self):
        pass 


#----------------Agent Class ----------------------
class DQNLearner(object):
    
    def __init__(self, my_Env=None, dt=1,  episodes=1000, max_steps=1000, 
                     gamma=0.9, explore_start = 1.0  , explore_stop = 0.01,  decay_rate=0.01 ,
                     Boltzmann_prob=False, reward_type='PB', Update_target_frequency=200,
                     learner_type='dqn', prior_exp_replay=False, dueling=False, importance_sampling=False, noisy_net=False,
                     memory_size = 1000000, batch_size = 64 , network_learning_rate=0.0025, plot_progress=False,
                     dirpath=None, learning_analysis_folder=None, run_number=0, noise_strength=0.0,
                     ):
        '''
        Data management
        '''
        self.Boltzmann_prob=Boltzmann_prob
        if dirpath==None:
            self.dirpath =os.path.dirname(os.path.abspath( __file__ ))
        else:
            self.dirpath=dirpath
        self.learning_analysis_folder=learning_analysis_folder
        
        print("current directory is : " + self.dirpath)
        self.result_folder=learner_type
        if prior_exp_replay:
            self.result_folder+='_per'
            if importance_sampling:
                self.result_folder+='_is'
        if dueling:
            self.result_folder+='_duel'        
        if noisy_net:
            self.result_folder+='_noisy_net'

        if Boltzmann_prob:
            self.results_path=self.dirpath+'/'+self.result_folder+ '/boltzmann/' + reward_type 
        else:
            self.results_path=self.dirpath+'/'+self.result_folder+ '/epsilon_greedy/' + reward_type
        print("Learner: Store results in:" + self.results_path)

        self.max_data=10000
        self.run_number=run_number  # To make trial runs distinguishable
        #print("Run number: " + str(self.run_number))
        self.plot_progress=plot_progress
        
        '''
        Environment
        '''
        if my_Env==None:
            self.my_Env=ays_env.AYS_Environment(dt=dt, reward_type=reward_type)
        else:
            self.my_Env=my_Env
        self.my_Env.reward_type=reward_type,
        self.my_Env.image_dir=self.results_path
        self.my_Env.run_number=self.run_number
        
        print( "Using the following environment:", type(my_Env))

        self.reward_type=reward_type
        observation_space=self.my_Env.observation_space
        self.state_size = len(observation_space)
        self.action_set=self.my_Env.action_space
        self.action_size=len(self.action_set)
        
        '''
        Memory
        '''        
        self.prior_exp_replay=prior_exp_replay
        self.importance_sampling=importance_sampling
        self.memory_size=memory_size
        if prior_exp_replay:
            if importance_sampling:
                self.memory=PERISMemory(int(memory_size))
            else:
                self.memory=PERMemory(int(memory_size))
        else:
            self.memory=Memory(int(memory_size))
        
        '''
        Network
        '''
        self.noisy_net=noisy_net
        self.batch_size=batch_size
        self.network_learning_rate= network_learning_rate
        if dueling:
            self.Network=DuellingDDQNetwork(self.state_size, self.action_size, self.network_learning_rate, self.batch_size, self.noisy_net)
        else:
            # In Fixed target Network is unspecified target Network included!
            self.Network=Fixed_targetDQNetwork(self.state_size, self.action_size, self.network_learning_rate, self.batch_size, self.noisy_net)
        
        
        self.Update_target_frequency=Update_target_frequency
        self.target_network = True
        self.next_Qs=self._get_next_Q(learner_type)
        if learner_type=='dqn':
            self.target_network = False
            self.Update_target_frequency=None
        

        self.my_Env.reset()
        
        '''        
        Learning / Simulation
        '''
        self.BestFinalLearner=False
        self.dt=dt
        self.total_episodes = episodes      # total number of episodes
        self.max_steps=max_steps            # steps per episode
        
        # Hyperparameters
        self.gamma= gamma
        
        # Exploration parameters for epsilon greedy strategy
        self.explore_start = explore_start      # exploration probability at start
        self.explore_stop = explore_stop        # minimum exploration probability 
        self.decay_rate = decay_rate            # exponential decay rate for exploration prob
        self.decay_step = 0
        
        ''' 
        Results / Analysis
        '''
        self.noise_strength=noise_strength
        self.loss=[]
        self.learning_progress=[]                 # Collects all gained rewards for estimating the total reward
        self.total_temporal_difference=0          # Here all temporal-difference values will be stored
        self.total_learned_episodes=0             # Here all learned episodes are stored, even if they are passed in different learning calls.
        self.mean_reward=0
        self.survived_steps=[]
        self.tested_survived_steps=[]     
        self.av_tot_reward=[]
        self.test_tot_reward=[]
        self.best_reward_arr=[]    # For finding at the end weights with the best reward
        self.av_loss=[]     # For loss function
        self.test_q_states=[]  # For average Q-value development of spec. states
        
        # Timing
        self.start_time=self.time_last_benchmark=timer()
        self.this_episode=0
        self.last_step_number=0
        self.episode_count=0
        
        self.V_max=0
        self.V_min=0
    
    def init_memory(self):    
        dummyRandomAgent=randomAgent(self.memory, self.action_size, self.prior_exp_replay)
        while dummyRandomAgent.experiences < self.memory_size:
            
            state = self.my_Env.reset()
            
            for step in range (self.max_steps):
                #Increase decay_step
                action=dummyRandomAgent.act()
                #action=0
                next_state, reward, done = self.my_Env.step(action)
                if done:
                    next_state=None

                dummyRandomAgent.observe((state, action, reward, next_state) )
                if dummyRandomAgent.experiences >= self.memory_size:
                    break
                if done:
                    break
                state=next_state
#             if self.plot_progress:
#                 print("Init Memory: ", dummyRandomAgent.experiences, "/", self.memory_size)
        
        self.memory=dummyRandomAgent.memory
        print("Memory is initialized!")
    
    """
    This is the essential learning method for the agent. It implements the learning process of the agent.
    It could be implemented outside as well. It uses the defining functions of Memory, Network, Agent and Environment!
    """
    def learn(self, episodes=0):
        # Initialize the decay rate (that will use to reduce epsilon) 
        #self.decay_step = 0
        if(episodes==0):
            episodes=self.total_episodes
        
        av_survived_steps=0
        av_tot_reward= 0
        av_num_episodes=100
         
        del self.best_reward_arr[:]
        self.Network.delete_stored_weights()
        for episode in range(episodes):
            # Make a new episode and observe the first state
            self.this_episode=episode
            state = self.my_Env.reset()
            total_reward=0
            self.decay_step +=1
            self.total_learned_episodes +=1
            learning_progress=[]   
            good_final_state=False
            
            if self.plot_progress:
                if episode==0:
                    print("episode: ", self.episode_count)
            self.episode_count+=1 
            
            for step in range (self.max_steps):                
                if not self.Boltzmann_prob:
                    # Predict the action according to epsilon-greedy policy
                    action=self.act( state)
                else:
                    # Predict action according to a Boltzmann-probability (=softmax)
                    action=self.act_Boltzmann_approach(state)
                
                #Perform the action and get the next_state, reward, and done information
                next_state, reward, done = self.my_Env.step(action)
                
                # Use only if we want to visualize the trajectory of 1 single episode!
#                 if self.plot_progress:
#                     learning_progress.append([state.tolist(), action, reward])

                total_reward +=reward
                
                if done or step==self.max_steps - 1:
                    #self.my_Env.plot_stock_dynamics()
                    # The episode ends so no next state
                    if self.my_Env._good_final_state():
                        reward=self._calculate_expected_final_reward(step)
                        # reward=self.my_Env.calculate_expected_final_reward(gamma=self.gamma, max_steps=self.max_steps, current_step=step)
                        good_final_state=True
                        
                    
                    self.observe((state, action, reward, None))   # If final state, next state is None!
                    self.replay()
                    self.last_step_number=step
                    state=None
                    
                    av_survived_steps+=step
                    av_tot_reward += total_reward
                    if self.episode_count%av_num_episodes == 0:
                        _test_survived_steps, test_total_reward =self._test_survived_steps()
                        self.survived_steps.append([self.episode_count, av_survived_steps/av_num_episodes])    # store in array
                        self.tested_survived_steps.append([self.episode_count, _test_survived_steps])    # store in array
                        
                        self.av_tot_reward.append([self.episode_count, av_tot_reward/av_num_episodes])
                        self.test_tot_reward.append([self.episode_count,test_total_reward])
                        
                        this_av_lost=sum(self.loss)/av_survived_steps
                        self.av_loss.append(this_av_lost/av_num_episodes)
                        
#                         V_av=self.get_test_q_values()
#                         self.test_q_states.append([self.episode_count, V_av])
                            
                        av_survived_steps=0   # reset counter
                        av_tot_reward=0 # reset av total reward                    
                        self.loss=[]    # reset loss function array
                        
                        # Only for this episode plotting!
#                         if self.plot_progress:
#                             self.plot_learning()
#                             
#                             #self.plot_Q_developement()
#                             #self.plot_loss_function(av_num_episodes)
#                             self.my_Env.plot_run(learning_progress)
#                             self.my_Env.plot_2D(learning_progress,
#                                                 self.results_path  )
#                             self.my_Env.save_traj_final_state(learning_progress, self.results_path, self.episode_count)
                    
                    break
                else: #if step < max_observe_step:
                    # Add experienced transition <st,at,reward_t,st+1> to memory
                    self.observe((state, action, reward, next_state))
                    
                ### LEARNING PART            
                self.replay()
                
                # next_state (s_t+1) is now our current state
                state = next_state  
                
            #self.benchmark()

        if self.plot_progress:    
            self.plot_learning()
            self.plot_loss_function(av_num_episodes)
            self.plot_Q_developement()

    """
    Here the only three methods needed for RFL are implemented:
        - act(state)
        - observe(sample)
        - replay()
    """   

    def act(self, state):
        """
        This function will do the part of the DQN-based prediction of the best next action
        With prob epsilon select a random action , otherwise select at=argmaxaQ(st,a)
        """
        
        if self.noisy_net:
            # In noisy net exploration is implicitly guaranteed by the noise on the output layer 
            action=self.get_optimal_action(state)
        else:
            # EPSILON GREEDY STRATEGY:Choose action a from state s using epsilon greedy.
            exploration_exploitation_tradeoff = np.random.rand()
            # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
            explore_probability = self.explore_stop + (self.explore_start - self.explore_stop) * np.exp(-self.decay_rate * self.decay_step)
            
            if (explore_probability > exploration_exploitation_tradeoff):
                # Make a random action (exploration)
                action = self._act_random()
            else:
                # Get action from Q-network (exploitation)
                action=self.get_optimal_action(state)
    
        return action
    
    
    def get_optimal_action(self, state):
        # Get action from Q-network.  Estimate the Qs values state according to the network. 
        # print("State: ", state)
        Qs = self.Network.predictOne(state)
        
        thisQmax=np.max(Qs)
        thisQmin=np.min(Qs)
        if thisQmax > self.V_max:
            self.V_max=thisQmax
        if thisQmin < self.V_min:
            self.V_min=thisQmin
            
#         print(self.V_min, self.V_max)
        # Take the highest possible Q value (= the best action)!
        action =np.argmax(Qs)
        return action
    
    def observe(self, sample):  # in (s, a, r, s_) format
        x, y, errors = self._get_targets([sample])
        #print(errors)
        if self.prior_exp_replay:
            self.memory.add(errors[0], sample)
        else:
            self.memory.add(sample)
        
        if self.target_network:
            if self.episode_count % self.Update_target_frequency == 0:
                self.Network.updateTargetModel()    
    
    
    def replay(self):
        batch_idx, batch, ISWeights = self.memory.sample(self.batch_size)
        x_train, y_train, errors = self._get_targets(batch)
        
        if self.prior_exp_replay:
            #update errors
            for i in range(len(batch)):
                self.memory.update(batch_idx[i], errors[i])
                
        this_loss= self.Network.train(x=x_train, y=y_train, weights=ISWeights)      

        self.loss.append( this_loss[0] )
    
    """
    This functions are only help functions to structure the RFL-needed functions above better. 
    However they are necessary for the logic of RFL algorithms.
    """            
    
    def _get_targets(self, batch):
        """
        In this function we prepare the training data set for the replay function. 
        It is dependent on the chosen Network and the Memory.
        """
        batchLen = len(batch)
        
        no_state = np.zeros(self.state_size)    # To account for final states
        #print(batch[0])
        states = np.array([ o[0] for o in batch ])
        next_states = np.array([ (no_state if o[3] is None else o[3]) for o in batch ])  # 3 ist action_size
        # Create training data set, 
        x = np.zeros((batchLen, self.state_size))
        y = np.zeros((batchLen, self.action_size))
         
        pred_Qs=self.Network.predict(states)
        # This is essentially the only difference between DQN and FDQN
        pred_next_Qs=self.Network.predict(next_states, target=True)

#         if self.target_network:
#             pred_next_Qs=self.Network.predict(next_states, target=True)
#         else:
#             pred_next_Qs=None
        
        # Use new weights as target function for the NEXT (!) state. 
        pred_nextDouble_Qs=self.Network.predict(next_states, target = False)
        
        #print("This Q:" , pred_Qs)
        #print("Next Q: ", pred_next_Qs)
        errors=np.zeros(batchLen)

        # Here do Experience Replay algorithm
        for i in range(batchLen):
            # o: observation, s: state, s_: next state, a: action, t: target, 
            o = batch[i]
            s = o[0]; a = o[1]; r = o[2]; s_ = o[3]
            
            t = pred_Qs[i]
            oldVal=t[a]
            if s_ is None:
                t[a] = r
            else:
                # DQN-Learning, possibly with extensions to basic DQN
                t[a] = r + self.gamma * self.next_Qs(pred_next_Qs[i], pred_nextDouble_Qs[i])
                
            x[i] = s
            y[i] = t
            errors[i] = abs(oldVal - t[a])
        
        return (x, y, errors)
    
    def _get_next_Q(self, learner_type):
        '''
        pred_next_Qs is from target network, pred_nextDouble_Qs is without target network 
        '''
        def simple_dqn(pred_next_Qs, pred_nextDouble_Qs):
            next_Q = np.amax(pred_nextDouble_Qs)  
            return next_Q
        
        def fixed_target(pred_next_Qs, pred_nextDouble_Qs):
            return np.amax(pred_next_Qs)
            
        def double_dqn(pred_next_Qs, pred_nextDouble_Qs):
        # Create training data set, 
            best_action=np.argmax(pred_nextDouble_Qs)
            return pred_next_Qs[best_action]
        
        if learner_type=='dqn':
            return simple_dqn
        elif learner_type=='fdqn':
            return fixed_target
        elif learner_type=='ddqn':
            return double_dqn
        elif learner_type=='c51':
            return None
        else:
            print("ERROR! The learner function you chose is not available! " + learner_type)
            print_debug_info()
            sys.exit(1)
    

    def _act_random(self):
        action_number = np.random.randint(self.action_size)
        return action_number
    
    def act_Boltzmann_approach(self,state):
        """
        This function will choose actions based on Boltzmann-distributed probabilities. 
        This is the most physical understanding of choosing an action.
        No information is lost, since we exploit all the information present in the estimated Q-values 
        produced by our network.
        
        Parameters:
        state
        beta
        """
        beta=1 # Might differ for different environment specifications
        Qs = self.Network.predictOne(state)
        
        expOAvs = np.ma.filled(np.exp(beta * Qs), 0)
        print(expOAvs)
        assert not np.any(np.isinf(expOAvs)), "action policy contains infinite!"
        As = np.arange(len(Qs))
        
        current_act = np.random.choice(As, p=(expOAvs / np.sum(expOAvs)))   # softmax probability
        
        return current_act
        
    
    def _dynamic_alpha(self):
        alpha_min=0.1
        alpha_max=1.
        
        alpha = alpha_min + (alpha_max - alpha_min) * np.exp(- self.decay_step*1./self.tau_alpha)

        return alpha
    
    
    def _calculate_expected_final_reward(self, current_step=0):
        """
        Get the reward in the last state, expecting from now on always default.
        This is important since we break up simulation at final state, but we do not want the agent to 
        find trajectories that are close (!) to final state and stay there, since this would
        result in a higher total reward.
        """
        reward_final_state=self.my_Env.reward_function()   
        remaining_steps=self.max_steps - current_step
        discounted_future_reward=0.
        for i in range(remaining_steps):
            discounted_future_reward += self.gamma**(i) * reward_final_state
        #print("Discounted future reward: " + str(discounted_future_reward))
        
        return discounted_future_reward
        
    
    """
    This function is only needed if we perform more than one (independent!) learning process.
    """    
    def reset_learner(self, run_number=None):
        self.Network.reset_weights()
        self.init_memory()  # Reinitialize memory for a full reset of the learner as well!
        self.decay_step=0
        
        # Every reset leads to a new learning process
        del self.survived_steps[:]          
        del self.tested_survived_steps[:]
        del self.test_tot_reward[:]
        del self.av_tot_reward[:]
        del self.loss[:]
        del self.av_loss[:]
        del self.test_q_states[:]
        
        self.episode_count=0
        
        # To get Q value also for first learned episode
#         V_av=self.get_test_q_values()
#         self.test_q_states.append([self.episode_count, V_av])
        
        if run_number is not None:
            self.run_number=run_number
            self.my_Env.run_number=run_number
       
       
    """
    These functions following from here, are only needed for evaluation and plotting and thus have no influence on the learning process itself.
    """
    def _test_survived_steps(self):
        test_points=self.my_Env.define_test_points()

        total_steps=0
        tot_reward=0
        which_result=[]
        
        last_state=self.my_Env.state
        
        for start_state in test_points:
            state=self.my_Env.reset_for_state(start_state)
        
            last_step=0
            for step in range(self.max_steps):
                # Take the biggest Q value (= the best action)!
                action =self.get_optimal_action(state)
                                        
                # Do the new chosen action in Environment
                new_state, reward, done = self.my_Env.step(action)
                tot_reward +=reward
                state=new_state
                if done:
                    which_result.append(self.my_Env._which_final_state())
                    last_step=step
                    break
            total_steps +=last_step
        
        av_survived_steps=total_steps/(len(test_points))
        av_tot_reward = tot_reward/(len(test_points))
        
        # Reset the state of the Environment to the state which it had before the _test_survived_steps call.
        self.my_Env.state = last_state
        
        #print(which_result)
        #print(av_survived_steps, av_tot_reward)
        
        return av_survived_steps, av_tot_reward

    
    
    def test_on_current_state(self, save_plot=True, show_plot=False):

        state=self.my_Env.reset_for_state()
        
        learning_progress=[]
        for step in range(self.max_steps):
            list_state=self.my_Env.get_plot_state_list()
            # Take the biggest Q value (= the best action)!
            action =self.get_optimal_action(state)
#             print(step, action)                      
            # Do the new chosen action in Environment
            new_state, reward, done = self.my_Env.step(action)
            
            if save_plot or show_plot:
                learning_progress.append([list_state, action, reward])  # TODO Fix for noisy state!
                
            state=new_state
            if done:
                break
        
        if show_plot:
            self.my_Env.plot_run(learning_progress)
        if save_plot:
            # Can also be done outside the IF condition
            self.my_Env.save_traj_final_state(learning_progress, self.results_path, self.episode_count)
            
        return(self.my_Env._which_final_state() )
        
        
    def get_test_q_values(self):
        test_states=self.my_Env.test_Q_states()
        V_tot=0
        for state in test_states:
            
            Qs= self.Network.predictOne(np.array(state)) 
            V=np.max(Qs)
            V_tot+=V
        V_tot/=len(test_states)
        return V_tot
    
    def plot_learning(self):
        #print(self.learning_progress)
        if self.learning_analysis_folder is None:
            save_path = self.results_path +'/survived_progress/' 
        else:
            save_path = self.results_path +'/survived_progress/' + self.learning_analysis_folder +'/'
        
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
            
        with open(save_path +str (self.run_number) + '_'+'survived_progress'+ '_noise' + str("%.2f" %self.noise_strength) +'.txt', 'w') as f:
            for i in range(len(self.survived_steps)):
                f.write("%s  %s  %s  %s  %s \n" % (self.survived_steps[i][0] ,
                                          self.survived_steps[i][1], 
                                          self.tested_survived_steps[i][1],
                                          self.av_tot_reward[i][1],
                                          self.test_tot_reward[i][1]) )
        f.close()
                    
    def plot_loss_function(self, av_num_episodes):
        #print(self.learning_progress)
        if self.learning_analysis_folder is None:
            save_path = self.results_path +'/survived_progress/' + str (self.run_number) + '_'
        else:
            save_path = self.results_path +'/survived_progress/' + self.learning_analysis_folder +'/' +str (self.run_number) + '_'

        with open(save_path + 'loss_function'+ '_noise' + str("%.2f" %self.noise_strength) +'.txt', 'w') as f:
            for i in range(len(self.av_loss)):
                f.write("%s  %s \n" % ( i*av_num_episodes,
                                        self.av_loss[i] ) )
        f.close()
        
        plt.figure(figsize=(10,7))
        time=np.arange(len(self.av_loss))
        plt.plot(50*time, self.av_loss, '-', color='orange',lw=1, label='Loss function ' + self.reward_type)
        
        plt.xlabel('# Episodes')
        plt.ylabel('Average Loss')
        plt.yscale('log')
        #plt.xlim(0,self.episode_count)
        plt.legend(loc='upper left', fontsize=14)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path + 'loss_function'+ '_noise' + str("%.2f" %self.noise_strength) +'.pdf')
        #plt.show()
        plt.close()
    
    
    def plot_Q_developement(self):
        # All states are normalized between [0,1]
        if self.learning_analysis_folder is None:
            save_path = self.results_path +'/survived_progress/' + str (self.run_number) + '_'
        else:
            save_path = self.results_path +'/survived_progress/' + self.learning_analysis_folder +'/' +str (self.run_number) + '_'
            
        with open(save_path +'av_q'+ '_noise' + str("%.2f" %self.noise_strength) +'.txt', 'w') as f:
            for i in range(len(self.test_q_states)):
                f.write("%s  %s \n" % (self.test_q_states[i][0] ,
                                       self.test_q_states[i][1]) )
        f.close()
        
        plt.figure(figsize=(10,6))
        
        plt.plot(*zip(*self.test_q_states), 'x-', color='olive',lw=1, label='Average max(Q(s,a))' + self.reward_type)
        
        plt.xlabel('# Episodes')
        plt.ylabel('Average state action value $Q(s,a)$')
        plt.xlim(0,self.episode_count)
        plt.legend(loc='upper left', fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path + 'av_q'+ '_noise' + str("%.2f" %self.noise_strength) +'.pdf')
        #plt.show()
        plt.close()
        
    

    """
    This function is only needed to evaluate the performance of the agent.
    """
    def benchmark(self):
        stoppoint= timer()
        time_passed_so_far= stoppoint-self.start_time
        time_passed_since_last_benchmark=stoppoint-self.time_last_benchmark
        
        print("Episode: " + str(self.this_episode), "Last step number: " + str(self.last_step_number))
        #print("Total passed time : " + str(time_passed_so_far))
        print("Time since last benchmark: " + str(time_passed_since_last_benchmark))
        
        self.time_last_benchmark=stoppoint
        
    
    
    