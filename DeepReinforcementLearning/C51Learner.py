from DeepReinforcementLearning.DQNLearner import DQNLearner
from DeepReinforcementLearning.DQNNetworks import CategoricalDQN, CategoricalDuelingDQN

import numpy as np
import random
import math


"""
Attention this is test code and not fully analyzed and working yet
@todo: Check if this code is really working.
"""

# Attention for the moment PER is not implemented yet!!!
class C51Learner(DQNLearner):
    def __init__(self, my_Env=None, dt=1,  episodes=1000, max_steps=1000, 
                     alpha=1., tau_alpha=0.1,  beta=1., gamma=0.9, 
                     explore_start = 1.0  , explore_stop = 0.01,  decay_rate=0.01 ,
                     Boltzmann_prob=False, reward_type='PB', Update_target_frequency=200,
                     learner_type='c51', prior_exp_replay=False, dueling=False, importance_sampling=False, noisy_net=False,
                     memory_size = 1000000, batch_size = 64 , network_learning_rate=0.0025, plot_progress=False,
                     dirpath=None, learning_analysis_folder=None, run_number=0, noise_strength=0.0, 
                     num_atoms=51,V_max=35, V_min=-20,
                     ):  
        if noisy_net:
            print("ERROR! Currently Noisy Nets are not implemented yet! Please do without!")
            exit(1)
        
        DQNLearner.__init__(self, my_Env, dt, episodes, max_steps,
                            alpha, tau_alpha, beta, gamma, 
                            explore_start, explore_stop, decay_rate, 
                            Boltzmann_prob, reward_type, Update_target_frequency, learner_type, prior_exp_replay, dueling, importance_sampling, noisy_net, 
                            memory_size, batch_size, network_learning_rate, plot_progress, 
                            dirpath, learning_analysis_folder, run_number, noise_strength
                            )
       
        '''
        Network
        '''
        self.num_atoms=num_atoms
        if dueling:
            self.Network=CategoricalDuelingDQN(state_size=self.state_size, action_size=self.action_size, 
                                learning_rate=self.network_learning_rate, batch_size=self.batch_size,
                                num_atoms= self.num_atoms)
        else:
            self.Network=CategoricalDQN(state_size=self.state_size, action_size=self.action_size, 
                                learning_rate=self.network_learning_rate, batch_size=self.batch_size,
                                num_atoms= self.num_atoms)
        
        # This parameters we estimate from previous runs!
        self.V_max = V_max # Max possible Q value inside PB
        self.V_min = V_min # Min possible Q value, 
         
        self.delta_z = (self.V_max - self.V_min) / float(self.num_atoms - 1)
        self.z = [self.V_min + i * self.delta_z for i in range(self.num_atoms)]
    
    def act(self, state):
        """
        Get action from model using epsilon-greedy policy
        """
        # EPSILON GREEDY STRATEGY:Choose action a from state s using epsilon greedy.
        exploration_exploitation_tradeoff = np.random.rand()
        # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
        explore_probability = self.explore_stop + (self.explore_start - self.explore_stop) * np.exp(-self.decay_rate * self.decay_step)
        
        if (explore_probability > exploration_exploitation_tradeoff):
            # Make a random action (exploration)
            action = self._act_random()
        else:
            action=self.get_optimal_action(state)
            
        return action
    

    def get_optimal_action(self, state):
        """
        Get optimal action for a state
        """
        #print("State= ", state)
        z = self.Network.predictOne(state) # Return a list [1x51 times number of possible actions]
        #print("Predicted: ", z)
        z_concat = np.vstack(z)
        q = np.sum(np.multiply(z_concat, np.array(self.z)), axis=1) 

        # Pick action with the biggest Q value
        action_idx = np.argmax(q)
        
        return action_idx
    
    
    def _get_targets(self, batch):
        """
        In this function we prepare the training data set for the replay function. 
        It is dependent on the chosen Network and the Memory.
        """
        batchLen = len(batch)
        
        no_state = np.zeros(self.state_size)    # To account for final states
        #print(batch[0])
        states = np.array([ o[0] for o in batch ])
        next_states = np.array([ (no_state if o[3] is None else o[3]) for o in batch ])  # 3 is next state
        # Create training data set, 
        x = np.zeros((batchLen, self.state_size))
        m_prob = [np.zeros((batchLen, self.num_atoms)) for i in range(self.action_size)]
         
        #print(batch)       
        #print(states, next_states)
        pred_Zs=self.Network.predict(states) 
        
        pred_next_Zs=self.Network.predict(next_states, target=True)

#         if self.target_network:
#             pred_next_Qs=self.Network.predict(next_states, target=True)
#         else:
#             pred_next_Qs=None
        
        # Use new weights as target function for the NEXT (!) state. 
        #pred_nextDouble_Zs=self.Network.predict(next_states, target = False)

        #errors=np.zeros(batchLen)


        # Get Optimal Actions for the next states (from distribution z)
        z_concat = np.vstack(pred_Zs)
        q = np.sum(np.multiply(z_concat, np.array(self.z)), axis=1) # length (num_atoms x num_actions)
        q = q.reshape((batchLen, self.action_size), order='F')
        #print(q)
        optimal_action_idxs = np.argmax(q, axis=1)
        #print(optimal_action_idxs)
        errors=np.zeros(batchLen)
        
        # Here do Experience Replay algorithm
        for i in range(batchLen):
            # o: observation, s: state, s_: next state, a: action, t: target, 
            o = batch[i]
            s = o[0]; 
            a = o[1]; 
            r = o[2]; 
            s_ = o[3]
            p = pred_next_Zs[optimal_action_idxs[i]][i]
            #print("p: " , p)
            #t = pred_Zs[i]
#             oldVal=t[a]
            if s_ is None:   # Terminal State!
                # Distribution collapses to a single point
                Tz = min(self.V_max, max(self.V_min, r))
                bj = (Tz - self.V_min) / self.delta_z 
                m_l, m_u = math.floor(bj), math.ceil(bj)
                m_prob[a][i][int(m_l)] += (m_u - bj)
                m_prob[a][i][int(m_u)] += (bj - m_l)
            else:
                for j in range(self.num_atoms):
                    Tz = min(self.V_max, max(self.V_min, r + self.gamma * self.z[j]))
                    bj = (Tz - self.V_min) / self.delta_z 
                    m_l, m_u = math.floor(bj), math.ceil(bj)
                    # Double DQN - Learning!
                    #print(s, a, r, m_l, m_u, Tz, bj)
                    
                    pj=pred_next_Zs[optimal_action_idxs[i]][i][j]
                    m_prob[a][i][int(m_l)] += pj * (m_u - bj)
                    m_prob[a][i][int(m_u)] += pj * (bj - m_l)
                
                    #print(i, m_l,m_u,  m_prob[a][i][int(m_l)])

            x[i] = s
            #print(m_prob)
            if self.prior_exp_replay:
                # Cross entropy based on 
                errors[i] = -1*np.sum(m_prob[a][i]*np.log(p))
                #print(m_prob[a][i], errors[i])
            else:
                errors=None
        #print(errors)
        return (x, m_prob, errors)
        
        

    def replay(self):
        batch_idx, batch, ISWeights = self.memory.sample(self.batch_size)
       
        
        if self.prior_exp_replay:
            if self.importance_sampling:
                # (samples, sequence_length)
                c51_weights=list(np.repeat(ISWeights[np.newaxis,:], self.action_size, axis=0))
        else:
            c51_weights=ISWeights
        
        #print(c51_weights)

        x_train, y_train, errors = self._get_targets(batch)

        if self.prior_exp_replay:
            #update errors
            for i in range(len(batch)):
                self.memory.update(batch_idx[i], errors[i])
                
        this_loss= self.Network.train(x=x_train, y=y_train, weights=c51_weights)      

        self.loss.append( this_loss[0] )
    
