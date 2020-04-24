"""
This is the analysis script for the DRL agent.
It can be used to run the learner at the AYS, the c:GLOBAL and the EXPLOIT environment.
This script can be used to analyze various success rates, 
either for single learning algorithm or to evaluate the best parameters. 
The results are stored in single text files, in which all essential hyper-parameters are stored as well. 

@author: Felix Strnad

"""


from __future__ import print_function
import numpy as np

import os, sys
 
import matplotlib 

import pandas as pd
import datetime

from DeepReinforcementLearning.DQNLearner import DQNLearner 
from DeepReinforcementLearning.C51Learner import C51Learner

from DeepReinforcementLearning.Basins import Basins

import c_global.partially_observable_cG as part_obs_c_global
import AYS.AYS_Environment as ays

import c_global.PB_cG as PB_cG_env






# Just for better debugging in the cluster-submitting
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


Env='AYS'
dt=1
max_steps=600 
if Env=='AYS':
    Update_target_frequency=100
    explore_start = 1.0 
    explore_stop = 0.01 
    decay_rate=0.001 
    memory_size = int(1e5)   # This number might need further exploration!
    V_max=35
    V_min=-20
    learning_analysis_folder='AYS'
    observables=dict(A=True,
                     Y=True,
                     S=True,
            )
    reward_types=['PB' ,'survive','final_state', 'ren_knowledge' , 'desirable_region',  ]
    #reward_types=['survive','final_state', 'ren_knowledge' , 'desirable_region', 'PB'   ]
#     reward_types=['PB' ,'final_state', 'ren_knowledge' , 'desirable_region','survive',  ]

elif Env=='cG':
    Update_target_frequency=200
    explore_start = 1.0 
    explore_stop = 0.00001 
    decay_rate=0.001 
    memory_size = int(1e5)   # This number might need further exploration!
    V_max=26
    V_min=-5
    learning_analysis_folder='LAGTPKS'
    observables=dict(   L=True,  
                    A=True,
                    G=True,
                    T=True,
                    P=True,
                    K=True,
                    S=True,
            )
    reward_types=['PB'  ,'survive','final_state', 'ren_knowledge' , 'desirable_region',  ]

else:  
    print("ERROR! The Environment you chose is not available: ", Env)

   
batch_size = 64     # Number of experiences the Batch replay can keep
network_learning_rate= 0.00025

Boltzmann_prob=False

parameters=['trials', 'successful_test', 'brown_fix_point','episodes',
             'gamma', 'update_frequency', 'batch_size',
             'reward_type', 'observables_states', 'noise_strength']


def test_episodes_continue_learning ( trials, max_episodes, episode_steps,
                                      gamma, batch_size, 
                                      Boltzmann_prob, reward_type,  Update_target_frequency, noise_strength,
                                      learner_type, prior_exp_replay, dueling, importance_sampling, noisy_net,
                                      index):

    
    test_type='episodes'

    if Env=='AYS':
        full_observables=dict( A=True, Y=True, S=True  )
        my_Env=ays.AYS_Environment(dt=dt, reward_type=reward_type)
        dirpath='../AYS'
    elif Env=='cG':
        full_observables=dict( L=True, A=True, G=True, T=True, P=True, K=True, S=True  )
        
        my_Env=part_obs_c_global.cG_LAGTPKS_Environment(dt=dt, reward_type=reward_type, )
        dirpath='../c_global/noisy_part_observable_c_global'
    
    else:
        print("ERROR! The environment you chose is not available!", Env)
        sys.exit(1)        
    

    observed_states=''
    for str_state in my_Env.observed_states():
        observed_states+=str_state
    
    print("Episodes: " + str(max_episodes) +
          "gamma: " + str(gamma) + 
          "batch_size: " + str(batch_size)+ 
          "observables: " + str(my_Env.observed_states))

    plot_progress=True
    if learner_type=='c51':
        num_atoms=25
        dqn_agent=C51Learner(my_Env=my_Env, dt=dt,  episodes=episode_steps, max_steps=max_steps, 
                         gamma=gamma, explore_start = explore_start  , explore_stop = explore_stop,  decay_rate=decay_rate ,
                         Boltzmann_prob=Boltzmann_prob, reward_type=reward_type, Update_target_frequency=Update_target_frequency,
                         learner_type=learner_type, prior_exp_replay=prior_exp_replay, dueling=dueling, importance_sampling=importance_sampling,noisy_net=noisy_net,
                         memory_size = memory_size, batch_size = batch_size , network_learning_rate=network_learning_rate, plot_progress=plot_progress,
                         dirpath=dirpath, learning_analysis_folder=observed_states, noise_strength=noise_strength, num_atoms=num_atoms,
                         V_max=V_max, V_min=V_min)
    
    else:
        dqn_agent=DQNLearner(my_Env=my_Env, dt=dt,  episodes=episode_steps, max_steps=max_steps, 
                         gamma=gamma, explore_start = explore_start  , explore_stop = explore_stop,  decay_rate=decay_rate ,
                         Boltzmann_prob=Boltzmann_prob, reward_type=reward_type, Update_target_frequency=Update_target_frequency,
                         learner_type=learner_type, prior_exp_replay=prior_exp_replay, dueling=dueling, importance_sampling=importance_sampling, noisy_net=noisy_net,
                         memory_size = memory_size, batch_size = batch_size , network_learning_rate=network_learning_rate, plot_progress=plot_progress,
                         dirpath=dirpath, learning_analysis_folder=observed_states, noise_strength=noise_strength)
   
    num_runs= int(max_episodes/episode_steps) + 1
    print("Trials: " + str(trials) + " Runs: " + str(num_runs))
    successful_test=np.zeros((num_runs))
    brown_fix_point=np.zeros((num_runs))
    
    observed_states=''
    for str_state in my_Env.observed_states():
        observed_states+=str_state
    results_path=dqn_agent.results_path

    for trial in range(trials):
        run_number=index*trials + trial
        # Here for every test the agent is new initialized
        dqn_agent.reset_learner(run_number=run_number)
#         dqn_agent.learn(1)  # start learning process
        run_episodes=0

        print("Run number: " + str(dqn_agent.run_number))
        for run in range(num_runs ):
            # Test the agents behavior 
            result=dqn_agent.test_on_current_state(save_plot=True, show_plot=True)  # save_plot=True if we Trajectory of test should be stored!
    
            # Here we train the agent with runs times episodes, we do this after evaluation to get the untrained agent as well.
            # After last evaluation we don't need to learn again.
            if run < num_runs-1:
                dqn_agent.learn()    
    

       
    print("Episodes Learning: Stored results into " + results_path)

    print("Vmax: ", dqn_agent.V_max, "Vmin: ", dqn_agent.V_min)
    del dqn_agent
        
        
def test_episodes_continue_learning_reduce_states (  trials, max_episodes, episode_steps,
                                      gamma, batch_size, 
                                      Boltzmann_prob, reward_type,  Update_target_frequency, noise_strength,
                                      learner_type, prior_exp_replay, dueling, importance_sampling, noisy_net,
                                      index):

    
    test_type='episodes'

    if Env=='AYS':
        full_observables=dict( A=True, Y=True, S=True  )
        dirpath='../AYS/noisy_part_observable_AYS'
    elif Env=='cG':
        full_observables=dict( L=True, A=True, G=True, T=True, P=True, K=True, S=True  )
        
        PB_Environment=False
        if PB_Environment:
            dirpath='../c_global/PB_cG'
        else:
            dirpath='../c_global/noisy_part_observable_c_global'
    elif Env=='exploit':
        dirpath='../exploit'
    else:
        print("ERROR! The environment you chose is not available!")
        sys.exit(1)  

    keys=list(full_observables.keys())
    for reduce in range(len(full_observables)):
        if Env=='AYS':
            my_Env=ays.noisy_partially_observable_AYS(dt=dt, reward_type=reward_type, observables=full_observables, noise_strength=noise_strength)
        elif Env=='cG':
            PB_Environment=False
            if PB_Environment:
                my_Env=PB_cG_env.PB_cG(dt=dt, reward_type=reward_type, 
                                        noise_strength=noise_strength)
            else:
                my_Env=part_obs_c_global.noisy_partially_observable_LAGTPKS(dt=dt, reward_type=reward_type,  
                                                                        observables=full_observables, noise_strength=noise_strength)
        else:
            print("ERROR! The environment you chose is not available!")
            sys.exit(1)
        

        observed_states=''
        for str_state in my_Env._observed_states():
            observed_states+=str_state
        
        print("Episodes: " + str(max_episodes) +
              "\n gamma: " + str(gamma) + 
              "\n batch_size: " + str(batch_size)+ 
              "\n observables: " + str(observed_states))
    
        plot_progress=True
        if learner_type=='c51':
            num_atoms=25
            dqn_agent=C51Learner(my_Env=my_Env, dt=dt,  episodes=episode_steps, max_steps=max_steps, 
                             gamma=gamma, 
                             explore_start = explore_start  , explore_stop = explore_stop,  decay_rate=decay_rate ,
                             Boltzmann_prob=Boltzmann_prob, reward_type=reward_type, Update_target_frequency=Update_target_frequency,
                             learner_type=learner_type, prior_exp_replay=prior_exp_replay, dueling=dueling, importance_sampling=importance_sampling,noisy_net=noisy_net,
                             memory_size = memory_size, batch_size = batch_size , network_learning_rate=network_learning_rate, plot_progress=plot_progress,
                             dirpath=dirpath, learning_analysis_folder=observed_states, noise_strength=noise_strength, num_atoms=num_atoms,
                             V_max=V_max, V_min=V_min)
        
        else:
            dqn_agent=DQNLearner(my_Env=my_Env, dt=dt,  episodes=episode_steps, max_steps=max_steps, 
                             gamma=gamma, 
                             explore_start = explore_start  , explore_stop = explore_stop,  decay_rate=decay_rate ,
                             Boltzmann_prob=Boltzmann_prob, reward_type=reward_type, Update_target_frequency=Update_target_frequency,
                             learner_type=learner_type, prior_exp_replay=prior_exp_replay, dueling=dueling, importance_sampling=importance_sampling, noisy_net=noisy_net,
                             memory_size = memory_size, batch_size = batch_size , network_learning_rate=network_learning_rate, plot_progress=plot_progress,
                             dirpath=dirpath, learning_analysis_folder=observed_states, noise_strength=noise_strength)
       
        num_runs= int(max_episodes/episode_steps) + 1
        print("Trials: " + str(trials) + " Runs: " + str(num_runs))
        successful_test=np.zeros((num_runs))
        brown_fix_point=np.zeros((num_runs))
        
        observed_states=''
        for str_state in my_Env._observed_states():
            observed_states+=str_state
        results_path=dqn_agent.results_path
    
        for trial in range(trials):
            run_number=index*trials + trial
            run_episodes=0
            # Here for every test the agent is new initialized
            dqn_agent.reset_learner(run_number=run_number)
            dqn_agent.learn(1)  # start learning process
            print("Run number: " + str(dqn_agent.run_number))
            for run in range(num_runs ):
                # Test the agents behavior 
                result=dqn_agent.test_on_current_state(False, show_plot=True)
                if result==Basins.GREEN_FP:
                    successful_test[run]+=1
                elif result==Basins.BROWN_FP:
                    brown_fix_point[run]+=1

                learning_results = pd.DataFrame(columns=parameters)
        
                file_type=(test_type+'/'+ observed_states + '/raw/result_' + str(index) 
                                    +'_episodes'+str(run_episodes) +'_gamma'+str("%.2f" %gamma)
                                    +'_udf' + str(Update_target_frequency)
                                    +'_batch_size' + str(batch_size)  + '_noise' + str("%.2f" %noise_strength)
                                    +'.csv')
            
                tmp_data=pd.DataFrame([ [trial+1, successful_test[run], run_episodes, gamma , 
                                              Update_target_frequency, batch_size, reward_type, full_observables, noise_strength ] ], columns=parameters)
                learning_results=learning_results.append(tmp_data)
                learning_results.to_csv(results_path + '/'+file_type, sep='\t', index=False)
                
                run_episodes=run*episode_steps
    
                # Here we train the agent with runs times episodes, we do this after evaluation to get the untrained agent as well.
                # After last evaluation we don't need to learn again.
                if run < num_runs-1:
                    dqn_agent.learn()    
        
        print("Episodes Learning: Stored results into " + results_path)
    
        print("Vmax: ", dqn_agent.V_max, "Vmin: ", dqn_agent.V_min)
        del dqn_agent

        full_observables[keys[reduce]]=False

def run_all_tests(trials, episodes, episode_steps, gamma, batch_size ,
                  Boltzmann_prob,Update_target_frequency, noise_strength,
                  prior_exp_replay, importance_sampling, noisy_net,
                  index ):
    
#     learner_types=[ 'dqn', 'ddqn', 'fdqn','c51', ]
    learner_types=[ 'ddqn', 'dqn','fdqn','c51', ]
    #learner_types=[ 'c51', 'dqn', 'fdqn', 'ddqn',  ]
    only_full_state=True
    for dueling in [True, False]:
        for reward_type in reward_types:
            for learner_type in learner_types:
                if only_full_state==True:
                    test_episodes_continue_learning(trials, episodes, episode_steps, gamma, batch_size, 
                                                    Boltzmann_prob, reward_type, Update_target_frequency, noise_strength,
                                                    learner_type, prior_exp_replay, dueling, importance_sampling, noisy_net, index )
                else:
                    test_episodes_continue_learning_reduce_states(trials, episodes, episode_steps, gamma, batch_size, 
                                                                  Boltzmann_prob, reward_type, Update_target_frequency, noise_strength, 
                                                                  learner_type, prior_exp_replay, dueling, importance_sampling, noisy_net, index)
 


def main(argv):
    eprint('Simulation started at ', datetime.datetime.now())
    
    job_id= int (argv[0])
    print("Running with manual id: ", job_id)
    
    gamma=0.96
    Boltzmann_prob=False
      
    trials=10
    episodes=7000
    episode_steps=500  
    noise_strength=0.00
    
    prior_exp_replay=True
    importance_sampling=True
    noisy_net=False
    
    run_all_tests(trials, episodes, episode_steps, gamma, batch_size, 
                  Boltzmann_prob, Update_target_frequency, noise_strength, 
                  prior_exp_replay, importance_sampling, noisy_net, job_id)
      
    
if __name__ == "__main__":
    main(sys.argv[1:])
    
    






