#import AYS_Environment as ays_env
import c_global.cG_LAGTPKS_Environment as c_global
import numpy as np
import pandas as pd
import sys,os
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText


pars=dict(  Sigma = 1.5 * 1e8,
            Cstar=5500,
            a0=0.03,
            aT=3.2*1e3,
            l0=26.4,
            lT=1.1*1e6,
            delta=0.01,
            m=1.5,
            g=0.02,
            p=0.04,
            Wp=2000,
            q0=20,
            qP=0.,
            b=5.4*1e-7,
            yE=120,
            wL=0.,
            eB=4*1e10,
            eF=4*1e10,
            i=0.25,
            k0=0.1,
            aY=0.,
            aB=1.5e4,
            aF=2.7e5,
            aR=9e-15,
            sS=1./50.,
            sR=1.,
            ren_sub=.5,
            carbon_tax=.5   ,
            i_DG=0.1, 
            L0=0.3*2480
            )

ics=dict(   L=2480.,  
            A=830.0,
            G=1125,
            T=5.053333333333333e-6,
            P=6e9,
            K=5e13,
            S=5e11
            )

dt=1
reward_type='PB'
my_Env=c_global.cG_LAGTPKS_Environment(dt=dt,pars=pars, reward_type=reward_type, ics=ics, plot_progress=True)


def read_trajectories(learner_type, reward_type, basin, policy='epsilon_greedy', episode=0):
    runs=[]
    # 0_path_[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]_episode0    limit=150
    limit=150
    parameters=['time','L', 'A', 'G', 'T', 'P', 'K', 'S' , 'action' , 'Reward' ]
    for i in range(limit):
        file_name=('./'+learner_type+'/' + policy +'/' +reward_type + '/DQN_Path/' +
                   basin+ '/' + str(i)+'_path_[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]_episode' + str(episode)+'.txt')

        if os.path.isfile(file_name):
            tmp_file= pd.read_csv(file_name, sep='\s+' ,header=None, names=parameters, skiprows=2, index_col=False) # Skiprow=2, since we calculate derived variables first!
            runs.append(tmp_file)
            
#             print(file_name)
        # For not too many files
        if len(runs) > 100:
            break
    print(len(runs))
    
    return runs

def get_LAGTPKS(learning_progress):
    time=learning_progress['time']

    L=learning_progress['L']
    A=learning_progress['A']
    G=learning_progress['G']
    T=learning_progress['T']
    P=learning_progress['P']
    K=learning_progress['K']
    S=learning_progress['S']

    actions= learning_progress['action']

    return time, L,A,G,T,P,K,S,actions


def management_distribution_part(learning_progress_arr, savepath, start_time=0, end_time=20, only_long_times=False):
    
    tot_my_actions=pd.DataFrame(columns=['action'])
    for learning_progress in learning_progress_arr:
        time, L_comp, A_comp, G_comp, T_comp, P_comp, K_comp, S_comp, actions = get_LAGTPKS(learning_progress)
        
        end_time_simulation=time.iloc[-1]
        if only_long_times:
            if end_time_simulation >100:
                print(end_time_simulation)
                my_actions= pd.DataFrame(actions[start_time:end_time])
        else:
            my_actions= pd.DataFrame(actions[start_time:end_time])
            
        tot_my_actions=pd.concat([tot_my_actions, my_actions]).reset_index(drop = True)
       
    tot_my_actions=tot_my_actions.to_numpy()
    d = np.diff(np.unique(tot_my_actions)).min()
    left_of_first_bin = tot_my_actions.min() - float(d)/2
    right_of_last_bin = tot_my_actions.max() + float(d)/2
    #print(d, left_of_first_bin, right_of_last_bin)
    right_of_last_bin = 7.5

    fig, ax= plt.subplots(figsize=(8,5))
    plt.hist(tot_my_actions, np.arange(left_of_first_bin, right_of_last_bin + d, d),density=True, edgecolor='grey',rwidth=0.9)

    plt.xlabel("Action number", fontsize=15)
    plt.ylabel("Probability",fontsize=15)
    #plt.xlim([0,7])
    
    box_text=''
    for i in range(len(c_global.cG_LAGTPKS_Environment.management_options)):
            box_text+=str(i ) + ": " + c_global.cG_LAGTPKS_Environment.management_options[i] +"\n"

    at = AnchoredText(box_text, prop=dict(size=14), frameon=True, 
                      loc='lower left', bbox_to_anchor=(1.0, .02),bbox_transform=ax.transAxes
                  )
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    #ax.axis('off')
    ax.add_artist(at)
    fig.tight_layout()    
    
    fig.savefig(savepath)
    
@np.vectorize
def std_err_bin_coefficient(n,p):
    #print(n,p)
    std_dev=p*(1-p)
    return np.sqrt(std_dev/n)

def action_timeline(savepath, learner_type='ddqn_per_is_duel', reward_type='PB', basin='GREEN_FP',
                    start_time=0, end_time=100):
    

    
    time_arr= np.arange(0,end_time)
    action_names=['default', 'Sub' , 'Tax','NP' , 'Sub+Tax', 'Sub+NP', 'Tax+NP', 'Sub+Tax+NP' ]
    tot_average_actions=pd.DataFrame(columns=action_names)
    tot_average_actions_errors=pd.DataFrame(columns=action_names)
    
    
    for episode in range(5000, 7000, 500):
        learning_progress_arr=read_trajectories(learner_type=learner_type, reward_type=reward_type, 
                                                basin=basin, policy='epsilon_greedy', episode=episode)
    
    print("len learning_arr:", len(learning_progress_arr))
    
    for t in time_arr:
        my_actions=[]
        for learning_progress in learning_progress_arr:
            time, L_comp, A_comp, G_comp, T_comp, P_comp, K_comp, S_comp, actions = get_LAGTPKS(learning_progress)
            end_time_simulation=time.iloc[-1]
            if t<end_time_simulation:
                my_actions.append(actions[t:t+1].values)
                #print(t, actions[t:t+1].values)
        
        #print(t, len(my_actions), my_actions)
        prob,bins=np.histogram(my_actions, np.arange(0,9), density=True)
        prob_arr= pd.Series(prob, index=tot_average_actions.columns)
        prob_err_arr=pd.Series(std_err_bin_coefficient(len(my_actions), prob_arr), index=tot_average_actions_errors.columns)       
        #print( prob_arr, prob_err_arr)
        tot_average_actions=tot_average_actions.append(prob_arr, ignore_index=True)
        tot_average_actions_errors=tot_average_actions_errors.append(prob_err_arr, ignore_index=True)

        #print(tot_average_actions)
        #print('end loop')

    fig, ax= plt.subplots(figsize=(7,4))
    plt.xlabel("Time t [year]", fontsize=12)
    plt.ylabel("Probability",fontsize=12)
    #plt.xlim(0,100)
    for management in reversed(action_names):
        plt.fill_between(time_arr, [sum(x) for x in zip(tot_average_actions[management], tot_average_actions_errors[management])],
                         [a_1 - a_2 for a_1, a_2 in zip(tot_average_actions[management], tot_average_actions_errors[management])],
                         alpha=0.4, )
        plt.plot(time_arr, tot_average_actions[management], label=management)
        
    
#     ax.add_artist(at)
    ax.legend(fontsize=12, fancybox=True, bbox_to_anchor=(1,1.),)
    fig.tight_layout()    
     
    fig.savefig(savepath)


action=0

# ddqn_per_is_duel_OUT_PB=read_trajectories(learner_type='ddqn_per_is_duel', reward_type='PB', basin='OUT_PB', policy='epsilon_greedy', episode=1500)
# ddqn_per_is_duel_GREEN_FP_PB=read_trajectories(learner_type='ddqn_per_is_duel', reward_type='PB', 
#                                                basin='GREEN_FP', policy='epsilon_greedy', episode=8000)
# 
# ddqn_per_is_duel_GREEN_FP_survive=read_trajectories(learner_type='ddqn_per_is_duel', reward_type='survive', 
#                                                     basin='GREEN_FP', policy='epsilon_greedy', episode=8000)


#for i in range(10):
# time, L_comp, A_comp, G_comp, T_comp, P_comp, K_comp, S_comp, actions = get_LAGTPKS(ddqn_per_is_duel_OUT_PB[1])
# my_Env.plot_trajectories_from_data(time, L_comp, A_comp, G_comp, T_comp, P_comp, K_comp, S_comp, actions, './images/phase_space_plots/cG_non_successful_example_trajectory.pdf')
# 
# time, L_comp, A_comp, G_comp, T_comp, P_comp, K_comp, S_comp, actions = get_LAGTPKS(ddqn_per_is_duel_GREEN_FP_PB[19])
# my_Env.plot_trajectories_from_data(time, L_comp, A_comp, G_comp, T_comp, P_comp, K_comp, S_comp, actions, './images/phase_space_plots/cG_successful_example_trajectory.pdf')

#management_distribution_part(ddqn_per_is_duel_OUT_PB, './images/management/cG_non_successful_prob_distribution.pdf', only_long_times=True)
#management_distribution_part(ddqn_per_is_duel_GREEN_FP_PB, './images/management/cG_successful_prob_distribution_PB.pdf')
#management_distribution_part(ddqn_per_is_duel_GREEN_FP_survive, './images/management/cG_successful_prob_distribution_survive.pdf')
action_timeline(savepath='./images/management/cG_action_timeline_PB.pdf',
                learner_type='ddqn_per_is_duel', reward_type='PB', basin='GREEN_FP',)
action_timeline(savepath='./images/management/cG_action_timeline_survive.pdf',
                learner_type='ddqn_per_is_duel', reward_type='survive', basin='GREEN_FP',)
# action_timeline(savepath='./images/management/cG_action_timeline_OUT_PB.pdf',
#                 learner_type='ddqn_per_is_duel', reward_type='PB', basin='OUT_PB',)


plt.show()



#my_Env.plot_trajectories_from_data(*zip(get_LAGTPKS(ddqn_per_is_duel_GREEN_FP_6000[0])), './images/')


