import os, sys
import pandas as pd
import matplotlib.pyplot as plt
SMALL_SIZE = 11
MEDIUM_SIZE = 12
BIGGER_SIZE = 13
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def read_one_trajectory_reward_progress(model='AYS', learner_type='ddqn_per_is_duel', reward_type='survive', policy='epsilon_greedy',states='AYS',
                                           noise_strength=0., full_episode_length=8000, run_idx=0):
    """
    Here we read the trajectories. Attention, in some cases the files are not written properly, s.t.
    some single lines might miss
    """
    runs=[]
    parameter_learning_progress=['episodes' , 'Survived steps' , 'tested_survived_steps', 'Average reward','test_tot_reward']
    limit=100
    episode_length=int (full_episode_length/100)
    file_name=('../' + model+'/'+learner_type+'/' + policy +'/' +reward_type + '/survived_progress/'+ states+'/' + str(run_idx)+
               '_survived_progress'+ '_noise' + str("%.2f" %noise_strength) +'.txt')

    if os.path.isfile(file_name):
        tmp_file= pd.read_csv(file_name, sep='\s+' ,header=None, names=parameter_learning_progress, index_col=False)
        #print(len(tmp_file))
        if len(tmp_file) >= episode_length:
            cut_tmp_file=tmp_file[0:episode_length]
        else:
            cut_tmp_file=tmp_file
    
        return cut_tmp_file
    else:
        print("No such file: ", file_name)
        sys.exit(1)




def read_learning_developement(learner_type, reward_type, policy='epsilon_greedy', noise_strength=0, full_episode_length=10000):
    """
    Here we read the trajectories. Attention, in some cases the files are not written properly, s.t.
    some single lines might miss
    """
    runs=[]
    parameters=['episodes' , 'Survived steps' , 'tested_survived_steps', 'Average reward','test_tot_reward']
    limit=110
    episode_length=int (full_episode_length/100)
    for i in range(limit):
        file_name=('./'+learner_type+'/' + policy +'/' +reward_type + '/survived_progress/AYS/' + str(i)+'_survived_progress'+ '_noise' + str("%.2f" %noise_strength) +'.txt')

        if os.path.isfile(file_name):
            tmp_file= pd.read_csv(file_name, sep='\s+' ,header=None, names=parameters, index_col=False)
            #print(len(tmp_file))
            if len(tmp_file) >= episode_length:
                cut_tmp_file=tmp_file[0:episode_length]
                #print(len(cut_tmp_file), len(tmp_file))
                #print(cut_tmp_file['episodes'].head(), cut_tmp_file['episodes'].tail())
                runs.append(cut_tmp_file)
            runs.append(tmp_file)
            #print(file_name)
    
    print(len(runs))
    
    return runs

def read_one_learning_developement(path, model, learner_type, reward_type, policy='epsilon_greedy', noise_strength=0, run_idx=0):
    """
    Here we read the trajectories. Attention, in some cases the files are not written properly, s.t.
    some single lines might miss
    """
    parameters=['episodes' , 'Survived steps' , 'tested_survived_steps', 'Average reward','test_tot_reward']
    file_name=(path+model+'/'+learner_type+'/' + policy +'/' +reward_type + '/survived_progress/AYS/' + str(run_idx)+'_survived_progress'+ '_noise' + str("%.2f" %noise_strength) +'.txt')

    if os.path.isfile(file_name):
        tmp_file= pd.read_csv(file_name, sep='\s+' ,header=None, names=parameters, index_col=False)
    else:
        print("ERROR! File does not exist!: ", file_name)   
        sys.exit(1) 
        
    
    print(len(tmp_file))
    
    return tmp_file


def plot_one_learning_developement(path, model, learner_type, reward_type, label='', run_index=0):
    plt.figure(figsize=(8,6))
    data=read_one_learning_developement(path, model=model, learner_type=learner_type, reward_type=reward_type)
    episodes=data['episodes']
    survived_steps=data['Survived steps']
    plt.plot(episodes, survived_steps, 'x-', lw=1, label=label)
    
    plt.xlabel('# Episodes', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylabel('Average Reward per Episode', fontsize=16)
    plt.xlim(0,7000)
    plt.legend(loc='upper left', fontsize=14)
    plt.tight_layout()
    #plt.savefig('/plot_folder/learning_success/learning_developement_reward.pdf')
    plt.show()

def plot_reward_developement(model='AYS',learner_type='ddqn_per_is_duel', reward_type='survive', policy='epsilon_greedy', states='AYS',
                                noise_strength=0., full_episode_length=8000, run_idx=0):
    
    reward_progress=read_one_trajectory_reward_progress(model,learner_type, reward_type, policy, states, noise_strength, full_episode_length, run_idx)
    colors=['#e41a1c','#377eb8','#4daf4a']
    
    plt.figure(figsize=(10,6))
    
    #plt.plot(reward_progress['episodes'], reward_progress['Survived steps'], 'x-', color='olive',lw=1, label='Survive Average ' + reward_type)
    #plt.plot(reward_progress['episodes'], reward_progress['tested_survived_steps'], 'x-', color='midnightblue', lw=1, label='Survive Test '+reward_type)
    
    plt.plot(reward_progress['episodes'], reward_progress['Average reward'], 'x-', color=colors[1],lw=1, label='Reward Average ' + reward_type)
    plt.plot(reward_progress['episodes'], reward_progress['test_tot_reward'], 'x-', color=colors[2], lw=1, label='Reward Test '+reward_type)
    
    
    plt.xlabel('# Episodes')
    plt.ylabel('Reward [au]')
    plt.xlim(0,full_episode_length)
    plt.legend(loc='upper left')
    plt.tight_layout()
    #plt.savefig(save_path + 'survived_progress'+ '_noise' + str("%.2f" %self.noise_strength) +'.pdf')
    plt.show()
    
def compare_reward_developement(model='AYS', learner_type1='dqn', learner_type2='ddqn_per_is_duel', 
                                reward_type='survive', policy='epsilon_greedy', states='AYS',
                                noise_strength=0., full_episode_length=8000, run_idx=0):
    reward_progress1=read_one_trajectory_reward_progress(model,learner_type1, reward_type, policy, states, noise_strength, full_episode_length, run_idx)
    reward_progress2=read_one_trajectory_reward_progress(model,learner_type2, reward_type, policy, states, noise_strength, full_episode_length, run_idx)

    colors=['#e41a1c','#377eb8','#4daf4a','#984ea3']
    
    plt.figure(figsize=(10,6))
    
    #plt.plot(reward_progress['episodes'], reward_progress['Survived steps'], 'x-', color='olive',lw=1, label='Survive Average ' + reward_type)
    #plt.plot(reward_progress['episodes'], reward_progress['tested_survived_steps'], 'x-', color='midnightblue', lw=1, label='Survive Test '+reward_type)
    
    #plt.plot(reward_progress1['episodes'], reward_progress1['Average reward'], 'x-', color=colors[2],lw=1, label='Reward Average ' + learner_type1)
    plt.plot(reward_progress1['episodes'], reward_progress1['test_tot_reward'], 'x-', color=colors[0], lw=1, label='Reward Test '+ learner_type1)
    
    #plt.plot(reward_progress2['episodes'], reward_progress2['Average reward'], 'x-', color=colors[3],lw=1, label='Reward Average ' + learner_type2)
    plt.plot(reward_progress2['episodes'], reward_progress2['test_tot_reward'], 'x-', color=colors[1], lw=1, label='Reward Test '+ learner_type2)
    
    plt.xlabel('# Episodes')
    plt.ylabel('Reward [au]')
    plt.xlim(0,full_episode_length)
    plt.legend(loc='upper left')
    plt.tight_layout()
    #plt.savefig(save_path + 'survived_progress'+ '_noise' + str("%.2f" %self.noise_strength) +'.pdf')
    plt.show()
    
    
# plot_reward_developement(model='AYS', learner_type='dqn', reward_type='PB', policy='epsilon_greedy',full_episode_length=6000, run_idx=10, noise_strength=0.00)
# compare_reward_developement(model='AYS', learner_type1='dqn',learner_type2='ddqn_per_is_duel', reward_type='PB', policy='epsilon_greedy',full_episode_length=6000, run_idx=20, noise_strength=0.00)
# plot_one_learning_developement(path='../',model='AYS',learner_type='ddqn_per_is_duel', reward_type='PB',label='ddqn_per_is_duel' )

