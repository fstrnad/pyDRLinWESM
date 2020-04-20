import numpy as np
import pandas as pd
import os.path

from matplotlib.font_manager import FontProperties
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText

fontp = {#'family' : 'normal',
        'weight' : 'normal',
        'size'   : 12}
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font', **fontp)

import seaborn as sns

def read_trajectories(file_path, noise_strength=0, full_episode_length=8000):
    """
    Here we read the trajectories. Attention, in some cases the files are not written properly, s.t.
    some single lines might miss
    """
    runs=[]
    parameters=['episodes' , 'Survived steps' , 'tested_survived_steps', 'Average reward','test_tot_reward']
    limit=110
    episode_length=int (full_episode_length/100)
    for i in range(limit):
        file_name=(file_path+ '/' + str(i)+'_survived_progress'+ '_noise' + str("%.2f" %noise_strength) +'.txt')

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
    
    print(file_path, len(runs))
    
    return runs

def read_loss(file_path, noise_strength=0, full_episode_length=8000):
    runs=[]
    limit=100
    episode_length=int(full_episode_length/50)
    parameters=['steps', 'loss']
    for i in range(limit):
        file_name=(file_path+ '/' + str(i)+'_loss_function'+ '_noise' + str("%.2f" %noise_strength) +'.txt')
        if os.path.isfile(file_name) and os.stat(file_name).st_size != 0:
            
            tmp_file= pd.read_csv(file_name, sep='\s+' ,header=None, names=parameters, index_col=False)
            if len(tmp_file) >= episode_length:
                cut_tmp_file=tmp_file[0:episode_length]
                runs.append(cut_tmp_file)

    print(len(runs))
    
    return runs

color_list_LAGTPKS=['orangered','midnightblue','olive' , 'orange', 'darkgoldenrod', 'green', 'tan']
#color_list_learners=['brown', 'orange', 'tan', 'darkcyan','forestgreen', 'blue']
#color_list_learners=['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e']
color_list_learners=['#7fc97f','#beaed4','#fdc086','#ffff99','#386cb0']
#color_list_learners=['#e41a1c','#984ea3','#ff7f00', '#377eb8','#4daf4a',]
#color_list_learners=['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99']

        
def compare_loss(data_arr, label_arr, savepath, LAGTPKS=True):
    if len(data_arr) != len(label_arr):
        print("ERROR! Data Array and label array must have equal length!")
    else:
        #sns.set()
        plt.figure(figsize=(8,6))
        if LAGTPKS==True:
            my_color_list=color_list_LAGTPKS
        else:
            my_color_list=color_list_learners
        steps_arr=[]
        loss_arr=[]
        for idx, data in enumerate(data_arr):
            survived_steps_arr=[]
            steps=data[0]['steps']
            loss_arr=[]
            step_arr=[]
            for i in range(len(data)):
                loss_arr.append(data[i]['loss'])
                step_arr.append(data[i]['steps'])
#            plt.plot(steps,survived_steps1, 'gx:', lw=1, label=label1)
#            plt.plot(steps,survived_steps2, 'bx:', lw=1, label=label2)
            
            """
            The different dataframes are aranged such that even if some steps are missing the values are 
            properly averaged.
            """
            df1=pd.DataFrame(loss_arr).melt()
            df2=pd.DataFrame(step_arr).melt()
            df1['variable']=df2['value']
            sns.lineplot(x="variable", y="value", data=df1 ,color =my_color_list[idx] , linestyle ='-', ci=95)
            
            
        #plt.plot(steps,survived_steps_arr[1], 'gx:', lw=1)
        plt.xlabel('# learned episodes', fontsize=16)
        plt.ylabel('Loss', fontsize=16)
        plt.xticks(fontsize=16)

        #plt.xticks(np.arange(min(steps_arr[0]),max(steps_arr[0])*50 , 1000))

        #plt.xlim(0,steps.iget(-1))
        h = plt.gca().get_lines()
        plt.legend(handles=h, labels=label_arr, loc='upper right', fontsize=16)
        plt.tight_layout()
        plt.savefig(savepath)    
        sns.despine()


def compare_reward_per_step(data_arr, label_arr, savepath, reward='Average reward', LAGTPKS=True):
    if len(data_arr) != len(label_arr):
        print("ERROR! Data Array and label array must have equal length!")
    else:
        #sns.set()
        plt.figure(figsize=(8,6))
        if LAGTPKS==True:
            my_color_list=color_list_LAGTPKS
        else:
            my_color_list=color_list_learners
        
        for idx, data in enumerate(data_arr):
            survived_steps_arr=[]
            steps=data[idx]['episodes']
            survived_steps_arr=[]
            step_arr=[]
            for i in range(len(data)):
                survived_steps_arr.append(data[i][reward])
                step_arr.append(data[i]['episodes'])
            
            #sns.tsplot(time=steps, data=survived_steps_arr , color =my_color_list[idx] , linestyle ='-')
            df1=pd.DataFrame(survived_steps_arr).melt()
            df2=pd.DataFrame(step_arr).melt()
            df1['variable']=df2['value']
            sns.lineplot(x="variable", y="value", data=df1 ,color =my_color_list[idx] , linestyle ='-', lw=2, ci=95)
        #plt.plot(steps,survived_steps_arr[1], 'gx:', lw=1)
        plt.xlabel('# Learned Episodes', fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.ylabel(reward + ' per Episode', fontsize=16)
        #plt.xlim(0,steps.iget(-1))
        h = plt.gca().get_lines()
        plt.legend(handles=h, labels=label_arr, loc='lower right', fontsize=16)
        plt.tight_layout()
        plt.savefig(savepath)    
        sns.despine()
        

def plot_knickPoints_2D_full(learner_type, reward_type='PB', label=None, colors=['tab:green','black','#1f78b4'], 
                        basin='GREEN_FP', savepath='./images/phase_space_plots/Knick_points_heatmap.pdf'):
    import matplotlib as mpl
    a_min,a_max=0.55,0.595
    y_min,y_max=0.36,0.47
    s_min,s_max=0.54,0.62
    
    k=20
    lst_FP_arr1=pd.DataFrame(columns=parameters)
    for episode in range(0,10000, 500):
        learning_progress_arr=read_trajectories(learner_type=learner_type, reward_type=reward_type, basin=basin, policy='epsilon_greedy', 
                                                 episode=episode)
        for idx, simulation in enumerate(learning_progress_arr):
            learning_progress=simulation
            A,Y,S=get_averaged_AYS(learning_progress, k)
            action=learning_progress['Action']
        
            if basin=='GREEN_FP':
                knick_point_green=find_green_knick_point(A, Y, S)
            else:
                knick_point_brown=find_brown_knick_point(A, Y, S)
    
            tmp_data_1= pd.DataFrame(learning_progress.iloc[knick_point_green]).T
#             print(tmp_data_1.iloc[0]['A'])
            if tmp_data_1.iloc[0]['A']>=a_min and tmp_data_1.iloc[0]['Y']<y_max and tmp_data_1.iloc[0]['S']<s_max:
                lst_FP_arr1=pd.concat([lst_FP_arr1, tmp_data_1]).reset_index(drop = True)
    
        
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(14,4))
    A_PB, Y_PB, S_PB= [0.5897435897435898, 0.36363636363636365, 0]
    
    ax1=ax[0] 
    ax2=ax[1]
    ax3=ax[2]

    a_min,a_max=0.55,0.595
    y_min,y_max=0.36,0.47
    s_min,s_max=0.54,0.62

    # AY
    ax1.set_title('A-Y Plane')
    ax1.set_xlabel('A [GtC]')
    ax1.set_ylabel("Y [%1.0e USD/yr]"%Y_scale)
    ax1.set_xlim(a_min, a_max)
    ax1.set_ylim(y_min, y_max)
    hist1=ax1.hist2d(lst_FP_arr1['A'], lst_FP_arr1['Y'],bins=30, norm=mpl.colors.LogNorm(), vmin=1, vmax=200, cmap='viridis')

    make_2d_ticks(ax1, boundaries=[[a_min,a_max], [y_min, y_max]], scale_1=A_scale, scale_2=Y_scale, x_mid_1=current_state[0], x_mid_2=current_state[1])
    
    ax1.axvline(x=A_PB, color='red', linestyle='--')
    ax1.axhline(y=Y_PB, color='red', linestyle='--')
    
    # AS
    ax2.set_title('A-S Plane')
    ax2.set_xlabel('A [GtC]')
    ax2.set_ylabel("S [%1.0e GJ]"%S_scale )
    
    hist2=ax2.hist2d(lst_FP_arr1['A'], lst_FP_arr1['S'],bins=30, norm=mpl.colors.LogNorm(), vmin=1, vmax=200, cmap='viridis')
    
    ax2.set_xlim(a_min, a_max)
    ax2.set_ylim(s_min, s_max)
    make_2d_ticks(ax2, boundaries=[[a_min,a_max], [s_min, s_max]], scale_1=A_scale, scale_2=S_scale, x_mid_1=current_state[0], x_mid_2=current_state[2])

    ax2.axvline(x=A_PB, color='red', linestyle='--')

    
    # YS
    ax3.set_title('Y-S Plane')
    ax3.set_xlabel("S [%1.0e GJ]"%S_scale )
    ax3.set_ylabel("Y [%1.0e USD/yr]"%Y_scale)
    
    hist3=ax3.hist2d(lst_FP_arr1['S'], lst_FP_arr1['Y'],bins=30, norm=mpl.colors.LogNorm(), vmin=1, vmax=200, cmap='viridis')
    
    ax3.set_xlim(s_min, s_max)
    ax3.set_ylim(y_min, y_max)
    make_2d_ticks(ax3, boundaries=[[s_min, s_max],[y_min,y_max]], scale_1=S_scale, scale_2=Y_scale, x_mid_1=current_state[2], x_mid_2=current_state[1])
    
    
    ax3.axhline(y=Y_PB, color='red', linestyle='--')

    #ax3.legend(label, loc='center left', bbox_to_anchor=(1, .9), fontsize=14,fancybox=True, shadow=True )    
    
    plt.colorbar(hist2[3], ax=ax3, )

    
    fig.tight_layout()
    fig.savefig(savepath)
