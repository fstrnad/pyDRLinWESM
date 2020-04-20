"""
This script allows a detailed analysis of the pathways found by the learner 
in the AYS model.

@author: Felix Strnad 
"""
import sys,os
import numpy as np
import pandas as pd
from plots.AYS_3D_figures import *
import scipy.integrate as integ


shelter_color='#ffffb3'



def read_trajectories(learner_type, reward_type, basin, policy='epsilon_greedy', episode=0):
    runs=[]
    #10_path_[0.5, 0.5, 0.5]_episode9500
    limit=150
    parameters=['A' , 'Y' , 'S' , 'Action' , 'Reward' ]
    for i in range(limit):
        file_name=('./'+learner_type+'/' + policy +'/' +reward_type + '/DQN_Path/' +
                   basin+ '/' + str(i)+'_path_[0.5, 0.5, 0.5]_episode' + str(episode)+'.txt')

        if os.path.isfile(file_name):
            tmp_file= pd.read_csv(file_name, sep='\s+' ,header=None, names=parameters, skiprows=1, index_col=False)
            runs.append(tmp_file)
            
#             print(file_name)
        # For not too many files
        if len(runs) > 100:
            break
    print(learner_type +' '+ reward_type + ' ' + basin + ':' ,len(runs))
    
    return runs

def read_one_trajectory(learner_type, reward_type, basin, policy='epsilon_greedy', episode=0, run_idx=0):

    file_name=('./'+learner_type+'/' + policy +'/' +reward_type + '/DQN_Path/' +
               basin+ '/' + str(run_idx)+'_path_[0.5, 0.5, 0.5]_episode' + str(episode)+'.txt')

    if os.path.isfile(file_name):
        tmp_file= pd.read_csv(file_name, sep='\s+' ,header=None, names=parameters, skiprows=1, index_col=False)
        return tmp_file

    else:
        #print("No trajectories available for this simulation! run_idx: ", run_idx, " episode: ", episode)
        return None
    






def plot_current_state_trajectories(ax3d, label=False):
    # Trajectories for the current state with all possible management options
    time = np.linspace(0, 300, 1000)

    for action_number in range(len(management_actions)):
        if label==True:
            this_label=management_options[action_number]
        else:
            this_label=None
        parameter_list=get_parameters(action_number)         
        my_color=color_list[action_number]
        traj_one_step=odeint(ays.AYS_rescaled_rhs, current_state,time , args=parameter_list[0])
        ax3d.plot3D(xs=traj_one_step[:,0], ys=traj_one_step[:,1], zs=traj_one_step[:,2],
                        color=my_color, alpha=.8, lw=2, label=this_label) 

def plot_run(ax3d, learning_progress, reward_type, alpha=1., color_set=True, own_color=None):
        
        #print(learning_progress)
        timeStart = 0
        intSteps = 10    # integration Steps
        dt=1
        sim_time_step=np.linspace(timeStart,dt, intSteps)
        
                
        ax3d.plot3D(xs=learning_progress['A'], ys=learning_progress['Y'], zs=learning_progress['S'],
                            alpha=alpha, lw=1)


def plot_2D_AYS(self, learning_progress, file_path):
    #print(learning_progress)
    
    start_state=learning_progress[0][0]
    states=np.array(learning_progress)[:,0]
    a_states=list(zip(*states))[0]
    y_states=list(zip(*states))[1]
    s_states=list(zip(*states))[2]
    
    actions=np.array(learning_progress)[:,1]
    rewards=np.array(learning_progress)[:,2]

    
    fig=plt.figure(figsize=(14,8))
    ax=fig.add_subplot(111)

    plt.plot(actions +1, ".")
    plt.plot(a_states, "b", label='A')
    plt.plot([a*0 + self.A_PB for a in a_states], "b:")
    
    plt.plot(y_states, "k", label='Y')
    plt.plot([y*0 + self.Y_SF for y in y_states], "k:")

    plt.plot(s_states, "y", label='S')
    plt.ylim(-0.1,)
    
    
    at = AnchoredText((" 1.0: " + self.management_options[0] +"\n" +
                       " 2.0: " + self.management_options[1] +"\n" + 
                       " 3.0: " + self.management_options[2] +"\n" +
                       " 4.0: " + self.management_options[3] ), 
                   prop=dict(size=14), frameon=True, 
                  loc='center right'
                  )
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at)    
     
    fontP = FontProperties()
    fontP.set_size(12)
    plt.xlabel('# Timesteps')
    plt.ylabel('rescaled dynamic variables')
    plt.legend(loc=9, bbox_to_anchor=(0.5, -0.2), ncol=3, prop=fontP)       
    
    plt.tight_layout()
    
    final_state=self._which_final_state().name
    save_path = (file_path +'/DQN_Path/'+ final_state +'/'+ 
                 str (self.run_number) + '_' +  '2D_time_series'+  str(start_state) + '.pdf' )
        
    
    #plt.savefig(self.image_dir + '/sucess_path/2D_time_series'+str(start_state)+'.pdf')    
    
    plt.savefig(save_path)    

    plt.close()
    print('Saved as figure in path:' + save_path)


def get_averaged_AYS(learning_progress, k):
    A=learning_progress['A'].rolling(k ).mean()
    Y=learning_progress['Y'].rolling(k ).mean()
    S=learning_progress['S'].rolling(k ).mean()
    
    return A, Y, S

def correct_averaged_AYS(A,Y,S):
    A[np.isnan(A)]=0.5
    Y[np.isnan(Y)]=0.5
    S[np.isnan(S)]=0.5
    return A, Y, S

def find_green_knick_point(A, Y, S):
    gradient_arrayA=np.gradient(A, S)
    max_change1=np.argmax((gradient_arrayA))

    gradient_arrayY=np.gradient( Y, S,edge_order=2)

    max_change=np.where( gradient_arrayY==np.nanmin(gradient_arrayY))[0][0]  # gets the index of the element, which is not NaN
    #print("[", A[max_change],",", Y[max_change],",", S[max_change],"],")
    return max_change

def find_brown_knick_point(A,Y,S):
    
    gradient_arrayY=(np.gradient(Y)) # In Y we have a significant change in the slope that goes to 0
    max_change=np.where( gradient_arrayY==np.nanmin(gradient_arrayY))[0][0] +1  # gets the index of the element, which is not NaN, +1 to get the point where the knick has happend    
    #print("[", A[max_change],",", Y[max_change],",", S[max_change],"],")
    return max_change

def find_shelter_point(S):
    return next(idx for idx, value in enumerate(S) if value > 0.75) 

def find_backwater_point(S):
    return next(idx for idx, value in enumerate(S) if value < 0.1) 

def cut_shelter_traj(A, Y):
    idx_Y= next(idx for idx, value in enumerate(Y) if value > 0.63) 
    idx_A= next(idx for idx, value in enumerate(A) if value < 0.4)
    if idx_Y<idx_A: 
        return idx_Y
    else:
        return idx_A

def get_percentage_of_max_action(action, idx_1, idx_2):
    x = action[idx_1:idx_2]
    unique, counts = np.unique(x, return_counts=True)
    distribution=np.asarray((unique, counts)).T
    #print(unique, counts, distribution)
    
    tot_len_array=idx_2 - idx_1
    max_count=np.max(counts)
    
    idx_max_action=np.argwhere(counts==max_count)
    #print(idx_max_action, unique)
    max_action = int(unique[idx_max_action][0])
    percentage=max_count/tot_len_array
    return max_action, percentage

def plot_part_trajectory(ax3d, A, Y, S, action, start_idx, end_idx):
    max_action, percentage=get_percentage_of_max_action(action, start_idx, end_idx )
    #print(np.mean(action[start_idx:end_idx]), end_idx, max_action, percentage)
    ax3d.plot3D(xs=A[start_idx:end_idx], ys=Y[start_idx:end_idx], zs=S[start_idx:end_idx], color=color_list[max_action], alpha=percentage,lw=1)


def plot_action_trajectory (ax3d, learning_progress, start_idx, end_idx, lw=1):  
    timeStart = 0
    intSteps = 2    # integration Steps
    dt=1
    sim_time_step=np.linspace(timeStart,dt, intSteps)
    
    cut_array=learning_progress[start_idx:end_idx].reset_index(drop = True)
    #print(cut_array)
    for index, row in cut_array.iterrows():
        #print(row)
        state=(row['A'], row['Y'], row['S'])
        
        action=int(row['Action'])
        parameter_list=get_parameters(action)
        
        my_color=color_list[action]

        traj_one_step=odeint(ays.AYS_rescaled_rhs, state, sim_time_step , args=parameter_list[0])
        # Plot trajectory
        ax3d.plot3D(xs=traj_one_step[:,0], ys=traj_one_step[:,1], zs=traj_one_step[:,2],
                        color=my_color, alpha=1., lw=lw)
        

def plot_point_cloud_positive_developement(ax3d, learning_progress, cut_shelter=False,with_traj=True):
        
        timeStart = 0
        intSteps = 10    # integration Steps
        dt=1
        sim_time_step=np.linspace(timeStart,dt, intSteps)
        
        k=20
        A,Y,S=get_averaged_AYS(learning_progress, k)
        action=learning_progress['Action']
        knick_point=find_green_knick_point(A, Y, S)
        A,Y,S=correct_averaged_AYS( A,Y,S)

        if learning_progress['S'][knick_point]<0.65 and knick_point >5:
            #knick_point=np.min( np.argpartition(gradient_array, k)[0:k])

        
            ax3d.scatter(xs=A[knick_point+k], ys=Y[knick_point+k], zs=S[knick_point+k],
                        alpha=1,lw=1, color='tab:green' )

            if with_traj:
                # Part before knick
#                 plot_part_trajectory(ax3d, learning_progress['A'], learning_progress['Y'], learning_progress['S'], action, 0, knick_point)
                plot_part_trajectory(ax3d, A, Y, S, action, 0, knick_point+k)
                # Part between knick and shelter
                idx_shelter=find_shelter_point(learning_progress['S'])
#                 plot_part_trajectory(ax3d, learning_progress['A'], learning_progress['Y'], learning_progress['S'], action, knick_point, idx_shelter)
                plot_part_trajectory(ax3d, A, Y, S, action, knick_point+k, idx_shelter+1)
                # Part of the shelter, where every trajectory leads to green FP
                if cut_shelter:
                    idx_cut_shelter=cut_shelter_traj(learning_progress['A'],learning_progress['Y'])
#                     plot_part_trajectory(ax3d, learning_progress['A'], learning_progress['Y'], learning_progress['S'], action, idx_shelter, idx_cut_shelter)
                    plot_part_trajectory(ax3d, A,Y,S, action, idx_shelter, idx_cut_shelter)
#                     ax3d.plot3D(learning_progress['A'][idx_shelter:idx_cut_shelter], learning_progress['Y'][idx_shelter:idx_cut_shelter], 
#                                 learning_progress['S'][idx_shelter:idx_cut_shelter],  color=shelter_color, alpha=.9)
                else:
                    ax3d.plot3D(A[idx_shelter:], Y[idx_shelter:], 
                                S[idx_shelter:],  color=shelter_color, alpha=.9)
                    #plot_part_trajectory(ax3d, learning_progress['A'], learning_progress['Y'], learning_progress['S'], action, idx_shelter-1, len(learning_progress['A']))
                    #plot_action_trajectory(ax3d, learning_progress, idx_shelter, len(learning_progress['A']))
    
        else:
            print('here!', knick_point)
    
def plot_point_cloud_positive_developement_averaged(ax3d, learning_progress, cut_shelter=False, with_traj=True):
        
        k=20
        A,Y,S=get_averaged_AYS(learning_progress, k)

        action=learning_progress['Action']
    
        knick_point=find_green_knick_point(A, Y, S)
        
        if learning_progress['S'][knick_point]<0.65 and knick_point >5:
            #knick_point=np.min( np.argpartition(gradient_array, k)[0:k])

        
            ax3d.scatter(xs=A[knick_point+k], ys=Y[knick_point+k], zs=S[knick_point+k],
                        alpha=1,lw=1, color='lawngreen' )

            # Part before knick
            if with_traj:
                plot_part_trajectory(ax3d, A,Y,S, action, 0, knick_point+k)
                # Part between knick and shelter
                idx_shelter=find_shelter_point(S)
                plot_part_trajectory(ax3d, A,Y,S, action, knick_point+k, idx_shelter)
                # Part of the shelter, where every trajectory leads to green FP
                if cut_shelter:
                    idx_cut_shelter=cut_shelter_traj(A,Y)
                    ax3d.plot3D(A[idx_shelter:idx_cut_shelter], Y[idx_shelter:idx_cut_shelter], 
                                S[idx_shelter:idx_cut_shelter],  color=shelter_color, alpha=.5)
                else:
                    ax3d.plot3D(A[idx_shelter:], Y[idx_shelter:], 
                                S[idx_shelter:], color=shelter_color, alpha=.5)
        else:
            print('here!', knick_point)    

          
def plot_point_cloud_negative_developement(ax3d, learning_progress, with_traj=True):
        
        timeStart = 0
        intSteps = 10    # integration Steps
        dt=1
        sim_time_step=np.linspace(timeStart,dt, intSteps)
        
        k=25
        A,Y,S=get_averaged_AYS(learning_progress, k)
        action=learning_progress['Action']
        knick_point=find_brown_knick_point(A, Y, S)
        A,Y,S=correct_averaged_AYS( A,Y,S)
        
        #backwater_point=find_backwater_point(S)
        if learning_progress['S'][knick_point]>0.35 and knick_point >5:
            #knick_point=np.min( np.argpartition(gradient_array, k)[0:k])
            
            ax3d.scatter(xs=A[knick_point+k], ys=Y[knick_point+k], zs=S[knick_point+k],
                        alpha=1,lw=1, color='black' )

            if with_traj:
                # Part before knick
#                 plot_part_trajectory(ax3d, learning_progress['A'], learning_progress['Y'], learning_progress['S'], action, 0, knick_point+1)
                plot_part_trajectory(ax3d, A,Y,S, action, 0, knick_point+k)
                # Part up to end (requires constant management)
                #plot_part_trajectory(ax3d, A,Y,S,action, knick_point+k, backwater_point+1)
                #plot_part_trajectory(ax3d, A,Y,S,action, backwater_point, len(S))
                plot_part_trajectory(ax3d, A,Y,S,action, knick_point+k, len(S))
#             ax3d.plot3D(xs=learning_progress['A'], ys=learning_progress['Y'], zs=learning_progress['S'],
#                             alpha=alpha, lw=1)
    
        else:
            print('here!', knick_point)
        
    
def plot_3D_AYS_basins(learning_progress_green, learning_progress_brown, cut_shelter_image=False, num_traj=50,ax=None ):
    if ax is None:
        if cut_shelter_image:       
            fig, ax3d=create_extract_figure(Azimut=-160, plot_boundary=True)
        else:
            fig, ax3d=create_figure(Azimut=-160, )
    else:
        ax3d=ax
    for i in range(0,num_traj):
        if len(learning_progress_green) > i and len(learning_progress_brown) > i:
            if cut_shelter_image:       
        #         plot_point_cloud_positive_developement_averaged(ax3d, learning_progress=runs_survive_green[i],cut_shelter=cut_shelter )
                plot_point_cloud_positive_developement(ax3d, learning_progress=learning_progress_green[i],cut_shelter=cut_shelter_image )
            else:
                plot_point_cloud_positive_developement(ax3d, learning_progress=learning_progress_green[i], )
          
            plot_point_cloud_negative_developement(ax3d, learning_progress=learning_progress_brown[i], )
            
            
    if ax is None:
        if cut_shelter_image:
            fig.savefig('./images/phase_space_plots/zoom3D_AYS_trajectory_many_paths.pdf')
        else:
            #plot_hairy_lines(200, ax3d)
            fig.savefig('./images/phase_space_plots/3D_AYS_trajectory_many_paths.pdf')
        



def plot_knickPoints_2D(learning_progress_arr_1, learning_progress_arr_2, label=None, colors=['tab:green','black','#1f78b4'], 
                        basins=[True,False], savepath='./images/phase_space_plots/Knick_points_2D.pdf'):
    k=20
    lst_FP_arr1=pd.DataFrame(columns=parameters)
    lst_FP_arr2=pd.DataFrame(columns=parameters)
    for idx, simulation in enumerate(learning_progress_arr_1):
        learning_progress=simulation
        A,Y,S=get_averaged_AYS(learning_progress, k)
        action=learning_progress['Action']
    
        if basins[0]:
            knick_point_1=find_green_knick_point(A, Y, S)
        else:
            knick_point_1=find_brown_knick_point(A, Y, S)

        tmp_data_1= pd.DataFrame(learning_progress.iloc[knick_point_1]).T
        lst_FP_arr1=pd.concat([lst_FP_arr1, tmp_data_1]).reset_index(drop = True)
    
    for idx, simulation in enumerate(learning_progress_arr_2):
        learning_progress=simulation
        A,Y,S=get_averaged_AYS(learning_progress, k)
        action=learning_progress['Action']
    
        if basins[1]:
            knick_point_2=find_green_knick_point(A, Y, S)
        else:
            knick_point_2=find_brown_knick_point(A, Y, S)        
        
        tmp_data_2= pd.DataFrame(learning_progress.iloc[knick_point_2]).T
        lst_FP_arr2=pd.concat([lst_FP_arr2, tmp_data_2]).reset_index(drop = True)

        
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(14,4))
    A_PB, Y_PB, S_PB= [0.5897435897435898, 0.36363636363636365, 0]
    
    ax1=ax[0] 
    ax2=ax[1]
    ax3=ax[2]

    a_min,a_max=0.55,0.6
    y_min,y_max=0.36,0.47
    s_min,s_max=0.54,0.62

    # AY
    ax1.set_title('A-Y Plane')
    ax1.set_xlabel('A [GtC]')
    ax1.set_ylabel("Y [%1.0e USD/yr]"%Y_scale)
    make_2d_ticks(ax1, boundaries=[[a_min,a_max], [y_min, y_max]], scale_1=A_scale, scale_2=Y_scale, x_mid_1=current_state[0], x_mid_2=current_state[1])

    ax1.plot(lst_FP_arr1['A'], lst_FP_arr1['Y'], 'd', color=colors[0], fillstyle='none')
    ax1.plot(lst_FP_arr2['A'], lst_FP_arr2['Y'], 'x', color=colors[1])
    ax1.axvline(x=A_PB, color='red', linestyle='--')
    ax1.axhline(y=Y_PB, color='red', linestyle='--')

    
    # AS
    ax2.set_title('A-S Plane')
    ax2.set_xlabel('A [GtC]')
    ax2.set_ylabel("S [%1.0e GJ]"%S_scale )
    make_2d_ticks(ax2, boundaries=[[a_min,a_max], [s_min, s_max]], scale_1=A_scale, scale_2=S_scale, x_mid_1=current_state[0], x_mid_2=current_state[2])

    ax2.plot(lst_FP_arr1['A'], lst_FP_arr1['S'], 'd', color=colors[0],fillstyle='none')
    ax2.plot(lst_FP_arr2['A'], lst_FP_arr2['S'], 'x', color=colors[1])
    ax2.axvline(x=A_PB, color='red', linestyle='--')

    
    # YS
    ax3.set_title('Y-S Plane')
    ax3.set_xlabel("Y [%1.0e USD/yr]"%Y_scale)
    ax3.set_ylabel("S [%1.0e GJ]"%S_scale )
    make_2d_ticks(ax3, boundaries=[[y_min,y_max], [s_min, s_max]], scale_1=Y_scale, scale_2=S_scale, x_mid_1=current_state[1], x_mid_2=current_state[2])
    
    ax3.plot(lst_FP_arr1['Y'], lst_FP_arr1['S'], 'd', color=colors[0],fillstyle='none')
    ax3.plot(lst_FP_arr2['Y'], lst_FP_arr2['S'], 'x', color=colors[1])
    
    ax3.axvline(x=Y_PB, color='red', linestyle='--', label='Boundaries')

    ax3.legend(label, loc='center left', bbox_to_anchor=(1, .9), fontsize=14,fancybox=True, shadow=True )    
    fig.tight_layout()
    fig.savefig(savepath)

def plot_action_3D_basins(learning_progress_arr, cut_shelter_image=False, num_plots=50,):
    
    if cut_shelter_image:       
        fig, ax3d=create_extract_figure(Azimut=-160, plot_boundary=True, label=None, colors=None)
    else:
        fig, ax3d=create_figure(Azimut=-160,label=None, colors=None )
    
    for idx, learning_progress in enumerate(learning_progress_arr):
        for i in range(0,num_plots):
            if len(learning_progress) > i:
                k=15
                A,Y,S=learning_progress[i]['A'],learning_progress[i]['Y'],learning_progress[i]['S'] 
                
                if cut_shelter_image and learning_progress[i]['S'].iloc[-1]>0.9:
                    idx_cut_shelter=cut_shelter_traj(A,Y)
                    plot_action_trajectory(ax3d, learning_progress[i], 0, idx_cut_shelter)
                else:
                    plot_action_trajectory(ax3d, learning_progress[i], 0, len(S))
                
                if learning_progress[i]['S'].iloc[-1]>0.9:
                    plot_point_cloud_positive_developement(ax3d, learning_progress[i], cut_shelter_image, with_traj=False)
                else:
                    plot_point_cloud_negative_developement(ax3d, learning_progress[i], with_traj=False)
                
                
    if cut_shelter_image:
        fig.savefig('./images/phase_space_plots/zoom3D_AYS_trajectory_actions_many_paths.pdf')
    else:
        #plot_hairy_lines(200, ax3d)
        fig.savefig('./images/phase_space_plots/3D_AYS_trajectory_actions_many_paths.pdf')
                    
                    
def plot_averaged_3D_basins(learning_progress_arr, cut_shelter_image=False, num_plots=50,label_arr=None, color_arr=None, ax=None  ):
    if color_arr is None:
        color_arr=['black', 'green']
    if ax is None:
        if cut_shelter_image:       
            fig, ax3d=create_extract_figure(Azimut=-160, plot_boundary=True, label=label_arr, colors=color_arr)
        else:
            fig, ax3d=create_figure(Azimut=-160,label=label_arr, colors=color_arr )
    else:
        ax3d=ax
    
    for idx, learning_progress in enumerate(learning_progress_arr):
        for i in range(0,num_plots):
            if len(learning_progress) > i:
                k=15
                A,Y,S=correct_averaged_AYS( *get_averaged_AYS(learning_progress[i], k))
                
                if cut_shelter_image and learning_progress[i]['S'].iloc[-1]>0.9:
                    idx_cut_shelter=cut_shelter_traj(A,Y)
                    ax3d.plot3D(xs=A[:idx_cut_shelter], ys=Y[:idx_cut_shelter], zs=S[:idx_cut_shelter],
                                color=color_arr[idx] , alpha=1., lw=1)
                else:
                    ax3d.plot3D(xs=A, ys=Y, zs=S, color=color_arr[idx], alpha=1., lw=1)
    if cut_shelter_image:
        fig.savefig('./images/phase_space_plots/zoom3D_AYS_averaged_trajectory_many_paths.pdf')
    else:
        #plot_hairy_lines(200, ax3d)
        fig.savefig('./images/phase_space_plots/3D_AYS_averaged_trajectory_many_paths.pdf')
    
    return ax3d

def find_color(option):
    if option in management_options:
        idx=management_options.index(option)
        return color_list[idx]
    else:
        print("ERROR! This management option does not exist: ", option)
        sys.exit(1)
    
def plot_management_dynamics(ax, option=None):
    if option is not None:
        time = np.linspace(0, 300, 1000)
        x0 = [0.5,0.5,0.5]
        
        if option in management_options:
            parameter_list=get_parameters(management_options.index(option))
            print(option)
            color=find_color(option)
            traj = integ.odeint(ays.AYS_rescaled_rhs, x0, time, args=parameter_list[0])
        
            ax.plot3D(xs=traj[:,0], ys=traj[:,1], zs=traj[:,2],
                                color=color)

       

def plot_example_figure(learning_progress_arr, cut_shelter=False, num_traj=0,num_hairs=300, ax=None, ticks=True,label=[], colors=[],
                        option=None,plot_traj=True, plot_boundary=True,
                        filename='./images/phase_space_plots/3D_AYS_example_trajectory.pdf', ):
    
    learning_progress= learning_progress_arr[num_traj]
    if ax is None:
        if cut_shelter:       
            fig, ax3d=create_extract_figure(Azimut=-160, plot_boundary=plot_boundary, label=None, colors=None)
        else:
            if option=='DG':
                Azimut=-177
                Elevation=65
            elif option=='ET':
                Azimut=-88
                Elevation=65
            else:
                Azimut=-167
                Elevation=25
            fig, ax3d=create_figure(Azimut=Azimut,Elevation=Elevation, label=label, colors=colors, ticks=ticks, plot_boundary=plot_boundary )
    else:
        ax3d=ax
    
    if option:
        plot_management_dynamics(ax3d, option)
#                 plot_management_dynamics(ax3d, 'ET')
    else: 
        plot_hairy_lines(num_hairs, ax3d)
  
    
    A,Y,S=learning_progress['A'],learning_progress['Y'],learning_progress['S'] 
    
    if cut_shelter and learning_progress['S'].iloc[-1]>0.9:
        idx_cut_shelter=cut_shelter_traj(A,Y)
        plot_action_trajectory(ax3d, learning_progress, 0, idx_cut_shelter, lw=4)
    else:
        if plot_traj:
            plot_action_trajectory(ax3d, learning_progress, 0, len(S),lw=4)
    
    fig.savefig(filename)
    
    return ax3d
                
def plot_learning_developement(learner_type='ddqn_per_is_duel', reward_type='survive', policy='epsilon_greedy', run_idx=0):
    
    #episodes=[0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000,4500, 5000, 6000, 7000, 8000, 9000]
    #episodes=[0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000,4500, 5000, 6000, 7000, 8000, 9000]
    episodes=[0, 500, 1000, 1500, 2000, 2500, 3000, 5000, 7000, 8000, 9000]

    nrows=3
    ncols=3
    fig=  plt.figure( figsize=(12,12))
    for i in range(0,nrows):
        for j in range(0,ncols):
            idx=i*ncols + j
            episode=episodes[idx]
            this_ax=fig.add_subplot(nrows, ncols, idx+1, projection='3d')
            this_ax.set_title("Episode: " + str(episode))
            this_ax.dist = 13
            create_axis_3d(this_ax)
            for basin in ['BROWN_FP', 'GREEN_FP', 'OUT_PB']:
            
                learning_progress=read_one_trajectory(learner_type, reward_type, basin, policy, episode, run_idx)
                #print(filepath)
                if learning_progress is not None:
                    
        
                    print("Path for " + basin+ " at episode: ", episode )
                    
            
                    this_ax.plot3D(xs=learning_progress['A'], ys=learning_progress['Y'], zs=learning_progress['S'], 
                                   color='green' if learning_progress['S'].iloc[-1]>0.9 else 'black', 
                            alpha=1., lw=1)
                    break

    fig.tight_layout()
    fig.savefig('./images/learning_success/learning_developement.pdf')

def management_distribution_part(learning_progress):
    k=20
    A,Y,S=get_averaged_AYS(learning_progress, k)

    action=learning_progress['Action']

    knick_point=find_green_knick_point(A, Y, S)

    A,Y,S=learning_progress['A'],learning_progress['Y'],learning_progress['S']
    learning_progress['Action'] 
    idx_shelter=find_shelter_point(S)
    
    if learning_progress['S'][knick_point]<0.65 and knick_point >5:
        #knick_point=np.min( np.argpartition(gradient_array, k)[0:k])
        my_actions= learning_progress['Action'][knick_point:idx_shelter]
        weights = np.ones_like(my_actions)/float(len(my_actions))
        my_actions.hist(bins=3, density=True, weights=weights) 

        plt.xlabel("Action number", fontsize=15)
        plt.ylabel("Probability",fontsize=15)
        plt.xlim([0,3])
        
    else:
        print('No knick point found!', knick_point)      
        

    

def heat_map_knick_point(learner_type, reward_type='PB', label=None, colors=['tab:green','black','#1f78b4'], 
                        basin='GREEN_FP', savepath='./images/phase_space_plots/Knick_points_heatmap.pdf'):
    import matplotlib as mpl
    import scipy.stats as st

    a_min,a_max=0.56,0.595
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
                tmp_data_1= pd.DataFrame(learning_progress.iloc[knick_point_green]).T
            else:
                knick_point_brown=find_brown_knick_point(A, Y, S)
                tmp_data_1= pd.DataFrame(learning_progress.iloc[knick_point_brown]).T
#             print(tmp_data_1.iloc[0]['A'])
            if tmp_data_1.iloc[0]['A']>=a_min and tmp_data_1.iloc[0]['Y']<y_max:
                lst_FP_arr1=pd.concat([lst_FP_arr1, tmp_data_1]).reset_index(drop = True)
    

        
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,4))
    A_PB, Y_PB, S_PB= [0.5897435897435898, 0.36363636363636365, 0]
    
    ax1=ax 

    # AY
    ax1.set_title('A-Y Plane')
    ax1.set_xlabel('A [GtC]')
    ax1.set_ylabel("Y [%1.0e USD/yr]"%Y_scale)
        
    ax1.set_xlim(a_min, a_max)
    ax1.set_ylim(y_min, y_max)

    #hist=ax1.hist2d(lst_FP_arr1['A'], lst_FP_arr1['Y'],bins=30, norm=mpl.colors.LogNorm(), vmin=1, vmax=200, cmap='viridis')
    nbins=100
    kde = st.kde.gaussian_kde([lst_FP_arr1['A']-0.00, lst_FP_arr1['Y']+0.015])
    xi, yi = np.mgrid[a_min:a_max:nbins*1j, y_min:y_max:nbins*1j]
    zi = kde(np.vstack([xi.flatten(), yi.flatten()]))
    zi/=zi.max()
    for idx, cell in enumerate(zi):
        if cell> 0.01 and cell<0.9 :
            zi[idx] +=0.08
            print('Here:', cell) 
    print(zi.shape)
     
    # Make the plot
    plt.pcolormesh(xi, yi, zi.reshape(xi.shape),cmap=plt.cm.Reds, vmin=0.00)

    make_2d_ticks(ax1, boundaries=[[a_min,a_max], [y_min, y_max]], scale_1=A_scale, scale_2=Y_scale, x_mid_1=current_state[0], x_mid_2=current_state[1])
    
    ax1.axvline(x=A_PB, color='red', linestyle='--')
    ax1.axhline(y=Y_PB, color='red', linestyle='--')
    #plt.colorbar(hist[3], ax=ax1, )
    plt.colorbar()
    # AY
#     ax2.set_title('A-Y Plane')
#     ax2.set_xlabel('A [GtC]')
#     ax2.set_ylabel("Y [%1.0e USD/yr]"%Y_scale)
#     make_2d_ticks(ax1, boundaries=[[a_min,a_max], [y_min, y_max]], scale_1=A_scale, scale_2=Y_scale, x_mid_1=current_state[0], x_mid_2=current_state[1])
# 
#     ax2.plot(lst_FP_arr1['A'], lst_FP_arr1['Y'], 'd', color=colors[0], fillstyle='none')
#     ax2.axvline(x=A_PB, color='red', linestyle='--')
#     ax2.axhline(y=Y_PB, color='red', linestyle='--')

    fig.tight_layout()
    fig.savefig(savepath)

def plot_knickPoints_2D_full(learner_type, reward_type='PB', label=None, colors=['tab:green','black','#1f78b4'], 
                        basin='GREEN_FP', savepath='./images/phase_space_plots/Knick_points_heatmap.pdf'):
    import matplotlib as mpl
    import scipy.stats as st
    a_min,a_max=0.56,0.595
    y_min,y_max=0.36,0.47
    s_min,s_max=0.32,0.63
#     s_min,s_max=0.54,0.62

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
                tmp_data_1= pd.DataFrame(learning_progress.iloc[knick_point_green]).T
            else:
                knick_point_brown=find_brown_knick_point(A, Y, S)
                tmp_data_1= pd.DataFrame(learning_progress.iloc[knick_point_brown]).T
    
#             print(tmp_data_1.iloc[0]['A'])
            if tmp_data_1.iloc[0]['A']>=a_min and tmp_data_1.iloc[0]['Y']<y_max and tmp_data_1.iloc[0]['S']<s_max:
                lst_FP_arr1=pd.concat([lst_FP_arr1, tmp_data_1]).reset_index(drop = True)
    
        
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(14,4))
    A_PB, Y_PB, S_PB= [0.5897435897435898, 0.36363636363636365, 0]
    
    ax1=ax[0] 
    ax2=ax[1]
    ax3=ax[2]

    nbins=100

    # AY
    ax1.set_title('A-Y Plane')
    ax1.set_xlabel('A [GtC]')
    ax1.set_ylabel("Y [%1.0e USD/yr]"%Y_scale)
    ax1.set_xlim(a_min, a_max)
    ax1.set_ylim(y_min, y_max)
    
    k = st.kde.gaussian_kde([lst_FP_arr1['A'], lst_FP_arr1['Y']+0.015])
    xi, yi = np.mgrid[a_min:a_max:nbins*1j, y_min:y_max:nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    zi/=zi.max() 
    for idx, cell in enumerate(zi):
        if cell> 0.01 and cell<0.9 :
            zi[idx] +=0.08
     
    # Make the plot
    ax1.pcolormesh(xi, yi, zi.reshape(xi.shape),cmap=plt.cm.Reds, vmin=0.)
    make_2d_ticks(ax1,boundaries=[[a_min,a_max], [y_min, y_max]], 
                  scale_1=A_scale, scale_2=Y_scale, x_mid_1=current_state[0], x_mid_2=current_state[1])
    
    ax1.axvline(x=A_PB, color='red', linestyle='--')
    ax1.axhline(y=Y_PB, color='red', linestyle='--')
    
    # AS
    ax2.set_title('A-S Plane')
    ax2.set_xlabel('A [GtC]')
    ax2.set_ylabel("S [%1.0e GJ]"%S_scale )
    ax2.set_xlim(a_min, a_max)
    ax2.set_ylim(s_min, s_max)
        
    k = st.kde.gaussian_kde([lst_FP_arr1['A'], lst_FP_arr1['S'] ])
    xi, yi = np.mgrid[a_min:a_max:nbins*1j, s_min:s_max:nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    zi/=zi.max()     
    for idx, cell in enumerate(zi):
        if cell> 0.01 and cell<0.9 :
            zi[idx] +=0.08
    
    ax2.pcolormesh(xi, yi, zi.reshape(xi.shape),cmap=plt.cm.Reds, vmin=0.)

    make_2d_ticks(ax=ax2, boundaries=[[a_min,a_max],[s_min, s_max]], 
                  scale_1=A_scale, scale_2=S_scale, x_mid_1=240, x_mid_2=5e11 )
    ax2.axvline(x=A_PB, color='red', linestyle='--')

    
    # SY
    ax3.set_title('S-Y Plane')
    ax3.set_xlabel("S [%1.0e GJ]"%S_scale )
    ax3.set_ylabel("Y [%1.0e USD/yr]"%Y_scale)
    ax3.set_xlim(s_min, s_max)
    ax3.set_ylim(y_min, y_max)
        
    k = st.kde.gaussian_kde([lst_FP_arr1['S'], lst_FP_arr1['Y']+0.015])
    xi, yi = np.mgrid[s_min:s_max:nbins*1j, y_min:y_max:nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    zi/= zi.max() 
    for idx, cell in enumerate(zi):
        if cell> 0.01 and cell<0.9 :
            zi[idx] +=0.08
    
    
    pcm=ax3.pcolormesh(xi, yi, zi.reshape(xi.shape),cmap=plt.cm.Reds, vmin=0.00)

    make_2d_ticks(ax=ax3, boundaries=[[s_min,s_max], [y_min, y_max]], scale_1=S_scale, scale_2=Y_scale, 
                  x_mid_1=current_state[2], x_mid_2=current_state[1])
    
    ax3.axhline(y=Y_PB, color='red', linestyle='--')

    #ax3.legend(label, loc='center left', bbox_to_anchor=(1, .9), fontsize=14,fancybox=True, shadow=True )    
    
    plt.colorbar(pcm,ax=ax3)

    
    fig.tight_layout()
    fig.savefig(savepath)




# 
# heat_map_knick_point(learner_type='dqn', reward_type='PB',
#                       savepath='./images/phase_space_plots/Knick_points_heatmap_DQN_PB.pdf')
# heat_map_knick_point(learner_type='ddqn_per_is_duel', reward_type='PB',
#                      savepath='./images/phase_space_plots/Knick_points_heatmap_DDDQN_PB.pdf')
# heat_map_knick_point(learner_type='ddqn_per_is_duel', reward_type='survive',
#                      savepath='./images/phase_space_plots/Knick_points_heatmap_DDDQN_survive.pdf')

# heat_map_knick_point(learner_type='ddqn_per_is', reward_type='PB',
#                      savepath='./images/phase_space_plots/Knick_points_heatmap_DDQN_per_is.pdf')
# heat_map_knick_point(learner_type='ddqn_per_is_duel', reward_type='PB', 
#                      savepath='./images/phase_space_plots/Knick_points_heatmap_DDDQN_PB.pdf')
# heat_map_knick_point(learner_type='ddqn_per_is_duel', reward_type='survive', 
#                      savepath='./images/phase_space_plots/Knick_points_heatmap_DDDQN_survive.pdf')



# plot_knickPoints_2D_full(learner_type='ddqn_per_is_duel', reward_type='PB', 
#                      savepath='./images/phase_space_plots/Knick_points_DDDQN_PB.pdf')
# plot_knickPoints_2D_full(learner_type='dqn', reward_type='PB', 
#                      savepath='./images/phase_space_plots/Knick_points_DQN_PB.pdf')
# plot_knickPoints_2D_full(learner_type='ddqn_per_is_duel', reward_type='survive', 
#                      savepath='./images/phase_space_plots/Knick_points_DDDQN_survive.pdf')
# plot_knickPoints_2D_full(learner_type='ddqn_per_is_duel', reward_type='survive', basin='BROWN_FP',
#                      savepath='./images/phase_space_plots/Knick_points_DDDQN_survive_brown.pdf')
# plot_knickPoints_2D_full(learner_type='dqn', reward_type='survive', basin='BROWN_FP',
#                      savepath='./images/phase_space_plots/Knick_points_DQN_survive_brown.pdf')




cut_shelter=True
num_traj=100


ddqn_per_is_duel_survive_green_6000=read_trajectories(learner_type='ddqn_per_is_duel', reward_type='survive', basin='GREEN_FP', policy='epsilon_greedy', episode=6000)
ddqn_per_is_duel_PB_green_6000=read_trajectories(learner_type='ddqn_per_is_duel', reward_type='PB', basin='GREEN_FP', policy='epsilon_greedy', 
                                                 episode=6000)
ddqn_per_is_duel_PB_green_2000=read_trajectories(learner_type='ddqn_per_is_duel', reward_type='PB', basin='GREEN_FP', 
                                                 policy='epsilon_greedy', episode=1000)
 
# ddqn_per_is_duel_survive_brown_6000=read_trajectories(learner_type='ddqn_per_is_duel', 
#                                                       reward_type='survive', basin='BROWN_FP', policy='epsilon_greedy', episode=5000)
 
#dqn_survive_green_6000=read_trajectories(learner_type='dqn', reward_type='survive', basin='GREEN_FP', policy='epsilon_greedy', episode=6000)
#dqn_PB_green_6000=read_trajectories(learner_type='dqn', reward_type='PB', basin='GREEN_FP', policy='epsilon_greedy', episode=6000)
#dqn_PB_brown_6000=read_trajectories(learner_type='dqn', reward_type='PB', basin='BROWN_FP', policy='epsilon_greedy', episode=6000)

ddqn_per_is_duel_survive_green_6000=read_trajectories(learner_type='ddqn_per_is_duel', reward_type='survive', basin='GREEN_FP', policy='epsilon_greedy', episode=6000)



# plot_knickPoints_2D(ddqn_per_is_duel_PB_green_6000, ddqn_per_is_duel_survive_green_6000, label=['Boundary-Distance', 'Survive'],
#                       colors=['tab:blue','tab:orange', ], basins=[True,True], savepath='./images/phase_space_plots/Knick_points_2D_rewards.pdf')
# plot_knickPoints_2D(ddqn_per_is_duel_survive_green_6000, ddqn_per_is_duel_survive_brown_6000, label=['Green FP', 'Black FP'],
#                     savepath='./images/phase_space_plots/Knick_points_2D_fixpoint.pdf')
# plot_knickPoints_2D(ddqn_per_is_duel_PB_green_6000, dqn_PB_green_6000, label=['DDQN PER IS Duel', 'DQN'],basins=[True,True], 
#                     colors=['tab:green', 'blue'], savepath='./images/phase_space_plots/Knick_points_2D_learner.pdf' )
# plot_knickPoints_2D(ddqn_per_is_duel_PB_green_6000, ddqn_per_is_duel_PB_green_2000, label=['6000', '2000'],basins=[True,True], 
#                     colors=['tab:green', 'blue'], savepath='./images/phase_space_plots/Knick_points_2D_episodes.pdf' )
#plot_3D_AYS_basins(ddqn_per_is_duel_PB_green_6000, ddqn_per_is_duel_survive_brown_6000, cut_shelter, num_traj)
#plot_3D_AYS_basins(dqn_PB_green_6000, ddqn_per_is_duel_survive_brown_6000, cut_shelter, num_traj)
#plot_averaged_3D_basins([ddqn_per_is_duel_survive_brown_6000,ddqn_per_is_duel_survive_green_6000], cut_shelter,40, ['Brown FP', 'Green FP'], ['black', 'green'])
#plot_averaged_3D_basins([ddqn_per_is_duel_PB_green_6000, ddqn_per_is_duel_survive_green_6000], cut_shelter,40, ['PB', 'Survive'], ['tab:green','tab:red', ])
#plot_averaged_3D_basins([ddqn_per_is_duel_PB_green_6000, dqn_PB_green_6000], cut_shelter,40, ['DDQN PER IS Duel', 'DQN'], ['tab:green','tab:red', ])
#plot_action_3D_basins([ddqn_per_is_duel_PB_green_6000, ddqn_per_is_duel_survive_brown_6000], cut_shelter_image=cut_shelter, num_plots=num_traj)
#plot_one_learning_developement(learner_type='ddqn_per_is_duel', reward_type='PB', label='DDQN PER IS', run_index=7)
#plot_learning_developement(learner_type='ddqn_per_is_duel', reward_type='PB', policy='epsilon_greedy', run_idx=7)
# for i in range(5):
#     plot_example_figure(ddqn_per_is_duel_PB_green_6000, False, i, 300)

# plot_example_figure(learning_progress_arr=ddqn_per_is_duel_PB_green_6000, cut_shelter=False, num_traj=3, num_hairs=300,
#                     ticks=True, filename='./images/exemplary_figures/AYS_phase_space.pdf')


ddqn_per_is_duel_out_1000=read_trajectories(learner_type='ddqn_per_is_duel', 
                                            reward_type='survive', basin='OUT_PB', policy='epsilon_greedy', episode=1000)
ddqn_per_is_duel_out_2000=read_trajectories(learner_type='ddqn_per_is_duel', 
                                            reward_type='survive', basin='OUT_PB', policy='epsilon_greedy', episode=2000)

# plot_example_figure(learning_progress_arr=ddqn_per_is_duel_out_1000, cut_shelter=False, num_traj=3, num_hairs=600, option='ET',
#                     ticks=False, filename='./images/exemplary_figures/AYS_phase_space_ET.pdf')


# for i in range(10):
#     plot_example_figure(learning_progress_arr=ddqn_per_is_duel_PB_green_6000, cut_shelter=False, num_traj=i, num_hairs=600, 
#                     option=None, plot_traj=True, ticks=True, label=None, colors=[],plot_boundary=True,
#                     filename='./images/exemplary_figures/AYS_phase_space_traj.pdf')

plot_example_figure(learning_progress_arr=ddqn_per_is_duel_PB_green_6000, cut_shelter=False, num_traj=5, num_hairs=600, 
                    option=None, plot_traj=False, ticks=True, label=[], colors=[],plot_boundary=True,
                    filename='./images/exemplary_figures/AYS_phase_space_PB.pdf')

#management_distribution_part(ddqn_per_is_duel_PB_green_6000[3])

plt.show()


    






