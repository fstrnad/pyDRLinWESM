"""
This is the implementation of the AYS Environment in the form
that it can used within the Agent-Environment interface 
in combination with the DRL-agent.

@author: Felix Strnad
"""

import sys

import numpy as np
from scipy.integrate import odeint
import AYS.ays_model as ays
import AYS.ays_general as ays_general


from DeepReinforcementLearning.Basins import Basins
from gym import Env

import mpl_toolkits.mplot3d as plt3d
import matplotlib.pyplot as plt

from matplotlib.font_manager import FontProperties
from matplotlib.offsetbox import AnchoredText
from plots.AYS_3D_figures import create_figure
import plots.AYS_3D_figures as ays_plot
import os
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


@np.vectorize
def inv_compactification(y, x_mid):
    if y == 0:
        return 0.
    if np.allclose(y, 1):
        return np.infty
    return x_mid * y / (1 - y)


    

from inspect import currentframe, getframeinfo

def get_linenumber():
    print_debug_info()
    print("Line: ")
    cf = currentframe()
    return cf.f_back.f_lineno

def print_debug_info():
    frameinfo = getframeinfo(currentframe())
    print ("File: ", frameinfo.filename)
    


class AYS_Environment(Env):
    """
    The environment is based on Kittel et al. 2017, and contains in part code adapted from 
    https://github.com/timkittel/ays-model/ . 
    This Environment describes the 3D implementation of a simple model for the development of climate change, wealth
    and energy transformation which is inspired by the model from Kellie-Smith and Cox.
    Dynamic variables are :
        - excess atmospheric carbon stock A
        - the economic output/production Y  (similar to wealth)
        - the renewable energy knowledge stock S
    
    Parameters
    ----------
         - sim_time: Timestep that will be integrated in this simulation step
          In each grid point the agent can choose between subsidy None, A, B or A and B in combination. 
    """
    dimensions=np.array( ['A','Y','S'] )
    management_options=['default', 'LG' , 'ET','LG+ET'  ]
    action_space=[(False, False), (True, False), (False, True), (True, True)]
    action_space_number=np.arange(len(action_space))
    # AYS example from Kittel et al. 2017:
    tau_A = 50
    tau_S = 50
    beta = 0.03
    beta_LG = 0.015
    eps = 147
    A_offset = 600
    theta = beta /(950-A_offset)    # beta / ( 950 - A_offset(=350) )
    #theta = 8.57e-5

    rho = 2.
    sigma = 4e12
    sigma_ET = sigma*0.5**(1/rho)
    #sigma_ET = 2.83e12

    phi = 4.7e10
    
    AYS0 = [240, 7e13, 5e11]
    
    possible_test_cases=[[0.4949063922255394, 0.4859623171738628, 0.5] , [0.42610779 ,0.52056811, 0.5]]
    
    
    def __init__(self, t0=0, dt=1 , reward_type='PB', image_dir='./images/', run_number=0, plot_progress=False):
        
        self.image_dir=image_dir
        self.run_number = run_number
        self.plot_progress=plot_progress
         
        # The grid defines the number of cells, hence we have 8x8 possible states
        self.final_state=False
        self.reward=0
        
        self.reward_type=reward_type
        self.reward_function=self.get_reward_function(reward_type)
        
        timeStart = 0
        intSteps = 10    # integration Steps
        self.t=self.t0=t0
        self.dt=dt
        
        self.sim_time_step=np.linspace(timeStart,dt, intSteps)
        
        self.green_fp=[0,1,1]
        self.brown_fp=[0.6,0.4,0]
        self.final_radius=0.05 # Attention depending on how large the radius is, the BROWN_FP can be reached!
        self.color_list=ays_plot.color_list
        
        self.X_MID= [240, 7e13, 5e11]
        
        # Definitions from outside
        self.current_state=[0.5, 0.5, 0.5]
        self.state=self.start_state=self.current_state
        self.observation_space=self.state

        """
        This values define the planetary boundaries of the AYS model
        """
        self.A_PB = self._compactification( ays.boundary_parameters["A_PB"] , self.X_MID[0])    # Planetary boundary
        self.Y_SF = self._compactification( ays.boundary_parameters["W_SF"] , self.X_MID[1])    # Social foundations as boundary
        self.S_LIMIT=0
        self.PB=[self.A_PB, self.Y_SF,0]
        
        print("Init AYS Environment!",
              "\nReward Type: " + reward_type,
              "\nSustainability Boundaries [A_PB, Y_SF, S_ren]: ", inv_compactification(self.PB, self.X_MID) )
    
      
    def step(self, action):
        """
        This function performs one simulation step in a RFL algorithm. 
        It updates the state and returns a reward according to the chosen reward-function.
        """

        next_t= self.t + self.dt
        self.state=self._perform_step(action, next_t)
        self.t=next_t
        if self._arrived_at_final_state():
            self.final_state = True
        
        reward=self.reward_function(action)   # TODO check if this might be needed before step is done to evaluate the current state, not the next state!
        if not self._inside_planetary_boundaries():
            self.final_state = True
            #print("Left planetary boundaries!" + str(self.state))
            reward=0
        return self.state, reward, self.final_state
    
    def _perform_step(self, action, next_t):
        parameter_list=self._get_parameters(action)
        
        traj_one_step=odeint(ays.AYS_rescaled_rhs, self.state, [self.t, next_t] , args=parameter_list[0], mxstep=50000)
        
        
        a = traj_one_step[:,0][-1]
        y = traj_one_step[:,1][-1]
        s = traj_one_step[:,2][-1]
        return np.array( (a,y,s) )
    
    def reset(self):
        #self.state=np.array(self.random_StartPoint())
        self.state=np.array(self.current_state_region_StartPoint())
        #self.state=np.array(self.current_state)
        self.final_state=False
        self.t=self.t0
        return self.state    
    
    def reset_for_state(self, state=None):
        if state==None:
            self.start_state=self.state=np.array(self.current_state)
        else:
            self.start_state=self.state=np.array(state)
        self.final_state=False
        self.t=self.t0
        return self.state
    
    def get_reward_function(self,choice_of_reward):
        """
        This function returns one function as a function pointer according to the reward type we chose 
        for this simulation.
        """
        def reward_final_state(action=0):
            """
            Reward in the final  green fixpoint_good 1. , else 0.
            """
            if self._good_final_state():
                reward=1.
            else:
                reward=-0.00000000000001
            return reward
        
        def reward_rel_share(action=0):
            """
            We want to:
            - maximize the knowledge stock of renewables S 
            - minimize the excess atmospheric carbon stock A 
            - maximize the economic output Y!
            """
            a,y,s = self.state
            if self._inside_planetary_boundaries():
                reward=1.
            else:
                reward=0.
            reward*=s 
            
            return reward       
        def reward_desirable_region(action=0):
            a,y,s = self.state
            desirable_share_renewable=0.3
            reward=0.
            if s >= desirable_share_renewable:
                reward=1.
            return reward
        
        def reward_survive(action=0):
            if self._inside_planetary_boundaries():
                reward=1.
            else:
                reward=-0.0000000000000001
            return reward
        
        def reward_survive_cost(action=0):
            cost_managment=0.03
            if self._inside_planetary_boundaries():
                reward=1.
                if self.management_options[action] != 'default':
                    reward -=cost_managment
            else:
                reward=-0.0000000000000001
            
            #print(action, reward)
            
            return reward
            
        def reward_distance_PB(action=0):
            a,y,s=  self.state
#             dist=[]
#             for idx, boundary in enumerate(self.PB):
#                 dist_this_variable=np.linalg.norm(self.state[idx] - boundary)
#                 dist.append(dist_this_variable)
#             reward=1
#             for distance in dist:
#                 reward *=distance
            norm=np.linalg.norm(self.state-self.PB)
            #norm = (self.state[0] - self.A_PB)**2 
            if self._inside_planetary_boundaries():
                reward=1.
            else:
                reward=0.
            reward*=(norm)
            #print(norm, reward)
            return reward
        
        if choice_of_reward=='final_state':
            return reward_final_state
        elif choice_of_reward=='ren_knowledge':
            return reward_rel_share
        elif choice_of_reward=='desirable_region':
            return reward_desirable_region
        elif choice_of_reward=='PB':
            return reward_distance_PB
        elif choice_of_reward=='survive':
            return reward_survive
        elif choice_of_reward=='survive_cost':
            return reward_survive_cost
        elif choice_of_reward==None:
            print("ERROR! You have to choose a reward function!\n",
                   "Available Reward functions for this environment are: PB, rel_share, survive, desirable_region!")
            exit(1)
        else:
            print("ERROR! The reward function you chose is not available! " + choice_of_reward)
            print_debug_info()
            sys.exit(1)
    
    def calculate_expected_final_reward(self, gamma, max_steps, current_step=0):
        """
        Get the reward in the last state, expecting from now on always default.
        This is important since we break up simulation at final state, but we do not want the agent to 
        find trajectories that are close (!) to final state and stay there, since this would
        result in a higher total reward.
        """
        reward_final_state=self.reward_function()   
        remaining_steps=max_steps - current_step
        discounted_future_reward=0.
        for i in range(remaining_steps):
            discounted_future_reward += gamma**(i) * reward_final_state
        #print("Discounted future reward: " + str(discounted_future_reward))
        return discounted_future_reward
    
     
    def _compactification(self,x, x_mid):
        if x == 0:
            return 0.
        if x == np.infty:
            return 1.
        return x / (x + x_mid)    
    
    def _inv_compactification(self, y, x_mid):
        if y == 0:
            return 0.
        if np.allclose(y, 1):
            return np.infty
        return x_mid * y / (1 - y)

    
    def _inside_planetary_boundaries(self):
        a, y, s = self.state
        is_inside = True
        if a > self.A_PB or y < self.Y_SF or s<self.S_LIMIT:
            is_inside = False
            #print("Outside PB!")
        return is_inside
        
    def _arrived_at_final_state(self):
        a,y,s=self.state
        if np.abs(a - self.green_fp[0]) < self.final_radius and np.abs(y - self.green_fp[1]) < self.final_radius and np.abs(s-self.green_fp[2])< self.final_radius:
            return True
        elif np.abs(a - self.brown_fp[0]) < self.final_radius and np.abs(y - self.brown_fp[1]) < self.final_radius and np.abs(s-self.brown_fp[2])< self.final_radius:
            return True
        else:
            return False
              
    def _good_final_state(self):
        a,y,s=self.state
        if np.abs(a - self.green_fp[0]) < self.final_radius and np.abs(y - self.green_fp[1]) < self.final_radius and np.abs(s-self.green_fp[2])< self.final_radius:
            return True
        else:
            return False
    
    def _which_final_state(self):
        a,y,s=self.state
        if np.abs(a - self.green_fp[0]) < self.final_radius and np.abs(y - self.green_fp[1]) < self.final_radius and np.abs(s-self.green_fp[2])< self.final_radius:
            #print("ARRIVED AT GREEN FINAL STATE WITHOUT VIOLATING PB!")
            return Basins.GREEN_FP
        elif np.abs(a - self.brown_fp[0]) < self.final_radius and np.abs(y - self.brown_fp[1]) < self.final_radius and np.abs(s-self.brown_fp[2])< self.final_radius:
            return Basins.BROWN_FP
        else:
            return Basins.OUT_PB
    
    def get_plot_state_list(self):
        return self.state.tolist()
    
    
    
    def prepare_action_set(self, state):
        return np.arange(len(self.action_space)-1)
    
    
    def random_StartPoint(self):
       
        self.state=(0,0,0)
        while not self._inside_planetary_boundaries(): 
            a = np.random.uniform()
            y = np.random.uniform()
            s = np.random.uniform()
            self.state=(a,y,s)
        #print("Start: " + str(self.state))
        return (a, y, s)
    
    def current_state_region_StartPoint(self):
        
        self.state=[0,0,0]
        limit_start=0.05
        while not self._inside_planetary_boundaries(): 
            #self.state=self.current_state + np.random.uniform(low=-limit_start, high=limit_start, size=3)
            self.state[0] = self.current_state[0] + np.random.uniform(low=-limit_start, high=limit_start)
            self.state[1] = self.current_state[1] + np.random.uniform(low=-limit_start, high=limit_start)
            self.state[2] = 0.5
            
        #print(self.state)
        return self.state
    
    
        
    def _inside_box(self):
        """
        This function is needed to check whether our system leaves the predefined box (1,1,1).
        If values turn out to be negative, this is physically false, and we stop simulation and treat as a final state.
        """
        inside_box=True
        for x in self.state:
            if x < 0:
                x=0
                inside_box=False
        return inside_box
        
    
    def _get_parameters(self,action_number=0):
        
        """
        This function is needed to return the parameter set for the chosen management option.
        Here the action numbers are really transformed to parameter lists, according to the chosen 
        management option.
        Parameters:
            -action_number: Number of the action in the actionset.
             Can be transformed into: 'default', 'degrowth' ,'energy-transformation' or both DG and ET at the same time
        """
        if action_number < len(self.action_space):
            action=self.action_space[action_number]
        else:
            print("ERROR! Management option is not available!" + str (action))
            print(get_linenumber())
            sys.exit(1)
 
        parameter_list=[(self.beta_LG if action[0] else self.beta  ,
                         self.eps, self.phi, self.rho, 
                         self.sigma_ET if action[1] else self.sigma, 
                         self.tau_A, self.tau_S, self.theta)]
        
        return parameter_list 
        
    
    def plot_run(self,learning_progress):
        timeStart = 0
        intSteps = 2    # integration Steps
        dt=1
        sim_time_step=np.linspace(timeStart,self.dt, intSteps)
        fig, ax3d=create_figure()
        start_state=learning_progress[0][0]
        
        for state_action in learning_progress:
            state=state_action[0]
            action=state_action[1]
            parameter_list=self._get_parameters(action)
            traj_one_step=odeint(ays.AYS_rescaled_rhs, state, sim_time_step , args=parameter_list[0])
            # Plot trajectory
            my_color=ays_plot.color_list[action]
            ax3d.plot3D(xs=traj_one_step[:,0], ys=traj_one_step[:,1], zs=traj_one_step[:,2],
                            color=my_color, alpha=1., lw=3)
        
        # Plot from startpoint only one management option to see if green fix point is easy to reach:
        #self.plot_current_state_trajectories(ax3d)
        ays_plot.plot_hairy_lines(300, ax3d)
        
        final_state=self._which_final_state().name
        
        plt.show()
        
        return final_state
    
    def observed_states(self):
        return self.dimensions
                   
        
    def plot_current_state_trajectories(self, ax3d):
        # Trajectories for the current state with all possible management options
        time = np.linspace(0, 300, 1000)

        for action_number in range(len(self.action_space)):
            parameter_list=self._get_parameters(action_number)         
            my_color=self.color_list[action_number]
            traj_one_step=odeint(ays.AYS_rescaled_rhs, self.current_state,time , args=parameter_list[0])
            ax3d.plot3D(xs=traj_one_step[:,0], ys=traj_one_step[:,1], zs=traj_one_step[:,2],
                        color=my_color, alpha=.7, label=None) 
        
    
    def save_traj_final_state(self, learners_path, file_path, episode):
        final_state=self._which_final_state().name
        
        states=np.array(learners_path)[:,0]
        start_state=states[0]
        a_states=list(zip(*states))[0]
        y_states=list(zip(*states))[1]
        s_states=list(zip(*states))[2]
        
        actions=np.array(learners_path)[:,1]
        rewards=np.array(learners_path)[:,2]
        
        full_file_path=file_path +'/DQN_Path/'+ final_state +'/'
        if not os.path.isdir(full_file_path):
            os.makedirs(full_file_path)
        
        text_path = (full_file_path+ 
                     str (self.run_number) + '_' +  'path_'+  str(start_state)+ '_episode'+str(episode)  + '.txt' )
        with open(text_path, 'w') as f:
            f.write("# A  Y  S   Action   Reward \n")
            for i in range(len(learners_path)):
                f.write("%s  %s  %s   %s   %s \n" % (a_states[i] ,y_states[i], s_states[i], actions[i] , rewards[i]))
        f.close()
        print('Saved :' + text_path)
    
      
    def save_traj(self,ax3d, fn):
        ax3d.legend(loc='best', prop={'size': 12})
        plt.savefig(fname=fn)
        plt.close()
    
    def define_test_points(self):
        testpoints=[
            [0.49711988, 0.49849855, 0.5], 
            [0.48654806, 0.51625583, 0.5],
            [0.48158348, 0.50938806, 0.5],
            [0.51743486, 0.45828958, 0.5],
            [0.52277734, 0.49468274, 0.5],
            [0.49387675, 0.48199759, 0.5],
            [0.45762969, 0.50656114, 0.5]
         ]
        return testpoints
    
    def test_Q_states(self):
        # The Q values are choosen here in the region of the knick and the corner 
        testpoints=[
            [0.5, 0.5, 0.5],
            [0.48158348, 0.50938806, 0.5], # points around current state
            [0.51743486, 0.45828958, 0.5],
            [0.52277734, 0.49468274, 0.5],
            [0.49711988, 0.49849855, 0.5], 
            [ 0.5642881652513302 , 0.4475774101441196 , 0.5494879542441825 ], # From here on for knick to green FP
            [ 0.5677565382994565 , 0.4388184256945361 , 0.5553589418072845 ],
            [ 0.5642881652513302 , 0.4475774101441196 , 0.5494879542441825 ],
            [ 0.5667064632786063 , 0.4417642808582638 , 0.5534355600174762 ],
            [ 0.5677565382994565 , 0.4388184256945361 , 0.5553589418072845 ],
            [ 0.5667064632786063 , 0.4417642808582638 , 0.5534355600174762 ],
            [ 0.5642881652513302 , 0.4475774101441196 , 0.5494879542441825 ],
            [ 0.5667064632786063 , 0.4417642808582638 , 0.5534355600174762 ],
            [ 0.5677565382994565 , 0.4388184256945361 , 0.5553589418072845 ],
            [ 0.5667064632786063 , 0.4417642808582638 , 0.5534355600174762 ],
            [ 0.565551647191721 , 0.4446849282686741 , 0.5514780427327116 ],
            [ 0.5667064632786063 , 0.4417642808582638 , 0.5534355600174762 ],
            [ 0.5732889740892303 , 0.40670386098365746 , 0.5555233190964499 ],
            [ 0.575824650184652 , 0.4053645419804867 , 0.4723020776953208 ],
            [ 0.5770448313058577 , 0.4048031241155815 , 0.418890921031026 ],  # From here on for knick to black FP
            [ 0.5731695199856403 , 0.40703303828389187 , 0.5611291038925613 ],
            [ 0.5742215704891825 , 0.42075928220225944 , 0.4638131691273601 ],
            [ 0.5763299679962532 , 0.411095026888074 , 0.4294020150808698 ],
            [ 0.5722546035810613 , 0.41315124675768045 , 0.5695919593600399 ],
            [ 0.5762062083990029 , 0.405168276738863 , 0.4567816125395152 ],
            [ 0.5762327254875753 , 0.4052313013623205 , 0.4568789522146076 ],
            [ 0.5770448313058577 , 0.4048031241155815 , 0.418890921031026 ],
            [ 0.5770448313058577 , 0.4048031241155815 , 0.418890921031026 ],
            [ 0.5726685871808355 , 0.40709323935138103 , 0.5727121746516005 ],
            [ 0.2841645298525685 , 0.5742868996790442 , 0.9699317116062534],   # From here on region of the shelter
            [ 0.32909951420599637, 0.6082136751752725 , 0.9751810127843358],
            [ 0.5649255262907135 , 0.4238116683903446 , 0.8009508342049909],
            [ 0.04143141196994614, 0.9467759116676885 , 0.9972458138530155],
            ]
        return testpoints
    
    
"""
This is the implementation of a partially observable environment. 
The internal environment is still deterministic, however does not capture the full state information.
Which state information is provided can be chosen arbitrarily.
"""

class noisy_partially_observable_AYS(AYS_Environment):
        
    def __init__(self, t0=0, dt=1 , reward_type='PB', image_dir='./images/', run_number=0, plot_progress=False,
                 observables=dict(  A=True,
                                    Y=True,
                                    S=True
                            ),
                 noise_strength=0.00 ):
        
        AYS_Environment.__init__(self, t0, dt, reward_type, image_dir, run_number, plot_progress)
        self.observables=observables        
        self.obs_array=self._which_measurable_parameters()
        self.observation_space=np.array(self.state)[self.obs_array]
        
        self.noise_strength=noise_strength
    
    def _which_measurable_parameters (self):     
        
        obs_idx_array=[]
        if self.observables['A']:
            obs_idx_array.append(0)
        if self.observables['Y']:
            obs_idx_array.append(1)
        if self.observables['S']:
            obs_idx_array.append(2)
        return obs_idx_array
    
    def observed_states(self):
        return self.dimensions[self.obs_array]
    
    def _add_noise (self, state):     
        noise=np.random.uniform(low=0, high=self.noise_strength, size=(len(state)))
        noisy_state = state + noise
        noisy_state[noisy_state>1.]=1.
        return noisy_state
        
    def step(self, action):
        """
        This function performs one simulation step in a RFL algorithm. 
        It updates the state and returns a reward according to the chosen reward-function.
        """

        next_t= self.t + self.dt
        self.state=self._perform_step(action, next_t)
        self.t=next_t
        if self._arrived_at_final_state():
            self.final_state = True
        
        reward=self.reward_function(action)
        if not self._inside_planetary_boundaries():
            self.final_state = True
            #print("Left planetary boundaries!" + str(self.state))
            reward=0
        
        # Adjust return state to agent
        part_state=self.state[self.obs_array]
        #print("True state in Environment: ", self.state)
        return_state=self._add_noise(part_state)
        
        return return_state, reward, self.final_state            



    """
    This functions are needed to reset the Environment to specific states
    """
    def reset(self):
        self.start_state=self.state=np.array(self.current_state_region_StartPoint())

        self.final_state=False
        self.t=self.t0
        return_state=self.state[self.obs_array]
        
        return return_state    
    
    
    def reset_for_state(self, state=None):
        if state==None:
            self.start_state=self.state=np.array(self.current_state)
        else:
            self.start_state=self.state=np.array(state)
        self.final_state=False
        self.t=self.t0

        return_state=self.state[self.obs_array]
#         print("Reset to state: " , return_state)

        return return_state  
