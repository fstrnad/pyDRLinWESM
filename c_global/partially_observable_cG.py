"""
This is the implementation of a partially observable environment. 
The internal environment is still deterministic, however does not capture the full state information.
Which state information is provided can be chosen arbitrarily.
@author: Felix Strnad

"""

from c_global.cG_LAGTPKS_Environment import cG_LAGTPKS_Environment, compactification
import numpy as np


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


class partially_observable_cG(cG_LAGTPKS_Environment):
    
    def __init__(self, t0=0, dt=1 , reward_type=None, image_dir=None, run_number=0, plot_progress=False,
                 ics=dict(  L=2480.,  
                            A=758.0,
                            G=1125,
                            T=5.053333333333333e-6,
                            P=6e9,
                            K=6e13,
                            S=5e11
                            ) , # ics defines the initial values!
                 pars=dict( Sigma = 1.5 * 1e8,
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
                            b=5.4*1e-7,
                            yE=147,
                            eB=4*1e10,
                            eF=4*1e10,
                            i=0.25,
                            k0=0.1,
                            aY=0.,
                            aB=3e5,
                            aF=5e6,
                            aR=7e-18,
                            sS=1./50.,
                            sR=1.,
                            ren_sub=1,
                            carbon_tax=.5,
                            i_DG=0.1,
                            L0=0,
                            )   , # pars contains the parameters for the global model
                 specs=[] ,# contains specifications for the global model as e.g. the INST_GREENHOUSE
                 observables=dict(  L=True,  
                                    A=True,
                                    G=True,
                                    T=True,
                                    P=True,
                                    K=True,
                                    S=True
                            ),
        ):
        
        
        cG_LAGTPKS_Environment.__init__(self, t0, dt, reward_type, image_dir, run_number, plot_progress, ics, pars, specs)
                
        self.observables=observables        
        self.obs_array=self._which_measurable_parameters()
        self.observation_space=self.state[self.obs_array]
        
        print("Partially observable c:GLOBAL space initialized!\n",
              "Agent can observe: ", self._observed_states()  )
    
    def _which_measurable_parameters (self):     
        
        obs_idx_array=[]
        if self.observables['L']:
            obs_idx_array.append(0)
        if self.observables['A']:
            obs_idx_array.append(1)        
        if self.observables['G']:
            obs_idx_array.append(2)
        if self.observables['T']:
            obs_idx_array.append(3)
        if self.observables['P']:
            obs_idx_array.append(4)
        if self.observables['K']:
            obs_idx_array.append(5)
        if self.observables['S']:
            obs_idx_array.append(6)
        return obs_idx_array
    
    def observed_states(self):
        return self.dimensions[self.obs_array]
    """
    This function is equal to the step function in cG_LAGTPKS considering the dynamics inside the model.
    However it returns only the measurable parameters 
    """
    def step(self, action):
        """
        This function performs one simulation step in a RFL algorithm. 
        It updates the state and returns a reward according to the chosen reward-function.
        """

        next_t= self.t + self.dt
        self._adjust_parameters(action)

        
        self.state=self._perform_step( next_t)
        self.t=next_t
        if self._arrived_at_final_state():
            self.final_state = True
        
        reward=self.reward_function(action)
        if not self._inside_planetary_boundaries():
            self.final_state = True
        trafo_state=compactification(self.state, self.current_state)
        
        return_state=trafo_state[self.obs_array]
        
        
        return return_state, reward, self.final_state       
    
    """
    This functions are needed to reset the Environment to specific states
    """
    def reset(self):
        self.start_state=self.state=np.array(self.current_state_region_StartPoint())
        trafo_state=compactification(self.state, self.current_state)

        self.final_state=False
        self.t=self.t0
        return_state=trafo_state[self.obs_array]
        
        return return_state    
    
    
    def reset_for_state(self, state=None):
        if state==None:
            self.start_state=self.state=self.current_state
        else:
            self.start_state=self.state=np.array(state)
        self.final_state=False
        self.t=self.t0
        trafo_state=compactification(self.state, self.current_state)
        return_state=trafo_state[self.obs_array]
#         print("Reset to state: " , return_state)

        return return_state  
        
       

"""
This is the implementation of a noisy environment. 
The internal environment is still deterministic, only the returned state is noisy such that the agent observed only noisy states.
The noise here is distributed uniformly. 
"""
class noisy_partially_observable_LAGTPKS(partially_observable_cG):
    
    def __init__(self, t0=0, dt=1 , reward_type=None, image_dir=None, run_number=0, plot_progress=False,
                 ics=dict(  L=2480.,  
                            A=758.0,
                            G=1125,
                            T=5.053333333333333e-6,
                            P=6e9,
                            K=6e13,
                            S=5e11
                            ) , # ics defines the initial values!
                 pars=dict( Sigma = 1.5 * 1e8,
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
                            b=5.4*1e-7,
                            yE=147,
                            eB=4*1e10,
                            eF=4*1e10,
                            i=0.25,
                            k0=0.1,
                            aY=0.,
                            aB=3e5,
                            aF=5e6,
                            aR=7e-18,
                            sS=1./50.,
                            sR=1.,
                            ren_sub=1,
                            carbon_tax=.5,
                            i_DG=0.1,
                            L0=0,
                            )   , # pars contains the parameters for the global model
                 specs=[] ,# contains specifications for the global model as e.g. the INST_GREENHOUSE
                 observables=dict(  L=True,  
                                    A=True,
                                    G=True,
                                    T=True,
                                    P=True,
                                    K=True,
                                    S=True
                            ),
                 noise_strength=0.00   ,
        ):
        
        
        partially_observable_cG.__init__(self, t0, dt, reward_type, image_dir, run_number, plot_progress, ics, pars, specs, observables)
                
        self.noise_strength=noise_strength
        
        print("Noisy Partially observable c:GLOBAL space initialized! \n",
              "Agent can observe: ", self.dimensions[self.obs_array]  )
        print("Noisy environment with strength: ", noise_strength )
        


    def _add_noise (self, state):     
        noise=np.random.uniform(low=0, high=self.noise_strength, size=(len(state)))
        
        noisy_state = state + noise
        noisy_state[noisy_state>1.]=1.
        return noisy_state
        
    """
    This function is equal to the step function in cG_LAGTPKS considering the dynamics inside the model.
    However it returns only the measurable parameters 
    """
    def step(self, action):
        """
        This function performs one simulation step in a RFL algorithm. 
        It updates the state and returns a reward according to the chosen reward-function.
        This is not affected by the noise. 
        """
        next_t= self.t + self.dt
        self._adjust_parameters(action)
        
        self.state=self._perform_step( next_t)
        self.t=next_t
        if self._arrived_at_final_state():
            self.final_state = True
        
        reward=self.reward_function(action)
        if not self._inside_planetary_boundaries():
            self.final_state = True
        trafo_state=compactification(self.state, self.current_state)
        
        part_state=trafo_state[self.obs_array]
        return_state=self._add_noise(part_state)
        
#         print("Step Noisy Environment", return_state)

        return return_state, reward, self.final_state       
    
        














# import numpy as np
# test_arr=np.array([0.5, 0.8, 0.9, 1.2, 0.3, 1.8])
# test_arr[test_arr > 1]=1
# print(test_arr)
        
                