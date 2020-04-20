from c_global.cG_LAGTPKS_Environment import compactification
from c_global.partially_observable_cG import noisy_partially_observable_LAGTPKS
import numpy as np

class PB_cG(noisy_partially_observable_LAGTPKS):
    
    def __init__(self, t0=0, dt=1 , reward_type=None, image_dir=None, noise_strength=0,run_number=0, plot_progress=False,
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
                            ren_sub=.5,
                            carbon_tax=.5,
                            i_DG=0.1,
                            L0=0,
                            )   , # pars contains the parameters for the global model
                 specs=[] ,# contains specifications for the global model as e.g. the INST_GREENHOUSE
        ):
        
        observables=dict(  L=True, A=True, G=True, T=True, P=True, K=True, S=True )
        noisy_partially_observable_LAGTPKS.__init__(self, t0, dt, reward_type, image_dir, run_number, plot_progress, ics, pars, specs, observables, noise_strength)
                
        self.observation_space=self._get_measurement_PB(self.state)
        
        print("PB-observable c:GLOBAL space initialized! \n",
              "Agent can observe: ", self._observed_states()  )
        
        self.ini_measured_state=np.array([self.iniDynVar['G'],  self.iniDynVar['P']])
    
    
    def _observed_states(self):
        return ['A_PB', 'W_PB', 'P_PB']
   
    def _get_measurement_PB(self, state):
        L,A,G,T,P,K,S = state
#         Leff=L
#         if self.Lprot:
#             Leff=max(L-self.L0, 0)
#         W=self.direct_W(Leff, G, P, K, S)
        return np.array([G, P])
        
     
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
        
        measured_states=self._get_measurement_PB(self.state)
        
        trafo_state=compactification(measured_states, self.ini_measured_state )
        return_state=self._add_noise(trafo_state)
#         print("Step PB", return_state)

        return return_state, reward, self.final_state       
    
    """
    This functions are needed to reset the Environment to specific states
    """
    def reset(self):
        self.start_state=self.state=np.array(self.current_state_region_StartPoint())
        measured_states=self._get_measurement_PB(self.state)

        trafo_state=compactification(measured_states, self.ini_measured_state)

        self.final_state=False
        self.t=self.t0
        return_state=self._add_noise(trafo_state)
        
        return return_state    
    
    
    def reset_for_state(self, state=None):
        if state==None:
            self.start_state=self.state=self.current_state
        else:
            self.start_state=self.state=np.array(state)
        self.final_state=False
        self.t=self.t0
        
        measured_states=self._get_measurement_PB(self.state)
        trafo_state=compactification(measured_states, self.ini_measured_state)
        return_state=self._add_noise(trafo_state)

#         print("Reset to state: " , return_state)

        return return_state      
        