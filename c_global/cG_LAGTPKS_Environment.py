"""
This is the implementation of the c:GLOBAL Environment in the form
that it can used within the Agent-Environment interface 
in combination with the DRL-agent.

@author: Felix Strnad
"""
import sys

import numpy as np
from scipy.integrate import odeint

from gym import Env
from collections import OrderedDict
from DeepReinforcementLearning.Basins import Basins
import mpl_toolkits.mplot3d as plt3d
import matplotlib.pyplot as plt

from matplotlib.font_manager import FontProperties
from matplotlib.offsetbox import AnchoredText
import matplotlib.ticker as ticker


import heapq as hq
import functools as ft
import operator as op
INFTY_SIGN = u"\u221E"


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

    

from inspect import currentframe, getframeinfo

def get_linenumber():
    print_debug_info()
    print("Line: ")
    cf = currentframe()
    return cf.f_back.f_lineno

def print_debug_info():
    frameinfo = getframeinfo(currentframe())
    print ("File: ", frameinfo.filename)
    
@np.vectorize
def compactification(x, x_mid):
    if x == 0:
        return 0.
    if x == np.infty:
        return 1.
    return x / (x + x_mid)

@np.vectorize
def inv_compactification(y, x_mid):
    if y == 0:
        return 0.
    if np.allclose(y, 1):
        return np.infty
    return x_mid * y / (1 - y)

class cG_LAGTPKS_Environment(Env):
    """
    This Environment describes the 7D implementation of the copan:GLOBAL model developed by Jobst Heitzig.
    The parameters are taken from Nitzbon et al. 2017. 
    The code contains implementation parts that go back to Jan Nitzbon 2016 
    Dynamic variables are :
        - terrestrial ("land") carbon L
        - excess atmospheric carbon stock A
        - geological carbon G
        - temperature T
        - population P
        - capital K
        - the renewable energy knowledge stock S
    
    Parameters (mainly Nitzbon et al. 2016 )
    ----------
        - sim_time: Timestep that will be integrated in this simulation step
          In each grid point the agent can choose between subsidy None, A, B or A and B in combination. 
        - Sigma = 1.5 * 1e8
        - CstarPI=4000
        - Cstar=5500
        - a0=0.03
        - aT=3.2*1e3
        - l0=26.4
        - lT=1.1*1e6
        - delta=0.01
        - m=1.5
        - g=0.02
        - p=0.04
        - Wp=2000
        - q0=20
        - b=5.4*1e-7
        - yE=147
        - eB=4*1e10
        - eF=4*1e10
        - i=0.25
        - k0=0.1
        - aY=0.
        - aB= 3e5 (varied, basic year 2000)
        - aF= 5e6 (varied, basic year 2000)
        - aR= 7e-18 (varied, basic year 2000)
        - sS=1./50.
        - sR=1.
    """
    management_options=['default', 
                        'Sub' , 'Tax','NP' ,
                        'Sub+Tax', 'Sub+NP', 'Tax+NP',
                        'Sub+Tax+NP' ]

#     management_options=['default', 
#                         'SubTax','DG' , 'NP',
#                         'SubTax+DG', 'SubTax+NP', 'DG+NP',
#                         'SubTax+DG+NP' ]
    action_space=[(False, False, False), 
                        (True, False,False), (False, True, False), (False, False, True),
                        (True, True, False), (True, False, True) , (False, True, True),
                        (True, True, True)
                        ]
    dimensions=np.array(['L', 'A', 'G', 'T', 'P', 'K', 'S'])
    
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
                            ren_sub=.5,
                            carbon_tax=.5,
                            i_DG=0.1,
                            L0=0,
                            )   , # pars contains the parameters for the global model
                 specs=[] # contains specifications for the global model as e.g. the INST_GREENHOUSE  
                 ):
        
        self.image_dir=image_dir
        self.run_number = run_number
        self.plot_progress=plot_progress
        # The grid defines the number of cells, hence we have 8x8 possible states
        self.final_state=False
        self.reward=0
        
        self.reward_function=self.get_reward_function(reward_type)
        
        timeStart = 0
        intSteps = 10    # integration Steps
        self.t=self.t0=t0
        self.dt=dt
        
        self.sim_time_step=np.linspace(timeStart,dt, intSteps)
        
        self.specs=specs
        self.setParams(paramDict=pars)
        self.setInitials(iniDict=ics)
        
        
        # Definitions from outside
        self.state=self.current_state=np.array([self.iniDynVar['L'], self.iniDynVar['A'], self.iniDynVar['G'], self.iniDynVar['T'], 
                                       self.iniDynVar['P'], self.iniDynVar['K'], self.iniDynVar['S'] 
                                       ])
        self.state=self.start_state=self.current_state
        self.Lprot=False
        self.observation_space=self.state
        
        # Planetary Boundaries for A, Y, P (population!)
        self.A_PB=945
        self.A_scale=1
        self.Y_PB=self.direct_Y(self.iniDynVar['L'], self.iniDynVar['G'], self.iniDynVar['P'], self.iniDynVar['K'], self.iniDynVar['S'])
        self.P_PB=1e6
        self.W_PB= (1- self.params['i'])*self.Y_PB / (1.01*self.iniDynVar['P'])   # Economic production in year 2000 and population in year 2000
        self.W_scale=1e3
        self.PB=np.array([self.A_PB, self.W_PB, self.P_PB])
        self.compact_PB=compactification(self.PB, self.ini_state)    

        self.P_scale=1e9
        self.reward_type=reward_type
        
        print("Initialized c:GLOBAL environment!" ,
              "\nReward type: " + str(reward_type),
              "\nPlanetary Boundaries are: " + str(self.PB),
              "\nInitial LAGTPKS-values are: " + str(self.ini_state),
              "\nInitial derived values are: Wini:"+str(self.Wini)+"Yini: "+str(self.Yini))
        
        self.color_list=['orangered', 'mediumvioletred', 'darkgreen', 'midnightblue', 'yellow', 'goldenrod', 'slategrey', 'olive' ] # Contains as many numbers as management options!

    """
    This function is only basic function an Environment needs to provide
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
            #print("Left planetary boundaries!" + str(self.state))
        trafo_state=compactification(self.state, self.current_state)
        #print(self.state, trafo_state)
    #    return self.state, reward, self.final_state
        return trafo_state, reward, self.final_state
    
    
    
    def _perform_step(self, next_t):
        
        #print(parameter_list[0])
        #print(self.state)
        
        traj_one_step=odeint(self.dDynVar, self.state, [self.t, next_t] , mxstep=50000)
        l = traj_one_step[:,0][-1]
        a = traj_one_step[:,1][-1]
        g = traj_one_step[:,2][-1]
        t = traj_one_step[:,3][-1]
        p = traj_one_step[:,4][-1]
        k = traj_one_step[:,5][-1]
        s = traj_one_step[:,6][-1]
        
        #l,a,g,t,p,k,s= self.state

        return np.array( (l,a,g,t,p,k,s) )
    
    """
    This functions are needed to reset the Environment to specific states
    """
    def reset(self):
        self.start_state=self.state=np.array(self.current_state_region_StartPoint())
        trafo_state=compactification(self.state, self.current_state)

        self.final_state=False
        self.t=self.t0
#        return self.state    
        return trafo_state    
    
    
    def reset_for_state(self, state=None):
        if state==None:
            self.start_state=self.state=self.current_state
        else:
            self.start_state=self.state=np.array(state)
        self.final_state=False
        self.t=self.t0
        trafo_state=compactification(self.state, self.current_state)
        
        return trafo_state
        
    
    """
    This function defines the reward the Environment returns to the player for a given action
    """
    def get_reward_function(self,choice_of_reward):
        """
        This function returns one function as a function pointer according to the reward type we chose 
        for this simulation.
        """
        def reward_final_state(action=0):
            """
            Reward in the final  green fixpoint_good 100. , else 0.
            """
            if self._good_final_state():
                reward=2.
            else:
                if self._inside_planetary_boundaries():
                    reward=1.
                else:
                    reward=0.
            return reward
        
        def reward_ren_knowledge(action=0):
            """
            We want to:
            - maximize the knowledge stock of renewables S 
            """
            l,a,g,t,p,k,s = self.state
            if self._inside_planetary_boundaries():
                reward=compactification(s, self.iniDynVar['S'])
            else:
                reward=0.
            
            return reward       
        def reward_desirable_region(action=0):
            l,a,g,t,p,k,s = self.state
            desirable_share_renewable=self.iniDynVar['S']
            reward=0.
            if s >= desirable_share_renewable:
                reward=1.
            return reward
        
        def reward_survive(action=0):
            if self._inside_planetary_boundaries():
                reward=1.
            else:
                reward=0.
            return reward
        
        def reward_survive_cost(action=0):
            cost_managment=0.03
            if self._inside_planetary_boundaries():
                reward=1.
                if self.management_options[action] != 'default':
                    reward -=cost_managment
            else:
                reward=-1e-30
            
            return reward
        
        def reward_distance_PB(action=0):
            L,A,G,T,P,K,S=  self.state
            Leff=L
            if self.Lprot:
                Leff=max(L-self.L0, 0)            
            W=self.direct_W(Leff, G, P, K, S)
            
            #norm=np.linalg.norm(np.array([(A - self.A_PB)/Aini , (W-self.W_PB)/Wini, (P-self.P_PB)/Pini ]))
            #norm = (self.state[0] - self.A_PB)**2 
            if self._inside_planetary_boundaries():
                norm=np.linalg.norm( self.compact_PB -  compactification( np.array([A, W, P]), self.ini_state))
                #print("reward-function: ", norm)
                reward=norm
            else:
                reward=0.
            
            return reward
         
        if choice_of_reward=='final_state':
            return reward_final_state
        elif choice_of_reward=='ren_knowledge':
            return reward_ren_knowledge
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
        else:
            print("ERROR! The reward function you chose is not available! " + choice_of_reward)
            print_debug_info()
            sys.exit(1)
   
    
    """
    This functions define the dynamics of the copan:GLOBAL model
    """
    def dDynVar(self, LAGTPKS, t):
        #auxiliary functions
        #photosynthesis
        def phot(L, A, T):
            return (self.params['l0']-self.params['lT']*T)*np.sqrt(A)/np.sqrt(self.params['Sigma'])
        #respiration
        def resp(L, T):
            return self.params['a0']+self.params['aT']*T
        #diffusion atmosphere <--> ocean
        def diff(L, A, G=0.):
            return self.params['delta']*(self.params['Cstar']-L-G-(1+self.params['m'])*A)
        
        def fert(P,W):
            return 2*self.params['p']*self.params['Wp']*W/(self.params['Wp']**2+W**2) 
    
        def mort(P,W):
            return self.params['q0']/(W) + self.params['qP']*P/self.params['Sigma']
        
        
        L, A, G, T, P, K, S= LAGTPKS
        #adjust to lower and upper bounds
        L=np.amin([np.amax([L, 1e-12]), self.params['Cstar']])
        A=np.amin([np.amax([A, 1e-12]), self.params['Cstar']])
        G=np.amin([np.amax([G, 1e-12]), self.params['Cstar']])
        T=np.amax([T, 1e-12])
        P=np.amax([P, 1e-12])
        K=np.amax([K, 1e-12])
        S=np.amax([S, 1e-12])

        # calculate T and A if instantaneous processes
        if 'INST_DIFF' in self.specs:
            A = (self.params['Cstar']-L-G) / (1.+self.params['m'])
        if 'INST_GH' in self.specs:
            T = A/self.params['Sigma']
        #calculate auxiliary quantities
        
        if self.Lprot:
            Leff=max(L-self.L0, 0)
        else:
            Leff=L
        Xb=self.params['aB']*Leff**2.
        Xf=self.params['aF']*G**2.
        Xr=self.params['aR']*S**2. 
        X=Xb+Xf+Xr
        
        expP=2./5.
        expK=2./5.
        if 'KproptoP' in self.specs:
#             expP=4./5.
#             expK=0.
            K = P*self.iniDynVar['K']/(self.iniDynVar['P'])
        if 'NproptoP' in self.specs:
            expP-=1./5.   # gives in combination expP=3./5
        Z=self.Z(P, K, X, expP, expK)
        
        #calculate derived variables
        B=self.B(Xb, Z)
        F=self.F(Xf, Z)
        R=self.R(Xr, Z)

        Y=self.Y(B, F, R)
        W=self.W(Y, P, L)

        #calculate derivatives of the dynamic variables
        dL = (phot(L, A, T) - resp(L, T)) * L - B
        #print(self.phot(L, A, T) *L  , self.resp(L, T)*L , B , T)
        dA = -dL + diff(L, A, G)
        dG = -F
        dT = self.params['g'] * (A/self.params['Sigma'] - T)
        dP = P * (fert(P,W)-mort(P,W))
        dK = self.params['i'] * Y - self.params['k0'] * K
        dS = self.params['sR']*R - self.params['sS']*S

        if 'INST_DIFF' in self.specs:
            dA = -(dL+dG)/(1.+self.params['m'])
        if 'INST_GH' in self.specs:
            dT = dA/self.params['Sigma']
        
        #print(t, self.Lprot, L,  self.L0 , Leff, B, phot(L, A, T) , resp(L, T) )
        #print(Y, K, self.params['i'], self.params['k0'], dK)
        #print(R, S, self.params['sS'], dS)
        #print(W, P, dP)
        return [dL, dA, dG, dT, dP, dK, dS]        

    def setInitials(self,iniDict):
        self.iniDynVar=OrderedDict()
        if 'L' in iniDict.keys():
            L = iniDict['L']
            try:
                assert 0 <= L <= self.params['Cstar'], "L must be between 0 and Cstar"
                try: assert L <= self.params['Cstar'] - self.iniDynVar['A'], "L must be <= Cstar - A"
                except: pass
            except: pass
            self.iniDynVar['L'] = L
        
        if 'A' in iniDict.keys():
            A = iniDict['A']
            try:
                assert 0 <= A <= self.params['Cstar'], "A must be between 0 and Cstar"
                try: assert A <= self.params['Cstar'] - self.iniDynVar['L'], "A must be <= Cstar - L"
                except: pass
            except: pass
            self.iniDynVar['A'] = A
        
        if 'G' in iniDict.keys():
            G = iniDict['G']
            try:
                assert 0 <= G <= self.params['Cstar'], "G must be between 0 and Cstar"
            except: pass
            self.iniDynVar['G'] = G
            
        if 'T' in iniDict.keys():
            T = iniDict['T']
            try:
                assert 0 <= T, "T must be non-negative"
            except: pass
            self.iniDynVar['T'] = T
        
        if 'P' in iniDict.keys():
            P = iniDict['P']
            try:
                assert 0 <= P, "P must be non-negative"
            except: pass
            self.iniDynVar['P'] = P
            
        if 'K' in iniDict.keys():
            K = iniDict['K']
            try:
                assert 0 <= K, "K must be non-negative"
            except: pass
            self.iniDynVar['K'] = K
        
        if 'S' in iniDict.keys():
            S = iniDict['S']
            try:
                assert 0 <= S, "S must be non-negative"
            except: pass
            self.iniDynVar['S'] = S
            
        self.Aini=self.iniDynVar['A']
        self.Pini=self.iniDynVar['P']
        
        
        Xb=self.params['aB']*self.iniDynVar['L']**2.
        Xf=self.params['aF']*self.iniDynVar['G']**2.
        Xr=self.params['aR']*self.iniDynVar['S']**2. 
        X=Xb+Xf+Xr
        
        expP=2./5.
        expK=2./5.
        Z=self.Z(self.iniDynVar['P'], self.iniDynVar['K'], X, expP, expK)
        
        #calculate derived variables
        self.Bini=self.B(Xb, Z)
        self.Fini=self.F(Xf, Z)
        self.Rini=self.R(Xr, Z)

        self.Yini=self.Y(self.Bini, self.Fini, self.Rini)
        self.Wini=self.W(self.Yini, self.Pini, self.iniDynVar['L'])
        
        self.ini_state=np.array([self.Aini, self.Wini, self.Pini])
            
    def setParams(self,paramDict):
        self.params={}
        if 'Cstar' in paramDict.keys():
            Cstar = paramDict['Cstar']
            assert 0 < Cstar, "Cstar must be positive"
            self.params['Cstar']=Cstar
            
        if 'Sigma' in paramDict.keys():
            Sigma = paramDict['Sigma']
            assert 0 < Sigma, "Sigma must be positive"
            self.params['Sigma'] = Sigma 
            
        if 'm' in paramDict.keys():
            m = paramDict['m']
            assert 0 < m, "m must be positive"
            self.params['m'] = m
        
        if 'a0' in paramDict.keys():
            a0 = paramDict['a0']
            assert 0 <= a0, "a0 must be non-negative"
            self.params['a0'] = a0
            
        if 'aT' in paramDict.keys():
            aT = paramDict['aT']
            assert 0 <= aT, "aT must be non-negative"
            self.params['aT'] = aT
        
        if 'l0' in paramDict.keys():
            l0 = paramDict['l0']
            assert 0 <= l0, "l0 must be non-negative"
            self.params['l0'] = l0
            
        if 'lT' in paramDict.keys():
            lT = paramDict['lT']
            assert 0 <= lT, "lT must be non-negative"
            self.params['lT'] = lT
        
        if 'delta' in paramDict.keys():
            delta = paramDict['delta']
            assert 0 < delta, "delta must be positive"
            self.params['delta'] = delta
 
        if 'g' in paramDict.keys():
            g = paramDict['g']
            assert 0 < g, "g must be positive"
            self.params['g'] = g
        
        if 'p' in paramDict.keys():
            p = paramDict['p']
            assert 0 <= p, "p must be non-negative"
            self.params['p'] = p
        
        if 'q0' in paramDict.keys():
            q0 = paramDict['q0']
            assert 0 <= q0, "p must be non-negative"
            self.params['q0'] = q0
        
        if 'qP' in paramDict.keys():
            qP = paramDict['qP']
            assert 0 <= qP, "p must be non-negative"
            self.params['qP'] = qP
        
        if 'Wp' in paramDict.keys():
            Wp = paramDict['Wp']
            assert 0 <= Wp, "p must be non-negative"
            self.params['Wp'] = Wp
            
        if 'yE' in paramDict.keys():
            yE = paramDict['yE']
            assert 0 <= yE, "p must be non-negative"
            self.params['yE'] = yE
            
        if 'wL' in paramDict.keys():
            wL = paramDict['wL']
            assert 0 <= wL, "p must be non-negative"
            self.params['wL'] = wL
            
        if 'eB' in paramDict.keys():
            eB = paramDict['eB']
            assert 0 <= eB, "eB must be non-negative"
            self.params['eB'] = eB
            
        if 'eF' in paramDict.keys():
            eF = paramDict['eF']
            assert 0 <= eF, "eF must be non-negative"
            self.params['eF'] = eF       
            
        if 'aY' in paramDict.keys():
            aY = paramDict['aY']
            assert 0 <= aY, "aY must be non-negative"
            self.params['aY'] = aY

        if 'aB' in paramDict.keys():
            aB = paramDict['aB']
            assert 0 <= aB, "aB must be non-negative"
            self.params['aB'] = aB

        if 'aF' in paramDict.keys():
            aF = paramDict['aF']
            assert 0 <= aF, "aF must be non-negative"
            self.params['aF'] = aF
        
        if 'aR' in paramDict.keys():
            aR = paramDict['aR']
            assert 0 <= aR, "aR must be non-negative"
            self.params['aR'] = aR
        
        if 'i' in paramDict.keys():
            i = paramDict['i']
            assert 0 <= i <= 1., "i must be between 0 and 1"
            self.params['i'] = i
        
        if 'k0' in paramDict.keys():
            k0 = paramDict['k0']
            assert 0 <= k0, "k0 must be non-negative"
            self.params['k0'] = k0
        
        if 'sR' in paramDict.keys():
            sR = paramDict['sR']
            assert 0 <=sR , "sR must be non-negative"
            self.params['sR']=sR
        
        if 'sS' in paramDict.keys():
            sS = paramDict['sS']
            assert 0 <=sS , "sS must be non-negative"
            self.params['sS']=sS
            
        if 'ren_sub' in paramDict.keys():
            ren_sub=paramDict['ren_sub']
        if 'carbon_tax' in paramDict.keys():
            carbon_tax=paramDict['carbon_tax']
        if 'i_DG' in paramDict.keys():
            i_DG=paramDict['i_DG']
        if 'L0' in paramDict.keys():
            L0=paramDict['L0']
            
            
        # Here default parameters before management is used
        self.aR_default=aR
        self.aB_default=aB
        self.aF_default=aF
        self.i_default=i
        
        self.L0=L0
        
        self.ren_sub=ren_sub
        self.carbon_tax=carbon_tax
        self.i_DG=i_DG
    
    # maritime stock
    
    def M(self, L, A, G):
        return self.params['Cstar']-L-A-G
    #economic production
    def Y(self, B, F, R):
        #return self.params['y'] * ( self.params['eB']*B + self.params['eF']*F )
        # Y = y * E     E = E_B + E_F + R
        return self.params['yE'] * ( self.params['eB']*B + self.params['eF']*F + R )
    #wellbeing
    def W(self, Y, P, L):
        return (1.-self.params['i']) * Y / P + self.params['wL']*L/self.params['Sigma']
    # energy sector
    #auxiliary
    def Z(self, P, K, X, expP=2./5, expK=2./5.):
        return P**expP * K**expK / X**(4./5.)
        
    def B(self, Xb, Z):
        return Xb * Z / self.params['eB']
    def F(self, Xf, Z):
        return Xf * Z / self.params['eF']
    def R(self,Xr, Z):
        return Xr * Z 
    
    def direct_Y(self, L,G,P,K,S):
        Xb=self.params['aB']*L**2.
        Xf=self.params['aF']*G**2.
        Xr=self.params['aR']*S**2.  
        X=Xb+Xf+Xr
        
        expP=2./5.
        expK=2./5.
        if 'KproptoP' in self.specs:
#             expP=4./5.
#             expK=0.
            K = P*self.iniDynVar['K']/(self.iniDynVar['P'])
        if 'NproptoP' in self.specs:
            expP-=1./5.   # gives in combination expP=3./5
        Z=self.Z(P, K, X, expP, expK)           
        B=self.B(Xb, Z)
        F=self.F(Xf, Z)
        R=self.R(Xr, Z)
        return self.Y(B, F, R)
    
    def direct_W(self,L,G,P,K,S):
        Y=self.direct_Y(L, G, P, K, S)
        return self.W(Y, P, L)
    
    def get_Aini(self,Lini, Gini):
        return (self.params['Cstar']-Lini-Gini)/(1.+self.params['m'])
        
    def get_Tini(self,Aini):
        return Aini/self.params['Sigma']
    
    def prepare_action_set(self, state):
        this_state_action_set=[]
        L, A, G, T, P, K, S= state
        for idx in range(len(self.action_space)):
            self._adjust_parameters(idx)
            W=self.direct_W(L, G, P, K, S)
            if W > self.W_PB:
                this_state_action_set.append(idx)
        return this_state_action_set
    def _adjust_parameters(self, action_number=0):
        """
        This function is needed to adjust the parameter set for the chosen management option.
        Here the action numbers are really transformed to parameter lists, according to the chosen 
        management option.
        Parameters:
            -action: Number of the action in the actionset.
             Can be transformed into: 'default', 'subsidy' 'carbon tax' 'Nature Protection ' or possible combinations
        """
        if action_number < len(self.action_space):
            action=self.action_space[action_number]
        else:
            print("ERROR! Management option is not available!" + str (action))
            print(get_linenumber())
            sys.exit(1)
        # subsidy 
        if action[0]:
            self.params['aR']=self.aR_default*(1+self.ren_sub) 
        else:
            self.params['aR']=self.aR_default
        # carbon tax
        if action[1]:
            self.params['aB']=self.aB_default*(1-self.carbon_tax)
            self.params['aF']=self.aF_default*(1-self.carbon_tax)
        else:
            self.params['aB']=self.aB_default
            self.params['aF']=self.aF_default          
        # nature protection
        if action[2]:
            self.Lprot=True
        else:
            self.Lprot=False
    
    """
    This functions are needed to define a final state and to cluster to Green or brown FP
    """
    def _inside_planetary_boundaries(self):
        L,A,G,T,P,K,S = self.state
        Leff=L
        if self.Lprot:
            Leff=max(L-self.L0, 0)
        W=self.direct_W(Leff, G, P, K, S)
        
        is_inside = True
        if A > self.A_PB or W < self.W_PB or P<self.P_PB:
            is_inside = False
            #print("Outside PB!")
        return is_inside
    
        
    """
    This functions are specific to sustainable management parameters, to decide whether we are inside/outside of planetary boundaries
    and whether the game is finished or not!
    """
    def current_state_region_StartPoint(self):
        
        self.state=np.ones(7)
        self._adjust_parameters(0)
        while not self._inside_planetary_boundaries(): 
            #self.state=self.current_state + np.random.uniform(low=-limit_start, high=limit_start, size=3)
            lower_limit=-.1
            upper_limit=.1
            rnd= np.random.uniform(low=lower_limit, high=upper_limit, size=(len(self.state),))
            self.state[0] = self.current_state[0] + 1e3*rnd[0]  #L
            self.state[1] = self.current_state[1] + 1e3*rnd[1]  #A
            self.state[2] = self.current_state[2] + 1e3*rnd[2]  #G
            
            self.state[3] = self.get_Tini(self.state[1])        # T
            self.state[4] = self.current_state[4] + 1e9*rnd[4]  # P
            self.state[5] = self.current_state[5] + 1e13*rnd[5] # K 
            self.state[6] = self.current_state[6] + 1e11*rnd[6] # S
            
        return self.state

        
    def  _arrived_at_final_state(self):
        L,A,G,T,P,K,S=self.state
        # Attention that we do not break up to early since even at large W it is still possible that A_PB is violated!
        if self.A_PB - A > 0 and self.direct_W(L, G, P, K, S) > 2.e6 and P > 1e10 and self.t>400:
            #print("end", self.t)
            return True
        else:
            return False
              
    def _good_final_state(self):
        L,A,G,T,P,K,S=self.state
        # Good Final State. TODO find a reasonable explanation for this values (maybe something like carbon budget...)!
        if self.A_PB - A > 60 and self.direct_W(L, G, P, K, S) > 2.8e6 and P > 1e10  :
            #print('Success!')
            return True
        else:
            return False
    
    def _which_final_state(self):
        l,a,g,t,p,k,s=self.state
        if self._inside_planetary_boundaries():
            #print("ARRIVED AT GREEN FINAL STATE WITHOUT VIOLATING PB!")
            return Basins.GREEN_FP
        else:
            return Basins.OUT_PB
    
    def get_plot_state_list(self):
        trafo_state=compactification(self.state, self.current_state)

        return trafo_state.tolist()
    
    def observed_states(self):
        return self.dimensions
    
    
    """
    This functions are only needed for visual reasons to get a better feeling for the environment dynamics and the learned trajectories. 
    However, this function do not influence the dynamics of the environment.
    """ 
    def plot_run(self,learning_progress, file_path):
        
        fig, ax3d=self.create_3D_figure()
        start_state=learning_progress[0][0]
        t=0
        for state_action in learning_progress:
            state=inv_compactification(state_action[0], self.current_state)
            action=state_action[1]
            self._adjust_parameters(action)
            
            traj_one_step=odeint(self.dDynVar, state, self.sim_time_step )
            L=traj_one_step[:,0]
            Leff=L
            for t in range(len(L)):
                if self.Lprot:
                    Leff[t]=max(L[t]-self.L0, 0)
            W_one_step=self.direct_W(Leff, traj_one_step[:,2], traj_one_step[:,4],traj_one_step[:,5], traj_one_step[:,6])
            # Plot trajectory
            my_color=self.color_list[action]
            #Plot only APS
            A=compactification(x=traj_one_step[:,1], x_mid=self.Aini)
            W=compactification(W_one_step, self.Wini)
            P=compactification(traj_one_step[:,4], self.Pini)
            ax3d.plot3D(xs=A, ys=W,  zs=P,
                            color=my_color, alpha=1., lw=3)
        
        # Plot from startpoint only one management option to see if green fix point is easy to reach:
        #self.plot_current_state_trajectories(ax3d)
        
        final_state=self._which_final_state().name
        save_path = (file_path +'/DQN_Path/'+ final_state +'/'+ 
                     str (self.run_number) + '_' +  '3D_Path'+  str(start_state) + '.pdf' )
        
        self.save_traj(ax3d, save_path)
        #fig.clean()
        
    
    def create_3D_figure(self):
        fig3d = plt.figure(figsize=(16,9))
        ax3d = plt3d.Axes3D(fig3d)

        ylim=1e15
        zlim=1e13
        font_size=16
        ax3d.set_xlabel("\n\n atmospheric carbon \nstock A [GtC]", size=font_size)
        ax3d.set_ylabel("\n\n Wellbeing W \n  [%1.0e USD/H yr]"%self.W_scale , size=font_size)
        ax3d.set_zlabel("\n population P [%1.0e H]"%self.P_scale, size=font_size)
        self._make_3d_ticks(ax3d)
        ax3d.view_init(90, 180)

        # plot A-boundary
        A_PB=compactification( self.A_PB, self.Aini)
        A_min=0
        
        W_PB=compactification( self.W_PB, self.Wini)
        W_max=1
        
        P_min=compactification( self.P_PB, self.Pini)
        P_max=1
        ax3d.set_xlim(0,compactification(self.params['Cstar'], self.Aini) )        
        ax3d.set_ylim(compactification(3e3, self.Wini),1)
        ax3d.set_zlim(compactification(self.P_PB, self.Pini), 1)
        
        corner_points=[[
                [A_PB , W_PB , P_min],
                [A_PB , W_max, P_min],
                [A_PB , W_max, P_max],
                [A_PB , W_PB , P_max],
                ],
                [
                [A_PB , W_PB , P_max],
                [A_min, W_PB, P_max],
                [A_min, W_PB, P_min],
                [A_PB , W_PB , P_min],
            ]]
        
        boundary_surface_PB = plt3d.art3d.Poly3DCollection(corner_points, alpha=0.25)
        boundary_surface_PB.set_color("gray")
        boundary_surface_PB.set_edgecolor("gray")
        ax3d.add_collection3d(boundary_surface_PB)
        
        # Plot Startpoint A Y P
        L,A,G,T,P,K,S = self.current_state
        for idx, legend_color in enumerate(self.color_list):
            ax3d.scatter(compactification(A,self.Aini), compactification(self.direct_W(L, G, P, K, S), self.Wini),compactification(P,self.Pini) ,
                         color=legend_color, label=self.management_options[idx])

        self.plot_current_state_trajectories(ax3d)
        ax3d.grid(False)
        return fig3d, ax3d
    
    def plot_random_trajectories_in_phase_space(self, ax3d, action):
        colortop = "green"
        colorbottom = "black"
        def random_start_points():
        
            self.state=np.ones(7)
            #self.state=self.current_state + np.random.uniform(low=-limit_start, high=limit_start, size=3)
            lower_limit=-1
            upper_limit=1
            rnd= np.random.uniform(low=lower_limit, high=upper_limit, size=(len(self.state),))
            self.state[0] = self.current_state[0] + 6e2*rnd[0]  #L
            self.state[1] = self.current_state[1] + 6e2*rnd[2]  #A
            self.state[2] = self.current_state[2] + 6e2*rnd[1]         #G
            self.state[3] = self.get_Tini(self.state[1])        # T
   
            self.state[4] = self.current_state[4]*np.abs(rnd[4]) #P
            self.state[5] = self.current_state[5]*np.abs(rnd[5]) #K
            self.state[6] = self.current_state[6]* np.abs(rnd[6]) #S
            return self.state
            
        ########################################
        # prepare the integration
        ########################################
        steps=500
        time = np.linspace(0, steps*self.dt, steps)
        num=300
        c_lagtpks=[]
        self._adjust_parameters(action)
        for i in range(num):
            c_lagtpks.append(random_start_points())

        for i in range(num):
            x0 = c_lagtpks[i]
            traj_one_step=odeint(self.dDynVar, x0, time)
            L=traj_one_step[:,0]
            for t in range(len(L)):
                if self.Lprot:
                    L[t]=max(L[t]-self.L0,0)
            W_one_step=self.direct_W(L, traj_one_step[:,2], traj_one_step[:,4],traj_one_step[:,5], traj_one_step[:,6])
            
            A=compactification(x=traj_one_step[:,1], x_mid=self.Aini)
            W=compactification(W_one_step, self.Wini)
            P=compactification(traj_one_step[:,4], self.Pini)
            

            
            ax3d.plot3D(xs=A, ys=W,  zs=P,
                            color=self.color_list[action], alpha=.3, lw=1)
            

    
     
    def plot_current_state_trajectories(self, ax3d):
        # Trajectories for the current state with all possible management options
        steps=1000
        time = np.linspace(0, steps*self.dt, steps)

        for action_number in range(len(self.action_space)):
            self._adjust_parameters(action_number)
            my_color=self.color_list[action_number]
            traj_one_step=odeint(self.dDynVar, self.current_state, time)
            L=traj_one_step[:,0]
            for t in range(len(L)):
                if self.Lprot:
                    L[t]=max(L[t]-self.L0,0)
            W_one_step=self.direct_W(L, traj_one_step[:,2], traj_one_step[:,4],traj_one_step[:,5], traj_one_step[:,6])
            # Plot trajectory
            my_color=self.color_list[action_number]
            #Plot only AWP
            A=compactification(x=traj_one_step[:,1], x_mid=self.Aini)
            W=compactification(W_one_step, self.Wini)
            P=compactification(traj_one_step[:,4], self.Pini)
            ax3d.plot3D(xs=A, ys=W,  zs=P, 
                            color=my_color, alpha=.8, lw=1, )
            ax3d.scatter(xs=A, ys=W,  zs=P, 
                            color=my_color, alpha=.8, s=1.0)
            
    
    def plot_2D(self, learning_progress, file_path, plot_example=False):
        start_state=learning_progress[0][0]
        states=np.array(learning_progress)[:,0]
        actions=np.array(learning_progress)[:,1]

        L=inv_compactification( np.array(list(zip(*states))[0]), self.iniDynVar['L'])
        A=inv_compactification( np.array(list(zip(*states))[1]), self.iniDynVar['A'])
        G=inv_compactification( np.array(list(zip(*states))[2]), self.iniDynVar['G'])
        T=inv_compactification( np.array(list(zip(*states))[3]), self.iniDynVar['T'])
        P=inv_compactification( np.array(list(zip(*states))[4]), self.iniDynVar['P'])
        K=inv_compactification( np.array(list(zip(*states))[5]), self.iniDynVar['K'])
        S=inv_compactification( np.array(list(zip(*states))[6]), self.iniDynVar['S'])
        
        Leff=np.ones(len(L))
        for t in range(len(states)):
            if self.action_space[actions[t]][2]:
                Leff[t]=max(L[t]-self.L0, 0)
            else:
                Leff[t]=L[t]
        
        Xb=self.params['aB']*Leff**2.
        Xf=self.params['aF']*G**2.
        Xr=self.params['aR']*S**2.  
        X=Xb+Xf+Xr
        
        expP=2./5.
        expK=2./5.
        if 'KproptoP' in self.specs:
#             expP=4./5.
#             expK=0.
            K = P*self.iniDynVar['K']/(self.iniDynVar['P'])
        if 'NproptoP' in self.specs:
            expP-=1./5.   # gives in combination expP=3./5
        Z=self.Z(P, K, X, expP, expK)
        
        
        B=self.B(Xb, Z)
        F=self.F(Xf, Z)
        R=self.R(Xr, Z)
        Y=self.Y(B, F, R)
        W=self.W(Y, P, L)
        
        t = self.dt*np.arange(len(L))
                
        fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(20,14))
        ax1=ax[0][0]
        ax2=ax[0][1]
        ax3=ax[1][0]
        ax4=ax[1][1]
        ax5=ax[2][0]
        ax6=ax[2][1]

        self.plot_carbon_cycle(fig, ax1, t, L, A, G, [], T)
        self.plot_energy_consumption(fig, ax2, t,B,F,R,S)
        self.plot_population_dynamics(fig, ax3, t, P, W)
        self.plot_capital_dynamics(fig, ax4, t, K, Y)
        self.plot_2D_actions(ax5, t,actions)
        
        fig.tight_layout()
        
        final_state=self._which_final_state().name
        if plot_example:
            save_path=file_path
        else:
            save_path = (file_path +'/DQN_Path/'+ final_state +'/'+ 
                     str (self.run_number) + '_' +  '2D_time_series'+  str(start_state) + '.pdf' )
            
        plt.savefig(save_path)    
        if self.plot_progress:
            plt.show()
        plt.close()
        print('Saved as figure in path:' + save_path)

    def plot_2D_actions(self, ax,time, actions):
        ax.plot(time, actions , ".")
        ax.set_title('Action choice')
        ax.set_xlabel('t [years]')
        ax.set_ylabel('Chosen Action')
        ax.set_ylim(0,7.2)
        ax.set_yticks(np.arange(8))
        box_text=''
        for i in range(len(self.management_options)):
            box_text+=str(i ) + ": " + self.management_options[i] +"\n"
        
        at = AnchoredText(box_text, prop=dict(size=14), frameon=True, 
                          loc='lower left', bbox_to_anchor=(1.0, .02),bbox_transform=ax.transAxes
                      )
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        #ax.axis('off')
        ax.add_artist(at)    
        
    def plot_carbon_cycle(self,fig, ax, t, L, A, G, M, T):
        Cstar=5500
        Sigma = 1.5 * 1e8
        #Sigma = 1
    
        alpha=0.5
        
        #ax.set_title('Carbon cyle')
        ax.set_xlabel('t [years]')
        ax.set_ylabel('L, A, G , M , T [GtC]')
        ax.set_ylim(0,Cstar)
        ax.fill_between(t, G+L+A, Cstar, color='b', alpha=alpha)
        pltM=ax.plot( [], [], lw=10, color='b', alpha=alpha, label='$M$')
    
        ax.fill_between(t, G, G+ L , color='g', alpha=alpha)
        pltL=ax.plot( [], [], lw=10, color='g', alpha=alpha,  label='$L$')
    
        ax.fill_between(t, G+L,G+L+A, color='c', alpha=alpha)
        A_PB=950
        plot_PB=ax.plot(t, G+L+A_PB, lw=2, ls=':', color='red',label='$A_{PB}$' )
        pltA=ax.plot( [], [], lw=10, color='c', alpha=alpha, label='$A$')  
     
        ax.fill_between(t, 0, G, color='gray', alpha=alpha)
        pltG=ax.plot( [], [], lw=10, color='gray', alpha=alpha, label='$G$')    
        
        pltT=ax.plot( t, T*Sigma, lw=2, ls='-', color='r', label='$T \\cdot \\Sigma $')
    
        plts = pltM+pltA+plot_PB + pltL+pltG + pltT
        #labs = [l.get_label() for l in plts]
        #ax.legend(plts, labs, loc=2, fontsize=16, bbox_to_anchor=(1.,1.0), fancybox=True)
        return plts

    def plot_energy_consumption(self, fig, ax ,t, B,F,R,S):
        ax.set_title('Energy shares')
        ax.set_xlabel('t [years]')
        #ax2=ax.twinx()    
    #     ax.set_ylabel('Absolute Energy Flow [GJ/year]')
        ax.set_ylabel('Relative Energy Shares [%]')
        
        e_B=e_F=4*1e10
        E_B=e_B*B
        E_F=e_F*F
        E=E_B+ E_F + R
        eps=1e-9
        #energy shares
        pltB=ax.plot( t, (E_B/E), lw=3, ls='--', color='g', label='$E_B/E$')
        #pltB=ax.plot( t, (B*e_B), lw=3, ls='--', color='g', label='$E_B$')
    
        plts = pltB
    
        if (np.sum(F) >eps):
            pltF=ax.plot( t, (E_F/(E)), lw=3, ls='--', color='gray', label='$E_F/E$')
            #pltF=ax.plot( t, (F*e_F), lw=3, ls='--', color='gray', label='$E_F$')
    
            plts+=pltF
        if (np.sum(R) >eps):
            pltR=ax.plot( t, (R/(E)), lw=3, ls='--', color='orange', label='$R/E$')
            #pltR=ax.plot( t, R, lw=3, ls='--', color='orange', label='$R$')
            ax2=ax.twinx()
            ax2.set_yscale('log')
            ax2.set_ylabel('Renewable Knowledge Stock S [GJ]')
            pltS=ax2.plot( t, S, lw=3, color='orange', label='$S$')
            plts+=pltR+pltS
    
            
            
        labs = [l.get_label() for l in plts]
        ax.legend(plts, labs, loc=1, fontsize=16, bbox_to_anchor=(1.,1.0), fancybox=True)
    
    
    def plot_population_dynamics(self,fig, ax,t, P, W):
        ax.set_title('Population dynamics')
    
        ax.set_xlabel('t [years]')
        ax.set_ylabel('Population P [H]')
        ax.set_yscale('log')
        ax2=ax.twinx()    
        ax2.set_ylabel('Wellbeing W [\$/(H $ \\cdot $ year)]')
        ax2.set_yscale('log')
    
        pltP=ax2.plot( t, P, lw=3, color='y', label='$P$')
        pltW=ax2.plot( t, W, lw=3, ls='--', color='y', label='$W$')
        
        pltW_PB=ax2.plot(t, np.ones(len(t))*self.W_PB, lw=1, color='red', label='$W_{PB}$')
        
        plts = pltP+pltW+pltW_PB
        labs = [l.get_label() for l in plts]
        ax.legend(plts, labs, loc=1, fontsize=16, bbox_to_anchor=(1.,1.0), fancybox=True)
    
    
    def plot_capital_dynamics(self,fig, ax, t, K, Y):
        ax.set_title('Capital dynamics')
    
        ax.set_xlabel('t [years]')
        ax.set_ylabel('Capital K [$]')
        ax.set_yscale('log')
        ax2=ax.twinx()    
        ax2.set_ylabel('Economic Production Y [\$/year]')
        ax2.set_yscale('log')
        pltK=ax2.plot( t, K, lw=3, color='b', label='$K$')
        pltY=ax2.plot( t, Y, lw=3, ls='--', color='magenta', label='$Y$')
        
        Y_PB=self.Y_PB
        pltY_PB=ax2.plot(t, np.ones(len(t))*Y_PB, lw=1, color='red', label='$Y_{PB}$')
        plts = pltK+pltY+pltY_PB
        labs = [l.get_label() for l in plts]
        ax.legend(plts, labs, loc=1, fontsize=16, bbox_to_anchor=(1.,1.0), fancybox=True)
    
    def plot_2D_default_trajectory(self, action_number, file_path):
        time_end=101
        time = np.linspace(0, time_end*self.dt, time_end)
        self._adjust_parameters(action_number)
        traj=odeint(self.dDynVar, self.current_state, time)
        action=self.action_space[action_number]
        
        L = traj[:,0]
        A = traj[:,1]
        G = traj[:,2]
        T = traj[:,3]
        P = traj[:,4]
        K = traj[:,5]
        S = traj[:,6]
        
        M=self.M(L, A, G)        
        
        Leff=np.ones(len(L))
        for t in range(len(L)):
            if action[2] :
                Leff[t]=max(L[t]-self.L0, 0)
            else:
                Leff[t]=L[t]
        Xb=self.params['aB']*Leff**2.
        Xf=self.params['aF']*G**2.
        Xr=self.params['aR']*S**2.  
        X=Xb+Xf+Xr
        
        expP=2./5.
        expK=2./5.
        if 'KproptoP' in self.specs:
#             expP=4./5.
#             expK=0.
            K = P*self.iniDynVar['K']/(self.iniDynVar['P'])
        if 'NproptoP' in self.specs:
            expP-=1./5.   # gives in combination expP=3./5
        Z=self.Z(P, K, X, expP, expK)
        
        
        B=self.B(Xb, Z)
        F=self.F(Xf, Z)
        R=self.R(Xr, Z)
        Y=self.Y(B, F, R)
        W=self.W(Y, P, L)
        
        P_comp=compactification(P, self.iniDynVar['P'])
        K_comp=compactification(K, self.iniDynVar['K'])
        S_comp=compactification(S, self.iniDynVar['S'])
        Y_comp=compactification(Y, self.Yini)
        W_comp=compactification(W, self.Wini)
        
        actions=np.ones(len(L))*action_number
        t = self.dt*np.arange(len(L))

        from matplotlib import gridspec
        
        
        fig = plt.figure(figsize=(10, 6)) 
        ax1=plt.subplot()
        
#         gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1]) 
        
#         ax1 = plt.subplot(gs[0])
#         ax2 = plt.subplot(gs[1])
        plot_time=len(A)-1
        if plot_time >300:
            plot_time=100
        
        ax1.set_xlim(0,plot_time)
        
        self.plot_all_dynamics(fig, ax1, L, A, G, T, P_comp, K_comp, S_comp, Y_comp, W_comp, B, F, R)
        
#         ax2.set_xlim(0,plot_time)
#         self.plot_2D_actions(ax2, t,actions)     

        fig.tight_layout()
        plt.savefig(file_path+'/2Dtrajectories_' + self.management_options[action_number] +'.pdf')
        #plt.show()
        
        text_path = (file_path +'/2Dtrajectories_' + self.management_options[action_number]+'.txt' )
        with open(text_path, 'w') as f:
            f.write("# L \t A \t G \t T \t P \t K \t S \t M \t Y \t  W \t  B \t F \t   R \n")
            for i in range(len(L)):
                f.write("%s  %s  %s  %s  %s  %s  %s  %s  %s  %s  %s  %s  %s \n" % 
                        (L[i] ,A[i], G[i], T[i] , P[i], K[i], S[i],
                         M[i], Y[i], W[i], B[i], F[i], R[i]))
        f.close()
   
    
    def plot_all_dynamics(self, fig, ax, L,A,G,T,P,K,S, Y, W, B, F, R):
        
        t = self.dt*np.arange(len(L))
        
        plts=self.plot_carbon_cycle(fig, ax, t, L, A, G, [], T)
        ax1_2=ax.twinx()
        ax1_2.set_ylim(0,1)

        #energy shares
        e_B=e_F=4*1e10
        E_B=e_B*B
        E_F=e_F*F
        E=E_B+ E_F + R
        eps=1e-9
        
        pltB=ax1_2.plot( t, (E_B/E), lw=2, ls='--', color='g', label='$E_B/E$')
        #pltB=ax.plot( t, (B*e_B), lw=3, ls='--', color='g', label='$E_B$')
    
        plts += pltB
    
        if (np.sum(F) >eps):
            pltF=ax1_2.plot( t, (E_F/(E)), lw=2, ls='--', color='gray', label='$E_F/E$')
            #pltF=ax.plot( t, (F*e_F), lw=3, ls='--', color='gray', label='$E_F$')
    
            plts+=pltF
        if (np.sum(R) >eps):
            pltR=ax1_2.plot( t, (R/(E)), lw=2, ls='--', color='orange', label='$R/E$')
            # ax1_2.set_ylabel('Renewable Knowledge Stock S [GJ]')
            pltS=ax1_2.plot( t, S, lw=2, color='orange', label='$S/(S+S_{mid})$')
            plts+=pltR+pltS
    
        
        pltP=ax1_2.plot( t, P, lw=2, color='y', label='$P/(P+P_{mid})$')
        pltW=ax1_2.plot( t, W, lw=2, ls='--', color='brown', label='$W/(W+W_{mid})$')   
        
        W_PB_comp=self.compact_PB[1]
        pltW_PB=ax1_2.plot(t, np.ones(len(t))*W_PB_comp, lw=2, color='brown', ls=':',label='$W_{PB}$')

        pltK=ax1_2.plot( t, K, lw=2, color='black', label='$K/(K+K_{mid})$')
        pltY=ax1_2.plot( t, Y, lw=2, ls='--', color='magenta', label='$Y/(Y+Y_{mid})$')
        
        plts += pltP+pltW + pltW_PB+pltK + pltY
        labs = [l.get_label() for l in plts]

        ax1_2.set_ylabel(r'Rel. shares [%] / Scaled variables: $x\rightarrow x/(x+x_{mid})$')
              
        ax.legend(plts, labs, loc='lower left', fontsize=13.5, bbox_to_anchor=(1.1,.01), fancybox=True)
        
    
    def plot_trajectories_from_data(self, time, L_comp, A_comp, G_comp, T_comp, P_comp, K_comp, S_comp, actions,file_path ):
        
        M=np.zeros(len(L_comp)); Y=np.zeros(len(L_comp)); W=np.zeros(len(L_comp)); B=np.zeros(len(L_comp)); F=np.zeros(len(L_comp)); R=np.zeros(len(L_comp))
         
        L=inv_compactification( np.array(L_comp), self.iniDynVar['L'])
        A=inv_compactification( np.array(A_comp), self.iniDynVar['A'])
        G=inv_compactification( np.array(G_comp), self.iniDynVar['G'])
        T=inv_compactification( np.array(T_comp), self.iniDynVar['T'])
        P=inv_compactification( np.array(P_comp), self.iniDynVar['P'])
        K=inv_compactification( np.array(K_comp), self.iniDynVar['K'])
        S=inv_compactification( np.array(S_comp), self.iniDynVar['S'])
        
        # Account for action that was chosen at certain step for derived variables
        for t in range(len(L)):
            action=actions[t]
            self._adjust_parameters(action)
            if self.Lprot:
                Leff=max(L[t]-self.L0, 0)
            else:
                Leff=L[t]
            
        
            Xb=self.params['aB']*Leff**2.
            Xf=self.params['aF']*G[t]**2.
            Xr=self.params['aR']*S[t]**2.  
            X=Xb+Xf+Xr
        
            expP=2./5.
            expK=2./5.
            if 'KproptoP' in self.specs:
    #             expP=4./5.
    #             expK=0.
                K = P[t]*self.iniDynVar['K']/(self.iniDynVar['P'])
            if 'NproptoP' in self.specs:
                expP-=1./5.   # gives in combination expP=3./5
            Z=self.Z(P[t], K[t], X, expP, expK)
        
            
            B[t]=self.B(Xb, Z)
            F[t]=self.F(Xf, Z)
            R[t]=self.R(Xr, Z)
            Y[t]=self.Y(B[t], F[t], R[t])
            W[t]=self.W(Y[t], P[t], L[t])
            
        Y_comp=compactification(Y, self.Yini)
        W_comp=compactification(W, self.Wini)
        
        t = self.dt*np.arange(len(L))
        
#         fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10,10))
#         ax1=ax[0]
#         ax2=ax[1]
        from matplotlib import gridspec
        
        fig = plt.figure(figsize=(10, 9)) 
        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1]) 
        
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])
        plot_time=len(A)-1
        if plot_time >300:
            plot_time=100
        
        ax1.set_xlim(0,plot_time)
        ax2.set_xlim(0,plot_time)
        
        
        self.plot_all_dynamics(fig, ax1, L, A, G, T, P_comp, K_comp, S_comp, Y_comp, W_comp, B, F, R)
        
        self.plot_2D_actions(ax2, t,actions)        
        
        fig.tight_layout()
        fig.savefig(file_path)
        
    
        
    def plot_carbon_cycle_lines(self,fig, ax, t, L, A, G, M, T):
        Cstar=5500
        Sigma = 1.5 * 1e8
        #Sigma = 1
    
        alpha=0.5
        
        ax.set_title('Carbon cyle')
        ax.set_xlabel('t [years]')
        ax.set_ylabel('carbon stock L, A, G , M[GtC]')
        
        pltM=ax.plot( t, M, lw=1, alpha=0.5, color='b', label='$M$')

        pltL=ax.plot( t, L, lw=1, alpha=0.5, color='g', label='$L$')

        A_PB=950
        plot_PB=ax.plot(t, A_PB*np.ones(len(t)), lw=1, color='red',label='$A_{PB}$' )
        pltA=ax.plot( t, L, lw=1, alpha=0.5, color='c', label='$A$')  
     
        ax.fill_between(t, 0, G, color='gray', alpha=alpha)
        pltG=ax.plot( [], [], lw=10, color='gray', alpha=alpha, label='$G$')    
        
        pltT=ax.plot( t, T*Sigma, lw=5, ls='--', color='r', label='$T \\cdot \\Sigma $')
    
        plts = pltM+pltA+plot_PB + pltL+pltG + pltT
        labs = [l.get_label() for l in plts]
        ax.legend(plts, labs, loc=1, fontsize=16, bbox_to_anchor=(1.,1.0), fancybox=True)


    def _make_3d_ticks(self, ax3d, boundaries = None,transformed_formatters=False,num_a = 12, num_y = 12, num_p = 12,):
        if boundaries is None:
            boundaries = [None]*3
        
        transf = ft.partial(compactification, x_mid=self.Aini)
        inv_transf = ft.partial(inv_compactification, x_mid=self.Aini)

        if boundaries[0] is None:
            start, stop = 0, self.params['Cstar'] - self.iniDynVar['G']
        else:
            start, stop = inv_transf(boundaries[0])
        formatters, locators = self.transformed_space(transf, inv_transf, axis_use=True, start=start, stop=stop, num=num_a)
        if transformed_formatters:
            new_formatters = []
            for el, loc in zip(formatters, locators):
                if el:
                    new_formatters.append("{:4.2f}".format(loc))
                else:
                    new_formatters.append(el)
            formatters = new_formatters
        #print(locators, formatters)
        ax3d.w_xaxis.set_major_locator(ticker.FixedLocator(locators))
        ax3d.w_xaxis.set_major_formatter(ticker.FixedFormatter(formatters))
        
        # W - ticks
        transf = ft.partial(compactification, x_mid=self.Wini)
        inv_transf = ft.partial(inv_compactification, x_mid=self.Wini)
    
        if boundaries[1] is None:
            start, stop = 4e3, np.infty
        else:
            start, stop = inv_transf(boundaries[1])
        formatters, locators = self.transformed_space(transf, inv_transf, axis_use=True, scale=self.W_scale, start=start, stop=stop, num=num_y)
        if transformed_formatters:
            new_formatters = []
            for el, loc in zip(formatters, locators):
                if el:
                    new_formatters.append("{:4.2f}".format(loc))
                else:
                    new_formatters.append(el)
            formatters = new_formatters
        ax3d.w_yaxis.set_major_locator(ticker.FixedLocator(locators))
        ax3d.w_yaxis.set_major_formatter(ticker.FixedFormatter(formatters))
        
        
        transf = ft.partial(compactification, x_mid=self.Pini)
        inv_transf = ft.partial(inv_compactification, x_mid=self.Pini)
    
        if boundaries[2] is None:
            start, stop = 1e6, np.infty
        else:
            start, stop = inv_transf(boundaries[2])
        formatters, locators = self.transformed_space(transf, inv_transf, axis_use=True, scale=self.P_scale, start=start, stop=stop, num=num_p)
        if transformed_formatters:
            new_formatters = []
            for el, loc in zip(formatters, locators):
                if el:
                    new_formatters.append("{:4.2f}".format(loc))
                else:
                    new_formatters.append(el)
            formatters = new_formatters
        ax3d.w_zaxis.set_major_locator(ticker.FixedLocator(locators))
        ax3d.w_zaxis.set_major_formatter(ticker.FixedFormatter(formatters))
        
    
    def transformed_space(self, transform, inv_transform,
                      start=0, stop=np.infty, num=20,
                      scale=1,
                      num_minors = 12,
                      endpoint=True,
                      axis_use=False,
                      boundaries=None,
                      minors=False):
        add_infty = False
        if stop == np.infty and endpoint:
            add_infty = True
            endpoint = False
            num -= 1
    
        locators_start = transform(start)
        locators_stop = transform(stop)
    
        major_locators = np.linspace(locators_start,
                               locators_stop,
                               num,
                               endpoint=endpoint)
    
        major_formatters = inv_transform(major_locators)
        # major_formatters = major_formatters / scale
    
        major_combined = list(zip(major_locators, major_formatters))
        # print(major_combined)
    
        if minors:
            _minor_formatters = np.linspace(major_formatters[0], major_formatters[-1], num_minors, endpoint=False)[1:]
            minor_locators = transform(_minor_formatters)
            minor_formatters = [np.nan] * len(minor_locators)
            minor_combined = list(zip(minor_locators, minor_formatters))
            # print(minor_combined)
        else:
            minor_combined=[]
        
        combined = list(hq.merge(minor_combined, major_combined, key = op.itemgetter(0)))
    
        # print(combined)
    
        if not boundaries is None:
            combined = [(l, f) for l, f in combined if boundaries[0] <= l <= boundaries[1] ]
    
        ret = tuple(map(np.array, zip(*combined)))
        if ret:
            locators, formatters = ret
        else:
            locators, formatters = np.empty((0,)), np.empty((0,))
        formatters = formatters / scale
    
        if add_infty:
            # assume locators_stop has the transformed value for infinity already
            locators = np.concatenate((locators, [locators_stop]))
            formatters = np.concatenate(( formatters, [ np.infty ]))
    
        if not axis_use:
            return formatters
    
        else:
            string_formatters = np.zeros_like(formatters, dtype="|U10")
            mask_nan = np.isnan(formatters)
            if add_infty:
                string_formatters[-1] = INFTY_SIGN
                mask_nan[-1] = True
            string_formatters[~mask_nan] = np.round(formatters[~mask_nan], decimals=2).astype(int).astype("|U10")
            return string_formatters, locators



    def save_traj_final_state(self, learners_path, file_path,episode ):
        final_state=self._which_final_state().name
        
        states=np.array(learners_path)[:,0]
        start_state=states[0]
        L=np.array(list(zip(*states))[0])
        A=np.array(list(zip(*states))[1])
        G=np.array(list(zip(*states))[2])
        T=np.array(list(zip(*states))[3])
        P=np.array(list(zip(*states))[4])
        K=np.array(list(zip(*states))[5])
        S=np.array(list(zip(*states))[6])
        time=np.arange(len(L))*self.dt
        actions=np.array(learners_path)[:,1]
        rewards=np.array(learners_path)[:,2]
        
        text_path = (file_path +'/DQN_Path/'+ final_state +'/'+ 
                     str (self.run_number) + '_' +  'path_'+  str(start_state)+ '_episode'+str(episode)  + '.txt' )
        
        with open(text_path, 'w') as f:
            f.write("# time \t L \t A \t G \t T \t P \t K \t S \t   Action \t  Reward \n")
            for i in range(len(learners_path)):
                f.write("%d %s  %s  %s  %s  %s  %s  %s  %s  %s  \n" % 
                        (time[i], L[i] ,A[i], G[i], T[i] , P[i], K[i], S[i],
                         actions[i], rewards[i]))
        print('Saved :' + text_path)
        
    
      
    def save_traj(self,ax3d, fn):
        ax3d.legend(loc='best', prop={'size': 12})
        plt.savefig(fname=fn)
        if self.plot_progress:
            plt.show()
        plt.close()
        
    def define_test_points(self):
        testpoints=[[2480, 758, 1125, 5.053333333333333e-6, 6e9, 6e13, 5e11 ],
                    [2480, 830, 1125, 5.053333333333333e-6, 6e9, 6e13, 5e11 ]
                     ]
        return testpoints
        
    
    
    
    
