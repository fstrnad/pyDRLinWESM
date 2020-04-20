import sys

from DQNLearner import DQNLearner 
from partially_observable_cG import partially_observable_cG, noisy_partially_observable_LAGTPKS

  
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


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
            ren_sub=1,
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

observables=dict(   L=False,  
                    A=False  ,
                    G=True,
                    T=True,
                    P=True,
                    K=True,
                    S=False,
            )   

dt=1
network_learning_rate= 0.00025
explore_start = 1.0 
explore_stop = 0.01
decay_rate=0.001 
batch_size = 64           # Number of experiences the Memory can keep
alpha = 0.8
gamma=0.96    # 0.98 Belongs exp(-0.02) which is often used for economic models to estimate future discounts 
noise_strength=.05
Boltzmann_prob=False
beta=50

Update_target_frequency=200



learner_type='ddqn' 
prior_exp_replay=True
dueling=True
importance_sampling=True
plot_progress=True
reward_type='PB'



def single_run():
    print('Start single run test!') 
    episodes=6000    
    episode_steps=500
    memory_size = int(1e5)
    max_steps=1000 
    tau_alpha=1e5
    print("Update Frequency: " + str(Update_target_frequency))
    print("Memory size: " + str(memory_size))
    #memoryUsed()
    dirpath='./part_observable_c_global/'
    my_Env=partially_observable_cG(dt=dt,pars=pars, reward_type=reward_type, ics=ics, observables=observables, plot_progress=plot_progress)
    #my_Env=ays_env.AYS_Environment(dt=dt, reward_type=reward_type)
    dqn_agent=DQNLearner(my_Env=my_Env, dt=dt,  episodes=episode_steps, max_steps=max_steps, 
                     alpha=alpha, tau_alpha=tau_alpha,  beta=beta, gamma=gamma, 
                     explore_start = explore_start  , explore_stop = explore_stop,  decay_rate=decay_rate ,
                     Boltzmann_prob=Boltzmann_prob, reward_type=reward_type, Update_target_frequency=Update_target_frequency,
                     learner_type=learner_type, prior_exp_replay=prior_exp_replay, dueling=dueling, importance_sampling=importance_sampling,
                     memory_size = memory_size, batch_size = batch_size , network_learning_rate=network_learning_rate, plot_progress=plot_progress,
                     dirpath=dirpath)
        
    num_runs= int(episodes/episode_steps) +1
    # Here for every test the agent is new initialize      d
    dqn_agent.reset_learner()
    print("Run number: " + str(dqn_agent.run_number))    
    for run in range(num_runs ):
        # Test the agents behavior   
        print(dqn_agent.test_on_current_state(True))
        
        # Here we train the agent with runs episodes, we do this after evaluation to get t  he untrained agent as well.
        dqn_agent.learn()
    
    print(dqn_agent.test_on_current_state(True))

    
single_run()
