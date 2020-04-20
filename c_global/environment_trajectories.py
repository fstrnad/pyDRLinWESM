#import AYS_Environment as ays_env
import c_global.cG_LAGTPKS_Environment as c_global


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

action=0
#fig3d, ax3d=my_Env.create_3D_figure()
#my_Env.plot_random_trajectories_in_phase_space(ax3d, action)
#my_Env.plot_random_trajectories_in_phase_space(ax3d, 1)
#my_Env.plot_random_trajectories_in_phase_space(ax3d, 2)
#my_Env.plot_random_trajectories_in_phase_space(ax3d,3)
#my_Env.save_traj(ax3d, fn='./images/trajectories' + my_Env.management_options[action] + '.pdf')
#my_Env.save_traj(ax3d, fn='./c_global/images/trajectories_management_options'+ '.pdf')

for action in (range(0,8)):
    my_Env.plot_2D_default_trajectory(action, './images/management/')



