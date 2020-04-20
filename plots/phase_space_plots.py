#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
from ays_general import __version__, __version_info__
import ays_model as aws
import ays_general
from pyviability import helper

import numpy as np

import scipy.integrate as integ
import scipy.optimize as opt

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plt3d
import matplotlib.ticker as ticker
from matplotlib import animation

import warnings as warn

import heapq as hq
import operator as op

import argparse, argcomplete

import pickle

import functools as ft

green_fp=[0,1,1]
final_radius=0.1
brown_fp=[0.6,0.4,0]

def good_final_state(state):
    a,y,s=state
    if np.abs(a - green_fp[0]) < final_radius and np.abs(y - green_fp[1]) < final_radius and np.abs(s-green_fp[2])< final_radius:
            return True
    else:
            return False
        
management_options=['default', 'LG' , 'ET','LG+ET'  ]
management_actions=[(False, False), (True, False), (False, True), (True, True)]        
def get_parameters(action_number=0):
        
    """
    This function is needed to return the parameter set for the chosen management option.
    Here the action numbers are really transformed to parameter lists, according to the chosen 
    management option.
    Parameters:
    -action: Number of the action in the actionset.
     Can be transformed into: 'default', 'degrowth' ,'energy-transformation' or both DG and ET at the same time
    """
    # AYS example from Kittel et al. 2017:
    tau_A = 50
    tau_S = 50
    beta = 0.03
    beta_LG = 0.015
    eps = 147
    theta = beta /(350)    # beta / ( 950 - A_offset(=600) )
    rho = 2.
    sigma = 4e12
    sigma_ET = sigma*0.5**(1/rho)
    phi = 4.7e10
    
    AYS0 = [240, 7e13, 5e11]
    
    APB = 345
    YSF = 4e13 
    
    
    if action_number < len(management_actions):
        action=management_actions[action_number]
    else:
        print("ERROR! Management option is not available!" + str (action))

    
    parameter_list=[(beta_LG if action[0] else beta  ,
                 eps, phi, rho, 
                 sigma_ET if action[1] else sigma, 
                 tau_A, tau_S, theta)]
    
    return parameter_list 
        
def plot_phase_space(dynamic):
    
    save_path='./images/phase_space_plots/phase_space_' + dynamic + '.pdf'
    num = 400
    shift_axis=(2400, 1e14, 1e12)
    aws_0 = np.random.rand(num, 3)
    #print(aws_0)
    # a small hack to make all the parameters available as global variables
    # aws.globalize_dictionary(aws.AWS_parameters, module=aws)
    aws.globalize_dictionary(aws.grid_parameters, module=aws)
    aws.globalize_dictionary(aws.boundary_parameters, module=aws)
    
    
#     parameter_dict = aws.get_management_parameter_dict(dynamic, aws.AYS_parameters)
#     parameter_list=[]
#     parameter_list.append(helper.get_ordered_parameters(aws._AYS_rhs, parameter_dict))
#     print(parameter_list)
#     
    
    parameter_list=get_parameters(management_options.index(dynamic))
    print(parameter_list)

    
    
    ########################################
    # prepare the integration
    ########################################
    time = np.linspace(0, 300, 1000)
    one_step=np.linspace(0,10,1000)
    #formatters, locators=get_ticks()
    
    
    
    colortop = "green"
    colorbottom = "black"
    
    fig, ax3d = ays_general.create_figure(A_mid=aws.A_mid, W_mid=aws.W_mid, S_mid=aws.S_mid)
    #fig = plt.figure(figsize=(18,8))
    #ax3d = plt3d.Axes3D(fig)
    
    ax3d.view_init(ays_general.ELEVATION_FLOW, ays_general.AZIMUTH_FLOW)
    #ax3d.view_init(elev=89, azim=270)
    S_scale = 1e9
    W_scale = 1e12
    ax3d.set_xlabel("\n\nexcess atmospheric carbon\nstock A [GtC]", size=16)
    ax3d.set_ylabel("\neconomic output Y [%1.0e USD/yr]"%W_scale, size=16)
    ax3d.set_zlabel("\n\nrenewable knowledge\nstock S [%1.0e GJ]"%S_scale, size=16)
    
    x0_test = [.9, 0.5, 0] # a, w, s
    
    # management trajectory with degrowth:
    # Here we get the hairy trajectories that are integrated via odeint
    for i in range(num):
        x0 = aws_0[i]
        traj = integ.odeint(aws.AYS_rescaled_rhs, x0, time, args=parameter_list[0])
        #print(traj[-1])
        
        ax3d.plot3D(xs=traj[:,0], ys=traj[:,1], zs=traj[:,2],
                        color=colorbottom if traj[-1,2]<0.5 else colortop, alpha=.3)
#         ax3d.scatter(*zip(traj[0]),color='grey')
#         ax3d.scatter(traj[-1][0], traj[-1][1], traj[-1][2],
#                      color='green' if good_final_state(traj[-1])else 'red' , alpha=0.5)
        
        
    #print(x0_test)
    traj_one_step=integ.odeint(aws.AYS_rescaled_rhs, x0_test,one_step , args=parameter_list[0])
    #traj_one_step=integ.odeint(aws.AYS_rescaled_rhs, green_fp,one_step , args=parameter_list[0])
    
    ax3d.plot3D(xs=traj_one_step[:,0], ys=traj_one_step[:,1], zs=traj_one_step[:,2],
                        color='red', alpha=.3)       
    
        
    ays_general.add_boundary(ax3d,
                                 sunny_boundaries=["planetary-boundary", "social-foundation"],
                                 **aws.grid_parameters, **aws.boundary_parameters)  
    #ax3d.set_xlim(0, )
    #ax3d.set_ylim(0, 10e13)
    #ax3d.set_zlim(0, 1e12)
    ax3d.grid(False)
    plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    
    plot_phase_space('default')
   
   
   
   
   
   
   
   
    

    
