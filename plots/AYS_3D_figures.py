import numpy as np
from scipy.integrate import odeint

import matplotlib.ticker as ticker

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plt3d
from matplotlib.font_manager import FontProperties
from matplotlib.offsetbox import AnchoredText
from matplotlib.lines import Line2D
import os, sys

import functools as ft
import heapq as hq

import AYS.ays_model as ays
import AYS.ays_general as ays_general

AZIMUTH_FLOW, ELEVATION_FLOW = -140, 30
AZIMUTH, ELEVATION = 5, 20
# AZIMUTH, ELEVATION = 110, 34
AZIMUTH, ELEVATION = 170, 10

A_scale = 1
S_scale = 1e9
Y_scale = 1e12
tau_A = 50
tau_S = 50
beta = 0.03
beta_DG = 0.015
eps = 147
A_offset = 300
#theta = beta /(950-A_offset)
theta = 8.57e-5

rho = 2.
sigma = 4e12
#sigma_ET = sigma*0.5**(1/rho)
sigma_ET = 2.83e12

phi = 4.7e10

current_state = [240, 7e13, 5e11]



color_list=['#e41a1c','#ff7f00','#4daf4a','#377eb8','#984ea3']
# color_list=['#e41a1c','#ff7f00','#377eb8','#33a02c','#4daf4a']
#color_list=['#386cb0','#beaed4','#fdc086','#ffff99','#7fc97f',]

shelter_color='#ffffb3'
management_options=['default', 'DG' , 'ET','DG+ET' ]
parameters=['A' , 'Y' , 'S' , 'Action' , 'Reward' ]

reward_types=['survive', 'survive_cost', 'desirable_region', 'rel_share', 'PB']
management_actions=[(False, False), (True, False), (False, True), (True, True)]
SMALL_SIZE = 13
MEDIUM_SIZE = 15
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


current_state = [240, 7e13, 5e11]


# a small hack to make all the parameters available as global variables
ays.globalize_dictionary(ays.grid_parameters, module=ays)
ays.globalize_dictionary(ays.boundary_parameters, module=ays)    


def create_figure(Azimut=170, Elevation=36, label=None, colors=None, ax=None, ticks=True, plot_boundary=True,):

    if ax is None:
        fig3d = plt.figure(figsize=(16,9))
        ax3d = plt3d.Axes3D(fig3d)
    else:
        ax3d=ax
        fig3d=None

    if ticks==True:
        make_3d_ticks(ax3d)
    else:
        ax3d.set_xticks([])
        ax3d.set_yticks([])
        ax3d.set_zticks([])
    
    
    A_PB=[10, 265]
    top_view=[25,170]
    AZIMUTH, ELEVATION =  Azimut,Elevation
    ax3d.view_init(ELEVATION, AZIMUTH)
    
    
    S_scale = 1e9
    Y_scale = 1e12
    ax3d.set_xlabel("\n\nexcess atmospheric carbon\nstock A [GtC]", )
    ax3d.set_ylabel("\n\neconomic output Y \n  [%1.0e USD/yr]"%Y_scale, )
    ax3d.set_zlabel("\n\nrenewable knowledge\nstock S [%1.0e GJ]"%S_scale,)

    # Add boundaries to plot
    if plot_boundary:
        ays_general.add_boundary(ax3d,
                                 sunny_boundaries=["planetary-boundary", "social-foundation"],
                                 **ays.grid_parameters, **ays.boundary_parameters)

    ax3d.grid(False)

    legend_elements = [] 
    if label is None:
        # For Management Options
        for idx in range(len(management_options)):
                legend_elements.append(Line2D([0], [0], lw=2, color=color_list[idx], label=management_options[idx]))
        
        #ax3d.scatter(*zip([0.5,0.5,0.5]), lw=1, color=shelter_color, label='Shelter')
    else:
        for i in range(len(label)):
            ax3d.scatter(*zip([0.5,0.5,0.5]), lw=1, color=colors[i], label=label[i])

    # For Startpoint
    ax3d.scatter(*zip([0.5,0.5,0.5]), lw=6, color='red')
    
    # For legend
    legend_elements.append(Line2D([0], [0], lw=2, label='current state',  marker='o', color='w', markerfacecolor='red', markersize=15))   
    ax3d.legend(handles=legend_elements,prop={'size': 14}, bbox_to_anchor=(0.85,.90), fontsize=20,fancybox=True, shadow=True)

    return fig3d, ax3d

def get_parameters(action_number=0):

    """
    This function is needed to return the parameter set for the chosen management option.
    Here the action numbers are really transformed to parameter lists, according to the chosen 
    management option.
    Parameters:
        -action: Number of the action in the actionset.
         Can be transformed into: 'default', 'degrowth' ,'energy-transformation' or both DG and ET at the same time
    """
    if action_number < len(management_actions):
        action=management_actions[action_number]
    else:
        print("ERROR! Management option is not available!" + str (action))
        sys.exit(1)

    parameter_list=[(beta_DG if action[0] else beta  ,
                     eps, phi, rho, 
                     sigma_ET if action[1] else sigma, 
                     tau_A, tau_S, theta)]

    return parameter_list 

def plot_hairy_lines(num, ax3d):
    colortop = "lime"
    colorbottom = "black"
    ays_0 = np.random.rand(num, 3)
    time = np.linspace(0, 300, 1000)
    
    parameter_list=get_parameters(0)         
    for i in range(num):
        x0 = ays_0[i]
        traj = odeint(ays.AYS_rescaled_rhs, x0, time, args=parameter_list[0])
        #print(traj[-1])
        
        ax3d.plot3D(xs=traj[:,0], ys=traj[:,1], zs=traj[:,2],
                        color=colorbottom if traj[-1,2]<0.5 else colortop, alpha=.15)
        


def create_extract_figure(Azimut= AZIMUTH_FLOW, plot_boundary=False, label=None, colors=None, ax=None):
    
    font_size=20
    
    if ax is None:
        fig3d = plt.figure(figsize=(12,9))
        ax3d = plt3d.Axes3D(fig3d)
    else:
        ax3d=ax
        fig3d=None
        
    #ax3d.set_title('Sustainable path inside planetary boundaries \nfound by machine learning', size=font_size+2)
    
    #ax3d.set_xticks([])
    #ax3d.set_yticks([])
    #ax3d.set_zticks([])
    a_min,a_max=0.54,0.65
    y_min,y_max=0.33,0.55
    s_min,s_max=0.0,1
    
    ax3d.set_xlim(a_min,a_max)
    ax3d.set_ylim(y_min,y_max)
    ax3d.set_zlim(s_min,s_max)
    
    make_3d_ticks(ax3d, boundaries=[[a_min,a_max], [y_min,y_max], [s_min,s_max]], num_a=4)
    A_PB=[10, 265]
    top_view=[10,-176]
    ax3d.view_init(top_view[0], top_view[1])
    #ax3d.view_init(ays_general.ELEVATION_FLOW, Azimut)
    
    
    ax3d.set_xlabel("\n carbon stock A [GtC]", )
    ax3d.set_ylabel("\n\neconomic output Y \n  [%1.0e USD/yr]"%Y_scale, )
    ax3d.set_zlabel("\n\nrenewable knowledge\nstock S [%1.0e GJ]"%S_scale,)

    # Add boundaries to plot
    if plot_boundary:
        ays_general.add_boundary(ax3d,
                                 sunny_boundaries=["planetary-boundary", "social-foundation"],
                                 plot_boundaries=[[a_min,a_max],
                                                   [y_min,y_max],
                                                   [s_min,s_max]],
                                 **ays.grid_parameters, **ays.boundary_parameters)

    # Plot Startpoint
    
    ax3d.grid(False)
    
    if label is None:
        legend_elements = [] 
#         [Line2D([0], [0], color='b', lw=4, label='Line'),
#                    Line2D([0], [0], marker='o', color='w', label='Scatter',
#                           markerfacecolor='g', markersize=15),
#                    Patch(facecolor='orange', edgecolor='r',
#                          label='Color Patch')]
        
        # For Management Options
        for idx in range(len(management_options)):
                legend_elements.append(Line2D([0], [0], lw=2, color=color_list[idx], label=management_options[idx]))
        
        #ax3d.scatter(*zip([0.5,0.5,0.5]), lw=1, color=shelter_color, label='Shelter')
    else:
        for i in range(len(label)):
            ax3d.scatter(*zip([0.5,0.5,0.5]), lw=1, color=colors[i], label=label[i])

    # For Startpoint
    ax3d.scatter(*zip([0.5,0.5,0.5]), lw=6, color='red')
    legend_elements.append(Line2D([0], [0], lw=2, label='current state',  marker='o', color='w', markerfacecolor='red', markersize=15))
    
    ax3d.legend(handles=legend_elements, prop={'size': 14}, bbox_to_anchor=(0.85,.90), fontsize=20,fancybox=True, shadow=True)

    return fig3d, ax3d

def create_axis_3d(ax3d):

    ax3d.set_xticks([])
    ax3d.set_yticks([])
    ax3d.set_zticks([])
    
    
    A_PB=[10, 265]
    top_view=[15,180]
    ax3d.view_init(top_view[0], top_view[1])
    
    
    S_scale = 1e9
    Y_scale = 1e12
    ax3d.set_xlabel("                   A", )
    ax3d.set_ylabel("Y", )
    ax3d.set_zlabel("S ",)

    # Add boundaries to plot
    ays_general.add_boundary(ax3d,
                             sunny_boundaries=["planetary-boundary", "social-foundation"],
                             **ays.grid_parameters, **ays.boundary_parameters)

        # For Startpoint
    ax3d.scatter(*zip([0.5,0.5,0.5]), lw=6, color='red')

    

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

def make_3d_ticks(ax3d, boundaries = None,transformed_formatters=False,S_scale = 1e9, Y_scale = 1e12,num_a = 12, num_y = 12, num_s = 12,):
    if boundaries is None:
        boundaries = [None]*3
    
    transf = ft.partial(compactification, x_mid=current_state[0])
    inv_transf = ft.partial(inv_compactification, x_mid=current_state[0])
    
    # A- ticks
    if boundaries[0] is None:
        start, stop = 0, np.infty
        ax3d.set_xlim(0,1)
    else:
        start, stop = inv_transf(boundaries[0])
        ax3d.set_xlim(*boundaries[0])
    formatters, locators = ays_general.transformed_space(transf, inv_transf, axis_use=True, start=start, stop=stop, num=num_a)
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
    
    # Y - ticks
    transf = ft.partial(compactification, x_mid=current_state[1])
    inv_transf = ft.partial(inv_compactification, x_mid=current_state[1])

    if boundaries[1] is None:
        start, stop = 0, np.infty
        ax3d.set_ylim(0,1)
    else:
        start, stop = inv_transf(boundaries[1])
        ax3d.set_ylim(*boundaries[1])

    formatters, locators = ays_general.transformed_space(transf, inv_transf, axis_use=True, scale=Y_scale, start=start, stop=stop, num=num_y)
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
    
    
    transf = ft.partial(compactification, x_mid=current_state[2])
    inv_transf = ft.partial(inv_compactification, x_mid=current_state[2])

    # S ticks
    if boundaries[2] is None:
        start, stop = 0, np.infty
        ax3d.set_zlim(0,1)
    else:
        start, stop = inv_transf(boundaries[2])
        ax3d.set_zlim(*boundaries[2])

    formatters, locators = ays_general.transformed_space(transf, inv_transf, axis_use=True, scale=S_scale, start=start, stop=stop, num=num_s)
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


def make_2d_ticks(ax, boundaries = None,transformed_formatters=False,scale_1 = 1, scale_2=1, num_1 = 8, num_2 = 8, x_mid_1=1, x_mid_2=1):
    if boundaries is None:
        boundaries = [None]*2
    
    transf = ft.partial(compactification, x_mid=x_mid_1)
    inv_transf = ft.partial(inv_compactification, x_mid=x_mid_1)
    
    # xaxis - ticks
    if boundaries[0] is None:
        start, stop = 0, np.infty
        ax.set_xlim(0,1)
    else:
        start, stop = inv_transf(boundaries[0])
        ax.set_xlim(*boundaries[0])
        
    formatters, locators = ays_general.transformed_space(transf, inv_transf, axis_use=True, scale=scale_1,start=start, stop=stop, num=num_1)
    if transformed_formatters:
        new_formatters = []
        for el, loc in zip(formatters, locators):
            if el:
                new_formatters.append("{:4.2f}".format(loc))
            else:
                new_formatters.append(el)
        formatters = new_formatters
    #print(locators, formatters)
    ax.xaxis.set_major_locator(ticker.FixedLocator(locators))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter(formatters))
    
    # yaxis - ticks
    transf = ft.partial(compactification, x_mid=x_mid_2)
    inv_transf = ft.partial(inv_compactification, x_mid=x_mid_2)

    if boundaries[1] is None:
        start, stop = 0, np.infty
        ax.set_ylim(0,1)
    else:
        start, stop = inv_transf(boundaries[1])
        ax.set_ylim(*boundaries[1])

    formatters, locators = ays_general.transformed_space(transf, inv_transf, axis_use=True, scale=scale_2, start=start, stop=stop, num=num_2)
    if transformed_formatters:
        new_formatters = []
        for el, loc in zip(formatters, locators):
            if el:
                new_formatters.append("{:4.2f}".format(loc))
            else:
                new_formatters.append(el)
        formatters = new_formatters
    ax.yaxis.set_major_locator(ticker.FixedLocator(locators))
    ax.yaxis.set_major_formatter(ticker.FixedFormatter(formatters))
    