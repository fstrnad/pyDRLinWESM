#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""
adapted from https://github.com/timkittel/ays-model/
"""

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

DG_BIFURCATION_END = "dg-bifurcation-end"
DG_BIFURCATION_MIDDLE = "dg-bifurcation-middle"
RUN_OPTIONS = [aws.DEFAULT_NAME] + list(aws.MANAGEMENTS) + [DG_BIFURCATION_END, DG_BIFURCATION_MIDDLE]

if __name__ == "__main__":

    # a small hack to make all the parameters available as global variables
    # aws.globalize_dictionary(aws.AWS_parameters, module=aws)
    aws.globalize_dictionary(aws.grid_parameters, module=aws)
    aws.globalize_dictionary(aws.boundary_parameters, module=aws)

    parser = argparse.ArgumentParser(description="sample trajectories of the AWS model")

    parser.add_argument("option", choices=RUN_OPTIONS, default=aws.DEFAULT_NAME, nargs="?",
                        help="choose either the default or one of the management options to show")
    parser.add_argument("-m", "--mode", choices=["all", "lake"], default="all",
                        help="which parts should be sampled (default 'all')")
    parser.add_argument("-n", "--num", type=int, default=400,
            help="number of initial conditions (default: 400)")
    parser.add_argument("--no-boundary", dest="draw_boundary", action="store_false",
                        help="remove the boundary inside the plot")
    parser.add_argument("-s", "--save-pic", metavar="file", default="",
                        help="save the picture to 'file'")
    parser.add_argument("-z", "--zero", action="store_true",
            help="compute the zero of the RHS in the S=0 plane")

    # use argcomplete auto-completion
    argcomplete.autocomplete(parser)

    args = parser.parse_args()


    # small hack for now
    args.options =[args.option]

    num = args.num
    aws_0 = np.random.rand(num,3)  # args.mode == "all"
    if args.mode == "lake":
        aws_0[0] = aws_0[0] * aws.A_PB / (aws.A_PB + aws.A_mid)

    ########################################
    # prepare the integration
    ########################################
    time = np.linspace(0, 300, 1000)

    # parameter_dicts = []
    parameter_lists = []
    for management in args.options:
        if management == DG_BIFURCATION_END:
            parameter_dict = aws.get_management_parameter_dict("degrowth", aws.AYS_parameters)
            parameter_dict["beta"] = 0.035
        elif management == DG_BIFURCATION_MIDDLE:
            parameter_dict = aws.get_management_parameter_dict("degrowth", aws.AYS_parameters)
            parameter_dict["beta"] = 0.027
        else:
            parameter_dict = aws.get_management_parameter_dict(management, aws.AYS_parameters)
        if args.zero:
            x0 = [0.5, 0.5, 0] # a, w, s
            print("fixed point(s) of {}:".format(management))
            # below the '0' is for the time t, means we want to get the zero points of our plot
            print(opt.fsolve(aws.AYS_rescaled_rhs, x0,
                             args=(0., ) + helper.get_ordered_parameters(aws._AYS_rhs, parameter_dict)))
            print()
        parameter_lists.append(helper.get_ordered_parameters(aws._AYS_rhs, parameter_dict))
    # colors = ["green", "blue", "red"]
    # assert len(parameter_lists) <= len(colors), "need to add colors"


    colortop = "green"
    colorbottom = "black"
    
    #print(aws_0)
    fig, ax3d = ays_general.create_figure(A_mid=aws.A_mid, W_mid=aws.W_mid, S_mid=aws.S_mid)
    ax3d.view_init(ays_general.ELEVATION_FLOW, ays_general.AZIMUTH_FLOW)

    for i in range(num):
        x0 = aws_0[i]
        # management trajectory with degrowth:
        for parameter_list in parameter_lists:
            # Here we get the hairy trajectories that are integrated via odeint
            traj = integ.odeint(aws.AYS_rescaled_rhs, x0, time, args=parameter_list)
            
            ax3d.plot3D(xs=traj[:,0], ys=traj[:,1], zs=traj[:,2],
                        color=colorbottom if traj[-1,2]<0.5 else colortop, alpha=.3)

        # below traj was default and traj2 was degrowth
        # if traj2[:,0].max() > aws.A_PB > traj[:,0].max() and traj[-1,2] < 1e10 and traj2[-1,2] > 1e10: # lake candidate!
            # # JH: transform so that W_SF,sigma_default go to 1/2 and infinity goes to 1:
            # ax3d.plot3D(xs=traj[:,0], ys=traj[:,1]/(aws.W_mid+traj[:,1]), zs=traj[:,2]/(aws.S_mid+traj[:,2]),
                        # color="red" if traj[-1,2]<1000 else "blue", alpha=.7)
            # ax3d.plot3D(xs=traj2[:,0], ys=traj2[:,1]/(aws.W_mid+traj2[:,1]), zs=traj2[:,2]/(aws.S_mid+traj2[:,2]),
                        # color="orange" if traj2[-1,2]<1000 else "cyan", alpha=.7)
            # #print(traj2[:,0].max() - traj[:,0].max())


    if args.draw_boundary:
        ays_general.add_boundary(ax3d,
                                 sunny_boundaries=["planetary-boundary", "social-foundation"],
                                 **aws.grid_parameters, **aws.boundary_parameters)

    if args.save_pic:
        print("saving to {} ... ".format(args.save_pic), end="", flush=True)
        fig.savefig(args.save_pic)
        print("done")

    plt.show()




