#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

import pyviability as viab
from pyviability import libviability as lv

from ays_general import __version__, __version_info__
import ays_model as aws
import ays_show, ays_general

from scipy import spatial as spat
from scipy.spatial import ckdtree
import numpy as np
import pickle, argparse, argcomplete
import itertools as it

import datetime as dt
import functools as ft
import os, sys

import matplotlib.pyplot as plt
from matplotlib import ticker as mticker


FILE_ERROR_MESSAGE = "{!r} seems to be an older aws file version or not a proper aws file, please use the '--reformat' option"

TRANSLATION = {
        "beta_DG" : r"$\beta_{0,LG}\, \left[\frac{\%}{\mathrm{a}}\right]$",
        "phi_CCS" : r"$\phi_{CCS}\, \left[\frac{\mathrm{GJ}}{\mathrm{GtC}}\right]$",
        "theta_SRM" : r"$\theta_{SRM}\, \left[\mathrm{a}^{-1}\mathrm{GJ}^{-1}\right]$",
        "sigma_ET" : r"$\sigma_{ET}\, \left[\mathrm{GJ}\right]$",
        }

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="show the TSM results of the AWS model")
    parser.add_argument("parameter", metavar="bifurcation-parameter",
                        help="the parameter which changes for the expected bifurcation")
    parser.add_argument("input_files", metavar="input-file", nargs="+",
                        help="input files with the contents from the TSM analysis")

    parser.add_argument("-s", "--save-pic", metavar="file", default="",
                        help="save the picture to 'file'")

    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="increase verbosity can be used as -v, -vv ...")

    # use argcomplete auto-completion
    argcomplete.autocomplete(parser)

    args = parser.parse_args()
    bifurcation_parameter = args.parameter

    for in_file in args.input_files:
        if not os.path.isfile(in_file):
            parser.error("can't find input file {!r}".format(in_file))

    cmp_list = [
        "model-parameters",
        "grid-parameters",
        "boundary-parameters",
        "managements",
        ]
    stacking_order = lv.REGIONS

    if args.verbose:
        print("stacking_order:", stacking_order)
        print()
        print("header comparison keys:", cmp_list)
        print()

    try:
        print("getting reference ... ", end="")
        reference_header, _ = ays_general.load_result_file(args.input_files[0], verbose=1)
    except IOError:
        parser.error(FILE_ERROR_MESSAGE.format(args.input_file))

    # remove the bifurcation_parameter from the reference and check at the same time that it really was in there
    reference_header["model-parameters"].pop(bifurcation_parameter)

    # check correct parameters
    bifurcation_parameter_list = []
    volume_lists = {r:[] for r in lv.REGIONS}
    for in_file in args.input_files:
        try:
            header, data = ays_general.load_result_file(in_file, verbose=1)
        except IOError:
            parser.error(FILE_ERROR_MESSAGE.format(args.input_file))
        # append the value of the bifurcation parameter to the list and check at the same time that it really was in there
        bifurcation_parameter_list.append(header["model-parameters"].pop(bifurcation_parameter))
        
        for el in cmp_list:
            if ays_general.recursive_difference(reference_header[el], header[el]):
                raise ValueError("incompatible headers")
        grid = np.asarray(data["grid"])
        states = np.asarray(data["states"])

        num_all = states.size

        for r in lv.REGIONS:
            volume_lists[r].append(np.count_nonzero(states == getattr(lv, r))/num_all)
    print()
    if bifurcation_parameter == "beta_DG":
        # multiply with 100 becuase it's shown in %
        bifurcation_parameter_list =list(map(lambda x: x*100, bifurcation_parameter_list))

    fig = plt.figure(figsize=(8, 9), tight_layout=True)
    ax = fig.add_subplot(111)

    bifurc_val = 1e-4
    def add_middles(arr, check_bifurc=False):
        new_arr = np.repeat(arr, 3)[:-2]
        new_arr[1::3] = 0.5 * (arr[:-1] + arr[1:])
        new_arr[2::3] = 0.5 * (arr[:-1] + arr[1:])
        if check_bifurc:
            for i in range(len(arr) - 1):
                i_next = i+1
                if (arr[i] > bifurc_val and arr[i_next] < bifurc_val) or (arr[i] < bifurc_val and arr[i_next] > bifurc_val):
                    new_arr[1::3][i] = arr[i]
                    new_arr[2::3][i] = arr[i_next]
        return new_arr

    argsort_param = np.argsort(bifurcation_parameter_list)
    bifurcation_parameter_list = np.asarray(bifurcation_parameter_list)[argsort_param]
    for key in volume_lists:
        volume_lists[key] = np.asarray(volume_lists[key])[argsort_param]

    bifurcation_parameter_list = add_middles(bifurcation_parameter_list)
    for key in volume_lists:
        volume_lists[key] = add_middles(volume_lists[key], check_bifurc=True)


    

    y_before = np.zeros_like(volume_lists[key]) # using the key from the for-loop before
    for r in stacking_order:
        vals = volume_lists[r]
        y_now = volume_lists[r] + y_before
        ax.fill_between(
                bifurcation_parameter_list, 
                y_before,
                y_now, 
                facecolor=lv.COLORS[getattr(lv, r)], lw=2, edgecolor="white")
        y_before += volume_lists[r]

    if bifurcation_parameter == "beta_DG":
        ax.plot([3]*2, [0,1], color="red", lw=5)
    ax.set_xlim(bifurcation_parameter_list[0], bifurcation_parameter_list[-1])
    ax.set_ylim(0, 1)


    xlabel = bifurcation_parameter
    if xlabel in TRANSLATION:
        xlabel = TRANSLATION[xlabel]
    ax.set_xlabel(xlabel)
    ax.set_ylabel("relative volume in phase space")

    if bifurcation_parameter == "beta_DG":
        pass
    else:
        ax.ticklabel_format(style="sci", axis="x", scilimits=(0,0),  useMathTex=True) # useMathTex is somehow ignored?

    text_xvals = {
            "beta_DG" : 2.3,
            "phi_CCS" : 5.4e10,
            "theta_SRM" : 6e-5,
            "sigma_ET" : 5e12,
            }
    if bifurcation_parameter in text_xvals:
        text_val = text_xvals[bifurcation_parameter]
    else:
        text_val = (bifurcation_parameter_list[0] + bifurcation_parameter_list[-1]) / 2

    shelter_y = 0.05
    glade_y = 0.12
    lake_y = 0.15
    sunny_up_y = 0.2
    dark_up_y = 0.35

    backw_y = 0.48
    sunny_down_y = 0.53
    dark_down_y = 0.7

    trench_y = 0.99


    # shelter
    ax.text(text_val, shelter_y, "$S$")
    # glade
    if bifurcation_parameter == "sigma_ET":
        ax.text(1.5e12, glade_y, "$G$")
    else:
        ax.text(text_val, glade_y, "$G$")
    # alke
    if bifurcation_parameter == "theta_SRM":
        ax.text(text_val - 0.3e-5, lake_y, "$L$")
    elif bifurcation_parameter == "beta_DG":
        ax.text(1.6, lake_y, "$L$")
    elif bifurcation_parameter == "sigma_ET":
        ax.text(text_val, 0.12, "$L$")
    elif bifurcation_parameter == "phi_CCS":
        ax.text(text_val - 0.1e10, lake_y, "$L$")
    else:
        ax.text(text_val, lake_y, "$L$")
    # remaining sunny upstream
    if bifurcation_parameter == "sigma_ET":
        ax.text(text_val, 0.18, "$U^{(+)}$")
    else:
        ax.text(text_val, sunny_up_y, "$U^{(+)}$")
    # remaining dark upstream
    ax.text(text_val, dark_up_y, "$U^-$")
    # backwaters
    if bifurcation_parameter == "beta_DG":
        ax.text(1.6, backw_y, "$W$")
    elif bifurcation_parameter == "sigma_ET":
        ax.text(text_val, 0.45, "$W$")
    else:
        ax.text(text_val, backw_y, "$W$")
    # remaining sunny downstream
    if bifurcation_parameter == "sigma_ET":
        ax.text(text_val, 0.51, "$D^{(+)}$")
    else:
        ax.text(text_val, sunny_down_y, "$D^{(+)}$")
    # remaining dark downstream
    ax.text(text_val, dark_down_y, "$D^-$")
    # eddies
    if bifurcation_parameter == "beta_DG":
        ax.text(3.15, sunny_down_y, "$E^+$")
        ax.text(3.15, dark_down_y, "$E^-$")
    # trench
    ax.text(text_val, trench_y, "$\Theta$")




    if args.save_pic:
        print("saving to {} ... ".format(args.save_pic), end="", flush=True)
        fig.savefig(args.save_pic)
        print("done")

    sys.stdout.flush()
    sys.stderr.flush()
    plt.show()









