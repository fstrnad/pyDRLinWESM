#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

from ays_general import __version__, __version_info__
import ays_general
import ays_model as ays

import pyviability as viab
from pyviability import helper
from pyviability import libviability as lv

import numpy as np
import scipy.optimize as opt

import time
import datetime as dt

import sys, os
import argparse, argcomplete

MANAGEMENTS = ays.MANAGEMENTS

boundaries_choices = ["planetary-boundary", "social-foundation", "both"]

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Analyze the AYS model with TSM using the PyViability package.",
    )

    # required arguments
    parser.add_argument("output_file", metavar="output-file",
                        help="output file where the TSM data is saved to")
    parser.add_argument("-b", "--boundaries", choices=boundaries_choices, required=True,
                        help="set the boundaries that will be considered for the run")

    # optional arguments
    parser.add_argument("--no-backscaling", action="store_false", dest="backscaling",
                        help="do not backscale the result afterwards")
    parser.add_argument("-d", "--dry-run", action="store_true",
                        help="do a dry run; perpare everything but then do not"
                        " actually run the TSM computation nor save a file")
    parser.add_argument("-e", "--eddies", action="store_true",
                        help="include eddies in the computation")
    parser.add_argument("-f", "--force", action="store_true",
                        help="if output-file exists already, overwrite it")
    parser.add_argument("-i", "--integrate", action="store_const",
                        dest="run_type", const="integration", default="linear",
                        help="integrate instead of using linear approx.")
    parser.add_argument("-n", "--no-save", action="store_true",
                        help="don't save the result")
    parser.add_argument("--num", type=int, default=ays.grid_parameters["n0"],
                        help="number of points per dimension for the grid")
    parser.add_argument("-p", "--set-parameter", nargs=2, metavar=("par", "val"),
                        action="append", dest="changed_parameters", default=[],
                        help="set a parameter 'par' to value 'val' "\
                        "(caution, eval is used for the evaluation of 'val'")
    parser.add_argument("--record-paths", action="store_true",
                        help="record the paths, direction and default / management option used, "\
                        "so a path can be reconstructed")
    parser.add_argument("--stop-when-finished", default=lv.TOPOLOGY_STEP_LIST[-1], metavar="computation-step",
                        choices=lv.TOPOLOGY_STEP_LIST,
                        help="stop when the computation of 'computation-step' is finished") 
    parser.add_argument("-z", "--zeros", action="store_true",
                        help="estimate the fixed point(s)")

    # management arguments
    management_group = parser.add_argument_group("management options")
    [management_group.add_argument("--"+MANAGEMENTS[m], "--"+m, action="append_const",
                                  dest="managements", const=m,
                                  default=[])
                            for m in MANAGEMENTS]

    # # add verbosity check
    # parser.add_argument("-v", "--verbosity", action="count", default=0,
                        # help="increase the output")
    verbosity = 2 # lower verbosity would actually be annoying

    # use argcomplete auto-completion
    argcomplete.autocomplete(parser)

    # do the actual parsing of the arguments
    args = parser.parse_args()

    OUTPUT_FILE_SUFFIX = ".out"
    if not args.dry_run and not args.output_file.endswith(OUTPUT_FILE_SUFFIX):
        parser.error("please use the suffix '{}' for 'output-file' (reason is actually the '.gitignore' file)".format(OUTPUT_FILE_SUFFIX))

    if not (args.force or args.dry_run):
        if os.path.isfile(args.output_file):
            parser.error("'{}' exists already, use '--force' option to overwrite".format(args.output_file))

    print()

    ays.grid_parameters["n0"] = args.num

    print("managements: {}".format(", ".join(args.managements) if args.managements else "(None)"))
    print()

    if args.changed_parameters:
        print("parameter changing:")
        combined_parameters = dict(ays.AYS_parameters)
        combined_parameters.update(ays.grid_parameters)
        combined_parameters.update(ays.boundary_parameters)
        for par, val in args.changed_parameters:
            for d in [ays.AYS_parameters, ays.grid_parameters, ays.boundary_parameters]:
                if par in d:
                    try:
                        val2 = eval(val, combined_parameters)
                    except BaseException as e:
                        print("couldn't evaluate {!r} for parameter '{}' because of {}: {}".format(val, par, e.__class__.__name__, str(e)))
                        sys.exit(1)
                    print("{} = {!r} <-- {}".format(par, val2, val))
                    d[ par ] = val2
                    break
            else:
                parser.error("'{}' is an unknown parameter".format(par))
    print()

    # a small hack to make all the parameters available as global variables
    ays.globalize_dictionary(ays.boundary_parameters, module=ays)
    ays.globalize_dictionary(ays.grid_parameters, module=ays)
    ays.globalize_dictionary(ays.grid_parameters)

    # manage and print the boundaries
    if args.boundaries == "both":
        ays.AYS_sunny = ays.AYS_sunny_PB_SF
        args.boundaries = ["planetary-boundary", "social-foundation"]
    elif args.boundaries == "planetary-boundary":
        ays.AYS_sunny = ays.AYS_sunny_PB
        args.boundaries = [args.boundaries]
    elif args.boundaries == "social-foundation":
        ays.AYS_sunny = ays.AYS_sunny_SF
        args.boundaries = [args.boundaries]
    else:
        assert False, "something went wrong here ..."
    assert isinstance(args.boundaries, list) and args.boundaries
    print("boundaries:")
    if "planetary-boundary" in args.boundaries:
        print("planetary / CO2 concentration:", end=" ")
        print("A_PB = {:6.2f} GtC above equ. <=> {:6.2f} ppm <=> a_PB = {:5.3f}".format(ays.A_PB, (ays.A_PB + ays.AYS_parameters["A_offset"]) / 840 * 400, ays.A_PB / (ays.A_mid + ays.A_PB)))
    if "social-foundation" in args.boundaries:
        print("social foundation / welfare limit:", end=" ")
        print("W_SF = {:4.2e} US$ <=> w_SF = {:5.3f}".format(ays.W_SF, ays.W_SF / (ays.W_mid + ays.W_SF)))

    # generate the grid, normalized to 1 in each dimension
    grid, scaling_vector, offset, x_step = viab.generate_grid(boundaries,
                                                         n0,
                                                         grid_type,
                                                         verbosity=verbosity)
    lv.STEPSIZE = 2 * x_step * max([1, np.sqrt( n0 / 80 )])  # prop to 1 / sqrt(n0)
    print("stepsize / gridstepsize: {:<5.3f}".format(lv.STEPSIZE / x_step))
    print()

    # generate the fitting states array
    states = np.zeros(grid.shape[:-1], dtype=np.int16)

    # mark the fixed point in infinity as shelter already
    states[ np.linalg.norm(grid - [0, 1, 1], axis=-1) < 5 * x_step] = -lv.SHELTER

    run_args = [offset, scaling_vector]
    run_kwargs = dict(returning=args.run_type)

    default_run = viab.make_run_function(ays.AYS_rescaled_rhs,
                                         helper.get_ordered_parameters(ays._AYS_rhs, ays.AYS_parameters),
                                         *run_args, **run_kwargs)

    print("recording-paths: {}".format(args.record_paths))
    print()

    if args.zeros:
        x0 = [0.5, 0.5, 0] # a, w, s
        # x0 = [ays.boundary_parameters["A_PB"], 0.5, 0] # A, w, s
        # print(x0)
        print("fixed point(s) of default:")
        # below the '0' is for the time t
        print(opt.fsolve(ays.AYS_rescaled_rhs, x0,
                         args=(0., ) + helper.get_ordered_parameters(ays._AYS_rhs, ays.AYS_parameters)))
        print()


    management_runs = []
    for m in args.managements:
        management_dict = ays.get_management_parameter_dict(m, ays.AYS_parameters)
        management_run = viab.make_run_function(ays.AYS_rescaled_rhs,
                                                helper.get_ordered_parameters(ays._AYS_rhs, management_dict),
                                                *run_args, **run_kwargs)
        management_runs.append(management_run)
        if args.zeros:
            print("fixed point(s) of {}:".format(m))
            # below the '0' is for the time t
            print(opt.fsolve(ays.AYS_rescaled_rhs, x0,
                             args=(0., ) + helper.get_ordered_parameters(ays._AYS_rhs, management_dict)))
            print()

    sunny = viab.scaled_to_one_sunny(ays.AYS_sunny, offset, scaling_vector)

    # out_of_bounds = [[False, True],   # A still has A_max as upper boundary
                     # [False, False],  # W compactified as w
                     # [False, False]]  # S compactified as s

    out_of_bounds = False # in a, w, s representation, doesn't go out of bounds of [0, 1)^3 by definition

    ays_general.register_signals()

    start_time = time.time()
    print("started: {}".format(dt.datetime.fromtimestamp(start_time).ctime()))
    print()
    if not args.dry_run:
        try:
            viab.topology_classification(grid, states, [default_run], management_runs,
                                            sunny, grid_type=grid_type,
                                            compute_eddies=args.eddies,
                                            out_of_bounds=out_of_bounds,
                                            remember_paths=args.record_paths,
                                            verbosity=verbosity,
                                            stop_when_finished=args.stop_when_finished,
                                            )
        except SystemExit as e:
            print()
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            print("interrupted by SystemExit or Signal {} [{}]".format(ays_general.NUMBER_TO_SIGNAL[e.args[0]], e.args[0]))
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            print()
    time_passed = time.time() - start_time

    print()
    print("run time: {!s}".format(dt.timedelta(seconds=time_passed)))
    print()

    if args.backscaling:
        grid = viab.backscaling_grid(grid, scaling_vector, offset)

    viab.print_evaluation(states)

    if not args.no_save:
        header = {
                "aws-version-info": __version_info__,
                "model": "AWS",
                "managements": args.managements,
                "boundaries": args.boundaries,
                "grid-parameters": ays.grid_parameters,
                "model-parameters": ays.AYS_parameters,
                "boundary-parameters": ays.boundary_parameters,
                "start-time": start_time,
                "run-time": time_passed,
                "viab-backscaling-done": args.backscaling,
                "viab-scaling-vector": scaling_vector,
                "viab-scaling-offset": offset,
                "input-args": args,
                "stepsize": lv.STEPSIZE,
                "xstep" : x_step,
                "out-of-bounds": out_of_bounds,
                "remember-paths": args.record_paths,
                "computation-status" : viab.get_computation_status(),
                }
        data = {"grid": grid,
                "states": states,
                }
        if args.record_paths:
            data["paths"] = lv.PATHS
            data["paths-lake"] = lv.PATHS_LAKE
        if not args.dry_run:
            ays_general.save_result_file(args.output_file, header, data, verbose=1)












