#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

import pyviability as viab
from pyviability import libviability as lv

import ays_model as aws
import ays_general

import scipy.spatial as spat
import numpy as np
import pickle, argparse, argcomplete
import sys, os
import datetime as dt
import functools as ft


import matplotlib.pyplot as plt

def RegionName2Option(vname, style="long"):
    if style=="long":
        return vname.replace("_", "-").lower()
    elif style=="short":
        return "".join(map(lambda x: x[0], vname.lower().split("_")))
    raise ValueError("unkown style: {!r}".format(style))


# check that there are no short option used twice
_all_regions_short = list(map(lambda x: RegionName2Option(x, style="short"), lv.REGIONS))
assert len(_all_regions_short) == len(set(_all_regions_short))
del _all_regions_short

# prepare all the stuff needed for the regions argument parsing
regions_dict_short = { RegionName2Option(region, style="short") : region for region in lv.REGIONS }
regions_dict_long = { RegionName2Option(region, style="long") : region for region in lv.REGIONS }
regions_dict = dict(regions_dict_long)
regions_dict.update(regions_dict_short)
regions_arguments = [("all", "a")] + list(zip(map(RegionName2Option, lv.REGIONS), map(ft.partial(RegionName2Option, style="short"), lv.REGIONS)))
regions_arguments_flattened = sorted([item for sublist in regions_arguments for item in sublist])

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="show the TSM results of the AWS model")
    parser.add_argument("input_file", metavar="input-file",
                        help="input file with the contents from the TSM analysis")

    boundaries_group = parser.add_mutually_exclusive_group()
    boundaries_group.add_argument("-b", "--plot-boundaries-transformed", metavar="boundaries",
                                  help="set the boundaries (in (a,w,s)-coordinates) as a list with shape (3,2)")
    boundaries_group.add_argument("--plot-boundaries-original", metavar="boundaries",
                                  help="set the boundariess (in (A,W,S)-coordinates) as a list with shape (3,2)")
    parser.add_argument("-d", "--defaults", default=[], nargs="+",
                        choices=["grid", "model", "boundary"],
                        help="show all the default values")

    paths_parser = parser.add_argument_group(title="analyze tool",
                                             description="tools for analyzing")
    analyze_group = paths_parser.add_mutually_exclusive_group()
    analyze_group.add_argument("--analyze-transformed", nargs=2, metavar=("point", "distance"),
                        help="analyze all points, that are closer to 'point' (in (a, w, s)-coordinates) than 'distance'")
    analyze_group.add_argument("--analyze-original", nargs=2, metavar=("point", "distance"),
                        help="analyze all points, that are closer to 'point' (in (A, W, S)-coordinates) than 'distance'")

    paths_parser.add_argument("--mark", metavar="color",
                              help="mark the points chosen by analyze as 'color' points")
    paths_parser.add_argument("--mark-alpha", metavar="opacity", type=float, default=1.,
                              help="mark the points chosen by analyze as 'color' points")
    paths_parser.add_argument("--show-path", action="store_true",
                              help="show a path for all points determined by '--analyze'")
    paths_parser.add_argument("--paths-outside", action="store_true",
                              help="paths go go out of the plotting boundaries")
    paths_parser.add_argument("--no-paths-lake-fallback", action="store_false", dest="paths_lake_fallback",
                              help="fallback to PATHS if NO INFO in PATHS_LAKE")

    regions_parser = parser.add_argument_group(title="plot regions",
                                               description="choose which regions are plotted and how")
    regions_parser.add_argument("-r", "--show-region", metavar="region", dest="regions", 
                                default=[], nargs="+", choices=regions_arguments_flattened,
                                help="choose the regions to be shown in the plot: " + 
                                     ", ".join(["{} ({})".format(region_long, region_short) for region_long, region_short in regions_arguments]))
    region_plotting_styles = ["points", "surface"]
    regions_parser.add_argument("--regions-style", choices=region_plotting_styles, default=region_plotting_styles[0],
                                help="choose the plotting style from: " + ", ".join(region_plotting_styles))
    regions_parser.add_argument("--alpha", type=float,
                                help="set the alpha value (opacity) of the plotted points")

    parser.add_argument("--paper", action="store_true",
                        help="create the picture for paper style")
    parser.add_argument("--reformat", action="store_true",
                        help="automatically reformat 'input-file' if necessary")
    parser.add_argument("-s", "--save-pic", metavar="file", default="",
                        help="save the picture to 'file'")
    parser.add_argument("-t", "--transformed-formatters", action="store_true",
                        help="show from 0 to 1 at each axis instead of 0 to infty")

    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="increase verbosity can be used as -v, -vv ...")

    # use argcomplete auto-completion
    argcomplete.autocomplete(parser)

    args = parser.parse_args()

    # if args.save_video and not args.animate:
        # parser.error("no use to produce a video without animating the plot")

    if not os.path.isfile(args.input_file):
        parser.error("can't find input file {!r}".format(args.input_file))
    if args.defaults:
        for d in args.defaults:
            print("defaults for {}:".format(d))
            if d == "grid":
                dic = aws.grid_parameters
            elif d == "model":
                dic = aws.AYS_parameters
            elif d == "boundary":
                dic = aws.boundary_parameters
            else:
                raise ValueError("Did you forget to change something here?")
            print(ays_general.recursive_dict2string(dic))
            print()
        sys.exit(0)

    # resolve the chosen regions and translate them to the names in pyviability
    if args.regions:
        if "a" in args.regions or "all" in args.regions:
            args.regions = lv.REGIONS
        else:
            args.regions = list(set(map(regions_dict.__getitem__, args.regions)))

    try:
        header, data = ays_general.load_result_file(args.input_file, auto_reformat=args.reformat, verbose=1)
    except IOError:
            parser.error("{!r} seems to be an older aws file version, please use the '--reformat' option".format(args.input_file))
    print()

    if not header["viab-backscaling-done"]:
        raise NotImplementedError("there is no plotting for unrescaled systems yet (and probably won't ever be)")

    LAKE_PLOT = False # only used for the plot of the lake in the paper
    # define paper style
    if args.paper:
        if not args.verbose:
            args.verbose = 1
        args.analyze_original = ("[241,7.01e13,0.501e12]", "0.0035")
        # read current state here directly?
        args.mark = "red"
        args.regions_style = "surface"
        if not args.verbose:
            args.verbose=1
        if args.regions == ["LAKE"]:
            LAKE_PLOT = True
            args.plot_boundaries_original = "[[0,400],[3.55e13,9e13],[0.2e12,1e12]]"
            args.alpha = 0.6
            args.mark_alpha = 0.4


    
    # to be used for eval(...) statements
    combined_parameters = dict(header["model-parameters"])
    combined_parameters.update(header["grid-parameters"])
    combined_parameters.update(header["boundary-parameters"])
    # for some computations
    A_mid = header["grid-parameters"]["A_mid"]
    W_mid = header["grid-parameters"]["W_mid"]
    S_mid = header["grid-parameters"]["S_mid"]
    X_mid = np.array([ A_mid, W_mid, S_mid ])

    if args.alpha is None:
        if args.regions_style == "points":
            args.alpha = 1/header["grid-parameters"]["n0"]
        else:
            args.alpha = 0.8

    # evaluate the boundaries string to an array
    if args.plot_boundaries_original is not None:
        args.plot_boundaries = args.plot_boundaries_original
    elif args.plot_boundaries_transformed is not None:
        args.plot_boundaries = args.plot_boundaries_transformed
    else:
        args.plot_boundaries = None
    if args.plot_boundaries is not None:
        args.plot_boundaries = np.array(eval(args.plot_boundaries, combined_parameters))
        if args.plot_boundaries_original is not None:
            args.plot_boundaries = args.plot_boundaries / (X_mid[:, np.newaxis] + args.plot_boundaries)
        assert args.plot_boundaries.shape == (3, 2)
        assert np.all(args.plot_boundaries >= 0) and np.all(args.plot_boundaries <= 1)


    if args.analyze_original is not None:
        args.analyze = args.analyze_original
    elif args.analyze_transformed is not None:
        args.analyze = args.analyze_transformed
    else:
        args.analyze = None
    if not args.analyze is None:
        path_x0 = np.array(eval(args.analyze[0], combined_parameters))
        if args.analyze_original is not None:
            path_x0 = path_x0 / ( X_mid + path_x0 )
        path_dist = float(eval(args.analyze[1]))
        assert path_x0.shape == (3,)
        assert np.all(path_x0 > 0) and np.all(path_x0 < 1)

    if args.show_path:
        if not header["remember-paths"]:
            parser.error("'{}' does not contain recorded paths".format(args.input_file))


    grid = data["grid"]
    states = data["states"]

    print("date: {}".format(dt.datetime.fromtimestamp(header["start-time"]).ctime()))
    print("duration: {!s}".format(dt.timedelta(seconds=header["run-time"])))
    print()
    print("management options: {}".format(", ".join(header["managements"]) if header["managements"] else "(None)"))
    pars = header["model-parameters"]  # just to make it shorter here
    for m in header["managements"]:
        ending = "_" + aws.MANAGEMENTS[m].upper()
        changed = False
        for key in pars:
            # choose the variables that are changed by the ending
            if key.endswith(ending):
                default_key = key[:-len(ending)]
                print("{} = {} <--> {} = {}".format(key, ays_general.formatted_value(pars[key]), default_key, ays_general.formatted_value(pars[default_key])))
    print()
    assert header["boundaries"], "no boundaries for computation?"
    print("boundaries:")
    for b in header["boundaries"]:
        if b == "planetary-boundary":
            A_PB = header["boundary-parameters"]["A_PB"]
            A_mid = header["grid-parameters"]["A_mid"]
            A_offset = header["model-parameters"]["A_offset"]
            print("planetary / CO2 concentration:", end=" ")
            print("A_PB = {:6.2f} GtC above equ. <=> {:6.2f} ppm <=> a_PB = {:5.3f}".format(A_PB, (A_PB + A_offset) / 840 * 400 , A_PB / (A_mid + A_PB)))
        elif b == "social-foundation":
            W_SF = header["boundary-parameters"]["W_SF"]
            W_mid = header["grid-parameters"]["W_mid"]
            print("social foundation / welfare limit:", end=" ")
            print("W_SF = {:4.2e} US$ <=> w_SF = {:5.3f}".format(W_SF, W_SF / (W_mid + W_SF)))
        else:
            print("{!r} (no further info)".format(b))
    print()
    print("stepsize / gridstepsize: {:<5.3f}".format(header["stepsize"] / header["xstep"]))
    print()
    print("points per dimension: {:4d}".format(header["grid-parameters"]["n0"]))
    print()
    print("paths recorded: {}".format(header["remember-paths"]))
    if args.analyze:
        print("showing for", path_x0, path_dist)
    print()

    ays_general.print_changed_parameters(header["model-parameters"], aws.AYS_parameters, prefix="changed model parameters:")
    ays_general.print_changed_parameters(header["grid-parameters"], aws.grid_parameters, prefix="changed grid parameters:")
    ays_general.print_changed_parameters(header["boundary-parameters"], aws.boundary_parameters, prefix="changed boundary parameters:")

    if args.verbose:
        print("#" * 70)
        print("# HEADER")
        print(ays_general.recursive_dict2string(header))
        print("# END HEADER")
        print("#" * 70)
        print()


    viab.print_evaluation(states)

    if args.regions or args.analyze is not None:
        print()

        if args.regions or args.show_path or args.mark is not None:
            figure_parameters = dict(header["grid-parameters"])
            figure_parameters["boundaries"] = args.plot_boundaries
            figure_parameters["num_a"] = 6
            fig, ax3d = ays_general.create_figure(transformed_formatters=args.transformed_formatters, **figure_parameters)

            ax_parameters = dict(header["boundary-parameters"])  # make a copy
            ax_parameters.update(header["grid-parameters"])
            ays_general.add_boundary(ax3d, sunny_boundaries=header["boundaries"], plot_boundaries=args.plot_boundaries, **ax_parameters)

            def isinside(x, bounds):
                if bounds is None:
                    return np.ones(np.shape(x)[:-1], dtype=bool)
                return np.all((bounds[:, 0] <= x) & ( x <= bounds[:, 1]), axis=-1)

            mask2 = isinside(grid, args.plot_boundaries)

            for region in args.regions:
                region_num = getattr(lv, region)
                mask = (states == region_num) &  mask2
                if args.regions_style == "points":
                    ax3d.plot3D(xs=grid[:, 0][mask], ys=grid[:, 1][mask], zs=grid[:, 2][mask],
                                color=lv.COLORS[region_num],
                                alpha=args.alpha,
                                linestyle="", marker=".", markersize=30,
                                )
                elif args.regions_style == "surface":

                    ###############################################################################
                    # a quick and straightforward alpha shape computation is implemented below

                    import numba as nb

                    @nb.jit
                    def nb_dot(x, y):
                        val = 0
                        for x_i, y_i in zip(x, y):
                            val += x_i * y_i
                        return val

                    @nb.jit
                    def nb_cross(x, y):
                        val = np.array([  x[1]*y[2] - x[2]*y[1],
                                 x[2]*y[0] - x[0]*y[2],
                                 x[0]*y[1] - x[1]*y[0] ])
                        return val

                    @nb.jit
                    def r2_circumsphere_tetrahedron_single(a, b, c, d):
                        ad = a - d
                        bd = b - d
                        cd = c - d

                        ad2 = nb_dot(ad, ad)
                        bd2 = nb_dot(bd, bd)
                        cd2 = nb_dot(cd, cd)

                        cross_1 = nb_cross(bd, cd)
                        cross_2 = nb_cross(cd, ad)
                        cross_3 = nb_cross(ad, bd)

                        q = ad2 * cross_1 + bd2 * cross_2 + cd2 * cross_3
                        p = 2 * np.abs( nb_dot(ad, cross_1) )
                        if p < 1e-10:
                            return np.infty
                        
                        r2 = nb_dot(q, q) / p**2

                        return r2

                    @nb.jit(nopython=True)
                    def r2_circumsphere_tetrahedron(a, b, c, d):
                        len_a = len(a)
                        r2 = np.zeros((len_a,))
                        for i in range(len_a):
                            r2[i] = r2_circumsphere_tetrahedron_single(a[i], b[i], c[i], d[i])
                        return r2

                    def get_faces(tetrahedron):
                        faces = np.zeros((4, 3))
                        for n, (i1, i2, i3) in enumerate([(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]):
                            faces[n] = tetrahedron[i1], tetrahedron[i2], tetrahedron[i3]
                        return faces

                    def get_single_faces(triangulation):
                        num_faces_single = 4
                        num_tetrahedrons = triangulation.shape[0]
                        num_faces = num_tetrahedrons * num_faces_single
                        faces = np.zeros((num_faces, 3), np.int_) # 3 is the dimension of the model
                        mask = np.ones((num_faces,), np.bool_)
                        for n in range(num_tetrahedrons):
                            faces[num_faces_single * n: num_faces_single * (n+1)] = get_faces(triangulation[n])
                        orderlist = ["x{}".format(i) for i in range(faces.shape[1])]
                        dtype_list = [(el, faces.dtype.str) for el in orderlist]
                        faces.view(dtype_list).sort(axis=0)
                        for k in range(num_faces-1):
                            if mask[k]:
                                if np.all(faces[k] == faces[k+1]):
                                    mask[k] = False
                                    mask[k+1] = False
                        single_faces = faces[mask]
                        return single_faces

                    ###############################################################################



                    alpha_radius = 0.05 # this would actually depend on the input resolution but I just hardcoded it, as it's code for the paper
                    basefilename = os.path.splitext(os.path.split(args.input_file)[-1])[0]
                    CACHE_FILE = ".{}-region{}-{}.cache".format(basefilename, region_num, lv.REGIONS[region_num])

                    mask = (states == region_num) &  mask2
                    region_points = grid[mask]

                    if os.path.exists(CACHE_FILE):
                        if args.verbose:
                            print()
                            print("found cache file ({}) with an existing alpha shape, loading ... ".format(CACHE_FILE), end="", flush=True)
                        with open(CACHE_FILE, "rb") as f:
                            outer_triangulation = pickle.load(f)
                        if args.verbose:
                            print("done")
                    else:
                        if args.verbose:
                            print()
                            print("computing alpha shape for {}: {}".format(region_num, lv.REGIONS[region_num]))
                        triangulation = spat.Delaunay(region_points)

                        tetrahedrons = region_points[triangulation.simplices]
                        radii2 = r2_circumsphere_tetrahedron(tetrahedrons[:, 0, :], tetrahedrons[:, 1, :], tetrahedrons[:, 2, :], tetrahedrons[:, 3, :])
                        reduced_triangulation = triangulation.simplices[radii2 < alpha_radius**2]
                        del radii2, triangulation, tetrahedrons

                        outer_triangulation = get_single_faces(reduced_triangulation)
                        if args.verbose:
                            print()
                            print("saving alpha_shape to cache file ({}) ... ".format(CACHE_FILE), end="", flush=True)
                        with open(CACHE_FILE, "wb") as f:
                            pickle.dump(outer_triangulation, f)
                        if args.verbose:
                            print("done")

                    if args.verbose:
                        print()
                        print("plotting hull for {}: {}".format(region_num, lv.REGIONS[region_num]))
                        print("color {} ({}: {})".format(lv.COLORS[region_num], region_num, lv.REGIONS[region_num]))
                        print("{} triangles)".format(outer_triangulation.shape[0]))
                        print()
                    ax3d.plot_trisurf(
                            region_points[:, 0], region_points[:, 1], region_points[:, 2], 
                            triangles=outer_triangulation,
                            color=lv.COLORS[region_num],
                            antialiased=True,
                            linewidth=(0.2 if LAKE_PLOT else 0.1),
                            shade=0,
                            alpha=args.alpha,
#                            alpha=0.5,
                            edgecolors="white",
                            zorder=10,
                            )
                else:
                    raise NotImplementedError("plotting style '{}' is not yet implemented".format(args.regions_style))
        if args.analyze:
            bounds = args.plot_boundaries
            print("compute indices of points that are to be analyzed ... ", end="", flush=True)
            diff = grid - path_x0
            mask = (np.linalg.norm(diff, axis=-1) <= path_dist)
            starting_indices = np.where(mask)[0].tolist()
            _starting_indices = list(starting_indices)
            print("done")
            print()
            if not starting_indices:
                print("your point and distance do not match any grid points")
            else:
                print("matched:")
                print("STATE : COUNT")
                _matched_states = states[mask]
                matched_states = sorted(np.unique(_matched_states))
                for s in matched_states:
                    print("{:>5} : {:>5}".format(s, np.count_nonzero(_matched_states == s)))
                    if args.verbose >= 2 and not args.show_path:
                        for y in grid[mask][_matched_states == s]:
                            x = X_mid * y / (1 - y)
                            print(y, "<==>" ,x)
                        print()
                if args.mark is not None:
                    # ax3d.plot3D(xs=[path_x0[0], path_x0[0]], ys=[path_x0[1], path_x0[1]], zs=bounds[2],
                    #             color=args.mark, zorder=1)
                    # ax3d.plot3D(xs=bounds[0], ys=[path_x0[1], path_x0[1]], zs=[path_x0[2], path_x0[2]],
                    #             color=args.mark, zorder=1)
                    # ax3d.plot3D(xs=[path_x0[0], path_x0[0]], ys=bounds[1], zs=[path_x0[2], path_x0[2]],
                    #             color=args.mark, zorder=1)
                    ax3d.plot3D(xs=grid[:, 0][mask], ys=grid[:, 1][mask], zs=grid[:, 2][mask],
                            color=args.mark, linestyle="", marker=".", markersize=30, zorder=2,
                                alpha=args.mark_alpha)
                print()
                if args.show_path:
                    plotting = lambda traj, choice: ax3d.plot3D(xs=traj[0], ys=traj[1], zs=traj[2],
                                                                color="lightblue" if choice == 0 else "black")
                    paths_outside = args.paths_outside
                    if paths_outside or bounds is None:
                        path_isinside = ays_general.dummy_isinside
                    else:
                        def path_isinside(x):
                            return np.all((bounds[:, 0] <= x) & ( x <= bounds[:, 1]))
                    ays_general.follow_indices(starting_indices,
                                               grid=grid,
                                               states=states,
                                               paths=data["paths"],
                                               trajectory_hook=plotting,
                                               verbose=args.verbose,
                                               isinside=path_isinside)

                    if lv.LAKE in matched_states:
                        if args.verbose < 2:
                            print("following lake inside of manageable region ...", end="", flush=True)
                        else:
                            print()
                            print("following LAKE points inside of manageable region")
                        starting_indices = [index for index in _starting_indices if states[index] == lv.LAKE]
                        plotting = lambda traj, choice: ax3d.plot3D(xs=traj[0], ys=traj[1], zs=traj[2],
                                                                    color="green" if choice == 0 else "brown")
                        ays_general.follow_indices(starting_indices,
                                                   grid=grid,
                                                   states=states,
                                                   paths=data["paths-lake"],
                                                   fallback_paths=data["paths"] if args.paths_lake_fallback else None,
                                                   trajectory_hook=plotting,
                                                   verbose=args.verbose,
                                                   isinside=path_isinside)

        if args.save_pic:
            print("saving to {} ... ".format(args.save_pic), end="", flush=True)
            fig.savefig(args.save_pic, dpi=100)
            print("done")

        sys.stdout.flush()
        sys.stderr.flush()
        plt.show()









