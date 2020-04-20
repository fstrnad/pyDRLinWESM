"""
adapted from https://github.com/timkittel/ays-model/
"""
from pyviability import libviability as lv

import heapq as hq
import functools as ft

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plt3d
import matplotlib.ticker as ticker
from matplotlib import animation

import numpy as np
import operator as op
import pickle
import signal
import sys
import warnings as warn


def versioninfo2version(v_info):
    return ".".join(map(str, v_info))

DEFAULT_VERSION_INFO = (0, 1)  # that's where it all started

version_info = __version_info__ = (0, 3)
version = __version__ = versioninfo2version(__version_info__)


"""
aws-file version changes:
# Note that the abbreviation aws is used here instead of ays, to keep the compatibility with the older files.
0.3: added 'computation-status'
0.2: the first ones with actual versioning, adding 'paths-lake' if paths has been given
no version or 0.1: the stuff from the beginning
"""

INFTY_SIGN = u"\u221E"

# AZIMUTH, ELEVATION = -140, 30
AZIMUTH_FLOW, ELEVATION_FLOW = -140, 30
AZIMUTH, ELEVATION = 5, 20
# AZIMUTH, ELEVATION = 110, 34
AZIMUTH, ELEVATION = 170, 10


# patch to remove padding at ends of axes:
###patch start###
from mpl_toolkits.mplot3d.axis3d import Axis
if not hasattr(Axis, "_get_coord_info_old"):
    def _get_coord_info_new(self, renderer):
        mins, maxs, centers, deltas, tc, highs = self._get_coord_info_old(renderer)
        mins += deltas / 4
        maxs -= deltas / 4
        return mins, maxs, centers, deltas, tc, highs
    Axis._get_coord_info_old = Axis._get_coord_info
    Axis._get_coord_info = _get_coord_info_new
###patch end###


def remove_inner(arr):
    arr = np.asarray(arr)
    assert len(arr.shape) == 2
    l, d = arr.shape
    left_array = np.ones((l, ), dtype=bool)
    # for i in range(l):


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

def transformed_space(transform, inv_transform,
                      start=0, stop=np.infty, num=12,
                      scale=1,
                      num_minors = 50,
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

def animate(fig, ax3d, fname):
    assert fname.endswith(".mp4"), "for now '.mp4' files for video only"
    def turning_animation(i):
        ax3d.view_init(ELEVATION, AZIMUTH + i)
    # Animate
    anim = animation.FuncAnimation(fig, turning_animation, 
                                   init_func=init,
                                   frames=360, interval=20, blit=True)
    # Save
    anim.save(fname, fps=30, extra_args=['-vcodec', 'libx264'])
    # ax3d.view_init(ELEVATION, AZIMUTH)

def create_figure(*bla, S_scale = 1e9, W_scale = 1e12, W_mid = None, S_mid = None, boundaries = None, transformed_formatters=False,
                  num_a = 12, num_y = 12, num_s = 12, **kwargs):


    kwargs = dict(kwargs)

    if boundaries is None:
        boundaries = [None]*3

    fig = plt.figure(figsize=(16,9))
    ax3d = plt3d.Axes3D(fig)
    ax3d.set_xlabel("\n\nexcess atmospheric carbon\nstock A [GtC]")
    ax3d.set_ylabel("\neconomic output Y [%1.0e USD/yr]"%W_scale)
    ax3d.set_zlabel("\n\nrenewable knowledge\nstock S [%1.0e GJ]"%S_scale)

    # make proper tickmarks:
    if "A_max" in kwargs:
        
        A_max = kwargs.pop("A_max")
        Aticks = np.linspace(0,A_max,11)
        ax3d.w_xaxis.set_major_locator(ticker.FixedLocator(Aticks))
        ax3d.w_xaxis.set_major_formatter(ticker.FixedFormatter(Aticks.astype("int")))
        if boundaries is None:
            ax3d.set_xlim(Aticks[0],Aticks[-1])
        else:
            ax3d.set_xlim(*boundaries[0])
    elif "A_mid" in kwargs:
        # Aticks
        A_mid = kwargs.pop("A_mid")
        transf = ft.partial(compactification, x_mid=A_mid)
        inv_transf = ft.partial(inv_compactification, x_mid=A_mid)

        if boundaries[0] is None:
            start, stop = 0, np.infty
        else:
            start, stop = inv_transf(boundaries[0])
        formatters, locators = transformed_space(transf, inv_transf, axis_use=True, start=start, stop=stop, num=num_a)
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

        if boundaries[0] is None:
            ax3d.set_xlim(0,1)
        else:
            ax3d.set_xlim(*boundaries[0])

    else:
        raise KeyError("can't find proper key for 'A' in kwargs that determines which representation of 'A' has been used")

    if kwargs:
        warn.warn("omitted arguments: {}".format(", ".join(sorted(kwargs))), stacklevel=2)
    
    # Y - ticks
    transf = ft.partial(compactification, x_mid=W_mid)
    inv_transf = ft.partial(inv_compactification, x_mid=W_mid)

    if boundaries[1] is None:
        start, stop = 0, np.infty
    else:
        start, stop = inv_transf(boundaries[1])
    formatters, locators = transformed_space(transf, inv_transf, axis_use=True, scale=W_scale, start=start, stop=stop, num=num_y)
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

    if boundaries[1] is None:
        ax3d.set_ylim(0,1)
    else:
        ax3d.set_ylim(*boundaries[1])

    #S - ticks
    transf = ft.partial(compactification, x_mid=S_mid)
    inv_transf = ft.partial(inv_compactification, x_mid=S_mid)

    if boundaries[2] is None:
        start, stop = 0, np.infty
    else:
        start, stop = inv_transf(boundaries[2])
    formatters, locators = transformed_space(transf, inv_transf, axis_use=True, scale=S_scale, start=start, stop=stop, num=num_s)
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

    if boundaries[2] is None:
        ax3d.set_zlim(0,1)
    else:
        ax3d.set_zlim(*boundaries[2])

    ax3d.view_init(ELEVATION, AZIMUTH)

    return fig, ax3d


def add_boundary(ax3d, *, sunny_boundaries, add_outer=False, plot_boundaries=None, **parameters):
# def add_boundary(ax3d, *, boundary = ["planetary-boundary"], add_outer=False, plot_boundaries=None, **parameters):
    """show boundaries of desirable region"""

    if not sunny_boundaries:
        # nothing to do
        return 

    # get the boundaries of the plot (and check whether it's an old one where "A" wasn't transformed yet
    if plot_boundaries is None:
        if "A_max" in parameters:
            a_min, a_max = 0, parameters["A_max"]
        elif "A_mid" in parameters:
            a_min, a_max = 0, 1
        w_min, w_max = 0, 1
        s_min, s_max = 0, 1
    else:
        a_min, a_max = plot_boundaries[0]
        w_min, w_max = plot_boundaries[1]
        s_min, s_max = plot_boundaries[2]

    plot_pb = False
    plot_sf = False
    if "planetary-boundary" in sunny_boundaries:
        A_PB = parameters["A_PB"]
        if "A_max" in parameters:
            pass # no transformation necessary
        elif "A_mid" in parameters:
            A_PB = A_PB / (A_PB + parameters["A_mid"])
        else:
            assert False, "couldn't identify how the A axis is scaled"
        if a_min < A_PB < a_max:
            plot_pb = True
    if "social-foundation" in sunny_boundaries:
        W_SF = parameters["W_SF"]
        W_SF = W_SF / (W_SF + parameters["W_mid"])
        if w_min < W_SF < w_max:
            plot_sf = True


    if plot_pb and plot_sf:
        corner_points_list = [[
                [A_PB , W_SF , s_min],
                [A_PB , w_max, s_min],
                [A_PB , w_max, s_max],
                [A_PB , W_SF , s_max],
                ],
                [
                [A_PB , W_SF , s_max],
                [a_min, W_SF, s_max],
                [a_min, W_SF, s_min],
                [A_PB , W_SF , s_min],
                ]]
    elif plot_pb:
        corner_points_list = [[[A_PB,w_min,s_min],[A_PB,w_max,s_min],[A_PB,w_max,s_max],[A_PB,w_min,s_max]]]
    elif plot_sf:
        corner_points_list = [[[a_min,W_SF,s_min],[a_max,W_SF,s_min],[a_max,W_SF,s_max],[a_min,W_SF,s_max]]]
    else:
        raise ValueError("something wrong with sunny_boundaries = {!r}".format(sunny_boundaries))

    boundary_surface_PB = plt3d.art3d.Poly3DCollection(corner_points_list, alpha=0.35)
    boundary_surface_PB.set_color("gray")
    boundary_surface_PB.set_edgecolor("gray")
    ax3d.add_collection3d(boundary_surface_PB)

    # elif boundary == "both":
        # raise NotImplementedError("will be done soon")
        # boundary_surface_both = plt3d.art3d.Poly3DCollection([[[0,.5,0],[0,.5,1],[A_PB,.5,1],[A_PB,.5,0]],
                                                        # [[A_PB,.5,0],[A_PB,1,0],[A_PB,1,1],[A_PB,.5,1]]])
        # boundary_surface_both.set_color("gray"); boundary_surface_both.set_edgecolor("gray"); boundary_surface_both.set_alpha(0.25)
        # ax3d.add_collection3d(boundary_surface_both)
    # else:
        # raise NameError("Unkown boundary {!r}".format(boundary))
    # 
    # if add_outer:
        # # add outer limits of undesirable view from standard view perspective:
        # undesirable_outer_stdview = plt3d.art3d.Poly3DCollection([[[0,0,0],[0,0,1],[0,.5,1],[0,.5,0]],
                                            # [[A_PB,1,0],[aws.A_max,1,0],[aws.A_max,1,1],[A_PB,1,1]],
                                            # [[0,0,0],[0,.5,0],[A_PB,.5,0],[A_PB,1,0],[aws.A_max,1,0],[aws.A_max,0,0]]])
        # undesirable_outer_stdview.set_color("gray"); undesirable_outer_stdview.set_edgecolor("gray"); undesirable_outer_stdview.set_alpha(0.25)
        # ax3d.add_collection3d(undesirable_outer_stdview)




def formatted_value(val):
    fmt = "!r"
    try:
        float(val)
    except (TypeError, ValueError):
        pass
    else:
        fmt = ":4.2e"
    return ("{"+fmt+"}").format(val)

def recursive_difference(x, y):
    if type(x) != type(y):
        raise TypeError("arguments need to be of the same type")
    if isinstance(x, dict):
        changed_pars = {}
        for key, val in x.items():
            if not key in y:
                changed_pars[key] = (val, None)
            else:
                ret = recursive_difference(val, y[key])
                if ret:
                    changed_pars[key] = ret
        return changed_pars
    else:
        changed = ()
        if isinstance(y, np.ndarray):
            if not np.allclose(x, y):
                changed = (x, y)
        elif x != y:
            changed = (x, y)
        return changed

def get_changed_parameters(pars, default_pars):
    return recursive_difference(pars, default_pars)

def print_changed_parameters(pars, default_pars, prefix=""):
    model_changed_pars = get_changed_parameters(pars, default_pars)
    if model_changed_pars:
        if prefix:
            print(prefix)
        for par in sorted(model_changed_pars):
            print(("{} = {} (default: {})").format(par, *map(formatted_value, model_changed_pars[par])))
        print()

def recursive_dict2string(dic, prefix="", spacing=" "*4):
    ret = ""
    for key in sorted(dic):
        assert isinstance(key, str)
        ret += prefix + key + " = "
        if isinstance(dic[key], dict):
            ret += "{\n"
            ret += recursive_dict2string(dic[key], prefix=prefix+spacing, spacing=spacing)
            ret += "}\n"
        else:
            ret += formatted_value(dic[key]) + "\n"
    if ret:
        ret = ret[:-1]
    return ret

def dummy_hook(*args, **kwargs):
    pass

def dummy_isinside(x):
    return True

def follow_indices(starting_indices, *,
        grid, states, paths,
        trajectory_hook=dummy_hook, isinside=dummy_isinside,
        fallback_paths=None,
        verbose = 0
        ):
    if verbose:
        print("starting points and states for paths:")
        for ind in starting_indices:
            print("{!s} --- {:>2}".format(grid[ind], states[ind]))
        print()
    plotted_indices = set()
    if verbose < 2:
        print("following and plotting paths ... ", end="", flush=True)
    for ind in starting_indices:
        if ind in plotted_indices:
            continue
        plotted_indices.add(ind)
        x0 = grid[ind]
        x1 = paths["reached point"][ind]
        if verbose >= 2:
            print("({}| {:>2d}) {} via {} ".format(ind, states[ind], x0, x1), end="")
        if isinside([x0, x1]):
            traj = list(zip(x0, x1))
            trajectory_hook(traj, paths["choice"][ind])
            next_ind = paths["next point index"][ind]
            if next_ind == lv.PATHS_INDEX_DEFAULT and fallback_paths is not None:
                if verbose >= 2:
                    print("FALLBACK ", end="")
                next_ind = fallback_paths["next point index"][ind]
            if next_ind != lv.PATHS_INDEX_DEFAULT:
                if next_ind != ind:
                    if verbose >= 2:
                        print("to ({}| {:>2d}) {}".format(next_ind, states[next_ind], grid[next_ind]))
                    starting_indices.append(next_ind)
                elif verbose >= 2:
                    print("STAYING")
            elif verbose >= 2:
                print("NO INFO")
        elif verbose >= 2:
            print("OUTSIDE")
    if verbose < 2:
        print("done")

def reformat(filename, *, verbose=0):
    """load file and then update header and data"""
    header, data = load_result_file(filename, version_check=False, verbose=verbose)

    # the actually change of the format is done in _reformat
    header, data = _reformat(header, data, verbose=verbose)

    save_result_file(filename, header, data, verbose=verbose)

def save_result_file(fname, header, data, *, verbose=0):
    """save 'header' and 'data' to 'fname'"""
    try:
        _check_format(header, data)
    except AssertionError:
        warn.warn("the generated 'header' and 'data' failed at least one consistency check, saving anyway")

    if verbose:
        print("saving to {!r} ... ".format(fname), end="", flush=True)
    with open(fname, "wb") as f:
        pickle.dump((header, data), f)
    if verbose:
        print("done")

def load_result_file(fname, *, 
                     version_check=True,
                     consistency_check=True,
                     auto_reformat=False,
                     verbose=0
                     ):
    """loads the file 'fname' and performs some checks
    
    note that the options are interdependent: 'auto_reformat' needs 'consistency_check' needs 'version_check'
    """
    if verbose:
        print("loading {} ... ".format(fname), end="", flush=True)
    with open(fname, "rb") as f:
        header, data = pickle.load(f)
    if verbose:
        print("done", flush=True)
    if not version_check:
        return header, data
    if "aws-version-info" in header and header["aws-version-info"] == __version_info__:
        return header, data
    if auto_reformat:
        header, data = _reformat(header, data, verbose=verbose)
        return header, data
    raise IOError("please reformat the file (from version {} to {})".format(versioninfo2version(header.pop("aws-version-info", DEFAULT_VERSION_INFO)), __version__))


DEFAULT_HEADER = {
                "aws-version-info": DEFAULT_VERSION_INFO,
                "model": "AWS",
                "managements": "unknown",
                "boundaries": ["unknown"],
                "grid-parameters": {},
                "model-parameters": {},
                "boundary-parameters": {},
                "start-time": 0,
                "run-time": 0,
                "viab-backscaling-done": None,
                "viab-scaling-vector": None,
                "viab-scaling-offset": None,
                "input-args": None,
                "stepsize": 0.,
                "xstep" : 1.,
                "out-of-bounds": None,
                "remember-paths": False,
                "computation-status": "",
                }


def _check_format(header, data):
    """consistency checks"""

    assert header["aws-version-info"] == __version_info__

    # check the header contains the right keys
    if set(header) != set(DEFAULT_HEADER):
        print("maybe your header was orrupted:")
        new_header_unknown = set(header).difference(DEFAULT_HEADER)
        new_header_missing = set(DEFAULT_HEADER).difference(header.keys())
        if new_header_unknown:
            print("unknown keys: " + ", ".join(new_header_unknown))
        if new_header_missing:
            print("missing keys: " + ", ".join(new_header_missing))
        raise KeyError("header has not the proper key set")


    # keys for data
    data_mandatory_keys = ["grid", "states"]
    data_optional_keys  = ["paths", "paths-lake"]
    # check data contains all necessary keys
    assert set(data_mandatory_keys).issubset(data.keys())
    # check data contains not more than possible keys
    assert set(data_mandatory_keys + data_optional_keys).issuperset(data.keys())
    # check that paths and paths-lake only arise together
    assert len(set(["paths", "paths-lake"]).intersection(data.keys())) in [0, 2]


def _reformat(header, data, verbose=0):
    """updating header and data and check consistency"""

    if verbose:
        print("startin reformatting ... ", end="", flush=True)

    if "aws-version-info" not in header:
        header["aws-version-info"] = (0, 1)

    # 0.1 or no version given
    if header["aws-version-info"] == (0, 1):
        # management has been renamed with the plural
        if "management" in header:
            header["managements"] = header.pop("management")
        if not "boundary-parameters" in header:
            header["boundary-parameters"] = {
                "A_PB": header["model-parameters"].pop("A_PB"),
                "W_SF": header["model-parameters"].pop("W_SF"),
            }

        if "paths" in data and isinstance(data["paths"], tuple):
            new_paths = {}
            new_paths["reached point"] = data["paths"][0]
            new_paths["next point index"] = data["paths"][1]
            new_paths["choice"] = data["paths"][2]
            data["paths"] = new_paths

        new_header = dict(DEFAULT_HEADER)  # copy it in case several files are processed
        new_header.update(header)

        header = new_header

    # 0.2 add paths-lake if paths is given in data
    if header["aws-version-info"] < (0, 2):
        if "paths" in data and not "paths-lake" in data:
            data["paths-lake"] = np.array([])

    # 0.3 add computation-status
    if header["aws-version-info"] < (0, 3):
        header["computation-status"] = ""  # everything ran through

        # always at the last step
        # set the new version-info
        header["aws-version-info"] = __version_info__

    if verbose:
        print("checking consistency of new header and data ... ", end="", flush=True)
    _check_format(header, data)

    if verbose:
        print("done")

    return header, data






ALL_SIGNALS = { x: getattr(signal, x)  for x in dir(signal)
               if x.startswith("SIG")
               and not x.startswith("SIG_")  # because they are just duplicates
               and not getattr(signal, x) == 0  # can register only for signals >0
               and not getattr(signal, x) == 28 # SIGWINCH [28] is sent when resizing the terminal ...
               and not x in ["SIGSTOP", "SIGKILL"]  # can't register these because you can't actually catch them (:
               }
NUMBER_TO_SIGNAL = { val: key for key, val in ALL_SIGNALS.items() }

def signal_handler(sig, frame):
    sys.exit(sig)

def register_signals(sigs = set(ALL_SIGNALS), handler=signal_handler, verbose=True):
    """
    register a signal handler for all given signals
    sigs:       (set-like) providing all the signals to be registered
                default: all possible signals 'ALL_SIGNALS'
    handler:    (function) the signal handler to be used
                default: signal_handler, which just raises a 'sys.exit(sig)' for the signal 'sig'
    verbose:    (bool) print a notification to stderr if the signal registering failed
    """
    sigs = set(sigs)
    # register all possible signals
    for sig in ALL_SIGNALS:
        sigclass = getattr(signal, sig)
        signum = sigclass.value
        # the line below checks whether the signal has been given for
        # registering in the form of either the name, the signal class or the
        # signal number
        if set([sig, sigclass, signum]).intersection(sigs):
            try:
                signal.signal(getattr(signal, sig), signal_handler)
            except Exception as e:
                if verbose:
                    print("ignoring signal registration: [{:>2d}] {} (because {}: {!s})".format(ALL_SIGNALS[sig], sig, e.__class__.__name__, e), file=sys.stderr)



