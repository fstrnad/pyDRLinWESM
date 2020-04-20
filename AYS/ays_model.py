"""
adapted from https://github.com/timkittel/ays-model/
"""
from __future__ import division, print_function

from AYS.ays_general import __version__, __version_info__
import pyviability as pv

import numpy as np
import warnings as warn
import sys

if sys.version_info[0] < 3:
    warn.warn("this code has been tested in Python3 only", category=DeprecationWarning)

if pv.version_info < (0, 2, 0):
    warn.warn(f"please get the correct version (0.2.0) of pyviability (not {pv.version})")
# assert pv.version_info == (0, 2, 0), "please get the correct version (0.2.0) of pyviability (and don't forget to (re)run the installation)"


NB_USING_NOPYTHON = True
USING_NUMBA = True
if USING_NUMBA:
    try:
        import numba as nb
    except ImportError:
        warn.warn("couldn't import numba, continuing without", ImportWarning)
        USING_NUMBA = False


if USING_NUMBA:
    jit = nb.jit
else:
    def dummy_decorator_with_args(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        else:
            return dummy_decorator_with_args
    jit = dummy_decorator_with_args




# long name (command option line style) : short name (lower case)
DEFAULT_NAME = "default"
MANAGEMENTS = {
    "degrowth": "dg",
    "solar-radiation": "srm",
    "energy-transformation": "et",
    "carbon-capture-storage": "ccs",
}

def get_management_parameter_dict(management, all_parameters):
    management_dict = dict(all_parameters) # make a copy
    if management == DEFAULT_NAME:
        return management_dict
    ending = "_" + MANAGEMENTS[management].upper()
    changed = False
    for key in management_dict:
        # choose the variables that are changed by the ending
        if key+ending in management_dict:
            changed = True
            management_dict[key] = management_dict[key+ending]
    if not changed:
        raise NameError("didn't find any parameter for management option "\
                        "'{}' (ending '{}')".format(management, ending))
    return management_dict


AYS_parameters = {}
AYS_parameters["A_offset"] = 600  # pre-industrial level corresponds to A=0

AYS_parameters["beta"] = 0.03  # 1/yr
AYS_parameters["beta_DG"] = AYS_parameters["beta"] / 2
AYS_parameters["epsilon"] = 147.  # USD/GJ
AYS_parameters["rho"] = 2.  # 1
AYS_parameters["phi"] = 47.e9  # GJ/GtC
AYS_parameters["phi_CCS"] = AYS_parameters["phi"] * 4 / 3 # 25% carbon taken away in form oc co2 from the system
AYS_parameters["sigma"] = 4.e12  # GJ
AYS_parameters["sigma_ET"] = AYS_parameters["sigma"] * .5 ** (1 / AYS_parameters["rho"])
AYS_parameters["tau_A"] = 50.  # yr
AYS_parameters["tau_S"] = 50.  # yr
AYS_parameters["theta"] = AYS_parameters["beta"] / (950 - AYS_parameters["A_offset"])  # 1/(yr GJ)
AYS_parameters["theta_SRM"] = 0.5 * AYS_parameters["theta"]

boundary_parameters = {}
boundary_parameters["A_PB"] = 945 - AYS_parameters["A_offset"]  # 450ppm
# boundary_parameters["A_PB_350"] = 735 - AYS_parameters["A_offset"]
boundary_parameters["W_SF"] = 4e13  # USD, year 2000 GWP

grid_parameters = {}

current_state = [240, 7e13, 5e11]

# rescaling parameters
grid_parameters["A_mid"] = current_state[0]
grid_parameters["W_mid"] = current_state[1]
grid_parameters["S_mid"] = current_state[2]

grid_parameters["n0"] = 40
grid_parameters["grid_type"] = "orthogonal"
border_epsilon = 1e-3

grid_parameters["boundaries"] = np.array([[0, 1],  # a: rescaled A
                [0, 1],  # w: resclaed W
                [0, 1]  # s: rescaled S
                ], dtype=float)

# use the full stuff in the S direction
grid_parameters["boundaries"][:2, 0] = grid_parameters["boundaries"][:2, 0] + border_epsilon
grid_parameters["boundaries"][:2, 1] = grid_parameters["boundaries"][:2, 1] - border_epsilon


def globalize_dictionary(dictionary, module="__main__"):
    if isinstance(module, str):
        module = sys.modules[module]

    for key, val in dictionary.items():
        if hasattr(module, key):
            warn.warn("overwriting global value / attribute '{}' of '{}'".format(key, module.__name__))
        setattr(module, key, val)


# JH: maybe transform the whole to log variables since W,S can go to infinity...
def _AYS_rhs(AYS, t=0, beta=None, epsilon=None, phi=None, rho=None, sigma=None, tau_A=None, tau_S=None, theta=None):
    A, W, S = AYS
    U = W / epsilon
    F = U / (1 + (S/sigma)**rho)
    R = U - F
    E = F / phi
    Adot = E - A / tau_A
    Wdot = (beta - theta * A) * W
    Sdot = R - S / tau_S
    return Adot, Wdot, Sdot


AYS_rhs = nb.jit(_AYS_rhs, nopython=NB_USING_NOPYTHON)
# AYS_rhs = _AYS_rhs  # used for debugging


#@jit(nopython=NB_USING_NOPYTHON)
def AYS_rescaled_rhs(ays, t=0, beta=None, epsilon=None, phi=None, rho=None, sigma=None, tau_A=None, tau_S=None, theta=None):
    a, y, s = ays
    # A, y, s = Ays

    s_inv = 1 - s
    s_inv_rho = s_inv ** rho
    K = s_inv_rho / (s_inv_rho + (S_mid * s / sigma) ** rho )

    a_inv = 1 - a
    w_inv = 1 - y
    Y = W_mid * y / w_inv
    A = A_mid * a / a_inv
    adot = K / (phi * epsilon * A_mid) * a_inv * a_inv * Y - a * a_inv / tau_A
    ydot = y * w_inv * ( beta - theta * A )
    sdot = (1 - K) * s_inv * s_inv * Y / (epsilon * S_mid) - s * s_inv / tau_S

    return adot, ydot, sdot


# @jit(nopython=NB_USING_NOPYTHON)
def AYS_sunny_PB(ays):
    return ays[:, 0] < A_PB / (A_PB + A_mid) # transformed A_PB  # planetary boundary

# @jit(nopython=NB_USING_NOPYTHON)
def AYS_sunny_SF(ays):
    return ays[:, 1] > W_SF / (W_SF + W_mid) # transformed W_SF  # social foundation

# @jit(nopython=NB_USING_NOPYTHON)
def AYS_sunny_PB_SF(ays):
    return np.logical_and(ays[:, 0] < A_PB / (A_PB + A_mid), ays[:, 1] > W_SF / (W_SF + W_mid)) # both transformed









