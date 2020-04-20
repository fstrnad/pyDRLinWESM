import numpy as np

@np.vectorize
def std_err_bin_coefficient(n,p):
    std_dev=p*(1-p)
    return np.sqrt(std_dev/n)
