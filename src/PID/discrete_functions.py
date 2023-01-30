"""
Utility functions to calculate the top node of the PID lattice in large(-ish)
systems, without having to compute all other nodes. Implements Williams &
Beer's I_min and Barrett's MMI.

Pedro Mediano, Jun 2021
"""
import numpy as np
import dit
from dit.pid.measures.imin import s_i

def imin_syn(d, get_MI = False):
    """
    Computes the top node of the PID lattice for dit.Distribution d using
    Williams & Beer's I_min PID function.

    Parameters
    ----------
    d : dit.Distribution
        The joint distribution of all sources and targets. The function assumes
        the target is the last variable in the distribution, and each other
        variable is considered a separate source.

    Returns
    -------
    syn : float
        The partial information value of the top node of the PID lattice.
    """
    target = d.rvs[-1]
    p_s = d.marginal(target)
    mi = dit.shannon.mutual_information(d, list(range(len(d.alphabet)-1)), target)
    imax = sum(p_s[s] * max(s_i(d, np.setdiff1d(d.rvs[:-1], source), target, s) for source in d.rvs[:-1]) for s in p_s.outcomes)
    if get_MI:
        return mi - imax, mi
    return mi - imax


def immi_syn(d):
    """
    Computes the top node of the PID lattice for dit.Distribution d using
    Barrett's I_mmi PID function.

    Parameters
    ----------
    d : dit.Distribution
        The joint distribution of all sources and targets. The function assumes
        the target is the last variable in the distribution, and each other
        variable is considered a separate source.

    Returns
    -------
    syn : float
        The partial information value of the top node of the PID lattice.
    """
    target = d.rvs[-1]
    mi = dit.shannon.mutual_information(d, list(range(len(d.alphabet)-1)), target)
    imax = max(dit.shannon.mutual_information(d, np.setdiff1d(d.rvs[:-1], source), target) for source in d.rvs[:-1])
    return mi - imax

def imin_red(d, get_MI = False):
    """
    Computes the bottom node of the PID lattice for dit.Distribution d using
    Williams & Beer's I_min PID function.

    Parameters
    ----------
    d : dit.Distribution
        The joint distribution of all sources and targets. The function assumes
        the target is the last variable in the distribution, and each other
        variable is considered a separate source.

    Returns
    -------
    red : float
        The partial information value of the top node of the PID lattice.
    """
    target = d.rvs[-1]
    p_s = d.marginal(target)
    mi = dit.shannon.mutual_information(d, list(range(len(d.alphabet)-1)), target)
    imin = sum(p_s[s] * min(s_i(d, source, target, s) for source in d.rvs[:-1]) for s in p_s.outcomes)
    if get_MI:
        return imin, mi
    return imin


def immi_red(d):
    """
    Computes the bottom node of the PID lattice for dit.Distribution d using
    Barrett's I_mmi PID function.

    Parameters
    ----------
    d : dit.Distribution
        The joint distribution of all sources and targets. The function assumes
        the target is the last variable in the distribution, and each other
        variable is considered a separate source.

    Returns
    -------
    red : float
        The partial information value of the top node of the PID lattice.
    """
    target = d.rvs[-1]
    mmi = min(dit.shannon.mutual_information(d, source, target) for source in d.rvs[:-1])
    return mmi


if __name__ == '__main__':
    # Some quick tests against the dit PID implementations in small systems
    nb_runs = 20
    for _ in range(nb_runs):
        L = np.random.randint(3, 6)
        d = dit.distconst.random_distribution(L, 2)

        # I_min
        true_syn  = dit.pid.PID_WB(d)[(tuple(range(L-1)),)]
        estim_syn = imin_syn(d)
        assert(np.isclose(true_syn, estim_syn))

        # I_mmi
        true_syn  = dit.pid.PID_MMI(d)[(tuple(range(L-1)),)]
        estim_syn = immi_syn(d)
        assert(np.isclose(true_syn, estim_syn))
