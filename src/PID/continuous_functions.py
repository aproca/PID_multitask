"""
Utility functions to calculate the top and bottom nodes of the PID lattice in
large(-ish) systems, without having to compute all other nodes. Implements
Barrett's MMI measure for Gaussian distributions, using Ince's Gaussian copula
transformation.

References:

    Barrett AB (2015). Exploration of synergistic and redundant information
    sharing in static and dynamical Gaussian systems. Physical Review E.

    Ince, R et al. (2017). A statistical framework for neuroimaging data
    analysis based on mutual information estimated via a gaussian copula. Human
    Brain Mapping, 38(3), 1541-1573.

Pedro Mediano, Mar 2022
"""
import numpy as np
from src.PID.gauss_rank_scaler import GaussRankScaler

__all__ = ['gc_immi_syn', 'gc_immi_red']

def _mi(S, src, tgt):
    """
    Internal method to compute mutual information between Gaussian variables.

    Parameters
    ----------
    S : np.ndarray
        Covariance matrix. Must be square and positive definite.
    src, tgt : iterables
        Iterables containing the indices of source and target variables in S.

    Returns
    -------
    mi : float
    """
    allidx = [s for s in src] + [t for t in tgt]
    Hx  = 0.5*np.linalg.slogdet(S[np.ix_(src, src)])[1]
    Hy  = 0.5*np.linalg.slogdet(S[np.ix_(tgt, tgt)])[1]
    Hxy = 0.5*np.linalg.slogdet(S[np.ix_(allidx, allidx)])[1]
    mi = Hx + Hy - Hxy

    return mi


def gc_immi_syn(X, src, tgt, get_MI = False):
    """
    Computes the top node of the PID lattice for a dataset X using Barrett's
    I_mmi PID function, and Ince's Gaussian copula estimator.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Data matrix.
    src : iter of iters
        Iterable of iterables, each containing the indices of a source variable
        in the columns of X.
    tgt : iterable
        Iterable containing the indices of the target variables in the columns
        of X.

    Returns
    -------
    syn : float
    """
    scaler = GaussRankScaler()
    sX = scaler.fit_transform(X)
    S = np.corrcoef(sX.T)

    allsrc = []
    for s in src:
        allsrc.extend([a for a in s])
    mi = _mi(S, allsrc, tgt)
    imax = max([_mi(S, np.setdiff1d(allsrc, s), tgt) for s in src])

    if get_MI:
        return mi - imax, mi
    return mi - imax


def gc_immi_red(X, src, tgt, get_MI = False):
    """
    Computes the bottom node of the PID lattice for a dataset X using Barrett's
    I_mmi PID function, and Ince's Gaussian copula estimator.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Data matrix.
    src : iter of iters
        Iterable of iterables, each containing the indices of a source variable
        in the columns of X.
    tgt : iterable
        Iterable containing the indices of the target variables in the columns
        of X.

    Returns
    -------
    red : float
    """
    scaler = GaussRankScaler()
    sX = scaler.fit_transform(X)
    S = np.corrcoef(sX.T)

    red = min([_mi(S, s, tgt) for s in src])

    if get_MI:
        allsrc = []
        for s in src:
            allsrc.extend([a for a in s])
        mi = _mi(S, allsrc, tgt)
        return red, mi
    return red
