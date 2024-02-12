# coding=utf-8
"""
Python implementation of StefCal calibration algorithm.

From https://github.com/OxfordSKA/bda/blob/master/pybda/stefcal.py
"""


import numpy as np
import numpy.linalg


def stefcal(a: np.array, b: np.array, tol: float=1.0e-8, niter: int=50, gstart: np.array=None) -> tuple:
    """
    Python implementation of StefCal calibration algorithm.

    Description:
        Minimize
            ``|| a - G' * b * G ||_F``
        where
            ``G = diag(g)``

    Args:
        a (array_like): Observed visibilities (full matrix)
        b (array_like): Model visibilities (full matrix)
        tol (float): Required tolerance (eg. 1.0e-8)
        niter (int): Maximum number of iterations (eg. 50)
        gstart (array_like): Initial values for g. If not present, set g = 1
                             Useful as a starting point if, for example, gains
                             at a previous time are known.

    Returns:
        tuple of g, nit, dg

        g (array_like): computed gains, g[i] corresponds to the ith receiver.
        nit (int): number of iterations required.
        dg (float): convergence achieved.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    n = a.shape[0]

    if gstart is not None:
        g = np.asarray(gstart)
    else:
        g = np.ones(n, np.complex128)

    dg = 1.0e30
    nit = niter
    omega = 0.5
    f0 = 1 - omega
    f1 = omega

    for i in range(niter):
        g_old = np.copy(g)
        for j in range(n):
            z = np.conj(g_old) * b[:, j]
            scalefactor = np.dot(np.conj(z), z)
            if scalefactor == 0:
                scalefactor = 1
            g[j] = np.dot(np.conj(z), a[:, j]) / scalefactor

        if i < 2:
            dg = np.linalg.norm(g - g_old) / np.linalg.norm(g)
            if dg <= tol:
                nit = i
                break
        elif i % 2 == 1:
            dg = np.linalg.norm(g - g_old) / np.linalg.norm(g)
            if dg <= tol:
                nit = i
                break
            else:
                g = f0 * g + f1 * g_old

    p = np.conj(g[0]) / np.abs(g[0])
    g = p * g

    return g, nit, dg