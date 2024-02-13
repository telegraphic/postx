import numpy as np
import healpy as hp

from astropy.constants import c
from astropy.coordinates import SkyCoord, EarthLocation

import pyuvdata.utils as uvutils

#SHORTHAND
sin, cos = np.sin, np.cos
SPEED_OF_LIGHT = c.value


def phase_vector(w: np.array, f: float, conj: bool=False, dtype='complex64') -> np.array:
    """ Compute Nx1 phase weight vector e(2πi w f) """
    c0 = np.exp(1j * 2 * np.pi * w * f, dtype=dtype)
    c0 = np.conj(c0) if conj else c0
    return c0


def hpix2sky(nside: int, pix_ids: np.ndarray) -> SkyCoord:
    """ Convert a healpix pixel_id into a SkyCoord

    Args:
        nside (int): Healpix NSIDE parameter
        pix_ids (np.array): Array of pixel IDs

    Returns:
        sc (SkyCoord): Corresponding SkyCoordinates
    """
    gl, gb = hp.pix2ang(nside, pix_ids, lonlat=True)
    sc = SkyCoord(gl, gb, frame='galactic', unit=('deg', 'deg'))
    return sc


def sky2hpix(nside: int, sc: SkyCoord) -> np.ndarray:
    """ Convert a SkyCoord into a healpix pixel_id

    Args:
        nside (int): Healpix NSIDE parameter
        sc (SkyCoord): Astropy sky coordinates array

    Returns:
        pix (np.array): Array of healpix pixel IDs
    """
    gl, gb = sc.galactic.l.to('deg').value, sc.galactic.b.to('deg').value
    pix = hp.ang2pix(nside, gl, gb, lonlat=True)
    return pix


def test_pix2sky():
    """ Small test routine for converting healpix pixel_id to and from SkyCoords """
    NSIDE = 32
    pix = np.arange(hp.nside2npix(NSIDE))
    sc  = hpix2sky(NSIDE, pix)
    pix_roundtrip = sky2hpix(NSIDE, sc)
    assert np.allclose(pix, pix_roundtrip)


def skycoord_to_lmn(src: SkyCoord, zen: SkyCoord) -> np.array:
    """ Calculate lmn coordinates for a SkyCoord, given current zenith

    Args:
        src (SkyCoord): SkyCoord of interest (can be SkyCoord vector of length N)
        zen (SkyCoord): Location of zenith

    Returns:
        lmn (np.array): Nx3 array of l,m,n values

    Notes:
        l = cos(DEC) sin(ΔRA)
        m = sin(DEC) cos(DEC0) - cos(DEC) sin(DEC0) cos(ΔRA)
        n = sqrt(1 - l^2 - m^2)

        Following Eqn 3.1 in
        http://math_research.uct.ac.za/~siphelo/admin/interferometry/3_Positional_Astronomy/3_4_Direction_Cosine_Coordinates.html
    """
    if src.frame.name == 'galactic':
        src = src.icrs

    DEC_rad = src.dec.to('rad').value
    RA_rad  = src.ra.to('rad').value

    RA_delta_rad = RA_rad - zen.icrs.ra.to('rad').value
    DEC_rad_0 = zen.icrs.dec.to('rad').value

    l = np.cos(DEC_rad) * np.sin(RA_delta_rad)
    m = (np.sin(DEC_rad) * np.cos(DEC_rad_0) - np.cos(DEC_rad) * np.sin(DEC_rad_0) * np.cos(RA_delta_rad))
    n = np.sqrt(1 - l**2 - m**2)
    lmn = np.column_stack((l,m,n))
    return lmn


def loc_xyz_ECEF_to_ENU(loc: EarthLocation, xyz: np.ndarray):
    """ Convert EarthLocation and antenna array in ECEF format to ENU

    Args:
        loc (EarthLocation): Astropy EarthLocation or array center
        xyz (np.array): ECEF XYZ coordinates (with array centroid subtracted)
                        i.e. xyz = XYZ_ECEF - XYZ_ECEF0

    """
    loc_xyz = list(loc.value)
    loc = loc.to_geodetic()
    enu = uvutils.ENU_from_ECEF(xyz + loc_xyz, loc.lat.to('rad'), loc.lon.to('rad'), loc.height)
    return loc, enu


def generate_lmn_grid(n_pix: int, abs_max: int=1, nan_below_horizon: bool=True):
    """ Generate a grid of direction cosine unit vectors

    Generates a square lmn grid across l=(-abs_max, abs_max), m=(-abs_max, abs_max).

    Notes:
        For unit pointing vector n = sqrt(1 - l^2 - m^2)
        If sqrt(l^2 + m^2) > 1, pointing is below horizon (unphysical)

    Args:
        n_pix (int): Number of pixels in grid (n_pix)
        abs_max (int): Maximum absolute values for l and m (default 1).
        nan_below_horizon (bool): If True, n is NaN below horizon.
                                    If False, n is 0 below horizon


    Returns:
        lmn (np.array): A (n_pix, n_pix, 3) array of (l,m,n) values.
    """
    l = np.linspace(abs_max, -abs_max, n_pix, dtype='float32')
    m = np.linspace(-abs_max, abs_max, n_pix, dtype='float32')
    lg, mg = np.meshgrid(l, m)

    lmn = np.zeros((n_pix, n_pix, 3), dtype='float32')
    lmn[:, :, 0] = lg
    lmn[:, :, 1] = mg
    if nan_below_horizon:
        ng     = np.sqrt(1 - lg**2 - mg**2)
        lmn[:, :, 2] = ng
    else:
        lmn[:, :, 2] = 0
    return lmn

def gaincal_vec_to_matrix(gc: np.array) -> np.array:
    """ Create a gain calibration matrix out of 4-pol vector

    Args:
        gc (np.array): Per-antenna gain calibration solutions.
                       Shape (N_ant, N_pol=2) dtype complex

    Returns:
        cal_mat (np.array): Per-visibility gain calibration solutions,
                            Shape (N_ant, N_ant, N_pol=4)
    """
    cal_mat = np.zeros((gc.shape[0], gc.shape[0], 4), dtype='complex64')

    cal_mat[..., 0] = np.outer(gc[..., 0], gc[..., 0])
    cal_mat[..., 1] = np.outer(gc[..., 0], gc[..., 1])
    cal_mat[..., 2] = np.outer(gc[..., 1], gc[..., 0])
    cal_mat[..., 3] = np.outer(gc[..., 1], gc[..., 1])
    return cal_mat
