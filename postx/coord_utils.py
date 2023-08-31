import ephem
import numpy as np
import datetime
from astropy.io import fits as pf
import h5py
from pygdsm import GSMObserver

from astropy.constants import c
from astropy.time import Time
import healpy as hp

from astropy.coordinates import SkyCoord

#SHORTHAND
sin, cos = np.sin, np.cos
SPEED_OF_LIGHT = c.value


def compute_w(xyz: np.array, H: float, d: float, conjugate: bool=False, in_seconds: bool=True) -> np.array:
    """ Compute geometric delay τ, equivalent to w term, for antenna array
    
    Args:
        xyz (np.array): Array of antenna XYZ locations, normally ENU coordinates
        H (float): Hourangle of pointing direction
        d (float): Declination of pointing direction
        conjugate (bool): Return -w if conjugate is true (default False)
        in_seconds (bool): Return data in seconds (True, default) or in meters (False)
    
    Returns:
        w (np.array): Array of w (geometric delay) terms, one per antenna
    """
    x, y, z = np.split(xyz, 3, axis=1)
    sh, sd = sin(H), sin(d)
    ch, cd = cos(H), cos(d)
    w  = cd * ch * x - cd * sh * y + sd * z
    w = -w if conjugate else w
    w = w / SPEED_OF_LIGHT if in_seconds else w
    return w.squeeze()


def generate_phase_vector(xyz: np.array, H: float, d: float, f: float, conj: bool=False) -> np.array:
    """ Generate e(2πi w f) phase vector for given antenna array + pointing direction
    
    First, computes the geometric delay (w term), then converts to a phase vector

    Args:
        xyz (np.array): Array of antenna XYZ locations, normally ENU coordinates
        H (float): Hourangle of pointing direction
        d (float): Declination of pointing direction
        conjugate (bool): Return -w if conjugate is true (default False)
    
     Returns:
        c0 (np.array): Array of phase delay terms, one per antenna       
    """
    w  = compute_w(xyz, H, d)
    c0 = phase_vector(w, f, conj)
    return c0

def phase_vector(w: np.array, f: float, conj: bool=False) -> np.array:
    """ Compute Nx1 phase weight vector e(2πi w f) """
    c0 = np.exp(1j * 2 * np.pi * w * f)
    c0 = np.conj(c0) if conj else c0
    return c0

def make_source(name: str, ra: str, dec: str, flux: float=1.0, epoch: int=2000) -> ephem.FixedBody:
    """ Create a pyEphem FixedBody radio source
    Args:
        name (str):   Name of source, e.g. CasA
        ra (str):     hh:mm:ss right ascension, e.g. 23:23:26
        dec (str):    dd:mm:ss declination e.g. 58:48:22.21
        flux (float): flux brightness in Jy
        epoch (int):  Defaults to J2000, i.e. 2000
    Returns:
        body (pyephem.FixedBody): Source as a pyephem fixed body
    """
    line = "%s,f,%s,%s,%s,%d"%(name,ra,dec,flux,epoch)
    body = ephem.readdb(line)
    return body


def skycoord_to_ephem(x: SkyCoord, name: str='', flux: float=1, epoch: int=2000) -> ephem.FixedBody:
    """ Convert an astropy SkyCoord to a pyephem FixedBody"""
    ra  = x.ra.to('hourangle').to_string(sep=':')
    dec = x.dec.to_string(sep=':')
    line = "%s,f,%s,%s,%s,%d"%(name,ra,dec,flux,epoch)
    body = ephem.readdb(line)
    return body

def ephem_to_skycoord(x: ephem.FixedBody) -> SkyCoord:
    s = SkyCoord(x.ra, x.dec, unit=('rad', 'rad'))
    return s

def test_ephem_skycoord(sc_array):
    """ Test routine for SkyCoord <-> FixedBody """
    for ii in range(len(sc_array)):
        _e0 = skycoord_to_ephem(sc[ii])
        _s0 = ephem_to_skycoord(_e0)
        assert np.isclose(_s0.ra, sc[ii].ra)
        assert np.isclose(_s0.dec, sc[ii].dec)

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
    DEC_rad = src.icrs.dec.to('rad').value
    RA_rad  = src.icrs.ra.to('rad').value

    RA_delta_rad = RA_rad - zen.ra.to('rad').value
    DEC_rad_0 = zen.icrs.dec.to('rad').value

    l = np.cos(DEC_rad) * np.sin(RA_delta_rad)
    m = (np.sin(DEC_rad) * np.cos(DEC_rad_0) - np.cos(DEC_rad) * np.sin(DEC_rad_0) * np.cos(RA_delta_rad))
    n = np.sqrt(1 - l**2 - m**2)
    lmn = np.column_stack((l,m,n))
    return lmn