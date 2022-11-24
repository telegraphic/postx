import ephem
import numpy as np
import datetime
from astropy.io import fits as pf
import h5py
from pygdsm import GSMObserver

from astropy.constants import c
from astropy.time import Time

#SHORTHAND
sin, cos = np.sin, np.cos
SPEED_OF_LIGHT = c.value

def polar_to_cartesian(θ,ϕ):
    x = np.sin(θ) * np.cos(ϕ)
    y = np.sin(θ) * np.sin(ϕ)
    z = np.cos(θ)   
    return x, y, z

def compute_w(xyz, H, d, conjugate=False, in_seconds=True):
    """ Compute geometric delay τ_g, equivalent to w term """
    x, y, z = np.split(xyz, 3, axis=1)
    sh, sd = sin(H), sin(d)
    ch, cd = cos(H), cos(d)
    w  = cd * ch * x - cd * sh * y + sd * z
    w = -w if conjugate else w
    w = w / SPEED_OF_LIGHT if in_seconds else w
    return w.squeeze()

def generate_phase_vector(xyz, H, d, f, conj=False):
    w  = compute_w(xyz, H, d)
    c0 = phase_vector(w, f, conj)
    return c0

def phase_vector(w, f, conj=False):
    """ Compute Nx1 phase weight vector """
    c0 = np.exp(1j * 2 * np.pi * w * f)
    c0 = np.conj(c0) if conj else c0
    return c0

def make_source(name, ra, dec, flux=1.0, epoch=2000):
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

from astropy.coordinates import SkyCoord

def skycoord_to_ephem(x, name='', flux=1, epoch=2000):
    ra  = x.ra.to('hourangle').to_string(sep=':')
    dec = x.dec.to_string(sep=':')
    line = "%s,f,%s,%s,%s,%d"%(name,ra,dec,flux,epoch)
    body = ephem.readdb(line)
    return body

def ephem_to_skycoord(x):
    s = SkyCoord(x._ra, x._dec, unit=('rad', 'rad'))
    return s

def test_ephem_skycoord(sc_array):
    for ii in range(len(sc_array)):
        _e0 = skycoord_to_ephem(sc[ii])
        _s0 = ephem_to_skycoord(_e0)
        assert np.isclose(_s0.ra, sc[ii].ra)
        assert np.isclose(_s0.dec, sc[ii].dec)