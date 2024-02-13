"""
Simple sky model class for ephemeris using pyephem
"""
import ephem
import numpy as np

from astropy.coordinates import SkyCoord, get_body, get_sun
from .ant_array import RadioArray

class RadioSource(SkyCoord):
    """ A SkyCoordinate with a magnitude """
    def __init__(self, *args, mag=1.0, unit=None, **kwargs):
        if unit is None:
            unit=('hourangle', 'degree')
        super().__init__(*args, unit=unit, **kwargs)
        self.mag = mag

def generate_skycat(observer: RadioArray):
    """ Generate a SkyModel for a given observer with major radio sources

    Args:
        observer (AntArray / ephem.Observer): Observatory instance

    Returns:
        skycat (SkyModel): A sky catalog with the A-team sources
    """
    skycat = {
        'Virgo_A':     RadioSource('12h 30m 49s',    '+12:23:28',    ),
        'Hydra_A':     RadioSource('09h 18m 5.6s',   '-12:5:44.0',   ),
        'Centaurus_A': RadioSource('13h 25m 27.6s',  '−43:01:09',    ),
        'Pictor_A':    RadioSource('05h 19m 49.72s', '−45:46:43.85', ),
        'Hercules_A':  RadioSource('16h 51m 08.15',  '+04:59:33.32', ),
        'Fornax_A':    RadioSource('03h 22m 41.7',   '−37:12:30',    ),
    }
    skycat.update(generate_skycat_solarsys(observer))
    return skycat

def generate_skycat_solarsys(observer: RadioArray):
    """ Generate Sun + Moon for observer """
    sun_gcrs  = get_body('sun', observer._ws('t'))
    moon_gcrs = get_body('moon', observer._ws('t'))
    skycat = {
        'Sun': RadioSource(sun_gcrs.ra, sun_gcrs.dec, mag=1.0),
        'Moon': RadioSource(moon_gcrs.ra, moon_gcrs.dec, mag=1.0),
    }
    return skycat

def sun_model(aa, t_idx=0) -> np.array:
    """ Generate sun flux model at given frequencies.

    Flux model values taken from Table 2 of Macario et al (2022).
    A 5th order polynomial is used to interpolate between frequencies.

    Args:
        aa (RadioArray): RadioArray to use for ephemeris / freq setup
        t_idx (int): timestep to use

    Returns:
        S (RadioSource): Model flux, in Jy

    Citation:
        Characterization of the SKA1-Low prototype station Aperture Array Verification System 2
        Macario et al (2022)
        JATIS, 8, 011014. doi:10.1117/1.JATIS.8.1.011014
        https://ui.adsabs.harvard.edu/abs/2022JATIS...8a1014M/abstract
    """
    f_i = (50, 100, 150, 200, 300)        # Frequency in MHz
    α_i = (2.15, 1.86, 1.61, 1.50, 1.31)  # Spectral index
    S_i = (5400, 24000, 5100, 81000, 149000)    # Flux in Jy

    p_S = np.poly1d(np.polyfit(f_i, S_i, 2))
    sun = RadioSource(get_sun(aa.workspace['t']), mag=p_S(aa.workspace['f'].to('MHz').value))

    return sun