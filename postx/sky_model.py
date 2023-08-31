"""
Simple sky model class for ephemeris using pyephem
"""
import ephem
import numpy as np


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


class SkyModel(object):
    """ Simple sky model class """
    def __init__(self, sources=[], date=None):
        self.sources = []
        self.read_sourcelist(sources)
        self.date    = date

    def __repr__(self):
        N = len(self.sources)
        #ret = f"<SkyModel with {N} sources> \n"
        ret = "<SkyModel with {0} sources> \n".format(N)
        ret += str(self.sources)
        return ret

    def __str__(self):
        return self.__repr__()

    def add_source(self, src):
        """ Add new source to sky model
        Args:
            src (ephem.FixedBody): Sky source to add
        """
        self.sources.append(src)

    def read_sourcelist(self, srclist):
        """ Read a list of sources, convert to FixedBody objects
        Args:
            srclist (list): A list of sources, each entry a string:
                           [name, ra, dec, flux]
        """
        for src in srclist:
            sobj = make_source(*src)
            self.add_source(sobj)

    def compute_ephemeris(self, observatory):
        """ Compute RA/DEC for all sources for given observatory
        Args:
            observatory (AntArray): pyephem Observer, or AntArray
        """
        self.date = observatory.date
        for ii in range(len(self.sources)):
            self.sources[ii].compute(observatory)

    def report(self):
        """ Report RA/DEC locations and alt/az """
        print("Ephemeris Date: {0}".format(self.date))
        print("SRC          RA           DEC        ALT          AZ")
        for src in self.sources:
            print("{0:12} {1}  {2}  {3}  {4}".format(src.name, src.ra, src.dec, src.alt, src.az))

    def getaltaz(self):
        """ Return alt/az for the first source in degrees"""
        return self.sources[0].alt*180/np.pi, self.sources[0].az*180/np.pi


def make_sky_model(sources, date=None):
    return SkyModel(sources, date)