import numpy as np
import pylab as plt
from .ant_array import RadioArray
from astropy.time import Time
import ephem
import h5py
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

def generate_skycat(observer: ephem.Observer):
    """ Generate a SkyModel for a given observer with major radio sources
    
    Args:
        observer (AntArray / ephem.Observer): Observatory instance
    
    Returns:
        skycat (SkyModel): A sky catalog with the A-team sources 
    """
    skycat = {
        'Virgo_A': SkyCoord('12h 30m 49s', '+12:23:28', unit=('hourangle', 'degree')),
        'Hydra_A': SkyCoord('09h 18m 5.6s', '-12:5:44.0',  unit=('hourangle', 'degree')),
        'Centaurus_A':  SkyCoord('13h 25m 27.6s', '−43:01:09', unit=('hourangle', 'degree')),
        'Pictor_A': SkyCoord('05h 19m 49.721s', '−45:46:43.85', unit=('hourangle', 'degree')),
        'Hercules_A': SkyCoord('16h 51m 08.15', '+04:59:33.32', unit=('hourangle', 'degree')),
        'Fornax_A': SkyCoord('03h 22m 41.7', '−37:12:30', unit=('hourangle', 'degree')),

    }
    skycat.update(generate_skycat_solarsys(observer))
    return skycat

def generate_skycat_solarsys(observer):
    """ Generate Sun + Moon for observer """
    sun  = ephem.Sun()
    moon = ephem.Moon()
    sun.compute(observer)
    moon.compute(observer) 
    skycat = {
        'Sun': SkyCoord(sun.ra, sun.dec, unit=('rad', 'rad')),
        'Moon':  SkyCoord(moon.ra, moon.dec, unit=('rad', 'rad'))
    }
    return skycat

class AllSkyViewer(object):
    """ An all-sky imager based on matplotlib imshow with WCS support 
    
    Provides the following:
        update() - updates WCS information
        get_pixel() - get pixel information
    """
    def __init__(self, observer=None, ts=None, f_mhz=None, n_pix=128):
        self.observer = observer
        self.wcsd     = None
        self.skycat = {}
        
        self.name = observer.name if hasattr(observer, 'name') else 'allsky'

        self.ts     = ts
        self.f_mhz  = f_mhz
        self.n_pix  = n_pix
        
        # Internal indexes 
        self._f_idx   = 0   # Frequency axis 
        self._p_idx   = 0   # pol axis
    
    def _update_wcs(self):
        
        self.observer.date = self.ts.datetime
        zen_ra, zen_dec = self.observer.radec_of(0, np.pi/2)
        
        self.wcsd = {
                 'SIMPLE': 'T',
                 'NAXIS': 2,
                 'NAXIS1': self.n_pix,
                 'NAXIS2': self.n_pix,
                 'CTYPE1': 'RA---SIN',
                 'CTYPE2': 'DEC--SIN',
                 'CRPIX1': self.n_pix // 2 + 1,
                 'CRPIX2': self.n_pix // 2 + 1,
                 'CRVAL1': np.rad2deg(zen_ra),
                 'CRVAL2': np.rad2deg(zen_dec),
                 'CDELT1': -360/np.pi / self.n_pix,
                 'CDELT2': 360/np.pi / self.n_pix  
            }
        
        for src in self.skycat.keys():
            if src in ('Sun', 'Moon'):
                self.skycat[src] = generate_skycat_solarsys(self.observer)[src]
        self.wcs = WCS(self.wcsd)
    
    def update(self, ts: Time=None, n_pix: int=None, f_mhz: float=None):
        """ Update WCS information on timestamp or other change 
        
        Args:
            ts (astropy.Time): New timestamp to use
            n_pix (int): Change number of pixels in image
            f_mhz (int): Change observing frequency
        """
        if ts is not None:
            self.ts = ts
        if n_pix is not None:
            self.n_pix = n_pix
        if f_mhz is not None:
            self.f_mhz = f_mhz
        self._update_wcs()
    
    def get_pixel(self, src: SkyCoord, f_idx: int=None) -> tuple:
        """ Return the pixel index for a given SkyCoord 
        
        Args:
            src (SkyCoord): sky coordinate of interest
            f_idx (int): Frequency index. TODO: remember why I added this arg? Potentially remove.
        """
        if f_idx is not None:
            self._f_idx = f_idx
        
        self._update_wcs()
        x, y = self.wcs.world_to_pixel(src)
        if ~np.isnan(x) and ~np.isnan(y):
            i, j = int(np.round(x)), int(np.round(y))
            return (i, j)
        else:
            return (0, 0)
        
    def load_skycat(self, skycat_dict: dict):
        """ Load a sky catalog 
        
        Args:
            skycat_dict (dict): Dictionary of ephem FixedBody radio sources
        """
        self.skycat = skycat_dict
    
    def new_fig(self, size: int=6):
        """ Create new matplotlib figure """
        plt.figure(self.name, figsize=(size, size), frameon=False)
    
    def plot(self, data: np.array=None, sfunc: np.ufunc=np.abs, overlay_srcs: bool=False,  overlay_grid: bool=True, 
                  title: str=None, colorbar: bool=False, return_data: bool=False, subplot_id: tuple=None, **kwargs):
        """ Plot all-sky image
        
        Args:
            data (np.array): Data to plot
            sfunc (np.unfunc): Scaling function to use, e.g. np.log, np.log10
            return_data (bool): If true, will return plot data as np array
        
        Plotting args:
            title (str): Title of plot. If not set, a title will be generated from LST and frequency.
            overlay_srcs (bool): Overlay sources in sky catalog (Default False)
            overlay_grid (bool): Overlay grid (Default true)
            colorbal (bool): Show colorbar (default False)
            subplot_id (tuple): Subplot ID, e.g. (2,2,1). Use if making multi-panel figures
            **kwargs: These are passed on to imshow()

        """
        # Update WCS and then create imshow
        self._update_wcs()
        if subplot_id is not None:
            plt.subplot(*subplot_id, projection=self.wcs)
        else:
            plt.subplot(projection=self.wcs)
        plt.imshow(sfunc(data), **kwargs)
        
        # Create title
        if title is None:
            lst_str = str(self.observer.sidereal_time())
            title = f'{self.name}:  {self.ts.iso}  \n LST: {lst_str}  |  freq: {self.f_mhz:.2f} MHz'
        plt.title(title)
        
        # Overlay a grid onto the imshow
        if overlay_grid: 
            plt.grid(color='white', ls='dotted')
        
        # Overlay sources in skycat onto image
        if overlay_srcs:
            for src, src_sc in self.skycat.items():
                x, y = self.wcs.world_to_pixel(src_sc)
                if not np.isnan(x) and not np.isnan(y):
                    plt.scatter(x, y, marker='x', color='red')
                    plt.text(x + 3, y + 3, src, color='white')
        
        # Turn on colorbar if requested
        if colorbar is True:
            plt.colorbar(orientation='horizontal')
            
        #plt.axis('off')
        if return_data:
            return data