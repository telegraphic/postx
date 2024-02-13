import numpy as np
import pylab as plt
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import healpy as hp

from .ant_array import RadioArray
from .sky_model import generate_skycat_solarsys, generate_skycat

class AllSkyViewer(object):
    """ An all-sky imager based on matplotlib imshow with WCS support

    Provides the following:
        update() - updates WCS information
        get_pixel() - get pixel information
    """
    def __init__(self, observer: RadioArray=None, skycat: dict=None, ts: Time=None, f_mhz: float=None, n_pix: int=128):
        self.observer = observer
        self.skycat = skycat if skycat is not None else generate_skycat(observer)
        self.name = observer.name if hasattr(observer, 'name') else 'allsky'

        self.ts     = ts
        self.f_mhz  = f_mhz
        self.n_pix  = n_pix
        self._update_wcs()


    def _update_wcs(self):
        """ Update World Coordinate System (WCS) information """
        zen_sc = self.observer.get_zenith()

        self.wcsd = {
                 'SIMPLE': 'T',
                 'NAXIS': 2,
                 'NAXIS1': self.n_pix,
                 'NAXIS2': self.n_pix,
                 'CTYPE1': 'RA---SIN',
                 'CTYPE2': 'DEC--SIN',
                 'CRPIX1': self.n_pix // 2 + 1,
                 'CRPIX2': self.n_pix // 2 + 1,
                 'CRVAL1': zen_sc.icrs.ra.to('deg').value,
                 'CRVAL2': zen_sc.icrs.dec.to('deg').value,
                 'CDELT1': -360/np.pi / self.n_pix,
                 'CDELT2': 360/np.pi / self.n_pix
            }

        self.wcs = WCS(self.wcsd)

    def _update_skycat(self):
        """ Update skycat with solar system objects """
        # Update Sun/Moon position (if needed)
        sm = generate_skycat_solarsys(self.observer)
        for key in sm.keys():
            if key in self.skycat.keys():
                self.skycat[key] = sm[key]

    def update(self):
        """ Update WCS information on timestamp or other change """
        self._update_wcs()
        self._update_skycat()

    def get_pixel(self, src: SkyCoord) -> tuple:
        """ Return the pixel index for a given SkyCoord

        Args:
            src (SkyCoord): sky coordinate of interest
        """

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

    def plot(self, data: np.array=None, pol_idx: int=0, sfunc: np.ufunc=np.abs,
                  overlay_srcs: bool=False,  overlay_grid: bool=True, return_data: bool=False,
                  title: str=None, colorbar: bool=False,  subplot_id: tuple=None, **kwargs):
        """ Plot all-sky image

        Args:
            data (np.array): Data to plot. Shape (N_pix, N_pix, N_pol). If not set, an image
                             will be generated from the ApertureArray object
            sfunc (np.unfunc): Scaling function to use, e.g. np.log, np.log10
            pol_idx (int): Polarization index to plot
            return_data (bool): If true, will return plot data as np array

        Plotting args:
            title (str): Title of plot. If not set, a title will be generated from LST and frequency.
            overlay_srcs (bool): Overlay sources in sky catalog (Default False)
            overlay_grid (bool): Overlay grid (Default true)
            colorbal (bool): Show colorbar (default False)
            subplot_id (tuple): Subplot ID, e.g. (2,2,1). Use if making multi-panel figures
            **kwargs: These are passed on to imshow()

        """
        if data is None:
            data = self.observer.make_image(self.n_pix, update=True)

        if data.shape[0] != self.n_pix:
            self.n_pix = data.shape[0]
            self.update()

        # Update WCS and then create imshow
        self._update_wcs()
        if subplot_id is not None:
            ax = plt.subplot(*subplot_id, projection=self.wcs)
        else:
            ax = plt.subplot(projection=self.wcs)
        im = plt.imshow(sfunc(data[..., pol_idx]), **kwargs)

        # Create title
        if title is None:
            ts = self.observer._ws('t')
            f  = self.observer._ws('f')
            lst_str = str(ts.sidereal_time('apparent'))
            title = f'{self.name}:  {ts.iso}  \n LST: {lst_str}  |  freq: {f.to("MHz").value:.2f} MHz'
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
            plt.colorbar(im, orientation='horizontal')


        if return_data:
            return data

    def mollview(self, hmap: np.array=None, sfunc=np.abs, n_side=64, fov=np.pi/2, apply_mask=True,
                 pol_idx: int=0, title=None, **kwargs):
        """ Healpix view """
        if hmap is None:
            hmap = self.observer.make_healpix(n_side=n_side, fov=fov, apply_mask=apply_mask)

        # Create title
        if title is None:
            ts = self.observer._ws('t')
            f  = self.observer._ws('f')
            lst_str = str(ts.sidereal_time('apparent'))
            title = f'{self.name}:  {ts.iso} | LST: {lst_str}  |  freq: {f.to("MHz").value:.3f} MHz'

        hp.mollview(sfunc(hmap[..., pol_idx]), coord='G', title=title, **kwargs)
        hp.graticule(color='white')
