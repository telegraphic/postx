"""
Basic antenna array geometry class
"""
import numpy as np
from pygdsm import GSMObserver
import pylab as plt

from astropy.constants import c
from astropy.units import Quantity
from astropy.time import Time
from astropy.coordinates import SkyCoord, AltAz, Angle, get_sun
import healpy as hp

from aavs_uv.vis_utils import  vis_arr_to_matrix
from aavs_uv.datamodel import UVX
from .coord_utils import phase_vector, skycoord_to_lmn, sky2hpix, hpix2sky

from astropy.constants import c
SPEED_OF_LIGHT = c.value


class RadioArray(object):
    """ RadioArray class, designed for post-correlation beamforming and all-sky imaging

    This class is a subclass of PyEphem's Observer(), and provides the following methods:

        get_zenith()     - get the ra/dec at zenith
        update()         - Recompute coordinates for imaging/beamforming
        make_image()     - Make an image using post-correlation beamforming
        make_healpix()   - Generate a grid of beam on healpix coordinates using post-x beamforming
        generate_gsm()   - Generate and view a sky model using the GSM
    """

    def __init__(self, vis: UVX, conjugate_data: bool=False, verbose: bool=False):
        """ Initialize RadioArray class (based on astropy EarthLocation)

        Args:
            vis (UV):                datamodel visibility dataclass
            conjugate_data (bool):   Flag to conjugate data (in case upper/lower triangle confusion)
            verbose (bool):          Print extra details to screen

        """

        self.vis = vis
        self.conjugate_data = conjugate_data
        self.verbose = verbose
        self.name = vis.name

        self.earthloc = vis.origin

        xyz0 = vis.antennas.attrs['array_origin_geocentric'].data
        self.xyz_enu  = vis.antennas.enu.data
        self.xyz_ecef = vis.antennas.ecef.data + xyz0

        self.n_ant = len(self.xyz_enu)

        # Setup frequency, time, and phase centre
        self.f = Quantity(vis.data.coords['frequency'].data, 'Hz')
        self.t = vis.timestamps

        self.phase_center = vis.phase_center

        # Create workspace dictionary, for storing state
        self.workspace = {}
        self.workspace['f_idx']   = 0
        self.workspace['f']       = self.f[0]
        self.workspace['t_idx']   = 0
        self.workspace['t']       = self.t[0]
        self.workspace['pol']     = vis.data.coords['polarization'][0]
        self.workspace['pol_idx'] = 0
        self.workspace['c0']      = np.ones(self.n_ant, dtype='complex64')
        self.workspace['data']    = np.zeros((self.n_ant, self.n_ant), dtype='complex64')

        # Healpix workspace
        self.workspace['hpx'] = {}

        # Setup Global Sky Model
        self.gsm       = GSMObserver()
        self.gsm.lat   = vis.origin.lat.to('rad').value
        self.gsm.lon   = vis.origin.lon.to('rad').value
        self.gsm.elev  = vis.origin.height.to('m').value
        self.gsm.date  = self.t[0].datetime

        # Call update to load correlation matrix
        self.update()

    def _generate_phase_vector(self, src: SkyCoord, conj: bool=False, coplanar: bool=False):
        """ Generate a phase vector for a given source

        Args:
            src (astropy.SkyCoord or ephem.FixedBody): Source to compute delays toward
            conj (bool); Conjugate data if True
            coplanar (bool): Treat array as coplanar if True. Sets antenna z-pos to zero

        Returns:
            c (np.array): Per-antenna phase weights
        """
        lmn = skycoord_to_lmn(src, self.get_zenith())
        ant_pos = self.xyz_enu
        if coplanar:
            ant_pos[..., 2] = 0

        t_g = np.einsum('id,pd', lmn, self.xyz_enu, optimize=True) / SPEED_OF_LIGHT
        c = phase_vector(t_g, self.workspace['f'].to('Hz').value, conj=conj)
        return c

    def _print(self, msg: str):
        """ Print a message if verbose flag is set"""
        if self.verbose:
            print(msg)

    @property
    def zenith(self):
        return self.get_zenith()

    def get_zenith(self) -> SkyCoord:
        """ Return the sky coordinates at zenith

        Returns:
           zenith (SkyCoord): Zenith SkyCoord object
        """
        zen_aa = AltAz(alt=Angle(90, unit='degree'), az=Angle(0, unit='degree'),
                       obstime=self.workspace['t'], location=self.earthloc)
        zen_sc = SkyCoord(zen_aa).icrs
        return zen_sc

    def get_sun(self) -> SkyCoord:
        """ Return the sky coordinates of the Sun
        Returns:
           sun_sc (SkyCoord): sun SkyCoord object
        """
        sun_sc = SkyCoord(get_sun(self.workspace['t']), location=self.earthloc)
        return sun_sc

    def update(self, t_idx: int=None, f_idx: int=None, pol_idx: int=0, update_gsm: bool=False):
        """ Update internal state

        Args:
            date (datetime): New observation datetime
            f_idx (int): Integer index for frequency axis
            pol_idx (int): Change polarization index (0--4)
            update_gsm (bool): Update GSM observed sky. Default False

        Notes:
            Call when changing datetime, frequency or polarization index
        """
        if t_idx is not None:
            self._print("Updating time index")
            self.workspace['t_idx'] = t_idx
            self.workspace['t'] = self.t[t_idx]
            self.gsm.date = self.workspace['t'].datetime

        if f_idx is not None:
            self._print("Updating freq idx")
            self.workspace['f_idx'] = f_idx

        if pol_idx is not None:
            self.workspace['pol']     = self.vis.data.coords['polarization'][pol_idx]
            self.workspace['pol_idx'] = pol_idx

        # Extract time/pol/freq slice and convert to matrix
        w = self.workspace
        d = self.vis.data[w['t_idx'], w['f_idx'], :,  w['pol_idx']]
        self.workspace['data'] = vis_arr_to_matrix(d, self.n_ant, 'upper', V=self.workspace['data'])

        if self.phase_center is not None:
            self._print("Updating phase vector")
            f = self.workspace['f'].to('Hz').value
            self.workspace['c0'] = self._generate_phase_vector(self.phase_center, conj=True)

        if self.conjugate_data:
            self._print("conjugating data")
            self.workspace['data'] = np.conj(self.workspace['data'])

        if update_gsm:
            self._print("Updating GSM")
            self.gsm.date = self.workspace['t'].datetime
            self.gsm.generate(self.workspace['f'].to('MHz').value)

    def _generate_weight_grid(self, n_pix: int, abs_max: int=1, nan_below_horizon: bool=True):
        """ Generate a grid of direction cosine pointing weights

        Generates a square lmn grid across l=(-abs_max, abs_max), m=(-abs_max, abs_max).

        Notes:
            For unit pointing vector, l^2 + m^2 + n^2 = 1

        Args:
            n_pix (int): Number of pixels in image
            abs_max (int): Maximum absolute values for l and m (default 1).
            nan_below_horizon (bool): If True, n is NaN below horizon.
                                      If False, n is 0 below horizon


        Notes:
            Generates a 2D array of coefficients (used internally).
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

        # i, j: pix idx
        # d: direction cosine lmn, and baseline XYZ (dot product)
        # p: antenna idx
        t_g = np.einsum('ijd,pd', lmn, self.xyz_enu, optimize=True) / SPEED_OF_LIGHT

        # geometric delay to phase weights
        c = phase_vector(t_g, self.workspace['f'].to('Hz').value) * self.workspace['c0']

        if nan_below_horizon:
            # Apply n factor to account for projection (Carozzi and Woan 2009 )
            c = np.einsum('ij,ijp->ijp', ng, c, optimize=True)

        self.workspace['lmn_grid'] = lmn
        self.workspace['cgrid'] = c
        self.workspace['cgrid_conj'] = np.conj(c)


    def plot_corr_matrix(self, log: bool=True, **kwargs):
        """ Plot correlation matrix """
        data = np.log(np.abs(self.workspace['data'])) if log else np.abs(self.workspace['data'])
        pol_label = self.vis.data.polarization.values[self.workspace['pol_idx']]
        plt.imshow(data, aspect='auto', **kwargs)
        plt.title(pol_label)
        plt.xlabel("Antenna P")
        plt.ylabel("Antenna Q")
        plt.colorbar()

    def plot_corr_matrix_4pol(self, **kwargs):
        """ Plot correlation matrix, for all pols """
        plt.figure(figsize=(10, 10))
        for ii in range(4):
            plt.subplot(2,2,ii+1)
            self.update(pol_idx=ii)
            self.plot_corr_matrix(**kwargs)
        plt.tight_layout()
        plt.show()


    def plot_antennas(self, x: str='E', y: str='N', overlay_names: bool=False, overlay_fontsize: str='x-small', **kwargs):
        """ Plot antenna locations in ENU

        Args:
            x (str): One of 'E', 'N', or 'U'
            y (str): One of 'E', 'N', or 'U'
            overlay_names (bool): Overlay the antenna names on the plot. Default False
            overlay_fontsize (str): Font size for antenna names 'xx-small', 'x-small', 'small', 'medium',
                                                                'large', 'x-large', 'xx-large'
        """
        enu_map = {'E':0, 'N':1, 'U':2}
        title = f"{self.name} | Lon: {self.earthloc.to_geodetic().lon:.2f} | Lat: {self.earthloc.to_geodetic().lat:.2f}"
        plt.scatter(self.xyz_enu[:, enu_map[x.upper()]], self.xyz_enu[:, enu_map[y.upper()]], **kwargs)
        plt.xlabel(f"{x} [m]")
        plt.ylabel(f"{y} [m]")

        if overlay_names:
            names = self.vis.antennas.attrs['identifier'].data
            for ii in range(self.n_ant):
                plt.text(self.xyz_enu[:, enu_map[x]][ii], self.xyz_enu[:, enu_map[y]][ii], names[ii], fontsize=overlay_fontsize)
        plt.title(title)


    def make_image(self, n_pix: int=128, update: bool=True) -> np.array:
        """ Make an image out of a beam grid

        Args:
            n_pix (int): Image size in pixels (N_pix x N_pix)
            update (bool): Rerun the grid generation (needed when image size changes).

        Returns:
            B (np.array): Image array in (x, y)
        """
        if update:
            self._generate_weight_grid(n_pix)
        w = self.workspace
        B = np.einsum('ijp,pq,ijq->ij', w['cgrid'], w['data'], w['cgrid_conj'], optimize=True)
        return np.abs(B)


    def make_healpix(self, n_side: int=128, fov: float=np.pi/2, update: bool=True, apply_mask: bool=True) -> np.array:
        """ Generate a grid of beams on healpix coordinates

        Args:
            n_side (int): Healpix NSIDE array size
            fov (float): Field of view in radians, within which to generate healpix beams. Defaults to pi/2.
            apply_mask (bool): Apply mask for pixels that are below the horizon (default True)
            update (bool): Update pre-computed pixel and coordinate data. Defaults to True.
                           Setting to False gives a speedup, but requires that pre-computed coordinates are still accurate.

        Returns:
            hpdata (np.array): Array of healpix data, ready for hp.mollview() and other healpy routines.
        """
        NSIDE = n_side
        NPIX  = hp.nside2npix(NSIDE)

        ws = self.workspace
        if ws['hpx'].get('n_side', 0) != NSIDE:
            ws['hpx']['n_side'] = NSIDE
            ws['hpx']['n_pix']  = NPIX
            ws['hpx']['pix0']   = np.arange(NPIX)
            ws['hpx']['sc']     = hpix2sky(NSIDE, ws['hpx']['pix0'])
            update = True

        NPIX = ws['hpx']['n_pix']
        sc   = ws['hpx']['sc']         # SkyCoord coordinates array
        pix0 = ws['hpx']['pix0']       # Pixel coordinate array

        if ws['hpx'].get('fov', 0) != fov:
            ws['hpx']['fov'] = fov
            update = True

        if update:
            sc_zen = self.get_zenith()
            pix_zen = sky2hpix(NSIDE, sc_zen)
            vec_zen = hp.pix2vec(NSIDE, pix_zen)

            mask = np.ones(shape=NPIX, dtype='bool')

            if apply_mask:
                pix_visible = hp.query_disc(NSIDE, vec=vec_zen, radius=fov)
                mask[pix_visible] = False
            else:
                mask = np.zeros_like(mask)
                pix_visible = pix0

            lmn = skycoord_to_lmn(sc[pix_visible], sc_zen)
            t_g = np.einsum('id,pd', lmn, self.xyz_enu, optimize=True) / SPEED_OF_LIGHT
            c = phase_vector(t_g, ws['f'].to('Hz').value)

            # Apply n factor to account for projection (Carozzi and Woan 2009 )
            c = np.einsum('i,ip->ip', lmn[:, 2], c, optimize=True)

            ws['hpx']['mask'] = mask
            ws['hpx']['lmn'] = lmn
            ws['hpx']['phs_vector'] = c * ws['c0']    # Correct for vis phase center (i.e.the Sun)


        mask = ws['hpx']['mask']       # Horizon mask
        c = ws['hpx']['phs_vector']    # Pointing phase vector

        B = np.abs(np.einsum('ip,pq,iq->i', c, ws['data'], np.conj(c), optimize=True))


        hpdata = np.zeros_like(pix0)

        hpdata[pix0[~mask]] = B
        return hpdata

    def generate_gsm(self) -> np.array:
        """ Generate a GlobalSkyModel orthographic map of observed sky

        Returns:
            pmap (array): 2D orthographic projection of observed sky
        """
        import healpy as hp
        import pylab as plt
        sky = self.gsm.generate(self.workspace['f'].to('MHz').value)
        pmap = hp.orthview(sky, half_sky=True, return_projected_map=True, flip='astro')
        plt.close()
        return pmap