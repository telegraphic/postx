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

from aavs_uv.vis_utils import  vis_arr_to_matrix_4pol
from aavs_uv.datamodel import UVX
from .coord_utils import phase_vector, skycoord_to_lmn, generate_lmn_grid, sky2hpix, hpix2sky, gaincal_vec_to_matrix

from astropy.constants import c
SPEED_OF_LIGHT = c.value




class ApertureArray(object):
    """ RadioArray class, designed for post-correlation beamforming and all-sky imaging

    This class is a subclass of PyEphem's Observer(), and provides the following methods:

        get_zenith()     - get the ra/dec at zenith
        update()         - Recompute coordinates for imaging/beamforming
        make_image()     - Make an image using post-correlation beamforming
        make_healpix()   - Generate a grid of beam on healpix coordinates using post-x beamforming
        generate_gsm()   - Generate and view a sky model using the GSM
    """

    def __init__(self, uvx: UVX, conjugate_data: bool=False, verbose: bool=False):
        """ Initialize RadioArray class (based on astropy EarthLocation)

        Args:
            vis (UV):                datamodel visibility dataclass
            conjugate_data (bool):   Flag to conjugate data (in case upper/lower triangle confusion)
            verbose (bool):          Print extra details to screen

        """

        self.uvx = uvx
        self.conjugate_data = conjugate_data
        self.verbose = verbose
        self.name = uvx.name

        self.earthloc = uvx.origin

        xyz0 = uvx.antennas.attrs['array_origin_geocentric'].data
        self.xyz_enu  = uvx.antennas.enu.data
        self.xyz_ecef = uvx.antennas.ecef.data + xyz0

        self.n_ant = len(self.xyz_enu)
        self.ant_names = uvx.antennas.identifier

        # Setup frequency, time, and phase centre
        self.f = Quantity(uvx.data.coords['frequency'].data, 'Hz')
        self.t = uvx.timestamps
        self.p = uvx.data.polarization.values

        # Phase center
        self.phase_center = uvx.phase_center

        self.calibration_mat = None

        # Setup current index dict and workspace
        self.idx = {'t': 0, 'f': 0, 'p': 0}
        self.workspace = {}

        self.bl_matrix = self._generate_bl_matrix()

        # Healpix workspace
        self._to_workspace('hpx', {})

    def _ws(self, key: str):
        """ Return value of current index for freq / pol / time or workspace entry

        Helper function to act as 'workspace'.
        Uses self.idx dictionary which stores selected index

        Args:
            key (str): One of f (freq), p (pol) or t (time)

        Returns
            Value of f/p/t array at current index in workspace

        """
        if key in ('f', 'p', 't'):
            return self.__getattribute__(key)[self.idx[key]]
        else:
            return self.workspace[key]

    def _in_workspace(self, key: str) -> bool:
        # Check if key is in workspace
        return key in self.workspace

    def _to_workspace(self, key: str, val):
        self.workspace[key] = val

    def get_zenith(self) -> SkyCoord:
        """ Return the sky coordinates at zenith

        Returns:
           zenith (SkyCoord): Zenith SkyCoord object
        """
        zen_aa = AltAz(alt=Angle(90, unit='degree'), az=Angle(0, unit='degree'),
                       obstime=self._ws('t'), location=self.earthloc)
        zen_sc = SkyCoord(zen_aa).icrs
        return zen_sc

    def get_sun(self) -> SkyCoord:
        """ Return the sky coordinates of the Sun
        Returns:
           sun_sc (SkyCoord): sun SkyCoord object
        """
        sun_sc = SkyCoord(get_sun(self._ws('t')), location=self.earthloc)
        return sun_sc

    def _generate_bl_matrix(self):
        """ Compute a matrix of baseline lengths """

        # Helper fn to compute length for one row
        def bl_len(xyz, idx):
            return np.sum(np.sqrt((xyz - xyz[idx])**2), axis=-1)

        bls = np.zeros((self.n_ant, self.n_ant), dtype='float32')
        # Loop over rows
        for ii in range(256):
            bls[ii] = bl_len(self.xyz_enu, ii)
        return bls

    def generate_phase_vector(self, src: SkyCoord, conj: bool=False, coplanar: bool=False, apply_cal: bool=True):
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
        c = phase_vector(t_g, self._ws('f').to('Hz').value, conj=conj, dtype='complex64')
        return c

    def generate_weight_grid(self, n_pix: int, abs_max: int=1, nan_below_horizon: bool=True):
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
        # Generate grid of (l, m, n) coordinates
        lmn = generate_lmn_grid(n_pix, abs_max, nan_below_horizon)

        # Compute Geometric delay t_g
        # lmn shape: (n_pix, n_pix, n_lmn=3)
        # xyz_enu shape: (n_antenna, n_xyz=3)
        # i, j: pix idx
        # d: direction cosine lmn, and baseline XYZ (dot product)
        # a: antenna idx
        t_g = np.einsum('ijd,ad', lmn, self.xyz_enu, optimize=True) / SPEED_OF_LIGHT

        # Convert geometric delay to phase weight vectors
        # t_g shape: (n_pix, n_pix, n_ant)
        pv_grid = phase_vector(t_g, self._ws('f').to('Hz').value)
        if self._in_workspace('c0'):
            pv_grid *= self._ws('c0')

        #if nan_below_horizon:
        #    # Apply n=sqrt(l2 + m2) factor to account for projection
        #    # See Carozzi and Woan (2009)
        #    pv_grid = np.einsum('ij,aij->aij', lmn, pv_grid, optimize=True)

        # Store in workspace
        self._to_workspace('lmn_grid', lmn)
        self._to_workspace('pv_grid', pv_grid)
        self._to_workspace('pv_grid_conj',  np.conj(pv_grid))

        return pv_grid, lmn

    def set_cal(self, cal):
        cal_mat = gaincal_vec_to_matrix(cal)
        self.workspace['cal'] = {'gaincal': cal, 'gaincal_matrix': cal_mat}

    def generate_vis_matrix(self, t_idx=None, f_idx=None):

        t_idx = self.idx['t'] if t_idx is None else t_idx
        f_idx = self.idx['f'] if f_idx is None else f_idx
        vis_sel = self.uvx.data[t_idx, f_idx].values
        vis_mat = vis_arr_to_matrix_4pol(vis_sel, self.n_ant)

        if self._in_workspace('cal'):
            vis_mat *= self.workspace['cal']['gaincal_matrix']
        return vis_mat

    def make_image(self, n_pix: int=128, update: bool=True) -> np.array:
        """ Make an image out of a beam grid

        Args:
            n_pix (int): Image size in pixels (N_pix x N_pix)
            update (bool): Rerun the grid generation (needed when image size changes).

        Returns:
            B (np.array): Image array in (x, y)
        """
        if update:
            self.generate_weight_grid(n_pix)
        w = self.workspace
        V = self.generate_vis_matrix()

        # For some reason, breaking into four pols is signifcantly quicker
        # Than adding 'x' as pol axis and using one-liner pij,pqx,qij->ijx
        B = np.zeros(shape=(w['pv_grid'].shape[1], w['pv_grid'].shape[2], V.shape[2]), dtype='complex64')
        B[..., 0] = np.einsum('pij,pq,qij->ij', w['pv_grid'], V[..., 0], w['pv_grid_conj'], optimize='greedy')
        B[..., 1] = np.einsum('pij,pq,qij->ij', w['pv_grid'], V[..., 1], w['pv_grid_conj'], optimize='greedy')
        B[..., 2] = np.einsum('pij,pq,qij->ij', w['pv_grid'], V[..., 2], w['pv_grid_conj'], optimize='greedy')
        B[..., 3] = np.einsum('pij,pq,qij->ij', w['pv_grid'], V[..., 3], w['pv_grid_conj'], optimize='greedy')

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

        ws = self._ws('hpx')
        if ws.get('n_side', 0) != NSIDE:
            ws['n_side'] = NSIDE
            ws['n_pix']  = NPIX
            ws['pix0']   = np.arange(NPIX)
            ws['sc']     = hpix2sky(NSIDE, ws['pix0'])

        if ws.get('fov', 0) != fov:
            ws['fov'] = fov

        NPIX = ws['n_pix']
        sc   = ws['sc']         # SkyCoord coordinates array
        pix0 = ws['pix0']       # Pixel coordinate array

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
            c = phase_vector(t_g, self._ws('f').to('Hz').value)

            # Apply n factor to account for projection (Carozzi and Woan 2009 )
            c = np.einsum('i,ip->ip', lmn[:, 2], c, optimize=True)

            ws['mask'] = mask
            ws['lmn'] = lmn
            ws['phs_vector'] = c       # Correct for vis phase center (i.e.the Sun)

        mask = ws['mask']       # Horizon mask
        c = ws['phs_vector']    # Pointing phase vector

        V = self.generate_vis_matrix()

        # For some reason, breaking into four pols is signifcantly quicker
        # Than adding 'x' as pol axis and using one-liner pij,pqx,qij->ijx
        B = np.zeros(shape=(ws['lmn'].shape[0], 4), dtype='float32')
        B[..., 0] = np.abs(np.einsum('ip,pq,iq->i', c, V[..., 0], np.conj(c), optimize=True))
        B[..., 1] = np.abs(np.einsum('ip,pq,iq->i', c, V[..., 1], np.conj(c), optimize=True))
        B[..., 2] = np.abs(np.einsum('ip,pq,iq->i', c, V[..., 2], np.conj(c), optimize=True))
        B[..., 3] = np.abs(np.einsum('ip,pq,iq->i', c, V[..., 3], np.conj(c), optimize=True))

        # Create a hpx array with shape (NPIX, 4) and insert above-horizon data
        hpdata = np.zeros((ws['pix0'].shape[0], 4), dtype='float32')
        hpdata[pix0[~mask]] = B
        return hpdata
