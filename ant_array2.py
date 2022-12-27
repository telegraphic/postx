"""
Basic antenna array geometry class
"""
import ephem
import numpy as np
import datetime
from astropy.io import fits as pf
import h5py
from pygdsm import GSMObserver

from astropy.constants import c
from astropy.time import Time
from astropy.coordinates import SkyCoord

from coord_utils import compute_w, polar_to_cartesian, phase_vector, generate_phase_vector, skycoord_to_ephem

#SHORTHAND
sin, cos = np.sin, np.cos
SPEED_OF_LIGHT = c.value 


class RadioArray(ephem.Observer):
    
    def __init__(self, lat, long, elev, f_mhz, antxyz_h5, 
                 t0=None, phase_center=None, conjugate_data=False, verbose=False):
        
        super().__init__()
        self.lat  = lat
        self.lon  = long
        self.elev = elev 
        self.date = t0 if t0 else datetime.datetime.now()
        
        with h5py.File(antxyz_h5, 'r') as h:
            self.xyz_local     = h['xyz_local'][:]       # TODO: Is this East-North-Up?
            self.xyz_celestial = h['xyz_celestial'][:]   # TODO: Calculate instead?
            self.n_ant = len(self.xyz_local)
        
        self.f =  f_mhz * 1e6
    
        # Set debug verbosity
        self.verbose = verbose

        # Correlation matrix and pointing vector w
        self.data = np.zeros((self.n_ant, self.n_ant), dtype='complex64')
        self.conjugate_data = conjugate_data
    
        # Create workspace dictionary, for storing state
        self.workspace = {}
        self.workspace['f_idx']  = 0
        self.workspace['f']      = self.f[0]
        
        # Setup phase center
        if phase_center is None:
            # c0 is phase weights vector 
            self.workspace['c0'] = np.ones(self.n_ant, dtype='complex64')
            self.phase_center    = 'ZENITH'
        else:
            self.phase_center    = phase_center
            H, d = self._compute_hourangle(phase_center)
            self.workspace['H0'] = H
            self.workspace['d0'] = d
            self.workspace['c0'] = generate_phase_vector(self.xyz_celestial, H, d, self.workspace['f'], conj=True)

        # Setup Global Sky Model
        self.gsm       = GSMObserver()
        self.gsm.lat   = lat
        self.gsm.lon   = long
        self.gsm.elev  = elev
        self.gsm.date  = self.date
    
    def _print(self, msg):
        if self.verbose:
            print(msg)

    def _compute_hourangle(self, src):
        if isinstance(src, SkyCoord):
            src = skycoord_to_ephem(src)
        src.compute(self)
        ra0, dec0 = self.radec_of(0, np.pi/2)
        H = ra0 - src.ra
        d = src.dec
        self._print(f"Time: {self.date}")
        self._print(f'{src.name} \tRA / DEC:  ({src.ra}, {src.dec}) \n\tALT / AZ:  ({src.alt}, {src.az})')
        self._print(f'ZENITH: ({ra0}, {dec0})')
        self._print(f'HA, D: ({H}, {d})')
        return H, d

    def load_fits_data(self, filename):
        fn_re = filename.replace('imag', 'real')
        fn_im = filename.replace('real', 'imag')
        d_re = pf.open(fn_re)[0].data
        d_im = pf.open(fn_im)[0].data
        
        self.data = np.zeros_like(d_re, dtype='complex128')
        self.data.real = d_re
        self.data.imag = d_im
            
    def load_h5_data(self, filename):
        self.h5   = h5py.File(filename, 'r')
        self._data = self.h5['data']
        self.update(f_idx=0)
        
    def update(self, date=None, f_idx=None, pol_idx=0, update_gsm=False):
        """ Update internal state
        
        Args:
            date (datetime): New observation datetime
            f_idx (int): Integer index for frequency axis
            pol_idx (int): Change polarization index (0--4)
            update_gsm (bool): Update GSM observed sky. Default False
        
        Notes:
            Call when changing datetime, frequency or polarization index
        """
        if date is not None:
            self._print("Updating datetime")
            self.date = date
            
        if f_idx is not None:
            self._print("Updating freq idx")
            self.workspace['f_idx'] = f_idx
            self.data  = self._data[f_idx, :, :, pol_idx]
            
        if self.phase_center is not None:
            self._print("Updating phase matrix")
            H, d = self._compute_hourangle(self.phase_center)
            f = self.workspace['f']
            self.workspace['H0'] = H
            self.workspace['d0'] = d
            self.workspace['c0'] = generate_phase_vector(self.xyz_celestial, H, d, f, conj=True)
            
        if self.conjugate_data:
            self._print("conjugating data")
            self.data = np.conj(self.data)
            
        if update_gsm:
            self._print("Updating GSM")
            self.gsm.generate(self.f[f_idx] / 1e6)

    def _generate_weight_grid(self, n_pix):
        """ Generate a grid of pointing weights 
        
        Args:
            n_pix (int): Number of pixels in image
        
        Notes:
            Generates a 2D array of coefficients (used internally).
        """
        l = np.linspace(1, -1, n_pix, dtype='float32')
        m = np.linspace(-1, 1, n_pix, dtype='float32')
        lg, mg = np.meshgrid(l, m)
        ng     = np.sqrt(1 - lg**2 - mg**2)

        lmn = np.zeros((n_pix, n_pix, 3), dtype='float32')
        lmn[:, :, 0] = lg
        lmn[:, :, 1] = mg
        lmn[:, :, 2] = ng

        # i, j: pix idx
        # d: direction cosine lmn, and baseline XYZ (dot product)
        # p: antenna idx 
        t_g = np.einsum('ijd,pd', lmn, self.xyz_local, optimize=True) / SPEED_OF_LIGHT

        # geometric delay to phase weights
        c = phase_vector(t_g, self.workspace['f']) * self.workspace['c0']
        
        self.workspace['cgrid'] = c
        self.workspace['cgrid_conj'] = np.conj(c)

    def make_image(self, n_pix=128, update_weight_grid=True):
        """ Make an image out of a beam grid 
        
        Args:
            n_pix (int): Image size in pixels (N_pix x N_pix)
            update_weight_grid (bool): Rerun the grid generation (needed when image size changes).
            
        Returns:
            B (np.array): Image array in (x, y)
        """
        if update_weight_grid:
            self._generate_weight_grid(n_pix)
        B = np.einsum('ijp,pq,ijq->ij', self.workspace['cgrid'], self.data, self.workspace['cgrid_conj'], optimize=True)
        return np.abs(B)

    def beamform(self, src):
        """ Form a beam toward a given source 
        
        Args:
            src (SkyCoord): Coordinates to point to
        
        Returns:
            B (np.array): Returned beamformed power for current frequency step.
        """
        
        H, d = self._compute_hourangle(src)
        c = generate_phase_vector(self.xyz_celestial, H, d, self.workspace['f'], conj=False)
        c *= self.workspace['c0']
        B = np.einsum('p,pq,q', c, self.data, np.conj(c), optimize=True)
        return np.abs(B)

    def generate_gsm(self):
        """ Generate a GlobalSkyModel orthographic map of observed sky
        
        Returns:
            pmap (array): 2D orthographic projection of observed sky
        """
        import healpy as hp
        import pylab as plt
        sky = self.gsm.generate(self.f[self._f_idx] / 1e6)
        pmap = hp.orthview(sky, half_sky=True, return_projected_map=True, flip='astro')
        plt.close()
        return pmap[::-1]