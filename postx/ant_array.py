"""
Basic antenna array geometry class
"""
import ephem
import numpy as np
import datetime
from astropy.io import fits as pf
import h5py
from pygdsm import GSMObserver
import pylab as plt

from astropy.constants import c
from astropy.time import Time
from astropy.coordinates import SkyCoord
import healpy as hp

from .coord_utils import phase_vector, ephem_to_skycoord, skycoord_to_ephem, skycoord_to_lmn, sky2hpix, hpix2sky

#SHORTHAND
sin, cos = np.sin, np.cos
from astropy.constants import c
SPEED_OF_LIGHT = c.value 


class RadioArray(ephem.Observer):
    """ RadioArray class, designed for post-correlation beamforming and all-sky imaging 
    
    This class is a subclass of PyEphem's Observer(), and provides the following methods:
        
        get_zenith()     - get the ra/dec at zenith
        load_fits_data() - load UVFITS data
        load_h5_data()   - load data in HDF5 format
        update()         - Recompute coordinates for imaging/beamforming
        make_image()     - Make an image using post-correlation beamforming
        make_healpix()   - Generate a grid of beam on healpix coordinates using post-x beamforming
        beamform()       - Generate a single beam using post-x beamforming
        generate_gsm()   - Generate and view a sky model using the GSM
    """
    def __init__(self, lat: str, long: str, elev: float, f_mhz: float, antxyz_h5: str, 
                 t0: Time=None, phase_center: SkyCoord=None, conjugate_data: bool=False, verbose: bool=False):
        """ Initialize RadioArray class (based on PyEphem observer)
        
        Args:
            lat (str):         Latitude of telescope (passed to pyephem)
            long (str):        Longtude of telescope (passed to pyephem)
            elev (float):      Elevation of telescope (passed to pyephem)
            f_mhz (np.array):  Frequency of observations, in MHz
            t0 (astropy.Time): Time of observation start
            antxyz_h5 (np.array):    HDF5 file with antenna locations
            phase_center (SkyCoord): Phase center of telescope, if not Zenith.
                                     Used to apply an extra geometric delay correction
            conjugate_data (bool):   Flag to conjugate data (in case upper/lower triangle confusion)
            verbose (bool):    Print extra details to screen
        
        Notes:
            antxyz_h5 file should have a 'xyz_local' dataset, which
            stores antenna positions in meters, references to zenith (local / ENU) 
        
        """
        super().__init__()
        self.lat  = lat
        self.lon  = long
        self.elev = elev 
        self.date = t0 if t0 else datetime.datetime.now()
        
        with h5py.File(antxyz_h5, 'r') as h:
            self.xyz_local     = h['xyz_local'][:]      
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
            self.phase_center    = self.get_zenith()
        else:
            self.phase_center    = phase_center
            self.workspace['c0'] = self._generate_phase_vector(self.phase_center, conj=True)
        
        # Healpix workspace
        self.workspace['hpx'] = {}
        
        # Setup Global Sky Model
        self.gsm       = GSMObserver()
        self.gsm.lat   = lat
        self.gsm.lon   = long
        self.gsm.elev  = elev
        self.gsm.date  = self.date
    
    def _generate_phase_vector(self, src: SkyCoord, conj: bool=False):
        """ Generate a phase vector for a given source 
        
        Args:
            src (astropy.SkyCoord or ephem.FixedBody): Source to compute delays toward
        
        Returns:
            c (np.array): Per-antenna phase weights
        """
        if isinstance(src, (ephem.FixedBody, ephem.Sun, ephem.Moon)):
            src.compute(self)
            src = ephem_to_skycoord(src)
            
        lmn = skycoord_to_lmn(src, self.get_zenith())
        t_g = np.einsum('id,pd', lmn, self.xyz_local, optimize=True) / SPEED_OF_LIGHT
        c = phase_vector(t_g, self.workspace['f'], conj=conj)
        return c
    
    def _print(self, msg: str):
        """ Print a message if verbose flag is set"""
        if self.verbose:
            print(msg)

    def _compute_hourangle(self, src: SkyCoord) -> tuple:
        """ Compute the hourangle between a source and current zenith 
        
        Args:
            src (SkyCoord or FixedBody): Source to compute hourangle of
        
        Returns:
            (H, d) (float, float): Hourangle and declination of the source w.r.t. zenith
        
        Notes:
            This method is deprecated, in favor of get_zenith() and using lmn coordinates
        """
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
        return (H, d)
    
    @property
    def time(self):
        return Time(self.date.datetime())
    
    @property
    def zenith(self):
        return self.get_zenith()
    
    def get_zenith(self) -> SkyCoord:
        """ Return the sky coordinates at zenith 
        
        Returns:
           zenith (SkyCoord): Zenith SkyCoord object
        """
        ra, dec = self.radec_of(0, np.pi/2)
        sc = SkyCoord(ra, dec, frame='icrs', unit=('rad', 'rad'))
        return sc

    def load_fits_data(self, filename: str):
        """ Load data in FITS format """
        fn_re = filename.replace('imag', 'real')
        fn_im = filename.replace('real', 'imag')
        d_re = pf.open(fn_re)[0].data
        d_im = pf.open(fn_im)[0].data
        
        self.data = np.zeros_like(d_re, dtype='complex128')
        self.workspace['f_idx'] = 0
        self.data.real = d_re
        self.data.imag = d_im
            
    def load_h5_data(self, filename: str):
        """ Load data in HDF5 format """
        self.h5   = h5py.File(filename, 'r')
        self._data = self.h5['data']
        
        dt = Time(self.h5['time'][0], format='jd').datetime
        self.f = self.h5['freqs'][:] * 1e6
        pc = SkyCoord(self.h5['ra'][0], self.h5['dec'][0], unit=('deg', 'deg'))
        self.phase_center = skycoord_to_ephem(pc)
        self.update(date=dt, f_idx=0)
        
    def update(self, date: datetime.datetime=None, f_idx: int=None, pol_idx: int=0, update_gsm: bool=False):
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
            if isinstance(date, Time):
                date = date.datetime
            self._print("Updating datetime")
            self.date = date
            self.gsm.date = self.date

        if f_idx is not None:
            self._print("Updating freq idx")
            self.workspace['f_idx'] = f_idx
            self.data  = self._data[f_idx, :, :, pol_idx]
            
        if self.phase_center is not None:
            self._print("Updating phase vector")
            f = self.workspace['f']
            self.workspace['c0'] = self._generate_phase_vector(self.phase_center, conj=True)
            
        if self.conjugate_data:
            self._print("conjugating data")
            self.data = np.conj(self.data)
            
        if update_gsm:
            self._print("Updating GSM")
            self.gsm.date = self.date
            self.gsm.generate(self.f[f_idx] / 1e6)

    def _generate_weight_grid(self, n_pix: int):
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
    
    def plot_corr_matrix(self, log: bool=True):
        """ Plot correlation matrix """
        data = np.log(np.abs(self.data)) if log else np.abs(self.data)
        plt.imshow(data, aspect='auto')
        plt.xlabel("Antenna P")
        plt.ylabel("Antenna Q")
        plt.colorbar()
    
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
        B = np.einsum('ijp,pq,ijq->ij', self.workspace['cgrid'], self.data, self.workspace['cgrid_conj'], optimize=True)
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

            lmn = skycoord_to_lmn(sc[pix_visible], sc_zen)
            t_g = np.einsum('id,pd', lmn, self.xyz_local, optimize=True) / SPEED_OF_LIGHT
            c = phase_vector(t_g, ws['f'])

            ws['hpx']['mask'] = mask
            ws['hpx']['lmn'] = lmn
            ws['hpx']['phs_vector'] = c * ws['c0']    # Correct for vis phase center (i.e.the Sun)
            
        
        mask = ws['hpx']['mask']       # Horizon mask
        c = ws['hpx']['phs_vector']    # Pointing phase vector
        
        B = np.abs(np.einsum('ip,pq,iq->i', c, self.data, np.conj(c), optimize=True))

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
        sky = self.gsm.generate(self.f[self.workspace['f_idx']] / 1e6)
        pmap = hp.orthview(sky, half_sky=True, return_projected_map=True, flip='astro')
        plt.close()
        return pmap[::-1]