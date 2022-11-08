"""
Basic antenna array geometry class
"""
import ephem
import numpy as np
import datetime
from astropy.io import fits as pf
import h5py

from astropy.constants import c
SPEED_OF_LIGHT = c.value

def find_max_idx(data):
    idx = np.unravel_index(np.argmax(data), data.shape)
    v = data[idx]
    return idx, v

def polar_to_cartesian(θ,ϕ):
    x = np.sin(θ) * np.cos(ϕ)
    y = np.sin(θ) * np.sin(ϕ)
    z = np.cos(θ)   
    return x, y, z

class AntArray(ephem.Observer):
    """ Antenna array
    Based on pyEphem's Observer class.
    Args:
        lat (str):       latitude of array centre, e.g. 44:31:24.88
        long (str):      longitude of array centre, e.g. 11:38:45.56
        elev (float):    elevation in metres of array centre
        date (datetime): datetime object, date and time of observation
        antennas (np.array): numpy array of antenna positions, in xyz coordinates in meters, relative to the array centre.
    """
    def __init__(self, lat, long, elev, date, antennas):
        super(AntArray, self).__init__()
        self.lat = lat
        self.long = long
        self.elev = elev
        self.date = date
        self.antennas = np.array(antennas)
        self.n_ant    = len(antennas)
        self.baselines = self._generate_baseline_ids()
        self.xyz       = self._compute_baseline_vectors()

    def _compute_baseline_vectors(self, autocorrs=True):
        """ Compute all the baseline vectors (XYZ) for antennas
        Args:
            autocorrs (bool): baselines should contain autocorrs (zero-length baselines)
        Returns:
            xyz (np.array): XYZ array of all antenna combinations, in ascending antenna IDs.
        """
        xyz   = self.antennas
        n_ant = self.n_ant

        if autocorrs:
            bls = np.zeros([n_ant * (n_ant - 1) // 2 + n_ant, 3])
        else:
            bls = np.zeros([n_ant * (n_ant - 1) // 2, 3])

        bls_idx = 0
        for ii in range(n_ant):
            for jj in range(n_ant):
                if jj >= ii:
                    if autocorrs is False and ii == jj:
                        pass
                    else:
                        bls[bls_idx] = xyz[ii] - xyz[jj]
                        bls_idx += 1
        return bls

    def _generate_baseline_ids(self, autocorrs=True):
        """ Generate a list of unique baseline IDs and antenna pairs
        Args:
            autocorrs (bool): baselines should contain autocorrs (zero-length baselines)
        Returns:
            ant_arr (list): List of antenna pairs that correspond to baseline vectors
        """
        ant_arr = []
        for ii in range(1, self.n_ant + 1):
            for jj in range(1, self.n_ant + 1):
                if jj >= ii:
                    if autocorrs is False and ii == jj:
                        pass
                    else:
                        ant_arr.append((ii, jj))
        return ant_arr

    def update(self, date):
        """ Update antenna with a new datetime """
        self.date = date

    def report(self):
        print(self)
        print(self.xyz)


class RadioArray(AntArray):
    def __init__(self, lat, long, elev, f_mhz, filename):
        ant_loc = np.loadtxt(filename, skiprows=7, delimiter='\t', dtype='str')
        ant_xyz = ant_loc[:, 1:].astype('float64')
        self.n_ant = len(ant_loc)
        
        self.f =  f_mhz * 1e6
        date = datetime.datetime.now()
        
        # Call super to setup antenna XYZ
        super().__init__(lat, long, elev, date, ant_xyz)
        
        self.data = np.zeros((self.n_ant, self.n_ant), dtype='complex128')
        self.w1   = np.ones(self.n_ant, dtype='complex128')
        self.w2   = np.ones(self.n_ant, dtype='complex128')
        
        # Set phase center
        self._θ_0   = 0
        self._ϕ_0   = 0
        self._t_g_0 = 0
        self._f_idx = 0
        self._phs_delay_mat = None
        
        self.a0_baselines = self.xyz[:self.n_ant]
        
    def load_fits_data(self, filename):
        fn_re = filename.replace('imag', 'real')
        fn_im = filename.replace('real', 'imag')
        d_re = pf.open(fn_re)[0].data
        d_im = pf.open(fn_im)[0].data
        self.data = np.array(d_re + 1j * d_im, dtype='complex128')
    
    def load_h5_data(self, filename):
        self.h5   = h5py.File(filename, 'r')
        self.data = self.h5['data'][0, :, :, 0]
        
    def set_freq(self, freq_idx, pol_idx=0):
        self._f_idx = freq_idx
        self.data  = self.h5['data'][freq_idx, :, :, pol_idx]
        
        if self._phs_delay_mat is not None:
            self.data *= self._phs_delay_mat
    
    def beamform(self, w=None):
        self.w = w if w is not None else self.w
        b = np.einsum('p,q,pq', self.w, np.conj(self.w), self.data)
        return b
    
    def incoherent_beamform(self, w=None):
        self.w = w if w is not None else self.w
        b = np.einsum('p,p,pp', self.w, np.conj(self.w), self.data)
        return b        
    
    def point(self, θ, ϕ):
        # Compute geometric delay        
        x, y, z = polar_to_cartesian(θ, ϕ)
        pvec = np.array((x, y, z), dtype='float128')
        self._point_vec(pvec)
    
    def phase_to_src(self, θ, ϕ):
        x, y, z = polar_to_cartesian(θ, ϕ)
        pvec = np.array((x, y, z), dtype='float128')
        t_g  = np.dot(self.a0_baselines, pvec) / SPEED_OF_LIGHT
        
        phs_delay = np.exp(-1j * 2 * np.pi * self.f[self._f_idx] * t_g)
        phs_delay_mat = np.outer(phs_delay, np.conj(phs_delay))
        if self._phs_delay_mat is not None:
            self.data *= np.conj(self._phs_delay_mat)
        self.data *= phs_delay_mat
        self._phs_delay_mat = phs_delay_mat
        return phs_delay_mat
    
    def set_phase_center(self, θ, ϕ):
        
        self._θ_phs = θ
        self._ϕ_phs = ϕ
        
        x_0, y_0, z_0 = polar_to_cartesian(θ, ϕ)
        pvec_0 = np.array((x_0, y_0, z_0), dtype='float128')
        t_g_0  = np.dot(self.a0_baselines, pvec_0) / SPEED_OF_LIGHT
        
        self._t_g_0 = t_g_0
        
    def _point_vec(self, pvec):
        t_g  = np.dot(self.a0_baselines, pvec) / SPEED_OF_LIGHT
        t_g -= self._t_g_0
        
        # convert time delay to phase 
        phase_weights = np.exp(1j * 2 * np.pi * self.f[self._f_idx] * t_g)
        self.w = phase_weights
        self.t_g = t_g

    def make_image(self, Npix=128):

        l = np.linspace(-1, 1, Npix)
        m = np.linspace(-1, 1, Npix)

        grid = np.zeros((Npix, Npix), dtype='float64')
        for xx in range(Npix):
            for yy in range(Npix):
                pvec = np.array((l[xx], m[yy], np.sqrt(1 - l[xx]**2 + m[yy]**2)))
                self._point_vec(pvec)
                grid[xx, yy] = np.abs(self.beamform())
        return grid


def make_antenna_array(lat, lon, elev, date, antennas):
    """ Generate a new AntArray object
    Args:
        lat (str):       latitude of array centre, e.g. 44:31:24.88
        lon (str):      longitude of array centre, e.g. 11:38:45.56
        elev (float):    elevation in metres of array centre
        date (datetime): datetime object, date and time of observation
        antennas (np.array): numpy array of antenna positions, in xyz coordinates in meters,
                             relative to the array centre.
    Returns:
        ant_arr (AntArray): New Antenna Array object
    """
    return AntArray(lat, lon, elev, date, antennas)
