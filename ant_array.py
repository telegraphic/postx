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

#SHORTHAND
sin, cos = np.sin, np.cos
SPEED_OF_LIGHT = c.value

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

def find_max_idx(data):
    idx = np.unravel_index(np.argmax(data), data.shape)
    v = data[idx]
    return idx, v

def polar_to_cartesian(θ,ϕ):
    x = np.sin(θ) * np.cos(ϕ)
    y = np.sin(θ) * np.sin(ϕ)
    z = np.cos(θ)   
    return x, y, z

def pointing_vec(za, az):
    x, y, z = polar_to_cartesian(za, az)
    pvec = np.array((x,y,z), dtype='float64')
    return pvec

def compute_phase_matrix(w, f, conj=False):
    """ Compute NxN phase weight matrix """
    phs = np.exp(1j * 2 * np.pi * w * f)
    phs = np.conj(phs) if conj else phs
    pmat = np.outer(phs, np.conj(phs))
    return pmat

def compute_uvw(xyz, H, d, in_seconds=True, conjugate=False):
    """ Converts X-Y-Z baselines into U-V-W
    Parameters
    ----------
    xyz: should be a numpy array [x,y,z] of baselines (NOT ANTENNAS!)
    H: float (float, radians) is the hour angle of the phase reference position
    d: float (float, radians) is the declination
    conjugate: (bool): Conjugate UVW coordinates?
    in_seconds (bool): Return in seconds (True) or meters (False)
    Returns uvw vector (in microseconds)
    Notes
    -----
    The transformation is precisely that from the matrix relation 4.1 in
    Thompson Moran Swenson:
              [sin(H), cos(H), 0],
              [-sin(d)*cos(H), sin(d)*sin(H), cos(d)],
              [cos(d)*cos(H), -cos(d)*sin(H), sin(d)]
    A right-handed Cartesian coordinate system is used where X
    and Y are measured in a plane parallel to the earth's equator, X in the meridian
    plane (defined as the plane through the poles of the earth and the reference point
    in the array), Y is measured toward the east, and Z toward the north pole. In terms
    of hour angle H and declination d, the coordinates (X, Y, Z) are measured toward
    (H = 0, d = 0), (H = -6h, d = O), and (d = 90"), respectively.
    Here (H, d) are usually the hour angle and declination of the phase reference
    position.
    """
    is_list = True
    if type(xyz) in (list, tuple):
        x, y, z = xyz
    if type(xyz) == type(np.array([1])):
        is_list = False
        try:
            #print xyz.shape
            x, y, z = np.split(xyz, 3, axis=1)
        except:
            print(xyz.shape)
            raise

    sh, sd = sin(H), sin(d)
    ch, cd = cos(H), cos(d)
    u  = sh * x + ch * y
    v  = -sd * ch * x + sd * sh * y + cd * z
    w  = cd * ch * x - cd * sh * y + sd * z

    if is_list:
        uvw = np.array((u, v, w))
    else:
        uvw = np.column_stack((u, v, w))

    if conjugate:
        uvw *= -1
    if in_seconds:
        return uvw / SPEED_OF_LIGHT
    else:
        return uvw

def compute_w(xyz, H, d, conjugate=False, in_seconds=True):
    """ Compute geometric delay τ_g, equivalent to w term """
    x, y, z = np.split(xyz, 3, axis=1)
    sh, sd = sin(H), sin(d)
    ch, cd = cos(H), cos(d)
    w  = cd * ch * x - cd * sh * y + sd * z
    w = -w if conjugate else w
    w = w / SPEED_OF_LIGHT if in_seconds else w
    return w.squeeze()

def generate_phase_matrix(xyz, H, d, f, conj=False):
    w = compute_w(xyz, H, d)
    pmat = compute_phase_matrix(w, f, conj)
    return pmat

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
    def __init__(self, lat, long, elev, f_mhz, antxyz_h5, 
                 t0=None, phase_center=None, conjugate_data=False, verbose=False):
        
        with h5py.File(antxyz_h5, 'r') as h:
            self.xyz_local     = h['xyz_local'][:]      # TODO: Is this East-North-Up?
            self.xyz_celestial = h['xyz_celestial'][:]   # TODO: Calculate instead?
            self.n_ant = len(self.xyz_local)
        
        self.f =  f_mhz * 1e6
        
        # Set date if 
        date = t0 if t0 else datetime.datetime.now()

        # Set debug verbosity
        self.verbose = verbose

        # Correlation matrix and pointing vector w
        self.data = np.zeros((self.n_ant, self.n_ant), dtype='complex128')
        self.w    = np.ones(self.n_ant, dtype='complex128')
        self.conjugate_data = conjugate_data

        # Call super to setup antenna XYZ
        super().__init__(lat, long, elev, date, self.xyz_local)

        # Set phase center
        self._f_idx = 0
        self._phs_delay_mat = None
        if phase_center is not None:
            self.phase_center = phase_center
            H, d = self._src_hourangle(phase_center)
            _f = self.f[self._f_idx]
            self._phs_delay_mat = generate_phase_matrix(self.xyz_celestial, H, d, _f, conj=True)

        # Setup Global Sky Model
        self.gsm       = GSMObserver()
        self.gsm.lat   = lat
        self.gsm.lon   = long
        self.gsm.elev  = elev
        self.gsm.date  = date
    
    def _print(self, msg):
        if self.verbose:
            print(msg)

    def _src_hourangle(self, src):
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
        self.data = np.array(d_re + 1j * d_im, dtype='complex128')

        if self._phs_delay_mat is not None:
            self._print("Applying phase delay matrix")
            self.data *= self._phs_delay_mat
    
    def load_h5_data(self, filename):
        self.h5   = h5py.File(filename, 'r')
        self._data = self.h5['data']
        self.update(f_idx=0)
        
    def update(self, date=None, f_idx=None, pol_idx=0, update_gsm=False):
        if date is not None:
            self._print("Updating datetime")
            self.date = date
        if f_idx is not None:
            self._print("Updating freq idx")
            self._f_idx = f_idx
            self.data  = self._data[f_idx, :, :, pol_idx]
        if self.phase_center is not None:
            self._print("Updating phase matrix")
            H, d = self._src_hourangle(self.phase_center)
            _f = self.f[self._f_idx]
            self._phs_delay_mat = generate_phase_matrix(self.xyz_celestial, H, d, _f, conj=True)
        if self.conjugate_data:
            self._print("conjugating data")
            self.data = np.conj(self.data)
        if self._phs_delay_mat is not None:
            self._print("Applying phase delay matrix")
            self.data *= self._phs_delay_mat
        if update_gsm:
            self._print("Updating GSM")
            self.gsm.generate(self.f[f_idx] / 1e6)

    
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
        pvec = np.array((x, y, z), dtype='float64')
        self._point_vec(pvec)
    
    def phase_to_src(self, src):
        H, d = self._src_hourangle(src)
        pm = generate_phase_matrix(self.xyz_celestial, H, d, self.f[self._f_idx], conj=False)

        self._phs_delay_mat = pm
        return pm
        
    def _point_vec(self, pvec):
        t_g  = np.dot(self.xyz_local, pvec) / SPEED_OF_LIGHT
        
        # convert time delay to phase 
        phase_weights = np.exp(1j * 2 * np.pi * self.f[self._f_idx] * t_g)
        self.w = phase_weights
        self.t_g = t_g

    def make_image(self, n_pix=128):

        l = np.linspace(1, -1, n_pix)
        m = np.linspace(1, -1, n_pix)

        grid = np.zeros((n_pix, n_pix), dtype='float64')
        for xx in range(n_pix):
            for yy in range(n_pix):
                lm2 = l[xx]**2 + m[yy]**2
                if lm2 < 1:
                    pvec = np.array((l[xx], m[yy], np.sqrt(1 - lm2)))
                    self._point_vec(pvec)
                    grid[yy, xx] = np.abs(self.beamform())
        return grid

    def _generate_weight_grid(self, n_pix):
        l = np.linspace(1, -1, n_pix, dtype='float32')
        m = np.linspace(1, -1, n_pix, dtype='float32')
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
        w = np.exp(1j * 2 * np.pi * self.f[self._f_idx] * t_g, dtype='complex64')
        self._wg = w
        self._wgc = np.conj(w)
        return w

    def make_image2(self, n_pix=128):
        self._generate_weight_grid(n_pix)
        B = np.einsum('ijp,pq,ijq->ij', self._wg, self.data, self._wgc, optimize=True)
        return B.real
    
    def plot_image(self, img=None, n_pix=128):
        import pylab as plt
        if img is None:
            img = self.make_image(n_pix)
        plt.imshow(np.log(img), extent=(-1, 1, 1, -1), interpolation='none')
    
    def generate_gsm(self):
        import healpy as hp
        import pylab as plt
        sky = self.gsm.generate(self.f[self._f_idx] / 1e6)
        pmap = hp.orthview(sky, half_sky=True, return_projected_map=True, flip='astro')
        plt.close()
        return pmap[::-1]


