from postx.aperture_array import ApertureArray
import numpy as np

from astropy.constants import c
from astropy.coordinates import SkyCoord
LIGHT_SPEED = c.value

import matplotlib as mpl
import pylab as plt


##################
# HELPER FUNCTIONS
##################

def db(x):
    """ Return dB value of power magnitude """
    return 10*np.log10(x)


def window2d(lmn: np.array, sigma: float=1) -> np.array:
    """ Apply 2D-Gaussian window to lmn data """
    w2d = (1/np.sqrt(2*np.pi*sigma**2))
    w2d = w2d * np.exp(-(1/(2*sigma**2)) * (lmn[..., :2]**2).sum(axis=2))
    return w2d


def fft_2d_4pol(x: np.array, NFFT: int):
    """ Apply 2D FFT with FFT shift to 4-pol data """
    x_shifted = np.fft.ifftshift(x, axes=(0, 1))
    return np.fft.ifftshift(np.fft.ifft2(x_shifted, axes=(0, 1)), axes=(0, 1))


#################
## CORE JISNU CAL
#################

def generate_aperture_image(beam_corr: np.array, lm_matrix: np.array, sigma: float=1.0, NFFT:int=513) -> np.array:
    """ Generate aperture illumination from beam correlation (far-field E-pattern)

    Args:
        beam_corr (np.array): Beam cross-correlation between a calibrator source and a grid of
                              beams (spanning the range in lm_matrix arg). This is a measure of
                              the far-field electric field pattern.
                              Shape: (N_v, N_v) and dtype: complex64
        lm_matrix (np.array): A grid of direction cosine unit vectors, where N_v is
                              the number of pointings across the grid, each with a (l, m, n) value.
                              This can be generated with coord_utils.generate_lmn_grid().
                              Shape: (N_v, N_v, N_lmn=3), dtype: float
                              TODO: Why does it work better with unphysical l,m > 1?
        sigma (float): Sets width of Gaussian window applied before FFT. Defaults to 1.0
        NFFT (int): Number of points in the 2D FFT (and final image). Defaults to 513.
    """
    # Apply 2D gaussian window to lm_matrix
    w2d = window2d(lm_matrix, sigma=sigma)
    Ndim = beam_corr.shape[0]

    # Apply windowing and padding to beam_corr
    pad_len = int((NFFT-Ndim)/2)
    beam_corr_windowed = np.einsum('ij,ijp->ijp', w2d, beam_corr)
    beam_corr_windowed_padded = np.pad(beam_corr_windowed, ((pad_len, pad_len), (pad_len, pad_len), (0, 0)), 'constant')

    # Generate aperture image from 2D FFT (on all 4 pols)
    aperture_image = fft_2d_4pol(beam_corr_windowed_padded, NFFT)

    return aperture_image


def jishnu_cal(aa: ApertureArray, cal_src: SkyCoord,
               abs_max: int=2, aperture_padding: float=3, NFFT: int=1025,
               min_baseline_len=None, gc=None,
               bad_antenna_thr: float=10) -> dict:
    """ Calibrate aperture array data using self-holography

    Implentation based on J. Thekkeppattu et al. (2024)
    https://ui.adsabs.harvard.edu/abs/2024RaSc...5907847T/abstract

    Computes gain and phase calibration via self-holography on visibility data.
    A post-correlation beam is formed toward a strong calibrator source,
    then cross-correlated against a grid of beams (also formed from visibilities).

    The resulting 'beam correlation' is a measure of the far-field electric field
    pattern. A Fourier transform of the electric field pattern yields the
    'aperture illumination', which shows the gain and phase of each antenna.

    Args:
        aa (ApertureArray): Aperture array object (contains antenna positions
                            and data).
        cal_src (SkyCoord): Calibration source to use, such as the Sun. The source
                            needs to be much brighter than the background.
        abs_max (int): Maximum value of direction cosines (l, m). Default value 2.
                       Used to form a grid of beams across bounds (-abs_max, abs_max).
                       Note that values where sqrt(l^2 + m^2) > 1 are unphysical
                       (i.e. (l,m,n) is no longer a unit vector!), but we gain resolution
                       for reasons that Jishnu will explain at some stage.
        aperture_padding (float): Padding, in meters, around the maximum extents
                                  of the antenna array. For example, 3 meters
                                  around a 35 meter AAVS2 station.
                                  Sets the dimensions of the returned aperture
                                  illumination.
        NFFT (int): Size of FFT applied to produce aperture illumination. Default 1025.
                    Values of 2^N + 1 are the best choice. Data will be zero-padded before
                    the FFT is applied, which improves resolution in final image.

    Returns:
        cal = {

            'lmn_grid': Grid of direction cosines l,m,n. Shape (N_pix, N_pix, N_pol=4)
            'beam_corr': beam correlation, complex-valued (N_pix, N_pix, N_pol=4)
            'aperture_img': aperture illumination, complex-valued (NFFT, NFFT, N_pol=4)
            'meas_corr': measured correlations agains source for each antenna
                         (N_ant, N_pol=4)
            'vis_matrix': Visibility matrix, complex-valued (N_ant, N_ant, N_pol=4)
            'aperture_size': width of aperture plane in meters
        }

    Citation:
        Implentation based on J. Thekkeppattu et al. (2024)
        https://ui.adsabs.harvard.edu/abs/2024RaSc...5907847T/abstract
    """
    # Compute number of pixels required
    aperture_dim = np.max(aa.xyz_enu) - np.min(aa.xyz_enu) + aperture_padding
    λ = LIGHT_SPEED / aa.f.to('Hz').value
    n_pix  = 2 * int(abs_max / (λ / aperture_dim)) + 1

    # Generate visibility matrix (N_ant, N_ant, N_pol=4)
    V = aa.generate_vis_matrix()

    if min_baseline_len:
        short_bls = aa.bl_matrix < min_baseline_len
        V[short_bls] = 0

    # Generate a grid of pointing vectors, and lmn direction cosine coords
    pv_grid, lmn = aa.generate_weight_grid(n_pix, abs_max=abs_max, nan_below_horizon=False)

    # Compute the phase vector required to phase up to the cal source
    pv_src = aa.generate_phase_vector(cal_src, coplanar=True)

    # Compute the measured correlation between the source and each antenna
    # This has shape (N_ant, N_pol=4)
    meas_corr_src = np.einsum('i,ilp,l->ip', pv_src[0], V, np.conj(pv_src[0]), optimize='optimal')

    # Compute the beam correlation (a measure of far-field radiation pattern)
    # beam_corr will have shape (N_pix, N_pix, N_pol=4)
    beam_corr = np.einsum('alm,ap->lmp', pv_grid, meas_corr_src, optimize='optimal')

    # Finally, compute the aperture illumination
    # aperture_img will have shape (NFFT, NFFT, N_pol=4)
    aperture_img = generate_aperture_image(beam_corr, lmn, 1.0, NFFT)

    cal = {
        'beam_corr': beam_corr,
        'aperture_img': aperture_img,
        'meas_corr': meas_corr_src,
        'lmn_grid': lmn,
        'vis_matrix': V,
        'aperture_size': (λ / (lmn[0,0,0] - lmn[0,1,0]))[0],
        'n_pix': n_pix,
        'bad_antenna_thr': bad_antenna_thr,
    }

    # Run find_bad_antennas before and after gaincal
    # (This flags any antennas with silly gain solutions)
    if aa._in_workspace('cal'):
        print("reusing existing bad antenna flags")
        cal['bad_antennas'] = aa.workspace['cal']['bad_antennas']

    cal = compute_gaincal(aa, cal)
    cal = find_bad_antennas(aa, cal, bad_antenna_thr)

    return cal


###########################
## POST JISHNU-CAL ROUTINES
###########################

def find_bad_antennas(aa: ApertureArray, cal: dict, power_thr_db: float=10) -> dict:
    """ Find antennas with power too low, or too high

        Flag antennas with:
            ant_power < median_power  - power_thr (in dB)
            ant_power > median_power  + power_thr (in dB)

        Args:
            aa (ApertureArray): Aperture Array to use
            cal (dict): Calibration dictionary from jishnu_cal
            power_thr_db (float): Antennas with power_thr_db below the median
                                  Will be flagged. Default 10 dB (10x too low).

        Returns:
            bad_ants (dict): Dictionary of bad antenna identifiers and indexes.
                             Keys are 'x', 'y'
    """
    # Helper function to merge antenna sets
    def merge_sets(x, y):
        z = np.unique(np.concatenate((x, y)))
        new_z = np.setdiff1d(z, x)
        if len(new_z) > 0:
            print(f"New flagged antennas: {new_z}")
        return z

    ant_ids = aa.uvx.antennas.identifier

    if 'bad_antennas' in cal.keys():
        xx_bad_idx = cal['bad_antennas']['x']['idx']
        yy_bad_idx = cal['bad_antennas']['y']['idx']
        print(f"Existing XX flags: {xx_bad_idx}")
        print(f"Existing YY flags: {yy_bad_idx}")
    else:
        xx_bad_idx = np.array([], dtype='int32')
        yy_bad_idx = np.array([], dtype='int32')

    #if 'meas_corr' in cal.keys():
    #    print("checking meas_corr for bad antennas...")
    #    ant_power_db = db(np.abs(cal['meas_corr']))
    #    p_med = np.median(ant_power_db, axis=0)
    #    # apply threshold
    ##    is_bad  = np.abs(ant_power_db - p_med) > power_thr_db
    #
    #    # Find the indexes of the bad antennas
    #    xx_bad_idx = merge_sets(xx_bad_idx, ant_ids[is_bad[..., 0]].antenna.values)
    #    yy_bad_idx = merge_sets(yy_bad_idx, ant_ids[is_bad[..., 1]].antenna.values)

    print("checking gaincal for bad antennas...")
    # Check if any solutions fail the bad_antenna_thr test
    gc = np.abs(cal['gaincal'])
    cal_gains_db = db(gc) - np.median(db(gc[gc > 0]))

    is_bad = np.abs(cal_gains_db) > power_thr_db

    # Make a new unique list of bad antennas indexes
    xx_bad_idx = merge_sets(xx_bad_idx, ant_ids[is_bad[..., 0]].antenna.values)
    yy_bad_idx = merge_sets(yy_bad_idx, ant_ids[is_bad[..., 1]].antenna.values)

    bad_ants = {
        'x': {'name': ant_ids[xx_bad_idx].values,  'idx': xx_bad_idx},
        'y': {'name': ant_ids[yy_bad_idx].values,  'idx': yy_bad_idx}
    }

    # Zero out bad antennas
    cal['gaincal'][..., 0][bad_ants['x']['idx']]  = 0
    cal['gaincal'][..., 1][bad_ants['y']['idx']]  = 0

    cal['bad_antennas'] = bad_ants

    return cal


def compute_gaincal(aa: ApertureArray, cal: dict) -> np.array:
    """ Compute gain calibration

    Args:
        aa (ApertureArray): Aperture Array to use
        cal (dict): Calibration dictionary from jishnu_cal

    Return:
        gaincal (np.array): Numpy array of gain calibrations
    """
    ant_power_x = np.abs(cal['meas_corr'])[..., 0]
    ant_power_y = np.abs(cal['meas_corr'])[..., 3]
    p_med_x = np.median(db(ant_power_x)[ant_power_x > 0])
    p_med_y = np.median(db(ant_power_y)[ant_power_y > 0])

    ant_power_lin_x = 10**((db(ant_power_x) - p_med_x)/10)
    ant_power_lin_y = 10**((db(ant_power_y) - p_med_y)/10)
    cal_gains = 1 / np.row_stack((ant_power_lin_x, ant_power_lin_y)).T

    cal_phs = np.angle(np.conj(cal['meas_corr']))[..., [0,3]]

    cal['gaincal']  = cal_gains * np.exp(1j * cal_phs)
    return cal


####################
## PLOTTING ROUTINES
####################

def plot_aperture_illumination(aa: ApertureArray, cal: dict, vmin: float=-40, phs_range: tuple=None):
    """ Plot aperture illumnation function magnitude and phase

    Plots a 2x2 grid for aperture illumination image, showing magnitude and phase
    for X and Y polarizations.

    Args:
        aa (ApertureArray): Aperture Array to use
        cal (dict): Calibration dictionary from jishnu_cal
        vmin (float): sets vmin in dB for magnitude plot colorscale range (vmin, 0)
                      Default value is -40 (-40 dB)
        phs_range (tuple): Sets phase scale range. Two floats in degrees, e.g. (-90, 90).
                           Default value is (-180, 180) degrees

    """

    bcorr = cal['beam_corr']
    ap_ex = cal['aperture_size']
    apim  = cal['aperture_img']

    plt.figure(figsize=(10,8))
    pidxs = [0, 3]
    extent = [-ap_ex/2, ap_ex/2, ap_ex/2, -ap_ex/2]
    for ii in range(2):
        pidx = pidxs[ii]
        img_db = 10 * np.log10(np.abs(apim[..., pidx]))
        img_db -= np.max(img_db)
        plt.subplot(2,2,2*ii+1)
        plt.title(f"{aa.p[pidx]} Magnitude")
        plt.imshow(img_db, cmap='inferno', vmin=vmin, extent=extent)
        plt.xlabel("E (m)")
        plt.ylabel("N (m)")
        plt.colorbar(label="dB")

        img_phs = np.rad2deg(np.angle(apim[..., pidx]))

        # Mask out areas where magnitude is low, so phase plot is cleaner
        phs_mask = img_db < -20
        img_phs = np.ma.array(img_phs, mask=phs_mask)

        # Create colormap for phase data
        phs_cmap = mpl.colormaps.get_cmap("viridis").copy()
        phs_cmap.set_bad(color='black')
        # set plotting limits, override if phs_range is set
        phs_range = (-180, 180) if phs_range is None else phs_range

        plt.subplot(2,2,2*ii+2)
        plt.title(f"{aa.p[pidx]} Phase]")
        plt.imshow(img_phs, cmap=phs_cmap, vmin=phs_range[0], vmax=phs_range[1], extent=extent)
        plt.xlabel("E (m)")
        plt.ylabel("N (m)")
        plt.colorbar(label="deg")

    plt.suptitle(f"{aa.name} station holography")
    plt.tight_layout()
    plt.show()
