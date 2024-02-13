from postx.aperture_array import ApertureArray
from postx.coord_utils import gaincal_vec_to_matrix
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


def jishnu_selfholo(aa: ApertureArray, cal_src: SkyCoord,
               abs_max: int=4, aperture_padding: float=3, NFFT: int=1025,
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


def meas_corr_to_phasecal(mc: np.array) -> np.array:
    """ Compute phase calibration coefficients from meas_corr

    Args:
        meas_corr (np.array): Measured correlations between reference beam.
                              and each antenna.
                              Shape (N_ant, N_pol=2), dtype complex.
    Returns:
        pc (np.array): Phase calibration solutions.
                       Shape (N_ant, N_pol=2), dtype complex
    """
    mc_xy = mc[..., [0, 3]]

    # Gain Calibration
    gc = 1 / np.abs(mc_xy)
    gc = np.ma.array(gc, mask=gc > np.median(gc) * 10)
    gc = np.ma.array(gc, mask=gc > np.median(gc.compressed()) * 10)

    # Phase calibration
    phs_ang = np.ma.array(np.angle(np.conj(mc_xy)), mask=gc.mask)
    phs_ang -= np.median(phs_ang)
    pc = np.exp(1j * phs_ang)
    pc[gc.mask] = 0
    pc.mask = gc.mask

    return pc


def jishnu_phasecal(aa: ApertureArray, cal_src: dict,
                    n_iter_max: int=50, target_phs_std: float=1.0):
    """ Iteratively apply Jishnu Cal phase calibration

    Args:
        aa (ApertureArray):
        cal_src (SkyCoord):
        n_iter_max (int): Maximum number of iterations. Default 50
        target_phs_std (float): Target phase STDEV (in deg) at which to stop iterating.

    Returns:
        cc_dict (dict): Phase calibration solution and runtime info, in dictionary with keys:
                        'phs_cal': complex-valued np.array of phase corrections.
                                   Shape: (N_ant, N_pol=2), complex data.
                        'n_iter': Number of iterations before break point reached.
                        'phs_std': np.array of STD reached at each iteration.
                                   Shape (N_iter), dtype float.
    """
    # Compute the phase vector required to phase up to the cal source
    pv_src = aa.generate_phase_vector(cal_src, coplanar=True)

    # Generate visibility matrix from aa
    V = aa.generate_vis_matrix()

    # Now, we loop over n_iter, until the STD of phase stops decreasing
    phs_iter_list = np.zeros(n_iter_max)
    for ii in range(n_iter_max):
        if ii == 0:
            meas_corr_src = np.einsum('i,ilp,l->ip', pv_src[0], V,
                                  np.conj(pv_src[0]), optimize='optimal')
            cc    = meas_corr_to_phasecal(meas_corr_src)

            cc_mat = gaincal_vec_to_matrix(cc)
            phs_std = np.std(np.angle(cc))
        else:
            meas_corr_iter = np.einsum('i,ilp,l->ip', pv_src[0], V * cc_mat,
                                       np.conj(pv_src[0]), optimize='optimal')
            cc_iter     = meas_corr_to_phasecal(meas_corr_iter)

            phs_std_iter = np.std(np.angle(cc_iter))
            if phs_std_iter >= phs_std:
                print(f"Iter {ii-1}: Iteration phase std minima reached, breaking")
                break
            elif target_phs_std > np.rad2deg(phs_std_iter):
                print(f"Iter {ii-1}: Target phase std reached, breaking")
                break
            else:
                cc_iter_mat = gaincal_vec_to_matrix(cc_iter)
                cc_mat *= cc_iter_mat

                # Update phs_std comparator
                phs_std = phs_std_iter
        # Log phase stdev iteration to
        phs_iter_list[ii] = phs_std
    cc_dict = {
        'phs_cal': cc_iter * cc,
        'n_iter': ii,
        'phs_std': phs_iter_list[:ii]
    }
    return cc_dict

def plot_jishnu_phasecal_iterations(cc_dict: dict):
    """ Plot the iterative phase corrections applied in phasecal

    Args:
        cc_dict (dict): Output dict from jishnu_phasecal
    """
    phs_std = np.rad2deg(cc_dict['phs_std'])
    plt.loglog(phs_std)
    plt.xlabel("Jishnu cal iteration")
    plt.ylabel("Phase STD [deg]")
    plt.ylim(phs_std[-1] / 1.5, phs_std[0])

####################
## PLOTTING ROUTINES
####################

def ant_xyz_to_image_idx(xyz_enu: np.array, cal: dict, as_int: bool=True) -> tuple:
    """ Convert an ENU antenna location to a pixel location in image

    Args:
        xyz_enu (np.array): Antenna positions, in meters, ENU
        cal (dict): Calibration dictionary from jishnu_cal

    Returns:
        an_x, an_y (np.array, np.array): Antenna locations in image. If as_int=True,
                                         these are rounded to nearest integer.

    Notes:
        Not currently used in plotting, as setting plt.imshow(extent=) keyword
        allows actual antenna positions in meters to be used.
    """
    NFFT = cal['aperture_img'].shape[0]
    an_x = (NFFT/2) + (xyz_enu[:, 0])*NFFT/cal['aperture_size'] + 1
    an_y = (NFFT/2) - (xyz_enu[:, 1])*NFFT/cal['aperture_size'] - 1
    if as_int:
        return np.round(an_x).astype('int32'), np.round(an_y).astype('int32')
    else:
        return an_x, an_y


def plot_aperture(aa: ApertureArray, cal: dict, pol_idx: int=0, plot_type: str='mag',
                  vmin: float=-40, phs_range: tuple=None, annotate: bool=False, s: int=None):
    """ Plot aperture illumnation for given polarization index

    Args:
        aa (ApertureArray): Aperture Array to use
        cal (dict): Calibration dictionary from jishnu_cal
        pol_idx (int): Polarization axis index. Default 0 (X-pol)
        plot_type (str): Either 'mag' for magnitude, or 'phs' for phase
        vmin (float): sets vmin in dB for magnitude plot colorscale range (vmin, 0)
                      Default value is -40 (-40 dB)
        phs_range (tuple): Sets phase scale range. Two floats in degrees, e.g. (-90, 90).
                           Default value is (-180, 180) degrees
        annotate (bool): Set to True to overlay antenna identifiers/names
        s (int): Sets circle size around antenna locations
    """
    bcorr = cal['beam_corr']
    ap_ex = cal['aperture_size']
    apim  = cal['aperture_img']

    # Compute normalized magnitude
    img_db = 10 * np.log10(np.abs(apim[..., pol_idx]))
    img_db -= np.max(img_db)

    # Mask out areas where magnitude is low, so phase plot is cleaner
    img_phs = np.rad2deg(np.angle(apim[..., pol_idx]))
    phs_mask = img_db < -20
    img_phs = np.ma.array(img_phs, mask=phs_mask)

    # Create colormap for phase data
    phs_cmap = mpl.colormaps.get_cmap("viridis").copy()
    phs_cmap.set_bad(color='black')

    # set plotting limits, override if phs_range is set
    phs_range = (-180, 180) if phs_range is None else phs_range
    extent = [-ap_ex/2, ap_ex/2, ap_ex/2, -ap_ex/2]

    if plot_type == 'mag':
        plt.title(f"{aa.p[pol_idx]} Magnitude")
        plt.imshow(img_db, cmap='inferno', vmin=vmin, extent=extent)
        plt.colorbar(label="dB")
    elif plot_type == 'phs':
        plt.title(f"{aa.p[pol_idx]} Phase]")
        plt.imshow(img_phs, cmap=phs_cmap, vmin=phs_range[0], vmax=phs_range[1], extent=extent)
        plt.colorbar(label='deg')
    else:
        raise RuntimeError("Need to plot 'mag' or 'phs'")
    plt.xlabel("E (m)")
    plt.ylabel("N (m)")

    if annotate:
        ix, iy = aa.xyz_enu[:, 0], -aa.xyz_enu[:, 1]
        ixy = np.column_stack((ix, iy))
        for ii in range(aa.n_ant):
            plt.annotate(text=aa.uvx.antennas.identifier[ii].values, xy=ixy[ii], color='white', size=8)

        circle_size = s if s else cal['aperture_img'].shape[0] / np.sqrt(aa.n_ant)
        plt.scatter(ix, iy, s=circle_size, facecolors='none', edgecolors='white', alpha=0.7)


def plot_aperture_xy(aa: ApertureArray, cal: dict, vmin: float=-40,
                               phs_range: tuple=None, annotate: bool=False, figsize: tuple=(10,8)):
    """ Plot aperture illumnation function magnitude and phase, both polarizations

    Plots a 2x2 grid for aperture illumination image, showing magnitude and phase
    for X and Y polarizations.

    Args:
        aa (ApertureArray): Aperture Array to use
        cal (dict): Calibration dictionary from jishnu_cal
        vmin (float): sets vmin in dB for magnitude plot colorscale range (vmin, 0)
                      Default value is -40 (-40 dB)
        phs_range (tuple): Sets phase scale range. Two floats in degrees, e.g. (-90, 90).
                           Default value is (-180, 180) degrees
        annotate (bool): Set to True to overlay antenna identifiers/names
        figsize (tuple): Size of figure, passed to plt.figure(figsize). Default (10, 8)

    """
    plt.figure(figsize=figsize)
    pidxs = [0, 3]
    for ii in range(2):
        pidx = pidxs[ii]

        plt.subplot(2,2,2*ii+1)
        plot_aperture(aa, cal, pidx, 'mag', vmin=vmin, annotate=annotate)

        plt.subplot(2,2,2*ii+2)
        plot_aperture(aa, cal, pidx, 'phs', vmin=vmin, annotate=annotate)

    plt.suptitle(f"{aa.name} station holography")
    plt.tight_layout()
