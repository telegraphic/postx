import pyuvdata
import numpy
import numpy as np
import os
from astropy.io import fits as pf
from astropy.wcs import WCS

def baseline_id_to_ant_ids(bl_ids):
    """ Convert baseline ID into an antenna pair.
    Uses MIRIAD convention for antennas > 256
    Returns a tuple of antenna IDs.
    """
    ant_ids = np.zeros(shape=(bl_ids.shape[0], 2), dtype='int32')
    ant_ids[bl_ids < 65536, 0] = bl_ids[bl_ids < 65536] // 256
    ant_ids[bl_ids < 65536, 1] = bl_ids[bl_ids < 65536] % 256
    ant_ids[bl_ids > 65536, 0] = (bl_ids[bl_ids > 65536] - 65536) // 2048
    ant_ids[bl_ids > 65536, 1] = (bl_ids[bl_ids > 65536] - 65536) % 2048
    return ant_ids

def read_uvfits(filename):
    with pf.open(filename) as hdu:

        # Read data shapes
        # Feel the pain of FITS
        n_ant  = hdu[1].header['NAXIS2']
        n_chan = hdu[0].header['NAXIS4']
        n_stokes = hdu[0].header['NAXIS3']

        # convert baselines into antenna indexes
        bls = hdu[0].data['BASELINE'].astype('int32')
        ant_ids = baseline_id_to_ant_ids(bls) 
        ant_ids -= 1                                    # Convert to zero-indexed
        a0, a1 = ant_ids[:, 0], ant_ids[:, 1]

        # Load matrix
        # (dec, ra, freq, stokes, complex), where complex has shape (real, imag, weight)
        hdu_data  = hdu[0].data['DATA'][:, 0, 0, :, :, :]
        hdu_flags = hdu_data[..., 2].astype('float32')

        # Zero bad data
        hdu_data[hdu_flags <= 0] = 0
        hdu_data  = np.ascontiguousarray(hdu_data[..., :2].astype('float32'))
        hdu_data  = hdu_data.view('complex64').squeeze()

        # Create vis matrix and fill
        vis_matrix = np.zeros(shape=(n_ant, n_ant, n_chan, n_stokes), dtype='complex64')
        vis_matrix[a0, a1] = hdu_data
        vis_matrix[a1, a0] = np.conj(hdu_data)
    return vis_matrix

def read_channel_dir(filelist, integrate=True):
    """ Read a list of UVFITS files in a directory """
    _vis_matrix = read_uvfits(filelist[0])
    vis_matrix_shape = [len(filelist)] + list(_vis_matrix.shape)
    vis_matrix = np.zeros(shape=vis_matrix_shape, dtype='complex64')
    vis_matrix[0] = _vis_matrix

    for ii, filename in enumerate(filelist[1:]):
        vis_matrix[ii + 1] = read_uvfits(filename)
    
    if integrate:
        vis_matrix = vis_matrix.mean(axis=0)
        vis_matrix = vis_matrix.transpose((2, 0, 1, 3))
    else:
        vis_matrix = vis_matrix.transpose((3, 0, 1, 2, 4))
    
    return vis_matrix

def generate_freqs(uvf):
    # Compute f0 - this matches WCS.pixel_to_world()
    n_chan = uvf[0].header['NAXIS4']
    fidx   = uvf[0].header['CRPIX4']
    fc     = uvf[0].header['CRVAL4']
    fd     = uvf[0].header['CDELT4']
    f0     = (1 - fidx) * fd + fc
    f_mhz  = (np.arange(n_chan) * fd + f0) / 1e6
    return f_mhz

def uvfits_to_dict(uvf):
    dd = {
        'u': uvf[0].data['UU'],
        'v': uvf[0].data['VV'],
        'w': uvf[0].data['WW'],
        'jd': uvf[0].data['DATE'][0],
        'ra': uvf[0].header['CRVAL5'],
        'dec': uvf[0].header['CRVAL6'],
        'f_mhz': generate_freqs(uvf)
    }
    return dd

if __name__ == "__main__":
    import glob
    import h5py

    filename_out = 'corr_mat_2020.05.20.h5'
    dpath='/data/2020_05_20_midday_channel_sweep'
    first_folder = 100
    last_folder  = 450
    n_folders    = last_folder - first_folder

    # First, spider and generate a list of files
    filedict = {}
    for folder_id in range(first_folder, last_folder+1):
        subpath = f'{dpath}/{folder_id}/merged'
        uv_files = sorted(glob.glob(f'{subpath}/*.uvfits'))
        filedict[folder_id] = uv_files
    
    # Read first folder to figure out shape
    print(f"Reading first subfolder ({first_folder}/{last_folder})...")
    _v = read_channel_dir(filedict[first_folder], integrate=True)
    v0_shape = _v.shape
    n_chans  = v0_shape[0]
    print(f"\tdata shape: {_v.shape}")

    # Now, generate a HDF5 object to write to
    print(f"Creating {filename_out}")
    with h5py.File(filename_out, 'w') as h:
        # Create collated dataset
        h_data_shape = [n_folders * n_chans] + list(v0_shape)[1:]
        print(f"\tCreating dataset, shape: {h_data_shape}")
        h_data = h.create_dataset('data', shape=(h_data_shape), dtype='complex64')
        h_data[:n_chans] = _v

        for ii in range(1, n_folders+1):
            print(f"Reading subfolder ({first_folder + ii}/{last_folder})...")
            try:
                _v = read_channel_dir(filedict[first_folder + ii], integrate=True)
                print(f"\tdata shape: {_v.shape}")
                i0, i1 = ii*n_chans, (ii+1)*n_chans
                if _v.shape == v0_shape:
                    h_data[i0:i1] = _v
                else:
                    print(f"WARNING: shape mismatch, skipping {_v.shape}")
            except:
                print("ERROR: Couldn't process. Attempting to continue...")

        


