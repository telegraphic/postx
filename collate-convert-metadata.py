import pyuvdata
import numpy
import numpy as np
import os
from astropy.io import fits as pf

def generate_freqs(uvf):
    # Compute f0 - this matches WCS.pixel_to_world()
    n_chan = uvf[0].header['NAXIS4']
    fidx   = uvf[0].header['CRPIX4']
    fc     = uvf[0].header['CRVAL4']
    fd     = uvf[0].header['CDELT4']
    f0     = (1 - fidx) * fd + fc
    f_mhz  = (np.arange(n_chan) * fd + f0) / 1e6
    return f_mhz

def uvfits_to_dict(filename):
    uvf = pf.open(filename)
    bl_ids = hdu[0].data['BASELINE'].astype('int32')
    ant_ids = baseline_id_to_ant_ids(bls)
    
    dd = {
        'jd': uvf[0].data['DATE'][0],
        'ra': uvf[0].header['CRVAL5'],
        'dec': uvf[0].header['CRVAL6'],
        'f_mhz': generate_freqs(uvf),
        'baselines': {
            'ant0': ant_ids[:, 0],
            'ant1': ant_ids[:, 1],
            'bl_ids': bl_ids,
            'u': uvf[0].data['UU'],
            'v': uvf[0].data['VV'],
            'w': uvf[0].data['WW'],
        },
    }
    return dd

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

if __name__ == "__main__":
    import glob
    import h5py

    filename_out = 'corr_mat_metadata_2020.05.20.h5'
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
    
    with h5py.File(filename_out, 'a') as h:
        h_data_shape = h['data'].shape
        
        h_freqs = h.create_dataset('freqs', shape=(h_data_shape[0],), dtype='float64')
        h_ra    = h.create_dataset('ra', shape=(h_data_shape[0],), dtype='float64')
        h_dec   = h.create_dataset('dec', shape=(h_data_shape[0],), dtype='float64')
        h_date  = h.create_dataset('time', shape=(h_data_shape[0],), dtype='float64')
        
        g_uvw = h.create_group('baselines')
        
        # Read first file
        dd0 = uvfits_to_dict(filedict[first_folder][0])
        for ds_name in dd['baselines'].keys():
            print(f"\tCreating dataset baselines/{ds_name}")
            g_uvw.create_dataset(ds_name, data=dd[ds_name])
        
        for ii in range(1, n_folders+1):
            print(f"Reading subfolder ({first_folder + ii}/{last_folder})...")
            dd = uvfits_to_dict(filedict[first_folder][0])
            n_chans = dd['f_mhz'].shape[0]
            assert n_chans == 32
            i0, i1 = ii*n_chans, (ii+1)*n_chans
            h_freqs[i0:i1] = dd['f_mhz']
            h_ra[i0:i1] = dd['ra']
            h_dec[i0:i1] = dd['ii']
            h_date[i0:i1] = dd['time_jd']
            

        


