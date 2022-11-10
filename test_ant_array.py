from ant_array import *

def _setup():
    """ Shared setup """
    filename_ant = 'test-data/eda2/antenna_locations.txt'
    filename_data = 'test-data/eda2/20200209/chan_204_20200209T034646_vis_real.fits'

    f_mhz = np.array([159.3750])
    lat = '-26:42:11:95'
    lon = '116:40:14.93'
    elev = 500

    eda = RadioArray(lat, lon, elev, f_mhz, filename_ant)
    eda.load_fits_data(filename_data)
    return eda

def test_beamform(eda):

    w = np.ones(256, dtype='complex64')
    print(eda.beamform(w))
    eda.point(0, 0)
    print(eda.beamform())
    # (2032755.3931818656+0j)
    # (2032755.3931818656+0j)
    # (2031523.3860585701763+4.7961634663806762546e-14j)

if __name__ == "__main__":
    eda = _setup()
    
    test_beamform(eda)

