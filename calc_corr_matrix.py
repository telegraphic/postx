import pyuvdata
import numpy


def calc_corr_matrix( uvfitsfile = "chan_204_20190818T120912.uvfits" , single_channel=-1 , n_ants=256, n_chan=32, n_pols=2, skip_ch=4, ignore_missing=False , outdir="./" ) :
   global g_debug_level

   phase_vis_fits = os.path.basename(uvfitsfile).replace(".uvfits","_corr_phase.fits" )

#   n_ants=256
   n_inputs = n_ants*2
   n_baselines=((n_inputs*(n_inputs-1))/2)
#   n_chan=32
#   n_pols=2
#   skip_ch=4

   pol=0

   UV = pyuvdata.UVData() 
   corr_matrix=numpy.zeros( (n_ants,n_ants) , dtype=numpy.complex64 )
   
   if os.path.exists( uvfitsfile ) :
      UV.read( uvfitsfile, file_type='uvfits') 

      hdu_phases = pyfits.PrimaryHDU()
      hdu_phases.data = numpy.zeros( (n_ants,n_ants))


      for ant1 in range(0,n_ants):
         for ant2 in range(ant1,n_ants):
             bl = UV.antnums_to_baseline(ant1, ant2) 
             bl_ind = numpy.where(UV.baseline_array == bl)[0]
       
             mean_vis =  0
             count    = 0
             
             start_range = skip_ch
             stop_range  = (n_chan-skip_ch)
             if single_channel >= 0 :
                start_range = single_channel
                stop_range  = start_range + 1
                print("DEBUG : single channel requested -> forced channel range to be %d - %d" % (start_range,stop_range))


             # if there is only one channel just use this channel
             if UV.data_array.shape[2] == 1 :
                start_range = 0
                stop_range  = 1
             
             for ch in range(start_range,stop_range) :
                 if single_channel<0 or ch == single_channel :
                     l = len(UV.data_array[bl_ind, 0, ch, pol])
                     if l > 0 :            
                        mean_vis += UV.data_array[bl_ind, 0, ch, pol][0]
                        count += 1
                     else :
                        print "WARNING : 0 values for antenna1 = %d, antenna2 = %d , ch = %d , pol = %d (bl_ind=%s)" % (ant1,ant2,ch,pol,bl_ind)

             if count > 0 :       
                mean_vis = mean_vis / count
                if g_debug_level > 0 :
                    print "Baseline %d-%d visibility calculated from %d channels" % (ant1,ant2,count)
             else :
                print "WARNING : 0 values for baseline %d-%d" % (ant1,ant2)
       
             corr_matrix[ant1,ant2] = mean_vis
             hdu_phases.data[ant1,ant2] = numpy.angle(corr_matrix[ant1,ant2])*(180.00/math.pi)
             if ant1 != ant2 :
                corr_matrix[ant2,ant1] = mean_vis.conjugate()             
                hdu_phases.data[ant2,ant1] = numpy.angle(corr_matrix[ant2,ant1])*(180.00/math.pi) 

      hdulist = pyfits.HDUList([hdu_phases])
      full_path=outdir + "/" + phase_vis_fits
      print "Saving file %s" % (full_path)
      hdulist.writeto( full_path,overwrite=True)

             
      if g_debug_level > 1 : 
         for ant1 in range(0,n_ants):
            line = ""
            for ant2 in range(0,n_ants):
               if ant1<16 and ant2<16 :            
                  re = corr_matrix[ant1,ant2].real
                  im = corr_matrix[ant1,ant2].imag
           
                  line +=  ("%06.2f+j%06.2f " % (re,im))
            
            print line      
   else :
      if ignore_missing :
         print "WARNING : uvfits file %s does not exist, trying to continue ..." % (uvfitsfile)
         corr_matrix=numpy.ones( (n_ants,n_ants) , dtype=numpy.complex64 )
      else :
         print "ERROR : uvfits file %s does not exist !!! -> cannot continue !" % (uvfitsfile)
         os.sys.exit(-1)
         
             
   return corr_matrix             
