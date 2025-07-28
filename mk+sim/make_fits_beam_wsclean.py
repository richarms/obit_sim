"""Convert a beam from Mattieu's .npy format to 4 fits files for MFBeam."""
import time
import concurrent.futures
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

import katpoint
from numba import jit
import pbeam

POL_MAP = {'XX': -5, 'YY': -6, 'XY': -7, 'YX': -8}
MK_POL_MAP = {'XX': 0, 'YY': 3, 'XY': 1, 'YX': 2}

def make_wcs(naxis, ctype, delt, refpix, rval):
	w = WCS(naxis=naxis)
	w.wcs.ctype = ctype
	w.wcs.cdelt = delt
	w.wcs.crpix = refpix
	w.wcs.crval = rval
	return w

def aips_timestamps(ds):
	"""Generate an array of AIPS timestamps from a :class:`katdal:DataSet`"""

	ONE_DAY_IN_SECONDS = 24*60*60.
	start = time.gmtime(ds.start_time.secs)
	start_day = time.strftime('%Y-%m-%d', start)
	midnight = time.mktime(time.strptime(start_day, '%Y-%m-%d'))
	return (ds.timestamps - midnight) / ONE_DAY_IN_SECONDS

@jit(nopython=True)
def _permute_axes(in_data):
	# Reorder from ra, dec, ant, matrix, time
	# to: freq, antenna, matrix, dec, ra
	nra, ndec, nant, nmatrix, nfreq = in_data.shape
	out = np.empty((nfreq, nant, nmatrix, ndec, nra), dtype=np.float32)
	for x in range(nra):
		for y in range(ndec):
			for a in range(nant):
				for m in range(nmatrix):
					for f in range(nfreq):
						out[f, a, m, y, x] = in_data[x, y, a, m, f]
	return out

def _interp_beam(input_beam, grid, ants, parangle, freqs):
	# Wrapper for multiprocessing
	out = np.empty((len(grid), len(grid), len(ants), 4, len(freqs)),dtype=np.float32)
	for i, azi in enumerate(grid):
		for j, elj in enumerate(grid):
			for a, ant in enumerate(ants):
				ant_type = ant.name[0]
				this_beam = input_beam[ant_type]
				beam_data = this_beam.beam_voltage_gain_full_pol(azi, elj, angle=parangle[a], frequency=freqs)
				#out[i, j, a, 0] = np.abs(beam_data[:, 0, 0])
				#out[i, j, a, 1] = 0.0
				#out[i, j, a, 2] = np.abs(beam_data[:, 1, 1])
				#out[i, j, a, 3] = 0.0
				out[i, j, a, 0] = beam_data[:, 0, 1].real
				out[i, j, a, 1] = beam_data[:, 0, 1].imag
				out[i, j, a, 2] = beam_data[:, 0, 1].real
				out[i, j, a, 3] = beam_data[:, 0, 1].imag
	return _permute_axes(out)

def _interp_beam2(input_beam, grid, parangle):
	# Wrapper for multiprocessing
	out = np.empty((len(grid), len(grid), 1, 4, len(parangle)),dtype=np.float32)
	gstart = grid[0]
	gend = grid[-1]
	for t, angle in enumerate(parangle):
		beam_data = input_beam.beam_voltage_gain_from_grid(((gstart, gend, len(grid)),) * 2, angle=angle)	
		out[:, :, 0, 0, t] = np.abs(beam_data[:, :, 0, 0])
		out[:, :, 0, 1, t] = 0.0
		out[:, :, 0, 2, t] = np.abs(beam_data[:, :, 1, 1])
		out[:, :, 0, 3, t] = 0.0
	return _permute_axes(out)

class ProgressUpdater:
    def __init__(self, num_items):
        self.num_items = num_items
        self.num_processed = 0

    def update(self, data):
        self.num_processed += 1
        print(f"Done processing {self.num_processed} of {self.num_items} inputs")

class ResultStorer:
	def __init__(self, result_array):
		self.outputs = {}
		self.results = result_array

	def init_result_store(self, future, t, a):
		self.outputs[future] = self.results[t, :, a]

	def copy_result(self, future):
		self.outputs[future][:] = future.result()
		#future._result = None


def make_wsclean_fits_beam_from_dataset2(ds, input_parms, beam_file, grid_size=256, grid_extent=(-2.0, 2.0), time_subsample=150, freq_subsample=32):
	"""Generate a FITS beam in wsclean format from the information in a katdal dataset and the .

	Inputs
	======
	ds : :class:`katdal.Dataset`
		The katdal dataset from which to obtain the metadata
	input_parms : string
		csv file listing the beam parameters as a function of frequency
	beam_file : string
		The output filename to write the beam file into
	grid_size : int (optional)
		The number of pixels in the image grid. Default: 128
	grid_extent : tuple (optional)
		A tuple defining the start and end values in degrees of the interpolated Az-el beam image.
		This is end exclusive, i.e. range is [start, stop)
		Default: (-4.0, 4.0)
	time_subsample : int
		Step size over timestamps in the dataset
	"""

	# Get parameters from Beam file
	in_beam_data = np.atleast_2d(np.loadtxt(input_parms, delimiter=',')).T
	in_freqs = in_beam_data[0]
	in_data = in_beam_data[1:]
	print(in_beam_data)
	print(in_freqs)
	print(in_data)

	# Set up frequencies
	num_freqs = len(ds.channel_freqs[::freq_subsample])
	f_width = ds.channel_width * freq_subsample
	f_start = ds.channel_freqs[0]
	freqs = ds.channel_freqs[::freq_subsample]

	# Get a beam object per output frequency
	beams = []
	in_data_freq = np.array([np.interp(freqs/1.e6, in_freqs, data) for data in in_data]).T
	for i, data in enumerate(in_data_freq/60.):
		Hsqu = data[0:2]
		Vsqu = data[2:4]
		Hwid = data[4:6]
		Vwid = data[6:8]
		beam = pbeam.PrimaryBeam(*pbeam.funcbeam_from_params(freqs[i]/1.e6, Hwid, Hsqu, Vwid, Vsqu, btype='cosine'))
		beams.append(beam)

	# Set up spatial grid
	grid_def = grid_extent + (grid_size,)
	grid, grid_delt = np.linspace(*grid_def, endpoint=False, retstep=True)
	grid_rpix = grid_size // 2
	beam_rval = grid[grid_rpix]
	im_centre = np.rad2deg(ds.catalogue.targets[0].radec())
	ra_rval = im_centre[0] - beam_rval
	dec_rval = im_centre[1] - beam_rval

	# Set up times
	num_times = len(ds.timestamps[::time_subsample])
	# TIME axis is MJD
	out_mjd = np.asarray([katpoint.Timestamp(t).to_mjd() * 24 * 60 * 60 for t in ds.timestamps[::time_subsample]])
	start_time = out_mjd[0]
	# Should be the same for MJD
	time_delt =  ds.dump_period * time_subsample

	fits_data = np.empty((num_times, num_freqs, 1, 4, grid_size, grid_size), dtype=np.float32)
	print(fits_data.shape, fits_data.size * 4 /1024/1024/1024)
	
	# results = ResultStorer(fits_data)
	# Multiprocess over timestamps (especially since that is the last axis of the output array)
	outputs = {}
	futures = []

	angles = ds.parangle[::time_subsample]

	with concurrent.futures.ProcessPoolExecutor() as pool:
		for f in range(num_freqs):
			# Submit tasks to processpool
			future = pool.submit(_interp_beam2, beams[f], grid, np.mean(angles, axis=1))
			#results.init_result_store(future, t, this_slice)
			#future.add_done_callback(results.copy_result)
			futures.append(future)
			outputs[future] = fits_data[:, f, :]
		# Get results as they arrive and put them into fits_data
		for f, future in enumerate(concurrent.futures.as_completed(futures)):
			print(f)
			outputs[future][:] = future.result()
			# Free the result from memory
			future._result = None

	t_ctype = ['RA---SIN', 'DEC--SIN', 'MATRIX  ', 'ANTENNA ', 'FREQ    ', 'TIME    ']
	t_delt = [-grid_delt, grid_delt, 1, 1, f_width, time_delt]
	t_refpix = [grid_rpix + 1, grid_rpix + 1, 1, 1, 1, 1]
	t_rval = [ra_rval, dec_rval, 0., 0., f_start, start_time]

	w = make_wcs(6, t_ctype, t_delt, t_refpix, t_rval)
	header = w.to_header()

	hdu = fits.PrimaryHDU(header=header)
	hdu.data = fits_data
	out = fits.HDUList([hdu])
	out.writeto(f'{beam_file}', overwrite=True)


def make_wsclean_fits_beam_from_dataset(ds, input_beam, beam_file, grid_size=256, grid_extent=(-1.0, 1.0), time_subsample=4, freq_subsample=1):
	"""Generate a FITS beam in wsclean format from the information in a katdal dataset.

	Inputs
	======
	ds : :class:`katdal.Dataset`
		The katdal dataset from which to obtain the metadata
	input_beam : dict
		A decitionary with keys corresponding to the first character of the antennas
		and values corresponding to :class:`beam.PrimaryBeam` for each antenna type
	beam_file : string
		The output filename to write the beam file into
	grid_size : int (optional)
		The number of pixels in the image grid. Default: 128
	grid_extent : tuple (optional)
		A tuple defining the start and end values in degrees of the interpolated Az-el beam image.
		This is end exclusive, i.e. range is [start, stop)
		Default: (-4.0, 4.0)
	time_subsample : int
		Step size over timestamps in the dataset
	"""
	num_ants = len(ds.ants)
	
	# Set up frequencies
	num_freqs = len(ds.channel_freqs[::freq_subsample])
	f_width = ds.channel_width * freq_subsample
	f_start = ds.channel_freqs[0]
	freqs = ds.channel_freqs[::freq_subsample]

	# Set up spatial grid
	grid_def = grid_extent + (grid_size,)
	grid, grid_delt = np.linspace(*grid_def, endpoint=False, retstep=True)
	grid_rpix = grid_size // 2
	beam_rval = grid[grid_rpix]
	im_centre = np.rad2deg(ds.catalogue.targets[0].radec())
	ra_rval = im_centre[0] - beam_rval
	dec_rval = im_centre[1] - beam_rval

	# Set up times
	num_times = len(ds.timestamps[::time_subsample])
	# TIME axis is MJD
	out_mjd = np.asarray([katpoint.Timestamp(t).to_mjd() * 24 * 60 * 60 for t in ds.timestamps[::time_subsample]])
	start_time = out_mjd[0]
	# Should be the same for MJD
	time_delt =  ds.dump_period * time_subsample

	fits_data = np.empty((num_times, num_freqs, num_ants, 4, grid_size, grid_size), dtype=np.float32)
	print(fits_data.shape, fits_data.size * 4 /1024/1024/1024)
	
	# results = ResultStorer(fits_data)
	# Multiprocess over timestamps (especially since that is the last axis of the output array)
	outputs = {}
	futures = []

	with concurrent.futures.ProcessPoolExecutor() as pool:
		for t in range(num_times):
			# Do antennas in chunks of 10
			chunk_size = 10
			for a in range(0, len(ds.ants), chunk_size):
				this_slice = slice(a, a + chunk_size)
				# Submit tasks to processpool
				future = pool.submit(_interp_beam, input_beam, grid, ds.ants[this_slice],
									 ds.parangle[t*time_subsample, this_slice], freqs)
				#results.init_result_store(future, t, this_slice)
				#future.add_done_callback(results.copy_result)
				futures.append(future)
				outputs[future] = fits_data[t, :, this_slice]
		# Get results as they arrive and put them into fits_data
		for future in concurrent.futures.as_completed(futures):
			outputs[future][:] = future.result()
			# Free the result from memory
			future._result = None

	t_ctype = ['RA---SIN', 'DEC--SIN', 'MATRIX  ', 'ANTENNA ', 'FREQ    ', 'TIME    ']
	t_delt = [-grid_delt, grid_delt, 1, 1, f_width, time_delt]
	t_refpix = [grid_rpix + 1, grid_rpix + 1, 1, 1, 1, 1]
	t_rval = [ra_rval, dec_rval, 0., 0., f_start, start_time]

	w = make_wcs(6, t_ctype, t_delt, t_refpix, t_rval)
	header = w.to_header()

	hdu = fits.PrimaryHDU(header=header)
	hdu.data = fits_data
	out = fits.HDUList([hdu])
	out.writeto(f'{beam_file}', overwrite=True)

def main():
	#mkat_beam = pbeam.PrimaryBeam(*pbeam.airybeam(FILE_MKAT, ellipticity=0.8))
	mkat_beam = pbeam.PrimaryBeam(*pbeam.funcbeam(0.9, 1500., squintel=2./60, squintaz=2./60, ellipticity=0.95, btype='airy'))


if __name__ == '__main__':
	main()
