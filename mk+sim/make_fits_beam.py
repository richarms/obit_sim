"""Convert a beam from Mattieu's .npy format to 4 fits files for MFBeam."""
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

import pbeam

from static import SKA_BEAM_OFFSET

POL_MAP = {'XX': -5, 'YY': -6, 'XY': -7, 'YX': -8}
MK_POL_MAP = {'XX': 0, 'YY': 3, 'XY': 1, 'YX': 2}

def make_wcs(naxis, ctype, delt, refpix, rval):
	w = WCS(naxis=naxis)
	w.wcs.ctype = ctype
	w.wcs.cdelt = delt
	w.wcs.crpix = refpix
	w.wcs.crval = rval
	return w


FILE_MKAT = '../MK+PBeams/MKATPBeam.npy'
FILE_SKA = '../MK+PBeams/SKAPBeam.npy'

# Set up beams
#mkat_beam = pbeam.PrimaryBeam(*pbeam.twod_beam_fullpol(FILE_MKAT))
#mkat_beam = pbeam.PrimaryBeam(*pbeam.twod_beam_fullpol(FILE_SKA))

#mkat_beam = pbeam.PrimaryBeam(*pbeam.airybeam(FILE_MKAT, ellipticity=0.8))
mkat_beam = pbeam.PrimaryBeam(*pbeam.funcbeam(1.0, 1500., btype='airy'))

# Set up a grid of az, alt for interpolation
grid_size = 256 # Make it even to get it uniform about a zero in the middle
grid_el = (-4.0, 4.0, grid_size)
grid_az = (-4.0, 4.0, grid_size)
grid = (grid_el, grid_az)
(el, el_delt), (az, az_delt) = np.linspace(*grid_el, endpoint=False, retstep=True), np.linspace(*grid_az, endpoint=False, retstep=True)
el_zero = list(el).index(0.)
az_zero = list(az).index(0.)

# Frequency axis
num_freqs = 1
bandwidth = 856.e6
f_start = 1360.e6

#freqs, f_width = np.linspace(f_start, f_start + bandwidth, num_freqs, retstep=True)
freqs, f_width = np.linspace(f_start, f_start + bandwidth, num_freqs, endpoint=False, retstep=True)
# AIPS IF table
iftab = fits.open('../STATIC/AIPS_IF.fits')[1]
iftab.data['CH WIDTH'] = f_width
iftab.data['TOTAL BANDWIDTH'] = bandwidth

# Need to make beam-freq-pol array to lookup in the loop below
beam_data = np.empty((1, 1, num_freqs, grid_size, grid_size, 4), dtype=np.complex64)
#for f, freq in enumerate(freqs):
#	print(freq)
for i, eli in enumerate(el):
	for j, azj in enumerate(az):
		#beam_data[0, 0, f] = mkat_beam.beam_voltage_gain_from_grid(grid, freq)
		beam = mkat_beam.beam_voltage_gain_full_pol(azj, eli, angle=0., frequency=freqs)
		beam_data[0, 0, :, i, j, 0] = beam[:, 0, 0]
		beam_data[0, 0, :, i, j, 1] = beam[:, 0, 1]
		beam_data[0, 0, :, i, j, 2] = beam[:, 1, 0]
		beam_data[0, 0, :, i, j, 3] = beam[:, 1, 1]

# Set up WCS for output fits files
t_ctype = ['AZIMUTH ', 'ELEVATIO', 'FREQ    ', 'STOKES  ', 'IF      ']
t_delt = [az_delt, el_delt, f_width, 1., 1.]
t_refpix = [az_zero + 1, el_zero + 1, 1., 1., 1.]
t_rval = [0., 0., f_start, 0., 1.]
# Generate template FITS files
for p in POL_MAP:
	t_rval[3] = POL_MAP[p]
	w = make_wcs(5, t_ctype, t_delt, t_refpix, t_rval)
	header = w.to_header()
	header['TELESCOP'] = 'MeerKAT'
	header['BUNIT'] = 'voltgain'
	header['OBJECT'] = f'{p}Beam'
	header['XPXOFF'] = 0.0
	header['YPXOFF'] = 0.0
	hdu = fits.PrimaryHDU(header=header)
	hdu.data = np.abs(beam_data[..., MK_POL_MAP[p]])
	#hdu.data = beam_data[..., MK_POL_MAP[p]].real
	#hdu.data = np.ones_like(beam_data[..., MK_POL_MAP[p]], dtype=np.float32)
	out = fits.HDUList([hdu, iftab])
	out.writeto(f'{p}SYMBeam.fits', overwrite=True)
	w2 = make_wcs(5, t_ctype, t_delt, t_refpix, t_rval)
	header = w2.to_header()
	header['TELESCOP'] = 'MeerKAT'
	header['BUNIT'] = 'DEGREE'
	header['OBJECT'] = f'{p}Beam'
	header['XPXOFF'] = 0.0
	header['YPXOFF'] = 0.0
	hdu2 = fits.PrimaryHDU(header=header)
	hdu2.data = np.rad2deg(np.angle(beam_data[..., MK_POL_MAP[p]]))
	#hdu2.data = beam_data[..., MK_POL_MAP[p]].imag
	#hdu2.data = np.ones_like(beam_data[..., MK_POL_MAP[p]], dtype=np.float32)
	out = fits.HDUList([hdu2, iftab])
	out.writeto(f'{p}PhSYMBeam.fits', overwrite=True)
