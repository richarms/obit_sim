import os
from interpolation.splines import UCGrid, filter_cubic, eval_cubic, nodes

import numpy as np
from scipy import interpolate
from scipy import constants
from scipy.special import j1

POL_MAPPING = ['hh', 'hv', 'vh', 'vv']


def radial_Ibeam(FILE):
    beam = np.load(FILE, allow_pickle=True).tolist()
    Ibeam = beam['VV'] + beam['HH']
    Ibeamnorm = Ibeam / 2.0
    centre = np.where(beam['loc'] == 0.)[0]
    # Get horizontal and vertical slices to circularise the beam
    Ibeamnorm_x = Ibeamnorm[centre].flatten()
    Ibeamnorm_y = Ibeamnorm[:, centre].flatten()
    radIbeam = np.mean((Ibeamnorm_x, Ibeamnorm_y), axis=0)
    circ_rad_Ibeam = np.mean((np.flip(radIbeam[1:centre[0] + 1]), radIbeam[centre[0]:]), axis=0)
    circ_rad_Ibeam = radIbeam[centre[0]:]
    locs = beam['loc'][centre[0]:]
    circ_interp_beam = RadialPrimaryBeam(circ_rad_Ibeam, locs, beam['refMHz'])
    outbeam = np.zeros_like(Ibeamnorm)
    for x, xoff in enumerate(beam['loc']):
        for y, yoff in enumerate(beam['loc']):
            radius = np.sqrt(xoff**2 + yoff**2)
            outbeam[x, y] = circ_interp_beam.beam_voltage_gain(radius)
    ref_freq = beam['refMHz']

    return {'I': outbeam}, beam['loc'], ref_freq

def airybeam(FILE, ellipticity=1.0, xoffset=0., yoffset=0.):
    beam = np.load(FILE, allow_pickle=True).tolist()
    circabeam = RadialAiryBeam(15.0, beam['refMHz'])
    outbeam = np.zeros_like(beam['VV'])
    for x, xoff in enumerate(beam['loc']):
        xoff = xoff + (xoffset / 3600.)
        for y, yoff in enumerate(beam['loc']):
            yoff = (yoff * ellipticity) + (yoffset / 3600.)
            radius = np.sqrt(xoff**2 + yoff**2)
            outbeam[x, y] = circabeam.beam_voltage_gain(radius)
            zerobeam = np.zeros_like(outbeam)
    return {'hh': outbeam, 'vv': outbeam, 'hv': zerobeam, 'vh': zerobeam}, beam['loc'], beam['refMHz']

def funcbeam(fwhm, ref_freq, eloffset=0., azoffset=0., squintel=0., squintaz=0., ellipticity=1.0, btype='gauss'):
    # Simulate a radially symmetric beam for the given FWHM in power beam (deg.)
    def gaussian(posns, fwhm):
        numerator = -4. * np.log(2) * posns * posns
        denom = fwhm * fwhm
        return np.exp(numerator/denom)

    def airy(posns, fwhm):
        beamoff = posns / fwhm
        numerator = j1(beamoff * 3.233)
        jinc = 2. * numerator / (beamoff * 3.233)
        jinc2 = jinc * jinc
        jinc2[np.where(posns==0.)] = 1.
        return jinc2

    def const(posns, fwhm, value=1.0):
        return(np.ones_like(posns) * value)


    if btype == 'gauss':
        bfunc = gaussian
    elif btype == 'airy':
        bfunc = airy
    elif btype == 'const':
        bfunc = const

    num_nodes = 1024
    start_off = -6.0
    end_off = 6.0
    start_off_el = -6.0 - eloffset
    end_off_el = 6.0 - eloffset
    start_off_az = -6.0 - azoffset
    end_off_az = 6.0 - azoffset
    grid = UCGrid((start_off_el, end_off_el, num_nodes), (start_off_az, end_off_az, num_nodes))
    elazpos = nodes(grid)
    elazpos[:,0] *= ellipticity
    elazpos2 = elazpos**2
    radii = np.sqrt(elazpos2[:,0] + elazpos2[:,1]).reshape(num_nodes, num_nodes)
    outbeam = np.sqrt(bfunc(radii, fwhm))
    zerobeam = np.zeros_like(outbeam)

    # Do the squinty beam - Add the squint to the first dimension of the grid
    sq_grid = UCGrid((start_off_el - squintel, end_off_el - squintel, num_nodes), (start_off_az - squintaz, end_off_az - squintaz, num_nodes))
    sq_elazpos = nodes(sq_grid)
    sq_elazpos[:,0] *= ellipticity
    sq_elazpos2 = sq_elazpos**2
    sq_radii = np.sqrt(sq_elazpos2[:,0] + sq_elazpos2[:,1]).reshape(num_nodes, num_nodes)
    sq_outbeam = np.sqrt(bfunc(sq_radii, fwhm))

    location = np.linspace(start_off, end_off, num_nodes, endpoint=True)
    ref_freq = ref_freq
    return {'hh': sq_outbeam, 'vv': outbeam, 'hv': zerobeam, 'vh': zerobeam}, location, ref_freq

def twod_Ibeam(FILE, normalise=True):
    beam = np.load(FILE, allow_pickle=True).tolist()
    Ibeam = 0.5 * (beam['HH'] + beam['VV'])
    normbeam = Ibeam / np.max(Ibeam) if normalise else Ibeam
    zerobeam = np.zeros_like(normbeam)
    return {'hh': normbeam, 'vv': normbeam, 'hv': zerobeam, 'vh': zerobeam}, beam['loc'], beam['refMHz']

def twod_beam_fullpol(FILE, normalise=False, zeroim=False):
    beam = np.load(FILE, allow_pickle=True).tolist()
    norm = {key: np.max(beam[key.upper()]) if normalise else 1.0 for key in POL_MAPPING}
    outbeam = {key: beam[key.upper()] / norm[key] for key in POL_MAPPING}
    if zeroim:
        for key in outbeam:
            outbeam[key].imag = 0.
    return outbeam, beam['loc'], beam['refMHz']


class PrimaryBeam():
    def __init__(self, voltage_pattern, location, ref_freq, offset=0., kx=2, ky=2):
        """A full pol primary beam pattern.

        Given a 2d beam voltage pattern at a particular reference
        frequency, this class provides interpolaters over frequency
        to that 2d voltage pattern. The input voltage pattern must be a dictionary
        with arrays of shape (Nx, Ny), the key indicates the type of beam (polarisation
        or Stokes).

        Object methods return the power beam or voltage beam value
        given an offset position from the phase center for each of the 4
        input polarisations.
        """
        self._reffreq = ref_freq * 1.e6
        step = location[1] - location[0]
        grid_spec = (location[0], location[-1], len(location))
        self._grid = UCGrid(grid_spec, grid_spec)
        self._raw_voltage = voltage_pattern
        self._xoff = offset 

        self._beam_coeffs_real = {key: filter_cubic(self._grid, raw_voltage.real)
                                  for key, raw_voltage in self._raw_voltage.items()}
        self._beam_coeffs_imag = {key: filter_cubic(self._grid, raw_voltage.imag)
                                  for key, raw_voltage in self._raw_voltage.items()}

    def polar_beam_voltage_gain_full_pol(self, radius=0., angle=0., frequency=None):
        """ Return interpolated value of the voltage pattern
        at the given frequency(Hz) and radius(deg)
        Angle is in degrees
        """
        # Convert scalars to arrays (all must be 1d)
        if frequency is None:
            frequency = self._reffreq
        frequency = np.atleast_1d(frequency)
        nfreqs = len(frequency)
        output = np.empty((nfreqs, 4), dtype=np.complex64)
        arad = np.deg2rad(angle)
        c = np.cos(arad)
        s = np.sin(arad)
        points = np.empty((nfreqs, 2), dtype=np.float32)
        # x = elevation, y = azimuth.
        # Rotation is anticlockwise.
        points[:, 1] = (s * radius * frequency / self._reffreq) + self._xoff
        points[:, 0] = (c * radius * frequency / self._reffreq)
        for i, key in enumerate(POL_MAPPING):
            c_real, c_imag = self._beam_coeffs_real[key], self._beam_coeffs_imag[key]
            eval_cubic(self._grid, c_real, points, output[:, i].real)
            eval_cubic(self._grid, c_imag, points, output[:, i].imag)
        # Return a 2x2 Jones matrix for the polarisations
        return output.reshape((-1, 2, 2))

    def beam_voltage_gain_full_pol(self, az, el, angle=0., frequency=None):
        """ Return interpolated value of the voltage pattern
        at the given frequency(Hz) and cartesian position
        Angle is in degrees
        """
        # Convert scalars to arrays (all must be 1d)
        if frequency is None:
            frequency = self._reffreq
        frequency = np.atleast_1d(frequency)
        nfreqs = len(frequency)
        output = np.empty((nfreqs, 4), dtype=np.complex64)
        arad = np.deg2rad(angle)
        c = np.cos(arad)
        s = np.sin(arad)
        rot = np.array([[c, -s], [s, c]])
        points = np.empty((nfreqs, 2), dtype=np.float32)
        scale = frequency / self._reffreq
        # x = elevation, y = azimuth.
        # Rotation is anticlockwise.
        points[:, 1] = (az * scale) + self._xoff
        points[:, 0] = el * scale
        points = points @ rot.T
        #print(points)
        for i, key in enumerate(POL_MAPPING):
            c_real, c_imag = self._beam_coeffs_real[key], self._beam_coeffs_imag[key]
            eval_cubic(self._grid, c_real, points, output[:, i].real)
            eval_cubic(self._grid, c_imag, points, output[:, i].imag)
        # Return a 2x2 Jones matrix for the polarisations
        return output.reshape((-1, 2, 2))

    def beam_power_gain_full_pol(self, radius=0., angle=0., frequency=None):
        """ Return interpolated value of the power pattern
        at the given frequency(Hz) and radius(deg) and angle(deg).
        """
        voltage_gain = self.beam_voltage_gain_full_pol(radius=radius, angle=angle, frequency=frequency, xoff=xoff)
        return np.abs(voltage_gain * voltage_gain.conjugate())

    def beam_voltage_gain_from_grid(self, grid, frequency=None):
        # gird is in the form returned by interpolation (start, stop, numelements)
        points = nodes(grid)
        output = np.empty((len(points), 4), dtype=np.complex64)
        if frequency is None:
            frequency = self._reffreq
        interp = np.empty_like(points)
        interp[:, 0] = (points[:, 0] * frequency / self._reffreq) + self._xoff
        interp[:, 1] = points[:, 1] * frequency / self._reffreq
        for i, key in enumerate(POL_MAPPING):
            c_real, c_imag = self._beam_coeffs_real[key], self._beam_coeffs_imag[key]
            eval_cubic(self._grid, c_real, interp, output[:, i].real)
            eval_cubic(self._grid, c_imag, interp, output[:, i].imag)
        return output.reshape((grid[0][-1], grid[1][-1], -1))

class RadialPrimaryBeam():
    def __init__(self, voltage_pattern, location, ref_freq):
        """A primary beam pattern that is radially circularly symmetric

        Given a radial beam voltage pattern at a particular reference
        frequency, this class provides interpolaters to the pattern
        at any radial offset from the phase centre.

        Parameters
        ----------
        voltage_pattern : 1d array of complex64
            Each element represents voltage of the primary beam at radial offsets
            specified in `location`.
        location : 1d array of floats
            The radial offset in degrees corresponding to each element of `voltage_pattern`
        ref_freq : float
            The reference frequency in MHz of the beam pattern.
        """
        self._reffreq = ref_freq * 1.e6
        self._rad = location
        self._voltage_pattern = voltage_pattern
        print(location.shape, voltage_pattern.shape)
        self._beam_interp = interpolate.interp1d(self._rad, self._voltage_pattern, kind='cubic', fill_value='extrapolate')

    def beam_voltage_gain(self, radius, frequency=None):
        """ Return interpolated value of the voltage pattern
        at the given frequency(Hz) and radius(deg)
        """
        if frequency is None:
            frequency = self._reffreq
        return self._beam_interp(radius * (frequency / self._reffreq))

    def beam_power_gain(self, radius, frequency=None):
        """ Return interpolated value of the power pattern
        at the given frequency(Hz) and radius(deg)
        """
        voltage_gain = self.beam_voltage_gain(radius, frequency)
        return np.abs(voltage_gain * voltage_gain.conjugate())

class RadialAiryBeam(): 
    def __init__(self, dish_diameter, ref_freq):
        self._reffreq = ref_freq * 1.e6
        reflamb = constants.c / self._reffreq
        self._FWHM = np.rad2deg(1.13 * reflamb / dish_diameter)
        radarray = np.linspace(0., 3., 1000)
        jincarray = self._jinc_radius(radarray)
        self._beam_interp = interpolate.interp1d(radarray, jincarray, fill_value='extrapolate')

    def _jinc_radius(self, rad):
        beamoff = rad / self._FWHM
        numerator = j1(beamoff * 3.233)
        jinc = 2. * numerator / (beamoff * 3.233)
        jinc2 = jinc * jinc
        jinc2[np.where(rad==0.)] = 1.
        return jinc2

    def beam_power_gain(self, radius, frequency=None):
        """ Return interpolated value of the power pattern
        at the given frequency(Hz) and radius(deg)
        """
        if frequency is None:
            frequency = self._reffreq
        return self._beam_interp(radius * frequency / self._reffreq)


    def beam_voltage_gain(self, radius, frequency=None):

        return np.sqrt(self.beam_power_gain(radius, frequency))
