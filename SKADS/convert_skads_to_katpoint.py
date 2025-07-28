# Convert sky models from csv format from SKA shared drive into a list of katpoint Targets

import argparse
import numpy as np
import katpoint

from scipy.optimize import curve_fit
from scipy.special import binom


NUM_KATPOINT_PARMS = 10



def fit_flux_model(nu, s, nu0, sigma, sref, stokes=1, order=2):
    """
    Fit a flux model of given order from Eqn 2. of
    Obit Development Memo #38, Cotton (2014)
    (ftp.cv.nrao.edu/NRAO-staff/bcotton/Obit/CalModel.pdf):
    s_nu = s_nu0 * exp(a0*ln(nu/nu0) + a1*ln(nu/nu0)**2 + ...)

    Very rarely, the requested fit fails, in which case fall
    back to a lower order, iterating until zeroth order. If all
    else fails return the weighted mean of the components.

    Finally convert the fitted parameters to a
    katpoint FluxDensityModel:
    log10(S) = a + b*log10(nu) + c*log10(nu)**2 + ...

    Parameters
    ----------
    nu : np.ndarray
        Frequencies to fit in Hz
    s : np.ndarray
        Flux densities to fit in Jy
    nu0 : float
        Reference frequency in Hz
    sigma : np.ndarray
        Errors of s
    sref : float
        Initial guess for the value of s at nu0
    stokes : int (optional)
        Stokes of image (in AIPSish 1=I, 2=Q, 3=U, 4=V)
    order : int (optional)
        The desired order of the fitted flux model (1: SI, 2: SI + Curvature ...)
    """

    if order > 4:
        raise ValueError("katpoint flux density models are only supported up to 4th order.")
    init = [sref, -0.7] + [0] * (order - 1)
    lnunu0 = np.log(nu/nu0)
    for fitorder in range(order, -1, -1):
        try:
            popt, _ = curve_fit(obit_flux_model, lnunu0, s, p0=init[:fitorder + 1], sigma=sigma)
        except RuntimeError:
            log.warn("Fitting flux model of order %d to CC failed. Trying lower order fit." %
                     (fitorder,))
        else:
            coeffs = np.pad(popt, ((0, order - fitorder),), "constant")
            return obit_flux_model_to_katpoint(nu0, stokes, *coeffs)
    # Give up and return the weighted mean
    coeffs = [np.average(s, weights=1./(sigma**2))] + [0] * order
    return obit_flux_model_to_katpoint(nu0, stokes, *coeffs)


def obit_flux_model(lnunu0, iref, *args):
    """
    Compute model:
    (iref*exp(args[0]*lnunu0 + args[1]*lnunu0**2) ...)
    """
    exponent = np.sum([arg * (lnunu0 ** (power + 1))
                       for power, arg in enumerate(args)], axis=0)
    return iref * np.exp(exponent)


def obit_flux_model_to_katpoint(nu0, stokes, iref, *args):
    """ Convert model from Obit flux_model to katpoint FluxDensityModel.
    """
    kpmodel = [0.] * NUM_KATPOINT_PARMS
    # +/- component?
    sign = np.sign(iref)
    nu1 = 1.e6
    r = np.log(nu1 / nu0)
    p = np.log(10.)
    exponent = np.sum([arg * (r ** (power + 1))
                       for power, arg in enumerate(args)])
    # Compute log of flux_model directly to avoid
    # exp of extreme values when extrapolating to 1MHz
    lsnu1 = np.log(sign * iref) + exponent
    a0 = lsnu1 / p
    kpmodel[0] = a0
    n = len(args)
    for idx in range(1, n + 1):
        coeff = np.poly1d([binom(j, idx) * args[j - 1]
                           for j in range(n, idx - 1, -1)])
        betai = coeff(r)
        ai = betai * p ** (idx - 1)
        kpmodel[idx] = ai
    # Set Stokes +/- based on sign of iref
    # or zero in the unlikely event that iref is zero
    # I, Q, U, V are last 4 elements of kpmodel
    kpmodel[stokes - 5] = sign
    return kpmodel

parser = argparse.ArgumentParser()
parser.add_argument("csv", help="CSV file containt the SKADS model from SKA")
parser.add_argument("katpoint", help="Output file containing a list of katpoint targets")
parser.add_argument("--center-ra", default=0., type=float, help="The center Right Ascension to shift the coordinates to (deg.)")
parser.add_argument("--center-dec", default=0., type=float, help="The center Declination to shift the coordinates to (deg.)")


args = parser.parse_args()

# First row is just header info
raw_data = np.genfromtxt(args.csv, delimiter=',')[1:]

# From the header:
# Column 1: Source ID
# Column 5: RA
# Column 6: Dec.
# Columns 10-14: log10 flux density at 151, 610, 1400, 4860, 18000 MHz
print(raw_data)
source_num = raw_data[:, 0]
right_ascension = raw_data[:, 4]
declination = raw_data[:, 5]
raw_flux_density = np.power(10, raw_data[:, 9:14])
flux_freqs = np.array([151., 610., 1400., 4860., 18000.]) * 1.e6
katpoint_rows = []
for i, flux in enumerate(raw_flux_density):
	coeffs = fit_flux_model(flux_freqs, flux, 1284.e6, flux*0.001, sref=flux[2], order=3)
	coeffs_str = " ".join([str(coeff) for coeff in coeffs])
	# Reproject the positions to center_ra, center_dec
	x, y = katpoint.sphere_to_plane['SIN'](0., 0., np.deg2rad(right_ascension[i]), np.deg2rad(declination[i]))
	out_ra, out_dec = np.rad2deg(katpoint.plane_to_sphere['SIN'](np.deg2rad(args.center_ra), np.deg2rad(args.center_dec), x, y))
	# Make a katpoint Target string
	target = f'TARG_{int(source_num[i])}, radec, {out_ra}, {out_dec}, 151. 18000. {coeffs_str}'
	print(target)
	katpoint_rows.append(target)

np.savetxt(args.katpoint, katpoint_rows, fmt='%s')
