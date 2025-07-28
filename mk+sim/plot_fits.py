# Functions to show fits as png in small cutouts around a centre target
import aplpy
import numpy as np
import os
from astropy.io import fits
from astropy import wcs
from matplotlib import pylab as plt

def plot_stamp(fitsfile, target, size, out, minmax=None, **gscale_kwargs):

    fo = remove_noncelestial_axes(fitsfile)
    ra, dec = np.rad2deg(target.astrometric_radec())

    # Force range scaling for pixels (pmin and pmax seem to be broken)
    if gscale_kwargs.get('vmin') is None and minmax=='psf':
        gscale_kwargs['vmin'] = np.nanmin(fo.data) * 0.98
    if gscale_kwargs.get('vmax') is None and minmax=='psf':
        gscale_kwargs['vmax'] = np.nanmax(fo.data) * 0.3
    fig = plt.figure(figsize=(10, 10,), dpi=300)
    fitsfig = aplpy.FITSFigure(fo, figure=fig)
    fitsfig.recenter(ra, dec, size)
    fitsfig.show_grayscale(interpolation='bicubic', **gscale_kwargs)
    fitsfig.add_colorbar()
    fitsfig.axis_labels.hide()
    fitsfig.tick_labels.hide()
    fitsfig.ticks.hide()
    plt.savefig(out)
    

def remove_noncelestial_axes(fitsfile):
    fh = fits.open(fitsfile)
    data = fh[0].data[0,0] # drops the size-1 axes
    header = fh[0].header
    mywcs = wcs.WCS(header).celestial
    new_header = mywcs.to_header()
    new_fh = fits.PrimaryHDU(data=data, header=new_header)
    return new_fh