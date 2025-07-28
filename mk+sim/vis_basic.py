# Constructor class for visibilities, weights and flags
import numpy as np
from dask import array as da
from katdal.lazy_indexer import DaskLazyIndexer
from numba_vis import numba_vis
import pickle


class VisConstructor(object):
    """Make visibilities for inserting into MockDataSet

    MockDataSet needs visibilities to be ndarray-like with
    a __getitem__ a shape a len and a dtype. Only construct visibilites
    when asked based on a slice of the input array and the dataset. 

    This is intended to be instantiated whenever a call to dataset.vis[slice]
    is called, and the returned vis are only calculated for the desired slice.
    """

    def __init__(self, dataset, targets, noise=True, mkat_beam=None, ska_beam=None):

        self.dtype = np.complex64
        self.shape = dataset.shape
        self.dataset = dataset
        self.targets = targets
        self.noise = noise
        self.mkat_beam = mkat_beam
        self.ska_beam = ska_beam

    def __len__(self):
        """Length operator"""
        return self.shape[0]

    def __iter__(self):
        """Iterator"""
        for index in range(len(self)):
            yield self[index]

    def __getitem__(self, slices):
        """Compute the visibilites only on the desired slices"""
        # ALWAYS time x frequency x corr_product axes.
        ndim = 3
        keep = list(slices) if isinstance(slices, tuple) else [slices]
        # Construct blanket slices if keep is smaller than ndim
        keep = keep[:ndim] + [slice(None)] * (ndim - len(keep))
        # Pass this keep tuple to numba_vis and return the computed visibilities
        return numba_vis(self.dataset, self.targets, noise=self.noise, mkb=self.mkat_beam, skb=self.ska_beam, keep=tuple(keep))


# Simple vis, wetghts, flags for insertion into MockDataSet
def vis_basic(dataset):
    vis = np.empty(dataset.shape, dtype=np.complex64)
    vis[:, :, :].real = 1.0
    vis[:, :, :].imag = 1.0
    return DaskLazyIndexer(vis)

def weights_basic(dataset):
    chunks = (1, dataset.shape[1], dataset.shape[2])
    weights = da.ones(dataset.shape, chunks=chunks, dtype=np.float32)
    return DaskLazyIndexer(weights)

def flags_basic(dataset, mask=None, edges=0.05):
    chunks = (1, dataset.shape[1], dataset.shape[2])
    edge_cut = int(dataset.shape[1] * edges)
    if mask is None:
        mask = np.zeros(dataset.shape[1], dtype=bool)
    mask[:edge_cut] = True
    mask[-edge_cut:] = True
    flags = da.broadcast_to(mask[np.newaxis, :, np.newaxis], shape=dataset.shape, chunks=chunks)
    return DaskLazyIndexer(flags)

def vis_vcz(dataset, sources):
    """Compute (van Cittert-Zernike) visibilities for a list
    of katpoint Targets with flux density models. These are
    can be passed to MockDataSet via sources.
    """
    chunks = (1, dataset.shape[1]/2, dataset.shape[2])
    pc = dataset.catalogue.targets[0]
    out_vis = da.zeros(dataset.shape, chunks=chunks, dtype=np.complex64)
    wl = constants.c / dataset.freqs
    # uvw in wavelengths for each channel
    uvw = np.array([dataset.u, dataset.v, dataset.w])
    uvw_wl = uvw[:, :, np.newaxis, :] / wl[np.newaxis, np.newaxis, :, np.newaxis]
    for target in sources:
        flux_freq = target.flux_density(dataset.freqs/1.e6)
        lmn = np.array(pc.lmn(*target.radec()))
        n = lmn[2]
        lmn[2] -= 1.
        # uvw_wl has shape (uvw, ntimes, nchannels, nbl), move uvw to
        # the last axis before np.dot
        exponent = 2j * np.pi * da.dot(np.moveaxis(uvw_wl, 0, -1), lmn)
        out_vis += flux_freq[np.newaxis, :, np.newaxis] * da.exp(exponent) / n
    return out_vis.astype(np.complex64)