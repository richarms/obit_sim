import numba
from numba import cuda
import katpoint
import numpy as np

from functools import partial
from scipy.interpolate import interp1d
from astropy import coordinates as c
from astropy import units as u

import time

from datetime import datetime

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, *kw)
        te = time.time()
        print(method.__name__, ':', te - ts)
        return result
    return timed

# Map stokes to polarisation vectors
MUELLER = np.array([[1., 1., 0., 0.],
                    [0., 0., 1., 1j],
                    [0., 0., 1., -1j],
                    [1., -1., 0., 0.]],
                   dtype=complex)

# Map polarization to element in MUELLER x STOKES
POL = {('h', 'h'): 0,
       ('h', 'v'): 1,
       ('v', 'h'): 2,
       ('v', 'v'): 3}

MKAT_SEFD_FILE = '../STATIC/SEFD_MKAT_L.npy'
SKA_SEFD_FILE = '../STATIC/SEFD_SKA_L.npy'

#@timeit
@numba.jit(nopython=True, parallel=True)
def K_ant(k_ant, uvw, l, m, n):
    """Calculate K-Jones term per antenna.
    Calculate the K-Jones term for a point source with the
    given position (l, m) at the given wavelengths.
    The K-Jones term is the geometrical delay in the approximate
    2D Fourier transform relationship between the complex
    visibility and the sky brightness distribution
    Parameters
    ----------
    k_ant : :class:`np.ndarray`, complex, shape (ntimes, nchans, nants)
        array to update with K-Jones term
    uvw : :class:`np.ndarray`, real, shape (3, ntimes, nants)
        uvw co-ordinates of antennas
    l : float
        direction cosine, right ascension
    m : float
        direction cosine, declination
    n : float
        sqrt(1 - l**2 - m**2)
    Returns
    -------
    :class: `np.ndarray`, complex, shape (ntimes, nchans, nants)
        K-Jones term per antenna
    """
    ntimes, nchans, nants = k_ant.shape
    nm = n - 1.
    for t in numba.prange(ntimes):
        for c in range(nchans):
            for a in range(nants):
                ui, vi, wi = a, a + nants, a + (nants * 2)
                phase = ((l * uvw[t, c, ui]) + (m * uvw[t, c, vi]) + (nm * uvw[t, c, wi]))
                k_ant[t, c, a] = np.exp(2.j * np.pi * phase)
    return k_ant


#@timeit
@numba.jit(nopython=True, parallel=True)
def add_model_vis(model, k_ant, scale, baselines, blpol, n):
    """Add model visibilities to model.
    Calculate model visibilities from the per-antenna K-Jones term and source
    Stokes I flux densities and add them to the model array.
    Parameters
    ----------
    model : :class:`np.ndarray`, complex, shape (ntimes, nchans, ncorr)
        array to add model visibilities to
    k_ant : :class:`np.ndarray`, complex, shape (ntimes, nchans, nants)
        per-antenna K-Jones term
    scale : :class:`np.ndarray`, complex, shape (ntimes, nchans, nbls, 4)
        The complex amplitude scale
    baselines : int array, shape (nbl, 2)
        Mapping from baseline number to antenna pair
    blpol : int array, shape (ncorr, 2)
        Mapping from correlation product (in katdal order) to baseline (in baselines) and pol number 
    Returns
    -------
    :class: `np.ndarray`, complex, shape (ntimes, nchans, nbls)
        array with model visibilities added to it
    """
    ntimes, nchans, nbls = model.shape
    for t in numba.prange(ntimes):
        for c in range(nchans):
            for b in range(nbls):
                bl, pol = blpol[b]
                ant_1, ant_2 = baselines[bl]
                model[t, c, b] += ((k_ant[t, c, ant_1] * k_ant[t, c, ant_2].conjugate()) / n * scale[t, c, bl, pol])
    return model

#@timeit
@numba.jit(nopython=True, parallel=True)
def add_model_vis_nobeam(model, k_ant, scale, baselines, blpol, n):
    """Add model visibilities to model.
    Calculate model visibilities from the per-antenna K-Jones term and source
    Stokes I flux densities and add them to the model array.
    Parameters
    ----------
    model : :class:`np.ndarray`, complex, shape (ntimes, nchans, ncorr)
        array to add model visibilities to
    k_ant : :class:`np.ndarray`, complex, shape (ntimes, nchans, nants)
        per-antenna K-Jones term
    scale : :class:`np.ndarray`, complex, shape (nchans, 4)
        The complex amplitude scale
    baselines : int array, shape (nbl, 2)
        Mapping from baseline number to antenna pair
    blpol : int array, shape (ncorr, 2)
        Mapping from correlation product (in katdal order) to baseline (in baselines) and pol number 
    Returns
    -------
    :class: `np.ndarray`, complex, shape (ntimes, nchans, nbls)
        array with model visibilities added to it
    """
    ntimes, nchans, nbls = model.shape
    for t in numba.prange(ntimes):
        for c in range(nchans):
            for b in range(nbls):
                bl, pol = blpol[b]
                ant_1, ant_2 = baselines[bl]
                model[t, c, b] += ((k_ant[t, c, ant_1] * k_ant[t, c, ant_2].conjugate()) / n * scale[c, pol])
    return model

#@timeit
def make_empty_visibilities(ntimes, channel_freqs, corr_products, channel_width,
                            dump_period, sefd_model_mkat, sefd_model_ska):
    """Generate an array of shape (ntimes, nchans, nbaselines) containing
    complex visibilities with added noise according of given SKA and MeerKAT
    dish SEFD models.
    """
    # Generate scale at channel_freqs
    # Shanpe is ant, pol, channels (ant[0] = mkat, ant[1] = ska, pol[0] = h, pol[1] = v)
    nchans = len(channel_freqs)
    sefd_ant = np.empty((2, 2, nchans))
    scale = sefd_model_mkat['correlator_efficiency'] * np.sqrt(dump_period * channel_width * 2.0)
    sefd_ant[0, 0] = np.sqrt(np.interp(channel_freqs, sefd_model_mkat['freqs'], sefd_model_mkat['H']))
    sefd_ant[0, 1] = np.sqrt(np.interp(channel_freqs, sefd_model_mkat['freqs'], sefd_model_mkat['V']))
    sefd_ant[1, 0] = np.sqrt(np.interp(channel_freqs, sefd_model_ska['freqs'], sefd_model_ska['H']))
    sefd_ant[1, 1] = np.sqrt(np.interp(channel_freqs, sefd_model_ska['freqs'], sefd_model_ska['V']))

    ant_lookup = ['m', 's']
    pol_lookup = ['h', 'v']

    antpol1 = np.array(
        [[ant_lookup.index(ant[0]), pol_lookup.index(ant[-1])] for ant in corr_products[:, 0]],
        dtype=np.int_
    )
    antpol2 = np.array(
        [[ant_lookup.index(ant[0]), pol_lookup.index(ant[-1])] for ant in corr_products[:, 1]],
        dtype=np.int_
    )

    out = _generate_noise(ntimes, nchans, antpol1, antpol2, sefd_ant, scale)
    return out

@numba.jit(nopython=True, parallel=True)
def _generate_noise(ntimes, nchans, antpol1, antpol2, sefd_ant, scale):
    nbase = len(antpol1)
    out_vis = np.empty((ntimes, nchans, nbase), dtype=np.complex128)
    for t in numba.prange(ntimes):
        for c in range(nchans):
            for b in range(nbase):
                # Ant type is string element 0 and pol type is string element -1
                ant1, pol1 = antpol1[b]
                ant2, pol2 = antpol2[b]
                sens1 = sefd_ant[ant1, pol1, c]
                sens2 = sefd_ant[ant2, pol2, c]
                #print(c, ant1, pol1, ant2, pol2, sens1, sens2, scale)
                factor = sens1 * sens2 / scale
                out_vis[t, c, b] = (np.random.standard_normal() +
                                    1j*np.random.standard_normal()) * factor
    return out_vis

@cuda.jit(device=True)
def _matmul_2d(A, B, out):
    # Multiply 2x2 complex matricies and put result in out
    for i in range(2):
        for j in range(2):
            out[i, j] = A[i, 0] * B[0, j] + A[i, 1] * B[1, j]

@cuda.jit
def _compute_scale(beam, rot, flux, bls_lookup, out):
    # Kernel for rotating and scaling a set of 2x2 coherencies
    # by the primary beam.
    # beam: provides the voltage beam per antenna per channel
    # rot: the rotation matrix per antenna
    # flux: the choherencies per channel
    # bls_lookup: an array of shape (nbl, 2) with ant1,ant2 mappingto baseline  
    # This must be called with a 2d array of threads (nchannel, nbaselines)

    # Channel and baseline of this thread
    time, chan, base = cuda.grid(3)

    # Exit if this thread is useless to us.
    if time >= beam.shape[0] or chan >= beam.shape[1] or base >= bls_lookup.shape[0]:
        return

    # Set up some intermediate result matricies
    beam_T = cuda.local.array((2,2), numba.complex128)
    rot_T = cuda.local.array((2,2), numba.float64)
    rotated = cuda.local.array((2,2), numba.complex128)
    tmp = cuda.local.array((2,2), numba.complex128)
    ant1, ant2 = bls_lookup[base]

    # Conjugate-transpose the antenna 2 matrix
    for i in range(2):
        for j in range(2):
            beam_T[i, j] = beam[time, chan, ant2, j, i].conjugate()
            rot_T[i, j] = rot[j, i, time, ant2]

    # Compute beam[ant1]*rot[ant1]*flux*rot[ant2].T.conj*beam[ant2].T.conj
    # Rotate HV from flux based on antenna parallactic angles
    _matmul_2d(flux[chan], rot_T, tmp)
    _matmul_2d(rot[:, :, time, ant1], tmp, rotated)
    # Apply beams to HV,
    # and store in appropriate element of out
    _matmul_2d(rotated, beam_T, tmp)
    _matmul_2d(beam[time, chan, ant1], tmp, out[time, chan, base])

#@timeit
def compute_scale(beam, rot, flux, bls_lookup, out, device=0):
    """Run the CUDA kernel to rotate and scale flux coherencies.

    Parameters
    ----------
    device : int, optional
        CUDA device index to use. If the specified device is not available the
        first device (``0``) will be selected instead.
    """
    # beam: provides the voltage beam per antenna per channel
    # rot: the rotation matrix per antenna
    # flux: the coherencies per channel
    # bls_lookup: an array of shape (nbl, 2) with ant1, ant2 mapping to baseline
    # out: the output complex64 array shape (nchans, nbl, 2, 2)
    # Attempt to select the requested CUDA device. The Docker image may only
    # expose a single GPU, so fall back to device 0 if the desired index is not
    # available.
    try:
        cuda.select_device(device)
    except IndexError:
        cuda.select_device(0)
    ntimes = beam.shape[0]
    nchans = beam.shape[1]
    nbase = bls_lookup.shape[0]
    # Copy arrays to the GPU
    beam_dev = cuda.to_device(beam)
    rot_dev = cuda.to_device(rot)
    flux_dev = cuda.to_device(flux)
    bls_lookup_dev = cuda.to_device(bls_lookup)
    # Allocate output on the GPU
    result_dev = cuda.device_array((ntimes, nchans, nbase, 2, 2), dtype=np.complex128)
    tpb = 32
    bpg = (nbase + (tpb - 1)) // tpb
    # Make a grid of (nchans, nbase) threads
    _compute_scale[(ntimes, nchans, bpg), (1, 1, tpb)](beam_dev, rot_dev, flux_dev, bls_lookup_dev, result_dev)
    result_dev.copy_to_host(ary=out)
    return out


def full_pol_scale_per_antenna(parangle, flux, position, freqs, antnum, baselines, mkb=None, skb=None):
    # Work out the flux density scaling per antenna. The input flux is
    # scaled by the primary beam as a function of frequency defaulting to 1
    # if no beams are given. If only mkat is given default
    # to mkat for all beams.
    # Position is l, m coordinates in radians of target.

    az, el = np.rad2deg(position)
    az = -az
    #offset = np.rad2deg(np.sqrt(l[0]**2 + m[0]**2))
    # -l because azimuth direction of beam is opposite of RA.
    #angle = np.rad2deg(np.arctan2(m[0], -l[0]))
    # Set up our interpolators for mkat and ska beams (ska = MeerKAT if no ska beams provided)
    if mkb is not None:
        mk_interp = partial(mkb.beam_voltage_gain_full_pol, az, el, frequency=freqs)
        sk_interp = mk_interp
    if skb is not None:
        sk_interp = partial(skb.beam_voltage_gain_full_pol, az, el, frequency=freqs)

    #source_angle = (angle + parangle) % 360
    nchans, _, _ = flux.shape
    ntimes, nants = parangle.shape
    nbase = len(baselines)
    # Only loop over antennas if we have a meerkat beam to interpolate
    mk_ants = [num for name, num in antnum.items() if name[0] == 'm' and mkb]
    sk_ants = [num for name, num in antnum.items() if name[0] == 's' and mkb]
    out = np.empty((ntimes, nchans, nbase, 2, 2), dtype=np.complex128)
    bvolt = np.zeros((ntimes, nchans, nants, 2, 2), dtype=np.complex128)
    # Loop over antennas and included polarisations to get the beam voltage
    # Use to make the beam radially symmetric
    # parangle = np.zeros((ntimes, len(antnum)),dtype=np.float32)
    for t in range(ntimes):
        # Set up beam attenuation array per antenna per pol
        bvolt[t, :, :] = np.eye(2, dtype=np.complex128) 
        # Loop over dishes and replace ones with beam values if we have a beam
        # The offsets are empircally determined from the current beam models in /MKAT+beams/...
        for i in mk_ants:
            #print(f'Timestamp: {t}, MKAnt: {i}, PA: {parangle[t,i]}, Angle: {source_angle[t, i]}')
            bvolt[t, :, i] = mk_interp(angle=parangle[t, i]) #, xoff=MKAT_BEAM_OFFSET)
            #print(f'BVOLT: {bvolt[t, 0, i]}')
        # Loop over SKA dishes
        for i in sk_ants:
            #print(f'Timestamp: {t}, SKAnt: {i}')
            bvolt[t, :, i] = sk_interp(angle=parangle[t, i]) #, xoff=SKA_BEAM_OFFSET)
            #print(f'BVOLT: {bvolt[t, 0, i]}')
    # Compute the rotation matrix per antenna to rotate pol response with parallactic angle 
    # Rotate parallactic angle by 90 degrees because +X and H pol are misaligned.   
    # TODO: get the sign of this correct    
    pol_angle = np.deg2rad(parangle)    
    c, s = np.cos(pol_angle), np.sin(pol_angle) 
    rot = np.array([[c, s],[-s, c]])
    # Pass it all into something that reorders the scale into (nchan, nbls)
    # and multiplies the Jones matricies
    compute_scale(bvolt, rot, flux, baselines, out)
    return out.reshape(out.shape[0:3] + (4,))

#@timeit
def calculate_flux_density(source, frequencies):
    # Convert flux from 4x1 array of stokes parameters to a
    # 2x2 array (Jones matrix) of coherencies per [[xx,xy],[yx,yy]]
    stokes_per_channel = source.flux_density_stokes(frequencies)
    flux = stokes_per_channel @ MUELLER.T
    # Make the pol fluxes a 2x2 Jones matrix
    return flux.reshape(-1, 2, 2)


def get_bl_pol_mapping(ants, corr_products): 
    # Create 3 objects from dataset. 
    # antnum: A dict mapping from antenna name to element in the dataset.ants array 
    # baselines: (nbl, 2) An array listing all unique baseline pairs in dataset.corr products 
    # bls_lookup: (ncorr, 2) An array mapping all corrlation products in order to element in baselines and pol
    # Mapping antenna array element to antenna name 
    antnum = {ant.name: i for i, ant in enumerate(ants)} 
    # Mapping from dataset.corr_products to antenna and polarisation index 
    blslines_pol = np.array([[antnum[name1[:-1]], antnum[name2[:-1]], 
                             POL[(name1[-1], name2[-1])]]  
                             for name1, name2 in corr_products]) 
    # Get unique baselines 
    baselines, blslines_pol_lookup = np.unique(blslines_pol[:, 0:2], axis=0, return_inverse=True) 
    bls_lookup = np.array([blslines_pol_lookup, blslines_pol[:, 2]]).T  
    return antnum, baselines, bls_lookup 


def numba_vis(dataset, targets, noise=True, mkb=None, skb=None, keep=(slice(None),) * 3):

    # Select the desired timestamps, channels, corr_products
    # Ensure these are always 1d array in case of scalar indexing
    ts = np.atleast_1d(dataset.timestamps[keep[0]])
    cf_Hz = np.atleast_1d(dataset.channel_freqs[keep[1]])
    cf_MHz = cf_Hz / 1.e6
    cf_iwl = (cf_Hz / katpoint.lightspeed)
    cp = np.atleast_2d(dataset.corr_products[keep[2]])

    out_shape = (nts, ncf, ncp) = len(ts), len(cf_Hz), len(cp)
    nants = len(dataset.ants)

    # Parallactic angle should be cached in dataset
    # TODO: Do this properly (its probable that katpoint gets it wrong)
    pangle = np.atleast_2d(dataset.parangle[keep[0]])

    ants = dataset.ants

    # Now do everything that is timestamp independenent or can be vectorised
    # over the given timestamps
    # Phase center of field (This assumes only one phase center
    # and only the first is of any interest
    pc = dataset.catalogue.targets[0]
    pcp = c.SkyCoord(*pc.radec(), unit=(u.rad, u.rad))
    array_ref = dataset.ref_ant.array_reference_antenna()

    # Get required mapping based on ants and corr_products
    antnum, baselines, bls_lookup = get_bl_pol_mapping(ants, cp)

    if noise:
        # SEFD and sensitivity per baseline for noise estimate
        sefd_model_mkat = np.load(MKAT_SEFD_FILE, allow_pickle=True).tolist()
        sefd_model_ska = np.load(SKA_SEFD_FILE, allow_pickle=True).tolist()
        out_vis = make_empty_visibilities(nts, cf_Hz, cp, dataset.channel_width,
                                          dataset.dump_period, sefd_model_mkat, sefd_model_ska)
    else:
        out_vis = np.zeros((nts, ncf, ncp), dtype=np.complex128)

    # UVW calculation can be vectorised over time axis
    uvw = pc.uvw(ants, ts, array_ref)
    uvw = np.hstack(uvw)
    uvw_wl = uvw[:, np.newaxis, :] * cf_iwl[np.newaxis, :, np.newaxis]
    k_ant = np.empty((nts, ncf, nants), dtype=np.complex128)
    # Add the target model for each target to the noise
    print(f'SIM: Time Chunk {datetime.utcfromtimestamp(ts[0])} -- {datetime.utcfromtimestamp(ts[-1])} Num Targets: {len(targets)}')
    for i,targ in enumerate(targets):
        # Get the flux and the lmn of the targets
        flux_density = calculate_flux_density(targ, cf_MHz)
        l, m, n = pc.lmn(*targ.radec(), antenna=array_ref)
        k_ant = K_ant(k_ant, uvw_wl, l, m, n)
        if mkb is not None:
            scale = full_pol_scale_per_antenna(pangle, flux_density, (l, m), cf_Hz,
                                               antnum, baselines, mkb=mkb, skb=skb)
            out_vis = add_model_vis(out_vis, k_ant, scale, baselines, bls_lookup, n)
        else:
            fd_flat = flux_density.reshape((ncf, 4))
            out_vis = add_model_vis_nobeam(out_vis, k_ant, fd_flat, baselines, bls_lookup, n)
    return out_vis.astype(np.complex64)
