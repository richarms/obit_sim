#!/usr/bin/env python

################################################################################
# Copyright (c) 2011-2019, National Research Foundation (Square Kilometre Array)
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy
# of the License at
#
#   https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

# Produce a CASA-compatible MeasurementSet from a MeerKAT Visibility Format
# (MVF) dataset using casacore.

from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()    # noqa: E402
from builtins import zip
from builtins import range

from collections import namedtuple
import os
import tarfile
import optparse
import time
import multiprocessing
import multiprocessing.sharedctypes
import queue
import urllib.parse

import numpy as np
import dask
import numba

import katpoint
import katdal
from katdal import averager
from katdal import ms_extra
from katdal import ms_async
from katdal.sensordata import telstate_decode
from katdal.lazy_indexer import DaskLazyIndexer
from katdal.flags import NAMES as FLAG_NAMES


SLOTS = 4    # Controls overlap between loading and writing

#Fake the options needed by the writer
class options:
    verbose = False
    model_data = False


def default_ms_name(args, centre_freq=None):
    """Infer default MS name from argument list and optional frequency label."""
    # Use the first dataset in the list to generate the base part of the MS name
    url_parts = urllib.parse.urlparse(args[0], scheme='file')
    # Create MS in current working directory (strip off directories)
    dataset_filename = os.path.basename(url_parts.path)
    # Get rid of the ".full" bit on RDB files (it's the same dataset)
    full_rdb_ext = '.full.rdb'
    if dataset_filename.endswith(full_rdb_ext):
        dataset_basename = dataset_filename[:-len(full_rdb_ext)]
    else:
        dataset_basename = os.path.splitext(dataset_filename)[0]
    # Add frequency to name to disambiguate multiple spectral windows
    if centre_freq:
        dataset_basename += '_%dHz' % (int(centre_freq),)
    # Add ".et_al" as reminder that we concatenated multiple datasets
    return '%s%s.ms' % (dataset_basename, "" if len(args) == 1 else ".et_al")


def load(dataset, indices, vis, weights, flags):
    """Load data from lazy indexers into existing storage.

    This is optimised for the MVF v4 case where we can use dask directly
    to eliminate one copy, and also load vis, flags and weights in parallel.
    In older formats it causes an extra copy.

    Parameters
    ----------
    dataset : :class:`katdal.DataSet`
        Input dataset, possibly with an existing selection
    indices : tuple
        Index expression for subsetting the dataset
    vis, weights, flags : array-like
        Outputs, which must have the correct shape and type
    """
    if isinstance(dataset.vis, DaskLazyIndexer):
        DaskLazyIndexer.get([dataset.vis, dataset.weights, dataset.flags], indices,
                            out=[vis, weights, flags])
    else:
        vis[:] = dataset.vis[indices]
        weights[:] = dataset.weights[indices]
        flags[:] = dataset.flags[indices]


@numba.jit(nopython=True, parallel=True)
def permute_baselines(in_vis, in_weights, in_flags, cp_index, out_vis, out_weights, out_flags):
    """Reorganise baselines and axis order.

    The inputs have dimensions (time, channel, pol-baseline), and the output has shape
    (time, baseline, channel, pol). cp_index is a 2D array which is indexed by baseline and
    pol to get the input pol-baseline.

    cp_index may contain negative indices if the data is not present, in which
    case it is filled with 0s and flagged.

    This could probably be optimised further: the current implementation isn't
    particularly cache-friendly, and it could benefit from unrolling the loop
    over polarisations in some way
    """
    # Workaround for https://github.com/numba/numba/issues/2921
    in_flags_u8 = in_flags.view(np.uint8)
    n_time, n_bls, n_chans, n_pols = out_vis.shape
    bstep = 128
    bblocks = (n_bls + bstep - 1) // bstep
    for t in range(n_time):
        for bblock in numba.prange(bblocks):
            bstart = bblock * bstep
            bstop = min(n_bls, bstart + bstep)
            for c in range(n_chans):
                for b in range(bstart, bstop):
                    for p in range(out_vis.shape[3]):
                        idx = cp_index[b, p]
                        if idx >= 0:
                            vis = in_vis[t, c, idx]
                            weight = in_weights[t, c, idx]
                            flag = in_flags_u8[t, c, idx] != 0
                        else:
                            vis = np.complex64(0 + 0j)
                            weight = np.float32(0)
                            flag = True
                        out_vis[t, b, c, p] = vis
                        out_weights[t, b, c, p] = weight
                        out_flags[t, b, c, p] = flag
    return out_vis, out_weights, out_flags


def makems(dataset, ms_name, no_auto_corr=True):

    tag_to_intent = {'gaincal': 'CALIBRATE_PHASE,CALIBRATE_AMPLI',
                     'bpcal': 'CALIBRATE_BANDPASS,CALIBRATE_FLUX',
                     'target': 'TARGET'}


    def antenna_indices(na, no_auto_corr):
        """Get default antenna1 and antenna2 arrays."""
        return np.triu_indices(na, 1 if no_auto_corr else 0)

    def corrprod_index(dataset):
        """The correlator product index (with -1 representing missing indices)."""
        corrprod_to_index = {tuple(cp): n for n, cp in enumerate(dataset.corr_products)}

        # ==========================================
        # Generate per-baseline antenna pairs and
        # correlator product indices
        # ==========================================

        def _cp_index(a1, a2, pol):
            """Create correlator product index from antenna pair and pol."""
            a1 = "%s%s" % (a1.name, pol[0].lower())
            a2 = "%s%s" % (a2.name, pol[1].lower())
            return corrprod_to_index.get((a1, a2), -1)

        # Generate baseline antenna pairs
        ant1_index, ant2_index = antenna_indices(len(dataset.ants), no_auto_corr)
        # Order as similarly to the input as possible, which gives better performance
        # in permute_baselines.
        bl_indices = list(zip(ant1_index, ant2_index))
        bl_indices.sort(key=lambda ants: _cp_index(dataset.ants[ants[0]],
                                                   dataset.ants[ants[1]],
                                                   pols_to_use[0]))
        # Undo the zip
        ant1_index[:] = [bl[0] for bl in bl_indices]
        ant2_index[:] = [bl[1] for bl in bl_indices]
        ant1 = [dataset.ants[a1] for a1 in ant1_index]
        ant2 = [dataset.ants[a2] for a2 in ant2_index]

        # Create actual correlator product index
        cp_index = [_cp_index(a1, a2, p)
                    for a1, a2 in zip(ant1, ant2)
                    for p in pols_to_use]
        cp_index = np.array(cp_index, dtype=np.int32)

        CPInfo = namedtuple("CPInfo", ["ant1_index", "ant2_index",
                                       "ant1", "ant2", "cp_index"])
        return CPInfo(ant1_index, ant2_index, ant1, ant2, cp_index)

    # Get list of unique polarisation products in the dataset
    pols_in_dataset = np.unique([(cp[0][-1] + cp[1][-1]).upper() for cp in dataset.corr_products])

    pols_to_use = pols_in_dataset

    # Extract one MS per spectral window in the dataset(s)
    for win in range(len(dataset.spectral_windows)):

        centre_freq = dataset.spectral_windows[win].centre_freq
        print('Extract MS for spw %d: centre frequency %d Hz'
              % (win, int(centre_freq)))

        basename = os.path.splitext(ms_name)[0]

        # Discard first N dumps which are frequently incomplete
        dataset.select(spw=win, scans='track')

        # The first step is to copy the blank template MS to our desired output
        # (making sure it's not already there)
        if os.path.exists(ms_name):
            raise RuntimeError("MS '%s' already exists - please remove it "
                               "before running this script" % (ms_name,))

        print("Will create MS output in " + ms_name)

        # Are we averaging?
        average_data = False

        # Determine the number of channels
        nchan = len(dataset.channels)

        print("\nUsing %s as the reference antenna. All targets and scans "
              "will be based on this antenna.\n" % (dataset.ref_ant,))
        # MS expects timestamps in MJD seconds
        start_time = dataset.start_time.to_mjd() * 24 * 60 * 60
        end_time = dataset.end_time.to_mjd() * 24 * 60 * 60
        # MVF version 1 and 2 datasets are KAT-7; the rest are MeerKAT
        telescope_name = 'MK_SIM'

        # increment scans sequentially in the ms
        scan_itr = 1
        print("\nIterating through scans in dataset(s)...\n")

        cp_info = corrprod_index(dataset)
        nbl = cp_info.ant1_index.size
        npol = len(pols_to_use)

        field_names, field_centers, field_times = [], [], []
        obs_modes = ['UNKNOWN']
        total_size = 0

        # Create the MeasurementSet
        table_desc, dminfo = ms_extra.kat_ms_desc_and_dminfo(
            nbl=nbl, nchan=nchan, ncorr=npol, model_data=False)
        ms_extra.create_ms(ms_name, table_desc, dminfo)

        ms_dict = {}
        ms_dict['ANTENNA'] = ms_extra.populate_antenna_dict([ant.name for ant in dataset.ants],
                                                            [ant.position_ecef for ant in dataset.ants],
                                                            [ant.diameter for ant in dataset.ants])
        ms_dict['FEED'] = ms_extra.populate_feed_dict(len(dataset.ants), num_receptors_per_feed=2)
        ms_dict['DATA_DESCRIPTION'] = ms_extra.populate_data_description_dict()
        ms_dict['POLARIZATION'] = ms_extra.populate_polarization_dict(ms_pols=pols_to_use)
        ms_dict['OBSERVATION'] = ms_extra.populate_observation_dict(
            start_time, end_time, telescope_name, dataset.observer, dataset.experiment_id)

        print("Writing static meta data...")
        ms_extra.write_dict(ms_dict, ms_name)

        # Pre-allocate memory buffers
        channel_freq_width = dataset.channel_width
        dump_av = 1
        time_av = dataset.dump_period
        tsize = dump_av
        in_chunk_shape = (tsize,) + dataset.shape[1:]
        scan_vis_data = np.empty(in_chunk_shape, dataset.vis.dtype)
        scan_weight_data = np.empty(in_chunk_shape, dataset.weights.dtype)
        scan_flag_data = np.empty(in_chunk_shape, dataset.flags.dtype)

        ms_chunk_shape = (SLOTS, tsize // dump_av, nbl, nchan, npol)
        raw_vis_data = ms_async.RawArray(ms_chunk_shape, scan_vis_data.dtype)
        raw_weight_data = ms_async.RawArray(ms_chunk_shape, scan_weight_data.dtype)
        raw_flag_data = ms_async.RawArray(ms_chunk_shape, scan_flag_data.dtype)
        ms_vis_data = raw_vis_data.asarray()
        ms_weight_data = raw_weight_data.asarray()
        ms_flag_data = raw_flag_data.asarray()

        # Need to limit the queue to prevent overwriting slots before they've
        # been processed. The -2 allows for the one we're writing and the one
        # the writer process is reading.
        work_queue = multiprocessing.Queue(maxsize=SLOTS - 2)
        result_queue = multiprocessing.Queue()
        writer_process = multiprocessing.Process(
            target=ms_async.ms_writer_process,
            args=(work_queue, result_queue, options, dataset.ants, cp_info, ms_name,
                  raw_vis_data, raw_weight_data, raw_flag_data))
        writer_process.start()

        try:
            slot = 0
            for scan_ind, scan_state, target in dataset.scans():
                s = time.time()
                scan_len = dataset.shape[0]
                if scan_state != 'track':
                    continue
                if target.body_type != 'radec':
                    continue
                print("scan %3d (%4d samples) loaded. Target: '%s'. Writing to disk..."
                      % (scan_ind, scan_len, target.name))

                # Get the average dump time for this scan (equal to scan length
                # if the dump period is longer than a scan)
                dump_time_width = min(time_av, scan_len * dataset.dump_period)

                # Get UTC timestamps
                utc_seconds = dataset.timestamps[:]
                # Update field lists if this is a new target
                if target.name not in field_names:
                    # Since this will be an 'radec' target, we don't need antenna
                    # or timestamp to get the (astrometric) ra, dec
                    ra, dec = target.radec()

                    field_names.append(target.name)
                    field_centers.append((ra, dec))
                    field_times.append(katpoint.Timestamp(utc_seconds[0]).to_mjd() * 60 * 60 * 24)
                    if options.verbose:
                        print("Added new field %d: '%s' %s %s"
                              % (len(field_names) - 1, target.name, ra, dec))
                field_id = field_names.index(target.name)

                # Determine the observation tag for this scan
                obs_tag = ','.join(tag_to_intent[tag]
                                   for tag in target.tags if tag in tag_to_intent)

                # add tag to obs_modes list
                if obs_tag and obs_tag not in obs_modes:
                    obs_modes.append(obs_tag)
                # get state_id from obs_modes list if it is in the list, else 0 'UNKNOWN'
                state_id = obs_modes.index(obs_tag) if obs_tag in obs_modes else 0

                # Iterate over time in some multiple of dump average
                ntime = utc_seconds.size
                ntime_av = 0

                for ltime in range(0, ntime - tsize + 1, tsize):
                    utime = ltime + tsize
                    tdiff = utime - ltime
                    out_freqs = dataset.channel_freqs

                    # load all visibility, weight and flag data
                    # for this scan's timestamps.
                    # Ordered (ntime, nchan, nbl*npol)
                    load(dataset, np.s_[ltime:utime, :, :],
                         scan_vis_data, scan_weight_data, scan_flag_data)

                    # This are updated as we go to point to the current storage
                    vis_data = scan_vis_data
                    weight_data = scan_weight_data
                    flag_data = scan_flag_data

                    out_utc = utc_seconds[ltime:utime]

                    # Select correlator products and permute axes
                    cp_index = cp_info.cp_index.reshape((nbl, npol))
                    vis_data, weight_data, flag_data = permute_baselines(
                        vis_data, weight_data, flag_data, cp_index,
                        ms_vis_data[slot], ms_weight_data[slot], ms_flag_data[slot])

                    # Increment the number of averaged dumps
                    ntime_av += tdiff

                    # Check if writer process has crashed and abort if so
                    try:
                        result = result_queue.get_nowait()
                        raise result
                    except queue.Empty:
                        pass

                    work_queue.put(ms_async.QueueItem(
                        slot=slot, target=target, time_utc=out_utc, dump_time_width=dump_time_width,
                        field_id=field_id, state_id=state_id, scan_itr=scan_itr))
                    slot += 1
                    if slot == SLOTS:
                        slot = 0

                work_queue.put(ms_async.EndOfScan())
                result = result_queue.get()
                if isinstance(result, Exception):
                    raise result
                scan_size = result.scan_size
                s1 = time.time() - s

                if average_data and utc_seconds.shape != ntime_av:
                    print("Averaged %s x %s second dumps to %s x %s second dumps"
                          % (np.shape(utc_seconds)[0], dataset.dump_period,
                             ntime_av, dump_time_width))

                scan_size_mb = float(scan_size) / (1024**2)

                print("Wrote scan data (%f MiB) in %f s (%f MiBps)\n"
                      % (scan_size_mb, s1, scan_size_mb / s1))

                scan_itr += 1
                total_size += scan_size

        finally:
            work_queue.put(None)
            writer_exc = None
            # Drain the result_queue so that we unblock the writer process
            while True:
                result = result_queue.get()
                if isinstance(result, Exception):
                    writer_exc = result
                elif result is None:
                    break
            writer_process.join()
        # This raise is deferred to outside the finally block, so that we don't
        # raise an exception while unwinding another one.
        if isinstance(writer_exc, Exception):
            raise writer_exc

        if total_size == 0:
            raise RuntimeError("No usable data found in MVF dataset "
                               "(pick another reference antenna, maybe?)")

        ms_dict = {}
        ms_dict['SPECTRAL_WINDOW'] = ms_extra.populate_spectral_window_dict(
            out_freqs, channel_freq_width * np.ones(len(out_freqs)))
        ms_dict['FIELD'] = ms_extra.populate_field_dict(
            field_centers, field_times, field_names)
        ms_dict['STATE'] = ms_extra.populate_state_dict(obs_modes)
        ms_dict['SOURCE'] = ms_extra.populate_source_dict(
            field_centers, field_times, field_names)

        print("\nWriting dynamic fields to disk....\n")
        # Finally we write the MS as per our created dicts
        ms_extra.write_dict(ms_dict, ms_name)

