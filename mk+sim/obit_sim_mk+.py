#!/usr/bin/env python

import numpy as np
import os
import logging

import katpoint

from katdal.lazy_indexer import DaskLazyIndexer

from functools import partial
from scipy import constants

from mock_dataset import MockDataSet

from katacomb import pipeline_factory
from katacomb.util import (setup_aips_disks,
                           get_and_merge_args)
import katacomb.configuration as kc

import numba_vis
from vis_basic import weights_basic, flags_basic, VisConstructor

import pbeam

from static import MKAT_BEAM_OFFSET, SKA_BEAM_OFFSET

MFIMAGE_CONFIG = 'obitconf/MKAT_wide_L.yaml'
SKA_BEAM = '../MK+PBeams/SKAPBeam.npy'
MKAT_BEAM = '../MK+PBeams/MKATPBeam.npy'

log = logging.getLogger('katacomb')
def configure_logging():
    log_handler = logging.StreamHandler()
    fmt = "[%(levelname)s] %(message)s"
    log_handler.setFormatter(logging.Formatter(fmt))
    log.addHandler(log_handler)
    log.setLevel('INFO')

def nba_vis(dataset, targets, mb=None, sb=None):
    return VisConstructor(dataset, targets, mkat_beam=mb, ska_beam=sb)

def run_mfimage(ds):
    select = {'scans': 'track',
              'nif': 8,
              'corrprods': 'cross',
              'pol': 'hh,hv,vh,vv'}

    uvblavg = {'FOV': 1000.0,
               'avgFreq': 0,
               'chAvg': 1,
               'maxInt': 0.01,
               'maxFact': 1.00}

    mfimage = {'Niter': 100000,
               'maxPSCLoop': 0,
               'maxASCLoop': 0,
               'prtLv': 2,
               'Catalog': '',
               'OutlierDist': 0.,
               'antSize': 0.,
               'Robust': -0.5,
               'minFlux': 0.00002,
               'maxFBW': 0.025,
               'FOV': 1.0,
               'do3D': False,
               'Reuse': 0.,
               'autoCen': 0.005,
               'maxRealtime': -1.0,
               'doGPU': True,
               'doGPUGrid': True,
               'GPU_no': [0],
               'minFList': [0.0001, 0.00002],
               'solAInt': 1.0,
               'solAType': 'L1',
               'SolAMode': 'A&P'}

    mfimage = get_and_merge_args('mfimage', MFIMAGE_CONFIG, mfimage)

    # Dummy CB_ID and Product ID and temp fits disk
    fd = kc.get_config()['fitsdirs']
    fd += [(None, './')]
    kc.set_config(output_id='OID_INTPIX', cb_id='CBID_1PB', fitsdirs=fd, aipsdirs=[(None, '/scratch/simdisk')])

    setup_aips_disks()

    pipeline = pipeline_factory('offline', ds,
                                katdal_select=select,
                                uvblavg_params=uvblavg,
                                mfimage_params=mfimage,
                                reuse=False)

    pipeline.execute()


configure_logging()

MKP_ANTS_LOC = os.path.join('..','MK+ArrayConfig','MeerKAT+.katpoint.txt')

# Get all of the antennas as katpoint descriptions
with open(MKP_ANTS_LOC, 'r') as antfile:
    mkp_antlist = antfile.read().splitlines()

# Set up a one dump scan on a single target
target = katpoint.Target("Tilda, radec, 180, -80.0, (1.0 1.0e6 0.0)")
# Always start with a sop to get sensorcache working as first dump is useless
# #NOTE: Could put this into MockDataSet
scan = [('track', 11, target), ('stop', 1, target)] #* 11 #* 80

targets = [katpoint.Target("Tilda1, radec, 12:00:00.0, -79:15:00.0, (1.0 1.0e6 -1.0)")]

# Set up a subarray with all antennas
subar = [{'antenna': mkp_antlist}]

# Set up a spectral window
nchan = 8096 #16
spws = [{'centre_freq': .856e9 + .856e9 / 2.,
         'num_chans': nchan,
         'channel_width': .856e9 / nchan,
         'sideband': 1,
         'band': 'L',
        }]

metadata = {'name': "Sim MK+ Obs",
            'version': 'Latest',
            'observer': 'Tilda',
            'description':  'Mock observation',
            'experiment_id': 'Mock',
            'obs_params': {
            'sb_id_code': 'MOCK_SBID',
            'description': 'Mock observation',
            'proposal_id': 'MOCK_PID',
            'observer': 'Tilda'
            }
           }

timestamps = {'start_time': 1.0 + (12 * 3600.),
              'dump_period': 4}     #12*3600./100}

# Set up radially symmetric beams
mkat_beam = pbeam.PrimaryBeam(*pbeam.twod_beam_fullpol(MKAT_BEAM), offset=MKAT_BEAM_OFFSET)
ska_beam = pbeam.PrimaryBeam(*pbeam.twod_beam_fullpol(SKA_BEAM), offset=SKA_BEAM_OFFSET)
#ska_beam = pbeam.PrimaryBeam(*pbeam.airybeam(SKA_BEAM))
# Set up 2d StokesI beam
#mkat_beam = pbeam.PrimaryBeam(*pbeam.twod_Ibeam(MKAT_BEAM))
#ska_beam = pbeam.PrimaryBeam(*pbeam.twod_Ibeam(SKA_BEAM))

ds = MockDataSet(subarrays=subar,
                 spws=spws,
                 dumps=scan,
                 vis=partial(nba_vis, targets=targets, mb=mkat_beam, sb=ska_beam),
                 weights=weights_basic,
                 flags=flags_basic,
                 metadata=metadata,
                 timestamps=timestamps)

run_mfimage(ds)
