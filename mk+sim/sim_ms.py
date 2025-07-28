import numpy as np
import os
import logging

import katpoint

import dask

from katdal.lazy_indexer import DaskLazyIndexer

from functools import partial
from scipy import constants

from mock_dataset import MockDataSet

import numba_vis
from vis_basic import weights_basic, flags_basic

import pbeam


from makems import makems

SKA_BEAM = '../MK+PBeams/SKAPBeam.npy'
MKAT_BEAM = '../MK+PBeams/MKATPBeam.npy'
MKP_ANTS_LOC = os.path.join('..','MK+ArrayConfig','MeerKAT+.katpoint.txt')

def nba_vis(dataset, targets, mb=None, sb=None):
    return DaskLazyIndexer(numba_vis.numba_vis(dataset, targets, mb, sb))

# Get all of the antennas as katpoint descriptions
with open(MKP_ANTS_LOC, 'r') as antfile:
    mkp_antlist = antfile.read().splitlines()

# Set up a one dump scan on a single target
target = katpoint.Target("Tilda, radec, 0.0, -80.0, (1.0 1.0e6 0.0)")
# Always start with a sop to get sensorcache working as first dump is useless
# #NOTE: Could put this into MockDataSet
scan = [('track', 10, target)] #* 80

targets = [katpoint.Target("Tilda1, radec, 00:00:00.0, -80:15:00.0, (1.0 1.0e6 -1.5)")]

# Set up a subarray with all antennas
subar = [{'antenna': mkp_antlist}]

# Set up a spectral window
nchan = 10
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
              'dump_period': 4}

mkat_beam = pbeam.RadialPrimaryBeam(*pbeam.radial_Ibeam(MKAT_BEAM))
ska_beam = pbeam.RadialPrimaryBeam(*pbeam.radial_Ibeam(SKA_BEAM))

ds = MockDataSet(subarrays=subar,
                 spws=spws,
                 dumps=scan,
                 vis=partial(nba_vis, targets=targets, mb=None, sb=None),
                 weights=weights_basic,
                 flags=flags_basic,
                 metadata=metadata,
                 timestamps=timestamps)

outname = target.name

makems(ds, outname + '.ms')
