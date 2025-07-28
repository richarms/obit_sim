#!/usr/bin/env python

import numpy as np
import os
import logging

import katpoint

from katdal.lazy_indexer import DaskLazyIndexer

from astropy import coordinates as c
from astropy import units as u

from functools import partial
from scipy import constants

from mock_dataset import MockDataSet

from katacomb import pipeline_factory
from katacomb.util import (setup_aips_disks,
                           get_and_merge_args)
import katacomb.configuration as kc

import numba_vis
from vis_basic import weights_basic, flags_basic, VisConstructor

from static import MKAT_BEAM_OFFSET, SKA_BEAM_OFFSET

import pbeam

MFIMAGE_CONFIG = '/obitconf/mfimage.yaml'
SKA_BEAM = '../MK+PBeams/SKAPBeam.npy'
MKAT_BEAM = '../MK+PBeams/MKATPBeam.npy'
MASKFILE = './sim.mask'

log = logging.getLogger('katacomb')
def configure_logging():
    log_handler = logging.StreamHandler()
    fmt = "[%(levelname)s] %(message)s"
    log_handler.setFormatter(logging.Formatter(fmt))
    log.addHandler(log_handler)
    log.setLevel('INFO')

def nba_vis(dataset, targets, noise=True, mb=None, sb=None):
    return VisConstructor(dataset, targets, noise=noise, mkat_beam=mb, ska_beam=sb)

def run_mfimage(ds):
    select = {'scans': 'track',
              'nif': 8,
              'corrprods': 'cross',
              'pol': 'hh,vv'}

    uvblavg = {'FOV': 1000.0,
               'avgFreq': 0,
               'chAvg': 1,
               'maxInt': 0.01,
               'maxFact': 1.00}

    mfimage = {'doGPU': True,
               'doGPUGrid': True,
               'GPU_no': [0],
               'Niter': 100000,
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
               'autoCen': 0.004,
               'maxRealtime': -1.0,
               'doGPU': True,
               'minFList': [0.0003, 0.00008],
               'solPMode': 'A&P',
               'solAInt': 1.0,
               'solAType': 'L1',
               'SolAMode': 'A&P',
               'Stokes': 'I'
               #'CLEANFile': 'sim.mask'
               }

    mfimage = get_and_merge_args('mfimage', MFIMAGE_CONFIG, mfimage)

    # Dummy CB_ID and Product ID and temp fits disk
    fd = kc.get_config()['fitsdirs']
    fd += [(None, './')]
    kc.set_config(output_id='OID_INTPIX', cb_id='CENTER', fitsdirs=fd, aipsdirs=[(None, '/scratch/simdisk')])

    setup_aips_disks()

    pipeline = pipeline_factory('offline', ds,
                                katdal_select=select,
                                uvblavg_params=uvblavg,
                                mfimage_params=mfimage,
                                reuse=False)

    pipeline.execute()


configure_logging()

MKP_ANTS_LOC = os.path.join('..','MK+ArrayConfig','MeerKAT+.katpoint.txt')
TARGET_FILE = os.path.join('..','SKADS','out100uJy.katpoint')


# Get all of the antennas as katpoint descriptions
with open(MKP_ANTS_LOC, 'r') as antfile:
    mkp_antlist = antfile.read().splitlines()
model_ant = katpoint.Antenna(mkp_antlist[0])

# Set up a one dump scan on a single target
target = katpoint.Target("Tilda, radec, 0.0, -80.0, (1.0 1.0e6 0.0)")
#target = katpoint.Target("Tilda, radec, 0:03:07.25, -80:55:12.4, (1.0 1.0e6 0.0)")
# Always start with a sop to get sensorcache working as first dump is useless
# #NOTE: Could put this into MockDataSet
scan = [('track', int(40), target), ('stop', 2, target)] #* 13 #* 80

targets = [katpoint.Target("Tilda1, radec, 00:00:00.0, -80:00:00.0, (1.0 1.0e6 -1.0)")]
with open(TARGET_FILE) as f:
    all_targs = f.read().splitlines()

cenra, cendec = 1.9089, -79.585297
scale = 0.3 / np.cos(np.deg2rad(-79.585297))

targets = []

for targ in all_targs:
    this_target = katpoint.Target(targ)
    sep = np.rad2deg(this_target.separation(target, antenna=model_ant))
    if sep < 1.0 and this_target.flux_density(1284.) < 0.004:
        ra, dec = np.rad2deg(this_target.radec())
        if dec < (cendec + 0.3) and dec > (cendec - 0.3):
          if ra < (cenra + scale) and ra > (cenra - scale):
            targets.append(this_target)
        if this_target.flux_density(1284.) > 0.1:
            print(this_target, sep, this_target.flux_density(1284.))

#targets.append(katpoint.Target("Tilda1, radec, 00:00:00.0, -80:00:00.0, (1.0 1.0e6 -1.0)"))
print(f'Num Targets : {len(targets)}')
targets += [katpoint.Target('TARG_59986373, radec, 0.5906296442064175, -79.5419558069681, 151. 18000. 1.2213072394842004 -0.7594591940442245 0.003644170857582036 -0.00041258726234065876 0.0 0.0 1.0 0.0 0.0 0.0')]
#targets = [katpoint.Target("Tilda1, radec, 00:00:00.0, -80:00:00.0, (1.0 1.0e6 -1.0)")]
#targets = [katpoint.Target('TARG_36157896, radec, 0.09371835714646977, -79.84108668637126, 151. 18000. 2.1103439872581613 -0.7536182214529148 0.001211834159882686 -0.00012375561491840553 0.0 0.0 1.0 0.0 0.0 0.0')]
# Generate mask with this list of targets
with open(MASKFILE, 'w') as f:
  for i in targets:
      coord = c.SkyCoord(*i.radec(), unit=(u.rad, u.rad), frame='icrs')
      strout = f'{coord.to_string("hmsdms", sep=" ", precision=1)} 14\n'
      f.write(strout)

# Set up a subarray with all antennas
subar = [{'antenna': mkp_antlist}]

# Set up a spectral window
nchan = 32

mask=None

spws = [{'centre_freq': .856e9 + .856e9 / 2.,
         'num_chans': nchan,
         'channel_width': (.856e9 * 0.9) / nchan,
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

timestamps = {'start_time': 1.0,
              'dump_period': 12.*3600/40}

# Set up radially symmetric beams
#ska_beam = pbeam.PrimaryBeam(*pbeam.airybeam(SKA_BEAM))

# Set up full beams
mkat_beam = pbeam.PrimaryBeam(*pbeam.twod_beam_fullpol(MKAT_BEAM), offset=MKAT_BEAM_OFFSET)
ska_beam = pbeam.PrimaryBeam(*pbeam.twod_beam_fullpol(SKA_BEAM), offset=SKA_BEAM_OFFSET)



ds = MockDataSet(subarrays=subar,
                 spws=spws,
                 dumps=scan,
                 vis=partial(nba_vis, targets=targets, noise=True, mb=None, sb=None),
                 weights=weights_basic,
                 flags=partial(flags_basic, mask=mask),
                 metadata=metadata,
                 timestamps=timestamps)

run_mfimage(ds)
#makems(ds, 'nobeam.ms')
