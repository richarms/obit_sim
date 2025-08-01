from datetime import datetime
import random

import numpy as np
from ephem.stars import stars

from katdal.dataset import DataSet, Subarray
from katdal import SpectralWindow
from katdal.h5datav3 import VIRTUAL_SENSORS, SENSOR_PROPS

from katdal.sensordata import SensorCache, RecordSensorGetter
from katdal.categorical import CategoricalData

import katpoint

ANTENNA_DESCRIPTIONS = [
    'm000, -30:42:39.8, 21:26:38.0, 1035.0, 13.5, -8.2435 -207.289 1.18 5872.443 5873.171, \
    0:01:23.5 0 -0:02:25.8 -0:03:01.9 0:00:42.6 -0:00:10.3 -0:11:37.7 0:03:03.5, 1.22',
    'm002, -30:42:39.8, 21:26:38.0, 1035.0, 13.5, -32.098 -224.248 1.202 5870.487 5871.204, \
    0:28:42.0 0 0:01:38.9 0:01:50.4 0:00:21.6 0:00:03.2 0:02:57.6, 1.22',
    'm005, -30:42:39.8, 21:26:38.0, 1035.0, 13.5, -102.08 -283.1155 1.454 5876.408 5876.854, \
    0:01:11.1 0 -0:00:50.9 -0:06:06.7 0:00:21.4 -0:00:10.6 -0:00:29.3, 1.22',
    'm017, -30:42:39.8, 21:26:38.0, 1035.0, 13.5, 199.641 -112.277 1.499 5868.708 5869.414, \
    0:47:17.0 0 -0:03:13.0 -0:06:45.8 0:00:42.8 0:00:05.4 0:06:26.5 0:03:35.6, 1.22',
    'm020, -30:42:39.8, 21:26:38.0, 1035.0, 13.5, 97.0245 -299.636 2.452 5852.26 5833.148, \
    -1:37:44.7 0 0:04:29.3 0:04:55.8 0:00:24.9 0:00:04.9 0:03:37.0 0:02:17.1, 1.22',
    'm021, -30:42:39.8, 21:26:38.0, 1035.0, 13.5, -295.9485 -327.2465 0.6725 5885.674 5866.658, \
    -0:22:35.3 0 0:02:41.4 -0:03:24.5 0:00:33.3 0:00:26.3 0:02:10.4 -0:04:09.4, 1.22',
    'm034, -30:42:39.8, 21:26:38.0, 1035.0, 13.5, 357.8245 -28.3215 1.515 5861.427 5861.286, \
    0:11:27.2 0 0:12:28.4 0:08:44.2 -0:00:34.1 0:01:29.7 -0:24:03.2 0:02:38.4, 1.22',
    'm041, -30:42:39.8, 21:26:38.0, 1035.0, 13.5, -287.531 -661.6855 2.5065 5870.621 5870.585, \
    0:06:55.8 0 -0:03:18.5 0:00:28.2 0:00:23.8 0:00:28.8 0:04:47.4 0:02:08.3, 1.22',
    'm042, -30:42:39.8, 21:26:38.0, 1035.0, 13.5, -361.6965 -460.3195 1.059 5870.792 5870.692, \
    -0:07:55.0 0 0:00:53.5 -0:00:10.7 -0:00:16.1 0:00:08.4 0:02:22.5 0:01:37.9, 1.22',
    'm043, -30:42:39.8, 21:26:38.0, 1035.0, 13.5, -629.8345 -128.3255 -2.1695 5860.531 5861.26, \
    0:21:49.1 0 -0:00:00.8 0:00:47.2 0:00:03.0 0:00:34.9 -0:09:30.3 0:01:51.5, 1.22',
    'm048, -30:42:39.8, 21:26:38.0, 1035.0, 13.5, -2805.594 2686.8905 -17.153 5869.018 5850.242, \
    1:21:25.0 0 0:05:18.8 0:11:29.4 0:00:17.0 0:00:05.6 0:04:21.3 -0:02:36.1, 1.22',
    'm049, -30:42:39.8, 21:26:38.0, 1035.0, 13.5, -3605.9285 436.507 -4.6515 5881.679 5882.156, \
    0:22:43.8 0 0:12:37.6 0:25:26.6 0:00:37.1 0:00:33.8 -0:08:53.6 -0:01:09.3, 1.22',
    'm050, -30:42:39.8, 21:26:38.0, 1035.0, 13.5, -2052.322 -843.685 -2.0395 5882.69 5882.889, \
    0:26:30.1 0 -0:01:18.0 -0:05:14.5 -0:00:01.8 -0:00:13.0 -0:12:05.9 0:02:41.4, 1.22',
    'm054, -30:42:39.8, 21:26:38.0, 1035.0, 13.5, 871.9915 -499.8355 5.8945 5878.312 5878.298, \
    0:01:58.0 0 -0:06:01.7 -0:03:59.5 0:00:35.1 -0:00:45.8 0:06:50.6 -0:00:58.1, 1.22',
    'm055, -30:42:39.8, 21:26:38.0, 1035.0, 13.5, 1201.798 96.4835 2.577 5849.908 5850.499, \
    -0:00:02.5 0 0:02:37.7 -0:05:55.1 -0:00:01.2 0:01:30.5 0:06:12.9 0:03:27.3, 1.22',
    'm056, -30:42:39.8, 21:26:38.0, 1035.0, 13.5, 1598.4195 466.6315 -0.502 5873.236 5854.456, \
    -0:25:01.4 0 0:02:18.3 0:07:57.5 -0:00:19.6 0:00:57.1 0:12:21.6 0:01:28.5, 1.22'
]

DEFAULT_SUBARRAYS = [{
    'antenna': ANTENNA_DESCRIPTIONS,
    # Auto-generated from 'antenna'
    # 'corr_products' : [],
}]

DEFAULT_SPWS = [{
    'centre_freq': .856e9 + .856e9 / 2.,
    'num_chans': 32768,
    'channel_width': .856e9 / 32768,
    'sideband': 1,
    'band': 'L',
}]

# Pick 10 random ephem stars as katpoint targets
_NR_OF_DEFAULT_TARGETS = 10
DEFAULT_TARGETS = [katpoint.Target("%s, star" % t) for t in
                   random.sample(stars.keys(), _NR_OF_DEFAULT_TARGETS)]

# Slew for 1 dumps then track for 4 on random targets
_SLEW_TRACK_DUMPS = (('slew', 1), ('track', 4))
DEFAULT_DUMPS = [(e, nd, t)
                 for t in DEFAULT_TARGETS
                 for e, nd in _SLEW_TRACK_DUMPS] * 20

DEFAULT_TIMESTAMPS = {
    # Auto-generated from current time
    # 'start_time': 1.0,
    'dump_period': 4.0,
}

DEFAULT_METADATA = {
    'name': "mock",
    'version': 'mock_version',
    'observer': 'ghost',
    'description':  'mock observation',
    'experiment_id': 'mock',
    'obs_params': {
        'sb_id_code': 'MOCK_SBID',
        'description': 'mock observation',
        'proposal_id': 'MOCK_PID',
        'observer': 'ghost'
        }
}


def DEFAULT_WEIGHTS(dataset):
    # Dump space is linear space between scaled dump indices
    start, stop = dataset.dumps[[0, -1]] / 1000.0
    return np.linspace(start, stop,
                       num=np.product(dataset.shape),
                       endpoint=True,
                       dtype=np.float32).reshape(dataset.shape)


def DEFAULT_FLAGS(dataset):
    flags = np.zeros(dataset.shape, dtype=bool)
    # Flag 10% of visibilities
    nflagged = int(0.1*flags.size)
    flags.ravel()[-nflagged:] = True
    return flags


def DEFAULT_VIS(dataset):
    # Visibilities of form (scan + dump*1j)
    vis = np.empty(dataset.shape, dtype=np.complex64)
    vis[:, :, :].real = np.full(dataset.shape, dataset.scan_indices[0])
    vis[:, :, :].imag = dataset.dumps[:, None, None]
    return vis


_CORR_PRODUCT_LOOKUP = {
    ('h', 'h'): 0,
    ('h', 'v'): 1,
    ('v', 'h'): 2,
    ('v', 'v'): 3
}

_CP_LOOKUP_LEN = len(_CORR_PRODUCT_LOOKUP)


def corr_product_index(a1, a2, na):
    """
    Generates a unique correlation product index given
    names for antenna one and two, as well as the number of antenna.

    Parameters
    ----------
    a1 : str
        MeerKAT antenna name, e.g. 'm006h'
    a2 : str
        MeerKAT antenna name, e.g. 'm018h'
    na : integer
        Number of antenna

    Returns
    -------
    integer
        Correlation product index
    """

    try:
        corr_id = _CORR_PRODUCT_LOOKUP[(a1[-1:], a2[-1:])]
    except KeyError:
        raise ValueError("Invalid correlation product "
                         "(%s, %s)" % (a1, a2))

    return (int(a1[:-1])*na + int(a2[:-1]))*_CP_LOOKUP_LEN + corr_id


class MockDataSet(DataSet):
    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        timestamps (optional) : dict
            Dictionary defining the start time and dump rate of the observation.
            Defaults to the current time and 4 second dump rates.
        subarrays (optional): list of dict
            List of dictionaries, each defining a subarray.
        spws (optional): list of dict
            List of dictionaries, each defining a spectral window.
            Defaults to MeerKAT 32K L band
        dumps (optional): list of tuples
            List of (event, number of dumps, target) tuples
        vis (optional): function
            Custom function defining the visibilities returned by
            the MockDataSet instance.

            The function must take a MockDataSet instance as its first
            argument and should have the following signature:

            .. code-block:: python

                def my_vis(dataset):
                    return np.zeros(dataset.shape, dtype=np.complex64)

            If you wish to pass in data orthogonal to dataset ``functools.partial``
            can be used to construct a ``callable`` with the correct signature.

            .. code-block:: python

                def my_vis(dataset, my_data):
                    shape = dataset.shape
                    ...
                    return np.array(..., dtype=np.complex64)

                ds = MockDataSet(vis=partial(my_vis, my_data=...))

        weights (optional): function
            Function from which the weights array is defined.
            Similar to vis, should create an ndarray of float32 values.
        flags (optional): function
            Function from which the vis array is defined.
            Similar to vis, should create an ndarray of boolean values.
        """
        super(MockDataSet, self).__init__(name='mock')
        self.observer = 'mock'
        self.description = "Mock observation"

        # Obtain any custom instructions for builiding
        # the mock observation
        metadata_defs = kwargs.get('metadata', DEFAULT_METADATA)
        timestamp_defs = kwargs.get('timestamps', DEFAULT_TIMESTAMPS)
        subarray_defs = kwargs.get('subarrays', DEFAULT_SUBARRAYS)
        dump_defs = kwargs.get('dumps', DEFAULT_DUMPS)
        spw_defs = kwargs.get('spws', DEFAULT_SPWS)

        self._vis_def = kwargs.get('vis', DEFAULT_VIS)
        self._flags_def = kwargs.get('flags', DEFAULT_FLAGS)
        self._weights_def = kwargs.get('weights', DEFAULT_WEIGHTS)

        self._create_metadata(metadata_defs)
        self._create_timestamps(timestamp_defs, dump_defs)

        # Now create the sensors cache now that
        # we have dump period and timestamps
        self.sensor = SensorCache({}, self._timestamps, self.dump_period,
                                  keep=self._time_keep, props=SENSOR_PROPS,
                                  virtual=VIRTUAL_SENSORS)

        self._create_subarrays(subarray_defs)
        self._create_spectral_windows(spw_defs)
        self._create_antenna_sensors(self.ants)

        try:
            ref_ant = self.ants[0]
        except IndexError:
            raise ValueError("No antenna were defined "
                             "for the default subarray")

        self._create_targets(dump_defs, ref_ant)
        self._create_scans(ref_ant, dump_defs)

        ncorrproducts = self.corr_products.shape[0]
        nchan = self.channels.shape[0]

        # Select everything upfront (weights + flags already set to all in superclass)
        self._time_keep = np.ones(self._ndumps, dtype=bool)
        self._corrprod_keep = np.ones(ncorrproducts, dtype=bool)
        self._freq_keep = np.ones(nchan, dtype=bool)
        ants = [ant.name for ant in self.ants]

        self._create_azel_sensors()

        self.select(ants=ants, spw=0, subarray=0)

    def _create_metadata(self, metadata):
        """
        Adds metadata to self

        Parameters
        ----------
        metadata : dict
            Dictionary of attributes to add to this object
        """
        for k, v in metadata.items():
            setattr(self, k, v)

    def _create_targets(self, dump_defs, ref_ant):
        """
        Iterates through dumps, adding targets to the catalogue.

        Parameters
        ----------
        dump_defs : list of tuples
            List of (event, number of dumps, target) tuples
        """
        for _, _, target in dump_defs:
            self.catalogue.add(target)

        self.catalogue.antenna = ref_ant

    def _create_timestamps(self, timestamp_defs, dump_defs):
        """
        Creates observation starting and ending times,
        as well as the dump period, indices and timestamps
        and number of dumps.

        Parameters
        ----------
        timestamp_defs : dict
            Dictionary { 'start_time' : float, 'dump_period' : float}
        dump_defs : list of tuples
            List of (event, number of dumps, target) tuples
        """
        try:
            start_time = timestamp_defs['start_time']
        except KeyError:
            epoch = datetime.utcfromtimestamp(0)
            start_time = (datetime.now() - epoch).total_seconds()

        dump_period = timestamp_defs.get('dump_period', 4.0)

        self._ndumps = ndumps = sum(nd for _, nd, _ in dump_defs)

        # Observation times
        self.start_time = katpoint.Timestamp(start_time)
        self.end_time = self.start_time + dump_period*ndumps
        self.dump_period = dump_period
        self.dumps = np.arange(ndumps)
        self._timestamps = np.arange(self.start_time.secs + (dump_period / 2.),
                                     self.end_time.secs,
                                     step=dump_period, dtype=np.float64)

    def _create_subarrays(self, subarray_defs):
        """
        Create subarrays, setting default subarray properties
        for this dataset to the first subarray.

        Parameters
        ----------
        subarray_defs : list of dicts
            List of subarray definition dictionaries
            { 'antenna' : list, 'corr_products' : list}
        """

        subarrays = []

        for subarray_def in subarray_defs:
            try:
                ants = subarray_def['antenna']
            except KeyError as e:
                raise KeyError("Subarray definition '%s' "
                               "missing '%s'" % (subarray_def, str(e)))

            ants = [a if isinstance(a, katpoint.Antenna)
                    else katpoint.Antenna(a) for a in ants]

            try:
                corr_products = subarray_def['corr_products']
            except KeyError:
                # Generate correlation products for all antenna pairs
                # including auto-correlations
                corr_products = np.array([
                    (a1.name + c1, a2.name + c2)
                    for i, a1 in enumerate(ants)
                    for a2 in ants[i:]
                    for c1 in ('h', 'v')
                    for c2 in ('h', 'v')])

            subarrays.append(Subarray(ants, corr_products))

        try:
            subarray = subarrays[0]
        except IndexError:
            raise ValueError("No subarrays were defined in '%s'" % subarray_defs)

        self.subarrays = subarrays
        self.subarray = 0
        self.inputs = subarray.inputs
        self.ants = subarray.ants
        self.corr_products = subarray.corr_products

        subarray_catdata = CategoricalData(self.subarrays, [0, self._ndumps])
        subarray_index_catdata = CategoricalData([self.subarray], [0, self._ndumps])
        self.sensor['Observation/subarray'] = subarray_catdata
        self.sensor['Observation/subarray_index'] = subarray_index_catdata

    def _create_spectral_windows(self, spw_defs):
        """
        Create spectral windows, setting the default spectral windows
        for this dataset to the first spectral window.

        Parameters
        ----------
        spw_defs : list of dictionaries
            List of dictionaries defining a spectral window
        """
        spws = []

        if not isinstance(spw_defs, (tuple, list)):
            spw_defs = [spw_defs]

        required = set(['centre_freq', 'channel_width', 'num_chans'])

        for spw_def in spw_defs:
            # Sanity check bare minimum is present
            if not required.issubset(spw_def.keys()):
                raise KeyError("Spectral Window definition '%s' "
                               "missing '%s'" % (spw_def, required))

            spws.append(SpectralWindow(**spw_def))

        try:
            spw = spws[0]
        except IndexError:
            raise ValueError("No Spectral Windows were defined")

        self.spectral_windows = spws
        self.spw = 0
        self.channel_width = spw.channel_width
        self.freqs = self.channel_freqs = spw.channel_freqs
        self.channels = np.arange(spw.num_chans)

        spw_catdata = CategoricalData(self.spectral_windows, [0, self._ndumps])
        spw_index_catdata = CategoricalData([self.spw], [0, self._ndumps])
        self.sensor['Observation/spw'] = spw_catdata
        self.sensor['Observation/spw_index'] = spw_index_catdata

    def _create_antenna_sensors(self, antenna):
        """
        Create antenna sensors.

        Parameters
        ----------
        antenna : list of :class:`katpoint.Antenna`
            Antenna objects
        """

        for ant in antenna:
            ant_catdata = CategoricalData([ant], [0, self._ndumps])
            self.sensor['Antennas/%s/antenna' % (ant.name,)] = ant_catdata

        # Extract array reference from first antenna (first 5 fields of description)
        array_ant_fields = ['array'] + antenna[0].description.split(',')[1:5]
        array_ant = katpoint.Antenna(','.join(array_ant_fields))
        self.sensor['Antennas/array/antenna'] = CategoricalData([array_ant], [0, self._ndumps])

    def _create_azel_sensors(self):

        for ant in self.ants:
            self.select()
            # Azimuth Elevation sensors per antenna
            az = np.recarray((self._ndumps,), dtype=[('timestamp', float), ('value', float)])
            el = np.recarray((self._ndumps,), dtype=[('timestamp', float), ('value', float)])
            for target in self.catalogue.targets:
                self.select(targets=target)
                scan_az, scan_el = target.azel(self.timestamps, ant)
                az['value'][self.dumps] = np.rad2deg(scan_az)
                el['value'][self.dumps] = np.rad2deg(scan_el)
            az['timestamp'] = self._timestamps
            el['timestamp'] = self._timestamps
            ant_az = RecordSensorGetter(az, name='Antennas/%s/pos_actual_scan_azim' % (ant.name,))
            ant_el = RecordSensorGetter(el, name='Antennas/%s/pos_actual_scan_elev' % (ant.name,))
            self.sensor['Antennas/%s/pos_actual_scan_azim' % (ant.name,)] = ant_az
            self.sensor['Antennas/%s/pos_actual_scan_elev' % (ant.name,)] = ant_el

    def _create_scans(self, ref_ant, dumps_def):
        """
        Setup reference antenna, as well as scans
        associated with it

        Parameters
        ----------
        ref_ant : :class:`katpoint.Antenna`
            Reference antenna
        dump_defs : list of tuples
            List of (event, number of dumps, target) tuples

        """
        self.ref_ant = ref_ant

        def _generate_ref_ant_compound_scans():
            """
            Divide dumps into periods of slewing and tracking at a target,
            yielding (dump_index, 'slew'/'track', target).
            """
            dump_index = 0

            for event, dumps, target in dumps_def:
                yield dump_index, event, target
                dump_index += dumps

        # Generate compound scans for the reference antenna
        ref_ant_compound_scans = _generate_ref_ant_compound_scans()

        # Labels seem to only involve tracks, separate them out
        label_scans = [tup for tup in ref_ant_compound_scans if tup[1] == 'track']
        events, values, _ = zip(*label_scans)
        label = CategoricalData(values, events + (self._ndumps,))

        # Generate dump indexes (events) and 'slew'/'track' (values)
        # and targets for the reference antenna
        events, values, targets = zip(*(_generate_ref_ant_compound_scans()))
        refant = CategoricalData(values, events + (self._ndumps,))
        # DO THIS BCOS h5datav3.py does it
        refant.add_unmatched(label.events)
        self.sensor['Antennas/%s/activity' % self.ref_ant.name] = refant

        # Derive scan states and indices from reference antenna data
        scan_index = CategoricalData(list(range(len(refant))), refant.events)

        self.sensor['Observation/scan_state'] = refant
        self.sensor['Observation/scan_index'] = scan_index

        # DO THIS BCOS h5datav3.py does it
        label.align(refant.events)

        # First track event unlikely to happen at dump 0
        if label.events[0] > 0:
            label.add(0, '')

        # Derive compound scan index from the label
        compscan_index = CategoricalData(range(len(label)), label.events)
        self.sensor['Observation/label'] = label
        self.sensor['Observation/compscan_index'] = compscan_index

        # Create categorical data for our targets
        targets = CategoricalData(targets, (events + (self._ndumps,)))
        # DO THIS BCOS h5datav3.py does it
        targets.align(refant.events)

        target_index = CategoricalData(targets.indices, targets.events)

        self.sensor['Observation/target'] = targets
        self.sensor['Observation/target_index'] = target_index

    @property
    def timestamps(self):
        return self._timestamps[self._time_keep]

    @property
    def weights(self):
        return self._weights_def(self)

    @property
    def flags(self):
        return self._flags_def(self)

    def _corr_product_indices(self):
        """
        Returns
        -------
        np.ndarray
            array of baseline ids
        """

        na = len(self.ants)

        return np.array([corr_product_index(a1, a2, na) for a1, a2
                        in self.corr_products])

    def selection_index(self, shape="TFB"):
        dim_sizes = {'T': self.shape[0],
                     'F': self.shape[1],
                     'B': self.shape[2]}

        select_shape = tuple(dim_sizes[d] for d in shape)
        size_mults = np.asarray(tuple(dim_sizes[d] for d in shape[1:]) + (1,))
        size_mults = size_mults[::-1].cumprod()[::-1]

        index = np.zeros(shape=select_shape, dtype=np.int32)

        for i, (dim, mult) in enumerate(zip(shape, size_mults)):
            broadcast = tuple(slice(None) if i == j else None
                              for j in range(len(shape)))

            if dim == 'T':
                dim_index = np.arange(dim_sizes['T'])
            elif dim == 'F':
                dim_index = np.arange(dim_sizes['F'])
            elif dim == 'B':
                dim_index = self._corr_product_indices()
            else:
                raise ValueError("Invalid dimension '%s'" % dim)

            index += dim_index[broadcast]*mult

        return index

    @property
    def vis(self):
        return self._vis_def(self)
