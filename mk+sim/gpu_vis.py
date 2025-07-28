from katcbfsim.rime import RimeTemplate
from katsdpsigproc import accel, resource
import numpy as np


def rime_factory(katds, targets):
    """ Set up a rime object from a katdal dataset """

    context = accel.create_some_context(False)
    queue = context.create_command_queue()
    n_antennas = len(katds.ants)
    n_baselines = n_antennas * (n_antennas + 1) // 2
    template = RimeTemplate(context, n_antennas)
    spectral_window = katds.spectral_windows[katds.spw]
    centre_frequency = spectral_window.centre_freq
    bandwidth = spectral_window.bandwidth
    n_channels = len(katds.channels)
    predict = template.instantiate(
            queue, centre_frequency, bandwidth,
            n_channels, 100000, targets, katds.ants,
            0.)
    predict.ensure_all_bound()
    predict.gain[:] = np.tile(np.identity(2, np.complex64), (n_channels, n_antennas, 1, 1))
    data = predict.buffer('out')
    host = data.empty_like()
    return predict, data, host

def run_predict(predict, target, timestamps):
    """ Run predict for the timestamps and targets selected in katds """
    host_shape = predict.slots['out'].buffer.shape
    output_data = np.empty((len(timestamps),) + host_shape[:-1], np.complex64)
    for i, timestamp in enumerate(timestamps):
        predict.set_phase_center(target)
        predict.set_time(timestamp)
        predict._update_scaled_phase()
        _update_flux_density(predict)
        predict._update_die()
        transfer_event = predict.command_queue.enqueue_marker()
        predict._run_predict()
        host = predict.slots['out'].buffer.get(predict.command_queue)
        output_data[i] = host[..., 0] + 1j * host[..., 1]
        output_collapse = host_shape[1] * host_shape[2] * host_shape[3]
    return output_data.reshape((len(timestamps), host_shape[0], output_collapse))

def _update_flux_density(predict):
        """Set the per-channel perceived flux density from the flux models of
        the sources and a simple beam model.

        This performs an **asynchronous** transfer to the GPU, and the caller
        must wait for it to complete before calling the function again.
        """

        # Maps Stokes parameters to linear feed coherencies. This is
        # basically a Mueller matrix, but shaped to give 2x2 rather than
        # 4x1 output.
        stokes = np.zeros((4, 2, 2), np.complex64)
        # I
        stokes[0, 0, 0] = 1
        stokes[0, 1, 1] = 1
        # Q
        stokes[1, 0, 0] = 1
        stokes[1, 1, 1] = -1
        # U
        stokes[2, 0, 1] = 1
        stokes[2, 1, 0] = 1
        # V
        stokes[3, 0, 1] = 1j
        stokes[3, 1, 0] = -1j
        # i is channel, j is source, k is Stokes parameter, lm are
        # indices of Jones matrices.
        np.einsum('ijk,klm->ijlm',
                  predict._flux_models, stokes,
                  out=predict._flux_density_host)
        predict._flux_density.set_async(predict.command_queue, predict._flux_density_host)