
import logging
import os
from shutil import copy, SameFileError

import katacomb.continuum_pipeline as cp
import katacomb.configuration as kc

import multiprocessing
from pretty import pretty

from katacomb.aips_path import next_seq_nr, path_exists

from katacomb import uv_factory, task_factory, export_images
from katacomb.util import log_obit_err

log = logging.getLogger('katacomb')

def run_mfbeam(self, uv_path, uv_sources):
        """
        Run the MFImage task
        """

        with uv_factory(aips_path=uv_path, mode="r") as uvf:
            merge_desc = uvf.Desc.Dict

        # Run MFImage task on merged file,
        out_kwargs = uv_path.task_output_kwargs(name='',
                                                aclass=cp.IMG_CLASS,
                                                seq=0)
        out2_kwargs = uv_path.task_output2_kwargs(name='',
                                                  aclass=cp.UV_CLASS,
                                                  seq=0)

        workdir = kc.get_config()['fitsdirs'][-1][-1]


        for i in self.beams:
            try:
                copy(i, workdir)
            except SameFileError:
                pass

        beamroot = os.path.basename(i)
        beamroot = 'SS' + beamroot[2:]
        beam_kwargs = {'in3DType': 'FITS',
                       'in3Disk': 0,
                       'in3File': beamroot}

        imager_kwargs = {}
        # Setup input file
        imager_kwargs.update(uv_path.task_input_kwargs())
        # Output file 1 (clean file)
        imager_kwargs.update(out_kwargs)
        # Output file 2 (uv file)
        imager_kwargs.update(out2_kwargs)
        imager_kwargs.update(beam_kwargs)
        imager_kwargs.update({
            'nThreads': multiprocessing.cpu_count(),
            'prtLv': self.prtlv,
            'Sources': uv_sources,
            'doPhase': False
        })

        # Finally, override with default parameters
        imager_kwargs.update(self.imager_params)

        log.info("MFBeam arguments %s" % pretty(imager_kwargs))

        imager = task_factory("MFBeam", **imager_kwargs)

        # Send stdout from the task to the log
        #with log_obit_err(log):
        imager.go()

def attach_SN_tables_to_image(self, uv_file, image_file):
        """
        Loop through each of the SN tables in uv_file that
        were produced by MFImage and copy and attach these to the
        image_file.
        Parameters
        ----------
        uv_file    : :class:`AIPSPath`
            UV file output from MFImage with SN tables attached.
        image_file : :class:`AIPSPath`
            Image (MA) file output from MFImage
        """

        uvf = uv_factory(aips_path=uv_file, mode='r')
        if uvf.exists:
            # Get all SN tables in UV file
            tables = uvf.tablelist
            taco_kwargs = {}
            taco_kwargs.update(uv_file.task_input_kwargs())
            taco_kwargs.update(image_file.task_output_kwargs())
            taco_kwargs['inTab'] = 'AIPS SN'
            taco_kwargs['nCopy'] = 1
            # Copy all SN tables
            SN_ver = [table[0] for table in tables if table[1] == 'AIPS SN']
            for ver in SN_ver:
                taco_kwargs.update({
                    'inVer': ver,
                    'outVer': ver
                    })
                taco = task_factory("TabCopy", **taco_kwargs)
                with log_obit_err(log):
                    taco.go()
            taco_kwargs['inTab'] = 'AIPS AN'
            taco_kwargs['nCopy'] = 1
            AN_ver = 1
            taco_kwargs.update({
                    'inVer': AN_ver,
                    'outVer': AN_ver
                    })
            taco = task_factory("TabCopy", **taco_kwargs)
            with log_obit_err(log):
                taco.go()

cp.PipelineImplementation._run_mfbeam = run_mfbeam
cp.PipelineImplementation._attach_SN_tables_to_image = attach_SN_tables_to_image

@cp.register_workmode('mfbeam')
class KatdalMFBeamPipeline(cp.KatdalPipelineImplementation):
    def __init__(self, katdata, beams, uvblavg_params={}, imager_params={},
                 katdal_select={}, nvispio=10240, prtlv=5,
                 clobber=set(['scans', 'avgscans']), reuse=False, uv_merge_path=None):
        """
        Initialise the Continuum Pipeline for offline imaging
        using a katdal dataset.

        Parameters
        ----------
        katdata : :class:`katdal.Dataset`
            katdal Dataset object
        uvblavg_params : dict
            Dictionary of UV baseline averaging task parameters
        imager_params : dict
            Dictionary of MFBeam task parameters
        katdal_select : dict
            Dictionary of katdal selection statements.
        nvispio : integer
            Number of AIPS visibilities per IO operation.
        prtlv : integer
            Chattiness of Obit tasks
        clobber : set or iterable
            Set or iterable of output files to clobber from the aipsdisk.
            Possible values include:
            1. `'scans'`, UV data files containing observational
                data for individual scans.
            2. `'avgscans'`, UV data files containing time-dependent
                baseline data for individual scans.
            3. `'merge'`, UV data file containing merged averaged scans.
            4. `'clean'`, Output images from MFImage.
            5. `'mfimage'`, Output UV data file from MFImage."
        reuse : bool
            Are we reusing a previous katdal export in the aipsdisk?
        beams : list
        	List of FITS files containing the beams
        """

        super(KatdalMFBeamPipeline, self).__init__(katdata)
        self.uvblavg_params = uvblavg_params
        self.imager_params = imager_params
        self.katdal_select = katdal_select
        self.nvispio = nvispio
        self.prtlv = prtlv
        self.clobber = clobber
        self.reuse = reuse
        self.beams = beams
        self.uv_merge_path = uv_merge_path

        self.odisk = len(kc.get_config()['fitsdirs'])

    def __enter__(self):
        return self

    def __exit__(self, etype, evalue, etraceback):
        if etype:
            log.exception('Exception executing continuum pipeline')
        self._cleanup()

    def execute_implementation(self):
        result_tuple = self._select_and_infer_files()
        uv_sources, target_indices, uv_files, clean_files = result_tuple
        if "mfimage" in self.clobber:
            self.cleanup_uv_files += uv_files
        if "clean" in self.clobber:
            self.cleanup_img_files += clean_files
        # Update MFImage source selection
        self.imager_params['Sources'] = uv_sources
        # Find the highest numbered merge file if we are reusing
        if self.reuse:
            uv_mp = self.ka.aips_path(aclass='merge', name=kc.get_config()['cb_id'])
            # Find the merge file with the highest seq #
            hiseq = next_seq_nr(uv_mp) - 1
            # hiseq will be zero if the aipsdisk has no 'merge' file
            if hiseq == 0:
                raise ValueError("AIPS disk at '%s' has no 'merge' file to reuse." %
                                 (kc.get_config()['aipsdirs'][self.disk - 1][-1]))
            else:
                # Get the AIPS entry of the UV data to reuse
                self.uv_merge_path = uv_mp.copy(seq=hiseq)
                log.info("Re-using UV data in '%s' from AIPS disk: '%s'" %
                         (self.uv_merge_path, kc.get_config()['aipsdirs'][self.disk - 1][-1]))
        else:
            # Get the default out path if required
            if self.uv_merge_path is None:
                self.uv_merge_path = self._get_merge_default()
            desired_seq = self.uv_merge_path.seq
            if desired_seq == 0:
                # Get a default uv_merge_path with an unused seq number
                self.uv_merge_path = self.uv_merge_path.copy(seq=next_seq_nr(self.uv_merge_path))
            # Make sure our desired output file doesn't already exist
            if path_exists(self.uv_merge_path):
                raise FileExistsError(f"Desired output path {self.uv_merge_path} already exists.")
            log.info('Exporting visibility data to %s', self.uv_merge_path)
            self._export_and_merge_scans(merge_path=self.uv_merge_path)
        if "merge" in self.clobber:
            self.cleanup_uv_files.append(self.uv_merge_path)
        self._run_mfbeam(self.uv_merge_path, uv_sources)

        self._get_wavg_img(clean_files)
        for uv, clean in zip(uv_files, clean_files):
            self._attach_SN_tables_to_image(uv, clean)

        export_images(clean_files, target_indices,
                      self.odisk, self.ka)

