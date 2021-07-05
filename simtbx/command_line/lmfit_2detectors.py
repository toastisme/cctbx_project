from __future__ import absolute_import, division, print_function

# LIBTBX_SET_DISPATCHER_NAME simtbx.diffBragg.lmfit_2detectors

import h5py
from dxtbx.model.experiment_list import ExperimentList
import pandas
from simtbx.diffBragg import ls49_utils
from scitbx.matrix import sqr, col
import lmfit
from simtbx.nanoBragg.anisotropic_mosaicity import AnisoUmats
ROTX_ID=0
ROTY_ID=1
ROTZ_ID=2
NCELLS_ID=9
G = False



class levmar_out:
    def __init__(self, x):
        self.x = x  # parameters


import numpy as np
import os
from dials.array_family import flex
from dxtbx.model.experiment_list import ExperimentListFactory
from libtbx.mpi4py import MPI
COMM = MPI.COMM_WORLD
from  libtbx.phil import parse

from simtbx.diffBragg import utils
from simtbx.diffBragg.phil import philz


script_phil = """
weights = True
  .type = bool
  .help = use weights in the target function
exp_ref_spec_file = None
  .type = str
  .help = path to 3 col txt file containing file names for exper, refl, spectrum (.lam)
method = None
  .type = str
  .help = minimizer method
nelder_mead_maxfev = None
  .type = int
  .help = max number of fevals
strong_factor = 10
  .type = int
  .help = strong pixel weight
weight_strongs_more = False
  .type = bool
  .help = if true, apply a factor to strong pixels when computing loss
number_of_xtals = 1
  .type = int
  .help = number of crystal domains to model per shot
strong_only = False
  .type = bool
  .help = only use the strong spot pixels
strong_dilation = None
  .type = int 
  .help = dilate the strong spot mask
optimizer = neldermead
  .type = str
  .help = type of optimization
sanity_test_input = True
  .type = bool
  .help = sanity test input
outdir = True
  .type = str
  .help = output folder
quiet = False
  .type = bool
  .help = silence most output
max_process = -1
  .type = int
  .help = max exp to process
atols = [0.0001, 0.0001]
  .type = floats(size=2)
  .help = atol and fatol to be passed to nelder mead minimizer (termination params)
embed_at_end = False
  .type = bool
  .help = embedto ipython at end of minimize
plot_at_end = False
  .type = bool
  .help = embedto ipython at end of minimize
init {
  Nabc = [100,100,100]
    .type = floats(size=3)
    .help = init for Nabc
  Ndef = [0,0,0]
    .type = floats(size=3)
    .help = init for Ndef
  RotXYZ = [0,0,0]
    .type = floats(size=3)
    .help = init for RotXYZ
  G = 1
    .type = float
    .help = init for scale factor
  G2 = 2
    .type = float
    .help = init for scale factor
}
mins {
  Nabc = [3,3,3] 
    .type = floats(size=3)
    .help = min for Nabc
  Ndef = [-200,-200,-200] 
    .type = floats(size=3)
    .help = min for Ndef
  RotXYZ = [-1,-1,-1]
    .type = floats(size=3)
    .help = min for rotXYZ in degrees
  G = 0
    .type = float
    .help = min for scale G
  G2 = 0
    .type = float
    .help = min for scale G2 (Rayonix)
}
maxs {
  eta = 0.1
    .type = float
    .help = maximum mosaic spread in degrees
  Nabc = [300,300,300] 
    .type = floats(size=3)
    .help = max for Nabc
  Ndef = [200,200,200] 
    .type = floats(size=3)
    .help = max for Ndef
  RotXYZ = [1,1,1]
    .type = floats(size=3)
    .help = max for rotXYZ in degrees
  G = 1e12
    .type = float
    .help = max for scale G
  G2 = 1e12
    .type = float
    .help = max for scale G2 rayonix
}
RotXYZ_refine = True
  .type = bool
  .help = refine flag
G_refine = True
  .type = bool
  .help = refine flag
Nabc_refine = True
  .type = bool
  .help = refine flag
Ndef_refine = False
  .type = bool
  .help = refine flag
tilt_refine = False
  .type = bool
  .help = refine the background plane
relative_tilt = True
  .type = bool
  .help = fit tilt coef relative to roi corner
detdist_refine = False
  .type = bool
  .help = refine sample to detector distance per shot
detxy_refine = False
  .type = bool
  .help = refine sample to detector distance per shot
ucell_refine = False
  .type = bool
  .help = refine the unit cell
eta_refine = False
  .type = bool
  .help = refine the mosaic sp v         
num_mosaic_blocks = 1
  .type = int
  .help = number of mosaic blocks
ucell_edge_perc = 10 
  .type = float
  .help = precentage for allowing ucell to fluctuate during refinement
ucell_ang_abs = 5
  .type = float
  .help = absolute angle deviation in degrees for unit cell angles to vary during refinement
det_factors = [1,1]
  .type = floats
  .help = scale factors that determine contribution from each detector to the target residual
no_Nabc_scale = False
  .type = bool
  .help = toggle Nabc scaling of the intensity
"""

philz = script_phil + philz
phil_scope = parse(philz)

class Script:
    def __init__(self):
        from dials.util.options import OptionParser

        self.params = self.parser = None
        if COMM.rank==0:
            self.parser = OptionParser(
                usage="",  # stage 1 (per-shot) diffBragg refinement",
                sort_options=True,
                phil=phil_scope,
                read_experiments=True,
                read_reflections=True,
                check_format=False,
                epilog="PyCuties")
        self.parser = COMM.bcast(self.parser)
        if COMM.rank == 0:
            self.params, _ = self.parser.parse_args(show_diff_phil=True)
        self.params = COMM.bcast(self.params)


    def init_parameters(self, tilt_abc, num_xtals=1):
        rad = np.pi/180.
        self.parameters = lmfit.Parameters()
        for i_xtal in range(num_xtals):
            P_NA = lmfit.Parameter("Ncells_a%d" % i_xtal, value=self.params.init.Nabc[0], min=self.params.mins.Nabc[0], max=self.params.maxs.Nabc[0],
                             vary=self.params.Nabc_refine)
            P_NB = lmfit.Parameter("Ncells_b%d" %i_xtal, value=self.params.init.Nabc[1], min=self.params.mins.Nabc[1], max=self.params.maxs.Nabc[1],
                             vary=self.params.Nabc_refine)
            P_NC = lmfit.Parameter("Ncells_c%d" % i_xtal, value=self.params.init.Nabc[2], min=self.params.mins.Nabc[2], max=self.params.maxs.Nabc[2],
                             vary=self.params.Nabc_refine)
            P_ND = lmfit.Parameter("Ncells_d%d" % i_xtal, value=self.params.init.Ndef[0], min=self.params.mins.Ndef[0], max=self.params.maxs.Ndef[0],
                                   vary=self.params.Ndef_refine)
            P_NE = lmfit.Parameter("Ncells_e%d" %i_xtal, value=self.params.init.Ndef[1], min=self.params.mins.Ndef[1], max=self.params.maxs.Ndef[1],
                                   vary=self.params.Ndef_refine)
            P_NF = lmfit.Parameter("Ncells_f%d" % i_xtal, value=self.params.init.Ndef[2], min=self.params.mins.Ndef[2], max=self.params.maxs.Ndef[2],
                                   vary=self.params.Ndef_refine)
            P_ROTX = lmfit.Parameter("RotX%d"% i_xtal, value=self.params.init.RotXYZ[0]*rad, min=self.params.mins.RotXYZ[0]*rad, max=self.params.maxs.RotXYZ[0]*rad,
                             vary=self.params.RotXYZ_refine)
            P_ROTY = lmfit.Parameter("RotY%d" % i_xtal, value=self.params.init.RotXYZ[1]*rad, min=self.params.mins.RotXYZ[1]*rad, max=self.params.maxs.RotXYZ[1]*rad,
                               vary=self.params.RotXYZ_refine)
            P_ROTZ = lmfit.Parameter("RotZ%d" % i_xtal, value=self.params.init.RotXYZ[2]*rad, min=self.params.mins.RotXYZ[2]*rad, max=self.params.maxs.RotXYZ[2]*rad,
                               vary=self.params.RotXYZ_refine)
            P_G = lmfit.Parameter("Gscale%d" % i_xtal, value=self.params.init.G, min=self.params.mins.G, max=self.params.maxs.G,
                            vary=self.params.G_refine)
            P_G2 = lmfit.Parameter("G2scale%d" % i_xtal, value=self.params.init.G2, min=self.params.mins.G2, max=self.params.maxs.G2,
                                  vary=self.params.G_refine)

            self.parameters.add(P_NA)
            self.parameters.add(P_NB)
            self.parameters.add(P_NC)
            self.parameters.add(P_ND)
            self.parameters.add(P_NE)
            self.parameters.add(P_NF)
            self.parameters.add(P_ROTX)
            self.parameters.add(P_ROTY)
            self.parameters.add(P_ROTZ)
            self.parameters.add(P_G)
            self.parameters.add(P_G2)

        for i_roi,(a,b,c) in enumerate(tilt_abc):
            self.parameters.add("tilt_a%d" % i_roi, value=a, vary=self.params.tilt_refine)#, min=, max=, )
            self.parameters.add("tilt_b%d" % i_roi, value=b, vary=self.params.tilt_refine)  # , min=, max=, )
            self.parameters.add("tilt_c%d" % i_roi, value=c, vary=self.params.tilt_refine)  # , min=, max=, )
        self.parameters.add("detdist_offset", value=0, vary=self.params.detdist_refine, min=-5, max=5)  # in millimeters
        self.parameters.add("detxy_offset", value=0, vary=self.params.detxy_refine, min=-5, max=5)  # panel offsets in millimeters
        return self.parameters

    def run(self):
        assert os.path.exists(self.params.exp_ref_spec_file)
        input_lines = None
        if COMM.rank==0:
            input_lines = open(self.params.exp_ref_spec_file, "r").readlines()
            if self.params.sanity_test_input:
                for line in input_lines:
                    for fname in line.strip().split():
                        if not os.path.exists(fname):
                            raise FileNotFoundError("File %s not there " % fname)
        input_lines = COMM.bcast(input_lines)

        for i_exp, line in enumerate(input_lines):
            if i_exp == self.params.max_process:
                break
            if i_exp % COMM.size != COMM.rank:
                continue

            # container for information regarding modelled pixels
            all_pix = PixelArrays()

            print("COMM.rank %d on shot  %d / %d" % (COMM.rank, i_exp + 1, len(input_lines)))
            exp, ref, spec = line.strip().split()

            tstamp = ls49_utils.get_tstamp(exp)
            rayonix_exp_name = os.path.join(ls49_utils.RAYONIX_DIR, "idx-%s_refined_moved.expt" % tstamp)
            rayonix_ref_name = os.path.join(ls49_utils.RAYONIX_DIR, "idx-%s_indexed.refl" % tstamp)
            #experiments
            jungfrau_E = ExperimentListFactory.from_json_file(exp)[0]
            rayonix_E = ExperimentListFactory.from_json_file(rayonix_exp_name)[0]
            two_expers = [jungfrau_E, rayonix_E]

            # reflection tables
            jungfrau_R = flex.reflection_table.from_file(ref)
            rayonix_R = flex.reflection_table.from_file(rayonix_ref_name)
            two_refls = [jungfrau_R, rayonix_R]

            two_adu_per_photon = [7, 0.46]  # jungfrau gain, rayonix gain
            total_roi_count = 0
            two_masks = ["new_jungfrau_mask_panel13.pickle", "mask_r4.pickle"]
            two_sigma_r = [10, 0.25]  # TODO sigma R for Rayonix
            two_pan_fast_slow = []
            self.two_SIM = []

            CRYSTAL = two_expers[0].crystal
            two_expers[1].crystal = CRYSTAL

            # set the spectrum
            self.params.simulator.spectrum.filename = spec

            for i_detector in range(2):
                adu_per_phot = two_adu_per_photon[i_detector]
                sigma_r = two_sigma_r[i_detector]
                refls = two_refls[i_detector]
                img_data = utils.image_data_from_expt(two_expers[i_detector])
                img_data /=adu_per_phot
                is_trusted = utils.load_mask(two_masks[i_detector])
                hotpix_mask = None
                if is_trusted is not None:
                    hotpix_mask = ~is_trusted
                sigma_rdout = sigma_r / adu_per_phot
                E = two_expers[i_detector]
                roi_packet = utils.get_roi_background_and_selection_flags(
                    refls, img_data, shoebox_sz=self.params.roi.shoebox_size,
                    reject_edge_reflections=self.params.roi.reject_edge_reflections,
                    reject_roi_with_hotpix=self.params.roi.reject_roi_with_hotpix,
                    background_mask=None, hotpix_mask=hotpix_mask,
                    bg_thresh=self.params.roi.background_threshold,
                    use_robust_estimation=not self.params.roi.fit_tilt,
                    set_negative_bg_to_zero=self.params.roi.force_negative_background_to_zero,
                    pad_for_background_estimation=self.params.roi.pad_shoebox_for_background_estimation,
                    sigma_rdout=sigma_rdout, deltaQ=self.params.roi.deltaQ, experiment=two_expers[i_detector],
                    weighted_fit=self.params.roi.fit_tilt_using_weights,
                    tilt_relative_to_corner=self.params.relative_tilt)

                rois, pids, tilt_abc, selection_flags, background = roi_packet
                if sum(selection_flags)==0:
                    if not self.params.quiet: print("No pixels slected, continuing")
                    continue
                #print("sel")
                rois = [roi for i_roi, roi in enumerate(rois) if selection_flags[i_roi]]
                tilt_abc = [abc for i_roi, abc in enumerate(tilt_abc) if selection_flags[i_roi]]
                pids = [pid for i_roi, pid in enumerate(pids) if selection_flags[i_roi]]

                #print("get fast slow 1")
                import time
                t = time.time()
                if is_trusted is None:
                    img_mask = np.ones(img_data, bool)
                else:
                    img_mask = is_trusted

                pix = process_rois(refls, two_expers[i_detector], self.params, rois, pids, tilt_abc,
                            img_data, img_mask, sigma_rdout, roi_id_offset=total_roi_count, det_id=i_detector)  # note : returns PixelArrays obj

                total_roi_count += len(rois)
                pan_fast_slow = np.ascontiguousarray((np.vstack([pix.panel_id,pix.fast, pix.slow]).T).ravel())
                pan_fast_slow = flex.size_t(pan_fast_slow)
                two_pan_fast_slow.append(pan_fast_slow)

                all_pix.add(pix)

                SIM = utils.simulator_from_expt_and_params(two_expers[i_detector], self.params)
                SIM.D.no_Nabc_scale = self.params.no_Nabc_scale
                SIM.num_xtals = self.params.number_of_xtals
                if self.params.eta_refine:
                    SIM.umat_maker = AnisoUmats(num_random_samples=self.params.num_mosaic_blocks)
                if self.params.refiner.randomize_devices is not None:
                    dev = np.random.choice(self.params.refiner.num_devices)
                else:
                    dev = COMM.rank % self.params.refiner.num_devices
                SIM.D.device_Id = dev
                self.two_SIM.append(SIM)

            #roi_id = np.array(roi_id)
            ##print(time.time()-t)
            #all_data = np.array(all_data)
            #all_sigmas = np.array(all_sigmas)
            ## note rare chance for sigmas to be nan if the args of sqrt is below 0
            #all_trusted = np.logical_and(np.array(all_trusted), ~np.isnan(all_sigmas))
            #npix_total = len(all_data)
            #all_fast = np.array(all_fast)
            #all_slow = np.array(all_slow)
            all_pix.numpify()

            npix_total = len(all_pix.data)

            t = time.time()
            maxfev = self.params.nelder_mead_maxfev * npix_total

            P = self.init_parameters(all_pix.tilt_abc, self.params.number_of_xtals)
            crystal = self.two_SIM[0].crystal.dxtbx_crystal  # same crystal for both exper
            ucell_man = utils.manager_from_crystal(crystal)
            ucell_vary_perc = self.params.ucell_edge_perc / 100.
            for name, val in zip(ucell_man.variable_names, ucell_man.variables):
                if "Ang" in name:
                    minval = val - ucell_vary_perc * val
                    maxval = val + ucell_vary_perc * val
                else:
                    val_in_deg = val*180/np.pi
                    minval = (val_in_deg-self.params.ucell_ang_abs)*np.pi/180.
                    maxval = (val_in_deg+self.params.ucell_ang_abs)*np.pi/180.
                P.add(name, value=val, vary=self.params.ucell_refine, min=minval, max=maxval)
                if not self.params.quiet: print("Unit cell variable %s (currently=%f) is bounded by %f and %f" % (name,  val, minval, maxval))
            rad = np.pi/180.
            eta_max = self.params.maxs.eta
            P.add("eta_a", value=0, min=0, max=eta_max*rad, vary=self.params.eta_refine)
            P.add("eta_b", value=0, min=0, max=eta_max*rad, vary=self.params.eta_refine)
            P.add("eta_c", value=0, min=0, max=eta_max*rad, vary=self.params.eta_refine)

            self.two_SIM[0].ucell_man = ucell_man
            self.two_SIM[1].ucell_man = ucell_man

            method = "Nelder-Mead"
            if self.params.method is not None:
                method = self.params.method
            verbose = not self.params.quiet
            fit_kws = {"max_nfev": maxfev}
            strong_flags = None
            bg_flags = None
            strong_factor = 1
            if self.params.weight_strongs_more:
                strong_flags = np.logical_and(all_pix.is_strong, all_pix.is_trusted)
                bg_flags = np.logical_and(all_pix.is_bg, all_pix.is_trusted)
                nbg = np.sum(bg_flags)
                nstrong = np.sum(strong_flags)
                strong_factor = self.params.strong_factor * nbg / float(nstrong)
                if not self.params.quiet:print("Effective strong factor =%f" % strong_factor)
            if not self.params.weights:
                all_pix.sigmas = np.ones_like(all_pix.data)

            d0,d1 = self.params.det_factors
            out = lmfit.minimize(lmfit_target, P, method=method,
                    args=(self.two_SIM, two_pan_fast_slow, all_pix.data, all_pix.sigmas, all_pix.is_trusted,
                          all_pix.fast, all_pix.slow, all_pix.roi_id, all_pix.num_rois, verbose,
                         strong_flags, bg_flags, strong_factor, d0,d1),
                    **fit_kws
                    )
            best_model,_ = lmfit_model(out.params, self.two_SIM, two_pan_fast_slow, _verbose=not self.params.quiet)
            tilt_background = get_lmfit_background(out.params, all_pix.num_rois , all_pix.fast, all_pix.slow, all_pix.roi_id)
            best_model += tilt_background
            P = out.params

            if self.two_SIM[0].num_xtals == 1:
                save_to_pandas(P, self.two_SIM[0], exp, self.params, two_expers[0], i_exp)

            rank_imgs_outdir = os.path.join(self.params.outdir, "imgs", "rank%d" % COMM.rank)
            if not os.path.exists(rank_imgs_outdir):
                os.makedirs(rank_imgs_outdir)
            basename = os.path.splitext(os.path.basename(exp))[0]
            img_path = os.path.join(rank_imgs_outdir, "%s_%s_%d.h5" % ("simplex", basename, i_exp))
            #save_model_Z(img_path, all_data, best_model, pan_fast_slow, sigma_rdout)

            if not self.params.quiet: print("Processed %d ROIS , maxfev=%d" %(len(all_pix.rois), maxfev))
            data_subimg, model_subimg = get_data_model_pairs(all_pix.rois, all_pix.roi_pid, all_pix.roi_id, best_model, all_pix.data)

            if self.params.plot_at_end:
                import pylab as plt
                fig,axs = plt.subplots(nrows=1,ncols=2)
                while 1:
                    for i, (d, m) in enumerate(zip(data_subimg, model_subimg)):
                        axs[0].clear()
                        axs[1].clear()
                        axs[1].imshow(m)
                        axs[0].imshow(d)
                        det_id = all_pix.detector_id[i]
                        axs[1].set_title("model %d (det %d)" % (i, det_id ))
                        axs[0].set_title("data %d (det %d)"% (i, det_id))
                        plt.draw()
                        plt.pause(2)
            if self.params.embed_at_end:
                from IPython import embed
                embed()

            comp = {"compression": "lzf"}
            with h5py.File(img_path, "w") as h5:
                for i_roi in range(len(data_subimg)):
                    h5.create_dataset("data/roi%d" % i_roi, data=data_subimg[i_roi], **comp)
                    h5.create_dataset("model/roi%d" % i_roi, data=model_subimg[i_roi], **comp)
                h5.create_dataset("rois", data=all_pix.rois)
                h5.create_dataset("pids", data=all_pix.roi_pid)
                h5.create_dataset("detector_ids", data=all_pix.detector_id)
                #h5.create_dataset("sigma_rdout", data=sigma_rdout)

            self.two_SIM[0].D.free_all()
            self.two_SIM[0].D.free_Fhkl2()
            self.two_SIM[1].D.free_all()
            self.two_SIM[1].D.free_Fhkl2()


def process_rois(refls, E, params, rois, pids, tilt_abc,
                 img_data, img_mask, sigma_rdout, roi_id_offset, det_id):

    print("Processing %d rois" % len(rois))
    pix = PixelArrays()
    all_is_strong = None
    pix.rois = rois
    pix.tilt_abc = tilt_abc
    pix.roi_pid = pids
    pix.detector_id = [det_id] * len(rois)
    if params.strong_only or params.weight_strongs_more:
        all_is_strong = utils.strong_spot_mask(refls, E.detector, params.strong_dilation)
    for i_roi in range(len(rois)):
        pid = pids[i_roi]
        x1, x2, y1, y2 = rois[i_roi]
        # trusted = slice(':') #is_trusted[pid, y1:y2, x1:x2]
        # Y, X = map(lambda x: x[trusted], np.indices((y2-y1, x2-x1)))
        Y, X = np.indices((y2 - y1, x2 - x1))
        # data = img_data[pid, y1:y2, x1:x2][trusted].ravel()
        dat_subimg = img_data[pid, y1:y2, x1:x2]
        data = dat_subimg.ravel()
        trusted = img_mask[pid, y1:y2, x1:x2].ravel()
        if params.weight_strongs_more:
            strongs = trusted * all_is_strong[pid, y1:y2, x1:x2].ravel()
            bgs = trusted * ~all_is_strong[pid, y1:y2, x1:x2].ravel()
            pix.is_strong += list(strongs)
            pix.is_bg += list(bgs)
        if params.strong_only:
            trusted = np.logical_and(trusted, all_is_strong[pid, y1:y2, x1:x2].ravel())
        pix.is_trusted += list(trusted)
        pix.sigmas += list(np.sqrt(data + sigma_rdout ** 2))
        pix.fast += list(X.ravel() + x1)
        pix.slow += list(Y.ravel() + y1)
        pix.data += list(data)

        roi_npix = len(data)  # np.sum(trusted)
        pix.panel_id += [pid] * roi_npix
        pix.roi_id += [i_roi + roi_id_offset] * roi_npix
        a, b, c = tilt_abc[i_roi]
        #bg = (X+x1)*a+(Y+y1)*b + c
        #from IPython import embed
        #embed()
        pix.a += [a] * roi_npix
        pix.b += [b] * roi_npix
        pix.c += [c] * roi_npix
    return pix

def get_data_model_pairs(rois, pids, roi_id, best_model, all_data):
    all_dat_img, all_mod_img = [], []
    for i_roi in range(len(rois)):
        x1, x2, y1, y2 = rois[i_roi]
        pid = pids[i_roi]
        mod = best_model[roi_id == i_roi].reshape((y2 - y1, x2 - x1))
        #dat = img_data[pid, y1:y2, x1:x2]
        dat = all_data[roi_id==i_roi].reshape((y2-y1, x2-x1))
        all_dat_img.append(dat)
        all_mod_img.append(mod)
        #print("Roi %d, max in data=%f, max in model=%f" %(i_roi, dat.max(), mod.max()))
    return all_dat_img, all_mod_img


def lmfit_model(x, two_SIM, two_pfs, _verbose=True):

    two_model_pix = []
    det_id = []
    for i_detector in range(2):
        verbose = _verbose and i_detector == 0
        SIM = two_SIM[i_detector]
        pfs = two_pfs[i_detector]
        npix = int(len(pfs) / 3)
        xax = col((-1, 0, 0))
        yax = col((0, -1, 0))
        zax = col((0, 0, -1))

        model_pix = None
        unitcell_variables = [x[name].value for name in SIM.ucell_man.variable_names]
        SIM.ucell_man.variables = unitcell_variables
        Bmatrix = SIM.ucell_man.B_recipspace
        SIM.D.Bmatrix = Bmatrix
        eta_a = x["eta_a"].value
        eta_b = x["eta_b"].value
        eta_c = x["eta_c"].value
        if hasattr(SIM, "umat_maker"):
            eta_tensor = eta_a, 0, 0, 0, eta_b, 0, 0, 0, eta_c
            umats, _, _ = SIM.umat_maker.generate_Umats(eta_tensor,
                                                        SIM.crystal.dxtbx_crystal, how=1, compute_derivs=False)
            SIM.D.set_mosaic_blocks(umats)
            SIM.D.vectorize_umats()

        for i_xtal in range(SIM.num_xtals):
            SIM.D.raw_pixels_roi *= 0
            if i_detector==0:
                scale = x["Gscale%d" % i_xtal].value
            else:
                scale = x["G2scale%d" % i_xtal].value
            Na = x["Ncells_a%d" % i_xtal].value
            Nb = x["Ncells_b%d" % i_xtal].value
            Nc = x["Ncells_c%d" % i_xtal].value
            Nd = x["Ncells_d%d" % i_xtal].value
            Ne = x["Ncells_e%d" % i_xtal].value
            Nf = x["Ncells_f%d" % i_xtal].value
            rotX = x["RotX%d" % i_xtal].value
            rotY = x["RotY%d" % i_xtal].value
            rotZ = x["RotZ%d" % i_xtal].value

            ## update parameters:
            RX = xax.axis_and_angle_as_r3_rotation_matrix(rotX, deg=False)
            RY = yax.axis_and_angle_as_r3_rotation_matrix(rotY, deg=False)
            RZ = zax.axis_and_angle_as_r3_rotation_matrix(rotZ, deg=False)
            M = RX * RY * RZ
            SIM.D.Umatrix = M * sqr(SIM.crystal.dxtbx_crystal.get_U())
            SIM.D.spot_scale = scale
            SIM.D.set_ncells_values(tuple([Na, Nb, Nc]))
            SIM.D.Ncells_def = Nd, Ne, Nf

            # SIM.D.verbose = 1
            # SIM.D.printout_pixel_fastslow = pfs[1],pfs[2]
            if verbose: print("\tXtal %d:" % i_xtal)
            if verbose: print("\tNcells=%f %f %f" % (Na, Nb, Nc))
            if verbose: print("\tNcells def=%f %f %f" % (Nd, Ne, Nf))
            if verbose: print("\tspot scale=%f" % (scale))
            angles = tuple([x * 180 / np.pi for x in [rotX, rotY, rotZ]])
            if verbose: print("\trotXYZ= %f %f %f (degrees)" % angles)
            SIM.D.add_diffBragg_spots(pfs)

            if model_pix is None:
                model_pix = SIM.D.raw_pixels_roi.as_numpy_array()[:npix]
            else:
                model_pix += SIM.D.raw_pixels_roi.as_numpy_array()[:npix]

        if verbose: print("\tunitcell= %3.5f %3.5f %3.5f %3.5f %3.5f %3.5f" % SIM.ucell_man.unit_cell_parameters)
        etas_deg = tuple([x * 180 / np.pi for x in [eta_a, eta_b, eta_c]])
        if verbose: print("\tetas=%f %f %f" % etas_deg)
        two_model_pix.append(model_pix)
        det_id += [i_detector] * len(model_pix)

    return np.hstack(two_model_pix), np.array(det_id)

def get_lmfit_background(x,  num_rois, all_fast, all_slow, all_roi):
    npix = len(all_fast)
    tilt_background = np.zeros(npix)
    #for i_det in range(2):
    for i_roi in range(num_rois):
        roi_sel = all_roi==i_roi  #np.logical_and(all_roi == i_roi, all_det_ids=i_det)
        a = x["tilt_a%d" % i_roi].value
        b = x["tilt_b%d" % i_roi].value
        c = x["tilt_c%d" % i_roi].value
        tilt_background[roi_sel] = all_fast[roi_sel]*a + all_slow[roi_sel]*b + c
    return tilt_background


def lmfit_target(x, two_SIM, two_pfs, data, sigmas, trusted, all_fast, all_slow, all_roi, num_rois, verbose=True,
                 is_strong=None, is_bg=None, strong_factor=10, det0_factor=1, det1_factor=1):
    model_pix, det_id = lmfit_model(x, two_SIM, two_pfs, _verbose=verbose)
    tilt_background = get_lmfit_background(x, num_rois, all_fast, all_slow, all_roi)
    model_pix += tilt_background
    resid = (model_pix - data) / sigmas
    det_factors = det0_factor, det1_factor
    if is_strong is not None:
        sumsq_b = 0
        sumsq_s = 0
        for i_det in range(2):
            det_factor = det_factors[i_det]
            det_is_bg = np.logical_and(is_bg, det_id==i_det)
            sumsq_b += det_factor*( (resid[det_is_bg] ** 2).sum())
            det_is_strong = np.logical_and(is_strong, det_id==i_det)
            sumsq_s += det_factor*strong_factor*(resid[det_is_strong] ** 2).sum()
        sumsq = sumsq_b + sumsq_s
        if verbose: print("Resid=%10.7g (bg: %10.7f; strong:%10.7f)" % (sumsq,sumsq_b,sumsq_s))
    else:
        sumsq = (resid[trusted] ** 2).sum()
        if verbose: print("Resid=%10.7g" % sumsq)
    return sumsq


def get_param_from_x(x,SIM):
    if SIM.num_xtals >1:
        raise NotImplemented("this method only supports 1xtal")
    scale = x["Gscale0"].value
    Na = x["Ncells_a0"].value
    Nb = x["Ncells_b0"].value
    Nc = x["Ncells_c0"].value
    rotX = x["RotX0"].value
    rotY = x["RotY0"].value
    rotZ = x["RotZ0"].value
    return scale, rotX, rotY, rotZ, Na, Nb, Nc


def save_to_pandas(x, SIM, orig_exp_name, params, expt, rank_exp_idx):
    rank_exper_outdir = os.path.join(params.outdir, "expers" , "rank%d" % COMM.rank)
    rank_pandas_outdir = os.path.join(params.outdir, "pandas" , "rank%d" % COMM.rank)
    for d in [rank_exper_outdir, rank_pandas_outdir]:
        if not os.path.exists(d):
            os.makedirs(d)

    if SIM.num_xtals > 1:
        raise NotImplemented("cant save pandas for multiple crystals yet")
    scale, rotX, rotY, rotZ, Na, Nb, Nc = get_param_from_x(x, SIM)
    if isinstance(x, lmfit.Parameters):
        unitcell_variables = [x[name].value for name in SIM.ucell_man.variable_names]
        SIM.ucell_man.variables = unitcell_variables
        Bmatrix = SIM.ucell_man.B_recipspace
        a_init,b_init,c_init,al_init,be_init,ga_init = SIM.crystal.dxtbx_crystal.get_unit_cell().parameters()
        SIM.crystal.dxtbx_crystal.set_B(Bmatrix)
        a,b,c,al,be,ga = SIM.crystal.dxtbx_crystal.get_unit_cell().parameters()
        eta_a = x["eta_a"].value
        eta_b = x["eta_b"].value
        eta_c = x["eta_c"].value
        xtal_scales = [scale]
    else:
        eta_a = eta_b = eta_c = 0
        a,b,c,al,be,ga = SIM.crystal.dxtbx_crystal.get_unit_cell().parameters()
        a_init,b_init,c_init,al_init,be_init,ga_init = SIM.crystal.dxtbx_crystal.get_unit_cell().parameters()
        xtal_scales = [scale**2]

    xax = col((-1, 0, 0))
    yax = col((0, -1, 0))
    zax = col((0, 0, -1))
    ## update parameters:
    RX = xax.axis_and_angle_as_r3_rotation_matrix(rotX, deg=False)
    RY = yax.axis_and_angle_as_r3_rotation_matrix(rotY, deg=False)
    RZ = zax.axis_and_angle_as_r3_rotation_matrix(rotZ, deg=False)
    M = RX * RY * RZ
    U = M * sqr(SIM.crystal.dxtbx_crystal.get_U())
    SIM.crystal.dxtbx_crystal.set_U(U)
    Amats = [SIM.crystal.dxtbx_crystal.get_A()]
    ncells_def_vals = [(0, 0, 0)]
    ncells_vals = [(Na, Nb, Nc)]
    eta = [0]
    lam0 = [-1]
    lam1 = [-1]
    df = pandas.DataFrame({
        #"panX": list(panX), "panY": list(panY), "panZ": list(panZ),
        #"panO": list(panO), "panF": list(panF), "panS": list(panS),
        "spot_scales": xtal_scales, "Amats": Amats, "ncells": ncells_vals,
        "eta_abc": [(eta_a, eta_b, eta_c)],
        "ncells_def": ncells_def_vals,
        #"bgplanes": bgplanes, "image_corr": image_corr,
        #"init_image_corr": init_img_corr,
        #"fcell_xstart": fcell_xstart,
        #"ucell_xstart": ucell_xstart,
        #"init_misorient": init_misori, "final_misorient": final_misori,
        #"bg_coef": bg_coef,
        "eta": eta,
        "rotX": rotX,
        "rotY": rotY,
        "rotZ": rotZ,
        "a": a, "b": b, "c": c, "al": al, "be": be, "ga": ga,
        "a_init": a_init, "b_init": b_init, "c_init": c_init, "al_init": al_init,
        "lam0": lam0, "lam1": lam1,
        "be_init": be_init, "ga_init": ga_init})
        #"scale_xpos": scale_xpos,
        #"ncells_xpos": ncells_xstart,
        #"bgplanes_xpos": bgplane_xpos})

    basename = os.path.splitext(os.path.basename(orig_exp_name))[0]
    opt_exp_path = os.path.join(rank_exper_outdir, "%s_%s_%d.expt" % ("simplex", basename, rank_exp_idx))
    pandas_path = os.path.join(rank_pandas_outdir, "%s_%s_%d.pkl" % ("simplex", basename, rank_exp_idx))
    expt.crystal = SIM.crystal.dxtbx_crystal
    #expt.detector = refiner.get_optimized_detector()
    new_exp_list = ExperimentList()
    new_exp_list.append(expt)
    new_exp_list.as_file(opt_exp_path)

    df["spectrum_filename"] = os.path.abspath(params.simulator.spectrum.filename)
    df["spectrum_stride"] = params.simulator.spectrum.stride
    df["total_flux"] = params.simulator.total_flux
    df["beamsize_mm"] = SIM.beam.size_mm
    df["exp_name"] = os.path.abspath(orig_exp_name)
    df["opt_exp_name"] = os.path.abspath(opt_exp_path)
    df["oversample"] = params.simulator.oversample

    df.to_pickle(pandas_path)


def save_model_Z(img_path, Zdata, Zmodel, pfs, sigma_r):
    pids = pfs[0::3]
    xs = pfs[1::3]
    ys = pfs[2::3]

    sigma = np.sqrt(Zdata + sigma_r**2)
    sigma2 = np.sqrt(Zmodel + sigma_r**2)
    Zdiff = Zmodel - Zdata
    Z = Zdiff / sigma
    Z2 = Zdiff / sigma2
    with h5py.File(img_path, "w") as h5:
        comp = {"compression": "lzf"}
        h5.create_dataset("Z_data_noise", data=Z, **comp)
        h5.create_dataset("Z_model_noise", data=Z2, **comp)
        h5.create_dataset("pids", data=pids, **comp)
        h5.create_dataset("ys", data=ys, **comp)
        h5.create_dataset("xs", data=xs, **comp)


class PixelArrays:
    def __init__(self):
        """simple organizer for pixel stuffs"""
        self.fast = []  # fast scan  coord
        self.slow = []  # slow scan coor
        self.roi_id = []  # roi identifier
        self.panel_id = []  # panel id
        self.data = []  # pixel data (in photon units)
        self.sigmas = []  # pixel data errors (in photon units)
        self.model = []  # model
        self.is_trusted = []  # is the pixel trusted
        self.is_strong = []  # is the pixel a strong pixel
        self.is_bg =[]   # is the pixel a background pixel
        self.a =[]  # background tilt plane a coef (multiplies fast scan coord)
        self.b = []  # background tilt plane b coef (multiplies slow scan coord)
        self.c =[]  # background tilt plane c coef (offset)
        self.detector_id = []  # detector identifier
        self.rois =[]
        self.tilt_abc = []
        self.roi_pid = []
        self.names = ["fast", "slow", "roi_id", "panel_id", "data", "sigmas", "model",
                    "is_trusted", "is_strong", "is_bg", "a", "b", "c", "detector_id",
                    "rois", "tilt_abc", "roi_pid", "detector_id"]

    @property
    def num_rois(self):
        return len(set(self.roi_id))

    def add(self, other):
        #self.__dict__.keys():
        for name in self.names:
            setattr(self, name,
                    getattr(self, name) + getattr(other, name))
                    #getattr(other, name) + getattr(self, name))

    def numpify(self):
        self.roi_id = np.array(self.roi_id)
        self.data = np.array(self.data)
        self.detector_id = np.array(self.detector_id)
        self.sigmas = np.array(self.sigmas)
        # note rare chance for sigmas to be nan if the args of sqrt is below 0
        self.is_trusted = np.logical_and(np.array(self.is_trusted), ~np.isnan(self.sigmas))
        self.fast = np.array(self.fast)
        self.slow = np.array(self.slow)


if __name__ == '__main__':
    from dials.util import show_mail_on_error
    with show_mail_on_error():
        script = Script()
        script.run()
