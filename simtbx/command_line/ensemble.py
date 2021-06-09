from __future__ import absolute_import, division, print_function

from cctbx import sgtbx, miller
import time
from collections import Counter
from scipy.optimize import basinhopping
import h5py
import pandas
from scitbx.matrix import sqr, col

# diffBragg internal parameter indices
FHKL_ID = 11

# LIBTBX_SET_DISPATCHER_NAME simtbx.diffBragg.ensemble

import numpy as np
np.seterr(invalid='ignore')
import os
from dials.array_family import flex
from dxtbx.model.experiment_list import ExperimentListFactory
from libtbx.mpi4py import MPI

COMM = MPI.COMM_WORLD
from libtbx.phil import parse

from simtbx.diffBragg import utils
from simtbx.diffBragg.phil import philz
from simtbx.diffBragg.refiners.parameters import NormalParameter, RangedParameter

hopper_phil = """
percentile_cut = None
  .type = float
  .help = percentile below which pixels are masked 
space_group = P6522
  .type = str
  .help = space group to refine structure factors in
best_pickle = None
  .type = str
  .help = path to a pandas pickle containing the best models for the experiments
skip = None
  .type = int
  .help = skip this many exp
betas {
  G = 0
    .type = float
    .help = restraint factor for scale
}
centers {
  G = 100
    .type = float
    .help = restraint target for scale
}
first_n = None
  .type = int
  .help = refine the first n shots only
hess = None
  .type = str
  .help = scipy minimize hessian argument, 2-point, 3-point, cs, or None
stepsize = 0.5
  .type = float
  .help = basinhopping stepsize
temp = 1
  .type = float
  .help = temperature for basin hopping algo
niter = 100
  .type = int
  .help = number of basin hopping iters
exp_ref_spec_file = None
  .type = str
  .help = path to 3 col txt file containing file names for exper, refl, spectrum (.lam)
method = L-BFGS-B
  .type = str
  .help = minimizer method
opt_det = None
  .type = str
  .help = path to experiment with optimized detector model
sanity_test_input = True
  .type = bool
  .help = sanity test input
outdir = True
  .type = str
  .help = output folder
quiet = False
  .type = bool
  .help = silence most output
sigmas {
  Fhkl = 1
    .type = float
    .help = sigma for Fhkl
  G = 1
    .type = float
    .help = sensitivity for scale factor
}
init {
  G = 1
    .type = float
    .help = init for scale factor
}
mins {
  Fhkl = 0
    .type = float
    .help = min for an Fhkl
  G = 0
    .type = float
    .help = min for scale G
}
maxs {
  Fhkl = 1e6
    .type = float
    .help = max for an Fhkl value
  G = 1e12
    .type = float
    .help = max for scale G
}
relative_tilt = True
  .type = bool
  .help = fit tilt coef relative to roi corner
no_Nabc_scale = False
  .type = bool
  .help = toggle Nabc scaling of the intensity
"""

philz = hopper_phil + philz
phil_scope = parse(philz)


class Script:
    def __init__(self):
        from dials.util.options import OptionParser

        self.params = self.parser = None
        if COMM.rank == 0:
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

    def run(self):
        assert os.path.exists(self.params.exp_ref_spec_file)
        assert self.params.best_pickle is not None
        input_lines = None
        best_models = None
        if COMM.rank == 0:
            input_lines = open(self.params.exp_ref_spec_file, "r").readlines()
            if self.params.sanity_test_input:
                for line in input_lines:
                    for fname in line.strip().split():
                        if not os.path.exists(fname):
                            raise FileNotFoundError("File %s not there " % fname)
            #if self.params.best_pickle is not None:
            #    if not self.params.quiet: print("reading pickle %s" % self.params.best_pickle)
            #    best_models = pandas.read_pickle(self.params.best_pickle)
        input_lines = COMM.bcast(input_lines)
        #best_models = COMM.bcast(best_models)
        if self.params.best_pickle is not None:
            if not self.params.quiet: print("reading pickle %s" % self.params.best_pickle)
            best_models = pandas.read_pickle(self.params.best_pickle)
        if self.params.skip is not None:
            input_lines = input_lines[self.params.skip:]

        if self.params.first_n is not None:
            input_lines = input_lines[:self.params.first_n]

        shot_roi_dict = count_rois(input_lines)
        # gether statistics, e.g. how many total ROIs
        nshots = len(shot_roi_dict)
        nrois = sum([len(shot_roi_dict[s]) for s in shot_roi_dict])
        print("Rank %d will load %d rois across %d shots" % (COMM.rank, nrois, nshots))

        # make a data modeler for each shot
        Modelers = {}
        bests = {}
        for i_exp in shot_roi_dict:
            print("COMM.rank %d on shot  %d" % (COMM.rank, i_exp + 1))

            # this corresponds to the expfile, reflfile, specfile for this shot
            line = input_lines[i_exp]
            # which rois to load from this shots refls
            rois_to_load = shot_roi_dict[i_exp]

            # the filenames
            exp, ref, spec = line.strip().split()

            # if there is a starting model, then load it
            best = None
            if best_models is not None:
                best = best_models.query("exp_name=='%s'" % exp)
                if len(best) == 0:
                    best = best_models.query("opt_exp_name=='%s'" % exp)
                if len(best) != 1:
                    raise ValueError("Should be 1 entry for exp %s in best pickle %s" % (exp, self.params.best_pickle))
            bests[i_exp] = best

            # dont think this is necessary, but doesnt matter
            self.params.simulator.spectrum.filename = spec

            # each shot gets a data modeler
            Modeler = DataModeler(self.params)
            # gather the data from the input files
            if not Modeler.GatherFromExperiment(exp, ref, self.params.space_group, rois_to_load):
                continue

            # store the modeler for later use(each rank has one modeler per shot in shot_roi_dict)
            Modelers[i_exp] = Modeler

        # count up the total number of pixels being modeled by this rank
        npix = [len(modeler.all_data) for modeler in Modelers.values()]
        print("Rank %d wil model %d pixels in total" %(COMM.rank, sum(npix)))
        COMM.barrier()

        # these are the experient ids corresponding to exper-ref-spectrum input file lines , for this rank
        i_exps = list(Modelers.keys())

        # make a SIM instance, use first Modeler as a template
        self.SIM = get_diffBragg_simulator(Modelers[i_exps[0]].E, self.params)
        setup_Fhkl_attributes(self.SIM, self.params, Modelers)

        # iterate over the shot-modelers on this rank, and set the
        # free parameters and spectrum for each shot
        for i_exp, Modeler in Modelers.items():
            # load spectra for other shots
            if bests[i_exp] is not None:
                total_flux = bests[i_exp].total_flux.values[0]
                spectrum_stride = bests[i_exp].spectrum_stride.values[0]
            else:
                total_flux = self.params.simulator.total_flux
                spectrum_stride = self.params.simulator.spectrum.stride
            spectra_file = input_lines[i_exp].strip().split()[2]
            spectrum = utils.load_spectra_file(spectra_file, total_flux, spectrum_stride, as_spectrum=True)

            # set parameter objects for this shot
            Modeler.PAR = Modeler.SimulatorParamsForExperiment(self.SIM, bests[i_exp])
            Modeler.spectrum = spectrum  # store the spectrum as part of the modeler
            #self.SIM.spectrum = spectrum
            #Modeler.xray_beams = self.SIM.beam.xray_beams

            # define the fcell global index for each modeled pixel
            Modeler.all_fcell_global_idx = np.array([self.SIM.i_fcell_from_asu[h] for h in Modeler.hi_asu_perpix])

        self.SIM.update_Fhkl = Fhkl_updater(self.SIM, Modelers)

        # there is variable 1 scale factor per shot
        # Bookkeeping:
        # each i_exp in shot_roi_dict should globally point to a single index
        shot_mapping = {}
        rank_exp_indices = COMM.gather(list(shot_roi_dict.keys()), root=0)
        ntimes = None
        if COMM.rank == 0:
            all_indices = [i_exp for indices in rank_exp_indices for i_exp in indices]
            # count how many ranks a shot is divided amongst, this number
            # should be reported back to all ranks to allow ease in computing
            # the restraint terms
            ntimes = Counter(all_indices)
            shot_mapping = {i_exp: ii for ii, i_exp in enumerate(set(all_indices))}
        shot_mapping = COMM.bcast(shot_mapping)
        ntimes = COMM.bcast(ntimes)


        # for GPU usage, allocate enough pixels!
        self.NPIX_TO_ALLOC = determine_per_rank_max_num_pix(Modelers)
        # TODO in case of randomize devices, shouldnt this be total max across all ranks?
        n = COMM.gather(self.NPIX_TO_ALLOC)
        if COMM.rank == 0:
            n = max(n)
        self.NPIX_TO_ALLOC = COMM.bcast(n)
        self.SIM.D.Npix_to_allocate = int(self.NPIX_TO_ALLOC)

        global_Nshots = len(shot_mapping)
        nparam_per_shot =1   # 1 scale factor per shot
        total_params = nparam_per_shot*global_Nshots + self.SIM.n_global_fcell
        Fhkl_xidx = list(range(global_Nshots, global_Nshots + self.SIM.n_global_fcell))


        rank_xidx = {}
        for i_exp in shot_roi_dict:
            xidx_start = shot_mapping[i_exp]*nparam_per_shot
            xidx = list(range(xidx_start, xidx_start+nparam_per_shot))
            xidx += Fhkl_xidx
            rank_xidx[i_exp] = xidx

        x0 = np.array([1] * total_params)

        sigma_rdout = self.params.refiner.sigma_r / self.params.refiner.adu_per_photon
        mpi_safe_makedirs(self.params.outdir)

        for i_exp in Modelers.keys():
            M = Modelers[i_exp]

            best_model, _ = model(x0[rank_xidx[i_exp]], self.SIM, M, compute_grad=False)
            best_model += M.all_background
            img_path = "rank%d_img%d_before.h5" %(COMM.rank, i_exp)
            img_path = os.path.join(self.params.outdir, img_path)
            save_model_Z(img_path, M.all_data, best_model, M.pan_fast_slow, sigma_rdout, M.all_trusted)

        x = Minimize(x0, rank_xidx, self.params, self.SIM, Modelers, ntimes, global_Nshots)

        for i_exp in Modelers:
            M = Modelers[i_exp]
            best_model, _ = model(x[rank_xidx[i_exp]], self.SIM, M, compute_grad=False)
            best_model += M.all_background
            img_path = "rank%d_img%d_after.h5" %(COMM.rank, i_exp)
            img_path = os.path.join(self.params.outdir, img_path)
            save_model_Z(img_path, M.all_data, best_model, M.pan_fast_slow, sigma_rdout, M.all_trusted)


def get_diffBragg_simulator(expt, params):
    SIM = utils.simulator_from_expt_and_params(expt, params)

    # this works assumes all crystals are of the same crystal system
    SIM.ucell_man = utils.manager_from_crystal(expt.crystal)

    SIM.D.no_Nabc_scale = params.no_Nabc_scale
    SIM.num_xtals = 1
    return SIM


def count_rois(lines):
    info = []
    for i_line, line in enumerate(lines):
        if i_line % COMM.size != COMM.rank:
            continue
        e,r,s = line.strip().split()
        R = flex.reflection_table.from_file(r)
        info.append((i_line, len(R)))
    info = COMM.reduce(info)
    info = COMM.bcast(info)
    shots, nref = zip(*info)
    if COMM.rank==0:
        print("Input is %d refls on %d shots" %(sum(nref), len(shots)))

    shots_and_rois_to_load = []
    if COMM.rank == 0:
        for i_line, nroi in info:
            shots_to_load = [i_line] * nroi
            rois_to_load = list(range(nroi))
            shots_and_rois_to_load += list(zip(shots_to_load, rois_to_load))

    shots_and_rois_to_load = COMM.bcast(shots_and_rois_to_load)
    shots_and_rois_to_load = np.array_split(shots_and_rois_to_load, COMM.size)[COMM.rank]
    from itertools import groupby
    gb = groupby(sorted(shots_and_rois_to_load, key=lambda x: x[0]), key=lambda x:x[0])
    shot_rois = {shot: [i_roi for _,i_roi in vals] for shot,vals in gb}
    out = "\nRank %d will model\n" %COMM.rank
    for shot in shot_rois:
        roi_s = ",".join(map(str, shot_rois[shot]))
        out += "\tShot %d; rois=%s\n" % (shot, roi_s)
    print(out+"\n")
    return shot_rois


class DataModeler:

    def __init__(self, params):
        """ params is a simtbx.diffBragg.hopper phil"""
        self.params = params
        self.SIM = None
        self.PAR = None
        self.spectrum = None
        self.E = None
        self.pan_fast_slow =None
        self.all_background =None
        self.roi_id =None
        self.u_id = None
        self.all_data =None
        self.all_sigmas =None
        self.all_trusted =None
        self.npix_total =None
        self.all_fast =None
        self.all_slow =None
        self.rois=None
        self.pids=None
        self.tilt_abc=None
        self.selection_flags=None
        self.background=None
        self.tilt_cov = None
        self.simple_weights = None
        self.ref_id = None

        self.Hi_asu = None
        self.hi_asu_perpix = None

    def GatherFromExperiment(self, exp, ref, sg_symbol, ref_indices=None):
        """
        :param exp: input experiment filename
        :param ref: intput reflection filename
        :param ref_indices: integer list corresponding to which reflections to load
        :return: True or False depending on success or failure
        """
        self.E = ExperimentListFactory.from_json_file(exp)[0]
        if self.params.opt_det is not None:
            opt_det_E = ExperimentListFactory.from_json_file(self.params.opt_det, False)[0]
            self.E.detector = opt_det_E.detector

        refls = flex.reflection_table.from_file(ref)
        refls["refl_idx"] = flex.int(list(range(len(refls))))
        if ref_indices is not None:
            refl_sel = np.zeros(len(refls), bool)
            for i_roi in ref_indices:
                refl_sel[i_roi] = True
            refls = refls.select(flex.bool(refl_sel))
        img_data = utils.image_data_from_expt(self.E)
        img_data /= self.params.refiner.adu_per_photon
        is_trusted = utils.load_mask(self.params.roi.hotpixel_mask)
        hotpix_mask = None
        if is_trusted is not None:
            hotpix_mask = ~is_trusted
        self.sigma_rdout = self.params.refiner.sigma_r / self.params.refiner.adu_per_photon
        roi_packet = utils.get_roi_background_and_selection_flags(
            refls, img_data, shoebox_sz=self.params.roi.shoebox_size,
            reject_edge_reflections=self.params.roi.reject_edge_reflections,
            reject_roi_with_hotpix=self.params.roi.reject_roi_with_hotpix,
            background_mask=None, hotpix_mask=hotpix_mask,
            bg_thresh=self.params.roi.background_threshold,
            use_robust_estimation=not self.params.roi.fit_tilt,
            set_negative_bg_to_zero=self.params.roi.force_negative_background_to_zero,
            pad_for_background_estimation=self.params.roi.pad_shoebox_for_background_estimation,
            sigma_rdout=self.sigma_rdout, deltaQ=self.params.roi.deltaQ, experiment=self.E,
            weighted_fit=self.params.roi.fit_tilt_using_weights,
            tilt_relative_to_corner=self.params.relative_tilt, ret_cov=True)

        self.rois, self.pids, self.tilt_abc, self.selection_flags, self.background, self.tilt_cov = roi_packet

        assert len(self.rois) == len(refls)

        if sum(self.selection_flags) == 0:
            if not self.params.quiet: print("No pixels slected, continuing")
            return False
        # print("sel")
        self.rois = [roi for i_roi, roi in enumerate(self.rois) if self.selection_flags[i_roi]]
        self.tilt_abc = [abc for i_roi, abc in enumerate(self.tilt_abc) if self.selection_flags[i_roi]]
        self.pids = [pid for i_roi, pid in enumerate(self.pids) if self.selection_flags[i_roi]]
        self.tilt_cov = [cov for i_roi, cov in enumerate(self.tilt_cov) if self.selection_flags[i_roi]]
        #self.refl_index = [refls[i_roi]["refl_index"] for i_roi in range(len(self.rois)) if self.selection_flags[i_roi]]

        # TODO assert that the order of refls and selection_flags was maintained ?
        refls = refls.select(flex.bool(self.selection_flags))
        Hi = list(refls["miller_index"])
        self.Hi_asu = utils.map_hkl_list(Hi, True, sg_symbol)

        all_data = []
        hi_asu_perpix = []
        all_pid = []
        all_fast = []
        all_slow = []
        all_trusted = []
        all_sigmas = []
        all_background = []
        roi_id = []
        #refl_id = []
        for i_roi in range(len(self.rois)):
            pid = self.pids[i_roi]
            x1, x2, y1, y2 = self.rois[i_roi]
            Y, X = np.indices((y2 - y1, x2 - x1))
            data = img_data[pid, y1:y2, x1:x2].copy()


            all_fast += list(X.ravel() + x1)
            all_slow += list(Y.ravel() + y1)

            data = data.ravel()
            trusted = is_trusted[pid, y1:y2, x1:x2].ravel()

            if self.params.percentile_cut is not None:
                lower_cut = np.percentile(data, self.params.percentile_cut)
                trusted[data < lower_cut] = False

            all_background += list(self.background[pid, y1:y2, x1:x2].ravel())
            all_trusted += list(trusted)
            all_sigmas += list(np.sqrt(data + self.sigma_rdout ** 2))
            all_data += list(data)
            npix = len(data)  # np.sum(trusted)
            all_pid += [pid] * npix
            hi_asu_perpix += [self.Hi_asu[i_roi]] * npix
            roi_id += [i_roi] * npix
            #refl_id += [self.refl_index[i_roi]] * npix
        pan_fast_slow = np.ascontiguousarray((np.vstack([all_pid, all_fast, all_slow]).T).ravel())
        self.pan_fast_slow = flex.size_t(pan_fast_slow)
        self.hi_asu_perpix = hi_asu_perpix
        self.all_background = np.array(all_background)
        self.roi_id = np.array(roi_id)
        #self.refl_id = np.array(refl_id)
        self.all_data = np.array(all_data)
        self.all_sigmas = np.array(all_sigmas)
        # note rare chance for sigmas to be nan if the args of sqrt is below 0
        self.all_trusted = np.logical_and(np.array(all_trusted), ~np.isnan(all_sigmas))
        self.npix_total = len(all_data)
        self.simple_weights = 1/self.all_sigmas**2
        self.u_id = set(self.roi_id)
        return True

    def SimulatorParamsForExperiment(self,SIM, best=None):
        """optional best parameter is a single row of a pandas datafame containing the starting
        models, presumably optimized from a previous minimzation using this program"""
        ParameterType = RangedParameter
        PAR = SimParams()

        # set per shot parameters
        p = ParameterType()
        p.sigma = self.params.sigmas.G
        if best is not None:
            p.init = best.spot_scales.values[0]
        else:
            p.init = self.params.init.G
        p.minval = self.params.mins.G
        p.maxval = self.params.maxs.G
        PAR.Scale = p

        #ucell_man = utils.manager_from_crystal(self.E.crystal)
        PAR.Umatrix = sqr(self.E.crystal.get_U())
        PAR.Bmatrix = sqr(self.E.crystal.get_B())
        PAR.Nabc = tuple(best.ncells.values[0])

        return PAR


def Minimize(x0, rank_xidx, params, SIM, Modelers, ntimes, nshots_total):
    if params.refiner.randomize_devices is not None:
        dev = np.random.choice(params.refiner.num_devices)
    else:
        dev = COMM.rank % params.refiner.num_devices
    SIM.D.device_Id = dev

    target = TargetFunc()
    niter = params.niter
    SIM.D.refine(FHKL_ID)
    args = (rank_xidx, SIM, Modelers, not params.quiet, params, True, ntimes)

    out = basinhopping(target, x0,
                       niter=niter,
                       minimizer_kwargs={'args': args, "method": params.method,
                                         "jac": True,
                                         'hess': params.hess},
                       T=params.temp,
                       callback=None,
                       disp=not params.quiet and COMM.rank==0,
                       stepsize=params.stepsize)

    P = out.x
    return P


def get_data_model_pairs(rois, pids, roi_id, best_model, all_data, strong_flags=None):
    all_dat_img, all_mod_img = [], []
    all_strong = []
    for i_roi in range(len(rois)):
        x1, x2, y1, y2 = rois[i_roi]
        pid = pids[i_roi]
        mod = best_model[roi_id == i_roi].reshape((y2 - y1, x2 - x1))
        if strong_flags is not None:
            strong = strong_flags[roi_id == i_roi].reshape((y2 - y1, x2 - x1))
            all_strong.append(strong)
        else:
            all_strong.append(None)
        # dat = img_data[pid, y1:y2, x1:x2]
        dat = all_data[roi_id == i_roi].reshape((y2 - y1, x2 - x1))
        all_dat_img.append(dat)
        all_mod_img.append(mod)
        # print("Roi %d, max in data=%f, max in model=%f" %(i_roi, dat.max(), mod.max()))
    return all_dat_img, all_mod_img, all_strong


class SimParams:
    def __init__(self):
        self.ucell = None
        self.RotXYZ = None
        self.Scale = None
        self.shift = None
        self.Nabc = None
        self.center = None # fdp edge center
        self.slope = None # fdp edge slope

@profile
def model(x, SIM, Modeler, compute_grad=True):

    pfs = Modeler.pan_fast_slow
    PAR = Modeler.PAR

    # update the simlator crystal model
    SIM.D.Umatrix = PAR.Umatrix
    SIM.D.Bmatrix = PAR.Bmatrix
    SIM.D.set_ncells_values(PAR.Nabc)
    #print(PAR.Nabc)
    #print(PAR.Umatrix)
    #print(PAR.Bmatrix)

    # update the energy spectrum
    SIM.beam.spectrum = Modeler.spectrum
    SIM.D.xray_beams = SIM.beam.xray_beams
    #print("Simulating %d energy channels" % len(SIM.D.xray_beams))
    #SIM.D.xray_beams = Modeler.xray_beams #SIM.beam.xray_beams
    #SIM.D.update_xray_beams(SIM.beam.xray_beams)

    # how many parameters we simulate
    npix = int(len(pfs) / 3)
    #print("Simulating %d pixels" % npix)
    nparam = len(x)
    J = np.zeros((nparam, npix))  # Jacobian

    # get the scale factor for this shots
    scale_reparam = x[0]
    scale = PAR.Scale.get_val(scale_reparam)

    # compute the forward model, and gradients where required
    SIM.D.add_diffBragg_spots(pfs)

    model_pix = scale*(SIM.D.raw_pixels_roi[:npix].as_numpy_array())

    if compute_grad:
        # compute the scale factor gradient term, which is related directly to the forward model
        scale_grad = model_pix / scale
        scale_grad = PAR.Scale.get_deriv(scale_reparam, scale_grad)
        J[0] += scale_grad

        # TODO add a dimension to get_derivative_pixels(FHKL_ID), such that pixels can hold information on multiple HKL
        fcell_grad = SIM.D.get_derivative_pixels(FHKL_ID)
        fcell_grad = scale * (fcell_grad[:npix].as_numpy_array())
        unique_i_fcell = set(Modeler.all_fcell_global_idx)
        for i_fcell in unique_i_fcell:
            sel = Modeler.all_fcell_global_idx==i_fcell
            # TODO sanity checks here ?
            this_fcell_grad = fcell_grad[sel]
            rescaled_amplitude = x[1+i_fcell]
            this_fcell_grad = SIM.Fhkl_modelers[i_fcell].get_deriv(rescaled_amplitude, this_fcell_grad)
            J[1+i_fcell,sel] += this_fcell_grad

    return model_pix, J


class TargetFunc:
    def __init__(self):
        self.all_x = []

    def __call__(self, x, *args, **kwargs):
        self.all_x.append(x)
        return target_func(x, *args, **kwargs)

@profile
def target_func(x, rank_xidx, SIM, Modelers, verbose=True, params=None, compute_grad=True, ntimes=None, save=None):
    verbose = verbose and COMM.rank==0
    t_start = time.time()
    timestamps = list(Modelers.keys())
    fchi = fG = 0
    g = np.zeros_like(x)

#   do a global update of the Fhkl parameters in the simulator object
    t_update = time.time()
    SIM.update_Fhkl(SIM, x)
    #update_Fhkl(SIM, x)
    t_update = time.time()-t_update


    all_t_model = 0
    for t in timestamps:
        restraint_n = ntimes[t]
        trusted_t = Modelers[t].all_trusted

        #  x_t should be [G,Fhkl0, Fhkl1, Fhkl2, ..]
        x_t = x[rank_xidx[t]]

        t_model = time.time()
        model_bragg, Jac = model(x_t, SIM, Modelers[t], compute_grad=compute_grad)
        t_model  = time.time()-t_model
        all_t_model += t_model

        Mod_t = Modelers[t]

        model_pix = model_bragg + Mod_t.all_background

        resid = (Mod_t.all_data - model_pix)

        sigma_rdout = params.refiner.sigma_r / params.refiner.adu_per_photon
        V = model_pix + sigma_rdout**2
        resid_square = resid**2

        # if a shot is divided across ranks, then the restraint term
        # for that shot will be computed n times, hence we need to reduce
        # restraint terms by that factor
        nn = 1. / restraint_n

        # update global target functional
        fchi += (.5*(np.log(2*np.pi*V) + resid_square / V))[trusted_t].sum()   # negative log Likelihood target

        # scale factor restraint
        G_rescaled = x_t[0]
        G = Mod_t.PAR.Scale.get_val(G_rescaled)
        delG = params.centers.G - G
        G_V = params.betas.G
        fG += nn*(.5*(np.log(2*np.pi*G_V) + delG**2/G_V))


        if compute_grad:
            grad_term = (0.5 / V * (1 - 2 * resid - resid_square / V))[trusted_t]
            Jac = Jac[:, trusted_t]
            g_t = np.array([np.sum(grad_term*Jac[param_idx]) for param_idx in range(Jac.shape[0])])

            # Fhkl term updates
            g[-SIM.n_global_fcell:] += g_t[-SIM.n_global_fcell:]

            # scale gradient term
            Gidx = rank_xidx[t][0]
            g[Gidx] += g_t[0]

            # scale restraint term
            g[Gidx] += nn*Mod_t.PAR.Scale.get_deriv(G_rescaled, -delG / G_V)

    # bring in data from all ranks
    t_mpi_start = time.time()
    fchi = COMM.bcast(COMM.reduce(fchi))
    fG = COMM.bcast(COMM.reduce(fG))
    g = COMM.bcast(COMM.reduce(g))
    t_mpi_done = time.time()

    # add the Fhkl restraints
    Fhkl_current = np.array([\
        SIM.Fhkl_modelers[i_fcell].get_val(x[-SIM.n_global_fcell+i_fcell])\
        for i_fcell in range(SIM.n_global_fcell)])
    Fhkl_init = np.array([SIM.Fhkl_modelers[i_fcell].init for i_fcell in range(SIM.n_global_fcell)])
    delta_F = Fhkl_init - Fhkl_current
    var_F = np.sum(Fhkl_init**2)
    f_Fhkl = np.sum(delta_F**2 / var_F)

    # update the restraint term for structure factor amplitudes
    Fhkl_rescaled = x[-SIM.n_global_fcell:]
    Fhkl_restraint_grad = -2*delta_F / var_F
    g[-SIM.n_global_fcell:] += np.array([\
        SIM.Fhkl_modelers[i_fcell].get_deriv(Fhkl_rescaled[i_fcell], Fhkl_restraint_grad[i_fcell])\
        for i_fcell in range(SIM.n_global_fcell)])


    f = fchi + fG + f_Fhkl
    chi = fchi / f *100
    gg = fG / f*100
    ff = f_Fhkl / f *100
    gnorm = np.linalg.norm(g)

    t_done = time.time()
    t_mpi = t_mpi_done - t_mpi_start
    t_total = t_done - t_start

    frac_mpi = t_mpi / t_total *100.
    frac_model = all_t_model / t_total * 100.
    frac_update = t_update / t_total * 100.

    if verbose:
        print("F=%10.7g (chi: %.1f%%, G: %.1f%%, Fhkl: %.1f%%); |g|=%10.7e; Total iter time=%.1f millisec (mpi: %.1f%% , model: %.1f%%, updateFhkl: %.1f%%)" \
              % (f, chi, gg, ff, gnorm, t_total*1000, frac_mpi, frac_model, frac_update))
    return f, g



def save_model_Z(img_path, Zdata, Zmodel, pfs, sigma_r, trusted):
    pids = pfs[0::3]
    xs = pfs[1::3]
    ys = pfs[2::3]

    sigma = np.sqrt(Zdata + sigma_r ** 2)
    sigma2 = np.sqrt(Zmodel + sigma_r ** 2)
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
        h5.create_dataset("trusted", data=trusted, **comp)


class Fhkl_updater:

    def __init__(self, SIM, Modelers):
        self.unique_hkl = set()
        for i_exp in Modelers:
            Hi_asu_in_exp = Modelers[i_exp].Hi_asu
            self.unique_hkl = self.unique_hkl.union(set(Hi_asu_in_exp))

        self.equiv_hkls = {}
        self.update_idx = flex.miller_index()
        for hkl_asu in self.unique_hkl:
            equivs = [i.h() for i in miller.sym_equiv_indices(SIM.space_group, hkl_asu).indices()]
            self.equiv_hkls[hkl_asu] = equivs
            for hkl_equiv in equivs:
                self.update_idx.append(hkl_equiv)

    @profile
    def __call__(self, SIM, x):
        #idx, data = SIM.D.Fhkl_tuple
        update_amps = []
        for hkl_asu in self.unique_hkl:
            i_fcell = SIM.i_fcell_from_asu[hkl_asu]
            rescaled_amplitude = x[-SIM.n_global_fcell+i_fcell]
            amp = SIM.Fhkl_modelers[i_fcell].get_val(rescaled_amplitude)
            update_amps += [amp]*len(self.equiv_hkls[hkl_asu])

        update_amps = flex.double(update_amps)
        SIM.D.quick_Fhkl_update((self.update_idx, update_amps))

def update_Fhkl(SIM, x):
    # NOTE, this is the slow version use Fhkl_updater for iterations during ensemble refinement
    #idx, data = SIM.D.Fhkl_tuple
    update_idx = flex.miller_index()
    update_amp = flex.double()

    for i_fcell in range(SIM.n_global_fcell):
        # get the asu miller index
        hkl_asu = SIM.asu_from_i_fcell[i_fcell]

        # get the current amplitude
        xval = x[-SIM.n_global_fcell+i_fcell]
        new_amplitude = SIM.Fhkl_modelers[i_fcell].get_val(xval)

        # now surgically update the p1 array in nanoBragg with the new amplitudes
        # (need to update each symmetry equivalent)
        equivs = [i.h() for i in miller.sym_equiv_indices(SIM.space_group, hkl_asu).indices()]
        for h_equiv in equivs:
            # get the nanoBragg p1 miller table index corresponding to this hkl equivalent
            #try:
            p1_idx = SIM.idx_from_p1[h_equiv]
            #except KeyError as err:
            #    if self.debug:
            #        self.print( h_equiv, err)
            #    continue
            SIM.Fdata[p1_idx] = new_amplitude  # set the data with the new value
            #update_amp.append(new_amplitude)
            #update_idx.append(h_equiv)
    ##SIM.D.quick_Fhkl_update((SIM.Fidx, SIM.Fdata))
    #SIM.D.quick_Fhkl_update((update_idx, update_amp))
    SIM.D.Fhkl_tuple = SIM.Fidx, SIM.Fdata  # update

def setup_Fhkl_attributes(SIM, params, Modelers):
    SIM.space_group_symbol = params.space_group
    SIM.space_group = sgtbx.space_group(sgtbx.space_group_info(symbol=params.space_group).type().hall_symbol())
    a,b = aggregate_Hi(Modelers)
    SIM.i_fcell_from_asu = a
    SIM.asu_from_i_fcell = b
    SIM.n_global_fcell = len(SIM.i_fcell_from_asu)

    # get the nanoBragg internal p1 positional index from the asu miller-index
    SIM.Fidx, SIM.Fdata = SIM.D.Fhkl_tuple
    SIM.idx_from_p1 = {h: i for i, h in enumerate(SIM.Fidx)}

    # get the initial amplitude value from the asu miller-index
    asu_hi = [SIM.asu_from_i_fcell[i_fcell] for i_fcell in range(SIM.n_global_fcell)]
    SIM.fcell_init_from_asu = {h: SIM.Fdata[SIM.idx_from_p1[h]] for h in asu_hi}

    SIM.Fhkl_modelers = []
    for i_fcell in range(SIM.n_global_fcell):
        p = RangedParameter()
        p.sigma = params.sigmas.Fhkl
        p.maxval = params.maxs.Fhkl
        p.minval = params.mins.Fhkl
        asu = SIM.asu_from_i_fcell[i_fcell]
        p.init = SIM.fcell_init_from_asu[asu]
        SIM.Fhkl_modelers.append(p)

    # sanity test, passes
    #for i in range(SIM.n_global_fcell):
    #    val1 = SIM.Fhkl_modelers[i].get_val(1)
    #    h = SIM.asu_from_i_fcell[i]
    #    val2 = SIM.crystal.miller_array.value_at_index(h)
    #    assert np.allclose(val1, val2)


def aggregate_Hi(Modelers):
    # aggregate all miller indices
    Hi_asu_all_ranks = []
    for i_exp in Modelers:
        Hi_asu_all_ranks += Modelers[i_exp].Hi_asu
    Hi_asu_all_ranks = COMM.reduce(Hi_asu_all_ranks)
    Hi_asu_all_ranks = COMM.bcast(Hi_asu_all_ranks)

    # this will map the measured miller indices to their index in the LBFGS parameter array self.x
    idx_from_asu = {h: i for i, h in enumerate(set(Hi_asu_all_ranks))}
    # we will need the inverse map during refinement to update the miller array in diffBragg, so we cache it here
    asu_from_idx = {i: h for i, h in enumerate(set(Hi_asu_all_ranks))}

    return idx_from_asu, asu_from_idx


def mpi_safe_makedirs(dname):
    if COMM.rank == 0:
        utils.safe_makedirs(dname)
    COMM.barrier()


def determine_per_rank_max_num_pix(Modelers):
    max_npix = 0
    for i_exp in Modelers:
        rois = Modelers[i_exp].rois
        x1, x2, y1, y2 = map(np.array, zip(*rois))
        npix = np.sum((x2 - x1) * (y2 - y1))
        max_npix = max(npix, max_npix)
        print("Rank %d, shot %d has %d pixels" % (COMM.rank, i_exp + 1, npix))
    print("Rank %d, max pix to be modeled: %d" % (COMM.rank, max_npix))
    return max_npix


if __name__ == '__main__':
    from dials.util import show_mail_on_error

    with show_mail_on_error():
        script = Script()
        script.run()
