from __future__ import absolute_import, division, print_function
from copy import deepcopy
from scipy.interpolate import interp1d
from scipy.optimize import basinhopping
from libtbx import easy_pickle
import h5py
from dxtbx.model.experiment_list import ExperimentList
import pandas
from scitbx.matrix import sqr, col
from simtbx.nanoBragg.anisotropic_mosaicity import AnisoUmats

# diffBragg internal parameter indices
ROTX_ID = 0
ROTY_ID = 1
ROTZ_ID = 2
NCELLS_ID = 9
UCELL_ID_OFFSET = 3
FP_FDP_ID = 22

# LIBTBX_SET_DISPATCHER_NAME simtbx.diffBragg.hopper_ensemble

import numpy as np
import os
from dials.array_family import flex
from dxtbx.model.experiment_list import ExperimentListFactory
from libtbx.mpi4py import MPI

COMM = MPI.COMM_WORLD
from libtbx.phil import parse

from simtbx.diffBragg import utils
from simtbx.diffBragg.phil import philz
from simtbx.diffBragg.refiners.parameters import RangedParameter

hopper_phil = """
refine_fdp_center_and_slope=True
  .type = bool
  .help = refine the two parameter model for fprime, fdblprime
  .help = if True, then fp_fdp file is ignored, as well as refine_fp_fdp_shift parameter
refine_fp_fdp_shift = False
  .type = bool
  .help = refine an energy shift in the fp, fdp 
  .help = gradients not supported, so method should be Nelder-Mead
best_pickle = None
  .type = str
  .help = path to a pandas pickle containing the best models for the experiments
skip = None
  .type = int
  .help = skip this many exp
fp_fdp_file = None
  .type = str
  .help = path to a 3 column data file with energies, fprimes, fdblprimes
atom_data_file = None
  .type = str
  .help = path to a 5 column data file with x,y,z,Bfactor,occupancy for
  .help = each heavy atom that undergoes anomalous scattering
  .help = each heavy atom is of the same species (currently only one species supported)
complex_F = None
  .type = str
  .help = path to a pickle file containing a complex-type miller array
  .help = that will override the mtz path (if provided) 
betas {
  RotXYZ = 0
    .type = float
    .help = restraint factor for the rotXYZ restraint
  Nabc = 0
    .type = float
    .help = restraint factor for the ncells abc
  G = 0
    .type = float
    .help = restraint factor for scale
}
centers {
  RotXYZ = [0,0,0]
    .type = floats(size=3)
    .help = restraint target for Umat rotations 
  Nabc = [100,100,100]
    .type = floats(size=3)
    .help = restraint target for Nabc
  G = 100
    .type = float
    .help = restraint target for scale
}
widths {
  RotXYZ = [0.02,0.02,0.02]
    .type = floats(size=3)
    .help = 1 sigma spread of the RotXYZ rotations
  Nabc = [30,30,30]
    .type = floats(size=3)
    .help = 1 sigma spread of the Nabc 
}
levmar {
  damper = 1e-5
    .type = float
    .help = damping coefficient
  maxiter = 100
    .type = int
    .help = max iterations
  up = 10
    .type = float 
    .help = scale-up factor
  down = 10
    .type = float
    .help = scale-down factor 
  eps4 = 1e-3
    .type = float
    .help = metric improvement threshold for accepting parameter shift
}
hybrid_iter = None
  .type = int
  .help = number of iters for second gradient based basinhop if running in hybrid mode
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
lsq = True
  .type = bool
  .help = minimizes least squares, if False, minimizes likelihood
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
  .type = float
  .help = strong pixel weight
weight_strongs_more = False
  .type = bool
  .help = if true, apply a factor to strong pixels when computing loss
opt_det = None
  .type = str
  .help = path to experiment with optimized detector model
number_of_xtals = 1
  .type = int
  .help = number of crystal domains to model per shot
strong_only = False
  .type = bool
  .help = only use the strong spot pixels
strong_dilation = None
  .type = int 
  .help = dilate the strong spot mask
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
plot_at_end = False
  .type = bool
  .help = plot subimgs at end of minimize
embed_at_end = False
  .type = bool
  .help = embedto ipython at end of minimize
sigmas {
  Nabc = [1,1,1]
    .type = floats(size=3)
    .help = init for Nabc
  Ndef = [1,1,1]
    .type = floats(size=3)
    .help = init for Ndef
  RotXYZ = [1,1,1]
    .type = floats(size=3)
    .help = init for RotXYZ
  G = 1
    .type = float
    .help = init for scale factor
  ucell = [1,1,1,1,1,1]
    .type = ints
    .help = sensitivity for unit cell params
}
init {
  fdp_center_and_slope = [0.5, 3.43, 7120, 0.4]
    .type = floats(size=4)
    .help = initial values a,b,c,d for the 
    .help = fdlprime model   a + b/(1+exp[-d*(lambda-c)])
    .help = the first two terms a and b are currently treated as constants, and the second two
    .help = correspond to the center of the edge and the steepness of the edge
    .help = and they are refined 
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
  shift = 0
    .type = int
    .help = initial value for the fp, fdp shift param
}
mins {
  fdp_center_and_slope = [0,0]
    .type = floats(size=2)
    .help = min edge center and min edge slope
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
}
maxs {
  fdp_center_and_slope = [100000,100]
    .type = floats(size=2)
    .help = max edge center and max edge slope
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
detdist_refine = False
  .type = bool
  .help = refine sample to detector distance per shot
tilt_refine = False
  .type = bool
  .help = refine the background plane
relative_tilt = True
  .type = bool
  .help = fit tilt coef relative to roi corner
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
no_Nabc_scale = False
  .type = bool
  .help = toggle Nabc scaling of the intensity
"""


philz = hopper_phil + philz
phil_scope = parse(philz)

# TODO tidy up the refine_fdp_center_and_slope logic

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
        input_lines = None
        best_models = None
        if COMM.rank == 0:
            input_lines = open(self.params.exp_ref_spec_file, "r").readlines()
            if self.params.sanity_test_input:
                for line in input_lines:
                    for fname in line.strip().split():
                        if not os.path.exists(fname):
                            raise FileNotFoundError("File %s not there " % fname)
            if self.params.best_pickle is not None:
                if not self.params.quiet: print("reading pickle %s" % self.params.best_pickle)
                best_models = pandas.read_pickle(self.params.best_pickle)
        input_lines = COMM.bcast(input_lines)
        best_models = COMM.bcast(best_models)
        if self.params.skip is not None:
            input_lines = input_lines[self.params.skip:]

        if self.params.first_n is not None:
            input_lines = input_lines[:self.params.first_n]

        # TODO verify object
        shot_roi_dict = count_rois(input_lines)
        # gether statistics, e.g. how many total ROIs
        nshots = len(shot_roi_dict)
        nrois = sum([len(shot_roi_dict[s]) for s in shot_roi_dict])
        print("Rank %d will load %d rois across %d shots" % (COMM.rank, nrois, nshots))

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
                if len(best) != 1:
                    raise ValueError("Should be 1 entry for exp %s in best pickle %s" % (exp, self.params.best_pickle))
            bests[i_exp] = best

            # dont think this is necessary, but doesnt matter
            self.params.simulator.spectrum.filename = spec
            # each shot gets a data modeler
            Modeler = DataModeler(self.params)
            # gather the data from the input files
            if not Modeler.GatherFromExperiment(exp, ref, rois_to_load):
                continue

            # store the modeler for later use(each rank has one modeler per shot in shot_roi_dict)
            Modelers[i_exp] = Modeler

        npix = [len(modeler.all_data) for modeler in Modelers.values()]
        print("Rank %d wil model %d pixels in total" %(COMM.rank, sum(npix)))
        COMM.barrier()

        # these are the experient ids correspondong to input file lines , for this rank
        i_exps = list(Modelers.keys())
        # make a SIM instance, use first Modeler as a template
        self.SIM = get_diffBragg_simulator(Modelers[i_exps[0]].E, self.params)


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

        # for each shot
        #    for each crystal
        #       0- scale
        #       1-3 Na,Nb,Nc
        #       4-6 RotX, RotY, RotZ
        #    7-(7+nucell)  unitcell parameters

        # each i_exp in shot_roi_dict should globally point to a single index
        # (sometimes this can be the same value, however if shots are skipped when gathering data above, then we must account for that)
        shot_mapping = {}
        rank_exp_indices = COMM.gather(list(shot_roi_dict.keys()))
        if COMM.rank == 0:
            all_indices = set([i_exp for indices in rank_exp_indices for i_exp in indices])
            shot_mapping = {i_exp: ii for ii, i_exp in enumerate(all_indices)}
        shot_mapping = COMM.bcast(shot_mapping)
        Nshots = len(shot_mapping)
        nucell_param = len(self.SIM.ucell_man.variables)
        nparam_per_shot =7*self.SIM.num_xtals + nucell_param
        total_params = nparam_per_shot*Nshots
        # initial parameters (all set to 1, 7 parameters (scale, rotXYZ, Ncells_abc) per crystal (sausage) and then the unit cell parameters
        if self.params.refine_fdp_center_and_slope:
            total_params += 2
        elif self.params.refine_fp_fdp_shift:
            total_params += 1

        x0 = [1] * total_params

        rank_xidx ={}
        for i_exp in shot_roi_dict:
            xidx_start = shot_mapping[i_exp]*nparam_per_shot
            xidx = list(range(xidx_start, xidx_start+nparam_per_shot))
            if self.params.refine_fdp_center_and_slope:
                xidx += [total_params-2, total_params-1]
            elif self.params.refine_fp_fdp_shift:
                xidx += [total_params-1]
            rank_xidx[i_exp] = xidx

        x = Minimize(x0, rank_xidx, self.params, self.SIM, Modelers)
        #def Minimize(x0, rank_xidx, params, SIM, Modelers):
        save_up(x, rank_xidx, Modelers, self.SIM)



def get_diffBragg_simulator(expt, params):
    complex_F = None
    if params.complex_F is not None:
        complex_F = easy_pickle.load(params.complex_F)
    SIM = utils.simulator_from_expt_and_params(expt, params, complex_F=complex_F)

    # this works assumes all crystals are of the same crystal system
    SIM.ucell_man = utils.manager_from_crystal(expt.crystal)

    SIM.D.no_Nabc_scale = params.no_Nabc_scale
    SIM.num_xtals = params.number_of_xtals
    #if params.eta_refine:
    #    SIM.umat_maker = AnisoUmats(num_random_samples=params.num_mosaic_blocks)
    if params.fp_fdp_file is not None and not params.refine_fdp_center_and_slope:
        en, fp, fdblp = np.loadtxt(params.fp_fdp_file).T
        wavelens_modeled, _ = zip(*SIM.beam.spectrum)
        en_model = utils.ENERGY_CONV / np.array(wavelens_modeled)

        fp_modeled = interp1d(en, fp, bounds_error=True)(en_model)
        fdp_modeled = interp1d(en, fdblp, bounds_error=True)(en_model)

        # preserve the initial
        SIM.fp_reference = deepcopy(fp_modeled)
        SIM.fdp_reference = deepcopy(fdp_modeled)

        fp_modeled, fdp_modeled = shift_fp_fdp(SIM.fp_reference, SIM.fdp_reference, params.init.shift)
        SIM.D.fprime_fdblprime = list(fp_modeled), list(fdp_modeled)
    if params.atom_data_file is not None:
        x, y, z, B, o = map(list, np.loadtxt(params.atom_data_file).T)
        SIM.D.heavy_atom_data = x, y, z, B, o
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

    #shot_roi_dict = {}
    #shots_to_load, rois_to_load = zip(*shots_and_rois_to_load)
    #for s in set(shots_to_load):
    #    shot_roi_dict[s] = []
    #for s,roi in shots_and_rois_to_load:
    #    shot_roi_dict[s].append(roi)
    #return shot_roi_dict


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

    def GatherFromExperiment(self, exp, ref, ref_indices=None):
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
        if sum(self.selection_flags) == 0:
            if not self.params.quiet: print("No pixels slected, continuing")
            return False
        # print("sel")
        self.rois = [roi for i_roi, roi in enumerate(self.rois) if self.selection_flags[i_roi]]
        self.tilt_abc = [abc for i_roi, abc in enumerate(self.tilt_abc) if self.selection_flags[i_roi]]
        self.pids = [pid for i_roi, pid in enumerate(self.pids) if self.selection_flags[i_roi]]
        self.tilt_cov = [cov for i_roi, cov in enumerate(self.tilt_cov) if self.selection_flags[i_roi]]
        #self.refl_index = [refls[i_roi]["refl_index"] for i_roi in range(len(self.rois)) if self.selection_flags[i_roi]]

        all_data = []
        all_pid = []
        all_fast = []
        all_slow = []
        all_trusted = []
        all_sigmas = []
        all_background = []
        roi_id = []
        #refl_id = []
        is_strong = None
        all_strongs = []
        all_bgs = []
        if self.params.strong_only or self.params.weight_strongs_more:
            is_strong = utils.strong_spot_mask(refls, self.E.detector, self.params.strong_dilation)
        for i_roi in range(len(self.rois)):
            pid = self.pids[i_roi]
            x1, x2, y1, y2 = self.rois[i_roi]
            Y, X = np.indices((y2 - y1, x2 - x1))
            data = img_data[pid, y1:y2, x1:x2].copy()

            if is_strong is not None:
                strong_region = is_strong[pid, y1:y2, x1:x2].ravel()

            all_fast += list(X.ravel() + x1)
            all_slow += list(Y.ravel() + y1)

            data = data.ravel()
            trusted = is_trusted[pid, y1:y2, x1:x2].ravel()
            if self.params.weight_strongs_more or self.params.strong_only:
                strongs = trusted * strong_region
                bgs = trusted * ~strong_region
                all_strongs += list(strongs)
                all_bgs += list(bgs)
            all_background += list(self.background[pid, y1:y2, x1:x2].ravel())
            all_trusted += list(trusted)
            all_sigmas += list(np.sqrt(data + self.sigma_rdout ** 2))
            all_data += list(data)
            npix = len(data)  # np.sum(trusted)
            all_pid += [pid] * npix
            roi_id += [i_roi] * npix
            #refl_id += [self.refl_index[i_roi]] * npix
        pan_fast_slow = np.ascontiguousarray((np.vstack([all_pid, all_fast, all_slow]).T).ravel())
        self.pan_fast_slow = flex.size_t(pan_fast_slow)
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
        PAR = SimParams()

        if self.params.refine_fdp_center_and_slope:
            offset, amp, center, slope = self.params.init.fdp_center_and_slope
            wavelens_modeled, _ = zip(*SIM.beam.spectrum)
            en_model = utils.ENERGY_CONV / np.array(wavelens_modeled)
            fdp_modeled = utils.f_double_prime(en_model, offset, amp, center, slope)
            fp_modeled = utils.f_prime(fdp_modeled)
            SIM.en_model = en_model  # NOTE, this assumes the energy axis shouldn't change across experiments
            SIM.fdp_amp = amp
            SIM.fdp_offset = offset
            SIM.D.fprime_fdblprime = list(fp_modeled), list(fdp_modeled)

            p_center, p_slope = RangedParameter(), RangedParameter()
            p_center.sigma, p_slope.sigma = 1, 1
            p_center.minval, p_slope.minval = self.params.mins.fdp_center_and_slope
            p_center.maxval, p_slope.maxval = self.params.maxs.fdp_center_and_slope
            p_center.init,  p_slope.init = center, slope
            PAR.center, PAR.slope = p_center, p_slope

        ucell_man = utils.manager_from_crystal(self.E.crystal)
        #TODO : how to do this when best is NOne?
        self.Umatrix = sqr(self.E.crystal.get_U())
        if best is not None:
            # Umatrix
            #self.params.init.RotXYZ = best[["rotX", "rotY", "rotZ"]].values[0]
            xax = col((-1, 0, 0))
            yax = col((0, -1, 0))
            zax = col((0, 0, -1))
            rotX,rotY,rotZ = best[["rotX", "rotY", "rotZ"]].values[0]
            RX = xax.axis_and_angle_as_r3_rotation_matrix(rotX, deg=False)
            RY = yax.axis_and_angle_as_r3_rotation_matrix(rotY, deg=False)
            RZ = zax.axis_and_angle_as_r3_rotation_matrix(rotZ, deg=False)
            M = RX * RY * RZ
            self.Umatrix = M * sqr(self.E.crystal.get_U())

            # Bmatrix:
            ucparam = best[["a","b","c","al","be","ga"]].values[0]
            ucell_man = utils.manager_from_params(ucparam)

            self.params.init.Nabc = tuple(best.ncells.values[0])
            self.params.init.G = best.spot_scales.values[0]

        # set per shot parameters
        PAR.Nabc = []
        PAR.RotXYZ = []
        PAR.Scale = []
        for i_xtal in range(SIM.num_xtals):
            for ii in range(3):
                p = RangedParameter()
                p.sigma = self.params.sigmas.Nabc[ii]
                p.init = self.params.init.Nabc[ii]
                # set the mosaic block size
                p.minval = self.params.mins.Nabc[ii]
                p.maxval = self.params.maxs.Nabc[ii]
                PAR.Nabc.append(p)

                p = RangedParameter()
                p.sigma = self.params.sigmas.RotXYZ[ii]
                p.init = 0 #self.params.init.RotXYZ[ii]
                p.minval = self.params.mins.RotXYZ[ii] * np.pi / 180.
                p.maxval = self.params.maxs.RotXYZ[ii] * np.pi / 180.
                PAR.RotXYZ.append(p)

            p = RangedParameter()
            p.sigma = self.params.sigmas.G
            p.init = self.params.init.G
            p.minval = self.params.mins.G
            p.maxval = self.params.maxs.G
            PAR.Scale.append(p)


        ucell_vary_perc = self.params.ucell_edge_perc / 100.
        PAR.ucell = []
        for name, val in zip(ucell_man.variable_names, ucell_man.variables):
            if "Ang" in name:
                minval = val - ucell_vary_perc * val
                maxval = val + ucell_vary_perc * val
            else:
                val_in_deg = val * 180 / np.pi
                minval = (val_in_deg - self.params.ucell_ang_abs) * np.pi / 180.
                maxval = (val_in_deg + self.params.ucell_ang_abs) * np.pi / 180.
            p = RangedParameter()
            p.sigma = 1
            p.minval = minval
            p.maxval = maxval
            p.init = val
            if not self.params.quiet: print(
                "Unit cell variable %s (currently=%f) is bounded by %f and %f" % (name, val, minval, maxval))
            PAR.ucell.append(p)

        if self.params.refine_fp_fdp_shift and not self.params.refine_fdp_center_and_slope:
            if self.params.method not in [None, "Nelder-Mead", "Powell"]:
                raise NotImplementedError("method %s not supported for refining shift" % self.params.method)
            p = RangedParameter()
            p.init = self.params.init.shift
            p.minval = -200
            p.maxval = 200
            p.sigma = 1
            PAR.shift = p

        return PAR


def Minimize(x0, rank_xidx, params, SIM, Modelers):
    if params.method is None:
        method = "Nelder-Mead"
    else:
        method = params.method

    if params.refiner.randomize_devices is not None:
        dev = np.random.choice(params.refiner.num_devices)
    else:
        dev = COMM.rank % params.refiner.num_devices
    SIM.D.device_Id = dev
    H = hopper_minima(SIM, Modelers, rank_xidx)
    if params.quiet:
        H = None
    if COMM.rank != 0:
        H = None
    if method in ["L-BFGS-B", "BFGS", "CG", "dogleg", "SLSQP", "Newton-CG", "trust-ncg", "trust-krylov", "trust-exact", "trust-ncg", "levmar"]:
        niter = params.niter
        SIM.D.refine(ROTX_ID)
        SIM.D.refine(ROTY_ID)
        SIM.D.refine(ROTZ_ID)
        SIM.D.refine(NCELLS_ID)
        SIM.D.refine(FP_FDP_ID)
        for i_ucell in range(len(SIM.ucell_man.variables)):
            SIM.D.refine(UCELL_ID_OFFSET + i_ucell)

        #def target_func(x, rank_xidx, SIM, Modelers, verbose=True, params=None, compute_grad=True):
        args = (rank_xidx, SIM, Modelers, not params.quiet, params, True)
        out = basinhopping(target_func, x0,
                           niter=niter,
                           minimizer_kwargs={'args': args, "method": method, "jac": True,
                                             'hess': params.hess },
                           T=params.temp,
                           callback=H,
                           disp=not params.quiet and COMM.rank==0,
                           stepsize=params.stepsize)
    else:
        #args = (self.SIM, self.pan_fast_slow, self.all_data,
        #        self.all_sigmas, self.all_trusted, self.all_background, pos_data, not self.params.quiet)
        args = (rank_xidx, SIM, Modelers,
                not params.quiet, params, False)
        out = basinhopping(target_func, x0,
                           niter=params.niter,
                           minimizer_kwargs={'args': args, "method": method},
                           T=params.temp,
                           callback=H,
                           disp=not params.quiet and COMM.rank==0,
                           stepsize=params.stepsize)

    P = out.x
    return P

def save_up(x, rank_xidx, Modelers, SIM):
    # NOTE fixme
    for i_exp in Modelers:
        M = Modelers[i_exp]
        best_model,_ = model(x[rank_xidx[i_exp]], SIM,M, compute_grad=False)
        best_model += M.all_background
        look_at_x(x,SIM, M.PAR)

    #if self.SIM.num_xtals == 1:
    #    save_to_pandas(x, self.SIM, exp, self.params, self.E, i_exp)

    #rank_imgs_outdir = os.path.join(self.params.outdir, "imgs", "rank%d" % COMM.rank)
    #if not os.path.exists(rank_imgs_outdir):
    #    os.makedirs(rank_imgs_outdir)
    #basename = os.path.splitext(os.path.basename(exp))[0]
    #img_path = os.path.join(rank_imgs_outdir, "%s_%s_%d.h5" % ("simplex", basename, i_exp))
    ## save_model_Z(img_path, all_data, best_model, pan_fast_slow, sigma_rdout)

    #data_subimg, model_subimg, strong_subimg = get_data_model_pairs(self.rois, self.pids, self.roi_id, best_model, self.all_data)  # img_data)

    #comp = {"compression": "lzf"}
    #with h5py.File(img_path, "w") as h5:
    #    for i_roi in range(len(data_subimg)):
    #        h5.create_dataset("data/roi%d" % i_roi, data=data_subimg[i_roi], **comp)
    #        h5.create_dataset("model/roi%d" % i_roi, data=model_subimg[i_roi], **comp)
    #    h5.create_dataset("rois", data=self.rois)
    #    h5.create_dataset("pids", data=self.pids)
    #    h5.create_dataset("sigma_rdout", data=self.sigma_rdout)


def chi_sq(self, model):
    resid = (self.all_data - model)[self.all_trusted] ** 2
    return (resid * self.simple_weights[self.all_trusted]).sum()



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


def look_at_x(x, SIM, PAR):
    num_per_xtal_params = SIM.num_xtals * (7)
    n_ucell_param = len(SIM.ucell_man.variables)
    #if n_ucell_param + num_per_xtal_params != len(x):
    #    raise ValueError("weird x")
    params_per_xtal = np.array_split(x[:num_per_xtal_params], SIM.num_xtals)

    for i_xtal in range(SIM.num_xtals):
        scale_reparam, rotX_reparam, rotY_reparam, rotZ_reparam, \
        Na_reparam, Nb_reparam, Nc_reparam = params_per_xtal[i_xtal]

        rotX = PAR.RotXYZ[i_xtal * 3].get_val(rotX_reparam)
        rotY = PAR.RotXYZ[i_xtal * 3 + 1].get_val(rotY_reparam)
        rotZ = PAR.RotXYZ[i_xtal * 3 + 2].get_val(rotZ_reparam)

        scale = PAR.Scale[i_xtal].get_val(scale_reparam)

        Na = PAR.Nabc[i_xtal * 3].get_val(Na_reparam)
        Nb = PAR.Nabc[i_xtal * 3 + 1].get_val(Nb_reparam)
        Nc = PAR.Nabc[i_xtal * 3 + 2].get_val(Nc_reparam)

        print("\tXtal %d:" % i_xtal)
        print("\tNcells=%f %f %f" % (Na, Nb, Nc))
        print("\tspot scale=%f" % (scale))
        angles = tuple([x * 180 / np.pi for x in [rotX, rotY, rotZ]])
        print("\trotXYZ= %f %f %f (degrees)" % angles)
    print("\tunitcell= %3.5f %3.5f %3.5f %3.5f %3.5f %3.5f" % SIM.ucell_man.unit_cell_parameters)
    if PAR.shift is not None:
        shift = PAR.shift.get_val(x[-1])
        print("\tfp_fdp shift= %3.1f" % shift)

class SimParams:
    def __init__(self):
        self.ucell = None
        self.RotXYZ = None
        self.Scale = None
        self.shift = None
        self.Nabc = None
        self.center = None # fdp edge center
        self.slope = None # fdp edge slope


def model(x, SIM, Modeler, verbose=True, compute_grad=True):
    verbose = False

    pfs = Modeler.pan_fast_slow
    PAR = Modeler.PAR
    # update the Umatrix reference
    SIM.D.Umatrix = Modeler.Umatrix

    # update the energy spectrum
    SIM.beam.spectrum = Modeler.spectrum
    SIM.D.xray_beams = SIM.beam.xray_beams

    num_per_xtal_params = SIM.num_xtals * (7)
    n_ucell_param = len(SIM.ucell_man.variables)
    #TODO profile setting this parmaeter, consider moving outside per-shot modeler
    if PAR.center is not None: # and PAR.slope is not None:
        assert n_ucell_param+num_per_xtal_params+2 == len(x)
        center = PAR.center.get_val(x[-2])
        slope = PAR.slope.get_val(x[-1])
        fdp = utils.f_double_prime(SIM.en_model, SIM.fdp_offset, SIM.fdp_amp, center, slope)
        fp = utils.f_prime(fdp)
        SIM.D.fprime_fdblprime = list(fp), list(fdp)
        if compute_grad:
            c_deriv_fdp = utils.f_double_prime(SIM.en_model, SIM.fdp_offset,
                                               SIM.fdp_amp, center, slope,
                                               deriv='c')
            c_deriv_fp = utils.f_prime(c_deriv_fdp)

            d_deriv_fdp = utils.f_double_prime(SIM.en_model, SIM.fdp_offset,
                                               SIM.fdp_amp, center, slope,
                                               deriv='d')
            d_deriv_fp = utils.f_prime(d_deriv_fdp)
            # TODO make setting this property more intuituive ?
            # currently its set as a tuple [A], [B], where
            # [A] is the list of all fprime derivatives,one parameter followed by the next
            # and [B] is the same, but for f double prime
            fp_term = list(c_deriv_fp)+list(d_deriv_fp)
            fdp_term = list(c_deriv_fdp)+list(d_deriv_fdp)
            SIM.D.fprime_fdblprime_derivs = fp_term, fdp_term

    elif PAR.shift is not None:
        assert n_ucell_param+num_per_xtal_params+1 == len(x)
        shift_val = PAR.shift.get_val(x[-1])
        fp_shift, fdp_shift = shift_fp_fdp(SIM.fp_reference, SIM.fdp_reference, int(np.round(shift_val)))
        SIM.D.fprime_fdblprime = list(fp_shift), list(fdp_shift)
    else:
        assert n_ucell_param+num_per_xtal_params == len(x)
    params_per_xtal = np.array_split(x[:num_per_xtal_params], SIM.num_xtals)
    unitcell_var_reparam = x[num_per_xtal_params:num_per_xtal_params+n_ucell_param]
    unitcell_variables = [PAR.ucell[i].get_val(xval) for i, xval in enumerate(unitcell_var_reparam)]
    SIM.ucell_man.variables = unitcell_variables
    Bmatrix = SIM.ucell_man.B_recipspace
    SIM.D.Bmatrix = Bmatrix
    if compute_grad:
        for i_ucell in range(len(unitcell_variables)):
            SIM.D.set_ucell_derivative_matrix(
                i_ucell + UCELL_ID_OFFSET,
                SIM.ucell_man.derivative_matrices[i_ucell])
        # NOTE scale factor gradient is computed directly from the forward model below
    npix = int(len(pfs) / 3)
    nparam = len(x)
    J = np.zeros((nparam, npix))  # note: order is: scale, rotX, rotY, rotZ, Na, Nb, Nc, ... (for each xtal), then ucell0, ucell1 , ucell2, ..
    model_pix = None
    for i_xtal in range(SIM.num_xtals):
        SIM.D.raw_pixels_roi *= 0
        scale_reparam, rotX_reparam, rotY_reparam, rotZ_reparam, \
        Na_reparam, Nb_reparam, Nc_reparam = params_per_xtal[i_xtal]

        rotX = PAR.RotXYZ[i_xtal * 3].get_val(rotX_reparam)
        rotY = PAR.RotXYZ[i_xtal * 3 + 1].get_val(rotY_reparam)
        rotZ = PAR.RotXYZ[i_xtal * 3 + 2].get_val(rotZ_reparam)

        ## update parameters:

        SIM.D.set_value(ROTX_ID, rotX)
        SIM.D.set_value(ROTY_ID, rotY)
        SIM.D.set_value(ROTZ_ID, rotZ)
        #RX = xax.axis_and_angle_as_r3_rotation_matrix(rotX, deg=False)
        #RY = yax.axis_and_angle_as_r3_rotation_matrix(rotY, deg=False)
        #RZ = zax.axis_and_angle_as_r3_rotation_matrix(rotZ, deg=False)
        #M = RX * RY * RZ
        #SIM.D.Umatrix = M * sqr(SIM.crystal.dxtbx_crystal.get_U())

        scale = PAR.Scale[i_xtal].get_val(scale_reparam)
        #SIM.D.spot_scale = scale

        Na = PAR.Nabc[i_xtal * 3].get_val(Na_reparam)
        Nb = PAR.Nabc[i_xtal * 3 + 1].get_val(Nb_reparam)
        Nc = PAR.Nabc[i_xtal * 3 + 2].get_val(Nc_reparam)
        SIM.D.set_ncells_values(tuple([Na, Nb, Nc]))

        # SIM.D.verbose = 1
        # SIM.D.printout_pixel_fastslow = pfs[1],pfs[2]
        if verbose: print("\tXtal %d:" % i_xtal)
        if verbose: print("\tNcells=%f %f %f" % (Na, Nb, Nc))
        if verbose: print("\tspot scale=%f" % (scale))
        angles = tuple([x * 180 / np.pi for x in [rotX, rotY, rotZ]])
        if verbose: print("\trotXYZ= %f %f %f (degrees)" % angles)
        SIM.D.add_diffBragg_spots(pfs)

        if model_pix is None:
            model_pix = scale*SIM.D.raw_pixels_roi.as_numpy_array()[:npix]
        else:
            model_pix += scale*SIM.D.raw_pixels_roi.as_numpy_array()[:npix]

        if compute_grad:
            scale_grad = model_pix / scale
            scale_grad = PAR.Scale[i_xtal].get_deriv(scale_reparam, scale_grad)
            J[7*i_xtal] += scale_grad

            rotX_grad = scale*SIM.D.get_derivative_pixels(ROTX_ID).as_numpy_array()[:npix]
            rotY_grad = scale*SIM.D.get_derivative_pixels(ROTY_ID).as_numpy_array()[:npix]
            rotZ_grad = scale*SIM.D.get_derivative_pixels(ROTZ_ID).as_numpy_array()[:npix]
            rotX_grad = PAR.RotXYZ[i_xtal*3].get_deriv(rotX_reparam, rotX_grad)
            rotY_grad = PAR.RotXYZ[i_xtal*3+1].get_deriv(rotY_reparam, rotY_grad)
            rotZ_grad = PAR.RotXYZ[i_xtal*3+2].get_deriv(rotZ_reparam, rotZ_grad)
            J[7*i_xtal + 1] += rotX_grad
            J[7*i_xtal + 2] += rotY_grad
            J[7*i_xtal + 3] += rotZ_grad

            Na_grad, Nb_grad, Nc_grad = [scale*d.as_numpy_array()[:npix] for d in SIM.D.get_ncells_derivative_pixels()]
            Na_grad = PAR.Nabc[i_xtal * 3].get_deriv(Na_reparam, Na_grad)
            Nb_grad = PAR.Nabc[i_xtal * 3 + 1].get_deriv(Nb_reparam, Nb_grad)
            Nc_grad = PAR.Nabc[i_xtal * 3 + 2].get_deriv(Nc_reparam, Nc_grad)
            J[7*i_xtal + 4] += Na_grad
            J[7*i_xtal + 5] += Nb_grad
            J[7*i_xtal + 6] += Nc_grad

            # note important to keep gradients in same order as the parameters x
            #ucell_grad = []
            for i_ucell in range(n_ucell_param):
                d = scale*SIM.D.get_derivative_pixels(UCELL_ID_OFFSET+i_ucell).as_numpy_array()[:npix]
                d = PAR.ucell[i_ucell].get_deriv(unitcell_var_reparam[i_ucell], d)
                #ucell_grad.append(d)
                J[7*SIM.num_xtals + i_ucell] += d

            if PAR.center is not None:# and PAR.slope is not None:
                center_grad, slope_grad = [scale * d.as_numpy_array()[:npix] for d in
                                             SIM.D.get_fp_fdp_derivative_pixels()]
                center_grad = PAR.center.get_deriv(x[-2], center_grad)
                slope_grad = PAR.slope.get_deriv(x[-1], slope_grad)
                J[-2] += center_grad
                J[-1] += slope_grad

    if verbose: print("\tunitcell= %3.5f %3.5f %3.5f %3.5f %3.5f %3.5f" % SIM.ucell_man.unit_cell_parameters)
    return model_pix, J



def target_func(x, rank_xidx, SIM, Modelers, verbose=True, params=None, compute_grad=True):
    verbose = verbose and COMM.rank==0
    timestamps = list(Modelers.keys())
    fchi = frot = fG = fN = 0
    g = np.zeros_like(x)
    for t in timestamps:
        trusted_t = Modelers[t].all_trusted
        #rank_x should be [G0,rXYZ0, Nabc0, G1,rXYZ1,Nabc1.. Ucell0,Ucell1,... shift, slow]
        x_t = x[rank_xidx[t]]

        model_bragg, Jac = model(x_t, SIM, Modelers[t],
                                 verbose=verbose, compute_grad=compute_grad)

        # Jac has shape of num_param x num_pix ( or its None if not computing grad)

        model_pix = model_bragg + Modelers[t].all_background

        W = 1/(Modelers[t].all_sigmas)**2
        resid = (model_pix - Modelers[t].all_data)

        fchi += (resid[trusted_t]**2 * W[trusted_t]).sum()

        G, RotXYZ, Nabc, ucparams = get_param_from_x(x_t,SIM, Modelers[t].PAR)

        delG = []
        del_Na = []
        del_Nb = []
        del_Nc = []
        del_rX = []
        del_rY = []
        del_rZ = []

        deg = 180 / np.pi
        for i_xtal in range(SIM.num_xtals):
            G0 = params.centers.G
            delG.append(G0 - G[i_xtal*3])
            fG += params.betas.G*delG[-1]**2

            rotX = deg*RotXYZ[i_xtal*3]
            rotY = deg*RotXYZ[i_xtal*3+1]
            rotZ = deg*RotXYZ[i_xtal*3+1]
            rotX0,rotY0,rotZ0 = params.centers.RotXYZ
            Na0,Nb0,Nc0 = params.centers.Nabc
            sig_rX, sig_rY, sig_rZ = params.widths.RotXYZ
            sig_Na, sig_Nb, sig_Nc = params.widths.Nabc
            del_rX.append(rotX0-rotX)
            del_rY.append(rotY0-rotY)
            del_rZ.append(rotZ0-rotZ)
            frot_term = (del_rX[-1]/sig_rX)**2+ (del_rY[-1]/sig_rY)**2 + (del_rZ[-1] / sig_rZ)**2
            frot += frot_term * params.betas.RotXYZ*frot

            del_Na .append(Na0 - Nabc[i_xtal*3])
            del_Nb .append(Nb0 - Nabc[i_xtal*3+1])
            del_Nc .append(Nc0 - Nabc[i_xtal*3+2])
            fN_term = (del_Na[-1] / sig_Na)**2 +(del_Nb[-1] / sig_Nb)**2 + (del_Nc[-1] / sig_Nc)**2
            fN += fN_term*params.betas.Nabc*fN

        if compute_grad:
            grad_term = (2*resid*W)[trusted_t]
            Jac = Jac[:, trusted_t]
            g_t = np.array([np.sum(grad_term*Jac[param_idx]) for param_idx in range(Jac.shape[0])])
            ber = params.betas.RotXYZ
            beN = params.betas.Nabc
            # update the per shot gradients
            n_perxtal = 7*SIM.num_xtals
            n_uc_param = len(SIM.ucell_man.variables)
            xidx_xtal = rank_xidx[t][:n_perxtal]
            Gidx = xidx_xtal[0::7]
            rX_idx= xidx_xtal[1::7]
            rY_idx= xidx_xtal[2::7]
            rZ_idx= xidx_xtal[3::7]
            Na_idx= xidx_xtal[4::7]
            Nb_idx= xidx_xtal[5::7]
            Nc_idx = xidx_xtal[6::7]
            uc_idx = rank_xidx[t][n_perxtal: n_perxtal+n_uc_param]

            for i_uc in range(n_uc_param):
                g[uc_idx[i_uc]] += g_t[n_perxtal+i_uc]

            for i_xtal in range(SIM.num_xtals):
                g[Gidx[i_xtal]] += -2*params.betas.G*delG[i_xtal]
                g[rX_idx[i_xtal]] += -ber*2*deg*del_rX[i_xtal]/sig_rX**2
                g[rY_idx[i_xtal]] += -ber*2*deg*del_rY[i_xtal]/sig_rY**2
                g[rZ_idx[i_xtal]] += -ber*2*deg*del_rZ[i_xtal]/sig_rZ**2
                g[Na_idx[i_xtal]] += -beN*2*del_Na[i_xtal]/sig_Na**2
                g[Nb_idx[i_xtal]] += -beN*2*del_Nb[i_xtal]/sig_Nb**2
                g[Nc_idx[i_xtal]] += -beN*2*del_Nc[i_xtal]/sig_Nc**2

                g[Gidx[i_xtal]] += g_t[7*i_xtal]
                g[rX_idx[i_xtal]] += g_t[7*i_xtal+1]
                g[rY_idx[i_xtal]] += g_t[7*i_xtal+2]
                g[rZ_idx[i_xtal]] += g_t[7*i_xtal+3]
                g[Na_idx[i_xtal]] += g_t[7*i_xtal+4]
                g[Nb_idx[i_xtal]] += g_t[7*i_xtal+5]
                g[Nc_idx[i_xtal]] += g_t[7*i_xtal+6]

            # update global parameter gradients: TODO
            if params.refine_fdp_center_and_slope:
                g[-2] += g_t[-2]  # for fdp shift
                g[-1] += g_t[-1]  # for fdp slope

    fchi = COMM.bcast(COMM.reduce(fchi))
    frot = COMM.bcast(COMM.reduce(frot))
    fN = COMM.bcast(COMM.reduce(fN))
    fG = COMM.bcast(COMM.reduce(fG))
    g = COMM.bcast(COMM.reduce(g))

    # TODO MPI step
    f = fchi + frot + fN + fG
    chi = fchi / f *100
    rot = frot / f*100
    G = fG / f*100
    n = fN / f*100
    gnorm = np.linalg.norm(g)
    #shift = 0
    #shift = Modelers[timestamps[0]].PAR.shift.get_val(x[-1])

    if compute_grad:
        if verbose: print("F=%10.7g (chi: %.1f%%, rot: %.1f%% N: %.1f%%, G:%.1f%%, shift=%2.1f), |g|=%10.7g" % (f, chi, rot, n, G,0, gnorm))
        return f, g
    else:
        if verbose: print("F=%10.7g (chi: %.1f%%, rot: %.1f%% N: %.1f%%, G:%.1f%%, shift=%2.1f), |g|=NA" % (f, chi, rot, n, G,0))
        return f


class hopper_minima:
    def __init__(self, SIM, Modelers, rank_xidx):
        self.minima = []
        self.SIM = SIM
        self.Modelers = Modelers
        self.rank_xidx = rank_xidx

    def __call__(self, x, f, accept):
        for i_exp in self.Modelers:
            look_at_x(x[self.rank_xidx[i_exp]], self.SIM, self.Modelers[i_exp].PAR)

        self.minima.append((f,x,accept))


def get_param_from_x(x, SIM, PAR):
    num_per_xtal_params = SIM.num_xtals * (7)
    n_ucell_param = len(SIM.ucell_man.variables)
    params_per_xtal = np.array_split(x[:num_per_xtal_params], SIM.num_xtals)
    unitcell_var_reparam = x[num_per_xtal_params:num_per_xtal_params+n_ucell_param]
    unitcell_variables = [PAR.ucell[i].get_val(xval) for i, xval in enumerate(unitcell_var_reparam)]
    SIM.ucell_man.variables = unitcell_variables
    RotXYZ = []
    Scales = []
    Nabc = []

    for i_xtal in range(SIM.num_xtals):
        scale_reparam, rotX_reparam, rotY_reparam, rotZ_reparam, \
            Na_reparam, Nb_reparam, Nc_reparam = params_per_xtal[i_xtal]

        rotX = PAR.RotXYZ[i_xtal * 3].get_val(rotX_reparam)
        rotY = PAR.RotXYZ[i_xtal * 3 + 1].get_val(rotY_reparam)
        rotZ = PAR.RotXYZ[i_xtal * 3 + 2].get_val(rotZ_reparam)
        RotXYZ += [rotX, rotY, rotZ]

        scale = PAR.Scale[i_xtal].get_val(scale_reparam)
        Scales.append(scale)

        Na = PAR.Nabc[i_xtal * 3].get_val(Na_reparam)
        Nb = PAR.Nabc[i_xtal * 3 + 1].get_val(Nb_reparam)
        Nc = PAR.Nabc[i_xtal * 3 + 2].get_val(Nc_reparam)
        Nabc += [Na, Nb, Nc]

    return Scales, RotXYZ, Nabc, SIM.ucell_man.unit_cell_parameters


def save_to_pandas(x, SIM, orig_exp_name, params, expt, rank_exp_idx):
    rank_exper_outdir = os.path.join(params.outdir, "expers", "rank%d" % COMM.rank)
    rank_pandas_outdir = os.path.join(params.outdir, "pandas", "rank%d" % COMM.rank)
    for d in [rank_exper_outdir, rank_pandas_outdir]:
        if not os.path.exists(d):
            os.makedirs(d)

    if SIM.num_xtals > 1:
        raise NotImplemented("cant save pandas for multiple crystals yet")
    scale, rotX, rotY, rotZ, Na, Nb, Nc,a,b,c,al,be,ga = get_param_from_x(x, SIM)
    shift = np.nan
    if SIM.shift_param is not None:
        shift = SIM.shift_param.get_val(x[-1])
    xtal_scales = [scale]
    eta_a = eta_b = eta_c = 0
    a_init, b_init, c_init, al_init, be_init, ga_init = SIM.crystal.dxtbx_crystal.get_unit_cell().parameters()

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
        # "panX": list(panX), "panY": list(panY), "panZ": list(panZ),
        # "panO": list(panO), "panF": list(panF), "panS": list(panS),
        "spot_scales": xtal_scales, "Amats": Amats, "ncells": ncells_vals,
        "eta_abc": [(eta_a, eta_b, eta_c)],
        "ncells_def": ncells_def_vals,
        "fp_fdp_shift": [shift],
        # "bgplanes": bgplanes, "image_corr": image_corr,
        # "init_image_corr": init_img_corr,
        # "fcell_xstart": fcell_xstart,
        # "ucell_xstart": ucell_xstart,
        # "init_misorient": init_misori, "final_misorient": final_misori,
        # "bg_coef": bg_coef,
        "eta": eta,
        "rotX": rotX,
        "rotY": rotY,
        "rotZ": rotZ,
        "a": a, "b": b, "c": c, "al": al, "be": be, "ga": ga,
        "a_init": a_init, "b_init": b_init, "c_init": c_init, "al_init": al_init,
        "lam0": lam0, "lam1": lam1,
        "be_init": be_init, "ga_init": ga_init})
    # "scale_xpos": scale_xpos,
    # "ncells_xpos": ncells_xstart,
    # "bgplanes_xpos": bgplane_xpos})

    basename = os.path.splitext(os.path.basename(orig_exp_name))[0]
    opt_exp_path = os.path.join(rank_exper_outdir, "%s_%s_%d.expt" % ("simplex", basename, rank_exp_idx))
    pandas_path = os.path.join(rank_pandas_outdir, "%s_%s_%d.pkl" % ("simplex", basename, rank_exp_idx))
    expt.crystal = SIM.crystal.dxtbx_crystal
    # expt.detector = refiner.get_optimized_detector()
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
    if params.opt_det is not None:
        df["opt_det"] = params.opt_det

    df.to_pickle(pandas_path)


def save_model_Z(img_path, Zdata, Zmodel, pfs, sigma_r):
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


def shift_fp_fdp(fp, fdp, shift, pad=1000):
    nsources = len(fp)

    # pad with the edge values
    fp_pad = np.pad(fp, pad, 'edge')
    fdp_pad = np.pad(fdp, pad, 'edge')

    sl = slice(pad,pad+nsources,1)
    fp_shift = np.roll(fp_pad, shift)[sl]
    fdp_shift = np.roll(fdp_pad, shift)[sl]
    return fp_shift, fdp_shift


if __name__ == '__main__':
    from dials.util import show_mail_on_error

    with show_mail_on_error():
        script = Script()
        script.run()
