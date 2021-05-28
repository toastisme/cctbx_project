from __future__ import absolute_import, division, print_function
from copy import deepcopy
from scipy.interpolate import interp1d
from scipy.optimize import lsq_linear
from scipy.optimize import basinhopping
from libtbx import easy_pickle
import h5py
from dxtbx.model.experiment_list import ExperimentList
import pandas
from scitbx.matrix import sqr, col
from simtbx.nanoBragg.anisotropic_mosaicity import AnisoUmats

ROTX_ID = 0
ROTY_ID = 1
ROTZ_ID = 2
NCELLS_ID = 9
UCELL_ID_OFFSET = 3


# LIBTBX_SET_DISPATCHER_NAME simtbx.diffBragg.hopper

class levmar_out:
    def __init__(self, x):
        self.x = x  # parameters


import numpy as np
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
niter_per_J = 3
  .type = int
  .help = if using gradient descent, compute gradients 
  .help = every niter_per_J iterations . 
rescale_params = True
  .type = bool
  .help = use rescaled range parameters
use_likelihood_target = False
  .type = bool
  .help = if True, then use negative log Likelihood derived from a gaussian noise model
  .help = as opposed to the least squares target equation
model_fdp = False
  .type = bool
  .help = use the sigmoid model for fdp
refine_fp_fdp_shift = False
  .type = bool
  .help = refine an energy shift in the fp, fdp 
  .help = gradients not supported, so method should be Nelder-Mead
best_pickle = None
  .type = str
  .help = path to a pandas pickle containing the best models for the experiments
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
    .help = restraint factor for the scale G 
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
    .help = restraint target for scale G 
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
skip = None
  .type = int
  .help = skip this many exp
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

from scipy.ndimage import binary_dilation, label, generate_binary_structure, find_objects


def label_strong_region(subimg):
    lab, nlab = label(subimg > np.percentile(subimg, 95))
    counts = [(lab == i).sum() for i in range(1, nlab + 1)]
    peakmask = lab == (np.argmax(counts) + 1)
    bs = generate_binary_structure(2, 1)
    peakmask = binary_dilation(peakmask, bs, iterations=2)
    return peakmask


class TargetFunc:
    def __init__(self, SIM, niter_per_J=1):
        self.niter_per_J = niter_per_J
        self.all_x = []
        self.old_J = None
        self.old_model = None
        self.delta_x = None
        self.iteration = 0
        self.minima = []
        self.SIM = SIM

    def at_minimum_quiet(self, x, f, accept):
        self.iteration = 0
        self.all_x = []
        self.minima.append((f,x,accept))

    def at_minimum(self, x, f, accept):
        self.iteration = 0
        self.all_x = []
        look_at_x(x,self.SIM)
        self.minima.append((f,x,accept))

    def __call__(self, x, *args, **kwargs):
        if self.all_x:
            self.delta_x = x - self.all_x[-1]
        update_terms = None
        if not self.iteration % (self.niter_per_J) == 0:
            update_terms = (self.delta_x, self.old_J, self.old_model)
        self.all_x.append(x)
        f, g, model, J = target_func(x, update_terms, *args, **kwargs)
        self.old_model = model
        self.old_J = J
        self.iteration += 1
        if g is not None:
            return f, g
        else:
            return f



from scipy.optimize import OptimizeResult


class LevMar:

    def __init__(self, data,trusted,sigmas, up, down, eps4):
        self.trusted = trusted
        self.weights = (1/sigmas**2)[trusted]
        self.W = np.diag(self.weights)
        self.data = data
        self.up = up
        self.down = down
        self.eps4 = eps4

    def chi_sq(self, model):
        resid = (self.data-model)[self.trusted]**2
        return (resid*self.weights).sum()

    def __call__(self, fun, x0, args=(), maxfev=None, **options):
        SIM,pfs,background,verbose,damper,maxiter = args
        x = np.array(x0).astype(float)
        iteration = 0
        curr_model_pix = curr_sumsq = J = None
        while 1:
            if iteration > maxiter:
                break
            if curr_model_pix is None:
                curr_model_pix, J = model(x, SIM, pfs, verbose=False, compute_grad=True)
                J = J[:, self.trusted].T
                curr_model_pix += background
                curr_sumsq = self.chi_sq(curr_model_pix)

            dBeta = self.data - curr_model_pix
            JTW = np.dot(J.T, self.W)
            JTWJ = np.dot(JTW,J)
            damp_diag_JTWJ = damper*np.diag(JTWJ)
            a = JTWJ + damp_diag_JTWJ  # should be nparam x nparm
            b = np.dot(JTW, dBeta[self.trusted])
            out = lsq_linear(a,b)
            delta_x = out.x
            #delta_x, stop_condition, niters, solver_resid = lsmr(a, b)[:4]

            new_x = x + delta_x
            new_model_pix, newJ = model(new_x, SIM, pfs, verbose=False, compute_grad=True)
            newJ = newJ[:,self.trusted].T
            new_model_pix += background
            new_sumsq = self.chi_sq(new_model_pix)
            shift_norm = np.dot(delta_x, np.dot(damp_diag_JTWJ, delta_x) + b)
            rho = (curr_sumsq-new_sumsq) / shift_norm
            if rho > self.eps4:
                damper = max(damper/self.down, 1e-10)
                x = new_x
                curr_sumsq = new_sumsq
                curr_model_pix = new_model_pix
                J = newJ
                step_success = True
            else:
                damper = min(damper*self.up, 1e10)
                step_success = False
            print("iter%5d: step %s; Resid=%5.5g, ShiftedResid=%5.5g; rho=%5.5g damper=%5.1f " % (
                iteration + 1, "succeed" if step_success else "fail", curr_sumsq, new_sumsq, rho , damper))
            iteration += 1

        return OptimizeResult(fun=new_sumsq, x=x, nit=iteration, nfev=iteration, success=iteration > 1)


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

        input_lines = input_lines[self.params.skip:]
        for i_exp, line in enumerate(input_lines):
            if i_exp == self.params.max_process:
                break
            if i_exp % COMM.size != COMM.rank:
                continue

            print("COMM.rank %d on shot  %d / %d" % (COMM.rank, i_exp + 1, len(input_lines)))
            exp, ref, spec = line.strip().split()
            best = None
            if best_models is not None:
                best = best_models.query("exp_name=='%s'" % exp)
                if len(best) != 1:
                    raise ValueError("Should be 1 entry for exp %s in best pickle %s" % (exp, self.params.best_pickle))
            self.params.simulator.spectrum.filename = spec
            Modeler = DataModeler(self.params)
            if not Modeler.GatherFromExperiment(exp, ref):
                continue
            Modeler.SimulatorFromExperiment(best)


            # initial parameters (all set to 1, 7 parameters (scale, rotXYZ, Ncells_abc) per crystal (sausage) and then the unit cell parameters
            nparam = 7 * Modeler.SIM.num_xtals + len(Modeler.SIM.ucell_man.variables)
            if self.params.refine_fp_fdp_shift:
                nparam += 1
            if self.params.rescale_params:
                x0 = [1] * nparam
            else:
                x0 = [np.nan]*nparam
                for i_xtal in range(Modeler.SIM.num_xtals):
                    x0[7*i_xtal] = Modeler.SIM.Scale_params[i_xtal].init
                    x0[7*i_xtal+1] = Modeler.SIM.RotXYZ_params[3*i_xtal].init
                    x0[7*i_xtal+2] = Modeler.SIM.RotXYZ_params[3*i_xtal+1].init
                    x0[7*i_xtal+3] = Modeler.SIM.RotXYZ_params[3*i_xtal+2].init
                    x0[7*i_xtal+4] = Modeler.SIM.Nabc_params[3*i_xtal].init
                    x0[7*i_xtal+5] = Modeler.SIM.Nabc_params[3*i_xtal+1].init
                    x0[7*i_xtal+6] = Modeler.SIM.Nabc_params[3*i_xtal+2].init

                nucell = len(Modeler.SIM.ucell_man.variables)
                for i_ucell in range(nucell):
                    x0[7*Modeler.SIM.num_xtals+i_ucell] = Modeler.SIM.ucell_params[i_ucell].init

                if np.isnan(x0[-1]):
                    x0[-1] = Modeler.SIM.shift_param.init
            x = Modeler.Minimize(x0)
            Modeler.save_up(x, exp, i_exp)


class DataModeler:

    def __init__(self, params):
        """ params is a simtbx.diffBragg.hopper phil"""
        self.params = params
        self.SIM = None
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

    def GatherFromExperiment(self, exp, ref):
        self.E = ExperimentListFactory.from_json_file(exp)[0]
        if self.params.opt_det is not None:
            opt_det_E = ExperimentListFactory.from_json_file(self.params.opt_det, False)[0]
            self.E.detector = opt_det_E.detector

        refls = flex.reflection_table.from_file(ref)
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

        all_data = []
        all_pid = []
        all_fast = []
        all_slow = []
        all_fast_relative = []
        all_slow_relative = []
        all_trusted = []
        all_sigmas = []
        all_background = []
        roi_id = []
        all_a, all_b, all_c = [], [], []
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
            all_fast += list(X.ravel() + x1)
            all_fast_relative += list(X.ravel())
            all_slow += list(Y.ravel() + y1)
            all_slow_relative += list(Y.ravel())
            all_data += list(data)
            npix = len(data)  # np.sum(trusted)
            all_pid += [pid] * npix
            roi_id += [i_roi] * npix
            a, b, c = self.tilt_abc[i_roi]
            all_a += [a] * npix
            all_b += [b] * npix
            all_c += [c] * npix
        pan_fast_slow = np.ascontiguousarray((np.vstack([all_pid, all_fast, all_slow]).T).ravel())
        self.pan_fast_slow = flex.size_t(pan_fast_slow)
        self.all_background = np.array(all_background)
        self.roi_id = np.array(roi_id)
        self.all_data = np.array(all_data)
        self.all_sigmas = np.array(all_sigmas)
        # note rare chance for sigmas to be nan if the args of sqrt is below 0
        self.all_trusted = np.logical_and(np.array(all_trusted), ~np.isnan(all_sigmas))
        self.npix_total = len(all_data)
        self.all_fast = np.array(all_fast)
        self.all_slow = np.array(all_slow)
        self.simple_weights = 1/self.all_sigmas**2
        self.u_id = set(self.roi_id)
        return True

    def SimulatorFromExperiment(self, best=None):
        """optional best parameter is a single row of a pandas datafame containing the starting
        models, presumably optimized from a previous minimzation using this program"""

        ParameterType = RangedParameter if self.params.rescale_params else NormalParameter

        complex_F = None
        if self.params.complex_F is not None:
            complex_F = easy_pickle.load(self.params.complex_F)
        if best is not None:
            # set the crystal Umat (rotational displacement) and Bmat (unit cell)
            # Umatrix
            xax = col((-1, 0, 0))
            yax = col((0, -1, 0))
            zax = col((0, 0, -1))
            rotX,rotY,rotZ = best[["rotX", "rotY", "rotZ"]].values[0]
            RX = xax.axis_and_angle_as_r3_rotation_matrix(rotX, deg=False)
            RY = yax.axis_and_angle_as_r3_rotation_matrix(rotY, deg=False)
            RZ = zax.axis_and_angle_as_r3_rotation_matrix(rotZ, deg=False)
            M = RX * RY * RZ
            U = M * sqr(self.E.crystal.get_U())
            self.E.crystal.set_U(U)

            # Bmatrix:
            ucparam = best[["a","b","c","al","be","ga"]].values[0]
            ucman = utils.manager_from_params(ucparam)
            self.E.crystal.set_B(ucman.B_recipspace)
            # mosaic block
            self.params.init.Nabc = tuple(best.ncells.values[0])
            # scale factor
            self.params.init.G = best.spot_scales.values[0]

        self.SIM = utils.simulator_from_expt_and_params(self.E, self.params, complex_F=complex_F)

        self.SIM.D.no_Nabc_scale = self.params.no_Nabc_scale
        self.SIM.num_xtals = self.params.number_of_xtals
        if self.params.eta_refine:
            self.SIM.umat_maker = AnisoUmats(num_random_samples=self.params.num_mosaic_blocks)
        self.SIM.Nabc_params = []
        self.SIM.RotXYZ_params = []
        self.SIM.Scale_params = []
        for i_xtal in range(self.SIM.num_xtals):
            for ii in range(3):
                p = ParameterType()
                p.sigma = self.params.sigmas.Nabc[ii]
                p.init = self.params.init.Nabc[ii]
                # set the mosaic block size
                p.minval = self.params.mins.Nabc[ii]
                p.maxval = self.params.maxs.Nabc[ii]
                self.SIM.Nabc_params.append(p)

                p = ParameterType()
                p.sigma = self.params.sigmas.RotXYZ[ii]
                p.init = 0
                p.minval = self.params.mins.RotXYZ[ii] * np.pi / 180.
                p.maxval = self.params.maxs.RotXYZ[ii] * np.pi / 180.
                self.SIM.RotXYZ_params.append(p)

            p = ParameterType()
            p.sigma = self.params.sigmas.G
            p.init = self.params.init.G
            p.minval = self.params.mins.G
            p.maxval = self.params.maxs.G
            self.SIM.Scale_params.append(p)

        ucell_man = utils.manager_from_crystal(self.E.crystal)
        ucell_vary_perc = self.params.ucell_edge_perc / 100.
        self.SIM.ucell_params = []
        for i_uc, (name, val) in enumerate(zip(ucell_man.variable_names, ucell_man.variables)):
            if "Ang" in name:
                minval = val - ucell_vary_perc * val
                maxval = val + ucell_vary_perc * val
            else:
                val_in_deg = val * 180 / np.pi
                minval = (val_in_deg - self.params.ucell_ang_abs) * np.pi / 180.
                maxval = (val_in_deg + self.params.ucell_ang_abs) * np.pi / 180.
            p = ParameterType()
            p.sigma = self.params.sigmas.ucell[i_uc]
            p.init = val
            p.minval = minval
            p.maxval = maxval
            if not self.params.quiet: print(
                "Unit cell variable %s (currently=%f) is bounded by %f and %f" % (name, val, minval, maxval))
            self.SIM.ucell_params.append(p)
        self.SIM.ucell_man = ucell_man
        # eta_max = self.params.maxs.eta
        # P.add("eta_a", value=0, min=0, max=eta_max * rad, vary=self.params.eta_refine)
        # P.add("eta_b", value=0, min=0, max=eta_max * rad, vary=self.params.eta_refine)
        # P.add("eta_c", value=0, min=0, max=eta_max * rad, vary=self.params.eta_refine)
        if self.params.model_fdp:
            offset, amp, center, slope = self.params.init.fdp_center_and_slope
            wavelens_modeled, _ = zip(*self.SIM.beam.spectrum)
            en_model = utils.ENERGY_CONV / np.array(wavelens_modeled)
            fdp_modeled = utils.f_double_prime(en_model, offset, amp, center, slope)
            fp_modeled = utils.f_prime(fdp_modeled)
            self.SIM.en_model = en_model  # NOTE, this assumes the energy axis shouldn't change across experiments
            self.SIM.fdp_amp = amp
            self.SIM.fdp_offset = offset
            self.SIM.D.fprime_fdblprime = list(fp_modeled), list(fdp_modeled)
            assert self.params.atom_data_file is not None
            assert complex_F is not None

        elif self.params.fp_fdp_file is not None:
            en, fp, fdblp = np.loadtxt(self.params.fp_fdp_file).T
            wavelens_modeled,_ = zip(*self.SIM.beam.spectrum)
            en_model = utils.ENERGY_CONV / np.array(wavelens_modeled)

            fp_modeled = interp1d(en, fp,bounds_error=True)(en_model)
            fdp_modeled = interp1d(en, fdblp,bounds_error=True)(en_model)

            # preserve the initial
            self.SIM.fp_reference = deepcopy(fp_modeled)
            self.SIM.fdp_reference = deepcopy(fdp_modeled)

            fp_modeled, fdp_modeled = shift_fp_fdp(self.SIM.fp_reference,self.SIM.fdp_reference, self.params.init.shift)
            self.SIM.D.fprime_fdblprime = list(fp_modeled), list(fdp_modeled)
            assert self.params.atom_data_file is not None
            assert complex_F is not None

        if self.params.atom_data_file is not None:
            x,y,z,B,o = map(list, np.loadtxt(self.params.atom_data_file).T)
            self.SIM.D.heavy_atom_data =x,y,z,B,o
        self.SIM.shift_param = None
        if self.params.refine_fp_fdp_shift:
            if self.params.method not in [None, "Nelder-Mead", "Powell"]:
                raise NotImplemented("method %s not supported for refining shift" % self.params.method)
            p = ParameterType()
            p.init = self.params.init.shift
            p.minval = -200
            p.maxval = 200
            p.sigma = 1
            self.SIM.shift_param = p

    def Minimize(self, x0):
        target = TargetFunc(SIM=self.SIM, niter_per_J=self.params.niter_per_J)

        if self.params.method is None:
            method = "Nelder-Mead"
        else:
            method = self.params.method

        if self.params.refiner.randomize_devices is not None:
            dev = np.random.choice(self.params.refiner.num_devices)
        else:
            dev = COMM.rank % self.params.refiner.num_devices
        self.SIM.D.device_Id = dev
        pos_data = self.all_data.copy()
        pos_data[ pos_data <0]= 0
        maxfev = self.params.nelder_mead_maxfev * self.npix_total
        H = hopper_minima(self.SIM)
        at_min = target.at_minimum
        if self.params.quiet:
            at_min = target.at_minimum_quiet # H = None
            #target.at_minimum = None
        out = None
        if method=="hybrid":
            # note for reference, the args for target_func
            args = (self.SIM, self.pan_fast_slow, self.all_data,
                    self.all_sigmas, self.all_trusted, self.all_background, not self.params.quiet, self.params, False)
            # in hybrid mode, start with a nelder-mead descent hop
            out = basinhopping(target, x0,
                               niter=self.params.niter,
                               minimizer_kwargs={'args': args, "method": "Nelder-Mead", 'options':{'maxfev': maxfev}},
                               T=self.params.temp,
                               callback=at_min,
                               disp=not self.params.quiet,
                               stepsize=self.params.stepsize)

        if method in ["hybrid", "L-BFGS-B", "BFGS", "CG", "dogleg", "SLSQP", "Newton-CG", "trust-ncg", "trust-krylov", "trust-exact", "trust-ncg", "levmar"]:
            if method == "hybrid":
                assert out is not None
                method = "L-BFGS-B"
                x0 = out.x
                niter = self.params.hybrid_iter
            else:
                niter = self.params.niter
            self.SIM.D.refine(ROTX_ID)
            self.SIM.D.refine(ROTY_ID)
            self.SIM.D.refine(ROTZ_ID)
            self.SIM.D.refine(NCELLS_ID)
            for i_ucell in range(len(self.SIM.ucell_man.variables)):
                self.SIM.D.refine(UCELL_ID_OFFSET + i_ucell)
                #self.SIM.D.set_ucell_derivative_matrix(
                #    i_ucell + UCELL_ID_OFFSET,
                #    self.SIM.ucell_man.derivative_matrices[i_ucell])

            args = (self.SIM, self.pan_fast_slow, self.all_data,
                    self.all_sigmas, self.all_trusted, self.all_background, not self.params.quiet, self.params, True)
            #from scipy.optimize import shgo
            #bounds = [(None, None)] *len(x0)
            #bounds = [(None,None)]*len(x0)#(-np.inf,np.inf)]+ [(-p,5)]*3 + [(3,500)]*3 + [ (self.SIM.ucell_params[i].minval, self.SIM.ucell_params[i].maxval) for i in range(len(self.SIM.ucell_params))]
            #out = shgo(lbfgs_func, bounds, args=args) #, options={"jac": True})#, minimizer_kwargs={"method": "SLSQP"} )
            #from IPython import embed
            #embed()
            if method=="levmar":
                raise NotImplemented("Nope Levmar")
                method = LevMar(self.all_data, self.all_trusted, self.all_sigmas, up=self.params.levmar.up,
                                down=self.params.levmar.down, eps4=self.params.levmar.eps4)
                args = (self.SIM, self.pan_fast_slow,
                        self.all_background, not self.params.quiet)
                args = args + (self.params.levmar.damper, self.params.levmar.maxiter)
            out = basinhopping(target, x0,
                               niter=niter,
                               minimizer_kwargs={'args': args, "method": method, "jac": True,
                                                 'hess': self.params.hess },
                               T=self.params.temp,
                               callback=at_min, #H,
                               disp=not self.params.quiet,
                               stepsize=self.params.stepsize)
        else:
            #args = (self.SIM, self.pan_fast_slow, self.all_data,
            #        self.all_sigmas, self.all_trusted, self.all_background, pos_data, not self.params.quiet)
            args = (self.SIM, self.pan_fast_slow, self.all_data,
                    self.all_sigmas, self.all_trusted, self.all_background, not self.params.quiet, self.params, False)
            out = basinhopping(target, x0,
                               niter=self.params.niter,
                               minimizer_kwargs={'args': args, "method": method, 'options':{'maxfev': maxfev}},
                               T=self.params.temp,
                               callback=at_min, #$H,
                               disp=not self.params.quiet,
                               stepsize=self.params.stepsize)

        if not self.params.rescale_params:
            X = np.array(target.all_x)
            sig = 1 / np.std(X, 0)
            sig2 = sig/ sig.sum()
            print("G", sig[0], sig2[0])
            print("rotX", sig[1], sig2[1])
            print("rotY", sig[2], sig2[2])
            print("rotZ", sig[3], sig2[3])
            print("Na", sig[4], sig2[4])
            print("Nb", sig[5], sig2[5])
            print("Nc", sig[6], sig2[6])
            for i_uc, name in enumerate(self.SIM.ucell_man.variable_names):
                print(name, sig[7+i_uc], sig2[7+i_uc])

        P = out.x
        return P

    def save_up(self, x, exp, i_exp):
        # NOTE fixme
        best_model,_ = model(x, self.SIM, self.pan_fast_slow, compute_grad=False)
        #best_model += self.all_background
        print("Optimized:")
        look_at_x(x,self.SIM)

        if self.SIM.num_xtals == 1:
            save_to_pandas(x, self.SIM, exp, self.params, self.E, i_exp)

        rank_imgs_outdir = os.path.join(self.params.outdir, "imgs", "rank%d" % COMM.rank)
        if not os.path.exists(rank_imgs_outdir):
            os.makedirs(rank_imgs_outdir)
        basename = os.path.splitext(os.path.basename(exp))[0]
        img_path = os.path.join(rank_imgs_outdir, "%s_%s_%d.h5" % ("simplex", basename, i_exp))
        # save_model_Z(img_path, all_data, best_model, pan_fast_slow, sigma_rdout)

        data_subimg, model_subimg, strong_subimg, bragg_subimg = get_data_model_pairs(self.rois, self.pids, self.roi_id, best_model, self.all_data, background=self.all_background)

        comp = {"compression": "lzf"}
        with h5py.File(img_path, "w") as h5:
            for i_roi in range(len(data_subimg)):
                h5.create_dataset("data/roi%d" % i_roi, data=data_subimg[i_roi], **comp)
                h5.create_dataset("model/roi%d" % i_roi, data=model_subimg[i_roi], **comp)
                if bragg_subimg[0] is not None:
                    h5.create_dataset("bragg/roi%d" % i_roi, data=bragg_subimg[i_roi], **comp)
            h5.create_dataset("rois", data=self.rois)
            h5.create_dataset("pids", data=self.pids)
            h5.create_dataset("sigma_rdout", data=self.sigma_rdout)

        if self.params.plot_at_end:
            import pylab as plt
            fig, axs = plt.subplots(nrows=1, ncols=2)
            while 1:
                for i, (d, m) in enumerate(zip(data_subimg, model_subimg)):
                    axs[0].clear()
                    axs[1].clear()
                    axs[1].imshow(m)
                    axs[0].imshow(d)
                    axs[1].set_title("model %d" % i)
                    axs[0].set_title("data %d" % i)
                    plt.draw()
                    plt.pause(1.3)
                    s = strong_subimg[i]
                    if s is not None:
                        sm = m.copy()
                        sd = d.copy()
                        sm[~s] = np.nan
                        sd[~s] = np.nan
                        axs[1].images[0].set_data(sm)
                        axs[0].images[0].set_data(sd)
                        axs[1].set_title("model %d (strong)" % i)
                        axs[0].set_title("data %d (strong)" % i)
                        plt.draw()
                        plt.pause(1.3)
        if self.params.embed_at_end:
            from IPython import embed
            embed()

        self.SIM.D.free_all()
        self.SIM.D.free_Fhkl2()

    def chi_sq(self, model):
        resid = (self.all_data - model)[self.all_trusted] ** 2
        return (resid * self.simple_weights[self.all_trusted]).sum()



def get_data_model_pairs(rois, pids, roi_id, best_model, all_data, strong_flags=None, background=None):
    all_dat_img, all_mod_img = [], []
    all_strong = []
    all_bragg = []
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
        if background is not None:
            bg = background[roi_id==i_roi].reshape((y2-y1, x2-x1))
            # assume mod does not contain background
            all_bragg.append(mod)
            all_mod_img.append(mod+bg)
        else:  # assume mod contains background
            all_mod_img.append(mod)
            all_bragg.append(None)
        # print("Roi %d, max in data=%f, max in model=%f" %(i_roi, dat.max(), mod.max()))
    return all_dat_img, all_mod_img, all_strong, all_bragg


def look_at_x(x, SIM):
    num_per_xtal_params = SIM.num_xtals * (7)
    n_ucell_param = len(SIM.ucell_man.variables)
    #if n_ucell_param + num_per_xtal_params != len(x):
    #    raise ValueError("weird x")
    params_per_xtal = np.array_split(x[:num_per_xtal_params], SIM.num_xtals)
    unitcell_variables = [SIM.ucell_params[i].get_val(xval) for i, xval in
                          enumerate(x[num_per_xtal_params:num_per_xtal_params+n_ucell_param])]

    for i_xtal in range(SIM.num_xtals):
        scale_reparam, rotX_reparam, rotY_reparam, rotZ_reparam, \
        Na_reparam, Nb_reparam, Nc_reparam = params_per_xtal[i_xtal]

        rotX = SIM.RotXYZ_params[i_xtal * 3].get_val(rotX_reparam)
        rotY = SIM.RotXYZ_params[i_xtal * 3 + 1].get_val(rotY_reparam)
        rotZ = SIM.RotXYZ_params[i_xtal * 3 + 2].get_val(rotZ_reparam)

        scale = SIM.Scale_params[i_xtal].get_val(scale_reparam)

        Na = SIM.Nabc_params[i_xtal * 3].get_val(Na_reparam)
        Nb = SIM.Nabc_params[i_xtal * 3 + 1].get_val(Nb_reparam)
        Nc = SIM.Nabc_params[i_xtal * 3 + 2].get_val(Nc_reparam)

        print("\tXtal %d:" % i_xtal)
        print("\tNcells=%f %f %f" % (Na, Nb, Nc))
        print("\tspot scale=%f" % (scale))
        angles = tuple([x * 180 / np.pi for x in [rotX, rotY, rotZ]])
        print("\trotXYZ= %f %f %f (degrees)" % angles)
    print("\tunitcell= %3.5f %3.5f %3.5f %3.5f %3.5f %3.5f" % SIM.ucell_man.unit_cell_parameters)
    if SIM.shift_param is not None:
        shift = SIM.shift_param.get_val(x[-1])
        print("\tfp_fdp shift= %3.1f" % shift)


@profile
def model(x, SIM, pfs, verbose=True, compute_grad=True):
    verbose = False
    #if compute_grad and SIM.num_xtals > 1:
    #    raise NotImplemented("Grads not implemented for multiple xtals")
    num_per_xtal_params = SIM.num_xtals * (7)
    n_ucell_param = len(SIM.ucell_man.variables)
    if SIM.shift_param is not None:
        assert n_ucell_param+num_per_xtal_params+1 == len(x)
        shift_val = SIM.shift_param.get_val(x[-1])
        fp_shift, fdp_shift = shift_fp_fdp(SIM.fp_reference, SIM.fdp_reference, int(np.round(shift_val)))
        SIM.D.fprime_fdblprime = list(fp_shift), list(fdp_shift)
    else:
        assert n_ucell_param+num_per_xtal_params == len(x)
    params_per_xtal = np.array_split(x[:num_per_xtal_params], SIM.num_xtals)
    unitcell_var_reparam = x[num_per_xtal_params:num_per_xtal_params+n_ucell_param]
    unitcell_variables = [SIM.ucell_params[i].get_val(xval) for i, xval in enumerate(unitcell_var_reparam)]
    SIM.ucell_man.variables = unitcell_variables
    Bmatrix = SIM.ucell_man.B_recipspace
    SIM.D.Bmatrix = Bmatrix
    if compute_grad:
        #SIM.D.refine(ROTX_ID)
        #SIM.D.refine(ROTY_ID)
        #SIM.D.refine(ROTZ_ID)
        #SIM.D.refine(NCELLS_ID)
        for i_ucell in range(len(unitcell_variables)):
        #    SIM.D.refine(UCELL_ID_OFFSET+i_ucell)
            SIM.D.set_ucell_derivative_matrix(
                i_ucell + UCELL_ID_OFFSET,
                SIM.ucell_man.derivative_matrices[i_ucell])
        # NOTE scale factor gradient is computed directly from the forward model below
    #else:
    #    SIM.D.fix(ROTX_ID)
    #    SIM.D.fix(ROTY_ID)
    #    SIM.D.fix(ROTZ_ID)
    #    SIM.D.fix(NCELLS_ID)
    #    for i_ucell in range(len(unitcell_variables)):
    #        SIM.D.fix(UCELL_ID_OFFSET+i_ucell)


    xax = col((-1, 0, 0))
    yax = col((0, -1, 0))
    zax = col((0, 0, -1))

    npix = int(len(pfs) / 3)
    nparam = len(x)
    J = np.zeros((nparam, npix))  # note: order is: scale, rotX, rotY, rotZ, Na, Nb, Nc, ... (for each xtal), then ucell0, ucell1 , ucell2, ..
    model_pix = None
    for i_xtal in range(SIM.num_xtals):
        #SIM.D.raw_pixels_roi *= 0 #todo do i matter?
        scale_reparam, rotX_reparam, rotY_reparam, rotZ_reparam, \
        Na_reparam, Nb_reparam, Nc_reparam = params_per_xtal[i_xtal]

        rotX = SIM.RotXYZ_params[i_xtal * 3].get_val(rotX_reparam)
        rotY = SIM.RotXYZ_params[i_xtal * 3 + 1].get_val(rotY_reparam)
        rotZ = SIM.RotXYZ_params[i_xtal * 3 + 2].get_val(rotZ_reparam)

        ## update parameters:

        SIM.D.set_value(ROTX_ID, rotX)
        SIM.D.set_value(ROTY_ID, rotY)
        SIM.D.set_value(ROTZ_ID, rotZ)
        #RX = xax.axis_and_angle_as_r3_rotation_matrix(rotX, deg=False)
        #RY = yax.axis_and_angle_as_r3_rotation_matrix(rotY, deg=False)
        #RZ = zax.axis_and_angle_as_r3_rotation_matrix(rotZ, deg=False)
        #M = RX * RY * RZ
        #SIM.D.Umatrix = M * sqr(SIM.crystal.dxtbx_crystal.get_U())

        scale = SIM.Scale_params[i_xtal].get_val(scale_reparam)
        #SIM.D.spot_scale = scale

        Na = SIM.Nabc_params[i_xtal * 3].get_val(Na_reparam)
        Nb = SIM.Nabc_params[i_xtal * 3 + 1].get_val(Nb_reparam)
        Nc = SIM.Nabc_params[i_xtal * 3 + 2].get_val(Nc_reparam)
        SIM.D.set_ncells_values(tuple([Na, Nb, Nc]))

        # SIM.D.verbose = 1
        # SIM.D.printout_pixel_fastslow = pfs[1],pfs[2]
        if verbose: print("\tXtal %d:" % i_xtal)
        if verbose: print("\tNcells=%f %f %f" % (Na, Nb, Nc))
        if verbose: print("\tspot scale=%f" % (scale))
        angles = tuple([x * 180 / np.pi for x in [rotX, rotY, rotZ]])
        if verbose: print("\trotXYZ= %f %f %f (degrees)" % angles)
        SIM.D.add_diffBragg_spots(pfs)

        pix = SIM.D.raw_pixels_roi[:npix]
        pix = pix.as_numpy_array()
        if model_pix is None:
            model_pix = scale*pix #SIM.D.raw_pixels_roi.as_numpy_array()[:npix]
        else:
            model_pix += scale*pix #SIM.D.raw_pixels_roi.as_numpy_array()[:npix]

        if compute_grad:
            scale_grad = model_pix / scale
            scale_grad = SIM.Scale_params[i_xtal].get_deriv(scale_reparam, scale_grad)
            J[7*i_xtal] += scale_grad

            rotX_grad = scale*SIM.D.get_derivative_pixels(ROTX_ID).as_numpy_array()[:npix]
            rotY_grad = scale*SIM.D.get_derivative_pixels(ROTY_ID).as_numpy_array()[:npix]
            rotZ_grad = scale*SIM.D.get_derivative_pixels(ROTZ_ID).as_numpy_array()[:npix]
            rotX_grad = SIM.RotXYZ_params[i_xtal*3].get_deriv(rotX_reparam, rotX_grad)
            rotY_grad = SIM.RotXYZ_params[i_xtal*3+1].get_deriv(rotY_reparam, rotY_grad)
            rotZ_grad = SIM.RotXYZ_params[i_xtal*3+2].get_deriv(rotZ_reparam, rotZ_grad)
            J[7*i_xtal + 1] += rotX_grad
            J[7*i_xtal + 2] += rotY_grad
            J[7*i_xtal + 3] += rotZ_grad

            Nabc_grad = SIM.D.get_ncells_derivative_pixels()
            #Na_grad = scale*SIM.D.get_Na_derivative_pixels()[:npix]
            Na_grad = scale*(Nabc_grad[0][:npix].as_numpy_array())
            Nb_grad = scale*(Nabc_grad[1][:npix].as_numpy_array())
            Nc_grad = scale*(Nabc_grad[2][:npix].as_numpy_array())

            #Na_grad, Nb_grad, Nc_grad = [scale*d.as_numpy_array()[:npix] for d in SIM.D.get_ncells_derivative_pixels()]
            Na_grad = SIM.Nabc_params[i_xtal * 3].get_deriv(Na_reparam, Na_grad)
            Nb_grad = SIM.Nabc_params[i_xtal * 3 + 1].get_deriv(Nb_reparam, Nb_grad)
            Nc_grad = SIM.Nabc_params[i_xtal * 3 + 2].get_deriv(Nc_reparam, Nc_grad)
            J[7*i_xtal + 4] += Na_grad
            J[7*i_xtal + 5] += Nb_grad
            J[7*i_xtal + 6] += Nc_grad

            # note important to keep gradients in same order as the parameters x
            #ucell_grad = []
            for i_ucell in range(n_ucell_param):
                d = scale*SIM.D.get_derivative_pixels(UCELL_ID_OFFSET+i_ucell).as_numpy_array()[:npix]
                d = SIM.ucell_params[i_ucell].get_deriv(unitcell_var_reparam[i_ucell], d)
                #ucell_grad.append(d)
                J[7*SIM.num_xtals + i_ucell] += d
            #J = np.vstack([scale_grad, rotX_grad, rotY_grad, rotZ_grad, Na_grad, Nb_grad, Nc_grad])
            #J = np.vstack((J, ucell_grad))

    if verbose: print("\tunitcell= %3.5f %3.5f %3.5f %3.5f %3.5f %3.5f" % SIM.ucell_man.unit_cell_parameters)
    return model_pix, J


def nelmead_func(x, SIM, pfs, data, sigmas, trusted, background, pos_data, verbose=True):
    model_pix, _ = model(x, SIM, pfs, verbose=verbose, compute_grad=False)
    model_pix += background
    #f = 0.33333
    #P = f * pos_data + (1-f)*model_pix
    #W = 1/((sigmas**2)*data + (0.1*P)**2)
    W = 1/sigmas**2
    resid = (model_pix - data)
    s = resid**2 * W
    sumsq = (s[trusted]).sum()
    if verbose: print("Resid=%10.7g" % sumsq)
    return sumsq


class Minimizer:

    def __init__(self):
        self.iteration = 0
        self.Jac = None

    def __call__(self, x, SIM, pfs, data, sigmas, trusted, background, verbose=True):
        compute_grad = False
        if self.iteration &(self.iteration-1) == 0:
            compute_grad = True

        model_bragg, Jac = model(x, SIM, pfs, verbose=verbose, compute_grad=compute_grad)
        if self.Jac is None:
            self.Jac = Jac
        # Jac has shape of num_param x num_pix

        model_pix = model_bragg + background

        W = 1/sigmas**2
        resid = (model_pix - data)
        grad_term = (2*resid*W)[trusted]
        self.Jac = self.Jac[:,trusted]
        g = np.array([np.sum(grad_term*Jac[param_idx]) for param_idx in range(Jac.shape[0])])

        f = (resid[trusted]**2 * W[trusted]).sum()
        gnorm = np.linalg.norm(g)
        if verbose: print("F=%10.7g, |G|=%10.7g" % (f, gnorm))
        return f, g


def target_func(x, udpate_terms, SIM, pfs, data, sigmas, trusted, background, verbose=True, params=None, compute_grad=True):

    if udpate_terms is not None:
        # if approximating the gradients, then fix the refiners
        # so we dont waste time computing them
        _compute_grad = False
        SIM.D.fix(NCELLS_ID)
        SIM.D.fix(ROTX_ID)
        SIM.D.fix(ROTY_ID)
        SIM.D.fix(ROTZ_ID)
        for i_ucell in range(len(SIM.ucell_man.variables)):
            SIM.D.fix(UCELL_ID_OFFSET + i_ucell)
    elif compute_grad:
        # actually compute the gradients
        _compute_grad = True
        SIM.D.let_loose(NCELLS_ID)
        SIM.D.let_loose(ROTX_ID)
        SIM.D.let_loose(ROTY_ID)
        SIM.D.let_loose(ROTZ_ID)
        for i_ucell in range(len(SIM.ucell_man.variables)):
            SIM.D.let_loose(UCELL_ID_OFFSET + i_ucell)
    else:
        _compute_grad = False
    model_bragg, Jac = model(x, SIM, pfs, verbose=verbose, compute_grad=_compute_grad)

    if udpate_terms is not None:
        # try a Broyden update ?
        # https://people.duke.edu/~hpgavin/ce281/lm.pdf  equation 19
        delta_x, prev_J, prev_model_bragg = udpate_terms
        delta_y = model_bragg - prev_model_bragg

        delta_J = (delta_y - np.dot(prev_J.T, delta_x))
        delta_J /= np.dot(delta_x,delta_x)
        Jac = prev_J + delta_J

    # Jac has shape of num_param x num_pix

    model_pix = model_bragg + background

    LL = params.use_likelihood_target

    #if not LL:
    W = 1/sigmas**2
    if LL:
        resid = (data - model_pix)  #minor technicality , to accommodate hand-written notes
    else:
        resid = (model_pix - data)

    G, rotX,rotY, rotZ, Na,Nb,Nc,a,b,c,al,be,ga = get_param_from_x(x, SIM)

    #TODO vectorize  / generalized framework for restraints
    G0 = params.centers.G
    delG = (G0-G)

    deg = 180 / np.pi
    rotX = deg*rotX
    rotY = deg*rotY
    rotZ = deg*rotZ
    rotX0,rotY0,rotZ0 = params.centers.RotXYZ
    Na0,Nb0,Nc0 = params.centers.Nabc
    sig_rX, sig_rY, sig_rZ = params.widths.RotXYZ
    sig_Na, sig_Nb, sig_Nc = params.widths.Nabc
    del_rX = rotX0-rotX
    del_rY = rotY0-rotY
    del_rZ = rotZ0-rotZ

    del_Na = Na0 - Na
    del_Nb = Nb0 - Nb
    del_Nc = Nc0 - Nc

    if LL:
        sigma_rdout = params.refiner.sigma_r / params.refiner.adu_per_photon
        V = model_pix + sigma_rdout**2
        resid_square = resid**2
        fchi = (.5*(np.log(2*np.pi*V) + resid_square / V))[trusted].sum()   # negative log Likelihood target
        # TODO make betas.Nabc a 3-vector and remove sig_* terms
        # TODo make this a method the __call__ method of a class, and cache these terms
        Na_V = params.betas.Nabc / sig_Na**2
        Nb_V = params.betas.Nabc / sig_Nb**2
        Nc_V = params.betas.Nabc / sig_Nc**2
        rx_V = params.betas.RotXYZ / sig_rX**2
        ry_V = params.betas.RotXYZ / sig_rY**2
        rz_V = params.betas.RotXYZ / sig_rZ**2
        fN = .5*(np.log(2*np.pi*Na_V) + del_Na**2  / Na_V)
        fN += .5*(np.log(2*np.pi*Nb_V) + del_Nb**2  / Nb_V)
        fN += .5*(np.log(2*np.pi*Nc_V) + del_Nc**2  / Nc_V)

        frot = .5*(np.log(2*np.pi*rx_V) + del_rX**2  / rx_V)
        frot += .5*(np.log(2*np.pi*ry_V) + del_rY**2  / ry_V)
        frot += .5*(np.log(2*np.pi*rz_V) + del_rZ**2  / rz_V)

        G_V = params.betas.G
        fG = .5*(np.log(2*np.pi*G_V) + delG**2  /G_V)

    else:
        fchi = (resid[trusted] ** 2 * W[trusted]).sum()   # weighted least squares target
        fN = params.betas.Nabc*((del_Na / sig_Na)**2 +(del_Nb / sig_Nb)**2 + (del_Nc / sig_Nc)**2)
        frot = params.betas.RotXYZ*((del_rX/sig_rX)**2+ (del_rY/sig_rY)**2 + (del_rZ / sig_rZ)**2)
        fG = params.betas.G*delG**2

    f = fchi + frot + fN + fG
    chi = fchi / f *100
    rot = frot / f*100
    n = fN / f*100
    gg = fG / f *100
    if compute_grad:
        if LL:
            grad_term = (0.5 /V * (1-2*resid - resid_square / V))[trusted]
        else:
            grad_term = (2*resid*W)[trusted]
        Jac_t = Jac[:,trusted]
        g = np.array([np.sum(grad_term*Jac_t[param_idx]) for param_idx in range(Jac_t.shape[0])])
        if LL:
            g[0] += -delG /G_V
            # TODO , do we need the deg conversion factor?
            g[1] += -deg * del_rX / rx_V
            g[2] += -deg * del_rY / ry_V
            g[3] += -deg * del_rZ / rz_V
            g[4] += -del_Na / Na_V
            g[5] += -del_Nb / Nb_V
            g[6] += -del_Nc / Nc_V
        else:
            ber = params.betas.RotXYZ
            beN = params.betas.Nabc
            g[0] += -2*params.betas.G*delG
            g[1] += -ber*2*deg*del_rX/sig_rX**2
            g[2] += -ber*2*deg*del_rY/sig_rY**2
            g[3] += -ber*2*deg*del_rZ/sig_rZ**2
            g[4] += -beN*2*del_Na/sig_Na**2
            g[5] += -beN*2*del_Nb/sig_Nb**2
            g[6] += -beN*2*del_Nc/sig_Nc**2
        gnorm = np.linalg.norm(g)

    if compute_grad:
        if verbose: print("F=%10.7g (chi: %.1f%%, rot: %.1f%% N: %.1f%%, G: %.1f%%), |g|=%10.7g" % (f, chi, rot, n, gg,gnorm))
    else:
        if verbose: print("F=%10.7g (chi: %.1f%%, rot: %.1f%% N: %.1f%%, G: %.1f%%), |g|=NA" % (f, chi, rot, n, gg))

    return f, g, model_bragg, Jac


class hopper_minima:
    def __init__(self, SIM):
        self.minima = []
        self.SIM = SIM

    def __call__(self, x, f, accept):
        look_at_x(x,self.SIM)
        self.minima.append((f,x,accept))


def get_param_from_x(x, SIM):
    num_per_xtal_params = SIM.num_xtals * (7)
    n_ucell_param = len(SIM.ucell_man.variables)
    params_per_xtal = np.array_split(x[:num_per_xtal_params], SIM.num_xtals)
    unitcell_var_reparam = x[num_per_xtal_params:num_per_xtal_params+n_ucell_param]
    unitcell_variables = [SIM.ucell_params[i].get_val(xval) for i, xval in enumerate(unitcell_var_reparam)]
    SIM.ucell_man.variables = unitcell_variables
    a,b,c,al,be,ga = SIM.ucell_man.unit_cell_parameters
    i_xtal = 0

    scale_reparam, rotX_reparam, rotY_reparam, rotZ_reparam, \
        Na_reparam, Nb_reparam, Nc_reparam = params_per_xtal[i_xtal]

    rotX = SIM.RotXYZ_params[i_xtal * 3].get_val(rotX_reparam)
    rotY = SIM.RotXYZ_params[i_xtal * 3 + 1].get_val(rotY_reparam)
    rotZ = SIM.RotXYZ_params[i_xtal * 3 + 2].get_val(rotZ_reparam)

    scale = SIM.Scale_params[i_xtal].get_val(scale_reparam)

    Na = SIM.Nabc_params[i_xtal * 3].get_val(Na_reparam)
    Nb = SIM.Nabc_params[i_xtal * 3 + 1].get_val(Nb_reparam)
    Nc = SIM.Nabc_params[i_xtal * 3 + 2].get_val(Nc_reparam)

    return scale, rotX, rotY, rotZ, Na, Nb, Nc,a,b,c,al,be,ga


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
