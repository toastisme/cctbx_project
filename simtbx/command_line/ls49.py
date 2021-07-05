from __future__ import absolute_import, division, print_function
from simtbx.command_line.hopper import look_at_x
from scitbx.matrix import sqr, col
import pylab as plt
import numpy as np
from simtbx.diffBragg import ls49_utils

# LIBTBX_SET_DISPATCHER_NAME simtbx.diffBragg.ls49

ROTX_ID = 0
ROTY_ID = 1
ROTZ_ID = 2
PANX_ID = 15
PANO_ID = 14
PANF_ID = 17
PANS_ID=18
PANY_ID = 16
PANZ_ID = 10
NCELLS_ID = 9
UCELL_ID_OFFSET = 3


import os
from scipy.optimize import basinhopping
from libtbx.mpi4py import MPI

COMM = MPI.COMM_WORLD
from libtbx.phil import parse

from simtbx.diffBragg import utils
from simtbx.command_line.hopper import hopper_phil, DataModeler
from simtbx.command_line.hopper import model
from simtbx.diffBragg.phil import philz
from simtbx.diffBragg.refiners.parameters import RangedParameter
from simtbx.command_line.hopper import get_data_model_pairs

script_phil = """
plot_minima = False
  .type = bool
  .help = plot minimia images
jungfrau_factor = 1
  .type = float
  .help = scale jungfrau contribution to the target residual
rayonix_factor = 1
  .type = float
  .help = scale rayonix contribution to the target residual
outlier_Z = None
  .type = float
  .help = Zscore to find and remove outlier ROIS that dominate the target residual
watch_parameters = False
  .type = bool
  .help = watch parameters during minimization
fix {
  G = False
    .type = bool
    .help = fix the scale factor
  Nabc = False
    .type = bool
    .help = fix the ncells
  RotXYZ = False
    .type = bool
    .help = fix the rotXYZ
  unitcell = False
    .type = bool
    .help = fix the unitcell
  jungfrau = False
    .type = bool
    .help = hold fixed the jungfrau camera
  rayonix = False
    .type = bool
    .help = hold fixed the rayonix camera
}
"""


philz = script_phil + hopper_phil + philz
phil_scope = parse(philz)

from scipy.ndimage import binary_dilation, label, generate_binary_structure, find_objects


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
        if COMM.rank == 0:
            input_lines = open(self.params.exp_ref_spec_file, "r").readlines()
            if self.params.sanity_test_input:
                for line in input_lines:
                    for fname in line.strip().split():
                        if not os.path.exists(fname):
                            raise FileNotFoundError("File %s not there " % fname)
        input_lines = COMM.bcast(input_lines)

        input_lines = input_lines[self.params.skip:]
        for i_exp, line in enumerate(input_lines):
            if i_exp == self.params.max_process:
                break
            if i_exp % COMM.size != COMM.rank:
                continue

            print("COMM.rank %d on shot  %d / %d" % (COMM.rank, i_exp + 1, len(input_lines)))
            # its assumed theseare the Jungfrau exp, ref, spec
            exp, ref, spec = line.strip().split()

            # inject the spectrum into the params
            self.params.simulator.spectrum.filename = spec

            tstamp = ls49_utils.get_tstamp(exp)
            rayonix_exp_name = os.path.join(ls49_utils.RAYONIX_DIR, "idx-%s_refined_moved.expt" % tstamp)
            rayonix_ref_name = os.path.join(ls49_utils.RAYONIX_DIR, "idx-%s_indexed.refl" % tstamp)
            from copy import deepcopy
            self.rayonix_params = deepcopy(self.params)
            # reflection tables
            #jungfrau_R = flex.reflection_table.from_file(ref)
            #rayonix_R = flex.reflection_table.from_file(rayonix_ref_name)
            #two_refls = [jungfrau_R, rayonix_R]
            self.rayonix_params.refiner.adu_per_photon = 0.46   # From asmit
            self.rayonix_params.roi.hotpixel_mask = os.path.join(ls49_utils.RAYONIX_DIR, "mask_r4.pickle")
            self.rayonix_params.refiner.sigma_r = 0.46*1.5  # note: just a guessed, such that the readout noise level is a bit larger than a photon

            # Note, verify the crystals are the same ?
            #from dxtbx.model.experiment_list import ExperimentListFactory
            #C = ExperimentListFactory.from_json_file(exp, False)[0].crystal
            #rayonixC = ExperimentListFactory.from_json_file(rayonix_exp_name, False)[0].crystal

            JungfrauModeler = DataModeler(self.params)
            RayonixModeler = DataModeler(self.rayonix_params)

            if not JungfrauModeler.GatherFromExperiment(exp, ref):
                continue
            if not RayonixModeler.GatherFromExperiment(rayonix_exp_name, rayonix_ref_name):
                continue

            # Note, for some reason the unit cells differ,
            #  however we need the same crystal model during refinement, so do this
            JungfrauModeler.E.crystal = RayonixModeler.E.crystal
            JungfrauModeler.SimulatorFromExperiment()
            RayonixModeler.SimulatorFromExperiment()


            # initial parameters (all set to 1, 7 parameters (scale, rotXYZ, Ncells_abc) per crystal (sausage) and then the unit cell parameters
            x0 = [1] * (7 * JungfrauModeler.SIM.num_xtals) + [1] * len(JungfrauModeler.SIM.ucell_man.variables)
            #x0 += [1]*6  # 6 detector distance parameters
            RayonixModeler.SIM.DetXYZ_params = []
            JungfrauModeler.SIM.DetXYZ_params = []
            RayonixModeler.SIM.DetOFS_params = []
            JungfrauModeler.SIM.DetOFS_params = []
            for i_shift in range(3):
                p = RangedParameter()
                p.init = 0
                p.sigma = 1
                p.minval = -10e-3
                p.maxval = 10e-3
                JungfrauModeler.SIM.DetXYZ_params.append(p)

                p = RangedParameter()
                p.init = 0
                p.sigma = 1
                p.minval = -2e-3
                p.maxval = 2e-3
                RayonixModeler.SIM.DetXYZ_params.append(p)

            p = RangedParameter()
            p.init = 0
            p.sigma = 1
            p.minval = -5*np.pi/180.
            p.maxval = 5*np.pi/180
            RayonixModeler.SIM.DetOFS_params.append(p)

            p = RangedParameter()
            p.init = 0
            p.sigma = 1
            p.minval = -5*np.pi/180.
            p.maxval = 5*np.pi/180
            JungfrauModeler.SIM.DetOFS_params.append(p)

            # add a parameter for the Rayonix overall scale factor ?
            #p = RangedParameter()
            #p.sigma = self.params.sigmas.G2
            #p.init = self.params.init.G2
            #p.minval = self.params.mins.G2
            #p.maxval = self.params.maxs.G2
            #RayonixModeler.SIM.Scale_params[0] = p
            #x0.append(1)  # add a scale parameter for the Rayonix thats separate from the Jungfrau
            #x = Modeler.Minimize(x0)
            #Modeler.save_up(x, exp, i_exp)
            niter = self.params.niter
            # toggle on the gradients
            RayonixModeler.SIM.D.refine(ROTX_ID)
            RayonixModeler.SIM.D.refine(ROTY_ID)
            RayonixModeler.SIM.D.refine(ROTZ_ID)
            RayonixModeler.SIM.D.refine(NCELLS_ID)
            #RayonixModeler.SIM.D.refine(PANX_ID)
            #RayonixModeler.SIM.D.refine(PANO_ID)
            #RayonixModeler.SIM.D.refine(PANY_ID)
            #RayonixModeler.SIM.D.refine(PANZ_ID)

            JungfrauModeler.SIM.D.refine(ROTX_ID)
            JungfrauModeler.SIM.D.refine(ROTY_ID)
            JungfrauModeler.SIM.D.refine(ROTZ_ID)
            JungfrauModeler.SIM.D.refine(NCELLS_ID)
            #JungfrauModeler.SIM.D.refine(PANO_ID)
            #JungfrauModeler.SIM.D.refine(PANX_ID)
            #JungfrauModeler.SIM.D.refine(PANY_ID)
            #JungfrauModeler.SIM.D.refine(PANZ_ID)
            n_ucell_p = len(JungfrauModeler.SIM.ucell_man.variables)
            for i_ucell in range(n_ucell_p):
                RayonixModeler.SIM.D.refine(UCELL_ID_OFFSET + i_ucell)
                JungfrauModeler.SIM.D.refine(UCELL_ID_OFFSET + i_ucell)

            #fixlist = np.array([False]*len(x0))
            #bounds = [(None,None)] * len(x0)
            #for i_xtal in range(RayonixModeler.SIM.num_xtals):
            #    xx=i_xtal*7
            #    if self.params.fix.G:
            #        fixlist[xx+0] = True
            #        bounds[xx+0] = (1,1)
            #    if self.params.fix.RotXYZ:
            #        fixlist[xx+1:xx+4] = True
            #        for i in range(1,4):
            #            bounds[xx+i] = (1,1)
            #    if self.params.fix.Nabc:
            #        fixlist[xx+4:xx+7] = True
            #        for i in range(4, 7):
            #            bounds[xx + i] = (1, 1)
            #xx = RayonixModeler.SIM.num_xtals * 7
            #if self.params.fix.unitcell:
            #    fixlist[xx:xx+n_ucell_p] = True
            #    for i in range(n_ucell_p):
            #        bounds[xx+i] = (1,1)
            #if self.params.fix.rayonix:  # fix the 2 rayonix detector parameters
            #    fixlist[xx+n_ucell_p:xx+n_ucell_p+2] = True
            #    bounds[xx + n_ucell_p] = (1, 1)
            #    bounds[xx + n_ucell_p+1] = (1, 1)

            #if self.params.fix.jungfrau:  # fix the 3 Jungfray detector parameters
            #    fixlist[xx+n_ucell_p+2:] = True
            #    for i in range(4):
            #        bounds[xx+n_ucell_p+2+i] = (1,1)
            fixlist =None
            bounds =None
            args = RayonixModeler, JungfrauModeler, not self.params.quiet, self.params.outlier_Z, fixlist, self.params.watch_parameters
            # callback protocol defined here, called once per basin minimia:
            H = hopper_minima(Rayonix=RayonixModeler, Jungfrau=JungfrauModeler, plot=self.params.plot_minima)
            if self.params.quiet:
                H = None
            target = ls49_target(self.params.jungfrau_factor, self.params.rayonix_factor)
            out = basinhopping(target, x0,
                               niter=niter,
                               minimizer_kwargs={'args': args, "method": self.params.method, "jac": True,
                                                 'hess': self.params.hess , "bounds": bounds},
                               T=self.params.temp,
                               callback=H,
                               disp=not self.params.quiet,
                               stepsize=self.params.stepsize)

            if self.params.plot_at_end:
                H.P.plot(out.x,show=True)


def get_models(x,Rayonix, Jungfrau):
    models = ls49_model(x, Rayonix, Jungfrau, verbose=True)

    jun_best_model = models["jungfrau"][0]
    jun_data_subimg, jun_model_subimg, _ = \
        get_data_model_pairs(Jungfrau.rois, Jungfrau.pids,
                            Jungfrau.roi_id, jun_best_model, Jungfrau.all_data )

    ray_best_model = models["rayonix"][0]
    ray_data_subimg, ray_model_subimg, _ = \
        get_data_model_pairs(Rayonix.rois, Rayonix.pids,
                             Rayonix.roi_id, ray_best_model, Rayonix.all_data )

    det_id = ["ray"]*len(ray_data_subimg) + ["jun"] * len(jun_data_subimg)
    roi_id = list(range(len(ray_data_subimg))) + list(range(len(jun_data_subimg)))

    data = ray_data_subimg + jun_data_subimg
    model = ray_model_subimg + jun_model_subimg

    return data, model, det_id, roi_id


class Plotter:

    def __init__(self, Rayonix, Jungfrau):
        self.nrows = 4
        self.ncols = 4
        nspots = len(Rayonix.u_id)+ len(Jungfrau.u_id)
        self.nspots = nspots
        self.nspot_per_img = self.nrows*self.ncols / 2
        self.nfigs = int(np.ceil(self.nspots/float(self.nspot_per_img)))  # four comparisons per image
        self.F = []
        self.A = []
        self.Rayonix = Rayonix
        self.Jungfrau = Jungfrau

        self._setup_figs()

    def _setup_figs(self):
        self.F = []
        self.A = []
        import pylab as plt
        for i_fig in range(self.nfigs):
            fig, axs = plt.subplots(nrows=self.nrows, ncols=self.ncols)
            self.F.append(fig)
            for row in range(self.nrows):
                for col in range(self.ncols):
                    axs[row, col].axis('off')
                    axs[row, col].set_xticks([])
                    axs[row, col].set_yticks([])
            self.A.append(axs)

    def plot(self, x, show=False, pause=0.01):
        data, model, det_id, roi_id = get_models(x, Rayonix=self.Rayonix, Jungfrau=self.Jungfrau)

        for i_spot in range(self.nspots):
            i_fig = int(i_spot / self.nspot_per_img)
            axs = self.A[i_fig]
            roi_idx = i_spot % self.nrows
            col_idx = int((i_spot%self.nspot_per_img) / self.nrows)

            if col_idx == 0:
                c0, c1 = 0, 1
            else:
                c0, c1 = 2, 3
            ax0, ax1 = axs[roi_idx, c0:c1+1]
            ax0.set_aspect("auto")
            ax1.set_aspect("auto")
            ax0.set_title("%s-%d (dat)" %(det_id[i_spot], roi_id[i_spot]), fontsize=8,pad=0)
            ax1.set_title("%s-%d (mod)" %(det_id[i_spot], roi_id[i_spot]),fontsize=8,pad=0)
            ax0.set_xticks([])
            ax1.set_xticks([])
            ax0.set_yticks([])
            ax1.set_yticks([])
            ax0.imshow(data[i_spot])
            ax1.imshow(model[i_spot])

        for i_fig in range(self.nfigs):
            self.F[i_fig].subplots_adjust(left=0, right=.99, bottom=0,top=0.95)
        if show:
            plt.show()
        else:
            plt.draw()
            plt.pause(pause)


class hopper_minima:
    def __init__(self, Rayonix, Jungfrau, plot=False):
        self.Ray = Rayonix
        self.Jun = Jungfrau
        self.P = Plotter(Rayonix=Rayonix, Jungfrau=Jungfrau)
        self.minima = []
        self.plot = plot

    def __call__(self, x, f, accept):
        look_at_x(x,self.Jun.SIM)
        self.minima.append((f,x,accept))
        if self.plot:
            self.P.plot(x, show=False, pause=0.5)


class ls49_target:

    def __init__(self, jungfrau_factor=1, rayonix_factor=1):
        self.flagged_rois = None
        self.roi_contributions = None
        self.jungfrau_factor = jungfrau_factor
        self.rayonix_factor = rayonix_factor

    def __call__(self, x, Rayonix, Jungfrau, verbose, outlier_Z=None, fixlist=None, watch_params=False):
        # model both detectors pixels
        models = ls49_model(x, Rayonix=Rayonix, Jungfrau=Jungfrau, verbose=verbose and watch_params)
        ray_model, ray_Jac = models["rayonix"]
        jun_model, jun_Jac = models["jungfrau"]

        # residulas
        ray_resid = (Rayonix.all_data - ray_model)
        jun_resid = (Jungfrau.all_data - jun_model)

        # stored simple weights as 1 over variances
        rayW = Rayonix.simple_weights*self.rayonix_factor
        junW = Jungfrau.simple_weights*self.jungfrau_factor
        ########################################
        trusted = Jungfrau.all_trusted
        f = (jun_resid**2 *junW)[trusted].sum()
        grad_term = (2 * jun_resid * junW)[trusted]
        Jac = jun_Jac[:,trusted] #np.hstack((ray_Jac, jun_Jac))[:, trusted]
        g = np.array([np.sum(grad_term * Jac[param_idx]) for param_idx in range(Jac.shape[0])])
        # if fixlist is not None:
        #    for param_idx in range(Jac.shape[0]):
        #        if fixlist[param_idx]:
        #            g[param_idx] = 0
        gnorm = np.linalg.norm(g)
        if verbose: print("F=%10.7g, |G|=%10.7g" % (f, gnorm))
        return f, g
        ########################################
        #concatenate the two detectors data, important to maintain order, first Ray then Jun, or vice versa, but never mix!
        resid = np.hstack((ray_resid, jun_resid))
        W = np.hstack((rayW, junW))
        trusted = np.hstack((Rayonix.all_trusted, Jungfrau.all_trusted))

        if outlier_Z is not None:
            # assumes rayonix data is stacked first!
            det_id = np.hstack(([0]*len(ray_resid),[1]*len(jun_resid)))
            roi_id = np.hstack((Rayonix.roi_id, Jungfrau.roi_id))

        # compute functional and gradients
        weighted_resid = (resid**2*W)
        if outlier_Z is not None:
            # assumes rayonix data is stacked first!
            #TODO only do this once
            #if self.flagged_outliers is None:
            self.roi_contributions = []
            for i_roi in Rayonix.u_id:
                is_roi_pix = np.logical_and(det_id==0, roi_id==i_roi)
                is_roi_pix = np.logical_and(trusted, is_roi_pix)
                contribution_from_roi = weighted_resid[is_roi_pix].sum()
                #print("\tRayonix roi %3d: resid=%10.7f" %(i_roi, contribution_from_roi))
                self.roi_contributions.append((contribution_from_roi,0, i_roi))
            for i_roi in Jungfrau.u_id:
                is_roi_pix = np.logical_and(det_id==1, roi_id==i_roi)
                is_roi_pix = np.logical_and(trusted, is_roi_pix)
                contribution_from_roi = weighted_resid[is_roi_pix].sum()
                #print("\tJungfrau roi %3d: resid=%10.7f" %(i_roi, contribution_from_roi))
                self.roi_contributions.append((contribution_from_roi, 1, i_roi))

                vals,_,_ = map(np.array,zip(*self.roi_contributions))

            if self.flagged_rois is None:
                self.flagged_rois = utils.is_outlier(vals, outlier_Z)

            for ii, (con, i_det, i_roi) in enumerate(self.roi_contributions):
                s =""
                if self.flagged_rois[ii]:
                    is_roi_pix = np.logical_and(det_id == i_det, roi_id == i_roi)
                    #if i_det==1 and self.jungfrau_factor == 1:  # only flag jungfrau pixels as outliers if jone is not scaling the jungfrau contribution
                    if i_det==0:
                        trusted[is_roi_pix] = False
                    #if i_det==1 and self.jungfrau_factor!=1:

                #    s = "(OUT)"
                #if i_det == 0:
                #    print("\tRayonix roi %3d: resid=%10.7f %s" %(i_roi, con,s))
                #else:
                #    print("\tJungfrau roi %3d: resid=%10.7f %s" %(i_roi, con,s))

        f = weighted_resid[trusted].sum()
        grad_term = (2*resid*W)[trusted]
        Jac = np.hstack((ray_Jac, jun_Jac))[:, trusted]
        g = np.array([np.sum(grad_term*Jac[param_idx]) for param_idx in range(Jac.shape[0])])
        #if fixlist is not None:
        #    for param_idx in range(Jac.shape[0]):
        #        if fixlist[param_idx]:
        #            g[param_idx] = 0
        gnorm = np.linalg.norm(g)
        if verbose: print("F=%10.7g, |G|=%10.7g" % (f, gnorm))

        return f, g


def ls49_model(x, Rayonix,Jungfrau,  verbose):
    #jun_x = x.copy()[:-1]
    #ray_x = x.copy()
    #copy rayonix scale factor parameter to proper position
    #ray_x[0] = ray_x[-1]
    #ray_x = ray_x[:-1]

    #i_det is 0 for rayonix, and 1 for jungfrau
    rayonix_model_pix, ray_J = model(x, Rayonix.SIM,Rayonix.pan_fast_slow, verbose=verbose,compute_grad=True)#, i_det=0)
    rayonix_model_pix += Rayonix.all_background

    jungfrau_model_pix, jun_J = model(x, Jungfrau.SIM, Jungfrau.pan_fast_slow, verbose=verbose,compute_grad=True)#, i_det=1)
    jungfrau_model_pix += Jungfrau.all_background

    return {"rayonix": [rayonix_model_pix, ray_J], "jungfrau": [jungfrau_model_pix, jun_J]}


def model2(x, SIM, pfs, verbose=True, compute_grad=True, i_det=0, n_jungfrau_panel_groups=1):
    #if compute_grad and SIM.num_xtals > 1:
    #    raise NotImplemented("Grads not implemented for multiple xtals")
    num_per_xtal_params = SIM.num_xtals * (7)
    n_ucell_param = len(SIM.ucell_man.variables)
    ndet_param = 4*n_jungfrau_panel_groups + 2  # XYZ for Jungfrau panelsand XY for Rayonix
    if n_ucell_param + num_per_xtal_params + ndet_param != len(x):
        raise ValueError("weird x")
    params_per_xtal = np.array_split(x[:num_per_xtal_params], SIM.num_xtals)
    unitcell_var_reparam = x[num_per_xtal_params: num_per_xtal_params+ n_ucell_param]
    detdist_var_reparam = x[num_per_xtal_params+n_ucell_param:]

    if i_det==0:  # Rayonix
        x_detx = detdist_var_reparam[0]
        x_dety = detdist_var_reparam[1]
        shiftX, shiftY = SIM.DetXYZ_params[0].get_val(x_detx) , SIM.DetXYZ_params[1].get_val(x_dety)
        shiftZ = 0
        shiftO = 0
    if i_det==1:  # Jungfrau  TODO: expand for multiple panel groups
        x_detx = detdist_var_reparam[2]
        x_dety = detdist_var_reparam[3]
        x_detz = detdist_var_reparam[4]
        x_deto = detdist_var_reparam[5]
        shiftX, shiftY, shiftZ = \
            SIM.DetXYZ_params[0].get_val(x_detx), SIM.DetXYZ_params[1].get_val(x_dety), SIM.DetXYZ_params[2].get_val(x_detz)
        shiftO = SIM.DetOFS_params[0].get_val(x_deto)

    npanels = len(SIM.detector)
    for pid in range(npanels):
        #TODO panel Rotation about O
        panel_rot_angF = panel_rot_angS = 0

        SIM.D.reference_origin = SIM.detector[pid].get_origin()

        SIM.D.update_dxtbx_geoms(SIM.detector, SIM.beam.nanoBragg_constructor_beam, pid,
                                 shiftO, panel_rot_angF, panel_rot_angS,
                                 shiftX, shiftY,shiftZ,force=False)

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
    J = np.zeros((nparam, npix))  # note: order is: scale, rotX, rotY, rotZ, Na, Nb, Nc, ... (for each xtal), then ucell0, ucell1 , ucell2, ucell3, xcam0, ycam0, xcam1, ycam1 zcam1
    model_pix = None
    for i_xtal in range(SIM.num_xtals):
        SIM.D.raw_pixels_roi *= 0
        scale_reparam, rotX_reparam, rotY_reparam, rotZ_reparam, \
        Na_reparam, Nb_reparam, Nc_reparam = params_per_xtal[i_xtal]

        rotX = SIM.RotXYZ_params[i_xtal * 3].get_val(rotX_reparam)
        rotY = SIM.RotXYZ_params[i_xtal * 3 + 1].get_val(rotY_reparam)
        rotZ = SIM.RotXYZ_params[i_xtal * 3 + 2].get_val(rotZ_reparam)

        ## update parameters:
        #TODO test other method for setting matrix
        RX = xax.axis_and_angle_as_r3_rotation_matrix(rotX, deg=False)
        RY = yax.axis_and_angle_as_r3_rotation_matrix(rotY, deg=False)
        RZ = zax.axis_and_angle_as_r3_rotation_matrix(rotZ, deg=False)
        M = RX * RY * RZ
        SIM.D.Umatrix = M * sqr(SIM.crystal.dxtbx_crystal.get_U())

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

        if model_pix is None:
            model_pix = scale*SIM.D.raw_pixels_roi.as_numpy_array()[:npix]
        else:
            model_pix += scale*SIM.D.raw_pixels_roi.as_numpy_array()[:npix]

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

            Na_grad, Nb_grad, Nc_grad = [scale*d.as_numpy_array()[:npix] for d in SIM.D.get_ncells_derivative_pixels()]
            Na_grad = SIM.Nabc_params[i_xtal * 3].get_deriv(Na_reparam, Na_grad)
            Nb_grad = SIM.Nabc_params[i_xtal * 3 + 1].get_deriv(Nb_reparam, Nb_grad)
            Nc_grad = SIM.Nabc_params[i_xtal * 3 + 2].get_deriv(Nc_reparam, Nc_grad)
            J[7*i_xtal + 4] += Na_grad
            J[7*i_xtal + 5] += Nb_grad
            J[7*i_xtal + 6] += Nc_grad

            # note important to keep gradients in same order as the parameters x
            for i_ucell in range(n_ucell_param):
                d = scale*SIM.D.get_derivative_pixels(UCELL_ID_OFFSET+i_ucell).as_numpy_array()[:npix]
                d = SIM.ucell_params[i_ucell].get_deriv(unitcell_var_reparam[i_ucell], d)
                J[7*SIM.num_xtals + i_ucell] += d


            dx = scale*SIM.D.get_derivative_pixels(PANX_ID).as_numpy_array()[:npix]
            dx = SIM.DetXYZ_params[0].get_deriv(x_detx, dx)

            dy = scale*SIM.D.get_derivative_pixels(PANY_ID).as_numpy_array()[:npix]
            dy = SIM.DetXYZ_params[0].get_deriv(x_dety, dy)

            if i_det==0:  # Rayonix
                J[7 * SIM.num_xtals + n_ucell_param] += dx
                J[7 * SIM.num_xtals + n_ucell_param + 1] += dy
            else:  # Jungfrau
                J[7*SIM.num_xtals + n_ucell_param + 2] += dx
                J[7*SIM.num_xtals + n_ucell_param+ 3] += dy
                dz = scale * SIM.D.get_derivative_pixels(PANZ_ID).as_numpy_array()[:npix]
                dz = SIM.DetXYZ_params[0].get_deriv(x_detz, dz)
                J[7 * SIM.num_xtals + n_ucell_param + 4] += dz

                do = scale * SIM.D.get_derivative_pixels(PANO_ID).as_numpy_array()[:npix]
                do = SIM.DetOFS_params[0].get_deriv(x_deto, do)
                J[7 * SIM.num_xtals + n_ucell_param + 5] += do

    if verbose: print("\tunitcell= %3.5f %3.5f %3.5f %3.5f %3.5f %3.5f" % SIM.ucell_man.unit_cell_parameters)
    if verbose: print("\tdetdist shift= %3.5g %3.5g %3.5g (mm)" % (shiftX*1000, shiftY*1000, shiftZ*1000))
    if verbose: print("\tdetdist shift= %3.5g deg" % (shiftO*180/np.pi))
    return model_pix, J


if __name__ == '__main__':
    from dials.util import show_mail_on_error

    with show_mail_on_error():
        script = Script()
        script.run()
