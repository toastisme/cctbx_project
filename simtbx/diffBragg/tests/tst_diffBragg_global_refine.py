from __future__ import division
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--plot", action='store_true')
parser.add_argument("--cuda", action='store_true')
parser.add_argument("--detdist", action='store_true', help='perturb then refine the detdist')
parser.add_argument("--ncells", action='store_true', help='perturb then refine the ncells')
parser.add_argument("--bg", action='store_true', help='refine bg planes... ')
parser.add_argument("--spotscale", action='store_true')
parser.add_argument("--bmatrix", action='store_true')
parser.add_argument("--umatrix", action='store_true')
parser.add_argument("--fcell", action='store_true')
parser.add_argument("--nshots", default=1, type=int)
parser.add_argument("--curvatures", action='store_true')
parser.add_argument("--psf", action='store_true')
parser.add_argument("--gain", action='store_true')
parser.add_argument("--iterfreeze", action="store_true")
parser.add_argument("--rescale", action="store_true")
parser.add_argument("--onlyindexed", action="store_true")
parser.add_argument("--testbg", action="store_true")
parser.add_argument("--testfcell", action="store_true")
parser.add_argument("--testUmatrix", action="store_true")
parser.add_argument("--bgoffsetonly", action="store_true")
parser.add_argument("--bgoffsetpositive", action="store_true")
parser.add_argument("--shufflebg", action="store_true")
parser.add_argument("--predictwithtruth", action="store_true")
parser.add_argument("--pershotucell", action="store_true")
parser.add_argument("--pershotncells", action="store_true")
parser.add_argument("--maxcalls", type=int, default=None, help="maximum number of iterations for refinement")
parser.add_argument("--displayedimage", type=int, default=0, help="if plotting during refinement, this is specifies which image to plot")
parser.add_argument("--perturbfcell", type=float, default=None, help="perturbation factor, 0.1 is small, 1 is large")
parser.add_argument("--fractionperturbed", type=float, default=0.1, help="Fraction of Fhkl to perturn")
parser.add_argument("--fcellsigmascale", type=float, default=None)
parser.add_argument("--outdir", type=str, default=None, help="output directory for refinement data")
parser.add_argument("--spectra", action="store_true", help="test with a refinable spectrum")
parser.add_argument("--perturbSpectra", action="store_true", help="perturb the spectrum")
parser.add_argument("--refineSpectra", action="store_true", help="refine the spectrum")
parser.add_argument("--testSpectra", action="store_true", help="test average energy after spectrum refinement")
parser.add_argument("--sz", type=float, help="shoebox size", default=15)
args = parser.parse_args()


from dxtbx.model.crystal import Crystal
from copy import deepcopy

from dxtbx.model import Panel
from cctbx import uctbx
from scitbx.matrix import rec, col
import numpy as np
from scipy.spatial.transform import Rotation
from scitbx.matrix import sqr

from simtbx.nanoBragg import shapetype

from simtbx.nanoBragg.nanoBragg_crystal import NBcrystal
from simtbx.nanoBragg.sim_data import SimData
from simtbx.diffBragg import utils
from simtbx.diffBragg.refiners.local_refiner import LocalRefiner
from simtbx.diffBragg.refiners.crystal_systems import TetragonalManager

# containers for GlobalRefine
shot_ucell_managers={}
shot_rois={}
shot_nanoBragg_rois={}
shot_roi_imgs={}
shot_spectra={}
shot_crystal_GTs={}
shot_crystal_models={}
shot_xrel={}
shot_yrel={}
shot_abc_inits={}
shot_asu={}
sel_flags ={}
shot_hkl={}
shot_panel_ids={}
nspot_per_shot = {}
shot_originZ_init = {}

# GLOBAL PARAMETERS

all_c_before = []
ucell = (55, 65, 75, 90, 95, 90)
ucell2 = (55, 65, 75, 90, 95, 90)
if args.bmatrix:
    ucell2 = (55.1, 65.2, 74.9, 90, 94.9, 90)
symbol = "P121"

ucell = (79, 79, 38, 90,90,90)
ucell2 = (79, 79, 38, 90,90, 90)
if args.bmatrix:
    ucell2 = (79.1, 79.1, 38.2, 90, 90, 90)
symbol = "P43212"

from simtbx.diffBragg.utils import  fcalc_from_pdb
miller_array_GT = fcalc_from_pdb(resolution=2, wavelength=1, algorithm='fft', ucell=ucell, symbol=symbol)
Ncells_gt = 12, 12, 12

N_SHOTS = args.nshots

np.random.seed(3142019)
detdists_gt = np.random.normal(150,0.1, N_SHOTS)
offsets = np.random.uniform(1, 3, N_SHOTS) * np.random.choice([1,-1], N_SHOTS)
originZ_gt = {}
all_dets = []
all_reso = []
waves = waves_perturbed = fluxes = None
for i_shot in range(N_SHOTS):

    # FIRST WE GENERATE SOME RANDOM IMAGES

    # generate a random raotation
    rotation = Rotation.random(num=1, random_state=100 + i_shot)[0]
    Q = rec(rotation.as_quat(), n=(4, 1))
    rot_ang, rot_axis = Q.unit_quaternion_as_axis_and_angle()

    # generate a small perturbation rotation
    perturb_rot_axis = np.random.random(3)
    perturb_rot_axis /= np.linalg.norm(perturb_rot_axis)
    perturb_rot_ang = 0
    if args.umatrix:
        perturb_rot_ang = np.random.choice([0.02, 0.03, 0.04, .05])  # degrees

    # make the ground truth crystal:
    if args.pershotucell:
        np.random.seed(i_shot)
        uc_a = np.random.uniform(78.5, 79.5)
        uc_c = np.random.uniform(37.5, 38.5)
        ucell_shot = uc_a, uc_a, uc_c, 90,90,90
        print("Shot %d cell:" % (i_shot+1), ucell_shot)
        a_real, b_real, c_real = sqr(uctbx.unit_cell(ucell_shot).orthogonalization_matrix()).transpose().as_list_of_lists()
    else:
        a_real, b_real, c_real = sqr(uctbx.unit_cell(ucell).orthogonalization_matrix()).transpose().as_list_of_lists()
    x = col((-1, 0, 0))
    y = col((0, -1, 0))
    z = col((0, 0, -1))
    rx, ry, rz = np.random.uniform(-180, 180, 3)
    RX = x.axis_and_angle_as_r3_rotation_matrix(rx, deg=True)
    RY = y.axis_and_angle_as_r3_rotation_matrix(ry, deg=True)
    RZ = z.axis_and_angle_as_r3_rotation_matrix(rz, deg=True)
    M = RX * RY * RZ
    a_real = M * col(a_real)
    b_real = M * col(b_real)
    c_real = M * col(c_real)
    C = Crystal(a_real, b_real, c_real, symbol)
    C.rotate_around_origin(rot_axis, rot_ang)

    # make the perturbed crystal model
    a2_real, b2_real, c2_real = sqr(uctbx.unit_cell(ucell2).orthogonalization_matrix()).transpose().as_list_of_lists()
    a2_real = M * col(a2_real)
    b2_real = M * col(b2_real)
    c2_real = M * col(c2_real)
    C2 = Crystal(a2_real, b2_real, c2_real, symbol)
    C2.rotate_around_origin(rot_axis, rot_ang)
    assert np.allclose(C2.get_U(), C.get_U())
    C2.rotate_around_origin(col(perturb_rot_axis), perturb_rot_ang)

    # Setup the simulation and create a realistic image
    # with background and noise
    # <><><><><><><><><><><><><><><><><><><><><><><><><>
    nbcryst = NBcrystal()
    nbcryst.dxtbx_crystal = C   # simulate ground truth
    nbcryst.thick_mm = 0.1
    nbcryst.Ncells_abc = Ncells_gt  # ground truth Ncells

    nbcryst.miller_array = miller_array_GT
    print("Ground truth ncells = %f" % (nbcryst.Ncells_abc[0]))

    # ground truth detector
    DET_gt = SimData.simple_detector(detdists_gt[i_shot], 0.177, (600, 600))
    originZ_gt[i_shot] = DET_gt[0].get_origin()[2]

    # initialize the simulator
    SIM = SimData()
    if args.spectra:

        if waves is None:
            from simtbx.nanoBragg.utils import ENERGY_CONV
            from scipy.signal import windows
            spec = SIM.beam.spectrum
            total_flux = spec[0][1]
            wave = spec[0][0]

            en = ENERGY_CONV / wave
            delta_en = 1.5
            ens = np.arange(en - 5, en + 6, delta_en)
            waves = ENERGY_CONV / ens
            num_energies = len(ens)
            fluxes = np.ones(num_energies) * total_flux / num_energies
            fluxes = fluxes*windows.hanning(num_energies)
            fluxes /= fluxes.sum()
            fluxes *= total_flux


            spectrum_GT = list(zip(waves, fluxes))
            gt_lambda0 = waves[0]
            gt_lambda1 = waves[1] - waves[0]
            spec_idx = np.arange(num_energies)
            assert np.allclose(waves, gt_lambda0 + spec_idx*gt_lambda1)
            lam0, lam1 = gt_lambda0, gt_lambda1

        SIM.beam.spectrum = list(zip(waves, fluxes))

    SIM.detector = DET_gt
    all_dets.append(DET_gt)

    # TODO get the detector model
    node = SIM.detector[0]
    node_d = node.to_dict()
    Origin = node_d["origin"][0], node_d["origin"][1], node_d["origin"][2]
    distance = Origin[2]
    print("Ground truth originZ=%f" % (SIM.detector[0].get_origin()[2]))

    # TODO perturb the detector model
    # copy the detector and update the origin
    det2 = deepcopy(SIM.detector)
    # alter the detector distance by 2 mm
    #detz_offset = np.random.normal(0, 0.1)  # TODO make me random per shot (like GDVN would have)
    detz_offset = offsets[i_shot]
    node_d["origin"] = Origin[0], Origin[1], Origin[2] + detz_offset
    det2[0] = Panel.from_dict(node_d)

    shot_originZ_init[i_shot] = Origin[2]
    if args.detdist:
        shot_originZ_init[i_shot] = Origin[2]+detz_offset

    SIM.crystal = nbcryst
    SIM.instantiate_diffBragg(oversample=0, auto_set_spotscale=True)
    SIM.D.nopolar = False
    SIM.D.default_F = 0
    SIM.D.progress_meter =  False
    #SIM.water_path_mm = 0.005
    SIM.water_path_mm = 0.15
    SIM.air_path_mm = 0.1
    SIM.add_air = True
    SIM.add_water = True
    SIM.include_noise = True
    SIM.D.add_diffBragg_spots()
    SIM.D.F000 = 0
    SPOTS = SIM.D.raw_pixels.as_numpy_array()
    SIM.D.readout_noise_adu = 1
    SIM._add_background()
    BACKGROUND_IMAGE = SIM.D.raw_pixels.as_numpy_array() - SPOTS
    SIM._add_noise()

    if args.psf:
        img_pre_psf = SIM.D.raw_pixels.as_numpy_array()
        v = SIM.D.verbose
        SIM.D.verbose = 8
        SIM.D.detector_psf_kernel_radius_pixels = 0
        fwhm = 76 / 56
        radius = 3
        SIM.D.apply_psf(shapetype.Fiber, fwhm, radius)
        SIM.D.verbose = v

    print("Using oversample %d" % SIM.D.oversample)

    # This is the ground truth image:
    img = SIM.D.raw_pixels.as_numpy_array()
    SIM.D.raw_pixels *= 0

    if args.psf:
        y = slice(450, 480,1)
        x = slice(650, 670, 1)
        print ("PSF max discrepancy: %f" % abs(img_pre_psf[y,x]- img[y,x]).max())

    # Simulate the perturbed image for comparison
    # perturbed detector:
    if args.detdist:
        SIM.detector = det2
        SIM.D.update_dxtbx_geoms(det2, SIM.beam.nanoBragg_constructor_beam, 0)
        print("Modified originZ=%f" % (det2[0].get_origin()[2]))
    # perturbed crystal:
    SIM.D.Bmatrix = C2.get_B()
    SIM.D.Umatrix = C2.get_U()
    nbcryst.dxtbx_crystal = C2

    if args.ncells:
        Ncells_abc2 = 14, 14, 14
        nbcryst.Ncells_abc = Ncells_abc2
        SIM.D.set_value(9, Ncells_abc2[0])
        print("Modified Ncells=%f" % Ncells_abc2[0])
    else:
        Ncells_abc2 = 12, 12, 12

    SIM.crystal = nbcryst
    # perturbed Ncells
    SIM.D.raw_pixels*=0
    SIM.D.add_diffBragg_spots()
    SPOTS2 = SIM.D.raw_pixels.as_numpy_array()

    SIM.D.raw_pixels *= 0

    if args.predictwithtruth:
        _spots = SPOTS
    else:
        _spots = SPOTS2

        spot_refls = utils.refls_from_sims(np.array([_spots]), SIM.detector, SIM.beam.nanoBragg_constructor_beam,
                                           thresh=20)
        out = utils.get_roi_background_and_selection_flags(spot_refls, np.array([img]), shoebox_sz=args.sz,
                                                           reject_edge_reflections=True, use_robust_estimation=True)
        spot_roi, panel_ids, tilt_abc, selection_flags, background = out

    if args.shufflebg:
        tilt_abc[:, 2] = np.random.permutation(tilt_abc[:, 2])

    #UcellMan = MonoclinicManager(
    #    a=ucell2[0],
    #    b=ucell2[1],
    #    c=ucell2[2],
    #    beta=ucell2[4]*np.pi/180.)

    UcellMan = TetragonalManager(
        a=ucell2[0],
        c=ucell2[2])

    if args.gain:
        img = img*1.1

    # TODO: the following need to be added to the refiner init function..
    nspot = len(spot_roi)
    nspot_per_shot[i_shot] = nspot

    nanoBragg_rois = []  # special nanoBragg format
    xrel, yrel, roi_imgs = [], [], []
    xcom, ycom = [],[]
    for i_roi, (x1, x2, y1, y2) in enumerate(spot_roi):
        nanoBragg_rois.append(((int(x1), int(x2)), (int(y1), int(y2))))
        yr, xr = np.indices((y2 - y1, x2 - x1))
        xrel.append(xr)
        yrel.append(yr)
        roi_imgs.append(img[y1:y2, x1:x2])
        xcom.append(.5*(x1 + x2))
        ycom.append(.5*(y1 + y2))

    q_spot = utils.x_y_to_q(xcom, ycom, SIM.detector, SIM.beam.nanoBragg_constructor_beam)
    reso = 1/np.linalg.norm(q_spot, axis=1)
    all_reso += list(reso)
    Ai = sqr(SIM.crystal.dxtbx_crystal.get_A()).inverse()
    Ai = Ai.as_numpy_array()
    HKL = np.dot(Ai, q_spot.T)
    HKLi = [np.ceil(h - 0.5).astype(int) for h in HKL]
    HKLi = [tuple(x) for x in np.vstack(HKLi).T]

    Hi_asu = utils.map_hkl_list(HKLi, anomalous_flag=True, symbol=symbol)

    shot_ucell_managers[i_shot] = UcellMan
    shot_rois[i_shot] = spot_roi
    shot_nanoBragg_rois[i_shot] = nanoBragg_rois
    shot_roi_imgs[i_shot] = roi_imgs
    shot_spectra[i_shot] = SIM.beam.spectrum
    if args.perturbSpectra:
        assert args.spectra
        if waves_perturbed is None:
            np.random.seed(3142019)
            lam0 = np.random.normal(gt_lambda0, gt_lambda0 * 0.002)
            lam1 = np.random.normal(gt_lambda1, abs(gt_lambda1) * 0.002)
            waves_perturbed = lam0 + spec_idx * lam1
        print("ENERGY TRUTH=%.4f" % (ENERGY_CONV/gt_lambda0))
        print("ENERGY PERTURBED=%.4f" % (ENERGY_CONV/lam0))
        shot_spectra[i_shot] = list(zip(waves_perturbed, fluxes))
        SIM.beam.spectrum = list(zip(waves_perturbed, fluxes))
    elif args.spectra:
        waves_perturbed = waves

    shot_crystal_GTs[i_shot]= C
    shot_crystal_models[i_shot]= SIM.crystal.dxtbx_crystal
    shot_xrel[i_shot]= xrel
    shot_yrel[i_shot]= yrel
    shot_abc_inits[i_shot]= tilt_abc
    shot_asu[i_shot]= Hi_asu  # TODO Im weird fix me
    shot_hkl[i_shot]= HKLi  # TODO Im weird fix me
    shot_panel_ids[i_shot]= [0]*nspot
    sel_flags[i_shot] = selection_flags

    if i_shot < N_SHOTS-1:
        SIM.D.free_all()  # CLEANGIUAGE

if args.detdist:
    SIM.D.oversample_omega = False  # necessary to refine detector distance

Hi_all_ranks, Hi_asu_all_ranks = [], []
for i in range(N_SHOTS):
    Hi_all_ranks += shot_hkl[i]
    Hi_asu_all_ranks += shot_asu[i]


print("Overall completeness\n<><><><><><><><>")
from cctbx.crystal import symmetry
from cctbx import miller
uc = shot_ucell_managers[0]
from cctbx.array_family import flex as cctbx_flex
params = uc.a, uc.b, uc.c, uc.al * 180 / np.pi, uc.be * 180 / np.pi, uc.ga * 180 / np.pi
symm = symmetry(unit_cell=params, space_group_symbol=symbol)
hi_flex_unique = cctbx_flex.miller_index(list(set(Hi_asu_all_ranks)))
mset = miller.set(symm, hi_flex_unique, anomalous_flag=True)
mset.setup_binner(d_min=2, d_max=999, n_bins=10)
mset.completeness(use_binning=True).show()
print("total miller vars=%d" % (len(set(Hi_asu_all_ranks))))

# this will map the measured miller indices to their index in the LBFGS parameter array self.x
idx_from_asu = {h: i for i, h in enumerate(set(Hi_asu_all_ranks))}
# we will need the inverse map during refinement to update the miller array in diffBragg, so we cache it here
asu_from_idx = {i: h for i, h in enumerate(set(Hi_asu_all_ranks))}

if args.perturbfcell is not None:
    pert = args.perturbfcell
    fobs = SIM.crystal.miller_array
    unique_Hi_asu_all_ranks = list(idx_from_asu.keys())
    h_vals = []
    for h in unique_Hi_asu_all_ranks:
        h_vals.append(fobs.value_at_index(tuple(h)))
    h_vals = np.array(h_vals)

    num_hkl = len(h_vals)
    num_perturb = int(args.fractionperturbed * num_hkl)

    indices_of_fhkl_chosen_for_perturbation = np.random.permutation(num_hkl)[:num_perturb]

    for i in indices_of_fhkl_chosen_for_perturbation:
        val = np.log(h_vals[i])
        new_val = np.random.uniform(val-pert, val+pert)
        h_vals[i] = np.exp(new_val)

    new_fobs = utils.update_miller_array_at_indices(fobs, indices=unique_Hi_asu_all_ranks, new_values=h_vals)
    SIM.crystal.miller_array = new_fobs
    SIM.update_Fhkl_tuple()

# always local parameters: rotations, spot scales, tilt coeffs
nrotation_param = 3*N_SHOTS
nscale_param = 1*N_SHOTS
ntilt_param = 0  # note: tilt means tilt plane
for i_shot in range(N_SHOTS):
    ntilt_param += 3 * nspot_per_shot[i_shot]

# unit cell parameters
nucell_param = len(shot_ucell_managers[0].variables)
n_pershot_ucell_param = 0
n_global_ucell_param = nucell_param
if args.pershotucell:
    n_pershot_ucell_param += nucell_param*N_SHOTS
    n_global_ucell_param = 0

# mosaic domain parameter m
n_ncell_param = 1
n_pershot_m_param = 0
n_global_m_param = n_ncell_param
if args.pershotncells:
    n_pershot_m_param = 1*N_SHOTS
    n_global_m_param = 0
n_ncells_def_param = 3

ndetz_param = len(detdists_gt)
n_eta_param = 3
n_perspot_scale_param = int(ntilt_param / 3.)
n_local_unknowns = n_ncells_def_param + nrotation_param + nscale_param \
                   + ntilt_param + ndetz_param + n_pershot_ucell_param \
                   + n_pershot_m_param + n_eta_param + n_perspot_scale_param

nfcell_param = len(idx_from_asu)
ngain_param = 1

n_spectra_param = 2
n_panelXYZ_params = 3
n_panelRot_params = 3

#n_global_unknowns = n_spectra_params + n_panelRot_params + n_panelXYZ_params
n_global_unknowns = nfcell_param + ngain_param + n_global_m_param + n_global_ucell_param + n_spectra_param + n_panelXYZ_params + n_panelRot_params
n_total_unknowns = n_local_unknowns + n_global_unknowns

RUC = LocalRefiner(
    n_total_params=n_total_unknowns,
    n_local_params=n_local_unknowns,
    local_idx_start=0,
    shot_ucell_managers=shot_ucell_managers,
    shot_rois=shot_roi_imgs,
    shot_nanoBragg_rois=shot_nanoBragg_rois,
    shot_roi_imgs=shot_roi_imgs,
    shot_spectra=shot_spectra,
    shot_crystal_GTs=shot_crystal_GTs,
    shot_crystal_models=shot_crystal_models,
    shot_xrel=shot_xrel,
    shot_yrel=shot_yrel,
    shot_abc_inits=shot_abc_inits,
    shot_asu=shot_asu,
    global_param_idx_start=n_local_unknowns,
    shot_panel_ids=shot_panel_ids,
    log_of_init_crystal_scales=None,
    all_crystal_scales=None,
    shot_detector_distance_init={i_shot:0 for i_shot in range(N_SHOTS)},
    global_ncells=not args.pershotncells,
    global_ucell=not args.pershotucell,
    sgsymbol=symbol,
    omega_kahn=None)


if args.refineSpectra:
    RUC.refine_spectra = True
    RUC.n_spectra_param = 2
    RUC.spectra_coefficients_sigma = .01, .01
    RUC.spectra_coefficients_init = 0, 1
    RUC.lambda_coef_ranges = [(-0.01, 0.01), (0.95, 1.05)]

RUC.output_dir = args.outdir
# dummie info (used only in real situations)
RUC.FNAMES = ["DUMMIE%d.imagefile" % i_shot for i_shot in range(N_SHOTS)]
RUC.PROC_FNAMES = ["DUMMIE.hdf5"]
RUC.BBOX_IDX = [range(len(shot_nanoBragg_rois[i_shot])) for i_shot in range(N_SHOTS)]
RUC.PROC_IDX = range(N_SHOTS)
# end dummie info

RUC.iteratively_freeze_parameters = args.iterfreeze
RUC.index_of_displayed_image = args.displayedimage
RUC.idx_from_asu = idx_from_asu
RUC.asu_from_idx = asu_from_idx
RUC.refine_background_planes = args.bg
RUC.refine_Umatrix = args.umatrix
RUC.refine_Bmatrix = args.bmatrix
RUC.refine_ncells = args.ncells
RUC.refine_crystal_scale = args.spotscale
RUC.refine_Fcell = args.fcell
RUC.refine_detdist = args.detdist
RUC.refine_gain_fac = args.gain
RUC.rescale_params = args.rescale
RUC.background_testing_mode = args.testbg
if args.maxcalls is not None:
    RUC.max_calls = args.maxcalls
else:
    RUC.max_calls = 1000
RUC.trad_conv_eps = 1e-7
RUC.trad_conv = True
RUC.trial_id = 0
RUC.selection_flags = sel_flags

RUC.plot_stride = 4
RUC.plot_spot_stride = 10
RUC.plot_residuals = False
RUC.plot_images = args.plot
RUC.setup_plots()

RUC.refine_rotZ = True
RUC.request_diag_once = False
RUC.S = SIM
if args.cuda:
    RUC.S.D.use_cuda=True
if not args.curvatures:
    RUC.S.D.compute_curvatures = False
RUC.has_pre_cached_roi_data = True
RUC.S.D.update_oversample_during_refinement = False
RUC.S.D.verbose = 2
RUC.use_curvatures = False
RUC.use_curvatures_threshold = 10
RUC.bg_offset_positive = args.bgoffsetpositive
RUC.bg_offset_only = args.bgoffsetonly
RUC.calc_curvatures = args.curvatures
RUC.poisson_only = False
RUC.verbose = True

RUC.big_dump = False
RUC.gt_ncells = Ncells_gt[0]
RUC.originZ_gt = originZ_gt
RUC.gt_ucell = ucell[0], ucell[2]
RUC.testing_mode = False

RUC.spot_scale_init = [1]*N_SHOTS
RUC.m_init = {i:Ncells_abc2[0] for i in range(N_SHOTS)}  # np.log(Ncells_abc2[0]-3)
RUC.ucell_inits = {i:shot_ucell_managers[i].variables for i in range(N_SHOTS)}

#RUC.S.D.update_oversample_during_refinement = False  # todo: decide
Fobs = RUC.S.crystal.miller_array_high_symmetry
RUC.Fref = miller_array_GT if RUC.refine_Fcell else None
#dmax, dmin = Fobs.d_max_min()
dmax, dmin = max(all_reso), min(all_reso)
RUC.binner_dmax = dmax + 1e-6
RUC.binner_dmin = dmin - 1e-6
RUC.binner_nbin = 10
RUC.scale_r1 = RUC.refine_Fcell #True
RUC.merge_stat_frequency = int(RUC.refine_Fcell)
RUC.print_resolution_bins = False
if args.fcellsigmascale is not None:
    RUC.fcell_sigma_scale = args.fcellsigmascale

RUC.run(setup_only=False)
if RUC.hit_break_to_use_curvatures:
    RUC.num_positive_curvatures = 0
    RUC.use_curvatures = True
    RUC.run(setup=False)

#RUC.calc_func = True
#RUC.compute_functional_and_graGdients()
#
#def func(x, RUC):
#    print("F: det dist %f" % RUC.x[-3])
#    RUC.calc_func = True
#    RUC.x = flex.double(x)
#    f, g = RUC.compute_functional_and_gradients()
#    return f
#
#
#def fprime(x, RUC):
#    print("G: det dist %f" % RUC.x[-3])
#    RUC.calc_func = False
#    RUC.x = flex.double(x)
#    RUC.x = flex.double(x)
#    f, g = RUC.compute_functional_and_gradients()
#    return 1*g.as_numpy_array()
#
#from scipy.optimize import fmin_l_bfgs_b
#
#bounds = [(-np.inf, np.inf)]*RUC.n
#bounds[-11] = -.1*np.pi/180, .1*np.pi/180  # rotX
#bounds[-10] = -.1*np.pi/180, .1*np.pi/180  # roty
#bounds[-9] = -.1*np.pi/180, .1*np.pi/180  # rotZ
#bounds[-8] = 50, 60  # a
#bounds[-7] = 60, 70  #  b
#bounds[-6] = 70, 80  # c
#bounds[-5] = 93*np.pi/180, 97*np.pi/180.  # beta
#bounds[-4] = 7, 30  # ncells
#bounds[-3] = -170, -150  # detdist
#bounds[-2] = 1, 1  # gain
#bounds[-1] = 1, 1  # scale
#
#print("GO!")
##out = fmin_l_bfgs_b(func=func, x0=np.array(RUC.x),
##                    fprime=fprime,args=[RUC]) #, bounds=bounds)
#out = fmin_l_bfgs_b(func=func, factr=1000, x0=np.array(RUC.x),
#                    fprime=fprime, maxls=100,
#                    pgtol=1e-7,
#                    args=(RUC,),
#                    bounds=bounds)

i_shot = 0
abc_init = RUC.ABC_INIT[i_shot]
n_spots = len(RUC.ABC_INIT[i_shot])
all_dev = []
all_dev_i = []
all_tilt = []
all_tilt_i = []
all_img = []
all_data =[]
for i_spot in range(n_spots):
    xr = RUC.XREL[i_shot][i_spot]
    yr = RUC.YREL[i_shot][i_spot]
    ai, bi, ci = RUC.ABC_INIT[i_shot][i_spot]
    a, b, c = RUC._get_bg_vals(i_shot, i_spot)
    (i1, i2), (j1, j2) = RUC.NANOBRAGG_ROIS[i_shot][i_spot]

    tilt_i = ai*xr + bi*yr + ci
    tilt_refined = a*xr + b*yr + c

    all_tilt.append(tilt_refined)
    all_tilt_i.append(tilt_i)

    real_bg = BACKGROUND_IMAGE[j1:j2, i1:i2]
    data = img[j1:j2, i1:i2]
    all_img.append(real_bg)
    all_data.append(data)
    dev_i = np.abs(tilt_i - real_bg).sum()
    dev = np.abs(tilt_refined - real_bg).sum()
    all_dev_i.append(dev_i)
    all_dev.append(dev)
all_dev = np.array(all_dev)
all_dev_i = np.array(all_dev_i)
print("Before reinfment: bg deviation mean=%.4f, med=%.4f, c std = %.4f" % (np.mean(all_dev_i), np.median(all_dev_i), np.std(all_dev_i)))
print("After reinfment:               mean=%.4f, med=%.4f, c std = %.4f" % (np.mean(all_dev), np.median(all_dev), np.std(all_dev)))
if args.testbg:
    assert np.mean(all_dev) < np.mean(all_dev_i)
if args.testfcell:
    print("Final Rfactor = %2.7g" % RUC.R_overall)
    assert RUC.R_overall < 0.006
if args.testUmatrix:
    print("Final Misorientation=%2.7g" % RUC.all_ang_off[0])
    assert RUC.all_ang_off[0] < 0.003

if args.testSpectra:
    coef = RUC._get_spectra_coefficients()
    waves_refined = coef[0] + coef[1]*waves_perturbed
    fluxsum = sum(fluxes)
    en_ref_com = ENERGY_CONV / (sum(fluxes * waves_refined) / fluxsum)
    en_com = ENERGY_CONV / (sum(fluxes * waves) / fluxsum)
    en_init_com = ENERGY_CONV / (sum(fluxes * waves_perturbed) / fluxsum)

    print("Before refinement: COM energy=%f" % en_init_com)
    print("AFTER refinement: COM energy=%f" % en_ref_com)
    print("Ground truth COM energy = %f" % en_com)
    assert abs(en_ref_com - en_com) < 1

if args.cuda:
    RUC.S.D.gpu_free()
print("OK!")
