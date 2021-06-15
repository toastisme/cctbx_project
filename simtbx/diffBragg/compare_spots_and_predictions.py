from dials.array_family import flex
import os
from libtbx.mpi4py import MPI
COMM = MPI.COMM_WORLD
from scipy.spatial import distance
from itertools import groupby
from simtbx.diffBragg import utils
from simtbx.command_line.hopper import model
import numpy as np
from dxtbx.model.experiment_list import ExperimentListFactory
import glob
import sys
import pandas


def get_pfs(E, rois, pids):
    """
    :param E: experimen -lrt *
    :param rois: list of rois, an roi is (x1,x2,y1,y2) where x is fast-scan, y is slow-scan
    :param pids:  list of panel ids, same length as rois
    :return: pfs object for diffBragg, and roi_id . pfs is 3 times the length of roi_id
    """
    npan = len(E.detector)
    nfast, nslow = E.detector[0].get_image_size()

    MASK = np.zeros((npan, nslow, nfast), bool)
    ROI_ID = np.zeros((npan, nslow, nfast), 'uint16')
    nspots = len(rois)

    for i_spot in range(nspots):
        x1, x2, y1, y2 = rois[i_spot]
        pid = pids[i_spot]
        MASK[pid, y1:y2, x1:x2] = True
        ROI_ID[pid, y1:y2, x1:x2] = i_spot
    p, s, f = np.where(MASK)
    roi_id = ROI_ID[p, s, f]
    pan_fast_slow = np.ascontiguousarray((np.vstack([p, f, s]).T).ravel())
    pan_fast_slow = flex.size_t(pan_fast_slow)
    return pan_fast_slow, roi_id

from dials.command_line.find_spots import working_phil
params = working_phil.extract()

params.spotfinder.threshold.dispersion.gain=9.481
params.spotfinder.threshold.dispersion.sigma_background=1
params.spotfinder.threshold.dispersion.sigma_strong=1.2
params.spotfinder.threshold.dispersion.kernel_size=[3,3]
params.spotfinder.threshold.dispersion.global_threshold=0
params.spotfinder.filter.min_spot_size=4
params.spotfinder.lookup.mask="/global/cfs/cdirs/m3562/der/master_files/newmask_withbad.pkl"

TAG = "expanded"
MIN_SIG=1e-3
use_mtz = True
CUDA = os.environ.get("DIFFBRAGG_USE_CUDA") is not None
if CUDA:
    print("USINGCUDA")
MTZ_FILE=None
MTZ_COL=None
FHKL = None
if use_mtz:
  MTZ_FILE ="5wp2_noAnom.mtz"
  MTZ_COL = "F(+),F(-)"
  from iotbx.reflection_file_reader import any_reflection_file
  F = any_reflection_file(MTZ_FILE).as_miller_arrays()[0]
  FHKL = set(list(F.indices()))

#BAD_FHKL = {}# set(map(tuple, np.loadtxt("hkls", int)))
BAD_FHKL = set(map(tuple, np.loadtxt("hkls_2k", int)))

deltaQ = 0.015
LOST_SCALE = 15  # TODO eliminate this garbage
cutoff_dist = 2
hnorm_cutoff=1/3.
quiet = True

pkls = glob.glob(sys.argv[1])
nold = 0
n = 0
for pkl in pkls:
    df = None
    if COMM.rank==0:
        df = pandas.read_pickle(pkl)
    df = COMM.bcast(df)
    for i_exp_name, exp_name in enumerate(df.opt_exp_name.values):
        if i_exp_name % COMM.size != COMM.rank:
            continue
        df_exp = df.query("opt_exp_name=='%s'" % exp_name)
        exp_name = df_exp.opt_exp_name.values[0]
        print("Loading expt")
        El = ExperimentListFactory.from_json_file(exp_name, True)
        print("Done Loading expt")
        reflections = flex.reflection_table.from_observations(El, params)
        refl_name = exp_name.replace(".expt", "_strong.refl")
        reflections.as_file(refl_name)

        _=utils.refls_to_q(reflections, El[0].detector, El[0].beam,update_table=True)

        print("Wrote %d refls for exper pair\n%s %s" %(len(reflections), refl_name, exp_name))

        rois, _ = utils.get_roi_deltaQ(reflections, deltaQ, El[0])
        pids = reflections['panel']

        pids_rois = sorted(list(zip(pids, rois)), key=lambda x: x[0])
        gb = groupby(pids_rois, key=lambda x: x[0])
        rois_per_panel = {pid:[x[1] for x in list(v)] for pid,v in gb}
        #pfs, roi_id = get_pfs(El[0], rois, pids)
        print("Modeling!")
        model = utils.roi_spots_from_pandas(df_exp, rois_per_panel, quiet=quiet,
                                            mtz_file=MTZ_FILE, mtz_col=MTZ_COL, cuda=CUDA,
                                            reset_Bmatrix=True,
                                            force_no_detector_thickness=True,
                                            norm_by_nsource=True)
        print("Done modeling")
        dists = []
        sel = []
        print("Finding Xcalcs")
        xycalcs = flex.vec3_double()
        for i_ref, (pid, roi) in enumerate(zip(pids, rois)):
            x1,x2,y1,y2 = roi
            img = model[pid][y1:y2, x1:x2]
            img *= LOST_SCALE
            if np.allclose(img,0):
                sel.append(False)
                xycalcs.append((0,0,0))
                dists.append(np.nan)
                continue
            #elif np.all(img < MIN_SIG):
            #    sel.append(False)
            #    xycalcs.append((0,0,0))
            #    dists.append(np.nan)
            #    continue
            else:
                Y,X = np.indices(img.shape)
                Isum = img.sum()
                xcal = (X*img).sum() / Isum + x1+0.5
                ycal = (Y*img).sum() / Isum + y1+0.5
                xobs, yobs, _ = reflections[i_ref]["xyzobs.px.value"]
                xycalcs.append((xcal, ycal, 0))
                dist = distance.euclidean((xcal, ycal),(xobs,yobs))
                dists.append(dist)
                if dist >cutoff_dist:
                    sel.append(False)
                    continue
                sel.append(True)
        print("Done Finding Xcalcs")
        reflections["xyzcal.px"] = xycalcs
        reflections["dists"] = flex.double(dists)
        indexed_refls = reflections.select(flex.bool(sel))
        hkl, hkl_i = utils.refls_to_hkl(indexed_refls, El[0].detector, El[0].beam, El[0].crystal,
                               update_table=True)
        #hkl_norm = np.linalg.norm(hkl-hkl_i, axis=1)
        #sel_hkl = flex.bool([hi in FHKL for hi in hkl_i])
        hkl_i_asu = utils.map_hkl_list(hkl_i, True, "P6522")
        sel_hkl = flex.bool([tuple(hi) not in BAD_FHKL for hi in hkl_i_asu])
        indexed_refls = indexed_refls.select(sel_hkl)
        #indexed_refls = indexed_refls.select(flex.bool(hkl_norm < hnorm_cutoff))

        nidx = len(indexed_refls)
        md = np.median(indexed_refls["dists"])
        out = "RANK %3d:  shot %d / %d\n------------------\nThe model predicted %d  / %d strong spots with median prediction offset of %f" \
              %(COMM.rank, i_exp_name+1, len(df), nidx, len(rois), md)
        groupA = df_exp.pred_offsets.values[0]
        md_groupA = np.median(groupA)
        out += "\n\tAs comparison, groupA was %d spots, with median prediction offset of %f" %(len(groupA), md_groupA)
        out += "\n\tOriginal exper: %s" % df_exp.exp_name.values[0]

        #old_refl_name = df_exp.exp_name.values[0].replace(".expt",".refl")
        old_refl_name = df_exp.refl_names.values[0]
        oldR= flex.reflection_table.from_file(old_refl_name)
        nold+= len(oldR)
        n += nidx

        #idx_refl_name = exp_name.replace(".expt", "_indexed1.refl")
        idx_refl_name = exp_name.replace(".expt", "_%s.refl" % TAG)
        indexed_refls.as_file(idx_refl_name)
        out += "\n\tNew indexed refls file: %s" % idx_refl_name
        print(out)
        #
        #bragg_pix, _ = model(x, SIM, pfs, verbose=True, compute_grad=False):


nold = COMM.reduce(nold)
n = COMM.reduce(n)
if COMM.rank==0:
    if len(pkls)==1:
        ofname = pkls[0].replace(".pkl", ".txt")
        o = open(ofname, "w")
        for e, eopt, s in zip(df.exp_name, df.opt_exp_name, df.spectrum_filename):
            r = eopt.replace(".expt", "_%s.refl" %TAG)
            o.write("%s %s %s\n" % (e, r, s))
        o.close()
        print("Saved exper ref file %s" % ofname)
    print("Nold=%d; Nnew=%d" % (nold, n) )
    print("Done.")
