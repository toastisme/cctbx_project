# LIBTBX_SET_DISPATCHER_NAME cctbx.xfel.center_scan

#!/usr/bin/env libtbx.python
from argparse import ArgumentParser
from joblib import Parallel, delayed
from copy import deepcopy
parser = ArgumentParser()
import os

parser.add_argument("--heat", action="store_true", help="display CC heat")
parser.add_argument("--xyz", nargs=3, type=float, help="initial xyz offset passed as e.g. -xyz 0.1 0.3 0")
parser.add_argument("--run", type=int, help="run number")
parser.add_argument("--trial", type=int, help="trial number (XFEL GUI)")
parser.add_argument("--group", type=int, help="run group number (XFEL GUI)")
parser.add_argument("--j", type=int,help="number of jobs (max is number of hardware cores on a node)")
parser.add_argument('--xscan', nargs=3, type=float, default=[-3,3, 0.5], help="scan range and resolution in X, in pixel units, passed as 3 args e.g. '--xscan -2 2 0.5' will scan for the X-center from -2 to 2 pixels at 0.5 pixel resolution ")
#parser.add_argument("--loadpath", type=str, help="file containing all.expt all.refl")
parser.add_argument('--yscan', nargs=3, type=float, default=[-3,3, 0.5], help="same as xscan, but for Y direction")
parser.add_argument('--zscan', nargs=3, type=float, default=None, help="same as xscan, but for Z direction")
parser.add_argument("--dmaxmin", nargs=2, type=float, default=[20,1.4])
parser.add_argument("--nbins", type=int, default=3000)
parser.add_argument("--combined", type=str, default=None)
parser.add_argument("--output_expt", type=str, default=None)

#parser.add_argument('--zscan', nargs=3, type=float, default=[0,0], help="scan range in Z in pixel units, passed as two args e.g. --zscan -5 5 1.5")
args = parser.parse_args()



from dials.array_family import flex
from dxtbx.model.experiment_list import ExperimentListFactory
import random
import numpy as np
from scipy import stats
from xfel.merging.application.utils.memory_usage import get_memory_usage
from dxtbx.model import Detector
from dxtbx.model import Experiment, ExperimentList


def shift_origin(DETECTOR, xyz_offset):
    DET = deepcopy(DETECTOR)
    hierarchy = DET.hierarchy()
    fast = hierarchy.get_local_fast_axis()
    slow = hierarchy.get_local_slow_axis()
    origin = hierarchy.get_local_origin()
    corrected_origin = (
	    origin[0] + xyz_offset[0],
	    origin[1] + xyz_offset[1],
	    origin[2] + xyz_offset[2]
	    )
    hierarchy.set_local_frame(fast, slow, corrected_origin)
    return DET

# PARAMS
RUN=args.run
TRIAL=args.trial
RUN_GROUP=args.group
NJ = args.j

# INPUT FILEz
expts_file = args.combined + ".expt"
refls_file = args.combined + ".refl"
#sub_folder=os.path.join("r%04d" % RUN, "%03d_rg%03d" % (TRIAL, RUN_GROUP), "combined")
#results_dir = "/global/cscratch1/sd/blaschke/lv95/common/results"
#loadpath = os.path.join(results_dir, sub_folder)
#expts_file = os.path.join(loadpath, "all.expt")
#refls_file = os.path.join(loadpath, "all.refl")

for fname in [expts_file, refls_file]:
    if not os.path.exists(fname):
       raise IOError("Path not exisiting:  %s" % fname)


print("Will load %s and %s" % (expts_file, refls_file))
expts = ExperimentListFactory.from_json_file(expts_file, check_format=False)

# GLOBAL
S0_VECS = [B.get_s0() for B in expts.beams()]
detector = expts.detectors()[0]
pixsize = detector[0].get_pixel_size()[0]
DET_DICT = expts.detectors()[0].to_dict()

# get that shit outta here- imageset memory shit
expts = None
del expts

def main(scans, jid):
    #refls = flex.reflection_table.from_file("/global/cscratch1/sd/blaschke/lv95/common/results/r0048/024_rg016/combined/all.refl")
    #expts = ExperimentListFactory.from_json_file("/global/cscratch1/sd/blaschke/lv95/common/results/r0048/024_rg016/combined/all.expt", check_format=False)
    #refls = flex.reflection_table.from_file("/global/cscratch1/sd/blaschke/lv95/common/results/r0130/035_rg028/combined/all.refl")
    #expts = ExperimentListFactory.from_json_file("/global/cscratch1/sd/blaschke/lv95/common/results/r0130/035_rg028/combined/all.expt", check_format=False)
    refls = flex.reflection_table.from_file(refls_file)

    n_bins = args.nbins
    d_max, d_min = args.dmaxmin
    d_inv_low, d_inv_high = 1/d_max, 1/d_min
    DETECTOR = Detector.from_dict(DET_DICT)
    correls = [] 
    #DETECTOR = expts[0].detector
   
    # powder from strongs code: 
    for i_scan, xyz_offset in enumerate(scans):
        sums0 = flex.double(n_bins)
        sums1 = flex.double(n_bins)
        sums2 = flex.double(n_bins)
        sums3 = flex.double(n_bins)
        sums4 = flex.double(n_bins)
        sums5 = flex.double(n_bins)
        sums6 = flex.double(n_bins)
        sums7 = flex.double(n_bins)
        panelsums = {
            0: sums0,
            1: sums1,
            2: sums2,
            3: sums3,
            4: sums4,
            5: sums5,
            6: sums6,
            7: sums7,
            }
        d_table = []

        DET = shift_origin(DETECTOR, xyz_offset)

        #DET = deepcopy(DETECTOR)
        #hierarchy = DET.hierarchy()
        #fast = hierarchy.get_local_fast_axis()
        #slow = hierarchy.get_local_slow_axis()
        #origin = hierarchy.get_local_origin()
        #corrected_origin = (
        #        origin[0] + xyz_offset[0],
        #        origin[1] + xyz_offset[1],
        #        origin[2] + xyz_offset[2]
        #        )
        #hierarchy.set_local_frame(fast, slow, corrected_origin)
        for i, s0 in enumerate(S0_VECS):
            #s0 = expt.beam.get_s0()
            sel = refls['id'] == i
            refls_sel = refls.select(sel)
            xyzobses = refls_sel['xyzobs.px.value']
            intensities = refls_sel['intensity.sum.value']
            panels = refls_sel['panel']

            for i_refl in range(len(refls_sel)):
                i_panel = panels[i_refl]
                panel = DET[i_panel]
                
                xy = xyzobses[i_refl][0:2]
                intensity = intensities[i_refl]
                res = panel.get_resolution_at_pixel(s0, xy)
                d_table.append((res, intensity))
                res_inv = 1/res
                i_bin = int(n_bins * (res_inv - d_inv_low) / (d_inv_high - d_inv_low))
                if i_bin < 0 or i_bin >= n_bins: continue
                panelsums[i_panel][i_bin] += 1 
                    
        xvalues = np.linspace(d_inv_low, d_inv_high, n_bins)

        offset = max(np.array(sums1))
        yvalues_list = []
        for i_sums, sums in enumerate([sums1, sums2, sums5, sums6]):
            yvalues = np.array(sums)
            yvalues_list.append(yvalues)

        # CHOOSE BESE CENTER
        correl = 0
        for i in range(len(yvalues_list)):
            for j in range(i+1, len(yvalues_list), 1):
                correl += stats.pearsonr(yvalues_list[i], yvalues_list[j])[0]
        x,y,z = xyz_offset
        correls.append( [correl,x,y,z])
        print("Job %d: scan (%d / %d) CC %f ... usage= %d" % 
            (jid, i_scan+1, len(scans), correl,get_memory_usage() ) )

    return correls


# SETUP SCAN
xstart, xstop, xres = args.xscan
ystart, ystop, yres = args.yscan
xscan = np.arange(xstart*pixsize, xstop*pixsize+1e-6, pixsize*xres)  # add 1e-6 so arange includes end point
yscan = np.arange(ystart*pixsize, ystop*pixsize+1e-6, pixsize*yres)
zscan = np.array([0]) 
if args.zscan is not None:
    zstart, zstop, zres = args.zscan
    zscan = np.arange(zstart*pixsize, zstop*pixsize+1e-6, pixsize*zres)

print(xscan)
print(yscan)


xinit = yinit = zinit = 0
if args.xyz is not None:
    xinit, yinit, zinit = args.xyz

scans = [ (x+xinit,y+yinit,z+zinit) for x in xscan for y in yscan for z in zscan]
scans_per_job = np.array_split(scans, NJ)
print(scans_per_job)

# RUN MAIN
res = Parallel(n_jobs=NJ)(delayed(main)(scans_per_job[j], j) for j in range(NJ))
res = [r for jobout in res for r in jobout if r]
best = sorted(res, key=lambda x: x[0])[::-1][0]
C,X,Y,Z = best 

if args.output_expt is not None:
    if not os.path.splitext(args.output_expt)[-1] == ".expt":
        args.output_expt += ".expt"

    new_det = shift_origin(detector, (X,Y,Z))
    El = ExperimentList()
    E = Experiment()
    E.detector = new_det
    El.append(E)
    #outname = "run%d_trial%d_group%d" % (RUN, TRIAL, RUN_GROUP)
    El.as_file(args.output_expt)

#with open( outname + ".txt" , "w") as out:
#    s = 'xyz_offset="%.5f %.5f %.5f"\n'%(X,Y,Z)
#    print(s)
#    out.write(s)
s = 'xyz_offset="%.5f %.5f %.5f"\n'%(X,Y,Z)
print(s)



if args.heat:
    c,x,y,z = zip(*res)
    import IPython;IPython.embed()
    nbinsY = len(yscan)
    nbinsX = len(xscan)
    ybins = np.linspace(np.min(y), np.max(y)+1e-6,nbinsY)
    xbins = np.linspace(np.min(x), np.max(x)+1e-6,nbinsX)
    heat = np.histogram2d( y,x,bins=(ybins,xbins), weights=c)[0]
    norm = np.histogram2d( y,x,bins=(ybins,xbins))[0]
    import pylab as plt
    plt.imshow(heat/norm, extent=(xbins[0], xbins[-1], ybins[-1], ybins[0]), 
	interpolation="nearest", cmap='magma')
    plt.plot([X], [Y], 'rx')
    plt.xlabel("x-shift")
    plt.ylabel("y-shift")
    plt.title(s)
    plt.show()
