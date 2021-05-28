# coding: utf-8
import glob
from joblib import Parallel,delayed

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--input", type=str, help="input pickle")
parser.add_argument("--glob", type=str, help="optional glob for input pickles", default=None)
parser.add_argument("--n", type=int, help="pixel cutoff", default=2)
parser.add_argument("--j", type=int, help="number of jobs", default=1)
parser.add_argument("--save", type=str, help="optional file name for saving figure output", default=None)
parser.add_argument("--signalcut", type=float, default=None, help="optional signal to backgroud cutoff")
args = parser.parse_args()
import h5py
from simtbx.diffBragg import ls49_utils
import numpy as np
import pandas
from scipy.ndimage.filters import gaussian_filter as GF
from scipy.spatial import distance, cKDTree
import os
from simtbx.diffBragg import utils
from dials.array_family import flex
from scipy.ndimage import label, maximum_filter,center_of_mass

def get_centroid(img):
    lab, nlab = label(img > np.percentile(img,90))
    com = center_of_mass(img, lab)
    return com, nlab

def get_centroid2(img):
    peakmask = maximum_filter(img, size=2)

print("pandas input")
if args.glob is not None:
    fnames = glob.glob(args.glob)
    print("found %d files in glob" % len(fnames))
    df = pandas.concat([pandas.read_pickle(f) for f in fnames])
else:
    df = pandas.read_pickle(args.input)

if "imgs" not in list(df):
    #df["imgs"] = [df.opt_exp_name.values[0].replace("expers", "imgs").replace(".expt", ".h5")][f.replace("expers", "imgs").replace(".expt", ".h5") for f in df.opt_exp_name]
    df['imgs'] = [f.replace("expers", "imgs").replace(".expt", ".h5") for f in df.opt_exp_name]
if "refl_names" not in list(df):
    df['refl_names'] = [f.replace(".expt", ".refl") for f in df.exp_name]
    #refls = [f.replace("_pathmod.expt", "_idx.refl") for f in df.exp_name]
    #df['refl_names'] = refls

def main(jid):
    dev_res =[]
    per_img_dists = []
    per_img_dials_dists =[]
    per_img_Z = []
    per_img_Z2 = []
    per_img_signal = []
    per_img_ref_index =[]
    per_img_vec_dists = []
    per_img_dials_vec_dists =[]
    img_names =[]
    per_img_shot_roi = []
    for ii, (imgf,reff) in  enumerate(df[["imgs","refl_names"]].values):
        if ii % args.j != jid:
            continue
        if jid==0:
            print("Processing %d / %d" % (ii+1, len(df)))
        h = h5py.File(imgf, "r")
        dat = h['data']
        if 'bragg' in list(h.keys()):
            mod = h['bragg']
        else:
            mod = h['model']
        nroi = len(dat.keys())
        R = flex.reflection_table.from_file(reff)
        R["refl_index"] = flex.int(range(len(R)))
        Rpp = utils.refls_by_panelname(R)
        pids = h['pids']
        rois = h['rois']
        all_dists = []
        all_dials_dists = []
        all_signal = []
        all_ref_index=[]
        all_vec_dists =[]
        all_dials_vec_dists = []
        all_Z =[]
        all_Z2 =[]
        all_shot_roi = []
        sigma_rdout = h["sigma_rdout"][()]
        for img_i_roi in range(nroi):
            ddd = dat["roi%d" % img_i_roi][()]
            mmm = mod["roi%d" % img_i_roi][()]
            signal = mmm.max() / np.median(mmm)
            if args.signalcut is not None and signal < args.signalcut:
                continue
            noise = np.sqrt(ddd + sigma_rdout**2)
            noise2 = np.sqrt(mmm + sigma_rdout**2)
            Z = (ddd-mmm) / noise
            Z2 = (ddd-mmm) / noise2
            roi_d = GF(ddd, 1) 
            roi_m = GF(mmm,0 )
            com_d, nlab_d = get_centroid(roi_d)
            com_m, nlab_m = get_centroid(roi_m)
            pid = pids[img_i_roi]
            ref_p = Rpp[pid]
            x1,x2,y1,y2 = rois[img_i_roi]
            if np.any(np.isnan(com_m)) or np.any(np.isnan(com_d)):
                continue
            if nlab_m != 1 or nlab_d != 1:
                continue
            xyz = np.array(ref_p["xyzobs.px.value"])
            xy = xyz[:,:2]
            tree = cKDTree(xy)
            y_com,x_com = com_d
            y_com += y1+0.5
            x_com += x1+0.5
            y_nelder, x_nelder = com_m
            y_nelder += y1+0.5
            x_nelder += x1+0.5
            res = tree.query_ball_point((x_com, y_com), r=7)
            if len(res) == 0:
                print("weird tree query multiple or no res")
                continue
            dists = [distance.euclidean(np.array(ref_p[i_r]['xyzobs.px.value'])[:2], (x_com,y_com)) for i_r in res]
            close = np.argmin(dists)
            i_roi = res[close]
            r = ref_p[i_roi]
            xcal,ycal,_ = r['xyzcal.px']
            xobs,yobs,_ = r['xyzobs.px.value']
            d_dials = distance.euclidean((x_com, y_com), (xcal, ycal))
            d = distance.euclidean((xobs, yobs), (x_nelder, y_nelder))
            
            # output
            dev_res.append( (d, d_dials,imgf, i_roi, nroi))

            # distances
            all_dists.append(d)
            all_dials_dists.append(d_dials)
            # Z-score sigmas
            all_Z.append(np.std(Z))
            all_Z2.append(np.mean(Z))
            
            # vector differences
            vec_dials_d = np.array((xobs, yobs)) - np.array((xcal, ycal))
            vec_d = np.array((xobs, yobs)) - np.array((x_nelder, y_nelder))
            all_vec_dists.append(vec_d)
            all_dials_vec_dists.append(vec_dials_d)
            all_shot_roi.append(img_i_roi)
            all_ref_index.append(r["refl_index"])
            all_signal.append(signal)

        img_names.append(imgf)
        per_img_ref_index.append(tuple(all_ref_index))
        per_img_signal.append(tuple(all_signal))
        per_img_dists.append(tuple(all_dists))
        per_img_dials_dists.append(tuple(all_dials_dists))
        per_img_vec_dists.append(tuple(all_vec_dists))
        per_img_dials_vec_dists.append(tuple(all_dials_vec_dists))
        per_img_Z.append(tuple(all_Z))
        per_img_Z2.append(tuple(all_Z2))
        per_img_shot_roi.append( all_shot_roi)

    return dev_res, per_img_dists, per_img_dials_dists, per_img_Z, per_img_Z2, per_img_vec_dists\
                ,per_img_dials_vec_dists, per_img_shot_roi, per_img_signal, img_names, per_img_ref_index

results = Parallel(n_jobs=args.j)(delayed(main)(j) for j in range(args.j))

dev_res =[]
per_img_dists = []
per_img_dials_dists =[]
per_img_Z = []
per_img_Z2 = []
per_img_vec_dists =[]
per_img_signal = []
per_img_ref_index =[]
per_img_dials_vec_dists =[]
per_img_shot_roi = []
img_names =[]
for r in results:
    dev_res +=r[0]
    per_img_dists += r[1]
    per_img_dials_dists +=r[2]
    per_img_Z += r[3]
    per_img_Z2 +=r[4]
    per_img_vec_dists += r[5]
    per_img_dials_vec_dists += r[6]
    per_img_shot_roi += r[7]
    per_img_signal += r[8]
    img_names += r[9]
    per_img_ref_index += r[10]

df_process = pandas.DataFrame(
    {"sigmaZ_PoissonDat" : per_img_Z,
    "Z_PoissonMod" : per_img_Z2,
    "signal_to_background" : per_img_signal,
    "pred_offsets" : per_img_dists,
    "refl_index": per_img_ref_index,
    "pred_offsets_dials" : per_img_dials_dists,
    "img_i_roi" : per_img_shot_roi,"imgs": img_names})

df = pandas.merge(df, df_process, on="imgs", how='inner')
# NOTE uncomment for LS49
#df['tstamp'] = [ls49_utils.get_tstamp(f) for f in df.imgs]

vecx = [ tuple([i for i,j in v]) for v in per_img_vec_dists]
vecy = [ tuple([j for i,j in v]) for v in per_img_vec_dists]
df["vec_x"] = vecx
df["vec_y"] = vecy
dials_vecx = [ tuple([i for i,j in v]) for v in per_img_dials_vec_dists]
dials_vecy = [ tuple([j for i,j in v]) for v in per_img_dials_vec_dists]
df["dials_vec_x"] = dials_vecx
df["dials_vec_y"] = dials_vecy
#===================================================================

d,d2,imgf,_,_ = zip(*dev_res)
from pylab import *
bins = linspace(min(d+d2), max(d+d2), 100)
hist(d, bins=bins, histtype='step', lw=2, label="after nelder-mead", color="C0")
hist(d2, bins=bins, histtype='step', lw=2, label="from dials", color="tomato")
d = np.array(d)
d2 = np.array(d2)
good_d = d[d < args.n]
before_good_d = d2[ d < args.n]

from itertools import groupby
dimg = list(zip(d, imgf))
gb = groupby(sorted(dimg, key=lambda x: x[1]), key=lambda x:x[1])

gb_results = {k:list(v) for k,v in gb}

#better_imgf = [v for sublist in [[ss[1] for ss in s] for s in stuff if len(s) > 1] for v in sublist]

better_d = []
better_imgf =[]
imgnames = list(gb_results.keys())
med_dists = []
for name in imgnames:
    img_dists = [v[0] for v in gb_results[name]]
    n_within_2 = sum([dist < args.n for dist in img_dists])
    if n_within_2 >= 2:
        better_d += img_dists
        better_imgf.append(name)
    med_dists.append(np.median(img_dists))

df["good_img"] = df.imgs.isin(better_imgf)
df_better = df.loc[df.imgs.isin(better_imgf)]
df_better.reset_index(inplace=True, drop=True)
df_better["predictions"] = df_better.refl_names
#df_better.to_pickle(args.input.replace(".pkl", "_stg2.pkl"))

nwithin = len(good_d)
print("good", np.median(good_d))
print("Before good", np.median(before_good_d))
hist(good_d, bins=bins, histtype='stepfilled', lw=2, label="%d spots within %d pix" % (nwithin,args.n), alpha=0.5, color="C0")
hist(before_good_d, bins=bins, histtype='stepfilled', lw=2, label="%d spots prior to refinement" % nwithin, alpha=0.5, color="tomato")
hist(better_d, bins=bins, histtype='stepfilled', lw=2, label="%d spots on images containing 2 or more \npredictions within %.1f pix of observations (%d images)" % (len(better_d),args.n, len(set(better_imgf))), alpha=0.5, color="k")
legend()
xlabel("pixels")
xlabel("|xobs - xcal| pixels")
xlabel("|xobs - xcal| pixels", fontsize=12)
title("xcal and xobs for %d refls\n (%s)" % (len(d), args.input), fontsize=12)
ax = gca()
ax.tick_params(labelsize=12)
print("Number of modeled refls within %d pix of obs: %d " % (args.n,nwithin))
if args.save is not None:
    plt.savefig(args.save)
    pkl_name = os.path.splitext(args.save)[0] + "_pandas.pkl"
    df.to_pickle(pkl_name)
else:
    show()


def make_plot(d, d2, imgf, n=2):
    bins = linspace(min(d + d2), max(d + d2), 100)
    hist(d, bins=bins, histtype='step', lw=2, label="after nelder-mead", color="C0")
    hist(d2, bins=bins, histtype='step', lw=2, label="from dials", color="tomato")
    d = np.array(d)
    d2 = np.array(d2)
    good_d = d[d < n]
    before_good_d = d2[d < n]

    from itertools import groupby
    dimg = list(zip(d, imgf))
    gb = groupby(sorted(dimg, key=lambda x: x[1]), key=lambda x: x[1])

    gb_results = {k: list(v) for k, v in gb}

    # better_imgf = [v for sublist in [[ss[1] for ss in s] for s in stuff if len(s) > 1] for v in sublist]

    better_d = []
    better_imgf = []
    imgnames = list(gb_results.keys())
    med_dists = []
    for name in imgnames:
        img_dists = [v[0] for v in gb_results[name]]
        n_within_2 = sum([dist < n for dist in img_dists])
        if n_within_2 >= 2:
            better_d += img_dists
            better_imgf.append(name)
        med_dists.append(np.median(img_dists))

    nwithin = len(good_d)
    hist(good_d, bins=bins, histtype='stepfilled', lw=2, label="%d spots within %d pix" % (nwithin, n), alpha=0.5,
         color="C0")
    hist(before_good_d, bins=bins, histtype='stepfilled', lw=2, label="%d spots prior to refinement" % nwithin,
         alpha=0.5, color="tomato")
    hist(better_d, bins=bins, histtype='stepfilled', lw=2,
         label="%d spots on images containing 2 or more \npredictions within %.1f pix of observations (%d images)" % (
         len(better_d), n, len(set(better_imgf))), alpha=0.5, color="k")
    legend()
    xlabel("pixels")
    xlabel("|xobs - xcal| pixels")
    xlabel("|xobs - xcal| pixels", fontsize=12)
    title("xcal and xobs for %d refls\n" % (len(d)), fontsize=12)
    ax = gca()
    ax.tick_params(labelsize=12)
    show()