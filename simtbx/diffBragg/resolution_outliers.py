from __future__ import print_function
from argparse import ArgumentParser
parser = ArgumentParser()

parser.add_argument("--input", type=str, help="input pickle, output of stg1_view_mp.py")
parser.add_argument("--statsPkl", type=str, default=None, help="input pickle, to be used for computing outlier statistics, should be the output of a previous run of this script")
parser.add_argument("--output", type=str, help="output pickle, pkl with filtered refls")
parser.add_argument("--threshR", type=float, help="resolution bin outliers are at least this man stdev above median", default=5)
parser.add_argument("--threshH", type=float, help="ASU hkl outliers are at least this man stdev above median", default=5)
parser.add_argument("--nbins", type=int, default=100, help="number of resolution bins")
parser.add_argument("--nasu", type=int, default=10, help="min multiplicity for filtering outliers amongst ASU equivalent HKLs")
parser.add_argument("--j", type=int, default=10, help="Number of jobs used to Parallel-write all new reflection tables")
parser.add_argument("--reflTag", type=str, default="filt", help="tag name for the filtered reflection tables, to be stored alongside the original reflection tables")
parser.add_argument("--reflOutdir", type=str, default=None, help="optional output directory for filtered reflections")
parser.add_argument("--expref", type=str, help="new experiments reflections spectrum filename for re-running stage 1 without outliers")
parser.add_argument("--save", action="store_true", help="save new refls, and exp ref file")
args = parser.parse_args()
import pandas
import os
from dials.array_family import flex
import numpy as np
from simtbx.diffBragg import utils

df = pandas.read_pickle(args.input) #"BB_753_8_pandas.pkl")
h,k,l = np.vstack([v for v in df.hkl.values]).T
sigZ = np.hstack([v for v in df.sigmaZ_PoissonDat.values])
res = np.hstack([r for r in df.resolution])
ref_idx = np.hstack([r for r in df.refls_idx])
ref_name = np.hstack([[i]*len(r) for i,r in zip(df.stage1_refls, df.resolution) ])

dfZ = pandas.DataFrame({"h":h, "k":k, "l":l, "refl_idx":ref_idx, "sigZ":sigZ, "res": res, "refl_file": ref_name})

## outliers within resolution shells:

# determine equally popultated resolution bins
res_split = np.array_split(np.sort(res), args.nbins)
bins = [r[0] for r in res_split] + [res_split[-1][-1]]
dfZ["res_bin"] = np.digitize(dfZ.res, bins)
gb_res = dfZ.groupby("res_bin")
gb_hkl = dfZ.groupby(["h", "k", "l"])
hkls = list(gb_hkl.groups.keys())

# initialize a column for outlier flags
dfZ["is_res_outlier"] = False
dfZ["is_hkl_outlier"] = False

# optional load a previous outlier pickle from this script
if args.statsPkl is not None:
    df_stats = pandas.read_pickle(args.statsPkl)
    gb_res_stats = df_stats.groupby("res_bin")
    gb_hkl_stats = df_stats.groupby(["h","k","l"])
    assert set(list(gb_res_stats.groups.keys())) == set(list(gb_res.groups.keys()))
    stats_hkls = list(gb_hkl_stats.groups.keys())
    missing_hkls = set(hkls).difference(stats_hkls)
    print("%d / %d hkl are not in the stats pkl" % ( len(missing_hkls), len(hkls)))
    #assert set(list(gb_hkl_stats.groups.keys())) == set(list(gb_hkl.groups.keys()))
else:
    gb_res_stats = gb_hkl_stats = None

# iterate over bins and compute outliers
active_resbins = set(list(gb_res.groups.keys()))
for r in range(1, args.nbins+1):
    if r not in active_resbins:
        print("Warning, there are empty resolution bins, try reducing the number of bins (currently using %d bins)" % args.nbins)
        continue

    df_r = gb_res.get_group(r)


    v = df_r.sigZ.values
    if gb_res_stats is not None:
        v = gb_res_stats.get_group(r).sigZ.values
    md = np.median(v)
    diff = np.abs(v-md)
    md_diff = np.median(diff)
    cutoff = md + md_diff*args.threshR*1.4836

    is_res_outlier = df_r.sigZ.values > cutoff
    print("bin %3d: %5.3f - %5.3f, cutoff=%.2f, Max sigZ in bin: %.2f, detected %d / %d outliers "\
          % (r-1,round(bins[r-1],3), round(bins[r],3), cutoff, df_r.sigZ.max(), sum(is_res_outlier), len(df_r)))
    dfZ.loc[df_r.index, "is_res_outlier"] = is_res_outlier

# filter amongst populations of hkl

nused =0
for H in hkls:
    df_H = gb_hkl.get_group(H)
    if len(df_H) < args.nasu:
        continue
    nused += 1

    v = df_H.sigZ.values
    if gb_hkl_stats is not None and H not in missing_hkls:
        v = gb_hkl_stats.get_group(H).sigZ.values
    md = np.median(v)
    diff = np.abs(v-md)
    md_diff = np.median(diff)
    cutoff = md + md_diff*args.threshH*1.4836

    #is_hkl_outlier = utils.is_outlier(df_H.sigZ.values, thresh=args.threshH)
    is_hkl_outlier = df_H.sigZ.values > cutoff
    print("%d / %d outliers for ASU %d,%d,%d" % ((sum(is_hkl_outlier),len(is_hkl_outlier))+H))
    dfZ.loc[df_H.index, "is_hkl_outlier"] = is_hkl_outlier

dfZ["is_outlier"] = np.logical_or(dfZ.is_res_outlier, dfZ.is_hkl_outlier)

print("%d/%d ASU HKLs had populations large enough for outlier testing" % (nused, len(hkls)))
print("\n\nTOTALS\n<><><><>")
print("Number of resolution bin outliers: %d" % dfZ.is_res_outlier.sum())
print("Number of outliers detected in populations of ASU equivalents: %d" % dfZ.is_hkl_outlier.sum())
print("Total number of flagged outliers: %d" % dfZ.is_outlier.sum())

if not args.save:
    print("Exiting, to save add the flag --save")
    exit()


dfZ.to_pickle(args.output)
print("\nWrote output to pandas pickle %s" % args.output)


#### Now open the reflection tables and save new ones without outliers
if args.reflOutdir is not None:
    if not os.path.exists(args.reflOutdir):
        os.makedirs(args.reflOutdir)


def new_refl_name(orig_refl_name):
    filt_name = os.path.splitext(orig_refl_name)[0] + "_%s.refl" % args.reflTag
    if args.reflOutdir is not None:
        basename = os.path.basename(filt_name)
        filt_name = os.path.abspath(os.path.join(args.reflOutdir, basename))
    return filt_name

# function for writing the filenames
def main(jid, filenames):
    for i_f, f in enumerate(filenames): #dfZ.refl_file.unique:
        if i_f % args.j != jid:
            continue
        df_file = dfZ.query("refl_file=='%s'" % f)
        R = flex.reflection_table.from_file(f)

        ref_idx = np.ones(len(R), bool)
        outliers = df_file.loc[df_file.is_outlier]
        if np.any(df_file.is_outlier):
            # this should be guarenteed if the bookkeeping was done properly
            assert outliers.refl_idx.max() < len(R)
        ref_idx[outliers.refl_idx.values] = False
        Rfilt = R.select(flex.bool(ref_idx))

        filt_name = new_refl_name(f)
        Rfilt.as_file(filt_name)
        #if jid==0:
        #    print("\rWrote file  %s (%d/%d)" % (filt_name, i_f+1, len(filenames)), flush=True,  end="")


from joblib import Parallel, delayed
Parallel(n_jobs=args.j)(delayed(main)(j,dfZ.refl_file.unique()) for j in range(args.j))

# make a new exper refl file
o = open(args.expref, "w")
for e,r,s in zip(df.exp_name, df.stage1_refls, df.spectrum_filename):
    new_r = new_refl_name(r)
    o.write("%s %s %s\n" % (e, new_r, s))
o.close()
print("\nWrote new exp ref file %s" % args.expref)
