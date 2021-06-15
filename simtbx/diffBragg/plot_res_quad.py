import numpy as np
import pandas
import sys

res_bins = [30.0051,
            4.0169,
            3.1895,
            2.7867,
            2.5320,
            2.3506,
            2.2121,
            2.1013,
            2.0099,
            1.9325,
            1.8658,
            1.8075,
            1.7558,
            1.7096,
            1.6679,
            1.6300]

df = pandas.read_pickle("basin7534_pids_pandas.pkl")


sels = []
for d,d2 in zip(df.pred_offsets, df.pred_offsets_dials):
    score = np.median(d)-np.median(d2)
    if score > 0:
        sel = False
    else:
        sel = True
    sels.append(sel)
df = df.loc[np.array(sels)]



all_d, all_d_dials, all_res = [], [], []
all_shot = []
all_quad = []



for i, (d, d_dials, res, pids) in enumerate(zip(df.pred_offsets, df.pred_offsets_dials, df.resolution, df.panel)):
    # H = np.histogram(
    all_d += list(d)
    all_d_dials += list(d_dials)
    all_res += list(res)
    all_shot += ([i] * len(d))
    all_quad += [int(pid / 64) for pid in pids]
    print(i, len(d))

bin_assign = np.digitize(all_res, res_bins)
all_d = np.array(all_d)
all_d_dials = np.array(all_d_dials)
all_shot = np.array(all_shot)
all_quad =np.array(all_quad)
nbins = len(res_bins)

Q = int(sys.argv[1])
res_diffs = []
for i in range(1,nbins):
    print(i, nbins)
    sel = bin_assign==i
    shots = all_shot[sel]
    quads = all_quad[sel]
    d = all_d[sel]
    d_dials = all_d_dials[sel]
    ushot = set(shots)
    all_m = []
    all_m_dials = []
    for i_shot in ushot:
        if i_shot % 100 ==0:
            print("i_shot %d / %d" % (i_shot, len(ushot)), end="\r", flush=True)
        sel_shot = np.logical_and(shots==i_shot, quads==Q)
        #sel_shot = shots==i_shot
        d_shot = d[sel_shot]
        d_dials_shot = d_dials[sel_shot]
        m = np.median(d_shot)
        m_dials = np.median(d_dials_shot)
        all_m.append(m)
        all_m_dials.append(m_dials)
    diffs = np.array(all_m_dials) - np.array(all_m)
    print(i)
    res_diffs.append(diffs)


P = []
for i_res, diffs in enumerate(res_diffs):
    diff_sort = np.sort(diffs)[::-1]
    pos = diff_sort >= 0
    x = np.arange(diff_sort.shape[0])
    npos = sum(pos)
    ntot = diff_sort.shape[0]
    perc_pos = npos / ntot * 100.
    perc_neg = 100 - perc_pos
    #P.append(perc_pos)
    v = np.median(diffs[~np.isnan(diffs)])
    P.append(v)


from pylab import *
figure()
bar( range(1,len(P)+1), P, color='C3')
ax = gca()

ax.set_xticks(list(range(1,len(res_bins))))

labs = ["%2.2f - %2.2f" % (round(x1,2),round(x2,2)) for x1,x2 in zip(res_bins[0:-1], res_bins[1:])]
ax.set_xticklabels(labs, rotation=270, ha='center')

xlabel("resolution ($\AA$)",fontsize=14)
#ylabel("percent", fontsize=14)
ylabel("pixels", fontsize=14)
ax.tick_params(labelsize=12)
subplots_adjust(bottom=.3,right=.983)
grid(1,alpha=1, axis='y')
#ax.set_yticks(range(0,101,10))
title("QUAD=%d" % Q)
show()