from simtbx.nanoBragg.utils import H5AttributeGeomWriter
import h5py
import numpy as np
import pandas
from dxtbx.model.experiment_list import ExperimentListFactory
from simtbx.diffBragg import utils
import sys

import glob

img_names =glob.glob(sys.argv[1])

adu_per_photon = 9.481  # TODO add to the pandas pickle
mask = utils.load_dials_flex_mask("/global/cfs/cdirs/m3562/der/master_files/newmask_withbad.pkl")

Zvals = []
for name in img_names:
    h = h5py.File(name, 'r')
    bragg = h['bragg']
    model = h['model']
    data = h['data']
    pids = h['pids']
    rois = h['rois']
    sigma_rdout = h['sigma_rdout'][()]
    sigma_r_img = sigma_rdout  # TODO add in the perpixel sigma readout values
    num_rois = len(rois)

    for i_roi in range(num_rois):
        x1, x2, y1, y2 = rois[i_roi]
        pid = pids[i_roi]
        trusted = mask[pid, y1:y2, x1:x2]
        mod_subimg = model['roi%d' % i_roi][()]
        data_subimg = data['roi%d' % i_roi][()]
        sigma_subimg = np.sqrt(mod_subimg + sigma_rdout**2)
        Z = data_subimg - mod_subimg
        Z /= sigma_subimg
        Zvals.append( Z[trusted].ravel())
Zvals = np.hstack(Zvals)

from pylab import *
vmin = min(Zvals)
vmax = max(Zvals)
hist(Zvals, bins=500, histtype='step', lw=2)
Zstd = np.std(Zvals)
title("%s\nZsigma = %f" % (sys.argv[1], Zstd))
show()
