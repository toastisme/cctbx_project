from simtbx.nanoBragg.utils import H5AttributeGeomWriter
import h5py
import numpy as np
import pandas
from dxtbx.model.experiment_list import ExperimentListFactory
from simtbx.diffBragg import utils
import sys


df = pandas.read_pickle(sys.argv[1])
if 'imgs' not in list(df):
    df['imgs'] = [f.replace("expers", "imgs").replace(".expt", ".h5") for f in df.opt_exp_name]
exper = ExperimentListFactory.from_json_file(df.exp_name.values[0],True)[0]

adu_per_photon = 9.481  # TODO add to the pandas pickle
print("Getting data from exper")
data = utils.image_data_from_expt(exper)
print("Done")
data /= adu_per_photon

num_imgs = 4
panel_Xdim, panel_Ydim = exper.detector[0].get_image_size()
img_shape = len(exper.detector), panel_Ydim, panel_Xdim
img_path = "_temp_image_view_image.py.hdf5"
writer_args = {"filename": img_path,
               "image_shape": img_shape,
               "num_images": num_imgs,
               "detector": exper.detector, "beam": exper.beam}


model_img = np.zeros(img_shape)
spots_img = np.zeros(img_shape)

h = h5py.File(df.imgs.values[0], 'r')
bragg = h['bragg']
model = h['model']
pids = h['pids']
rois = h['rois']
sigma_rdout = h['sigma_rdout'][()]
sigma_r_img = sigma_rdout  # TODO add in the perpixel sigma readout values
num_rois = len(rois)

print("Updating model images")
for i_roi in range(num_rois):
    x1,x2,y1,y2 = rois[i_roi]
    pid = pids[i_roi]
    mod_subimg = model['roi%d' % i_roi]
    bragg_subimg = bragg['roi%d' % i_roi]
    spots_img[pid, y1:y2, x1:x2] = bragg_subimg
    model_img[pid, y1:y2, x1:x2] = mod_subimg

print("Done")

with H5AttributeGeomWriter(**writer_args) as writer:
    pids, ys, xs = np.where(model_img == 0)
    model_img[pids, ys, xs] = data[pids, ys, xs]
    # model_img *= self.params.refiner.adu_per_photon
    writer.add_image(model_img)
    writer.add_image(data)

    Zimg = model_img - data
    Zimg /= np.sqrt(model_img + sigma_r_img ** 2)

    Zimg = Zimg * 0.1 + 1
    Zimg[pids, ys, xs] = 1

    Zimg2 = model_img - data
    Zimg2 /= np.sqrt(model_img + sigma_r_img ** 2)

    Zimg2 = Zimg2 * 0.1 + 1
    Zimg2[pids, ys, xs] = 1

    writer.add_image(Zimg2)
    writer.add_image(spots_img)

import os
os.system("dials.image_viewer %s" % img_path)
