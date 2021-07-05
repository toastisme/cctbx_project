from __future__ import division
from simtbx.diffBragg.utils import get_diffBragg_instance
import numpy as np

print ("Instantiating diffBragg")
D = get_diffBragg_instance()
D.vectorize_umats()
print ("Declaring a refinement manager")
rotX = 0
D.refine(rotX)  # defaults full image as ROI
D.initialize_managers()
print ("Running diffBragg on full image")
D.add_diffBragg_spots()
img_sh = D.raw_pixels.as_numpy_array().shape
print ("Extracting derivative pixels")
full_deriv_image = D.get_derivative_pixels(rotX).as_numpy_array()
print()
print ("Re-running but using only the ROIs")
rois = (20, 50, 10, 80), (10, 20, 80, 90)
for x1, x2, y1, y2 in rois:
    print("Declarind ROI %d %d %d %d" % (x1, x2, y1, y2))
    D.raw_pixels*= 0
    D.region_of_interest = ((x1, x2), (y1, y2))
    print("Running diffBRagg on ROI")
    D.add_diffBragg_spots()
    print("Extracting derivative pixels")
    roi_deriv_image = D.get_derivative_pixels(rotX).as_numpy_array()
    assert np.allclose(roi_deriv_image, full_deriv_image.reshape(img_sh)[y1:y2, x1:x2].ravel())
    print("LOOP OVER")
print("DONE!")
