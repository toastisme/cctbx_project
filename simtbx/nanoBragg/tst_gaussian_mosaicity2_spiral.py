from __future__ import absolute_import, division, print_function
from scitbx.array_family import flex
from scitbx.matrix import sqr,col
from simtbx.nanoBragg import nanoBragg
import libtbx.load_env # possibly implicit
import math
from libtbx.test_utils import approx_equal
from six.moves import range
from scipy import special
import numpy as np
from simtbx.nanoBragg.tst_gaussian_mosaicity import plotter, plotter2, check_distributions

"""Nick Sauter 6/21/2020
The purpose of function run_uniform() is to provide UMAT and d_UMAT_d_eta with the following
1. Large (scalable) ensemble of UMATs for use by nanoBragg.
2. UMATs are parameterized by eta, the half-angle of mosaic rotation (degrees) assuming Gaussian distribution
3. eta is the standard deviation of the Gaussian
4. The UMATs and d_UMAT_d_eta are completely determined by eta.
5. The UMATs are implemented using the axis & angle approach.
6. Unit axes are drawn randomly from the unit sphere by scitbx.
7. The angle theta is applied, as well as -theta; therefore the number of UMATs is always even.
8. Theta is uniformly chosen over the CDF of the Normal distribution.
9. Therefore the UMAT derivative with respect to eta requires the inverse erf function, supplied by SciPy.
"""

MOSAIC_SPREAD = 2  # top hat half width rotation in degrees


def search_directions(N=1000, seed=12345):
  """
  See Journal of Magnetic Resonance 138, 288–297 (1999)
  equation A6
  :param N: number of points on hemisphere
  :param seed: random seed
  :return: Nx3 numpy array of unit vectors
  """
  np.random.seed(seed)
  ti = (np.arange(1, N + 1) - 0.5) / N
  THETA = np.arccos(ti)
  PHI = np.sqrt(np.pi * N) * np.arcsin(ti)

  u_vecs = np.zeros((N, 3))
  x = np.sin(THETA) * np.cos(PHI)
  y = np.sin(THETA) * np.sin(PHI)
  z = np.cos(THETA)
  u_vecs[:N, 0] = x
  u_vecs[:N, 1] = y
  u_vecs[:N, 2] = z

  return u_vecs


def run_uniform(eta_angle, Naxes=10, verbose=True, Nang=10):
  print("Generate %d axes and %d angles per axes" % (Naxes, 2*Nang))
  UMAT = flex.mat3_double()
  d_UMAT_d_eta = flex.mat3_double()

  rot_axes = search_directions(Naxes)
  for ax in rot_axes:
    site = col(ax)
    for i_ang in range(Nang):
      ang_idx = i_ang / Nang
      d = np.sqrt(2)*special.erfinv(ang_idx)
      m = np.pi/180*eta_angle*d
      for ang_sign in [1, -1]:
        UMAT.append( site.axis_and_angle_as_r3_rotation_matrix(ang_sign*m,deg=False) )
        d_umat = site.axis_and_angle_as_r3_derivative_wrt_angle(ang_sign*m, deg=False)
        d_UMAT_d_eta.append((math.pi/180.) * ang_sign*d * d_umat)

  #sanity check on the gaussian distribution
  if verbose:
    nm_angles = check_distributions.get_angular_rotation(UMAT)
    nm_rms_angle = math.sqrt(flex.mean(nm_angles*nm_angles))
    print("Normal rms angle is ", nm_rms_angle)

  return UMAT, d_UMAT_d_eta

def check_finite(mat1, mat2, dmat1, eps):
  for im in range(len(mat1)):
    m1 = sqr(mat1[im])
    m2 = sqr(mat2[im])
    f_diff = (m2 - m1)/eps
    deriv = sqr(dmat1[im])
    assert approx_equal(f_diff.elems, deriv.elems)

def tst_all(make_plots):
  eps =0.0001 # degrees
  UM , d_UM = run_uniform(eta_angle=MOSAIC_SPREAD)
  UMp , d_UMp = run_uniform(eta_angle=MOSAIC_SPREAD+eps)
  UMm , d_UMm = run_uniform(eta_angle=MOSAIC_SPREAD-eps)
  check_finite (UM, UMp, d_UM, eps)
  check_finite (UM, UMm, d_UM, -eps)

  UM_2 , d_UM2 = run_uniform(eta_angle=MOSAIC_SPREAD/2.)
  if make_plots:
    P = plotter(UM,UM_2)
  Q = plotter2(UM,UM_2,make_plots, eps=1e-3) # suggested by Holton:
  """apply all the UMATs to a particular starting unit vector and take the rms of the resulting end points.
    You should not see a difference between starting with a vector with x,y,z = 0,0,1 vs
    x,y,z = 0.57735,0.57735,0.57735.  But if the implementation is wrong and the UMATs are being made by
    generating Gaussian-random rotations about the three principle axes, then the variance of
    0,0,1 will be significantly smaller than that of 0.57735,0.57735,0.57735."""

if __name__=="__main__":
  import sys
  make_plots = "--plot" in sys.argv
  tst_all(make_plots)
  print("OK")
