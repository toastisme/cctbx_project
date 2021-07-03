from __future__ import absolute_import, division, print_function
import math, random
from scitbx import matrix
from scitbx.array_family import flex
from six.moves import range

# -----------------------------------------------------------------------------
# simplified form of surface integral for spherical harmonic (l = m)
# http://en.wikipedia.org/wiki/Table_of_spherical_harmonics
def ylm(lm,c,t,p):
  y = c * math.pow(math.sin(t),lm) * complex(math.cos(lm*p),math.sin(lm*p))
  return y * y.conjugate() * math.sin(t)

# -----------------------------------------------------------------------------
def add_point(lm,c,R):

  x = matrix.col( [0,0,1] )
  new_x = R * x

  theta = math.acos(new_x[2])                    # theta = [0, pi]
  phi = math.atan2(new_x[1],new_x[0]) + math.pi  # phi = [0, 2pi)

  return ylm(lm,c,theta,phi)

# -----------------------------------------------------------------------------
def test_uniform_rotation_matrix(N=10000,choice=2,verbose=False):
  r"""
  The surface integral of a spherical harmonic function with its conjugate
  should be 1. (http://mathworld.wolfram.com/SphericalHarmonic.html, Eq 7)

  From Mathematica,

  l = 10;
  m = 10;
  y = SphericalHarmonicY[l, m, \[Theta], \[Phi]];
  Integrate[y*Conjugate[y]*Sin[\[Theta]], {\[Theta], 0, Pi}, {\[Phi], 0, 2*Pi}]

  should yield 1.

  By picking uniformly random points on a sphere, the surface integral can be
  numerically approximated.

  The results in the comments below are for N = 1 000 000.
  """
  if (choice == 0):
    # l=1, m=1
    # result = (0.883199394206+0j) (0.883824001444+0j)
    lm = 1
    c = -0.5 * math.sqrt(1.5/math.pi)
  elif (choice == 1):
    # l = 5, m = 5
    # result = (0.959557841214+0j) (0.959331535539+0j)
    lm = 5
    c = -(3/32) * math.sqrt(77/math.pi)
  else:
    # l = 10, m = 10
    # result = (0.977753926603+0j) (0.97686871766+0j)
    lm = 10
    c = (1/1024) * math.sqrt(969969/math.pi)

  result = [ 0.0, 0.0 ]
  for i in range(N):
    R  = [ matrix.sqr(flex.random_double_r3_rotation_matrix()),
           matrix.sqr(flex.random_double_r3_rotation_matrix_arvo_1992()) ]
    for j in range(len(result)):
      result[j] += add_point(lm,c,R[j])

  # multipy by area at the end, each point has an area of 4pi/N
  point_area = 4.0*math.pi/N  # surface area of unit sphere / number of points
  for i in range(len(result)):
    result[i] = point_area * result[i]
    if (verbose):
      print(result[i], end=' ')
  if (verbose):
    print()

  assert(result[0].real > 0.85)
  assert(result[0].real < 1.15)
  assert(result[1].real > 0.85)
  assert(result[1].real < 1.15)

if (__name__ == '__main__'):
  flex.set_random_seed(0)
  for i in range(3):
    test_uniform_rotation_matrix(N=1000, choice=i, verbose=False)
  print('OK')
