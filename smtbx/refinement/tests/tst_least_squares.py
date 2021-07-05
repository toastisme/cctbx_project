from __future__ import absolute_import, division, print_function
from libtbx import object_oriented_patterns as oop
from scitbx.linalg import eigensystem, svd
from scitbx import matrix
from scitbx.lstbx import normal_eqns_solving
from cctbx import sgtbx, crystal, xray, adptbx, uctbx
import cctbx.sgtbx.lattice_symmetry
from cctbx import euclidean_model_matching as emma
from cctbx.array_family import flex
from cctbx.development import random_structure
from smtbx.refinement import least_squares
from smtbx.refinement import constraints
from smtbx.refinement.restraints import origin_fixing_restraints
import smtbx.utils
from libtbx.test_utils import approx_equal
import libtbx.utils
import math
import sys
import random
import re
from six.moves import range

tested_ls_engines = (
  least_squares.normal_eqns.non_linear_ls_with_separable_scale_factor_BLAS_2,
)
try:
  from fast_linalg import env
  if env.initialised:
    tested_ls_engines += (  least_squares.normal_eqns.non_linear_ls_with_separable_scale_factor_BLAS_3,)
except ImportError:
  print('Skipping fast_linalg checks')
# we use a wrapper to make sure non_linear_ls_with_separable_scale_factor
# is not an argument with a default value, so as to protect ourself from
# forgetting to specify that argument in the tests, which would result
# in the wrong feature to be tested, potentially.
def tested_crystallographic_ls(
  observations, reparametrisation,
  non_linear_ls_with_separable_scale_factor,
  may_parallelise,
  **kwds):
  return least_squares.crystallographic_ls(
    observations, reparametrisation,
    non_linear_ls_with_separable_scale_factor,
    may_parallelise,
    **kwds)

class refinement_test(object):

  ls_cycle_repeats = 1

  def __init__(self, ls_engine, parallelise):
    self.ls_engine = ls_engine
    self.parallelise = parallelise

  def run(self):
    if self.ls_cycle_repeats == 1:
      self.do_run()
    else:
      print("%s in %s" % (self.purpose, self.hall))
      for n in range(self.ls_cycle_repeats):
        self.do_run()
        print('.', end=' ')
        sys.stdout.flush()
      print()


class site_refinement_test(refinement_test):

  debug = 1
  purpose = "site refinement"

  def do_run(self):
    self.exercise_ls_cycles()
    self.exercise_floating_origin_restraints()

  def __init__(self, ls_engine, parallelise):
    refinement_test.__init__(self, ls_engine, parallelise)
    sgi = sgtbx.space_group_info("Hall: %s" % self.hall)
    cs = sgi.any_compatible_crystal_symmetry(volume=1000)
    xs = xray.structure(crystal.special_position_settings(cs))
    for sc in self.scatterers():
      sc.flags.set_use_u_iso(False).set_use_u_aniso(False)\
              .set_grad_site(True)
      xs.add_scatterer(sc)
    self.reference_xs = xs.as_emma_model()
    self.xray_structure = xs

    mi = cs.build_miller_set(d_min=0.5, anomalous_flag=False)
    ma = mi.structure_factors_from_scatterers(xs, algorithm="direct").f_calc()
    self.fo_sq = ma.norm().customized_copy(
      sigmas=flex.double(ma.size(), 1.))

  def exercise_floating_origin_restraints(self):
    n = self.n_independent_params
    eps_zero_rhs = 1e-6
    connectivity_table = smtbx.utils.connectivity_table(self.xray_structure)
    reparametrisation = constraints.reparametrisation(
      structure=self.xray_structure,
      constraints=[],
      connectivity_table=connectivity_table)
    obs = self.fo_sq.as_xray_observations()
    ls = tested_crystallographic_ls(
      obs, reparametrisation,
      non_linear_ls_with_separable_scale_factor=self.ls_engine,
      may_parallelise=self.parallelise,
      weighting_scheme=least_squares.unit_weighting(),
      origin_fixing_restraints_type=oop.null())
    ls.build_up()
    unrestrained_normal_matrix = ls.normal_matrix_packed_u()
    assert len(unrestrained_normal_matrix) == n*(n+1)//2
    ev = eigensystem.real_symmetric(
      unrestrained_normal_matrix.matrix_packed_u_as_symmetric())
    unrestrained_eigenval = ev.values()
    unrestrained_eigenvec = ev.vectors()

    ls = tested_crystallographic_ls(
      obs,
      reparametrisation,
      non_linear_ls_with_separable_scale_factor=self.ls_engine,
      may_parallelise=self.parallelise,
      weighting_scheme=least_squares.unit_weighting(),
      origin_fixing_restraints_type=
      origin_fixing_restraints.homogeneous_weighting)
    ls.build_up()

    # Let's check that the computed singular directions span the same
    # space as the expected ones
    singular_test = flex.double()
    jac = ls.reparametrisation.jacobian_transpose_matching_grad_fc()
    m = 0
    for s in ls.origin_fixing_restraint.singular_directions:
      assert s.norm() != 0
      singular_test.extend(jac*s)
      m += 1
    for s in self.continuous_origin_shift_basis:
      singular_test.extend(flex.double(s))
      m += 1
    singular_test.reshape(flex.grid(m, n))
    assert self.rank(singular_test) == len(self.continuous_origin_shift_basis)

    assert ls.opposite_of_gradient()\
             .all_approx_equal(0, eps_zero_rhs),\
           list(ls.gradient())
    restrained_normal_matrix = ls.normal_matrix_packed_u()
    assert len(restrained_normal_matrix) == n*(n+1)//2
    ev = eigensystem.real_symmetric(
      restrained_normal_matrix.matrix_packed_u_as_symmetric())
    restrained_eigenval = ev.values()
    restrained_eigenvec = ev.vectors()

    # The eigendecomposition of the normal matrix
    # for the unrestrained problem is:
    #    A = sum_{0 <= i < n-p-1} lambda_i v_i^T v_i
    # where the eigenvalues lambda_i are sorted in decreasing order
    # and p is the dimension of the continous origin shift space.
    # In particular A v_i = 0, n-p <= i < n.
    # In the restrained case, it becomes:
    #    A' = A + sum_{n-p <= i < n} mu v_i^T v_i

    p = len(self.continuous_origin_shift_basis)
    assert approx_equal(restrained_eigenval[p:], unrestrained_eigenval[:-p],
                        eps=1e-12)
    assert unrestrained_eigenval[-p]/unrestrained_eigenval[-p-1] < 1e-12

    if p > 1:
      # eigenvectors are stored by rows
      unrestrained_null_space = unrestrained_eigenvec.matrix_copy_block(
        i_row=n-p, i_column=0,
        n_rows=p, n_columns=n)
      assert self.rank(unrestrained_null_space) == p

      restrained_space = restrained_eigenvec.matrix_copy_block(
        i_row=0, i_column=0,
        n_rows=p, n_columns=n)
      assert self.rank(restrained_space) == p

      singular = flex.double(
        self.continuous_origin_shift_basis)
      assert self.rank(singular) == p

      rank_finder = flex.double(n*3*p)
      rank_finder.resize(flex.grid(3*p, n))
      rank_finder.matrix_paste_block_in_place(unrestrained_null_space,
                                              i_row=0, i_column=0)
      rank_finder.matrix_paste_block_in_place(restrained_space,
                                              i_row=p, i_column=0)
      rank_finder.matrix_paste_block_in_place(singular,
                                              i_row=2*p, i_column=0)
      assert self.rank(rank_finder) == p
    else:
      # this branch handles the case p=1
      # it's necessary to work around a bug in the svd module
      # ( nx1 matrices crashes the code )
      assert approx_equal(
        restrained_eigenvec[0:n].angle(
          unrestrained_eigenvec[-n:]) % math.pi, 0)
      assert approx_equal(
        unrestrained_eigenvec[-n:].angle(
          flex.double(self.continuous_origin_shift_basis[0])) % math.pi, 0)

    # Do the floating origin restraints prevent the structure from floating?
    xs = self.xray_structure.deep_copy_scatterers()
    ls = tested_crystallographic_ls(
      obs,
      reparametrisation,
      non_linear_ls_with_separable_scale_factor=self.ls_engine,
      may_parallelise=self.parallelise,
      weighting_scheme=least_squares.unit_weighting(),
      origin_fixing_restraints_type=
      origin_fixing_restraints.atomic_number_weighting
    )
    barycentre_0 = xs.sites_frac().mean()
    while True:
      xs.shake_sites_in_place(rms_difference=0.15)
      xs.apply_symmetry_sites()
      barycentre_1 = xs.sites_frac().mean()
      delta = matrix.col(barycentre_1) - matrix.col(barycentre_0)
      moved_far_enough = 0
      for singular in self.continuous_origin_shift_basis:
        e = matrix.col(singular[:3])
        if not approx_equal(delta.dot(e), 0, eps=0.01, out=None):
          moved_far_enough += 1
      if moved_far_enough: break

    # one refinement cycle
    ls.build_up()
    ls.solve()
    shifts = ls.step()

    # That's what floating origin restraints are for!
    # Note that in the presence of special position, that's different
    # from the barycentre not moving along the continuous shift directions.
    # TODO: typeset notes about that subtlety.
    for singular in self.continuous_origin_shift_basis:
      assert approx_equal(shifts.dot(flex.double(singular)), 0, eps=1e-12)

  def rank(cls, a):
    """ row rank of a """
    rank_revealing = svd.real(a.deep_copy(),
                              accumulate_u=False, accumulate_v=False)
    return rank_revealing.numerical_rank(rank_revealing.sigma[0]*1e-9)
  rank = classmethod(rank)

  def exercise_ls_cycles(self):
    xs = self.xray_structure.deep_copy_scatterers()
    connectivity_table = smtbx.utils.connectivity_table(xs)
    emma_ref = xs.as_emma_model()
    # shaking must happen before the reparametrisation is constructed,
    # otherwise the original values will prevail
    xs.shake_sites_in_place(rms_difference=0.1)
    reparametrisation = constraints.reparametrisation(
      structure=xs,
      constraints=[],
      connectivity_table=connectivity_table)
    ls = tested_crystallographic_ls(
      self.fo_sq.as_xray_observations(), reparametrisation,
      non_linear_ls_with_separable_scale_factor=self.ls_engine,
      may_parallelise=self.parallelise,
      weighting_scheme=least_squares.mainstream_shelx_weighting(a=0),
      origin_fixing_restraints_type=
      origin_fixing_restraints.atomic_number_weighting)

    cycles = normal_eqns_solving.naive_iterations(
      ls,
      gradient_threshold=1e-12,
      step_threshold=1e-7,
      track_all=True)

    assert approx_equal(ls.scale_factor(), 1, eps=1e-5), ls.scale_factor()
    assert approx_equal(ls.objective(), 0), ls.objective()

    match = emma.model_matches(emma_ref, xs.as_emma_model()).refined_matches[0]
    assert match.rt.r == matrix.identity(3)
    for pair in match.pairs:
      assert approx_equal(match.calculate_shortest_dist(pair), 0, eps=1e-4)


class adp_refinement_test(refinement_test):

  random_u_cart_scale = 0.2
  purpose = "ADP refinement"

  class refinement_diverged(RuntimeError):
    pass

  def __init__(self, ls_engine, parallelise):
    refinement_test.__init__(self, ls_engine, parallelise)
    sgi = sgtbx.space_group_info("Hall: %s" % self.hall)
    cs = sgi.any_compatible_crystal_symmetry(volume=1000)
    xs = xray.structure(crystal.special_position_settings(cs))
    for i, sc in enumerate(self.scatterers()):
      sc.flags.set_use_u_iso(False).set_use_u_aniso(True)\
              .set_grad_u_aniso(True)
      xs.add_scatterer(sc)
      site_symm = xs.site_symmetry_table().get(i)
      u_cart = adptbx.random_u_cart(u_scale=self.random_u_cart_scale)
      u_star = adptbx.u_cart_as_u_star(cs.unit_cell(), u_cart)
      xs.scatterers()[-1].u_star = site_symm.average_u_star(u_star)
    self.xray_structure = xs

    mi = cs.build_miller_set(d_min=0.5, anomalous_flag=False)
    ma = mi.structure_factors_from_scatterers(xs, algorithm="direct").f_calc()
    self.fo_sq = ma.norm().customized_copy(
      sigmas=flex.double(ma.size(), 1.))

  def do_run(self):
    self.exercise_ls_cycles()

  def exercise_ls_cycles(self):
    xs = self.xray_structure.deep_copy_scatterers()
    xs.shake_adp() # it must happen before the reparamtrisation is constructed
                   # because the ADP values are read then and only then.
    connectivity_table = smtbx.utils.connectivity_table(xs)
    reparametrisation = constraints.reparametrisation(
      structure=xs,
      constraints=[],
      connectivity_table=connectivity_table)
    ls = tested_crystallographic_ls(
      self.fo_sq.as_xray_observations(), reparametrisation,
      non_linear_ls_with_separable_scale_factor=self.ls_engine,
      may_parallelise=self.parallelise,
      weighting_scheme=least_squares.mainstream_shelx_weighting(a=0),
      origin_fixing_restraints_type=oop.null())

    try:
      cycles = normal_eqns_solving.naive_iterations(
        ls,
        gradient_threshold=1e-12,
        track_all=True)

      assert approx_equal(ls.scale_factor(), 1, eps=1e-4)
      assert approx_equal(ls.objective(), 0)
      assert cycles.gradient_norm_history[-1] < cycles.gradient_threshold

      for sc0, sc1 in zip(self.xray_structure.scatterers(), xs.scatterers()):
        assert approx_equal(sc0.u_star, sc1.u_star)
    except RuntimeError as err:
      import re
      m = re.search(
        r'^cctbx::adptbx::debye_waller_factor_exp: \s* arg_limit \s+ exceeded'
        r'.* arg \s* = \s* ([\d.eE+-]+)', str(err), re.X)
      assert m is not None, eval
      print("Warning: refinement of ADP's diverged")
      print('         argument to debye_waller_factor_exp reached %s' % m.group(1))
      print('Here is the failing structure')
      xs.show_summary()
      xs.show_scatterers()
      raise self.refinement_diverged()


class p1_test(object):

  hall = "P 1"
  n_independent_params = 9
  continuous_origin_shift_basis = [ (1,0,0)*3,
                                    (0,1,0)*3,
                                    (0,0,1)*3 ]

  def scatterers(self):
    yield xray.scatterer("C1", (0.1, 0.2, 0.3))
    yield xray.scatterer("C2", (0.4, 0.7, 0.8))
    yield xray.scatterer("C3", (-0.1, -0.8, 0.6))


class p2_test(object):

  hall = "P 2x"
  n_independent_params = 7
  continuous_origin_shift_basis = [ (1,0,0, 1, 1,0,0) ]

  def scatterers(self):
    yield xray.scatterer("C1", (0.1, 0.2, 0.3))
    yield xray.scatterer("C2", (-0.3, 0, 0)) # on 2-axis
    yield xray.scatterer("C3", (0.4, 0.1, -0.1)) # near 2-axis

class pm_test(object):

  hall = "P -2x"
  n_independent_params = 8
  continuous_origin_shift_basis = [ (0,1,0, 1,0, 0,1,0),
                                    (0,0,1, 0,1, 0,0,1) ]

  def scatterers(self):
    yield xray.scatterer("C1", (0.1, 0.2, 0.3)) # near mirror plance
    yield xray.scatterer("C2", (0, -0.3, 0.4)) # on mirror plane
    yield xray.scatterer("C3", (0.7, 0.1, -0.1))

class all_special_position_test(object):
  """ cod_ma_xs/2104451.pickle (generated by cctbx/omz/cod_select_and_pickle.py)
      It was pointed out by Ralf that it crashes smtbx-refine:
      it turned out to be a bug in floating origin restraint
      and this is a regression test for that bug fix.
  """

  hall = "C 2c -2"
  n_independent_params = 12
  continuous_origin_shift_basis = [ (0, 1)*6 ]

  def scatterers(self):
    yield xray.scatterer('Ba', (0.500000, 0.369879, 0.431121))
    yield xray.scatterer('Mg', (-0.000000, 0.385261, -0.084062))
    yield xray.scatterer('F1', (0.000000, 0.291730, 0.783213))
    yield xray.scatterer('F2', (0.000000, 0.328018, 0.303193))
    yield xray.scatterer('F3', (0.000000, 0.506766, 0.591744))
    yield xray.scatterer('F4', (0.500000, 0.414262, 1.002902))

class twin_test(object):

  def __init__(self, ls_engine, parallelise):
    self.ls_engine = ls_engine
    self.parallelise = parallelise
    # Let's start from some X-ray structure
    self.structure = xs = xray.structure(
      crystal_symmetry=crystal.symmetry(
      unit_cell=(7.6338, 7.6338, 9.8699, 90, 90, 120),
      space_group_symbol='hall:  P 3 -2c'),
      scatterers=flex.xray_scatterer((
        xray.scatterer( #0
      label='LI',
      site=(0.032717, 0.241544, 0.254924),
      u=(0.000544, 0.000667, 0.000160,
         0.000326, 0.000072, -0.000030)),
        xray.scatterer( #1
      label='NA',
      site=(0.033809, 0.553123, 0.484646),
      u=(0.000554, 0.000731, 0.000174,
         0.000212, 0.000032, -0.000015)),
        xray.scatterer( #2
      label='S1',
      site=(0.000000, 0.000000, -0.005908),
      u=(0.000370, 0.000370, 0.000081,
         0.000185, 0.000000, 0.000000)),
        xray.scatterer( #3
      label='S2',
      site=(0.333333, 0.666667, 0.211587),
      u=(0.000244, 0.000244, 0.000148,
         0.000122, 0.000000, 0.000000)),
        xray.scatterer( #4
      label='S3',
      site=(0.666667, 0.333333, 0.250044),
      u=(0.000349, 0.000349, 0.000094,
         0.000174, 0.000000, 0.000000)),
        xray.scatterer( #5
      label='O1',
      site=(0.000000, -0.000000, 0.154207),
      u=(0.000360, 0.000360, 0.000149,
         0.000180, 0.000000, 0.000000),
      occupancy=0.999000),
        xray.scatterer( #6
      label='O2',
      site=(0.333333, 0.666667, 0.340665),
      u=(0.000613, 0.000613, 0.000128,
         0.000306, 0.000000, 0.000000)),
        xray.scatterer( #7
      label='O3',
      site=(0.666667, 0.333333, 0.112766),
      u=(0.000724, 0.000724, 0.000118,
         0.000362, 0.000000, 0.000000)),
        xray.scatterer( #8
      label='O4',
      site=(0.225316, 0.110088, -0.035765),
      u=(0.000477, 0.000529, 0.000213,
         0.000230, 0.000067, -0.000013)),
        xray.scatterer( #9
      label='O5',
      site=(0.221269, 0.442916, 0.153185),
      u=(0.000767, 0.000286, 0.000278,
         0.000210, 0.000016, -0.000082)),
        xray.scatterer( #10
      label='O6',
      site=(0.487243, 0.169031, 0.321690),
      u=(0.000566, 0.000582, 0.000354,
         0.000007, 0.000022, 0.000146))
      )))

    # We could use miller.array.twin_data below but we want to make
    # a few extra checks along the way. So we do it by hand...

    # Start by producing indices and Fo^2 for a 1st domain,
    # with an overall scale factor
    mi_1 = self.structure.build_miller_set(d_min=0.5, anomalous_flag=False
                                           ).map_to_asu()
    fo_sq_1 = mi_1.structure_factors_from_scatterers(
        self.structure, algorithm='direct').f_calc().norm()

    # Then introduce a twin law ...
    self.twin_law = sgtbx.rt_mx('y,x,-z')
    # For the record, we have a twinning by merohedry:
    assert self.twin_law not in xs.space_group().build_derived_laue_group()
    assert self.twin_law in sgtbx.lattice_symmetry.group(xs.unit_cell())
    # ... and compute the indices and Fo^2 for the 2nd domain
    fo_sq_2 = fo_sq_1.change_basis(sgtbx.change_of_basis_op(self.twin_law)
                                   ).map_to_asu()

    # Then add the two domains together with a twin fraction
    self.twin_fraction = 0.3 + 0.2*(1 + flex.random_double())
    matches = fo_sq_1.match_indices(fo_sq_2)
    # sanity check for a merohedral twin
    assert matches.singles(0).size() == 0
    assert matches.singles(1).size() == 0
    pairs = matches.pairs()
    fo_sq_1 = fo_sq_1.select(pairs.column(1))
    fo_sq_2 = fo_sq_2.select(pairs.column(0))
    self.fo_sq = fo_sq_1.customized_copy(
      data=(fo_sq_1.data()*(1 - self.twin_fraction) +
            fo_sq_2.data()*self.twin_fraction),
      sigmas=flex.double(fo_sq_1.size(), 1))

  def run(self):
    self.fo_sq = self.fo_sq.sort(by_value="packed_indices")
    self.exercise(fixed_twin_fraction=True)
    self.exercise(fixed_twin_fraction=False)
    self.fo_sq = self.fo_sq.sort(by_value="resolution")
    self.exercise(fixed_twin_fraction=True)
    self.exercise(fixed_twin_fraction=False)

  def exercise(self, fixed_twin_fraction):
    # Create a shaken structure xs ready for refinement
    xs0 = self.structure
    emma_ref = xs0.as_emma_model()
    xs = xs0.deep_copy_scatterers()
    xs.shake_sites_in_place(rms_difference=0.15)
    xs.shake_adp()
    for sc in xs.scatterers():
      sc.flags.set_use_u_iso(False).set_use_u_aniso(True)
      sc.flags.set_grad_site(True).set_grad_u_aniso(True)

    # Setup L.S. problem
    connectivity_table = smtbx.utils.connectivity_table(xs)
    shaken_twin_fraction = (
      self.twin_fraction if fixed_twin_fraction else
      self.twin_fraction + 0.1*flex.random_double())
    # 2nd domain in __init__
    twin_components = (xray.twin_component(
        twin_law=self.twin_law.r(), value=shaken_twin_fraction,
        grad=not fixed_twin_fraction),)
    reparametrisation = constraints.reparametrisation(
      structure=xs,
      constraints=[],
      connectivity_table=connectivity_table,
      twin_fractions=twin_components)
    obs = self.fo_sq.as_xray_observations(twin_components=twin_components)
    ls = tested_crystallographic_ls(
      obs, reparametrisation,
      non_linear_ls_with_separable_scale_factor=self.ls_engine,
      may_parallelise=self.parallelise,
      weighting_scheme=least_squares.unit_weighting(),
      origin_fixing_restraints_type=
      origin_fixing_restraints.atomic_number_weighting)

    # Refine till we get back the original structure (so we hope)
    cycles = normal_eqns_solving.levenberg_marquardt_iterations(
      ls,
      gradient_threshold=1e-12,
      step_threshold=1e-6,
      track_all=True)

    # Now let's start to check it all worked
    assert ls.n_parameters == 63 if fixed_twin_fraction else 64

    match = emma.model_matches(emma_ref, xs.as_emma_model()).refined_matches[0]
    assert match.rt.r == matrix.identity(3)
    for pair in match.pairs:
      assert approx_equal(match.calculate_shortest_dist(pair), 0, eps=1e-4), pair

    if fixed_twin_fraction:
      assert ls.twin_fractions[0].value == self.twin_fraction
    else:
      assert approx_equal(ls.twin_fractions[0].value, self.twin_fraction,
                          eps=1e-2)

    assert approx_equal(ls.scale_factor(), 1, eps=1e-5)
    assert approx_equal(ls.objective(), 0)


class site_refinement_in_p1_test(p1_test, site_refinement_test): pass

class site_refinement_in_p2_test(p2_test, site_refinement_test): pass

class site_refinement_in_pm_test(pm_test, site_refinement_test): pass

class site_refinement_with_all_on_special_positions(all_special_position_test,
                                                    site_refinement_test):
  pass


class adp_refinement_in_p1_test(p1_test, adp_refinement_test): pass

class adp_refinement_in_p2_test(p2_test, adp_refinement_test): pass

class adp_refinement_in_pm_test(pm_test, adp_refinement_test): pass


def exercise_normal_equations(ls_engine, parallelise):
  site_refinement_with_all_on_special_positions(ls_engine, parallelise).run()

  for klass in (adp_refinement_in_p1_test,
                adp_refinement_in_pm_test,
                adp_refinement_in_p2_test):
    for i in range(4):
      try:
        klass(ls_engine, parallelise).run()
        break
      except adp_refinement_test.refinement_diverged:
        print("Warning: ADP refinement diverged, retrying...")
    else:
      print ("Error: ADP refinement diverged four times in a row (%s)"
             % klass.__name__)

  site_refinement_in_p1_test(ls_engine, parallelise).run()
  site_refinement_in_pm_test(ls_engine, parallelise).run()
  site_refinement_in_p2_test(ls_engine, parallelise).run()

class special_positions_test(object):

  delta_site   = 0.1 # % (of unit cell c for constrained atoms)
  delta_u_star = 0.1 # %

  def __init__(self, ls_engine, parallelise, n_runs, **kwds):
    libtbx.adopt_optional_init_args(self, kwds)
    self.ls_engine = ls_engine
    self.parallelise = parallelise
    self.n_runs = n_runs
    self.crystal_symmetry = crystal.symmetry(
      unit_cell=uctbx.unit_cell((5.1534, 5.1534, 8.6522, 90, 90, 120)),
      space_group_symbol='Hall: P 6c')
    self.structure = xray.structure(
      self.crystal_symmetry.special_position_settings(),
      flex.xray_scatterer((
        xray.scatterer('K1',
                        site=(0, 0, -0.00195),
                        u=self.u_cif_as_u_star((0.02427, 0.02427, 0.02379,
                                                0.01214, 0.00000, 0.00000))),
        xray.scatterer('S1',
                       site=(1/3, 2/3, 0.204215),
                       u=self.u_cif_as_u_star((0.01423, 0.01423, 0.01496,
                                               0.00712, 0.00000, 0.00000 ))),
        xray.scatterer('Li1',
                       site=(1/3, 2/3, 0.815681),
                       u=self.u_cif_as_u_star((0.02132, 0.02132, 0.02256,
                                               0.01066, 0.00000, 0.00000 ))),
        xray.scatterer('O1',
                       site=(1/3, 2/3, 0.035931),
                       u=self.u_cif_as_u_star((0.06532, 0.06532, 0.01669,
                                               0.03266, 0.00000, 0.00000 ))),
        xray.scatterer('O2',
                       site=(0.343810, 0.941658, 0.258405),
                       u=self.u_cif_as_u_star((0.02639,  0.02079, 0.05284,
                                               0.01194, -0.00053,-0.01180 )))
      )))
    mi = self.crystal_symmetry.build_miller_set(anomalous_flag=False,
                                                d_min=0.5)
    fo_sq = mi.structure_factors_from_scatterers(
      self.structure, algorithm="direct").f_calc().norm()
    self.fo_sq = fo_sq.customized_copy(sigmas=flex.double(fo_sq.size(), 1))

  def u_cif_as_u_star(self, u_cif):
    return adptbx.u_cif_as_u_star(self.crystal_symmetry.unit_cell(), u_cif)

  def shake_point_group_3(self, sc):
    _, _, c, _, _, _ = self.crystal_symmetry.unit_cell().parameters()

    x, y, z = sc.site
    z += random.uniform(-self.delta_site, self.delta_site)/c
    sc.site = (x, y, z)

    u11, _, u33, _, _, _ = sc.u_star
    u11 *= 1 + random.uniform(-self.delta_u_star, self.delta_u_star)
    u33 *= 1 + random.uniform(-self.delta_u_star, self.delta_u_star)
    sc.u_star = (u11, u11, u33, u11/2, 0, 0)

  def run(self):
    if self.n_runs > 1:
      print('small inorganic refinement with many special positions')
      for i in range(self.n_runs):
        print('.', end=' ')
        self.exercise()
      print()
    else:
      self.exercise()

  def exercise(self):
    xs0 = self.structure
    xs = xs0.deep_copy_scatterers()
    k1, s1, li1, o1, o2 = xs.scatterers()
    self.shake_point_group_3(k1)
    self.shake_point_group_3(s1)
    self.shake_point_group_3(li1)
    self.shake_point_group_3(o1)
    o2.site = tuple(
      [ x*(1 + random.uniform(-self.delta_site, self.delta_site))
        for x in o2.site])
    o2.u_star = tuple(
      [ u*(1 + random.uniform(-self.delta_u_star, self.delta_u_star))
        for u in o2.u_star])

    for sc in xs.scatterers():
      sc.flags.set_use_u_iso(False).set_use_u_aniso(True)
      sc.flags.set_grad_site(True).set_grad_u_aniso(True)
      connectivity_table = smtbx.utils.connectivity_table(xs)
    reparametrisation = constraints.reparametrisation(
      structure=xs,
      constraints=[],
      connectivity_table=connectivity_table)
    ls = tested_crystallographic_ls(
      self.fo_sq.as_xray_observations(), reparametrisation,
      non_linear_ls_with_separable_scale_factor=self.ls_engine,
      may_parallelise=self.parallelise,
      weighting_scheme=least_squares.unit_weighting(),
      origin_fixing_restraints_type=
      origin_fixing_restraints.atomic_number_weighting)

    cycles = normal_eqns_solving.levenberg_marquardt_iterations(
      ls,
      gradient_threshold=1e-12,
      step_threshold=1e-7,
      track_all=True)

    ## Test whether refinement brought back the shaked structure to its
    ## original state
    match = emma.model_matches(xs0.as_emma_model(),
                               xs.as_emma_model()).refined_matches[0]
    assert match.rt.r == matrix.identity(3)
    assert not match.singles1 and not match.singles2
    assert match.rms < 1e-6

    delta_u_carts= (   xs.scatterers().extract_u_cart(xs.unit_cell())
                    - xs0.scatterers().extract_u_cart(xs.unit_cell())).norms()
    assert flex.abs(delta_u_carts) < 1e-6

    assert approx_equal(ls.scale_factor(), 1, eps=1e-4)

    ## Test covariance matrix
    jac_tr = reparametrisation.jacobian_transpose_matching_grad_fc()
    cov = ls.covariance_matrix(
      jacobian_transpose=jac_tr, normalised_by_goof=False)\
        .matrix_packed_u_as_symmetric()
    m, n = cov.accessor().focus()
    # x,y for point group 3 sites are fixed: no variance or correlation
    for i in (0, 9, 18, 27,):
      assert cov.matrix_copy_block(i, 0, 2, n) == 0

    # u_star coefficients u13 and u23 for point group 3 sites are fixed
    # to 0: again no variance or correlation with any other param
    for i in (7, 16, 25, 34,):
      assert cov.matrix_copy_block(i, 0, 2, n).as_1d()\
             .all_approx_equal(0., 1e-20)

    # u_star coefficients u11, u22 and u12 for point group 3 sites
    # are totally correlated, with variances in ratios 1:1:1/2
    for i in (3, 12, 21, 30,):
      assert cov[i, i] != 0
      assert approx_equal(cov[i, i], cov[i+1, i+1], eps=1e-15)
      assert approx_equal(cov[i, i+1]/cov[i, i], 1, eps=1e-12)
      assert approx_equal(cov[i, i+3]/cov[i, i], 0.5, eps=1e-12)

def exercise_floating_origin_dynamic_weighting(ls_engine,
                                               parallelise,
                                               verbose=False):
  from cctbx import covariance
  import scitbx.math

  worst_condition_number_acceptable = 10

  # light elements only
  xs0 = random_structure.xray_structure(elements=['C', 'C', 'C', 'O', 'N'],
                                        use_u_aniso=True)
  msg = "light elements in %s ..." % (
    xs0.space_group_info().type().hall_symbol())
  if verbose:
    print(msg, end=' ')
  fo_sq = xs0.structure_factors(d_min=0.8).f_calc().norm()
  fo_sq = fo_sq.customized_copy(sigmas=flex.double(fo_sq.size(), 1.))
  xs = xs0.deep_copy_scatterers()
  xs.shake_adp()
  xs.shake_sites_in_place(rms_difference=0.1)
  for sc in xs.scatterers():
    sc.flags.set_grad_site(True).set_grad_u_aniso(True)
  ls = tested_crystallographic_ls(
    fo_sq.as_xray_observations(),
    constraints.reparametrisation(
      structure=xs,
      constraints=[],
      connectivity_table=smtbx.utils.connectivity_table(xs)),
    non_linear_ls_with_separable_scale_factor=ls_engine,
    may_parallelise=parallelise,
    weighting_scheme=least_squares.unit_weighting(),
    origin_fixing_restraints_type=
    origin_fixing_restraints.atomic_number_weighting)
  ls.build_up()
  lambdas = eigensystem.real_symmetric(
    ls.normal_matrix_packed_u().matrix_packed_u_as_symmetric()).values()
  # assert the restrained L.S. problem is not too ill-conditionned
  cond = math.log10(lambdas[0]/lambdas[-1])
  if verbose:
    print("normal matrix condition: %.1f" % cond)
  assert cond < worst_condition_number_acceptable, msg

  # one heavy element
  xs0 = random_structure.xray_structure(
    space_group_info=sgtbx.space_group_info('hall: P 2yb'),
    elements=['Zn', 'C', 'C', 'C', 'O', 'N'],
    use_u_aniso=True)
  msg = "one heavy element + light elements (synthetic data) in %s ..." % (
    xs0.space_group_info().type().hall_symbol())
  if verbose:
    print(msg, end=' ')
  fo_sq = xs0.structure_factors(d_min=0.8).f_calc().norm()
  fo_sq = fo_sq.customized_copy(sigmas=flex.double(fo_sq.size(), 1.))
  xs = xs0.deep_copy_scatterers()
  xs.shake_adp()
  xs.shake_sites_in_place(rms_difference=0.1)
  for sc in xs.scatterers():
    sc.flags.set_grad_site(True).set_grad_u_aniso(True)
  ls = tested_crystallographic_ls(
    fo_sq.as_xray_observations(),
    constraints.reparametrisation(
      structure=xs,
      constraints=[],
      connectivity_table=smtbx.utils.connectivity_table(xs)),
    non_linear_ls_with_separable_scale_factor=ls_engine,
    may_parallelise=parallelise,
    weighting_scheme=least_squares.mainstream_shelx_weighting(),
    origin_fixing_restraints_type=
    origin_fixing_restraints.atomic_number_weighting)
  ls.build_up()
  lambdas = eigensystem.real_symmetric(
    ls.normal_matrix_packed_u().matrix_packed_u_as_symmetric()).values()
  # assert the restrained L.S. problem is not too ill-conditionned
  cond = math.log10(lambdas[0]/lambdas[-1])
  if verbose:
    print("normal matrix condition: %.1f" % cond)
  assert cond < worst_condition_number_acceptable, msg

  # are esd's for x,y,z coordinates of the same order of magnitude?
  var_cart = covariance.orthogonalize_covariance_matrix(
    ls.covariance_matrix(),
    xs.unit_cell(),
    xs.parameter_map())
  var_site_cart = covariance.extract_covariance_matrix_for_sites(
      flex.size_t_range(len(xs.scatterers())),
      var_cart,
      xs.parameter_map())
  site_esds = var_site_cart.matrix_packed_u_diagonal()
  indicators = flex.double()
  for i in range(0, len(site_esds), 3):
    stats = scitbx.math.basic_statistics(site_esds[i:i+3])
    indicators.append(stats.bias_corrected_standard_deviation/stats.mean)
  assert indicators.all_lt(2)

  # especially troublesome structure with one heavy element
  # (contributed by Jonathan Coome)
  xs0 = xray.structure(
    crystal_symmetry=crystal.symmetry(
      unit_cell=(8.4519, 8.4632, 18.7887, 90, 96.921, 90),
      space_group_symbol="hall: P 2yb"),
    scatterers=flex.xray_scatterer([
      xray.scatterer( #0
                      label="ZN1",
                      site=(-0.736683, -0.313978, -0.246902),
                      u=(0.000302, 0.000323, 0.000054,
                         0.000011, 0.000015, -0.000004)),
      xray.scatterer( #1
                      label="N3B",
                      site=(-0.721014, -0.313583, -0.134277),
                      u=(0.000268, 0.000237, 0.000055,
                         -0.000027, 0.000005, 0.000006)),
      xray.scatterer( #2
                      label="N3A",
                      site=(-0.733619, -0.290423, -0.357921),
                      u=(0.000229, 0.000313, 0.000053,
                         0.000022, 0.000018, -0.000018)),
      xray.scatterer( #3
                      label="C9B",
                      site=(-1.101537, -0.120157, -0.138063),
                      u=(0.000315, 0.000345, 0.000103,
                         0.000050, 0.000055, -0.000017)),
    xray.scatterer( #4
                    label="N5B",
                    site=(-0.962032, -0.220345, -0.222045),
                    u=(0.000274, 0.000392, 0.000060,
                       -0.000011, -0.000001, -0.000002)),
    xray.scatterer( #5
                    label="N1B",
                    site=(-0.498153, -0.402742, -0.208698),
                    u=(0.000252, 0.000306, 0.000063,
                       0.000000, 0.000007, 0.000018)),
    xray.scatterer( #6
                    label="C3B",
                    site=(-0.322492, -0.472610, -0.114594),
                    u=(0.000302, 0.000331, 0.000085,
                       0.000016, -0.000013, 0.000037)),
    xray.scatterer( #7
                    label="C4B",
                    site=(-0.591851, -0.368163, -0.094677),
                    u=(0.000262, 0.000255, 0.000073,
                       -0.000034, 0.000027, -0.000004)),
    xray.scatterer( #8
                    label="N4B",
                    site=(-0.969383, -0.204624, -0.150014),
                    u=(0.000279, 0.000259, 0.000070,
                       -0.000009, 0.000039, 0.000000)),
    xray.scatterer( #9
                    label="N2B",
                    site=(-0.470538, -0.414572, -0.135526),
                    u=(0.000277, 0.000282, 0.000065,
                       0.000003, 0.000021, -0.000006)),
    xray.scatterer( #10
                    label="C8A",
                    site=(-0.679889, -0.158646, -0.385629),
                    u=(0.000209, 0.000290, 0.000078,
                       0.000060, 0.000006, 0.000016)),
    xray.scatterer( #11
                    label="N5A",
                    site=(-0.649210, -0.075518, -0.263412),
                    u=(0.000307, 0.000335, 0.000057,
                       -0.000002, 0.000016, -0.000012)),
    xray.scatterer( #12
                    label="C6B",
                    site=(-0.708620, -0.325965, 0.011657),
                    u=(0.000503, 0.000318, 0.000053,
                       -0.000058, 0.000032, -0.000019)),
    xray.scatterer( #13
                    label="C10B",
                    site=(-1.179332, -0.083184, -0.202815),
                    u=(0.000280, 0.000424, 0.000136,
                       0.000094, 0.000006, 0.000013)),
    xray.scatterer( #14
                    label="N1A",
                    site=(-0.838363, -0.532191, -0.293213),
                    u=(0.000312, 0.000323, 0.000060,
                       0.000018, 0.000011, -0.000008)),
    xray.scatterer( #15
                    label="C3A",
                    site=(-0.915414, -0.671031, -0.393826),
                    u=(0.000319, 0.000384, 0.000078,
                       -0.000052, -0.000001, -0.000020)),
    xray.scatterer( #16
                    label="C1A",
                    site=(-0.907466, -0.665419, -0.276011),
                    u=(0.000371, 0.000315, 0.000079,
                       0.000006, 0.000036, 0.000033)),
    xray.scatterer( #17
                    label="C1B",
                    site=(-0.365085, -0.452753, -0.231927),
                    u=(0.000321, 0.000253, 0.000087,
                       -0.000024, 0.000047, -0.000034)),
    xray.scatterer( #18
                    label="C11A",
                    site=(-0.598622, 0.053343, -0.227354),
                    u=(0.000265, 0.000409, 0.000084,
                       0.000088, -0.000018, -0.000030)),
    xray.scatterer( #19
                    label="C2A",
                    site=(-0.958694, -0.755645, -0.337016),
                    u=(0.000394, 0.000350, 0.000106,
                       -0.000057, 0.000027, -0.000005)),
    xray.scatterer( #20
                    label="C4A",
                    site=(-0.784860, -0.407601, -0.402050),
                    u=(0.000238, 0.000296, 0.000064,
                       0.000002, 0.000011, -0.000016)),
    xray.scatterer( #21
                    label="C5A",
                    site=(-0.784185, -0.399716, -0.475491),
                    u=(0.000310, 0.000364, 0.000062,
                       0.000044, -0.000011, -0.000017)),
    xray.scatterer( #22
                    label="N4A",
                    site=(-0.630284, -0.043981, -0.333143),
                    u=(0.000290, 0.000275, 0.000074,
                       0.000021, 0.000027, 0.000013)),
    xray.scatterer( #23
                    label="C10A",
                    site=(-0.545465, 0.166922, -0.272829),
                    u=(0.000369, 0.000253, 0.000117,
                       0.000015, -0.000002, -0.000008)),
    xray.scatterer( #24
                    label="C9A",
                    site=(-0.567548, 0.102272, -0.339923),
                    u=(0.000346, 0.000335, 0.000103,
                       -0.000016, 0.000037, 0.000023)),
    xray.scatterer( #25
                    label="C11B",
                    site=(-1.089943, -0.146930, -0.253779),
                    u=(0.000262, 0.000422, 0.000102,
                       -0.000018, -0.000002, 0.000029)),
    xray.scatterer( #26
                    label="N2A",
                    site=(-0.843385, -0.537780, -0.366515),
                    u=(0.000273, 0.000309, 0.000055,
                       -0.000012, -0.000005, -0.000018)),
    xray.scatterer( #27
                    label="C7A",
                    site=(-0.674021, -0.136086, -0.457790),
                    u=(0.000362, 0.000378, 0.000074,
                       0.000043, 0.000034, 0.000016)),
    xray.scatterer( #28
                    label="C8B",
                    site=(-0.843625, -0.264182, -0.102023),
                    u=(0.000264, 0.000275, 0.000072,
                       -0.000025, 0.000019, -0.000005)),
    xray.scatterer( #29
                    label="C6A",
                    site=(-0.726731, -0.261702, -0.502366),
                    u=(0.000339, 0.000472, 0.000064,
                       0.000062, -0.000003, 0.000028)),
    xray.scatterer( #30
                    label="C5B",
                    site=(-0.577197, -0.376753, -0.020800),
                    u=(0.000349, 0.000353, 0.000066,
                       -0.000082, -0.000022, 0.000014)),
    xray.scatterer( #31
                    label="C2B",
                    site=(-0.252088, -0.497338, -0.175057),
                    u=(0.000251, 0.000342, 0.000119,
                       0.000020, 0.000034, -0.000018)),
    xray.scatterer( #32
                    label="C7B",
                    site=(-0.843956, -0.268811, -0.028080),
                    u=(0.000344, 0.000377, 0.000078,
                       -0.000029, 0.000059, -0.000007)),
    xray.scatterer( #33
                    label="F4B",
                    site=(-0.680814, -0.696808, -0.115056),
                    u=(0.000670, 0.000408, 0.000109,
                       -0.000099, 0.000139, -0.000031)),
    xray.scatterer( #34
                    label="F1B",
                    site=(-0.780326, -0.921249, -0.073962),
                    u=(0.000687, 0.000357, 0.000128,
                       -0.000152, -0.000011, 0.000021)),
    xray.scatterer( #35
                    label="B1B",
                    site=(-0.795220, -0.758128, -0.075955),
                    u=(0.000413, 0.000418, 0.000075,
                       0.000054, 0.000045, 0.000023)),
    xray.scatterer( #36
                    label="F2B",
                    site=(-0.945140, -0.714626, -0.105820),
                    u=(0.000584, 0.001371, 0.000108,
                       0.000420, 0.000067, 0.000134)),
    xray.scatterer( #37
                    label="F3B",
                    site=(-0.768914, -0.701660, -0.005161),
                    u=(0.000678, 0.000544, 0.000079,
                       -0.000000, 0.000090, -0.000021)),
    xray.scatterer( #38
                    label="F1A",
                    site=(-0.109283, -0.252334, -0.429288),
                    u=(0.000427, 0.001704, 0.000125,
                       0.000407, 0.000041, 0.000035)),
    xray.scatterer( #39
                    label="F4A",
                    site=(-0.341552, -0.262864, -0.502023),
                    u=(0.000640, 0.000557, 0.000081,
                       -0.000074, 0.000042, -0.000052)),
    xray.scatterer( #40
                    label="F3A",
                    site=(-0.324533, -0.142292, -0.393215),
                    u=(0.000471, 0.001203, 0.000134,
                       0.000333, -0.000057, -0.000220)),
    xray.scatterer( #41
                    label="F2A",
                    site=(-0.312838, -0.405405, -0.400231),
                    u=(0.002822, 0.000831, 0.000092,
                       -0.000648, 0.000115, 0.000027)),
    xray.scatterer( #42
                    label="B1A",
                    site=(-0.271589, -0.268874, -0.430724),
                    u=(0.000643, 0.000443, 0.000079,
                       0.000040, 0.000052, -0.000034)),
    xray.scatterer( #43
                    label="H5B",
                    site=(-0.475808, -0.413802, 0.004402),
                    u=0.005270),
    xray.scatterer( #44
                    label="H6B",
                    site=(-0.699519, -0.326233, 0.062781),
                    u=0.019940),
    xray.scatterer( #45
                    label="H3B",
                    site=(-0.283410, -0.484757, -0.063922),
                    u=0.029990),
    xray.scatterer( #46
                    label="H1B",
                    site=(-0.357103, -0.451819, -0.284911),
                    u=0.031070),
    xray.scatterer( #47
                    label="H10A",
                    site=(-0.495517, 0.268296, -0.256187),
                    u=0.027610),
    xray.scatterer( #48
                    label="H2B",
                    site=(-0.147129, -0.535141, -0.174699),
                    u=0.017930),
    xray.scatterer( #49
                    label="H7A",
                    site=(-0.643658, -0.031387, -0.475357),
                    u=0.020200),
    xray.scatterer( #50
                    label="H1A",
                    site=(-0.912757, -0.691043, -0.227554),
                    u=0.033320),
    xray.scatterer( #51
                    label="H7B",
                    site=(-0.933670, -0.241189, -0.010263),
                    u=0.021310),
    xray.scatterer( #52
                    label="H11B",
                    site=(-1.107736, -0.155470, -0.311996),
                    u=0.041500),
    xray.scatterer( #53
                    label="H9A",
                    site=(-0.539908, 0.139753, -0.382281),
                    u=0.007130),
    xray.scatterer( #54
                    label="H10B",
                    site=(-1.265944, -0.029610, -0.212398),
                    u=0.030910),
    xray.scatterer( #55
                    label="H3A",
                    site=(-0.934728, -0.691149, -0.450551),
                    u=0.038950),
    xray.scatterer( #56
                    label="H5A",
                    site=(-0.833654, -0.487479, -0.508239),
                    u=0.031150),
    xray.scatterer( #57
                    label="H6A",
                    site=(-0.742871, -0.242269, -0.558157),
                    u=0.050490),
    xray.scatterer( #58
                    label="H9B",
                    site=(-1.120150, -0.093752, -0.090706),
                    u=0.039310),
    xray.scatterer( #59
                    label="H11A",
                    site=(-0.593074, 0.054973, -0.180370),
                    u=0.055810),
    xray.scatterer( #60
                    label="H2A",
                    site=(-0.999576, -0.842158, -0.340837),
                    u=0.057030)
    ]))
  fo_sq = xs0.structure_factors(d_min=0.8).f_calc().norm()
  fo_sq = fo_sq.customized_copy(sigmas=flex.double(fo_sq.size(), 1.))
  for hydrogen_flag in (True, False):
    xs = xs0.deep_copy_scatterers()
    if not hydrogen_flag:
      xs.select_inplace(~xs.element_selection('H'))
    xs.shake_adp()
    xs.shake_sites_in_place(rms_difference=0.1)
    for sc in xs.scatterers():
      sc.flags.set_grad_site(True).set_grad_u_aniso(False)
    ls = tested_crystallographic_ls(
      fo_sq.as_xray_observations(),
      constraints.reparametrisation(
        structure=xs,
        constraints=[],
        connectivity_table=smtbx.utils.connectivity_table(xs)),
      non_linear_ls_with_separable_scale_factor=ls_engine,
      may_parallelise=parallelise,
      weighting_scheme=least_squares.unit_weighting(),
      origin_fixing_restraints_type=
      origin_fixing_restraints.atomic_number_weighting)

    ls.build_up()
    lambdas = eigensystem.real_symmetric(
      ls.normal_matrix_packed_u().matrix_packed_u_as_symmetric()).values()
    # assert the restrained L.S. problem is not too ill-conditionned
    cond = math.log10(lambdas[0]/lambdas[-1])
    msg = ("one heavy element + light elements (real data) %s Hydrogens: %.1f"
           % (['without', 'with'][hydrogen_flag], cond))
    if verbose: print(msg)
    assert cond < worst_condition_number_acceptable, msg


    # are esd's for x,y,z coordinates of the same order of magnitude?
    var_cart = covariance.orthogonalize_covariance_matrix(
      ls.covariance_matrix(),
      xs.unit_cell(),
      xs.parameter_map())
    var_site_cart = covariance.extract_covariance_matrix_for_sites(
        flex.size_t_range(len(xs.scatterers())),
        var_cart,
        xs.parameter_map())
    site_esds = var_site_cart.matrix_packed_u_diagonal()
    indicators = flex.double()
    for i in range(0, len(site_esds), 3):
      stats = scitbx.math.basic_statistics(site_esds[i:i+3])
      indicators.append(stats.bias_corrected_standard_deviation/stats.mean)
    assert indicators.all_lt(1)


def run():
  libtbx.utils.show_times_at_exit()
  import sys
  from libtbx.option_parser import option_parser
  command_line = (option_parser()
    .option(None, "--start_with_parallel",
            action="store_true",
            default=False)
    .option(None, "--fix_random_seeds",
            action="store_true",
            default=False)
    .option(None, "--runs",
            type='int',
            default=1)
    .option(None, "--verbose",
            action="store_true",
            default=False)
    .option(None, "--skip-twin-test",
            dest='skip_twin_test',
            action="store_true",
            default=False)
  ).process(args=sys.argv[1:])
  opts = command_line.options
  if opts.fix_random_seeds:
    flex.set_random_seed(1)
    random.seed(2)
  n_runs = opts.runs
  if n_runs > 1: refinement_test.ls_cycle_repeats = n_runs

  for parallelise in ((True, False) if opts.start_with_parallel else
                      (False, True)):
    print (("Parallel" if parallelise else "Serial") +
           " computation of all Fc(h) and their derivatives")
    for ls_engine in tested_ls_engines:
      m = re.search(r'BLAS_(\d)', ls_engine.__name__)
      print("\tNormal matrix accumulation with BLAS level %s" %(m.group(1)))
      exercise_normal_equations(ls_engine, parallelise)
      exercise_floating_origin_dynamic_weighting(ls_engine, parallelise,
                                                 opts.verbose)
      special_positions_test(ls_engine, parallelise, n_runs).run()
      if not opts.skip_twin_test:
        twin_test(ls_engine, parallelise).run()

if __name__ == '__main__':
  run()
