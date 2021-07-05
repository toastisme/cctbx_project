from __future__ import absolute_import, division, print_function
from cctbx.array_family import flex

import boost_adaptbx.boost.python as bp
from six.moves import zip
ext = bp.import_ext("cctbx_adp_restraints_ext")
from cctbx_adp_restraints_ext import *

from cctbx import crystal
from cctbx import adptbx
import scitbx.restraints

from cctbx.geometry_restraints import weight_as_sigma
import sys

class energies_iso(scitbx.restraints.energies):

  def __init__(self,
        plain_pair_sym_table,
        xray_structure,
        parameters,
        use_u_local_only,
        use_hd,
        wilson_b=None,
        compute_gradients=True,
        gradients=None,
        normalization=False,
        collect=False):
    scitbx.restraints.energies.__init__(self,
      compute_gradients=compute_gradients,
      gradients=gradients,
      gradients_size=xray_structure.scatterers().size(),
      gradients_factory=flex.double,
      normalization=normalization)
    unit_cell = xray_structure.unit_cell()
    if(use_u_local_only):
      u_isos = xray_structure.scatterers().extract_u_iso()
      #assert (u_isos < 0.0).count(True) == 0 XXX
    else:
      u_isos = xray_structure.extract_u_iso_or_u_equiv()
    if(use_hd):
      selection = xray_structure.all_selection()
    else:
      selection = ~xray_structure.hd_selection()
    energies = crystal.adp_iso_local_sphere_restraints_energies(
      pair_sym_table=plain_pair_sym_table,
      orthogonalization_matrix=unit_cell.orthogonalization_matrix(),
      sites_frac=xray_structure.sites_frac(),
      u_isos=u_isos,
      selection = selection,
      use_u_iso = xray_structure.use_u_iso(),
      grad_u_iso= xray_structure.scatterers().extract_grad_u_iso(),
      sphere_radius=parameters.sphere_radius,
      distance_power=parameters.distance_power,
      average_power=parameters.average_power,
      min_u_sum=1.e-6,
      compute_gradients=compute_gradients,
      collect=collect)
    self.number_of_restraints += energies.number_of_restraints
    self.residual_sum += energies.residual_sum
    if (compute_gradients):
      self.gradients += energies.gradients
    if (not collect):
      self.u_i = None
      self.u_j = None
      self.r_ij = None
    else:
      self.u_i = energies.u_i
      self.u_j = energies.u_j
      self.r_ij = energies.r_ij
    if (    wilson_b is not None
        and wilson_b > 0
        and parameters.wilson_b_weight is not None
        and parameters.wilson_b_weight > 0):
      wilson_u = adptbx.b_as_u(wilson_b)
      u_diff = flex.mean(u_isos) - wilson_u
      self.number_of_restraints += 1
      if(compute_gradients):
        g_wilson = 2.*u_diff/u_isos.size()/wilson_u
        g_wilson = flex.double(u_isos.size(), g_wilson)
        norm1 = self.gradients.norm()
        norm2 = g_wilson.norm()
        if(norm2 > 0 and parameters.wilson_b_weight_auto):
          w = norm1 / norm2 * parameters.wilson_b_weight
        else:
          w = parameters.wilson_b_weight
      else:
        w = parameters.wilson_b_weight
      self.residual_sum += w * u_diff**2 / wilson_u
      if (compute_gradients):
        self.gradients = self.gradients + w * g_wilson
    self.finalize_target_and_gradients()

class adp_aniso_restraints(object):
  def __init__(self, xray_structure, restraints_manager, use_hd,
               selection = None):
    # Pairwise ADP restraints: 3 mix cases supported:
    #  o - ()
    #  o - o
    # () - ()
    # In SHELX this called SIMU restraints
    unit_cell = xray_structure.unit_cell()
    n_grad_u_iso = xray_structure.n_grad_u_iso()
    u_cart = xray_structure.scatterers().extract_u_cart(unit_cell)
    u_iso  = xray_structure.scatterers().extract_u_iso()
    scatterers = xray_structure.scatterers()
    sites_cart = xray_structure.sites_cart()
    if(selection is not None):
      sites_cart = sites_cart.select(selection)
    bond_proxies_simple = restraints_manager.pair_proxies(sites_cart =
      sites_cart).bond_proxies.simple
    if(selection is None):
      selection = flex.bool(scatterers.size(), True)
    hd_selection = xray_structure.hd_selection()
    result = eval_adp_aniso_restraints(
      scatterers=scatterers,
      u_cart=u_cart,
      u_iso=u_iso,
      bond_proxies=bond_proxies_simple,
      selection=selection,
      hd_selection=hd_selection,
      n_grad_u_iso=n_grad_u_iso,
      use_hd=use_hd)
    self.target = result.target
    self.number_of_restraints = result.number_of_restraints
    if (n_grad_u_iso == 0):
      self.gradients_iso = None
    else :
      self.gradients_iso = result.gradients_iso()
    self.gradients_aniso_cart = result.gradients_aniso_cart()
    self.gradients_aniso_star = adptbx.grad_u_cart_as_u_star(unit_cell,
                                               self.gradients_aniso_cart)

@bp.inject_into(adp_similarity)
class _():

  def _show_sorted_item(self, f, prefix):
    adp_labels = ("U11","U22","U33","U12","U13","U23")
    deltas = self.deltas()
    if self.use_u_aniso == (False, False):
      adp_labels = ["Uiso"]
      deltas = deltas[:1]
    print("%s          delta    sigma   weight" %(prefix), end=' ', file=f)
    if len(adp_labels) == 1:
      print("residual", file=f)
    else: print("rms_deltas residual", file=f)
    rdr = None
    for adp_label,delta in zip(adp_labels, deltas):
      if (rdr is None):
        if len(adp_labels) == 1:
          rdr = " %6.2e" %self.residual()
        else:
          rdr = "   %6.2e %6.2e" % (self.rms_deltas(), self.residual())
      print("%s %-4s %9.2e %6.2e %6.2e%s" % (
        prefix, adp_label, delta, weight_as_sigma(weight=self.weight), self.weight, rdr), file=f)
      rdr = ""

@bp.inject_into(shared_adp_similarity_proxy)
class _():

  def deltas_rms(self, params):
    return adp_similarity_deltas_rms(params=params, proxies=self)

  def residuals(self, params):
    return adp_similarity_residuals(params=params, proxies=self)

  def show_sorted(self,
        by_value,
        u_cart,
        u_iso,
        use_u_aniso,
        site_labels=None,
        f=None,
        prefix="",
        max_items=None):
    _show_sorted_impl(self=self,
        proxy_type=adp_similarity,
        proxy_label="ADP similarity",
        item_label="scatterers",
        by_value=by_value, u_cart=u_cart, u_iso=u_iso,
        use_u_aniso=use_u_aniso, sites_cart=None,
        site_labels=site_labels, f=f, prefix=prefix,
        max_items=max_items)

@bp.inject_into(adp_u_eq_similarity)
class _():

  def _show_sorted_item(self, f, prefix):
    print("%s Mean Ueq=%6.2e" %(prefix, self.mean_u_eq), file=f)
    print("%s weight=%6.2e sigma=%6.2e rms_deltas=%6.2e residual=%6.2e"\
      %(prefix, self.weight, weight_as_sigma(weight=self.weight),
        self.rms_deltas(), self.residual()), file=f)

@bp.inject_into(shared_adp_u_eq_similarity_proxy)
class _():

  def deltas_rms(self, params):
    return adp_u_eq_similarity_deltas_rms(params=params, proxies=self)

  def residuals(self, params):
    return adp_u_eq_similarity_residuals(params=params, proxies=self)

  def show_sorted(self,
        by_value,
        u_cart,
        u_iso,
        use_u_aniso,
        site_labels=None,
        f=None,
        prefix="",
        max_items=None):
    _show_sorted_impl(self=self,
        proxy_type=adp_u_eq_similarity,
        proxy_label="Ueq similarity",
        item_label="scatterers",
        by_value=by_value, u_cart=u_cart, u_iso=u_iso,
        use_u_aniso=use_u_aniso, sites_cart=None,
        site_labels=site_labels, f=f, prefix=prefix,
        max_items=max_items)

@bp.inject_into(adp_volume_similarity)
class _():

  def _show_sorted_item(self, f, prefix):
    print("%s Mean Volume=%6.2e" %(prefix, self.mean_u_volume), file=f)
    print("%s weight=%6.2e sigma=%6.2e rms_deltas=%6.2e residual=%6.2e"\
      %(prefix, self.weight, weight_as_sigma(weight=self.weight),
        self.rms_deltas(), self.residual()), file=f)

@bp.inject_into(shared_adp_volume_similarity_proxy)
class _():

  def deltas_rms(self, params):
    return adp_volume_similarity_deltas_rms(params=params, proxies=self)

  def residuals(self, params):
    return adp_volume_similarity_residuals(params=params, proxies=self)

  def show_sorted(self,
        by_value,
        u_cart,
        u_iso,
        use_u_aniso,
        site_labels=None,
        f=None,
        prefix="",
        max_items=None):
    _show_sorted_impl(self=self,
        proxy_type=adp_volume_similarity,
        proxy_label="ADP volume similarity",
        item_label="scatterers",
        by_value=by_value, u_cart=u_cart, u_iso=u_iso,
        use_u_aniso=use_u_aniso, sites_cart=None,
        site_labels=site_labels, f=f, prefix=prefix,
        max_items=max_items)

@bp.inject_into(isotropic_adp)
class _():

  def _show_sorted_item(self, f, prefix):
    adp_labels = ("U11","U22","U33","U12","U13","U23")
    print("%s         delta    sigma   weight rms_deltas residual" % (prefix), file=f)
    rdr = None
    for adp_label,delta in zip(adp_labels, self.deltas()):
      if (rdr is None):
        rdr = "   %6.2e %6.2e" % (self.rms_deltas(), self.residual())
      print("%s %s %9.2e %6.2e %6.2e%s" % (
        prefix, adp_label, delta, weight_as_sigma(weight=self.weight),
        self.weight, rdr), file=f)
      rdr = ""

@bp.inject_into(shared_isotropic_adp_proxy)
class _():

  def deltas_rms(self, params):
    return isotropic_adp_deltas_rms(params=params, proxies=self)

  def residuals(self, params):
    return isotropic_adp_residuals(params=params, proxies=self)

  def show_sorted(self,
        by_value,
        u_cart,
        u_iso,
        use_u_aniso,
        site_labels=None,
        f=None,
        prefix="",
        max_items=None):
    _show_sorted_impl(self=self,
        proxy_type=isotropic_adp,
        proxy_label="Isotropic ADP",
        item_label="scatterer",
        by_value=by_value, u_cart=u_cart, u_iso=u_iso,
        use_u_aniso=use_u_aniso, sites_cart=None,
        site_labels=site_labels, f=f, prefix=prefix,
        max_items=max_items)


@bp.inject_into(fixed_u_eq_adp)
class _():

  def _show_sorted_item(self, f, prefix):
    adp_label = "Ueq"
    print("%s          delta    sigma   weight" %(prefix), end=' ', file=f)
    print("residual", file=f)
    rdr = " %6.2e" %self.residual()
    print("%s %-4s %9.2e %6.2e %6.2e%s" % (
      prefix, adp_label, self.delta(), weight_as_sigma(weight=self.weight),
      self.weight, rdr), file=f)

@bp.inject_into(shared_fixed_u_eq_adp_proxy)
class _():

  def deltas_rms(self, params):
    return fixed_u_eq_adp_deltas_rms(params=params, proxies=self)

  def residuals(self, params):
    return fixed_u_eq_adp_residuals(params=params, proxies=self)

  def show_sorted(self,
        by_value,
        u_cart,
        u_iso,
        use_u_aniso,
        site_labels=None,
        f=None,
        prefix="",
        max_items=None):
    _show_sorted_impl(self=self,
        proxy_type=fixed_u_eq_adp,
        proxy_label="Fixed Ueq ADP",
        item_label="scatterer",
        by_value=by_value, u_cart=u_cart, u_iso=u_iso,
        use_u_aniso=use_u_aniso, sites_cart=None,
        site_labels=site_labels, f=f, prefix=prefix,
        max_items=max_items)


@bp.inject_into(rigid_bond)
class _():

  def _show_sorted_item(self, f, prefix):
    print("%s   delta_z    sigma   weight residual" % (prefix), file=f)
    print("%s %9.2e %6.2e %6.2e %6.2e" % (
      prefix, self.delta_z(), weight_as_sigma(weight=self.weight),
      self.weight, self.residual()), file=f)

@bp.inject_into(shared_rigid_bond_proxy)
class _():

  def deltas(self, params):
    return rigid_bond_deltas(params=params, proxies=self)

  def residuals(self, params):
    return rigid_bond_residuals(params=params, proxies=self)

  def show_sorted(self,
        by_value,
        sites_cart,
        u_cart,
        site_labels=None,
        f=None,
        prefix="",
        max_items=None):
    _show_sorted_impl(self=self,
        proxy_type=rigid_bond,
        proxy_label="Rigid bond",
        item_label="scatterers",
        by_value=by_value, u_cart=u_cart, u_iso=None,
        use_u_aniso=None, sites_cart=sites_cart,
        site_labels=site_labels, f=f, prefix=prefix,
        max_items=max_items)

@bp.inject_into(rigu)
class _():

  def _show_sorted_item(self, f, prefix):
    print("%s   delta_z    sigma   weight residual" % (prefix), file=f)
    print("%s %9.2e %6.2e %6.2e %6.2e" % (
      prefix, self.delta_33(), weight_as_sigma(weight=self.weight),
      self.weight, self.residual33()), file=f)
    print("%s %9.2e %6.2e %6.2e %6.2e" % (
      prefix, self.delta_13(), weight_as_sigma(weight=self.weight),
      self.weight, self.residual13()), file=f)
    print("%s %9.2e %6.2e %6.2e %6.2e" % (
      prefix, self.delta_13(), weight_as_sigma(weight=self.weight),
      self.weight, self.residual23()), file=f)

@bp.inject_into(shared_rigu_proxy)
class _():

  def deltas(self, params):
    return rigu_deltas(params=params, proxies=self)

  def residuals(self, params):
    return rigu_residuals(params=params, proxies=self)

  def show_sorted(self,
        by_value,
        sites_cart,
        u_cart,
        site_labels=None,
        f=None,
        prefix="",
        max_items=None):
    _show_sorted_impl(self=self,
        proxy_type=rigu,
        proxy_label="Rigu bond",
        item_label="scatterers",
        by_value=by_value, u_cart=u_cart, u_iso=None,
        use_u_aniso=None, sites_cart=sites_cart,
        site_labels=site_labels, f=f, prefix=prefix,
        max_items=max_items)

def _show_sorted_impl(self,
      proxy_type,
      proxy_label,
      item_label,
      by_value,
      u_cart,
      u_iso=None,
      use_u_aniso=None,
      sites_cart=None,
      site_labels=None,
      f=None,
      prefix="",
      max_items=None):
  assert by_value in ["residual", "rms_deltas", "delta"]
  assert site_labels is None or len(site_labels) == u_cart.size()
  assert sites_cart is None or len(sites_cart) == u_cart.size()
  assert [u_iso, use_u_aniso].count(None) in (0,2)
  if (f is None): f = sys.stdout
  print("%s%s restraints: %d" % (prefix, proxy_label, self.size()), file=f)
  if (self.size() == 0): return
  if (max_items is not None and max_items <= 0): return

  from cctbx.adp_restraints import adp_restraint_params
  if use_u_aniso is None:  use_u_aniso = []
  if sites_cart is None:  sites_cart = []
  if u_iso is None: u_iso = []
  params = adp_restraint_params(
    sites_cart=sites_cart, u_cart=u_cart, u_iso=u_iso, use_u_aniso=use_u_aniso)
  if (by_value == "residual"):
    if proxy_type in (adp_similarity, adp_u_eq_similarity, adp_volume_similarity,\
                      isotropic_adp, fixed_u_eq_adp, rigid_bond, rigu):
      data_to_sort = self.residuals(params=params)
    else:
      raise AssertionError
  elif (by_value == "rms_deltas"):
    if proxy_type in (adp_similarity, adp_u_eq_similarity, adp_volume_similarity,\
                      isotropic_adp, fixed_u_eq_adp):
      data_to_sort = self.deltas_rms(params=params)
    else:
      raise AssertionError
  elif (by_value == "delta"):
    if proxy_type in (rigid_bond, rigu):
      data_to_sort = flex.abs(self.deltas(params=params))
    else:
      raise AssertionError
  else:
    raise AssertionError
  i_proxies_sorted = flex.sort_permutation(data=data_to_sort, reverse=True)
  if (max_items is not None):
    i_proxies_sorted = i_proxies_sorted[:max_items]
  item_label_blank = " " * len(item_label)
  print("%sSorted by %s:" % (prefix, by_value), file=f)
  dynamic_proxy_types = (
    adp_u_eq_similarity,
    adp_volume_similarity,
    )
  for i_proxy in i_proxies_sorted:
    proxy = self[i_proxy]
    s = item_label
    if proxy_type in (isotropic_adp, fixed_u_eq_adp):
      if (site_labels is None): l = str(proxy.i_seqs[0])
      else:                     l = site_labels[proxy.i_seqs[0]]
      print("%s%s %s" % (prefix, s, l), file=f)
      s = item_label_blank
    elif proxy_type in dynamic_proxy_types:
      restraint = proxy_type(params=params, proxy=proxy)
      restraint._show_sorted_item(f=f, prefix=prefix)
      print("%s         delta" % (prefix), file=f)
      for n, i_seq in enumerate(proxy.i_seqs):
        if (site_labels is None): l = str(i_seq)
        else:                     l = site_labels[i_seq]
        print("%s %s %9.2e" % (prefix, l, restraint.deltas()[n]), file=f)
    else:
      for n, i_seq in enumerate(proxy.i_seqs):
        if (site_labels is None): l = str(i_seq)
        else:                     l = site_labels[i_seq]
        print("%s%s %s" % (prefix, s, l), file=f)
        s = item_label_blank
    if proxy_type in (adp_similarity, isotropic_adp, fixed_u_eq_adp,\
                      rigid_bond, rigu):
      restraint = proxy_type(params=params, proxy=proxy)
      restraint._show_sorted_item(f=f, prefix=prefix)
    elif proxy_type in dynamic_proxy_types:
      pass
    else:
      raise AssertionError
  n_not_shown = self.size() - i_proxies_sorted.size()
  if (n_not_shown != 0):
    print(prefix + "... (remaining %d not shown)" % n_not_shown, file=f)
