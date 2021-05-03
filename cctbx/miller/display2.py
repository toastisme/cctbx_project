
# TODO:
#  - cached scenes

from __future__ import absolute_import, division, print_function
from libtbx.utils import Sorry, to_str
from cctbx import miller
from cctbx import crystal
from cctbx.array_family import flex
import libtbx.phil
from scitbx import graphics_utils
from libtbx import object_oriented_patterns as oop # causes crash in easy_mp.multi_core_run
from math import sqrt
import math, traceback
import math
from six.moves import zip

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

class MplColorHelper:
  def __init__(self, cmap_name, start_val, stop_val):
    self.cmap_name = cmap_name
    self.cmap = plt.get_cmap(cmap_name)
    self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
    self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)
  def get_rgb(self, val):
    rgba = self.scalarMap.to_rgba(val)
    return rgba[0], rgba[1], rgba[2], rgba[3]


nanval = float('nan')
inanval = -42424242 # TODO: find a more robust way of indicating missing flex.int data



def generate_systematic_absences(array,
                                  expand_to_p1=False,
                                  generate_bijvoet_mates=False):
  from cctbx import crystal
  sg = array.space_group()
  base_symm = crystal.symmetry(
    unit_cell=array.unit_cell(),
    space_group=sg.build_derived_reflection_intensity_group(False))
  base_array = array.customized_copy(crystal_symmetry=base_symm)
  complete_set = base_array.complete_set().customized_copy(
    crystal_symmetry=array.crystal_symmetry())
  absence_array = complete_set.sys_absent_flags()
  if (generate_bijvoet_mates):
    absence_array = absence_array.generate_bijvoet_mates()
  if (expand_to_p1):
    absence_array = absence_array.expand_to_p1().customized_copy(
      crystal_symmetry=array.crystal_symmetry())
      #niggli_cell().expand_to_p1()
  return absence_array


def nth_power_scale(dataarray, nth_power):
  """
  set nth_power to appropriate number between 0 and 1 for dampening the
  difference between the smallest and the largest values.
  If nth_power < 0 then an automatic value is computed that maps the smallest
  values to 0.1 of the largest values
  """
  absdat = flex.abs(dataarray).as_double()
  absdat2 = graphics_utils.NoNansArray(absdat) # much faster than flex.double([e for e in absdat if not math.isnan(e)])
  maxdat = flex.max(absdat2)
  mindat = max(1e-10*maxdat, flex.min(absdat2) )
  # only autoscale for sensible values of maxdat and mindat
  if nth_power < 0.0 and maxdat > mindat : # amounts to automatic scale
    nth_power = math.log(0.2)/(math.log(mindat) - math.log(maxdat))
  datascaled = flex.pow(absdat, nth_power)
  return datascaled, nth_power


def ExtendMillerArray(miller_array, nsize, indices=None ):
  millarray = miller_array.deep_copy()
  if indices:
    assert (indices.size()==nsize)
    millarray._indices.extend( indices )
  if millarray.sigmas() is not None:
    millarray._sigmas.extend( flex.double(nsize, nanval) )
  millarray._data = ExtendAnyData(millarray._data, nsize)
  return millarray


def ExtendAnyData(data, nsize):
  if isinstance(data, flex.bool):
    # flex.bool cannot be extended with NaN so cast the data to ints and extend with inanval instead
    data = data.as_int().extend( flex.int(nsize, inanval) )
  if isinstance(data, flex.hendrickson_lattman):
    data.extend( flex.hendrickson_lattman(nsize, (nanval, nanval, nanval, nanval)) )
  if isinstance(data, flex.int) or isinstance(data, flex.long) \
        or isinstance(data, flex.size_t):
    data.extend( flex.int(nsize, inanval) )
  if isinstance(data, flex.float) or isinstance(data, flex.double):
    data.extend( flex.double(nsize, nanval) )
  if isinstance(data, flex.complex_double):
    data.extend( flex.complex_double(nsize, nanval) )
  if isinstance(data, flex.vec3_double):
    data.extend( flex.vec3_double(nsize, (1.,1.,1.)) )
  return data


def MergeData(array, show_anomalous_pairs=False):
  if show_anomalous_pairs:
    merge = array.merge_equivalents()
    multiplicities = merge.redundancies()
    asu, matches = multiplicities.match_bijvoet_mates()
    mult_plus, mult_minus = multiplicities.hemispheres_acentrics()
    anom_mult = flex.int(
      min(p, m) for (p, m) in zip(mult_plus.data(), mult_minus.data()))
    #flex.min_max_mean_double(anom_mult.as_double()).show()
    anomalous_multiplicities = miller.array(
      miller.set(asu.crystal_symmetry(),
                 mult_plus.indices(),
                 anomalous_flag=False), anom_mult)
    anomalous_multiplicities = anomalous_multiplicities.select(
      anomalous_multiplicities.data() > 0)

    array = anomalous_multiplicities
    multiplicities = anomalous_multiplicities
  else:
    merge = array.merge_equivalents()
    array = merge.array()
    multiplicities = merge.redundancies()
  return array, multiplicities, merge


class scene(object):
  """
  Data for visualizing a Miller array graphically, either as a 3D view or
  a 2D zone of reciprocal space.  Currently used in phenix.data_viewer, but
  easily extensible to any other graphics platform (e.g. a PNG embedded in
  a web page).
  """
  def __init__(self, miller_array, settings, merge=None, foms_array=None,
   fullprocessarray=True):
    self.miller_array = miller_array
    self.renderscale = 100.0
    self.foms_workarray = foms_array
    self.SceneCreated = False
    if isinstance(miller_array.data(), flex.std_string):
      return
    self.settings = settings
    self.merge_equivalents = False
    if not self.miller_array.is_unique_set_under_symmetry():
      self.merge_equivalents = merge
    self.multiplicities = None
    self.fomlabel = ""
    self.foms = flex.double(self.miller_array.size(), float('nan'))
    self._is_using_foms = False
    self.fullprocessarray = fullprocessarray
    if self.miller_array.is_complex_array():
      # Colour map coefficient as a circular rainbow with saturation as a function of FOMs
      # process the foms miller array and store the foms data for later use when computing colours
      if foms_array:
        assert ( self.miller_array.size() == foms_array.size() )
        self.foms_workarray, dummy = self.process_input_array(foms_array)
        if not self.foms_workarray:
          return
        self.foms = self.foms_workarray.data()
        self.fomlabel = foms_array.info().label_string()
        self._is_using_foms = True
    self.work_array, self.multiplicities = self.process_input_array(self.miller_array)
    if not self.work_array:
      return
    array = self.work_array
    uc = array.unit_cell()
    self.unit_cell = uc
    self.slice_selection = None
    self.axis_index = None
    if (settings.slice_mode):
      self.axis_index = ["h","k","l"].index(self.settings.slice_axis)
      self.slice_selection = miller.simple_slice(
        indices=array.indices(),
        slice_axis=self.axis_index,
        slice_index=settings.slice_index)
      #if (self.slice_selection.count(True) == 0):
        #raise ValueError("No data selected!")
    index_span = array.index_span()
    self.colourlabel = self.miller_array.info().label_string()
    self.d_min = array.d_min()
    self.min_dist = 0.0
    self.nth_power_scale_radii = settings.nth_power_scale_radii
    self.hkl_range = index_span.abs_range()
    self.axes = [ uc.reciprocal_space_vector((self.hkl_range[0],0,0)),
                  uc.reciprocal_space_vector((0,self.hkl_range[1],0)),
                  uc.reciprocal_space_vector((0,0,self.hkl_range[2])) ]
    self.generate_view_data()
    if (self.slice_selection is not None):
      self.indices = self.work_array.indices().select(self.slice_selection).deep_copy()
      self.data = self.data.select(self.slice_selection)
      self.phases = self.phases.select(self.slice_selection)
      self.radians = self.radians.select(self.slice_selection)
      self.ampl = self.ampl.select(self.slice_selection)
      if self.sigmas:
        self.sigmas = self.sigmas.select(self.slice_selection)
      if foms_array:
        self.foms = self.foms.select(self.slice_selection)
    else :
      self.indices = array.indices()
    self.points = uc.reciprocal_space_vector(self.indices) * self.renderscale
    n_points = self.points.size()
    if not fullprocessarray:
      self.radii = flex.double()
      self.radii = ExtendAnyData(self.radii, n_points)
      self.colors = flex.vec3_double()
      self.colors = ExtendAnyData(self.colors, n_points)
    self.missing_flags = flex.bool(self.radii.size(), False)
    self.sys_absent_flags = flex.bool(self.radii.size(), False)
    if (settings.show_missing):
      self.generate_missing_reflections()
    if (settings.show_systematic_absences) and (not settings.show_only_missing):
      self.generate_systematic_absences()
    n_points = self.points.size()
    assert (self.colors.size() == n_points)
    assert (self.indices.size() == n_points)
    assert (self.radii.size() == n_points)
    assert (self.missing_flags.size() == n_points)
    assert (self.sys_absent_flags.size() == n_points)
    assert (self.data.size() == n_points)
    assert (self.phases.size() == n_points)
    assert (self.radians.size() == n_points)
    assert (self.ampl.size() == n_points)
    if self.sigmas:
      assert (self.sigmas.size() == n_points)
    if foms_array:
      assert (self.foms.size() == n_points)
    else:
      self.foms = flex.double(n_points, float('nan'))
    self.dres = uc.d(self.indices )
    self.clear_labels()
    self.SceneCreated = True


  def process_input_array(self, arr):
    array = arr.deep_copy()
    work_array = arr
    multiplicities = None
    try:
      if self.merge_equivalents :
        array, multiplicities, merge = MergeData(array, self.settings.show_anomalous_pairs)
      settings = self.settings
      data = array.data()
      #import code, traceback; code.interact(local=locals(), banner="".join( traceback.format_stack(limit=10) ) )
      self.missing_set = oop.null()
      #if (array.is_xray_intensity_array()):
      #  data.set_selected(data < 0, flex.double(data.size(), 0.))
      if (array.is_unique_set_under_symmetry()) and (settings.map_to_asu):
        array = array.map_to_asu()
        if (multiplicities is not None):
          multiplicities = multiplicities.map_to_asu()

      if (settings.d_min is not None):
        array = array.resolution_filter(d_min=settings.d_min)
        if (multiplicities is not None):
          multiplicities = multiplicities.resolution_filter(
            d_min=settings.d_min)
      self.filtered_array = array.deep_copy()
      if (settings.expand_anomalous):
        if not array.is_unique_set_under_symmetry():
          raise Sorry("Error! Cannot generate bijvoet mates of unmerged reflections.")
        array = array.generate_bijvoet_mates()
        original_symmetry = array.crystal_symmetry()

        if (multiplicities is not None):
          multiplicities = multiplicities.generate_bijvoet_mates()
      if (self.settings.show_missing):
        self.missing_set = array.complete_set().lone_set(array)
        if self.settings.show_anomalous_pairs:
          self.missing_set = self.missing_set.select(
            self.missing_set.centric_flags().data(), negate=True)
      if (settings.expand_to_p1):
        if not array.is_unique_set_under_symmetry():
          raise Sorry("Error! Cannot expand unmerged reflections to P1.")
        original_symmetry = array.crystal_symmetry()
        array = array.expand_to_p1().customized_copy(
          crystal_symmetry=original_symmetry)
        #array = array.niggli_cell().expand_to_p1()
        #self.missing_set = self.missing_set.niggli_cell().expand_to_p1()
        self.missing_set = self.missing_set.expand_to_p1().customized_copy(
          crystal_symmetry=original_symmetry)
        if (multiplicities is not None):
          multiplicities = multiplicities.expand_to_p1().customized_copy(
              crystal_symmetry=original_symmetry)
      data = array.data()
      self.r_free_mode = False
      self.phases = flex.double(data.size(), float('nan'))
      self.radians = flex.double(data.size(), float('nan'))
      self.ampl = flex.double(data.size(), float('nan'))
      self.sigmas = None
      self.singletonsiness = flex.double(data.size(), 0.0 )
      if isinstance(data, flex.bool):
        self.r_free_mode = True
        data_as_float = flex.double(data.size(), 0.0)
        data_as_float.set_selected(data==True, flex.double(data.size(), 1.0))
        data = data_as_float
        self.data = data #.deep_copy()
      else :
        if isinstance(data, flex.double):
          self.data = data #.deep_copy()
        elif isinstance(data, flex.hendrickson_lattman):
          # for now display HL coefficients as a simple sum
          self.data = flex.double( [a+b+c+d for a,b,c,d in data] )
        elif isinstance(data, flex.complex_double):
          self.data = data #.deep_copy()
          self.ampl = flex.abs(data)
          self.phases = flex.arg(data) * 180.0/math.pi
          # purge nan values from array to avoid crash in fmod_positive()
          #b = flex.bool([bool(math.isnan(e)) for e in self.phases])
          b = graphics_utils.IsNansArray( self.phases )
          # replace the nan values with an arbitrary float value
          self.phases = self.phases.set_selected(b, 42.4242)
          # Cast negative degrees to equivalent positive degrees
          self.phases = flex.fmod_positive(self.phases, 360.0)
          self.radians = flex.arg(data)
          # replace the nan values with an arbitrary float value
          self.radians = self.radians.set_selected(b, 0.424242)
        elif hasattr(array.data(), "as_double"):
          self.data = data
        else:
          raise RuntimeError("Unexpected data type: %r" % data)
        if array.sigmas() is not None:
          self.sigmas = array.sigmas()
        else:
          self.sigmas = None
      if array.anomalous_flag() and array.is_unique_set_under_symmetry():
        # Label singletons with 1 or -1 if any. True singletons are only acentric
        self.singletonsiness.set_selected( array.select_acentric().match_bijvoet_mates()[1].singles("+"), 1.0 )
        self.singletonsiness.set_selected( array.select_acentric().match_bijvoet_mates()[1].singles("-"), -1.0 )
      work_array = array
    except Exception as e:
      print(to_str(e) + "".join(traceback.format_stack(limit=10)))
      raise e
      return None, None
    work_array.set_info(arr.info() )
    multiplicities = multiplicities
    return work_array, multiplicities


  def generate_view_data(self):
    from scitbx.array_family import flex
    #from scitbx import graphics_utils
    settings = self.settings
    data_for_colors = data_for_radii = None
    if not self.fullprocessarray:
      return
    data = self.data #self.work_array.data()
    sigmas = self.sigmas
    if (isinstance(data, flex.double) and data.all_eq(0)):
      data = flex.double(data.size(), 1)
    if ((self.multiplicities is not None) and
        (settings.scale_colors_multiplicity)):
      data_for_colors = self.multiplicities.data().as_double()
      assert data_for_colors.size() == data.size()
    elif isinstance(data, flex.complex_double):
      data_for_colors = self.phases
      foms_for_colours = self.foms
       # assuming last part of the labels indicates the phase label as in ["FCALC","PHICALC"]
      self.colourlabel = "Phase of " + self.miller_array.info().label_string()
    elif (settings.sigma_color) and sigmas is not None:
      data_for_colors = sigmas.as_double()
      self.colourlabel = "Sigma of " + self.miller_array.info().label_string()
    else :
      data_for_colors = flex.abs(data.deep_copy())
    uc = self.work_array.unit_cell()
    self.min_dist = min(uc.reciprocal_space_vector((1,1,1))) * self.renderscale
    min_radius = 0.05 * self.min_dist
    max_radius = 0.5 * self.min_dist
    if ((self.multiplicities is not None) and
        (settings.scale_radii_multiplicity)):
      data_for_radii = self.multiplicities.data().as_double()
      if (settings.sigma_radius) and sigmas is not None:
        data_for_radii = sigmas * self.multiplicities.as_double()
      assert data_for_radii.size() == data.size()
    elif (settings.sigma_radius) and sigmas is not None:
      data_for_radii, self.nth_power_scale_radii = nth_power_scale(flex.abs(sigmas.as_double().deep_copy()),
                                       settings.nth_power_scale_radii)
    else :
      data_for_radii, self.nth_power_scale_radii = nth_power_scale(flex.abs(data.deep_copy()),
                                       settings.nth_power_scale_radii)
    if (settings.slice_mode):
      data = data.select(self.slice_selection)
      if (not settings.keep_constant_scale):
        data_for_radii = data_for_radii.select(self.slice_selection)
        data_for_colors = data_for_colors.select(self.slice_selection)
        foms_for_colours = foms_for_colours.select(self.slice_selection)
    # Computing rgb colours of each reflection is slow so make a small array
    # of precomputed colours to use as a lookup table for each reflection
    if isinstance(data, flex.complex_double):
      COL = MplColorHelper(settings.color_scheme, 0, 360)
      rgbcolarray = [ COL.get_rgb(d)[0:3] for d in range(360) ]
      if self.isUsingFOMs():
        colors = graphics_utils.map_to_rgb_colourmap(
          data_for_colors=data_for_colors,
          colormap=rgbcolarray,
          selection=flex.bool(data_for_colors.size(), True),
          attenuation = foms_for_colours,
          map_directly=True,
          color_all=False
          )
      else:
        colors = graphics_utils.map_to_rgb_colourmap(
          data_for_colors=data_for_colors,
          colormap=rgbcolarray,
          selection=flex.bool(data_for_colors.size(), True),
          attenuation = None,
          map_directly=True,
          color_all=False
          )
    else:
      # Use a colour gradient from matplotlib
      COL = MplColorHelper(settings.color_scheme, 0, 199)
      colorgradientarray = flex.vec3_double([COL.get_rgb(d)[0:3] for d in range(200) ])
      # Do the table lookup in C++ for speed improvement
      colors = graphics_utils.map_to_rgb_colourmap(
        data_for_colors=data_for_colors,
        colormap=colorgradientarray,
        selection=flex.bool(data_for_colors.size(), True),
        powscale = settings.color_powscale,
        attenuation = None,
        color_all=False
        )

    if (settings.slice_mode) and (settings.keep_constant_scale):
      colors = colors.select(self.slice_selection)
      data_for_radii = data_for_radii.select(self.slice_selection)
    #if (settings.sqrt_scale_radii) and (not settings.scale_radii_multiplicity):
    #  data_for_radii = flex.sqrt(flex.abs(data_for_radii))
    if len(data_for_radii):
      #dat2 = flex.abs(flex.double([e for e in data_for_radii if not math.isnan(e)]))
      dat2 = flex.abs(flex.double( graphics_utils.NoNansArray( data_for_radii, 0.1 ) ))
      # don't divide by 0 if dealing with selection of Rfree array where all values happen to be zero
      scale = max_radius/(flex.max(dat2) + 0.001)
      radii = data_for_radii * (self.settings.scale * scale)
      assert radii.size() == colors.size()
    else:
      radii = flex.double()
      max_radius = 0
    self.radii = radii
    self.max_radius = max_radius
    self.min_radius = min_radius
    self.colors = colors
    if isinstance(data, flex.complex_double):
      self.foms = foms_for_colours
    #print(min_dist, min_radius, max_radius, flex.min(data_for_radii), flex.max(data_for_radii), scale)


  def isUsingFOMs(self):
    #return len([e for e in self.foms if math.isnan(e)]) != self.foms.size()
    return self._is_using_foms


  def ExtendData(self, nextent):
    self.data = ExtendAnyData(self.data, nextent )
    self.phases = ExtendAnyData(self.phases, nextent )
    self.ampl = ExtendAnyData(self.ampl, nextent )
    self.foms = ExtendAnyData(self.foms, nextent )
    self.radians = ExtendAnyData(self.radians, nextent )
    if self.sigmas is not None:
      self.sigmas = ExtendAnyData(self.sigmas, nextent)


  def generate_missing_reflections(self):
    from cctbx import miller
    from cctbx.array_family import flex
    settings = self.settings
    array = self.work_array
    uc = array.unit_cell()
    if (settings.show_only_missing):
      self.colors = flex.vec3_double()
      self.points = flex.vec3_double()
      self.radii = flex.double()
      self.missing_flags = flex.bool()
      self.indices = flex.miller_index()
      self.data = flex.double()
      self.phases = flex.double()
      self.ampl = flex.double()
      self.foms = flex.double()
      self.radians = flex.double()
      if self.sigmas is not None:
        self.sigmas = flex.double()
      self.sys_absent_flags = flex.bool()
    if (settings.slice_mode):
      slice_selection = miller.simple_slice(
        indices=self.missing_set.indices(),
        slice_axis=self.axis_index,
        slice_index=settings.slice_index)
      missing = self.missing_set.select(slice_selection).indices()
    else :
      missing = self.missing_set.indices()
    n_missing = missing.size()
    if (n_missing > 0):
      points_missing = uc.reciprocal_space_vector(missing) * 100.
      self.points.extend(points_missing)
      if (settings.color_scheme == "heatmap"):
        self.colors.extend(flex.vec3_double(n_missing, (0.,1.,0.)))
      elif (not settings.color_scheme in ["rainbow","redblue"]):
        self.colors.extend(flex.vec3_double(n_missing, (1.,0,0)))
      else :
        self.colors.extend(flex.vec3_double(n_missing, (1.,1.,1.)))
      self.radii.extend(flex.double(n_missing, self.settings.scale * self.max_radius/2.0 ))
      self.missing_flags.extend(flex.bool(n_missing, True))
      self.indices.extend(missing)
      self.ExtendData(n_missing)
      self.sys_absent_flags.extend(flex.bool(n_missing, False))


  def generate_systematic_absences(self):
    from cctbx import miller
    from cctbx import crystal
    from cctbx.array_family import flex
    settings = self.settings
    array = self.filtered_array # XXX use original array here!
    absence_array = generate_systematic_absences(
      array=self.filtered_array,
      expand_to_p1=settings.expand_to_p1,
      generate_bijvoet_mates=settings.expand_anomalous)
    if (settings.slice_mode):
      slice_selection = miller.simple_slice(
        indices=absence_array.indices(),
        slice_axis=self.axis_index,
        slice_index=settings.slice_index)
      absence_array = absence_array.select(slice_selection)
    absence_flags = absence_array.data()
    if (absence_flags.count(True) == 0):
      print("No systematic absences found!")
    else :
      new_indices = flex.miller_index()
      for i_seq in absence_flags.iselection():
        hkl = absence_array.indices()[i_seq]
        #if (hkl in self.indices):
        #  j_seq = self.indices.index(hkl)
        #  self.colors[j_seq] = (1.0, 0.5, 1.0)
        #  self.radii[j_seq] = self.max_radius
        #  self.missing_flags[j_seq] = False
        #  self.sys_absent_flags[j_seq] = True
        #else :
        if hkl in self.indices:
          print("Systematically absent reflection %s is unexpectedly present in %s" %(hkl, array.info().label_string()) )
        else:
          new_indices.append(hkl)
      if (new_indices.size() > 0):
        uc = self.work_array.unit_cell()
        points = uc.reciprocal_space_vector(new_indices) * 100
        self.points.extend(points)
        n_sys_absent = new_indices.size()
        self.radii.extend(flex.double(new_indices.size(), self.max_radius/2.0))
        self.indices.extend(new_indices)
        self.missing_flags.extend(flex.bool(new_indices.size(), False))
        self.sys_absent_flags.extend(flex.bool(new_indices.size(), True))
        self.ExtendData(n_sys_absent)
        if (settings.color_scheme == "redblue"):
          self.colors.extend(flex.vec3_double(new_indices.size(), (1.,1.0,0.)))
        else :
          self.colors.extend(flex.vec3_double(new_indices.size(), (1.,0.5,1.)))


  def clear_labels(self):
    self.label_points = set([])


  def get_resolution_at_point(self, k):
    hkl = self.indices[k]
    return self.unit_cell.d(hkl)


  def get_reflection_info(self, k):
    hkl = self.indices[k]
    d_min = self.unit_cell.d(hkl)
    if (self.missing_flags[k]) or (self.sys_absent_flags[k]):
      value = None
    else :
      value = self.data[k]
    return (hkl, d_min, value)


# list of all colour maps from https://matplotlib.org/examples/color/colormaps_reference.html
colormaps = " ".join(plt.colormaps())
# set the default selected colour map to "brg" for phil attribute "color_scheme" below
colormaps = colormaps.replace("brg ", "*brg ")

philstr = """
  data = None
    .type = path
    .optional = False
  labels = None
    .type = str
  symmetry_file = None
    .type = str
  black_background = True
    .type = bool
  show_axes = True
    .type = bool
  nth_power_scale_radii = 0.0
    .type = float
  scale = 1
    .type = float
  #phase_color = False
  #  .type = bool
  sigma_color = False
    .type = bool
  sigma_radius = False
    .type = bool
  expand_to_p1 = False
    .type = bool
  expand_anomalous = False
    .type = bool
  inbrowser = True
    .type = bool
  show_missing = False
    .type = bool
  show_only_missing = False
    .type = bool
  show_systematic_absences = False
    .type = bool
  slice_mode = False
    .type = bool
  keep_constant_scale = True
    .type = bool
  slice_axis = *h k l
    .type = choice
  slice_index = 0
    .type = int
  color_scheme = %s
    .type = choice
  color_powscale = 1.0
    .type = float
  show_labels = True
    .type = bool
  uniform_size = False
    .type = bool
  d_min = None
    .type = float
  map_to_asu = False
    .type = bool
  scale_radii_multiplicity = False
    .type = bool
  scale_colors_multiplicity = False
    .type = bool
  show_anomalous_pairs = False
    .type = bool
""" %colormaps

master_phil = libtbx.phil.parse( philstr )
params = master_phil.fetch().extract()

def reset_settings():
  """
  Reset all display settings to their default values as specified in the phil definition string
  """
  global master_phil
  global philstr
  global params
  params = master_phil.fetch(source = libtbx.phil.parse( philstr) ).extract()


def settings():
  """
  Get a global phil parameters object containing the display settings
  """
  global params
  return params
