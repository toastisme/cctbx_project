from __future__ import absolute_import, division, print_function
import scitbx
from scitbx.matrix import col
import math

from collections import Iterable
from simtbx.diffBragg import diffBragg
from scitbx.array_family import flex
import numpy as np
from simtbx.nanoBragg import shapetype, nanoBragg
from simtbx.nanoBragg.nanoBragg_crystal import NBcrystal
from simtbx.nanoBragg.nanoBragg_beam import NBbeam
from copy import deepcopy
from scitbx.matrix import sqr
from simtbx.nanoBragg.tst_gaussian_mosaicity2 import run_uniform
from simtbx.nanoBragg.anisotropic_mosaicity import generate_Umats


def Amatrix_dials2nanoBragg(crystal):
  """
  returns the A matrix from a cctbx crystal object
  in nanoBragg format
  :param crystal: cctbx crystal
  :return: Amatrix as a tuple
  """
  sgi = crystal.get_space_group().info()
  cb_op = sgi.change_of_basis_op_to_primitive_setting()
  dtrm = sqr(cb_op.c().r().as_double()).determinant()
  if not dtrm == 1:
    raise ValueError('You need to convert your crystal model to its primitive setting first')
  Amatrix = sqr(crystal.get_A()).transpose()
  return Amatrix


def determine_spot_scale(beam_size_mm, crystal_thick_mm, mosaic_vol_mm3):
  """
  :param beam_size_mm:  diameter of beam focus (millimeter)
  :param crystal_thick_mm: thickness of crystal (millimeter)
  :param mosaic_vol_mm3:  volume of a mosaic block in crystal (cubic mm)
  :return: roughly the number of exposed mosaic blocks
  """
  if beam_size_mm <= crystal_thick_mm:
    illum_xtal_vol = crystal_thick_mm * beam_size_mm ** 2
  else:
    illum_xtal_vol = crystal_thick_mm ** 3
  return illum_xtal_vol / mosaic_vol_mm3


class SimData:
  def __init__(self):
    self.detector = SimData.simple_detector(180, 0.1, (512, 512))
    self.seed = 1
    self.crystal = NBcrystal()
    self.add_air = False
    self.Umats_method = 2
    self.add_water = True
    self.water_path_mm = 0.005
    self.air_path_mm = 0
    self.using_diffBragg_spots = False
    nbBeam = NBbeam()
    nbBeam.unit_s0 = (0, 0, -1)
    self.beam = nbBeam
    self.using_cuda = False
    self.using_omp = False
    self.rois = None
    self.readout_noise = 3
    self.gain = 1
    self.psf_fwhm = 0
    self.include_noise = True
    self.background_raw_pixels = None  # background raw pixels, should be a 2D flex double array
    self.backrground_scale = 1  # scale factor to apply to background raw pixels
    self.functionals = []
    self.mosaic_seeds = 777, 777
    self.D = None # nanoBragg instance
    self.panel_id = 0

  @property
  def background_raw_pixels(self):
    return self._background_raw_pixels

  @background_raw_pixels.setter
  def background_raw_pixels(self, val):
    self._background_raw_pixels = val

  @property
  def gain(self):
    return self._gain

  @gain.setter
  def gain(self, val):
    self._gain = val

  @staticmethod
  def default_panels_fast_slow(detector):
    Npanel = len(detector)
    nfast, nslow = detector[0].get_image_size()
    slows, fasts = np.indices((nslow, nfast))
    fasts = list(map(int, np.ravel(fasts)))
    slows = list(map(int, np.ravel(slows)))
    fasts = fasts*Npanel
    slows = slows*Npanel
    pids = []
    for pid in range(Npanel):
      pids += [pid]*(nfast*nslow)

    npix = nslow*nfast*Npanel
    panels_fasts_slows = np.zeros(npix*3, int)
    panels_fasts_slows[0::3] = pids
    panels_fasts_slows[1::3] = fasts
    panels_fasts_slows[2::3] = slows
    return flex.size_t(panels_fasts_slows)

  @property
  def air_path_mm(self):
    return self._air_path_mm

  @air_path_mm.setter
  def air_path_mm(self, val):
    self._air_path_mm = val

  @property
  def water_path_mm(self):
    return self._water_path_mm

  @water_path_mm.setter
  def water_path_mm(self, val):
    self._water_path_mm = val

  @property
  def crystal(self):
    return self._crystal

  @crystal.setter
  def crystal(self, val):
    self._crystal = val

  @property
  def beam(self):
    return self._beam

  @beam.setter
  def beam(self, val):
    self._beam = val

  @property
  def seed(self):
    return self._seed

  @seed.setter
  def seed(self, val):
    self._seed = val

  @staticmethod
  def Umats(mos_spread_deg, n_mos_doms=None, isotropic=True,
            seed=777, norm_dist_seed=777, method=0, angles_per_axis=10, num_axes=10,
            crystal=None):
    """

    :param mos_spread_deg: a float (if method is in [0,1,2], 3-tuple (if method is 3), or 6-tuple (if method is 4)
    :param n_mos_doms: for method 0 or 1, how many mosaic domains
    :param isotropic: only for method 0; for each rotation angle, also model its negative
    :param seed: random seed for method 0 axes
    :param norm_dist_seed: random seed for method 0 angles
    :param method: 0,1,2,3 or 4  TODO explain methods
    :param angles_per_axis: if method in [2,3,4] how many hemispher samples for rotation axes
    :param num_axes: if method in [2,3,4] how many rotation angles per axis on hemisphere
    :param crystal: if method in [3,4], crystal model for producing the anisptropic mosaicity model
    :return:  Umats, Umats_prime and Umats_dblprime (the derivatives will be None  depending on method)
    """
    UMAT_dblprime = None
    if method == 0:
      assert n_mos_doms is not None
      # this is the legacy method
      UMAT_nm = flex.mat3_double()
      mersenne_twister = flex.mersenne_twister(seed=seed)
      scitbx.random.set_random_seed(norm_dist_seed)
      rand_norm = scitbx.random.normal_distribution(mean=0, sigma=mos_spread_deg * np.pi / 180.)
      g = scitbx.random.variate(rand_norm)
      mosaic_rotation = g(n_mos_doms)
      for m in mosaic_rotation:
        site = col(mersenne_twister.random_double_point_on_sphere())
        if mos_spread_deg > 0:
          UMAT_nm.append(site.axis_and_angle_as_r3_rotation_matrix(m, deg=False))
        else:
          UMAT_nm.append(site.axis_and_angle_as_r3_rotation_matrix(0, deg=False))
        if isotropic and mos_spread_deg > 0:
          UMAT_nm.append(site.axis_and_angle_as_r3_rotation_matrix(-m, deg=False))
      UMAT_prime = None
    else:
      if method == 1:
        assert n_mos_doms is not None
        print("Setting Umats using method 1")
        if mos_spread_deg == 0 or n_mos_doms == 1:
          UMAT_nm = [(1, 0, 0, 0, 1, 0, 0, 0, 1)]
          UMAT_prime = [(0, 0, 0, 0, 0, 0, 0, 0, 0)]
        else:
          UMAT_nm, UMAT_prime = run_uniform(mos_spread_deg, n_mos_doms)
      elif method in [2, 3, 4]:
        if method in [3, 4] and crystal is None:
          raise ValueError("Crystal cannot be None for methods 3,4 which makes anisotropic mosaicity models")
        if mos_spread_deg == 0:
          UMAT_nm = [(1, 0, 0, 0, 1, 0, 0, 0, 1)]
          UMAT_prime = [(0, 0, 0, 0, 0, 0, 0, 0, 0)]
        else:
          if method == 2:  # isotropic eta
            print("Setting Umats using method 2")
            eta = mos_spread_deg
            eta_tensor = eta, 0, 0, 0, eta, 0, 0, 0, eta
            generation_method = 2
          elif method == 3:
            print("Setting Umats using method 3")
            if not isinstance(mos_spread_deg, Iterable):
              raise ValueError("for method 3 of Umats, mosaic_spread_def needs to be a 3-tuple")
            if len(mos_spread_deg) != 3:
              raise ValueError("for method 3 of Umats, mosaic_spread_def needs to be a 3-tuple")
            eta_a, eta_b, eta_c = mos_spread_deg
            eta_tensor = eta_a, 0, 0, 0, eta_b, 0, 0, 0, eta_c
            generation_method=1
          elif method==4:
            raise NotImplementedError("Not yet supporting full 6-parameter mosaic spread")
            if not isinstance(mos_spread_deg, Iterable):
              raise ValueError("for method 4 of Umats, mosaic_spread_def needs to be a 6-tuple")
            if len(mos_spread_deg) != 6:
              raise ValueError("for method 4 of Umats, mosaic_spread_def needs to be a 6-tuple")
            eta_a, eta_b, eta_c, eta_d, eta_e, eta_f = mos_spread_deg
            eta_tensor = eta_a, eta_d, eta_f, eta_d, eta_b, eta_e, eta_f, eta_e, eta_c
            generation_method=0

          UMAT_nm, UMAT_prime, UMAT_dblprime = generate_Umats(eta_tensor,
                                                              crystal=crystal,
                                                              compute_derivs=True,
                                                              plot=None,
                                                              num_random_samples=n_mos_doms,
                                                              how=generation_method)

    if UMAT_dblprime is not None:
      return UMAT_nm, UMAT_prime, UMAT_dblprime
    else:
      return UMAT_nm, UMAT_prime

  @property
  def Umats_method(self):
    return self._Umats_method

  @Umats_method.setter
  def Umats_method(self, val):
    if val not in [0, 1, 2, 3, 4]:
      raise ValueError("Umats method needs to be 0,1,2,3, or 4 (but 4 aint yet supported)")
    self._Umats_method = val

  @property
  def psf_fwhm(self):
    return self._psf_fwhm

  @psf_fwhm.setter
  def psf_fwhm(self, val):
    self._psf_fwhm = val

  @property
  def readout_noise(self):
    return self._readout_noise

  @readout_noise.setter
  def readout_noise(self, val):
    self._readout_noise = val

  @property
  def add_air(self):
    return self._add_air

  @add_air.setter
  def add_air(self, val):
    self._add_air = val

  @property
  def add_water(self):
    return self._add_water

  @add_water.setter
  def add_water(self, val):
    self._add_water = val

  @property
  def detector(self):
    return self._detector

  @detector.setter
  def detector(self, val):
    self._detector = val

  @property
  def rois(self):
    return self._rois

  @rois.setter
  def rois(self, val):
    self._rois = val

  @property
  def using_omp(self):
    return self._using_omp

  @using_omp.setter
  def using_omp(self, val):
    assert val in (True, False)
    self._using_omp = val

  @property
  def using_diffBragg_spots(self):
    return self._using_diffBragg_spots

  @using_diffBragg_spots.setter
  def using_diffBragg_spots(self, val):
    assert(val in [True, False])
    self._using_diffBragg_spots = val

  @property
  def using_cuda(self):
    return self._using_cuda

  @using_cuda.setter
  def using_cuda(self, val):
    assert(val in [True, False])
    self._using_cuda = val

  @property
  def include_noise(self):
    return self._include_noise

  @include_noise.setter
  def include_noise(self, val):
    self._include_noise = val

  def update_Fhkl_tuple(self):
    if self.crystal.miller_array is not None:
      if self.using_diffBragg_spots and self.crystal.miller_is_complex:
        Freal, Fimag = zip(*[(val.real, val.imag) for val in self.crystal.miller_array.data()])
        Freal = flex.double(Freal)
        Fimag = flex.double(Fimag)
        self.D.Fhkl_tuple = self.crystal.miller_array.indices(), Freal, Fimag
      else:
        self.D.Fhkl_tuple = self.crystal.miller_array.indices(), self.crystal.miller_array.data(), None

  def _crystal_properties(self):
    if self.crystal is None:
      return
    self.D.xtal_shape = self.crystal.xtal_shape

    self.update_Fhkl_tuple()

    ## TODO: am I unnecessary?
    #self.D.unit_cell_tuple = self.crystal.dxtbx_crystal.get_unit_cell().parameters()
    if self.using_diffBragg_spots:
      self.D.Omatrix = self.crystal.Omatrix
      self.D.Bmatrix = self.crystal.dxtbx_crystal.get_B() #
      self.D.Umatrix = self.crystal.dxtbx_crystal.get_U()
      if self.crystal.isotropic_ncells:
        self.D.Ncells_abc = self.crystal.Ncells_abc[0]
      else:
        self.D.Ncells_abc_aniso = self.crystal.Ncells_abc
      if self.crystal.Ncells_def is not None:
        self.D.Ncells_def = self.crystal.Ncells_def

      if self.crystal.anisotropic_mos_spread_deg is not None:
        mosaicity = self.crystal.anisotropic_mos_spread_deg
        self.Umats_method = 3 if 3 == len(mosaicity) else 4
        crystal=self.crystal.dxtbx_crystal
      else:
        mosaicity = self.crystal.mos_spread_deg
        self.Umats_method = 2
        crystal=None
      self.update_umats(mosaicity, self.crystal.n_mos_domains, crystal)

    else:
      self.D.xtal_shape = self.crystal.xtal_shape
      self.update_Fhkl_tuple()
      self.D.Amatrix = Amatrix_dials2nanoBragg(self.crystal.dxtbx_crystal)
      #Nabc = tuple([int(round(x)) for x in self.crystal.Ncells_abc])
      Nabc = self.crystal.Ncells_abc
      if len(Nabc) == 1:
        Nabc = Nabc[0], Nabc[0], Nabc[0]
      self.D.Ncells_abc = Nabc
      # TODO fix for anisotropic
      self.D.mosaic_spread_deg = self.crystal.mos_spread_deg
      self.D.mosaic_domains = self.crystal.n_mos_domains
      mos_blocks, _ = SimData.Umats(self.crystal.mos_spread_deg,
                                    self.crystal.n_mos_domains,
                                    seed=self.mosaic_seeds[0], norm_dist_seed=self.mosaic_seeds[1],
                                    angles_per_axis=self.crystal.mos_angles_per_axis, num_axes=self.crystal.num_mos_axes)
      self.D.set_mosaic_blocks(mos_blocks)

  def update_umats(self, mos_spread, mos_domains, crystal=None):
    #TODO remove arguments from this function as they are already in crystal attribute
    if not hasattr(self, "D"):
      print("Cannot set umats if diffBragg is not yet instantiated")
      return
    # TODO anisotropic case
    if isinstance(mos_spread, Iterable):
      # TODO does this matter ?
      ave_spread =  sum(mos_spread) / len(mos_spread)
      assert ave_spread > 0
      self.D.mosaic_spread_deg = ave_spread
      self.crystal.mos_spread_deg = ave_spread
      self.crystal.anisotropic_mosaic_spread_deg = mos_spread
      assert self.Umats_method in [3, 4]
      assert crystal is not None
      self.D.has_anisotropic_mosaic_spread = True
    else:
      self.D.mosaic_spread_deg = mos_spread
      self.crystal.mos_spread_deg = mos_spread
      self.crystal.anisotropic_mosaic_spread_deg = None
      assert self.Umats_method in [0, 1, 2]
      self.D.has_anisotropic_mosaic_spread = False

    self.D.mosaic_domains = mos_domains
    self.crystal.n_mos_domains = mos_domains
    Umats_data = SimData.Umats(mos_spread, mos_domains, method=self.Umats_method,
                               angles_per_axis=self.crystal.mos_angles_per_axis,
                               crystal=crystal,
                               num_axes=self.crystal.num_mos_axes)
    if len(Umats_data) == 2:
      Umats, Umats_prime = Umats_data
      Umats_dbl_prime = None
    else:
      assert len(Umats_data) == 3
      Umats, Umats_prime, Umats_dbl_prime = Umats_data

    self.D.set_mosaic_blocks(Umats)
    if self.Umats_method == 3 and Umats_prime is not None:
      Umats_prime = Umats_prime[0::3] + Umats_prime[1::3] + Umats_prime[2::3]
      assert len(Umats_prime) == 3*len(Umats)
      if Umats_dbl_prime is not None:
        Umats_dbl_prime = Umats_dbl_prime[0::3] + Umats_dbl_prime[1::3] + Umats_dbl_prime[2::3]
        assert len(Umats_dbl_prime) == 3*len(Umats)

    if Umats_prime is not None:
      self.D.set_mosaic_blocks_prime(Umats_prime)
    if Umats_dbl_prime is not None:
      print("Setting second derivatives")
      self.D.set_mosaic_blocks_dbl_prime(Umats_dbl_prime)

    # here we move the umats from the flex mat3 into vectors of Eigen:
    self.D.vectorize_umats()

  def _beam_properties(self):
    self.D.xray_beams = self.beam.xray_beams
    self.D.beamsize_mm = self.beam.size_mm

  def _seedlings(self):
    self.D.seed = self.seed
    self.D.calib_seed = self.seed
    self.D.mosaic_seed = self.seed

  def determine_spot_scale(self):
    if self.crystal is None:
      return 1
    if self.beam.size_mm <= self.crystal.thick_mm:
      illum_xtal_vol = self.crystal.thick_mm * self.beam.size_mm**2
    else:
      illum_xtal_vol = self.crystal.thick_mm**3
    mosaic_vol = self.D.xtal_size_mm[0]*self.D.xtal_size_mm[1]*self.D.xtal_size_mm[2]
    return illum_xtal_vol / mosaic_vol

  def update_nanoBragg_instance(self, parameter, value):
    setattr(self.D, parameter, value)

  @property
  def panel_id(self):
    return self._panel_id

  @panel_id.setter
  def panel_id(self, val):
    if val >= len(self.detector):
      raise ValueError("panel id cannot be larger than the number of panels in detector (%d)" % len(self.detector))
    if val <0:
      raise ValueError("panel id cannot be negative!")
    if self.D is None:
      self._panel_id = 0
    else:
      self.D.set_dxtbx_detector_panel(self.detector[int(val)], self.beam.nanoBragg_constructor_beam.get_s0())
      self._panel_id = int(val)

  def instantiate_nanoBragg(self, verbose=0, oversample=0, device_Id=0, adc_offset=0,
                            default_F=1e3, interpolate=0):

    self.instantiate_diffBragg(verbose=verbose, oversample=oversample, device_Id=device_Id,
                               adc_offset=adc_offset, default_F=default_F, interpolate=interpolate,
                               use_diffBragg=False)

  def instantiate_diffBragg(self, verbose=0, oversample=0, device_Id=0,
                            adc_offset=0, default_F=1e3, interpolate=0, use_diffBragg=True,
                            auto_set_spotscale=False):

    if not use_diffBragg:
      self.D = nanoBragg(self.detector, self.beam.nanoBragg_constructor_beam,
                         verbose=verbose, panel_id=int(self.panel_id))
    else:
      self.D = diffBragg(self.detector,
                         self.beam.nanoBragg_constructor_beam,
                         verbose)
    self.using_diffBragg_spots = use_diffBragg
    self._seedlings()
    self.D.interpolate = interpolate
    self._crystal_properties()
    self._beam_properties()
    if auto_set_spotscale:
      self.D.spot_scale = self.determine_spot_scale()
    self.D.adc_offset_adu = adc_offset
    self.D.default_F = default_F

    if oversample > 0:
      self.D.oversample = int(oversample)

    if self.using_cuda:
      self.D.device_Id = device_Id
    if not self.using_diffBragg_spots:
      self._full_roi = self.D.region_of_interest
    else:
      self.D.vectorize_umats()

  def generate_simulated_image(self, instantiate=False):
    if instantiate:
      self.instantiate_diffBragg()
    self._add_nanoBragg_spots()
    self._add_background()
    if self.include_noise:
      self._add_noise()
    return self.D.raw_pixels.as_numpy_array()

  def _add_nanoBragg_spots(self):
    if self.using_diffBragg_spots:
      self.D.add_diffBragg_spots()
    else:
      rois = self.rois
      if rois is None:
        rois = [self._full_roi]
      _rawpix = None  # cuda_add_spots doesnt add spots, it resets each time.. hence we need this
      for roi in rois:
        if len(roi)==4:
          roi = (roi[0], roi[1]), (roi[2],roi[3])
        self.D.region_of_interest = roi
        if self.using_cuda:
          self.D.add_nanoBragg_spots_cuda()
          if _rawpix is None and len(rois) > 1:
            _rawpix = deepcopy(self.D.raw_pixels)
          elif _rawpix is not None:
            _rawpix += self.D.raw_pixels

        elif self.using_omp:
          from boost_adaptbx.boost.python import streambuf  # will deposit printout into dummy StringIO as side effect
          from six.moves import StringIO
          self.D.progress_meter = False
          self.D.add_nanoBragg_spots_nks(streambuf(StringIO()))
        else:
          self.D.add_nanoBragg_spots()

      if self.using_cuda and _rawpix is not None:
        self.D.raw_pixels = _rawpix

  def _add_background(self):
    if self.background_raw_pixels is not None:
      self.D.raw_pixels += self.background_raw_pixels
    else:
      if self.add_water:
        print('add water %f mm' % self.water_path_mm)
        water_scatter = flex.vec2_double([
            (0, 2.57), (0.0365, 2.58), (0.07, 2.8), (0.12, 5), (0.162, 8), (0.18, 7.32), (0.2, 6.75),
           (0.216, 6.75), (0.236, 6.5), (0.28, 4.5), (0.3, 4.3), (0.345, 4.36), (0.436, 3.77), (0.5, 3.17)])
        self.D.Fbg_vs_stol = water_scatter
        self.D.amorphous_sample_thick_mm = self.water_path_mm
        self.D.amorphous_density_gcm3 = 1
        self.D.amorphous_molecular_weight_Da = 18
        self.D.add_background(1, 0)

    if self.add_air:
      print("add air %f mm" % self.air_path_mm)
      air_scatter = flex.vec2_double([(0, 14.1), (0.045, 13.5), (0.174, 8.35), (0.35, 4.78), (0.5, 4.22)])
      self.D.Fbg_vs_stol = air_scatter
      self.D.amorphous_sample_thick_mm = self.air_path_mm
      self.D.amorphous_density_gcm3 = 1.2e-3
      self.D.amorphous_sample_molecular_weight_Da = 28  # nitrogen = N2
      self.D.add_background(1, 0)

  def _add_noise(self):
    self.D.detector_psf_kernel_radius_pixels = 5
    self.D.detector_psf_type = shapetype.Unknown
    self.D.detector_psf_fwhm_mm = self.psf_fwhm
    self.D.readout_noise = self.readout_noise
    self.D.quantum_gain = self.gain
    self.D.add_noise()

  @staticmethod
  def simple_detector(detector_distance_mm, pixelsize_mm, image_shape,
                      fast=(1, 0, 0), slow=(0, -1, 0)):
    from dxtbx.model.detector import DetectorFactory
    import numpy as np
    trusted_range = 0, 2e14
    detsize_s = image_shape[0]*pixelsize_mm
    detsize_f = image_shape[1]*pixelsize_mm
    cent_s = (detsize_s + pixelsize_mm*2)/2.
    cent_f = (detsize_f + pixelsize_mm*2)/2.
    beam_axis = np.cross(fast, slow)
    origin = -np.array(fast)*cent_f - np.array(slow)*cent_s + beam_axis*detector_distance_mm

    return DetectorFactory.make_detector("", fast, slow, origin,
                                         (pixelsize_mm, pixelsize_mm), image_shape, trusted_range)


if __name__ == "__main__":
  S = SimData()
  img = S.generate_simulated_image()
  print ("Maximum pixel value: %.3g" % img.max())
  print ("Minimum pixel value: %.3g" % img.min())
