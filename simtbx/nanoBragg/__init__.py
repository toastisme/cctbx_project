from __future__ import absolute_import, division, print_function
import boost_adaptbx.boost.python as bp
import cctbx.uctbx # possibly implicit
ext = bp.import_ext("simtbx_nanoBragg_ext")
from scitbx.array_family import flex
from simtbx_nanoBragg_ext import *
from scitbx.matrix import col, sqr

from dxtbx.imageset import MemReader
from dxtbx.imageset import ImageSet, ImageSetData
from dxtbx.model.experiment_list import Experiment, ExperimentList
from dxtbx.model import CrystalFactory
from dxtbx.model import BeamFactory
from dxtbx.model import DetectorFactory
from dxtbx.format import cbf_writer


@bp.inject_into(ext.nanoBragg)
class _():

  def __getattr__(self,name):
    """assemble miller array of structure factors used to compute spot intensities from the internal C cube array
       how do we specify docstrings for individual overriden members? """
    if name == "Fhkl":
      from cctbx.crystal import symmetry
      cs = symmetry(unit_cell = self.unit_cell_Adeg,space_group="P 1")
      from cctbx.miller import set, array
      indices,data = self.Fhkl_tuple
      mset = set(crystal_symmetry=cs, anomalous_flag=True, indices=indices)
      return array(mset, data=data).set_observation_type_xray_amplitude()

  def __setattr__(self,name,value):
    """use a P1 anomalous=True miller array to initialize the internal C cube array with structure factors for the spot intensities
       how do we specify docstrings for individual overriden members? """
    if name in ["Fhkl"]:
      value=value.expand_to_p1()
      value=value.generate_bijvoet_mates()
      assert value.space_group_info().type().lookup_symbol() == "P 1"
      # handle exception by expanding to P1
      assert value.anomalous_flag() == True
      # handle exception by copying all F(hkl) to F(-h-k-l)
      #assert values are amplitudes # not sure how to guarantee this
      self.unit_cell_Adeg = value.unit_cell()
      #self.mock_up_group = value.space_group()
      #self.mock_up_anomalous_flag = value.anomalous_flag()
      self.Fhkl_tuple = (value.indices(),value.data())
    else:
      super(ext.nanoBragg,self).__setattr__(name,value)

  def to_smv_format_py(self,fileout,intfile_scale=0.0,debug_x=-1,debug_y=-1,
    rotmat=False,extra=None,verbose=False,gz=False):

    byte_order = "little_endian";

    #recast the image file write to Python to afford extra options: rotmat, extra, gz
    if gz:
      from libtbx.smart_open import for_writing
      outfile = for_writing(file_name=fileout+".gz", gzip_mode="wb")
    else:
      outfile = open(fileout,"wb");

    outfile.write(("{\nHEADER_BYTES=1024;\nDIM=2;\nBYTE_ORDER=%s;\nTYPE=unsigned_short;\n"%byte_order).encode());
    outfile.write(b"SIZE1=%d;\nSIZE2=%d;\nPIXEL_SIZE=%g;\nDISTANCE=%g;\n"%(
      self.detpixels_fastslow[0],self.detpixels_fastslow[1],self.pixel_size_mm,self.distance_mm));
    outfile.write(b"WAVELENGTH=%g;\n"%self.wavelength_A);
    outfile.write(b"BEAM_CENTER_X=%g;\nBEAM_CENTER_Y=%g;\n"%self.beam_center_mm);
    outfile.write(b"ADXV_CENTER_X=%g;\nADXV_CENTER_Y=%g;\n"%self.adxv_beam_center_mm);
    outfile.write(b"MOSFLM_CENTER_X=%g;\nMOSFLM_CENTER_Y=%g;\n"%self.mosflm_beam_center_mm);
    outfile.write(b"DENZO_X_BEAM=%g;\nDENZO_Y_BEAM=%g;\n"%self.denzo_beam_center_mm);
    outfile.write(b"DIALS_ORIGIN=%g,%g,%g\n"%self.dials_origin_mm);
    outfile.write(b"XDS_ORGX=%g;\nXDS_ORGY=%g;\n"%self.XDS_ORGXY);
    outfile.write(b"CLOSE_DISTANCE=%g;\n"%self.close_distance_mm);
    outfile.write(b"PHI=%g;\nOSC_START=%g;\nOSC_RANGE=%g;\n"%(self.phi_deg,self.phi_deg,self.osc_deg));
    outfile.write(b"TIME=%g;\n"%self.exposure_s);
    outfile.write(b"TWOTHETA=%g;\n"%self.detector_twotheta_deg);
    outfile.write(b"DETECTOR_SN=000;\n");
    outfile.write(b"ADC_OFFSET=%g;\n"%self.adc_offset_adu);
    outfile.write(b"BEAMLINE=fake;\n");
    if rotmat:
      from scitbx.matrix import sqr
      RSABC = sqr(self.Amatrix).inverse().transpose()
      outfile.write( ("DIRECT_SPACE_ABC=%s;\n"%(",".join([repr(a) for a in RSABC.elems]))).encode() )
    if extra is not None:
      outfile.write(extra.encode())
    outfile.write(b"}\f");
    assert outfile.tell() < 1024, "SMV header too long, please edit this code and ask for more bytes."
    while ( outfile.tell() < 1024 ): outfile.write(b" ")
    from six import PY3
    if PY3:
      # Python3-compatible method for populating the output buffer.
      # Py2 implementation is more elegant in that the streambuf may be passed to C++,
      #   and the data are gzipped in chunks (default 1024). Py3 will not accept this method
      #   as it is PyString-based, with no converter mechanisms to bring data into PyBytes.
      # The Py3 method brings the full data in one chunk into PyBytes and then populates
      #   the output buffer in Python rather than C++.
      image_bytes = self.raw_pixels_unsigned_short_as_python_bytes(intfile_scale,debug_x,debug_y)
      ptr = 0; nbytes = len(image_bytes)
      while (ptr < nbytes): # chunked output necessary to prevent intermittent MemoryError
        outfile.write(image_bytes[ptr : min(ptr + 65536, nbytes)])
        ptr += 65536
      outfile.close();
      return
    from boost_adaptbx.boost.python import streambuf
    self.to_smv_format_streambuf(streambuf(outfile),intfile_scale,debug_x,debug_y)

    outfile.close();

  @property
  def beam(self):
    # Does this handle the conventions ? Im always confused about where the beam is pointing, whats s0 and whats beam_vector
    beam_dict = {'direction': self.beam_vector, #tuple([x for x in self.beam_vector]),  # TODO: is this correct?
                  'divergence': 0.0,  # TODO
                  'flux': self.flux,
                  'polarization_fraction': self.polarization,  #TODO
                  'polarization_normal': col(self.polar_vector).cross(col(self.beam_vector)),
                  'sigma_divergence': 0.0,  # TODO
                  'transmission': 1.0,  #TODO ?
                  'wavelength': self.wavelength_A}
    beam = BeamFactory.from_dict(beam_dict)
    return beam

  @property
  def crystal(self):
    crystal = None
    # dxtbx crystal description
    if self.Amatrix is not None:
      A = sqr(self.Amatrix).inverse().elems
      # is this always P-1 ?
      real_a = A[0], A[3], A[6]
      real_b = A[1], A[4], A[7]
      real_c = A[2], A[5], A[8]
      cryst_dict = {'__id__': 'crystal',
                     'real_space_a': real_a,
                     'real_space_b': real_b,
                     'real_space_c': real_c,
                     'space_group_hall_symbol': ' P 1'}
      crystal = CrystalFactory.from_dict(cryst_dict)
    return crystal

  @property
  def detector(self):
    # monolithic camera description
    pixsize = self.pixel_size_mm
    im_shape = self.detpixels_fastslow
    fdet = self.fdet_vector
    sdet = self.sdet_vector
    origin = self.dials_origin_mm
    det_descr = {'panels':
                   [{'fast_axis': fdet,
                     'slow_axis': sdet,
                     'gain': self.quantum_gain,
                     'identifier': '',
                     'image_size': im_shape,
                     'mask': [],
                     'material': '',  # TODO
                     'mu': 0.0,  # TODO
                     'name': 'Panel',
                     'origin': origin,
                     'pedestal': 0.0,
                     'pixel_size': (pixsize, pixsize),
                     'px_mm_strategy': {'type': 'SimplePxMmStrategy'},
                     'raw_image_offset': (0, 0),  # TODO
                     'thickness': 0.0,  # TODO
                     'trusted_range': (-1e3, 1e10),  # TODO
                     'type': ''}]}
    detector = DetectorFactory.from_dict(det_descr)
    return detector

  @property
  def imageset(self):
    format_class = FormatBraggInMemory(self.raw_pixels)
    reader = MemReaderNamedPath("virtual_Bragg_path", [format_class])
    reader.format_class = FormatBraggInMemory
    imageset_data = ImageSetData(reader, None)
    imageset = ImageSet(imageset_data)
    imageset.set_beam(self.beam)
    imageset.set_detector(self.detector)

    return imageset

  def as_explist(self):
    """
    return experiment list for simulated image
    """
    exp = Experiment()
    exp.crystal = self.crystal
    exp.beam = self.beam
    exp.detector = self.detector
    exp.imageset = self.imageset
    explist = ExperimentList()
    explist.append(exp)

    return explist

  def to_cbf(self, cbf_filename):
    writer = cbf_writer.FullCBFWriter(imageset=self.imageset)
    writer.write_cbf(cbf_filename, index=0)


def make_imageset(data, beam, detector):
  format_class = FormatBraggInMemoryMultiPanel(data)
  reader = MemReaderNamedPath("virtual_Bragg_path", [format_class])
  reader.format_class = FormatBraggInMemory
  imageset_data = ImageSetData(reader, None)
  imageset = ImageSet(imageset_data)
  imageset.set_beam(beam)
  imageset.set_detector(detector)
  return imageset


class FormatBraggInMemoryMultiPanel:

  def __init__(self, raw_pixels_lst):
    if not isinstance(raw_pixels_lst[0], flex.double):
      raw_pixels_lst = [flex.double(data) for data in raw_pixels_lst]
    self.raw_pixels_panels = tuple(raw_pixels_lst)
    panel_shape = self.raw_pixels_panels[0].focus()
    self.mask = tuple([flex.bool(flex.grid(panel_shape), True)]*len(self.raw_pixels_panels) )  # TODO: use nanoBragg internal mask

  def get_path(self, index):
    if index == 0:
      return "Virtual"
    else:
      raise ValueError("index must be 0 for format %s" % self.__name__)

  def get_raw_data(self):
    """
    return as a tuple, multi panel with 1 panel
    currently nanoBragg doesnt support simulating directly to a multi panel detector
    so this is the best we can do..
    """
    return self.raw_pixels_panels

  def get_mask(self, goniometer=None):
    """dummie place holder for mask, consider using internal nanoBragg mask"""
    return self.mask


class FormatBraggInMemory:

  def __init__(self, raw_pixels):
    self.raw_pixels = raw_pixels
    panel_shape = self.raw_pixels.focus()
    #self._filenames = ["InMemoryBraggPath"]  # TODO: CBFLib complains if no datablock path provided which comes from path
    self.mask = flex.bool(flex.grid(panel_shape), True)  # TODO: use nanoBragg internal mask

  def get_path(self, index):
    if index == 0:
      return "Virtual"
    else:
      raise ValueError("index must be 0 for format %s" % self.__name__)

  def get_raw_data(self):
    """
    return as a tuple, multi panel with 1 panel
    currently nanoBragg doesnt support simulating directly to a multi panel detector
    so this is the best we can do..
    """
    return self.raw_pixels,

  def get_mask(self, goniometer=None):
    """dummie place holder for mask, consider using internal nanoBragg mask"""
    return self.mask,

  #def paths(self):
  #  return ["InMemoryBraggPath"]  # TODO: CBFLib complains if no datablock path provided which comes from path

class MemReaderNamedPath(MemReader):

  def __init__(self, path,  *args, **kwargs):
    self.dummie_path_name = path
    super(MemReaderNamedPath, self).__init__(*args, **kwargs)

  def paths(self):
    """Necessary to have non zero string for CBFLib writer for some reason..."""
    return ["%s_%d" % (self.dummie_path_name, i) for i, _ in enumerate(self._images)]
