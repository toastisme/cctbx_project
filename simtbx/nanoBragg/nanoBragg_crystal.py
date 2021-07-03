"""
organizer for setting the nanoBragg crystal properties
"""
from __future__ import absolute_import, division, print_function
from collections import Iterable
from simtbx.nanoBragg import shapetype
from scitbx.matrix import sqr
from cctbx import sgtbx


class NBcrystal(object):

    def __init__(self):
        ucell = (79.1, 79.1, 38.4, 90, 90, 90)
        self.xtal_shape = "gauss"  # shapetype.Gauss
        self.Ncells_abc = (10, 10, 10)
        self.Ncells_def = None
        self.mos_angles_per_axis = 10
        self.num_mos_axes = 10
        self.mos_spread_deg = 0
        self.anisotropic_mos_spread_deg = None
        self.n_mos_domains = 1
        self.thick_mm = 0.1
        self.symbol = "P43212"
        self.miller_array = NBcrystal.dummie_Fhkl(ucell, self.symbol)
        self.isotropic_ncells = True
        self.dxtbx_crystal = NBcrystal.dxtbx_crystal_from_ucell_and_symbol(
            ucell_tuple_Adeg=ucell, symbol=self.symbol)

    @property
    def has_anisotropic_mosaicity(self):
        return self.anisotropic_mos_spread_deg is not None

    @property
    def space_group_info(self):
        info = sgtbx.space_group_info(symbol=self.symbol)
        return info

    @property
    def miller_array_high_symmetry(self):
        return self.miller_array.customized_copy(space_group_info=self.space_group_info)

    @property
    def symbol(self):
        return self._symbol

    @symbol.setter
    def symbol(self, val):
        self._symbol = val

    @property
    def Omatrix(self):
        """
        Change of basis operator
        """
        sgi = self.dxtbx_crystal.get_space_group().info()
        to_p1 = sgi.change_of_basis_op_to_primitive_setting()
        return sqr(to_p1.c_inv().r().transpose().as_double())

    @property
    def dxtbx_crystal(self):
        return self._dxtbx_crystal

    @dxtbx_crystal.setter
    def dxtbx_crystal(self, val):
        self._dxtbx_crystal = val

    @property
    def miller_array(self):
        return self._miller_array

    @miller_array.setter
    def miller_array(self, val):
        if isinstance(val.data()[0], complex):
            self.miller_is_complex = True
        else:
            self.miller_is_complex = False
            if str(val.observation_type) == "xray.intensity":
                val = val.as_amplitude_array()
        cb_op = val.space_group_info().change_of_basis_op_to_primitive_setting()
        val = val.expand_to_p1()
        val = val.generate_bijvoet_mates()
        dtrm = sqr(cb_op.c().r().as_double()).determinant()
        if not dtrm == 1:
            val = val.change_basis(cb_op)
        self._miller_array = val

    @property
    def Ncells_abc(self):
        return self._Ncells_abc

    @Ncells_abc.setter
    def Ncells_abc(self, val):
        self._Ncells_abc = val

    @property
    def Ncells_def(self):
        return self._Ncells_def

    @Ncells_def.setter
    def Ncells_def(self, val):
        self._Ncells_def = val

    @property
    def anisotropic_mos_spread_deg(self):
        return self._anisotropic_mos_spread_deg

    @anisotropic_mos_spread_deg.setter
    def anisotropic_mos_spread_deg(self, val):
        if val is not None:
            if not isinstance(val, Iterable):
                raise TypeError("anisotropic_mos_spread_deg needs top be a 3-tuple or 6-tuple")
            elif len(val) not in [3, 6]:
                raise ValueError("Anisotropic mosaicity should be either a 3-tuple or a 6-tuple")
        self._anisotropic_mos_spread_deg = val

    @property
    def mos_spread_deg(self):
        return self._mos_spread_deg

    @mos_spread_deg.setter
    def mos_spread_deg(self, val):
        self._mos_spread_deg = val

    @property
    def n_mos_domains(self):
        return self._n_mos_domains

    @n_mos_domains.setter
    def n_mos_domains(self, val):
        self._n_mos_domains = val

    @property
    def xtal_shape(self):
      if self._xtal_shape == "gauss":
        return shapetype.Gauss
      elif self._xtal_shape == "gauss_argchk":
        return shapetype.Gauss_argchk
      elif self._xtal_shape == "round":
        return shapetype.Round
      elif self._xtal_shape == "square":
        return shapetype.Square
      else:
        return shapetype.Tophat

    @xtal_shape.setter
    def xtal_shape(self, val):
        self._xtal_shape = val

    @property
    def thick_mm(self):
        return self._thick_mm

    @thick_mm.setter
    def thick_mm(self, val):
        self._thick_mm = val

    @staticmethod
    def dxtbx_crystal_from_ucell_and_symbol(ucell_tuple_Adeg, symbol):
        """
        :param ucell_tuple_Adeg:  unit cell tuple a,b,c al, be, ga in Angstom and degrees
        :param symbol: lookup symbol for space group, e.g. 'P1'
        :return:a default crystal in conventional orientation, a along x-axis
        """
        from cctbx import crystal
        from dxtbx.model.crystal import CrystalFactory
        symm = crystal.symmetry("%f,%f,%f,%f,%f,%f" % ucell_tuple_Adeg, symbol)

        ucell = symm.unit_cell()
        O = ucell.orthogonalization_matrix()
        real_space_a = O[0], O[3], O[6]
        real_space_b = O[1], O[4], O[7]
        real_space_c = O[2], O[5], O[8]

        hall_symbol = symm.space_group_info().type().hall_symbol()

        return CrystalFactory.from_dict(
            {'__id__': 'crystal',
                       'real_space_a': real_space_a,
                       'real_space_b': real_space_b,
                       'real_space_c': real_space_c,
                       'space_group_hall_symbol': hall_symbol})

    @staticmethod
    def dummie_Fhkl(ucell, symbol):
        from simtbx.diffBragg.utils import fcalc_from_pdb
        Fhkl = fcalc_from_pdb(resolution=2, algorithm="fft", wavelength=1, symbol=symbol, ucell=ucell)
        return Fhkl
