
from __future__ import absolute_import, division, print_function
from numpy import sin, cos, arcsin


class RangedParameter:

  def __init__(self):
    self.minval = 0
    self.maxval = 1
    self.sigma = None
    self.init = None

  #@property
  #def init(self):
  #  return self.__init

  #@init.setter
  #def init(self, val):
  #  if val is not None:
  #    if val < self.minval:
  #      raise ValueError("Parameter cannot be initialized to less than the minimum")
  #    if val >self.maxval:
  #      raise ValueError("Parameter cannot be initialized to more than the maximum")
  #  self._init = val

  @property
  def maxval(self):
    return self._maxval

  @maxval.setter
  def maxval(self, val):
    self._maxval = val

  @property
  def minval(self):
    return self._minval

  @minval.setter
  def minval(self, val):
    self._minval = val

  @property
  def rng(self):
    if self.minval >= self.maxval:
      raise ValueError("minval (%f) for RangedParameter must be less than the maxval (%f)" % (self.minval, self.maxval))
    return self.maxval - self.minval

  def get_val(self, x_current):
    sin_arg = self.sigma * (x_current - 1) + arcsin(2 * (self.init - self.minval) / self.rng - 1)
    val = (sin(sin_arg) + 1) * self.rng / 2 + self.minval
    return val

  def get_deriv(self, x_current, deriv):
    cos_arg = self.sigma * (x_current - 1) + arcsin(2 * (self.init - self.minval) / self.rng - 1)
    dtheta_dx = self.rng / 2 * cos(cos_arg) * self.sigma
    return deriv*dtheta_dx

  def get_second_deriv(self, x_current, deriv, second_deriv):
    sin_arg = self.sigma * (x_current - 1) + arcsin(2 * (self.init - self.minval) / self.rng - 1)
    cos_arg = self.sigma * (x_current - 1) + arcsin(2 * (self.init - self.minval) / self.rng - 1)
    dtheta_dx = self.rng / 2 * cos(cos_arg) * self.sigma
    d2theta_dx2 = -sin(sin_arg)*self.sigma*self.sigma * self.rng / 2.
    return dtheta_dx*dtheta_dx*second_deriv + d2theta_dx2*deriv


class Parameters:

  def __init__(self):
    self.Ncells_abc =[]
    self.Ncells_def = []
    self.rotXYZ = []
    self.Bmatrix = []
    self.spot_scale = []
    self.eta = []
    self.wavelen_offset =[]
    self.wavelen_scale = []

  def add_Ncells_abc(self, val):
    #if len(val) != 3:
    #  raise ValueError("Ncells abc must be a 3-tuple")
    self.Ncells_abc.append(val)

  def add_Ncells_def(self, val):
    self.Ncells_def.append(val)

  def add_spot_scale(self, val):
    self.spot_scale.append(val)

  def add_rotXYZ(self, val):
    self.rotXYZ.append(val)

  def add_Bmatrix(self, val):
    self.Bmatrix.append(val.elems)

  def add_eta(self,val):
    self.eta.append(val)

  def add_wavelen_offset(self, val):
    self.wavelen_offset.append(val)

  def add_wavelen_scale(self, val):
    self.wavelen_scale.append(val)
