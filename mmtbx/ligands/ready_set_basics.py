from __future__ import absolute_import, division, print_function
import math

from scitbx import matrix
import six
from six.moves import range

def construct_xyz(ba, bv,
                  aa, av,
                  da, dv,
                  period=3,
                  ):
  assert ba is not None
  assert aa is not None
  assert da is not None
  rn = matrix.col(ba.xyz)
  rca = matrix.col(aa.xyz)
  rc = matrix.col(da.xyz)
  rcca = rc -rca

  e0 = (rn - rca).normalize()
  e1 = (rcca - (rcca.dot(e0))*e0).normalize()
  e2 = e0.cross(e1)

  pi = math.pi
  alpha = math.radians(av)
  phi = math.radians(dv)

  rh_list = []
  for n in range(0, period):
    rh = rn + bv * (math.sin(alpha)*(math.cos(phi + n*2*pi/period)*e1 +
                                     math.sin(phi + n*2*pi/period)*e2) -
                    math.cos(alpha)*e0)
    rh_list.append(rh)
  return rh_list

def generate_atom_group_atom_names(rg, names, return_Nones=False, verbose=True):
  '''
  Generate all alt. loc. groups of names
  '''
  atom_groups = rg.atom_groups()
  atom_altlocs = {}
  for ag in atom_groups:
    for atom in ag.atoms():
      atom_altlocs.setdefault(atom.parent().altloc, [])
      atom_altlocs[atom.parent().altloc].append(atom)
  if len(atom_altlocs)>1 and '' in atom_altlocs:
    for key in atom_altlocs:
      if key=='': continue
      for atom in atom_altlocs['']:
        atom_altlocs[key].append(atom)
    del atom_altlocs['']
  for key, value in six.iteritems(atom_altlocs):
    atoms=[]
    for name in names:
      for atom in value:
        if atom.name.strip()==name.strip():
          atoms.append(atom)
          break
      else:
        if return_Nones:
          atoms.append(None)
        else:
          if verbose:
            print('not all atoms found. missing %s from %s' % (name, names))
          break
    if len(atoms)!=len(names):
      yield None, None
    else:
      yield atoms[0].parent(), atoms
