
import numpy as np
from pyscf import gto


def hellmann_feynman_mo(mol,ia):
  """Computes <|\nabla r^{-1}|> for a given atom index ia
  in the molecule mol.
  """
  mol.set_rinv_origin_(mol.atom_coord(ia))
  return -mol.atom_charge(ia) * mol.intor('cint1e_drinv_sph', comp=3)

