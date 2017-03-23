

import numpy as np
from pyscf import gto


def calc_eri3c(mol,auxmol,IklM):
  nao = mol.nao_nr()
  naux = auxmol.nao_nr()

  # Merging M1 and M2 to compute the integrals.
  atm, bas, env = gto.conc_env(mol._atm, mol._bas, mol._env, auxmol._atm, auxmol._bas, auxmol._env)

  eri3c = np.empty((nao,nao,naux))
  pi = 0
  for i in range(mol.nbas):
      pj = 0
      for j in range(mol.nbas):
          pk = 0
          for k in range(mol.nbas, mol.nbas+auxmol.nbas):
              shls = (i, j, k)
              buf = gto.getints_by_shell(IklM, shls, atm, bas, env)
              di, dj, dk = buf.shape
              eri3c[pi:pi+di,pj:pj+dj,pk:pk+dk] = buf
              pk += dk
          pj += dj
      pi += di
  return eri3c


def calc_eri2c(mol,auxmol,Ikl):
  nao = mol.nao_nr()
  naux = auxmol.nao_nr()

  # Merging M1 and M2 to compute the integrals.
  atm, bas, env = gto.conc_env(mol._atm, mol._bas, mol._env, auxmol._atm, auxmol._bas, auxmol._env)

  eri2c = np.empty((naux,naux))
  pk = 0
  for k in range(mol.nbas, mol.nbas+auxmol.nbas):
      pl = 0
      for l in range(mol.nbas, mol.nbas+auxmol.nbas):
          shls = (k, l)
          buf = gto.getints_by_shell(Ikl, shls, atm, bas, env)
          dk, dl = buf.shape
          eri2c[pk:pk+dk,pl:pl+dl] = buf
          pl += dl
      pk += dk
  return eri2c


