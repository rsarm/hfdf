

import numpy as np
from pyscf import gto


def integral_one_gaussian_from_overlap(mol):
  """
  Outputs the integral of only one Gaussian
  using as second Gaussian in <i|j> an 1s
  basis funtion of an He atom with
  alpha=1e-15 and coeff=1.00000. The result
  is divided by the norm of the He function.
  """
  # Creating auxiliar Molecule
  intmol=gto.Mole()
  intmol.atom = '''He  0. 0. 0.'''
  intmol.basis = 'xxx'
  intmol.build()

  # Merging molecules
  atm_x, bas_x, env_x = gto.conc_env(intmol._atm, intmol._bas, intmol._env,
                                        mol._atm,    mol._bas,    mol._env)
  # Computing the overlap
  HF_partial=gto.moleintor.getints('cint1e_ovlp_sph', atm_x, bas_x, env_x,
                                                      comp=1, hermi=0,
                                                      aosym='s1', out=None)
  return np.hstack(HF_partial[:,:1])[1:]/gto.gto_norm(0,1e-15)










def hellmann_feynman_df(mol,ia):
  """
  Outputs the integral
        Z_I*<i|(r-R_I)/|r-R_I|^3|j>
  of only one Gaussian using as second Gaussian
  an 1s basis funtion of an He atom with
  alpha=1e-15 and coeff=1.00000. The result
  is divided by the norm of the He function.
  """

  # Creating auxiliar Molecule
  intmol=gto.Mole()
  intmol.atom = '''He  0. 0. 0.'''
  intmol.basis = 'xxx'
  intmol.build()

  # Merging molecules
  atm_x, bas_x, env_x = gto.conc_env(intmol._atm, intmol._bas, intmol._env,\
                                        mol._atm,    mol._bas,    mol._env)


  PTR_RINV_ORIG   = 4
  env_x[PTR_RINV_ORIG:PTR_RINV_ORIG+3] = mol.atom_coord(ia)
  HF_partial=gto.moleintor.getints('cint1e_drinv_sph', atm_x, bas_x, env_x,
                                                       comp=3, hermi=0,
                                                       aosym='s1', out=None)

  return -mol.atom_charge(ia)*\
          np.hstack(HF_partial[:,:,:1])[1:]/gto.gto_norm(0,1e-15)






# Examples
if __name__ == '__main__':

  #from pyscf import gto
  #import numpy as np

  mol = gto.Mole()
  mol.atom = '''H  -0.37 0. 0.; H 0.37 0. 0.'''
  mol.basis = 'sto-3g'
  mol.build();

  print '\n=====Example 1:====='

  # This gives the forces of each term of the basis in which then
  # the density i expanded and then it has top be multiplied by
  # the density expansion coefficients
  print hellmann_feynman_df(mol,0)

  print '''
=====Example 2:=====
     Example for Hydrogen with the sto-3g basis set:
     The integral cint1e_ovlp_sph <He_aux|H_orbital> have to
     give N/M^2, where N and M are defined below
     in the example (M1, M2, M3, N1, N2 and N3)
     N is the norm of the Gaussian \int exp(-2*alpha*r^2)dr.

     The basis xxx set for He is
     H    S
          1e-15                  1.00000000
     The sto-3g basis set for H is
     H    S
          3.42525091             0.15432897
          0.62391373             0.53532814
          0.16885540             0.44463454
  '''
  # Computing the norms:
  N1=gto.gto_norm(0,3.42525091)
  M1=gto.gto_norm(0,3.42525091/2)
  N2=gto.gto_norm(0,0.62391373)
  M2=gto.gto_norm(0,0.62391373/2)
  N3=gto.gto_norm(0,0.16885540)
  M3=gto.gto_norm(0,0.16885540/2)
  print 'From the Norms M,N:',0.15432897*N1/M1/M1+0.53532814*N2/M2/M2+0.44463454*N3/M3/M3
  print 'From the integrals:',integral_one_gaussian_from_overlap(mol)[0]
  print ''

  print '''
=====Example 3:=====
  '''

  # Creating auxiliar Molecule
  intmol=gto.Mole()
  intmol.atom = '''He  0. 0. 0.'''
  intmol.basis = 'xxx'
  intmol.build()

  # Merging molecules
  atm_x, bas_x, env_x = gto.conc_env(intmol._atm, intmol._bas, intmol._env,\
                                        mol._atm,    mol._bas,    mol._env)
  PTR_RINV_ORIG   = 4
  env_x[PTR_RINV_ORIG:PTR_RINV_ORIG+3] = mol.atom_coord(0)
  hf_int=gto.moleintor.getints('cint1e_drinv_sph', atm_x, bas_x, env_x,
                                                        comp=3, hermi=0,
                                                        aosym='s1', out=None)[0]
  print 'Raw:\n ', hf_int[1:,1:]
  print 'Normalized:\n', hf_int/gto.gto_norm(0,1e-15)
