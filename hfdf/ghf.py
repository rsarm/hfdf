

import numpy as np

from pyscf import gto, scf, grad

import df_integrals
import mo_integrals
import normalization
import eri









def get_hf(molstr,fitness = 'repulsion',basis='sto3g',auxbasis='weigend',ref=True):
    """
    Function to calculate the Hellmann-Feynman forces from an electronic density
    expanded in Gaussian-type atomic orbitals.

    Returns a dictionary with the Hellmann-Feynman forces computed from the
    DF expansion, from the MO, the reference forces, energies and number
    of electrons.

    Arguments:

    molstr   :: expecting something like '''H  -0.37 0. 0. ; H  0.37 0. 0. '''

    fitness  :: Integral to inimize to find the DF coefficients.
                Can be 'repulsion' or 'density'.

    basis    :: Regular basis.

    auxbasis :: Auxiliary basis set to fit the density.

    ref      :: Selects wheather or not the reference forces are
                computed for comparison.
    """

    # Main Molecule object (M1).
    mol = gto.Mole()
    mol.atom = molstr
    mol.basis = basis
    mol.build()

    # Creating auxiliary Molecule object for the fitting (M2).
    auxmol = gto.Mole()
    auxmol.atom = molstr
    auxmol.basis = auxbasis
    auxmol.build()

    # Selecting the correspondent integral to minimize.
    if fitness=='density'  : IklM='cint3c1e_sph'; Ikl='cint1e_ovlp_sph'
    if fitness=='repulsion': IklM='cint3c2e_sph'; Ikl='cint2c2e_sph'

    # Computing the necessary integrals.
    eri2c=eri.calc_eri2c(mol,auxmol,Ikl )
    eri3c=eri.calc_eri3c(mol,auxmol,IklM)


    hf_forces_df=np.ones([mol.natm,3])
    number_of_electrons=[100000.]  # A list to make it a mutable object? #nasty

    #Definig a inner function
    def get_vhf(mol, dm, *args, **kwargs):
        """This function needs to be defined here as it depends on eri2c and eri3c,
        which are computed above.
        """
        naux = eri2c.shape[0]
        nao  = mol.nao_nr()
        rho  = np.einsum('ijp,ij->p', eri3c, dm);
        rho  = np.linalg.solve(eri2c, rho)

        jmat = np.einsum('p,ijp->ij', rho, eri3c)
        kpj  = np.einsum('ijp,jk->ikp', eri3c, dm)
        pik  = np.linalg.solve(eri2c, kpj.reshape(-1,naux).T)
        kmat = np.einsum('pik,kjp->ij', pik.reshape(naux,nao,nao), eri3c)

        # There is something I don't remember now with the normalization
        # of AO in PySCF. It seems it is only needed to multiply all the
        # DF coefficients by np.sqrt(4*np.pi):
        rho_normalized = rho * np.sqrt(4*np.pi) #normalization.normalize_df(auxmol)[0]
        #rho_normalized =  rho * normalization.normalize_df(auxmol)[0]



        # Compute the Hellmann Feynman Forces from the DF expansion.
        for atom_id in range(mol.natm):
            for axis in [0,1,2]:
                norm_aux=df_integrals.integral_one_gaussian_from_overlap(auxmol)
                hf=df_integrals.hellmann_feynman_df(auxmol,atom_id,axis)
                coul=grad.grad_nuc(mol)[atom_id][axis]

                hf_forces_df[atom_id][axis]=hf.dot(rho_normalized)+coul

        number_of_electrons[0]=norm_aux.dot(rho_normalized)

        return jmat - kmat * .5

    mf = scf.RHF(mol)
    mf.verbose = 0
    # Solving the scf RHF.
    e_ref=mf.kernel()

    # Compute the reference Hartree-Fock forces
    if ref==True:
      # Compute the reference Hartree-Fock forces
      g = grad.rhf.Gradients(mf)
      hf_forces_ref=g.grad()

      # Hellmann-Feynman Forces from the MO
      hf_forces_mo=np.ones([mol.natm,3])

      P=mf.from_chk()
      for i in range(mol.natm):
        F_en_0=mo_integrals.hellmann_feynman_mo(mol,i)/2.
        F_nn=grad.grad_nuc(mol)
        F_en=np.einsum('ij,kji->k',P,F_en_0)*2. # 2*\Sum P_ij*Phi_i*Phi_j
        hf_forces_mo[i]=F_en+F_nn[i]
    else:
      hf_forces_ref = None
      hf_forces_mo  = None


    mf.get_veff = get_vhf  # This substitutes the function 'get_veff' in mf by the
    # implementation of density fitting 'get_vhf' which prints the d_l coefficients.
    # This is done here after the SCF was solved.

    e_fit=mf.energy_tot()  # Run this only to get the density basis coefficients.
    # Here the function 'get_vhf' is called and give access to the coefficients
    # prints the d_l coefficients.

    return {'hf_df'  : hf_forces_df,
            'hf_mo'  : hf_forces_mo,
            'hf_ref' : hf_forces_ref,
            'nelect' : number_of_electrons[0],
            'e_fit'  : e_fit*27.2114,
            'e_ref'  : e_ref*27.2114
            }




# Examples
if __name__ == '__main__':

  print '''
=====Example 1:=====
  '''

  molstr = '''H  -0.37 0. 0. \n H  0.37 0. 0. '''
  #molstr = '''O  -0.37 0. 0. ; O  0.37 0. 0. '''

  hf_dict=get_hf(molstr,fitness = 'repulsion',basis='sto3g',auxbasis='weigend',ref=True)


  print hf_dict.keys(),'\n'

  print 'nelect:'
  print hf_dict['nelect']
  print 'hf_df:'
  print hf_dict['hf_df']
  print 'hf_mo:'
  print hf_dict['hf_mo']
  print 'hf_ref:'
  print hf_dict['hf_ref']
  print 'e_fit:'
  print hf_dict['e_fit']
  print 'e_ref:'
  print hf_dict['e_ref']
