

import numpy as np

from pyscf import gto, scf, grad

import df_integrals
import mo_integrals
import normalization
import eri






def _get_all(mol,auxmol,fitness, dm):
    """Does the density fitting and return the DF coefficients and
    the necessary matrices to compute the HF energy with the fitted
    density.

    There is something I don't remember now with the normalization
    of AO in PySCF. It seems it is only needed to multiply all the
    DF coefficients by np.sqrt(4*np.pi):
    rho_normalized =  rho * normalization.normalize_df(auxmol)#[0]
    """

    # Selecting the correspondent integral to minimize.
    if fitness=='den' or fitness=='density'  :
        IklM='cint3c1e_sph'; Ikl='cint1e_ovlp_sph'
    if fitness=='rep' or fitness=='repulsion':
        IklM='cint3c2e_sph'; Ikl='cint2c2e_sph'

    # Computing the necessary integrals.
    eri2c=eri.calc_eri2c(mol,auxmol,Ikl )
    eri3c=eri.calc_eri3c(mol,auxmol,IklM)

    # Solving the system of equations obtained from the minimization.
    naux = eri2c.shape[0]
    nao  = mol.nao_nr()
    rho  = np.einsum('ijp,ij->p', eri3c, dm);
    rho  = np.linalg.solve(eri2c, rho)

    # Getting the relevant matrices to compute the energy with the DF.
    jmat = np.einsum('p,ijp->ij', rho, eri3c)
    kpj  = np.einsum('ijp,jk->ikp', eri3c, dm)
    pik  = np.linalg.solve(eri2c, kpj.reshape(-1,naux).T)
    kmat = np.einsum('pik,kjp->ij', pik.reshape(naux,nao,nao), eri3c)

    return jmat, kmat, np.sqrt(4*np.pi)*rho




def _build_mole(molstr,basis):
    """Builds a PySCF Mole object."""

    mol = gto.Mole()
    mol.atom = molstr
    mol.basis = basis
    mol.build()

    return mol




def _get_hellmann_feynman_df(rho_normalized,mol,auxmol):
    """Compute the Hellmann Feynman Forces from the DF expansion.
    """

    _hf_forces_df = np.ones([mol.natm,3])

    for atom_id in range(mol.natm):
        hf=df_integrals.hellmann_feynman_df(auxmol,atom_id)
        coul=grad.grad_nuc(mol)[atom_id]#[axis]

        _hf_forces_df[atom_id]=np.dot(rho_normalized,hf)+coul

    return -_hf_forces_df




def _get_hellmann_feynman_mo(mol,density_matrix):
    """Hellmann-Feynman Forces from the MO."""

    _hf_forces_mo=np.ones([mol.natm,3])

    for i in range(mol.natm):
      F_en_0=mo_integrals.hellmann_feynman_mo(mol,i)/2.
      F_nn=grad.grad_nuc(mol)
      F_en=np.einsum('ij,kji->k',density_matrix,F_en_0)*2. # 2*\Sum P_ij*Phi_i*Phi_j
      _hf_forces_mo[i]=F_en+F_nn[i]

    return -_hf_forces_mo




def _get_dip_moment(rho_normalized,auxmol):
    """Returns dipole moment computed from the DF expansion.

    It is done following the function 'dip_moment' writen
    in 'pyscf/scf/hf.py'.
    """

    integral_of_ao=df_integrals.integral_one_gaussian_polarization(auxmol)

    el_dip  = np.dot(rho_normalized, integral_of_ao)

    charges = auxmol.atom_charges()
    coords  = auxmol.atom_coords()

    nucl_dip = np.einsum('i,ix->x', charges, coords)

    return nucl_dip - el_dip




def _get_number_of_electrons(auxmol,rho_normalized):
    """Returns the number of electrons computed from the DF expansion.
    """

    integral_of_ao=df_integrals.integral_one_gaussian_from_overlap(auxmol)

    return integral_of_ao.dot(rho_normalized)




def _write_df_coefficients(auxmol,rho_normalized):
    """Return a string with the DF coefficients in xyz format.
    """

    denstr=str(len(auxmol.atom))+'\nMol 0.0 0.0\n'

    iini = 0
    for a in auxmol.atom:
        bas_len=sum(normalization.df_basis_dict[auxmol.basis][a[0]])

        denstr+=str(a[0])+" ".join("%16.12f" % x for x in a[1])
        denstr+=" ".join("%16.12f" % x for x in rho_normalized[iini:iini+bas_len])+'\n'
        iini+=bas_len

    return denstr




def _save_df_coefficients(auxmol,rho_normalized,file_name):
    """Save a file in xyz format with the DF coefficients."""

    den_file=open(file_name,'w')

    print >>den_file, len(auxmol.atom)
    print >>den_file, 'Mol 0.0 0.0'

    iini = 0
    for a in auxmol.atom:
        bas_len=sum(normalization.df_basis_dict[auxmol.basis][a[0]])

        print >>den_file, a[0]," ".join("%16.12f" % x for x in a[1]),'  ',
        print >>den_file, " ".join("%16.12f" % x for x in rho_normalized[iini:iini+bas_len])
        iini+=bas_len

    den_file.close()







def write_dfc(molstr,fitness = 'repulsion',basis='sto3g',auxbasis='weigend'):
    """Save a file in xyz format with the DF coefficients.

    Returns a dictionary with the energies and number of electrons.

    Arguments:

    molstr   :: expecting something like '''H  -0.37 0. 0. ; H  0.37 0. 0. '''

    fitness  :: Integral to minimize to find the DF coefficients.
                Can be 'repulsion' or 'density'.

    basis    :: Regular basis.

    auxbasis :: Auxiliary basis set to fit the density.

    """

    mol    = _build_mole(molstr,basis)    # Main Molecule object.
    auxmol = _build_mole(molstr,auxbasis) # Auxiliary Molecule for the DF.

    number_of_electrons = [1.]  # Lists to have a mutable object.
    denstr=['xxx']
    dip_mom = [1.]

    def get_vhf(mol, dm, *args, **kwargs):
        """This function needs to be defined here as it depends on eri2c and eri3c,
        which are computed above.
        """

        jmat, kmat, rho_normalized = _get_all(mol,auxmol,fitness, dm)

        denstr[0] =_write_df_coefficients(auxmol, rho_normalized)

        number_of_electrons[0] = _get_number_of_electrons(auxmol,rho_normalized)

        dip_mom[0] = _get_dip_moment(rho_normalized,auxmol)

        return jmat - kmat * .5


    mf = scf.RHF(mol)
    mf.verbose = 0
    e_ref=mf.kernel() # Solving the scf RHF.

    mf.get_veff = get_vhf  # This substitutes the function 'get_veff' in mf by the
    # implementation of the density fitting 'get_vhf'.
    # This is done here after the SCF was solved.

    e_fit=mf.energy_tot()  # Here the function 'get_vhf' is called and give access
    # to the coefficients prints the d_l coefficients.

    return {'nelect' : number_of_electrons[0],
            'e_fit'  : e_fit*27.2114,
            'e_ref'  : e_ref*27.2114,
            'dfc_str': denstr[0],
            'dip_mom': dip_mom[0]
           }




def get_hf(molstr,fitness = 'repulsion',basis='sto3g',auxbasis='weigend',ref=True):
    """
    Function to calculate the Hellmann-Feynman forces from an electronic density
    expanded in Gaussian-type atomic orbitals.

    Returns a dictionary with the Hellmann-Feynman forces computed from the
    DF expansion, from the MO, the reference forces, energies and number
    of electrons.

    Arguments:

    molstr   :: expecting something like '''H  -0.37 0. 0. ; H  0.37 0. 0. '''

    fitness  :: Integral to minimize to find the DF coefficients.
                Can be 'repulsion' or 'density'.

    basis    :: Regular basis.

    auxbasis :: Auxiliary basis set to fit the density.

    ref      :: Selects wheather or not the reference forces are
                computed for comparison.

    save_den :: Selects wheather or not the DF coeffs are saved to file.

    """

    # Main Molecule object (M1).
    mol    = _build_mole(molstr,basis)

    # Creating auxiliary Molecule object for the fitting (M2).
    auxmol = _build_mole(molstr,auxbasis)

    # 'global-inside-the-function' obejects to be modified inside
    # the inner function get_vhf().
    hf_forces_df        = np.ones([mol.natm,3])
    number_of_electrons = [1.]  # A list to have a mutable object.


    def get_vhf(mol, dm, *args, **kwargs):
        """This function needs to be defined here as it depends on eri2c and eri3c,
        which are computed above.
        """

        jmat, kmat, rho_normalized = _get_all(mol,auxmol,fitness, dm)

        # To modify the hf_forces_df it has to be term by term:
        _hf_forces_df=_get_hellmann_feynman_df(rho_normalized,mol,auxmol)
        #
        for atom_id in range(mol.natm):
            for axis in [0,1,2]:
                hf_forces_df[atom_id][axis]=_hf_forces_df[atom_id][axis]

        number_of_electrons[0]= _get_number_of_electrons(auxmol,rho_normalized)

        return jmat - kmat * .5



    mf = scf.RHF(mol)
    mf.verbose = 0
    e_ref=mf.kernel() # Solving the scf RHF.

    # Compute the reference forces
    if ref==True:
        # Compute the reference Hartree-Fock forces
        g = grad.rhf.Gradients(mf)
        hf_forces_ref=-g.grad() # Contains already the Coulomb part.

        # Hellmann-Feynman Forces from the MO
        hf_forces_mo = _get_hellmann_feynman_mo(mol,mf.from_chk())

    else:
        hf_forces_ref = None
        hf_forces_mo  = None

    mf.get_veff = get_vhf  # This substitutes the function 'get_veff' in mf by the
    # implementation of the density fitting 'get_vhf'.
    # This is done here after the SCF was solved.

    e_fit=mf.energy_tot()  # Here the function 'get_vhf' is called and give access
    # to the coefficients prints the d_l coefficients.

    return {'hf_df'  : hf_forces_df *27.2114/0.529177,
            'hf_mo'  : hf_forces_mo *27.2114/0.529177,
            'hf_ref' : hf_forces_ref*27.2114/0.529177,
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
