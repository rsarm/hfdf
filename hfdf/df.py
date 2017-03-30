

import numpy as np

from pyscf import gto, scf, grad

import df_integrals
import mo_integrals
import normalization
import eri



class density_fitting(object):
    """Class density_fitting.

    Properties that can be computed from a DF expansion
    can be obtained from density_fitting objects.
    """

    def __init__(self,molstr,fitness='density', basis='631gs', auxbasis='weigend'):
        # Input
        self.molstr   = molstr
        self.fitness  = fitness
        self.basis    = basis
        self.auxbasis = auxbasis

        #
        self._mol     = _build_mole(molstr,basis)    # Main Molecule object.
        self._auxmol  = _build_mole(molstr,auxbasis) # Auxiliary Molecule for the DF.

        # Getting the DF coefficients
        _df           = get_df(self.molstr, self.fitness, self.basis, self.auxbasis)

        self.den      = _df['dfc']
        self.efit     = _df['e_fit']
        self.eref     = _df['e_ref']
        self.dfstr    = _df['dfc_str']
        self._dm      = _df['density_matrix']





    def get_number__of_electrons(self):
        """Number of electrons that is obtained from the DF expansion."""

        self.nelect = _get_number_of_electrons(self._auxmol,self.den)

        return self.nelect




    def get_dip_moment(self):
        """Dipole moment computed from the DF expansion."""

        self.dip_moment = _get_dip_moment(self.den,self._auxmol)

        return self.dip_moment




    def get_hf_forces(self):
        """Hellmann-Feynman Forces from the DF expansion."""

        self.hf_forces = _get_hellmann_feynman_df(self.den,self._auxmol)  * 27.2114/0.529177

        return self.hf_forces




    def get_hf_forces_mo(self):
        """Hellmann-Feynman Forces from the MO."""

        self.hf_forces_mo = _get_hellmann_feynman_mo(self._mol,self._dm)  * 27.2114/0.529177

        return self.hf_forces_mo




    def get_forces_ref(self):
        """Reference forces from the Hartree-Fock calculation."""

        mf = scf.RHF(self._mol)
        mf.verbose = 0
        e_ref=mf.kernel() # Solving the scf RHF.

        # Compute the reference Hartree-Fock forces
        g = grad.rhf.Gradients(mf)

        self.forces_ref = -g.grad()  * 27.2114/0.529177 # Contains already the Coulomb part.

        return self.forces_ref











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




def _get_hellmann_feynman_df(rho_normalized,auxmol):
    """Compute the Hellmann Feynman Forces from the DF expansion.
    """

    _hf_forces_df = np.ones([auxmol.natm,3])

    for atom_id in range(auxmol.natm):
        hf=df_integrals.hellmann_feynman_df(auxmol,atom_id)
        coul=grad.grad_nuc(auxmol)[atom_id]#[axis]

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




def get_df(molstr, fitness, basis, auxbasis):
    """This is used to initialize the class density_fitting.
    This function is called in __init__() and performs a DF
    calculation and returns the DF coefficients, the enrgies
    and the density matrix.
    """

    mol    = _build_mole(molstr,basis)    # Main Molecule object.
    auxmol = _build_mole(molstr,auxbasis) # Auxiliary Molecule for the DF.

    denstr         = ['xxx']
    rho_normalized = [1.]
    density_matrix = [1.]

    def get_vhf(mol, dm, *args, **kwargs):
        """This function needs to be defined here as it depends on eri2c and eri3c,
        which are computed above.
        """

        jmat, kmat, rho_normalized[0] = _get_all(mol,auxmol,fitness, dm)

        denstr[0] =_write_df_coefficients(auxmol, rho_normalized[0])

        return jmat - kmat * .5


    mf = scf.RHF(mol)
    mf.verbose = 0
    e_ref=mf.kernel() # Solving the scf RHF.

    density_matrix[0]=mf.from_chk()

    mf.get_veff = get_vhf  # This substitutes the function 'get_veff' in mf by the
    # implementation of the density fitting 'get_vhf'.
    # This is done here after the SCF was solved.

    e_fit=mf.energy_tot()  # Here the function 'get_vhf' is called and give access
    # to the coefficients prints the d_l coefficients.

    return {'e_fit'  : e_fit*27.2114,
            'e_ref'  : e_ref*27.2114,
            'dfc_str': denstr[0],
            'dfc'    : rho_normalized[0],
            'density_matrix' : density_matrix[0]
           }

















# Examples
if __name__ == '__main__':




    from ase.atoms import Atoms

    #from hfdf.df import df, get_df



    def to_str(m):
        """xxx."""

        return [[m.get_chemical_symbols()[n],tuple([i for i in m.positions[n]])]
                for n in range(m.get_number_of_atoms())]


    mol=Atoms('H2') #expecting a diatomic molecule formula like O2

    mol.positions[0,0]=0.84

    df=density_fitting(to_str(mol))

    print df.get_number__of_electrons()
    print df.get_dip_moment()
    print df.get_hf_forces()
    print df.get_hf_forces_mo()
    print df.get_forces_ref()

