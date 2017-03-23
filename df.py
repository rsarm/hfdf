#!/home/rsarmiento/anaconda2/bin/python


import numpy as np
from pyscf import gto, scf, grad
import read_xyz as xyz
import sys,os

import forces4df as fdf
import normalization
import eri





den_file=open(os.path.basename(sys.argv[1]).split('.')[0]+'.den','w')



molstr=xyz.get_block(sys.argv[1])[2:]

# Main Molecule object (M1).
mol = gto.Mole()
mol.atom =molstr
mol.basis = '631gs'
mol.build()

# Creating auxiliary Molecule object for the fitting (M2).
auxmol = gto.Mole()
auxmol.atom = molstr
auxmol.basis = 'sto-3g'
auxmol.build()



# Function to minimize.
fitness = 'repulsion'

# Selecting the correspondent integral to minimize.
if fitness=='density'  : IklM='cint3c1e_sph'; Ikl='cint1e_ovlp_sph'
if fitness=='repulsion': IklM='cint3c2e_sph'; Ikl='cint2c2e_sph'


# Computing the necessary integrals.
eri2c=eri.calc_eri2c(mol,auxmol,Ikl)
eri3c=eri.calc_eri3c(mol,auxmol,IklM)


# Compute the Hellmann Feynman integrals.
atom_id=0
axis=0
hf=fdf.HellmannFeynman_df(auxmol,atom_id,axis) #*np.sqrt(np.pi*4) # Accounts for the difference
norm_aux=fdf.OneGaussian_from_Overlap(auxmol)  #*np.sqrt(np.pi*4) # in normalization
#print 'hf  :',hf
#print 'Ovlp:',norm_aux
Coul=grad.grad_nuc(mol)[atom_id][axis]


def get_vhf(mol, dm, *args, **kwargs):
    """This function needs to be defined here as it depends on eri2c and eri3c,
    which are computed above. This is a bit a la 'por la pata del blumer' but
    currently I don't have another solution because the density lives only
    inside the function.
    """
    naux = eri2c.shape[0]
    nao  = mol.nao_nr()
    rho  = np.einsum('ijp,ij->p', eri3c, dm);
    rho  = np.linalg.solve(eri2c, rho)

    jmat = np.einsum('p,ijp->ij', rho, eri3c)
    kpj  = np.einsum('ijp,jk->ikp', eri3c, dm)
    pik  = np.linalg.solve(eri2c, kpj.reshape(-1,naux).T)
    kmat = np.einsum('pik,kjp->ij', pik.reshape(naux,nao,nao), eri3c)

    print >>den_file, len(mol.atom)
    print >>den_file, 'Molecule', sys.argv[1]

    iini = 0
    for i in mol.atom:
        bas_len=sum(normalization.df_basis_dict[auxmol.basis][i.split()[0]])
        print >>den_file, i.strip('\n'),'  ',
        print >>den_file, " ".join("%16.12f" % x for x in rho[iini:iini+bas_len])
        #print >>den_file, " ".join("%16.12f" % x for x in rho[iini:iini+bas_len[i.split()[0]]])
        iini+=bas_len

    rho_norm=rho*normalization.normalize_df(auxmol)[0]
    #print normalization.normalize_df(auxmol)
    #print 'rho_norm.shape', rho_norm.shape
    #print 'rho.shape', rho.shape

    print '\nF_EN = %12.6f'  % (hf.dot(rho_norm))
    print   'F_NN = %12.6f'  % (Coul)
    print   'Ftot = %12.6f'  % (hf.dot(rho_norm)+Coul)
    print   'Norm = %12.12f' % (norm_aux.dot(rho_norm))
    #print '\nRHO   =' , rho_norm
    #rhosave=open('rho.np','w'); np.save(rhosave,rho); rhosave.close()

    return jmat - kmat * .5



mf = scf.RHF(mol)
mf.verbose = 0

e_ref=mf.kernel() # Solving the scf RHF.
g = grad.rhf.Gradients(mf)
g.grad()

P=mf.from_chk()


mf.get_veff = get_vhf  # This substitutes the function 'get_veff' in mf by the
# implementation of density fitting 'get_vhf' which prints the d_l coefficients.
# This is done here after the SCF was solved.

e_fit=mf.energy_tot()  # Run this only to get the density basis coefficients.
# Here the function 'get_vhf' is called give access to the coefficients
# prints the d_l coefficients.

den_file.close()



print '\nHellmann-Feynman Forces:'
print '           x                y                z'
for i in range(mol.natm):
  F_en_0=fdf.HellmannFeynman(mol,i)/2.
  F_nn=grad.grad_nuc(mol)
  F_en=np.einsum('ij,kji->k',P,F_en_0)*2. # 2*\Sum P_ij*Phi_i*Phi_j
  Ft=F_en+F_nn[i]

  print i, mol.atom_symbol(0),
  print '%16.9f  %16.9f %16.9f' % (Ft[0],Ft[1],Ft[2])


print '\nMol_file   E_fit        E_ref       dE'
print sys.argv[1],' ',
print ' %0.4f ' % (e_fit*27.2114),
print ' %0.4f ' % (e_ref*27.2114),
print ' %0.4f [eV]' % ((e_fit-e_ref)*27.2114)












































