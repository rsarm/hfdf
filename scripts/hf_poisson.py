
import numpy as np
from scipy import special
import sys

from pyscf import gto


# This file defines two functions to compute the Hellmann-Feynmann
# forces by using the analitical solution for the electric force
# of a Gaussian charge distrubution on a point charge.



def force_en_1Gz(alpha,center,R,cartdir):
    """Force of a Gaussian charge distribution on
    a charge point.
    """

    dd=(R-center)[cartdir-1]
    R2=np.einsum('i,i',R-center,R-center)

    if R2==0.: return 0.
    R2sqrt=np.sqrt(R2)

    x=2*np.sqrt(alpha/np.pi)*np.exp(-alpha*R2)/R2\
     -special.erf(np.sqrt(alpha)*R2sqrt)/np.power(R2sqrt,3)

    return dd*x/np.power(np.pi/alpha,3/2.)


def force_nn_1Gz(center,R,cartdir):
    """Coulomb forces."""

    z={'H':1,'C':6,'N':7,'O':8,'F':9}

    dd=(R-center)[cartdir-1]
    R2=np.dot(R-center,R-center)

    if R2==0.: return 0

    R2sqrt=np.sqrt(R2)
    return dd/np.power(R2sqrt,3)







# Examples
if __name__ == '__main__':


  import sys
  import read_xyz as xyz

  #molstr=xyz.read(sys.argv[1])
  molstr=xyz.get_block(sys.argv[1])[2:]


  print '''
=====Example 1:=====
  '''
  mol = gto.Mole()
  mol.atom =molstr
  mol.basis = '631g'
  mol.build()

  F11=force_en_1Gz(2.02525091,mol.atom_coord(0),
                              mol.atom_coord(1),1)*1.000000
  F2=force_nn_1Gz(mol.atom_coord(0),
                  mol.atom_coord(1),1)
  F1=0.3570601*F11
  F_Total=F1+F2

  print '\nF_EN  = %12.6f' % (F1)
  print   'F_NN  = %12.6f' % (F2)
  print   'F_Tot = %12.6f' % (F_Total)
  print   'Norm  = %12.6f' % (np.power(np.pi/2.0252509,3/2.)*0.3570601*3.5447)
  print   'Norm  = %12.6f' % (gto.gto_norm(0,2.0252509)*0.3570601*3.5447)


  print '''
=====Example 2:=====
  '''

  mol.basis = 'sto3g'
    

  F11=force_en_1Gz(15.6752927,mol.atom_coord(0),
                              mol.atom_coord(1),1)*0.0186886
  F12=force_en_1Gz(3.6063578 ,mol.atom_coord(0),
                              mol.atom_coord(1),1)*0.0631670
  F13=force_en_1Gz(1.2080016 ,mol.atom_coord(0),
                              mol.atom_coord(1),1)*0.1204609
  F14=force_en_1Gz(0.4726794 ,mol.atom_coord(0),
                              mol.atom_coord(1),1)
  F15=force_en_1Gz(0.2018100 ,mol.atom_coord(0),
                              mol.atom_coord(1),1)
  F2=force_nn_1Gz(mol.atom_coord(0),
                  mol.atom_coord(1),1)
  F1=0.19586357*(F11+F12+F13)+0.07120454*F14-0.00381857*F15
  F_Total=F1+F2
  print F1
  print F2
  print 'F_Total=',F_Total



  print '''
=====Example 4:=====
  '''

  import hf_poisson as poi

  mol4 = gto.Mole()
  mol4.atom = '''H  -0.37 0. 0.; H 0.37 0. 0.'''
  mol4.basis = 'xxx'
  mol4.build();

  print hellmann_feynman_df(mol4,1,0)*3.5447


  F11=poi.force_en_1Gz(2.02525091,mol.atom_coord(0),
                             mol.atom_coord(1),1)*1.000000
  F2=poi.force_nn_1Gz(mol.atom_coord(0),
                  mol.atom_coord(1),1)
  F1=F11
  F_Total=F1+F2

  print '\nF_EN  = %12.6f' % (F1)
  print   'F_NN  = %12.6f' % (F2)
  print   'F_Tot = %12.6f' % (F_Total)
  print   'Norm  = %12.6f' % (np.power(np.pi/2.0252509,3/2.)*0.3570601)
  print   'Norm  = %12.6f' % (gto.gto_norm(0,2.0252509)*0.3570601)

