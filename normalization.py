
import numpy as np
from pyscf import gto


# Basis sets that to fit the density.
df_basis_dict={
  'sto-3g' :{'H':[ 1,  0],
             'C':[ 2,  3],
             'N':[ 2,  3],
             'O':[ 2,  3]
            },
  'weigend':{'H':[ 3,  3,  5,  0,  0],
             'C':[ 6, 12, 15,  7,  9],
             'N':[ 6, 12, 15,  7,  9],
             'O':[ 6, 12, 15,  7,  9]
            },
  'cc-pv5z':{'H':[ 5, 12, 15, 14,  9,  0],
             'O':[ 5, 15, 20, 21, 18, 12],
             'C':[ 5, 15, 20, 21, 18, 12]
            },
  'cc-pvtz':{'H':[ 3,  6,  5,  0],
             'O':[ 3,  9, 10,  7],
             'C':[ 3,  9, 10,  7]
            }
  }


norm_by_l_dict={
  'S': np.sqrt(4*np.pi),
  'P': np.sqrt(4*np.pi/3),
  'D': np.sqrt(4*np.pi/15),
  'F': np.sqrt(4*np.pi/105),
  'G': np.sqrt(4*np.pi/315),
  'H': np.sqrt(4*np.pi/1155)
  }

norm_by_l_list=[
  np.sqrt(4*np.pi),      # S
  np.sqrt(4*np.pi/3),    # P
  np.sqrt(4*np.pi/15),   # D
  np.sqrt(4*np.pi/105),  # F
  np.sqrt(4*np.pi/315),  # G
  np.sqrt(4*np.pi/1155)  # H
  ]


def total_num_basis(mol):
    """xxx."""

    return sum([sum(df_basis_dict[mol.basis][mol.atom_symbol(i)]) for i in range(mol.natm)])



def normalize_df(mol):
    """xxx."""

    norm_vector=np.empty(total_num_basis(mol))

    c=0
    for i in range(mol.natm):
      for j, num_ao_l in enumerate(df_basis_dict[mol.basis][mol.atom_symbol(i)]):
        for k in range(num_ao_l):
          #print i,j,k,'  ', norm_by_l_list[j]
          norm_vector[c]=norm_by_l_list[j]
          c+=1

    return norm_vector






















# Examples
if __name__ == '__main__':

  print '''
=====Example 1:=====
  '''

  # Creating auxiliar Molecule
  x=gto.Mole()
  x.atom = '''H  -0.37 0. 0. ; H  0.37 0. 0. '''
  x.basis = 'weigend'
  x.build()

  print x.basis, 'basis set:'
  print normalize_df(x)

