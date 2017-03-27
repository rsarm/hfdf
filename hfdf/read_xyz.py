import sys
import numpy as np



def read(mol_file):

  print """
  read_xyz.read: Deprecated!
  Use xyz.get_block('file.xyz')[2:] instead
  """

  f=open(mol_file,'r')
  l=f.readlines()
  f.close()

  return l[2:]




def get_block(mol_file):

  f=open(mol_file,'r')
  lf=f.readlines()
  f.close()

  return lf



Z={'H':1.,'C':6.,'N':7,'O':8.}


def write_formated(numpy_array,_format=None):
  """Write numpy arrays in a nice format and
     without brackets.

     Usage:
     >>> x=np.random.rand(5)
     >>> print write_formated(x)
     >>> print read_xyz.write_formated(x, "%12.3f")
  """

  if _format==None:
    return ' '.join("%12.8f" % (k) for k in numpy_array)
  else:
    return ' '.join( _format % (k) for k in numpy_array)



def get_atoms_str(molec_file):
  """xxx."""

  lf=get_block(molec_file)

  nat=int(lf[0])
  atomic_data_str=np.array([lf[i+2].split() for i in range(nat)])

  return atomic_data_str





def get_atoms_float(molec_file):
  """xxx."""

  atomic_data_str=get_atoms_str(molec_file)

  for i,j in enumerate(atomic_data_str):
    atomic_data_str[i][0]=Z[j[0]]

  return atomic_data_str.astype(float)


def get_label(molec_file):
  """xxx."""

  return get_block(molec_file)[1]










class mol_xyz(object):
  """
  xxx.
  """
  def __init__(self,xyz_file):
    #self.natm=0
    #self.z=[]
    #self.atms=[]
    self._mol_file='Not loaded yet!'
    #self.mol=np.array([])
    #self.R=0
    #self.label='Not loaded yet!
    self.get_molecule(xyz_file)
    


  def get_molecule(self,xyz_file):
    self._mol_file  = xyz_file
    self.label      = get_label(xyz_file)
    self.mol        = get_atoms_float(xyz_file)
    self.R          = self.mol[:,1:4]
    self.natm       = self.mol.shape[0]
    self.z          = self.mol[:,0]
    self.atms       = get_atoms_str(xyz_file)[:,0].tolist()

    return self.mol

  def atom_coord(self,atom_n):
    if self._mol_file!='Not loaded yet!':
      return self.mol[atom_n][1:]
    else:
       return None

  def atom_charge(self,atom_n):
    if self._mol_file!='Not loaded yet!':
      return self.z[atom_n]
    else:
       return None

  def distance(self,atom_i,atom_j):
    if self._mol_file!='Not loaded yet!':
      return np.linalg.norm(self.mol[atom_i][1:]-self.mol[atom_j][1:])
    else:
       return None
  

