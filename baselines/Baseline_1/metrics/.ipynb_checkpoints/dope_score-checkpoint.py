"""
Original code from Degiacomi Lab's 'molearn' repo:
https://github.com/Degiacomi-Lab/molearn/blob/master/src/molearn/scoring/dope_score.py

Modified to have option of normalized DOPE scores.
"""

import os
import numpy as np
from copy import deepcopy
# from ..utils import ShutUp, random_string
try:
    import modeller
    from modeller import *
    from modeller.scripts import complete_pdb
    from modeller.optimizers import ConjugateGradients
except Exception as e:
    print('Error importing modeller: ')
    print(e)

from multiprocessing import get_context


"""
this import didn't work, pasting here instead of:
from ..utils import ShutUp, random_string
"""
# import os
# import numpy as np
import sys
import torch
import random
import string

def random_string(length=32):
    '''
    generate a random string of arbitrary characters. Useful to generate temporary file names.

    :param length: length of random string
    '''
    return ''.join(random.choice(string.ascii_letters)
                    for n in range(length))
    
class ShutUp:
    
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, *args):
        sys.stdout.close()
        sys.stdout = self._stdout


class DOPE_Score:
    '''
    This class contains methods to calculate dope without saving to save and load PDB files for every structure. 
    Atoms in a biobox coordinate tensor are mapped to the coordinates in the modeller model directly.
    '''
    atom_map = {('ILE', 'CD1'):('ILE', 'CD')}

    def __init__(self, 
                 mol, 
                 normalize=False):
        '''
        :param biobox.Molecule mol: One example frame to gain access to the topology. 
        Mol will also be used to save a temporary pdb file that will be reloaded in modeller to create the initial modeller Model.
        '''

        self.normalize = normalize

        # set residues names with protonated histidines back to generic HIS name (needed by DOPE score function)
        testH = mol.data["resname"].values
        testH[testH == "HIE"] = "HIS"
        testH[testH == "HID"] = "HIS"
        _mol = deepcopy(mol)
        _mol.data["resname"] = testH

        alternate_residue_names = dict(CSS=('CYX',))
        atoms = ' '.join(list(_mol.data['name'].unique()))
        tmp_file = f'tmp{random_string()}.pdb'
        _mol.write_pdb(tmp_file, conformations=[0], split_struc=False)
        log.level(0, 0, 0, 0, 0)
        env = environ()
        env.libs.topology.read(file='$(LIB)/top_heav.lib')
        env.libs.parameters.read(file='$(LIB)/par.lib')
        self.fast_mdl = complete_pdb(env, tmp_file)
        self.fast_fs = selection(self.fast_mdl.chains[0])
        self.fast_ss = self.fast_fs.only_atom_types(atoms)
        atom_residue = _mol.get_data(columns=['name', 'resname', 'resid'])
        atom_order = []
        first_index = next(iter(self.fast_ss)).residue.index
        offset = atom_residue[0, 2]-first_index
        for i, j in enumerate(self.fast_ss):
            if i < len(atom_residue):
                for j_residue_name in alternate_residue_names.get(j.residue.name, (j.residue.name,)):
                    if [j.name, j_residue_name, j.residue.index+offset] == list(atom_residue[i]):
                        atom_order.append(i)
                    else:
                        where_arg = (atom_residue==(np.array([j.name, j_residue_name, j.residue.index+offset], dtype=object))).all(axis=1)
                        where = np.where(where_arg)[0]
                        if len(where)==0:
                            if (j_residue_name, j.name) in self.atom_map:
                                alt_residue_name, alt_name = self.atom_map[(j_residue_name, j.name)]
                                where_arg = (atom_residue==(np.array([alt_name, alt_residue_name, j.residue.index+offset], dtype=object))).all(axis=1)
                                where = np.where(where_arg)[0]
                            else:
                                print(f'Cant find {j.name} in the atoms {atom_residue[atom_residue[:,2]==j.residue.index+offset]} try adding a mapping to DOPE_Score.atom_map')
                        atom_order.append(int(where))
        self.fast_atom_order = atom_order
        # check fast dope atoms
        reverse_map = {value:key for key, value in self.atom_map.items()}
        for i, j in enumerate(self.fast_ss):
            if i<len(atom_residue):
                assert _mol.data['name'][atom_order[i]]==j.name or reverse_map[(_mol.data['resname'][atom_order[i]], _mol.data['name'][atom_order[i]])][1]==j.name
        self.cg = ConjugateGradients()
        os.remove(tmp_file)

    def get_dope(self, frame, refine=False):
        '''
        Get the dope score. Injects coordinates into modeller and uses `mdl.build(build_method='INTERNAL_COORDINATES', initialize_xyz=False)` to reconstruct missing atoms.
        If a error is thrown by modeller or at any stage, we just return a fixed large value of 1e10.
        
        :param numpy.ndarray frame: shape [N, 3]
        :param bool refine: (default: False) If True, relax the structures using a maximum of 50 steps of ConjugateGradient descent
        :returns: Dope score as calculated by modeller. If error is thrown we just simply return 1e10.
        :rtype: float
        '''
        
        # expect coords to be shape [N, 3] use .cpu().numpy().copy() before passing here and make sure it is scaled correctly
        try:
            frame = frame.astype(float)
            self.fast_fs.unbuild()
            for i, j in enumerate(self.fast_ss):
                if i+1<frame.shape[0]:
                    j.x, j.y, j.z = frame[self.fast_atom_order[i], :]
            self.fast_mdl.build(
                build_method='INTERNAL_COORDINATES', 
                initialize_xyz=False
            )
            if refine == 'both':
                with ShutUp():
                    if self.normalize:
                        dope_unrefined = self.fast_mdl.assess_normalized_dope()
                        self.cg.optimize(self.fast_fs, max_iterations=50)
                        dope_refined = self.fast_mdl.assess_normalized_dope()
                    else:
                        dope_unrefined = self.fast_fs.assess_dope()
                        self.cg.optimize(self.fast_fs, max_iterations=50)
                        dope_refined = self.fast_fs.assess_dope()
                    return dope_unrefined, dope_refined
            with ShutUp():
                if refine:
                    self.cg.optimize(self.fast_fs, max_iterations=50)
                if self.normalize:
                    dope_score = self.fast_mdl.assess_normalized_dope()
                else:
                    dope_score = self.fast_fs.assess_dope()

            return dope_score
        except Exception:
            print('MODELLER experienced an error: returning 1e10')
            return 1e10
        
    def get_all_dope(self, coords, refine=False):
        '''
        Expect a array of frames. return array of DOPE score value.
        
        :param numpy.ndarray coords: shape [B, N, 3]
        :param bool refine: (default: False) If True, relax the structures using a maximum of 50 steps of Conjugate Gradient descent
        :returns: float array shape [B]
        :rtype: np.ndarray
        '''
        
        # expect coords to be shape [B, N, 3] use .cpu().numpy().copy() before passing here and make sure it is scaled correctly
        dope_scores = []
        for frame in coords:
            frame = frame.astype(float)
            self.fast_fs.unbuild()
            for i, j in enumerate(self.fast_ss):
                if i+1<frame.shape[0]:
                    j.x, j.y, j.z = frame[self.fast_atom_order[i], :]
            self.fast_mdl.build(
                build_method='INTERNAL_COORDINATES', 
                initialize_xyz=False
            )
            if refine:
                self.cg.optimize(self.fast_fs, max_iterations=50)
            
            if self.normalize:
                dope_scores.append(self.fast_mdl.assess_normalized_dope())
            else:
                dope_scores.append(self.fast_fs.assess_dope())
            
        return np.array(dope_scores)


def set_global_score(score, kwargs):
    '''
    Make score a global variable.
    This is used when initializing a multiprocessing process.
    '''
    
    global worker_dope_score
    worker_dope_score = score(**kwargs)  # mol = mol, data_dir=data_dir, **kwargs)


def process_dope(coords, kwargs):
    '''
    Worker function for multiprocessing class
    '''
    
    return worker_dope_score.get_dope(coords,**kwargs)


class Parallel_DOPE_Score:
    '''
    a multiprocessing class to get modeller DOPE scores.
    A typical use case would looke like::

      score_class = Parallel_DOPE_Score(mol, **kwargs)
      results = []
      for frame in coordinates_array:
          results.append(score_class.get_score(frame))
      .... # DOPE will be calculated asynchronously in background
      #to retrieve the results
      results = np.array([r.get() for r in results])
    '''
    
    def __init__(self, 
                 mol, 
                 normalize=False,
                 processes=-1, 
                 context='spawn', 
                 **kwargs):
        '''
        :param biobox.Molecule mol: biobox molecule containing one example frame of the protein to be analysed. This will be passed to DOPE_Score class instances in each thread.
        :param int processes: (default: -1) Number of processes argument to pass to multiprocessing.pool. This controls the number of threads created.
        :param \*\*kwargs: additional kwargs will be passed multiprocesing.pool during initialisation.
        '''
        
        # set a number of processes as user desires, capped on number of CPUs
        if processes > 0:
            processes = min(processes, os.cpu_count())
        else:
            processes = os.cpu_count()
        self.processes = processes
        self.mol = deepcopy(mol)
        score = DOPE_Score
        ctx = get_context(context)
        self.pool = ctx.Pool(
            processes=processes, 
            initializer=set_global_score,
            initargs=(score, dict(mol=mol, normalize=normalize)),
            **kwargs
        )
        self.process_function = process_dope

    def __reduce__(self):
        return (self.__class__, (self.mol, self.processes))

    def get_score(self, coords, **kwargs):
        '''
        :param np.array coords: # shape (N, 3) numpy array
        '''
        # is copy necessary?
        score = self.pool.apply_async(
            self.process_function, 
            (coords.copy(), kwargs)
        )
        return score

