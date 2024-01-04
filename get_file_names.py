import numpy as np
import os
from tqdm import tqdm
DATA_DIR = #PUT DATA DIR HERE
PROTEIN_NAME = DATA_DIR.split('/')[-1]

pdgs0to2 = []
pdgs2to4 = []
pdgs4to6 = []
pdgs6to8 = [] 
pdgs8to10 = []

listname_table = {'0 to 2 us': pdgs0to2,
                    '2 to 4 us': pdgs2to4,
                    '4 to 6 us': pdgs4to6,
                    '6 to 8 us': pdgs6to8,
                    '8 to 10 us':pdgs8to10}

filename_table = {'0 to 2 us': 'pdgs0to2.npy',
                    '2 to 4 us': 'pdgs2to4.npy',
                    '4 to 6 us': 'pdgs4to6.npy',
                    '6 to 8 us': 'pdgs6to8.npy',
                    '8 to 10 us':'pdgs8to10.npy'}

PARENT_DIR = f'file_names_{PROTEIN_NAME}/'
if not os.path.isdir(PARENT_DIR):
    os.mkdir(PARENT_DIR)

for i, entry in enumerate(tqdm(os.listdir(DATA_DIR))):
    print(entry)
    curr = listname_table[entry]
    for filename in os.listdir(DATA_DIR + '/' + entry):
        curr.append(filename)

    np.save(PARENT_DIR + filename_table[entry], curr)



