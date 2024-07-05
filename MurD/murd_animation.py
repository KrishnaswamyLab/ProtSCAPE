import os

# Set the directory where the PDB files are stored
pdb_dir = '/Users/siddharthviswanath/Desktop/Yale/Research projects/ProtSCAPE/MurD/linear_interpolants_100/'

# Load the PDB files and create states
for i in range(100):
    pdb_file = os.path.join(pdb_dir, f'murd_atomic_frame_{i}.pdb')
    # cmd.do(f'print {pdb_file}')
    cmd.load(pdb_file, 'animation', state=i+1)

# Set the frame rate and play the animation
cmd.mset("1 x100")  # Define the frames for the animation (100 states)
cmd.mplay()  # Play the animation
