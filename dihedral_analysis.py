import os, glob
from shutil import which

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import MDAnalysis as mda
from MDAnalysis.analysis import rms
from MDAnalysis.analysis import dihedrals

## Helper functions

def gen_trajectories(data_dir, output_dir, dcd_filename):

	if which('mdconvert') is None:
		raise Exception("mdconvert utility not found. Fix: pip install mdtraj")

	esc_data_dir = data_dir.replace(" ", "\\ ")
	cmd = "find " + esc_data_dir + os.sep + "*.pdb -type f -print0 | sort -Vz | xargs -0 mdconvert -o " + output_dir + os.sep + dcd_filename 
	os.system(cmd)

def de_shaw_filesort(filename):

	base = os.path.basename(filename)
	fname = os.path.splitext(base)[0]
	frame = fname.split("-")[1]
	
	return int(frame)


parent_dir = "/gpfs/gibbs/pi/krishnaswamy_smita/de_shaw"
prot_name = "GB3"
workspace_dir = "/home/db2454/ProGSNN"
traj_dirs = ["0 to 2 us", "2 to 4 us",  "4 to 6 us",  "6 to 8 us",  "8 to 10 us"]

## Load topology file from frame 0

pdb_files = sorted(glob.glob(os.path.join(parent_dir, prot_name, traj_dirs[0]) + os.sep + "*.pdb"), key=de_shaw_filesort)
top_file = pdb_files[0] 

## Generate trajectories

dcd_dir = os.path.join(workspace_dir, prot_name)
if not os.path.isdir(dcd_dir):
	os.mkdir(dcd_dir)

dcd_files = []
for traj_dir in traj_dirs:
	
	dcd_fname = traj_dir.replace(" ", "_") + ".dcd"
	dcd_file = dcd_dir + os.sep + dcd_fname
	dcd_files.append(dcd_file)

	if not os.path.isfile(dcd_file):

		gen_trajectories(os.path.join(parent_dir, prot_name, traj_dir), dcd_dir, dcd_fname)

## Concatenate and load trajectory

u = mda.Universe(top_file, *dcd_files, in_memory=True, dt=5)
traj_len = len(u.trajectory)

## Analysis

rgyr_timelapse = []
sim_time = []
for ts in u.trajectory:
	time = u.trajectory.time
	sim_time.append(time)
	rgyr = u.atoms.radius_of_gyration()
	rgyr_timelapse.append(rgyr)
	print("Frame: {:3d}/{:d}, Time: {:4.1f} ps, Rgyr: {:.4f} A".format(ts.frame, traj_len-1, time, rgyr))

u.trajectory[0] # set to first frame
rmsd_analysis = rms.RMSD(u, select='backbone', groupselections=['name CA', 'protein'])
rmsd_analysis.run()
rmsd_df = pd.DataFrame(rmsd_analysis.rmsd[:, 2:], columns=['Backbone', 'C-alphas', 'Protein'], index=rmsd_analysis.rmsd[:, 1])
rmsd_df.index.name = 'Time (ps)'

protein = u.select_atoms('protein')
rama = dihedrals.Ramachandran(protein).run()
janin = dihedrals.Janin(protein).run()

## Plotting

plt.figure(figsize=(6,3), dpi=200)
plt.plot(np.multiply(0.001, sim_time), rgyr_timelapse, linestyle='--', linewidth=0.3)
plt.xlabel("Time (ns)")
plt.ylabel("Radius of gyration (A)")
plt.tight_layout()
plt.savefig("rgyr.png")
plt.close()

plt.figure(figsize=(6,3), dpi=200)
rmsd_df.plot(title='RMSD', linestyle='--', linewidth=0.3, alpha=0.75, ax=plt.gca())
plt.legend(frameon=False, fontsize=6)
plt.tight_layout()
plt.savefig("rmsd.png")
plt.close()

plt.figure(figsize=(4,4), dpi=200)
rama.plot(color='black', marker='.', ref=True, ax=plt.gca(), alpha=0.3, s=0.5)
plt.tight_layout()
plt.savefig("rama.png")
plt.close()

#plt.figure()
#plt.scatter(rama.angles[:,0,])
#plt.tight_layout()
#plt.savefig("rama_cbytime.png")
#plt.close()

plt.figure(figsize=(4,4), dpi=200)
janin.plot(ref=True, marker='.', color='black', ax=plt.gca(), alpha=0.3, s=0.5)
plt.tight_layout()
plt.savefig("janin.png")
plt.close()
