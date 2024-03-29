These trajectories and corresponding analysis were downloaded from the ATLAS database (https://www.dsimb.inserm.fr/ATLAS).

ATLAS gathers standardized molecular dynamics simulations of protein structures accompanied by their analysis in the form of interactive diagrams and trajectory visualization. All the raw trajectories as well as the results of analysis are available for download.


 • ARCHIVE CONTENT •
Protein structure and trajectory files:
     - PDB_Name.pdb: Protein structure in PDB format used to launch the molecular dynamics (after energy minimisation and equilibrations).
     - PDB_Name_corresp.tsv: Correspondence between the author residue number (PDB file / CIF file) and the new numbering (starting from 1).
     - PDB_Name_R{1,2,3}.tpr: File containing the starting structure of the simulation, the molecular topology and all the simulation parameters.
     - PDB_Name_R{1,2,3}.xtc: MD trajectory without solvent molecules. Each trajectory is of 100 ns with frames saved every 100 ps (1,000 frames in total).

Results of MD simulation analysis:
     - PDB_Name_Bfactor.tsv: Experimental B-factor values of the initial PDB structure (in Å²).
     - PDB_Name_RMSF.tsv: RMSF values for the 3 MD replicates (in Å).
     - PDB_Name_Neq.tsv: Local deformability index Neq for the 3 MD replicates.
     - PDB_Name_RMSD.tsv: RMSD values for the 3 MD replicates (in Å / ns).
     - PDB_Name_gyr.tsv: Gyration radius values for the 3 MD replicates (in Å / ns).
     - PDB_Name_contacts.tsv: Residue-level interactions co-crystallized with the protein of interest (chain, ligands, ions and nucleotides within 6 Å threshold of the α-carbon of the residue).
     - PDB_Name_pLDDT.tsv: AlphaFold2 pLDDT. pLDDT is the per-residue prediction confidence metric, not a flexibility measurement.

See the ATLAS about page (https://www.dsimb.inserm.fr/ATLAS/about.html) for details.

 • MOLECULAR DYNAMICS PARAMETERS •
Download the ATLAS GROMACS molecular dynamics parameters (.mdp) and force-field files (CHARMM36m) with https://www.dsimb.inserm.fr/ATLAS/data/download/ATLAS_parameters.zip or using the REST API with `curl -X 'GET' 'https://www.dsimb.inserm.fr/ATLAS/api/MD_parameters' -o MD_parameters.zip`


 • LICENCE •
The ATLAS database is released under a Creative Commons Attribution-NonCommercial 4.0 International license (CC-BY-NC). This license is one of the Creative Commons licenses and allows users to share and adapt the dataset if they give credit to the copyright holder and do not use the dataset for any commercial purposes. See https://creativecommons.org/licenses/by-nc/4.0/ for details.
Please contact us for commercial use.


 • CONTACTS •
For scientific collaboration, please contact:
      Tatiana GALOCHKINA, Associate Professor, Université Paris Cité
      tatiana.galochkina (at) u-paris.fr
  or
      Jean-Christophe GELLY, Professor, Université Paris Cité
      jean-christophe.gelly (at) u-paris.fr 

