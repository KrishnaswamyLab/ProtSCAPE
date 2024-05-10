This trajectory dataset was downloaded from the ATLAS database (https://www.dsimb.inserm.fr/ATLAS).  

ATLAS gathers standardized molecular dynamics simulations of protein structures accompanied by their analysis in the form of interactive diagrams and trajectory visualization. All the raw trajectories as well as the results of analysis are available for download.


 • ARCHIVE CONTENT •
     - PDB_Name.pdb: Protein structure in PDB format used to launch the molecular dynamics (after energy minimisation and equilibrations).
     - PDB_Name_R{1,2,3}.tpr: File containing the starting structure of the simulation, the molecular topology and all the simulation parameters.
     - PDB_Name_R{1,2,3}.xtc: MD trajectory file containing only the protein (no solution molecules), with PBC removed and frames aligned. Each trajectory is of 100 ns with frames saved every 10 ps (10,000 frames in total).

See the ATLAS about page (https://www.dsimb.inserm.fr/ATLAS/about.html) for details.


 • MOLECULAR DYNAMICS PARAMETERS •
Download the ATLAS GROMACS molecular dynamics parameters (.mdp) and force-field files (CHARMM36m) with https://www.dsimb.inserm.fr/ATLAS/data/download/ATLAS_parameters.zip or using the REST API with `curl -X 'GET' 'https://www.dsimb.inserm.fr/ATLAS/api/MD_parameters' -o MD_parameters.zip`


 • LICENSE •
The ATLAS database is released under a Creative Commons Attribution-NonCommercial 4.0 International license (CC-BY-NC). This license is one of the Creative Commons licenses and allows users to share and adapt the dataset if they give credit to the copyright holder and do not use the dataset for any commercial purposes. See https://creativecommons.org/licenses/by-nc/4.0/ for details.
Please contact us for commercial use.


 • CONTACTS •
For scientific collaboration, please contact:
      Tatiana GALOCHKINA, Associate Professor, Université Paris Cité
      tatiana.galochkina (at) u-paris.fr
  or
      Jean-Christophe GELLY, Professor, Université Paris Cité
      jean-christophe.gelly (at) u-paris.fr 

