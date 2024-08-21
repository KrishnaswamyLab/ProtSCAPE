import mdtraj as md
import numpy as np
import torch
import torch_geometric.data as Data
import pickle
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
def load_data():
    traj = md.load("1ptq_A_R1.xtc", top="1ptq_A.pdb")
    return traj

def one_hot_encode(residues):
    amino_acid_to_index = {amino_acid: i for i, amino_acid in enumerate(set(residues))}
    indices = [amino_acid_to_index[aa] for aa in residues]
    feats = []
    for i in range(len(indices)):
        arr = np.zeros(20)
        arr[indices[i]] = 1
        feats.append(arr)
    return torch.tensor(feats, dtype=torch.float)

def compute_com_dihedrals_matrix(residue_coords):
    num_residues = len(residue_coords)
    dihedral_matrix = np.zeros((num_residues, num_residues))
    
    for i in range(num_residues - 3):
        for j in range(i + 3, num_residues):
            p0 = residue_coords[i]
            p1 = residue_coords[i + 1]
            p2 = residue_coords[i + 2]
            p3 = residue_coords[j]

            # Compute vectors between consecutive COMs
            b0 = -1.0 * (p1 - p0)
            b1 = p2 - p1
            b2 = p3 - p2

            # Normalize b1 so that it does not influence the magnitude of the angle
            b1 /= np.linalg.norm(b1)

            # Compute the normal vectors to the planes formed by (p0, p1, p2) and (p1, p2, p3)
            v = b0 - np.dot(b0, b1) * b1
            w = b2 - np.dot(b2, b1) * b1

            # Compute the dihedral angle
            x = np.dot(v, w)
            y = np.dot(np.cross(b1, v), w)
            angle = np.arctan2(y, x)
            
            # Store the dihedral angle in the matrix
            dihedral_matrix[i, j] = angle
            dihedral_matrix[j, i] = angle  # Symmetric

    return dihedral_matrix

def create_pyg_graph(traj, frame_idx, property):
    frame = traj[frame_idx]
    residue_names = [residue.name for residue in frame.top.residues]
    
    node_features = one_hot_encode(residue_names)
    
    residue_coords = []
    for residue in frame.top.residues:
        atom_indices = [atom.index for atom in residue.atoms]
        atom_coords = frame.xyz[0][atom_indices]
        mean_coords = np.mean(atom_coords, axis=0)
        residue_coords.append(mean_coords)
    
    residue_coords = np.array(residue_coords)
    
    # Compute pairwise distances between COM of residues
    pairwise_distances = np.linalg.norm(residue_coords[:, np.newaxis, :] - residue_coords[np.newaxis, :, :], axis=-1)
    
    # Compute residue COM-based dihedral angles and differences
    dihedral_matrix = compute_com_dihedrals_matrix(residue_coords)
    # import pdb; pdb.set_trace()
    # Combine distances and dihedral differences into a tensor with shape (num_residues, num_residues, 2)
    coords_combined = np.stack((pairwise_distances, dihedral_matrix), axis=-1)

    timepoint = traj.time[frame_idx]
    if property == 'rog':
        y = md.compute_rg(frame)
    elif property == 'sasa':
        y = md.shrake_rupley(frame, mode='residue')
    
    graph = Data.Data(x=node_features, coords=torch.tensor(coords_combined, dtype=torch.float), time=timepoint, num_nodes=len(residue_names), y=y[0])
    
    nn = NearestNeighbors(n_neighbors=5+1, metric='euclidean')
    nn.fit(residue_coords)
    _, indices = nn.kneighbors(residue_coords)
    
    edge_index = []
    for i in range(len(indices)):
        for j in indices[i][1:]:
            edge_index.append([i, j])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    graph.edge_index = edge_index
    return graph

if __name__ == "__main__":
    traj = load_data()
    graphs = []
    property = 'rog'
    
    for frame_idx in tqdm(range(traj.n_frames)):
        graph = create_pyg_graph(traj, frame_idx, property)
        graphs.append(graph)
    
    output_filename = f"graphs{property}_new.pkl"
    with open(output_filename, 'wb') as f:
        pickle.dump(graphs, f)

    print(f"Graphs saved to {output_filename}")
