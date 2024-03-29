import mdtraj as md
import numpy as np
import torch
import torch_geometric.data as Data
import pickle
from sklearn.neighbors import NearestNeighbors
# Open the XTC trajectory file
def load_data():
    traj = md.load("16pk_A_R1.xtc", top= "16pk_A.pdb")
    # import pdb; pdb.set_trace()
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

def create_pyg_graph(traj, frame_idx):
    # Extract coordinates and residue names for the specified frame
    frame = traj[frame_idx]
    residue_names = [residue.name for residue in frame.top.residues]
    residue_coords = []
    
    # One-hot encode residue features
    node_features = one_hot_encode(residue_names)
    
    for residue in frame.top.residues:
        atom_indices = [atom.index for atom in residue.atoms]
        atom_coords = frame.xyz[0][atom_indices]
        mean_coords = np.mean(atom_coords, axis=0)
        residue_coords.append(mean_coords)

    residue_coords = np.array(residue_coords)

    timepoint = frame_idx
    
    rog = md.compute_rg(frame)
    # import pdb; pdb.set_trace()
    # Construct PyTorch Geometric graph
    graph = Data.Data(x=node_features, coords=residue_coords, time=timepoint, num_nodes=len(residue_names), y = rog[0])
    nn = NearestNeighbors(n_neighbors=5+1, metric='euclidean')
    nn.fit(residue_coords)
    _, indices = nn.kneighbors(residue_coords)
    edge_index = []
    for i in range(len(indices)):
        for j in indices[i][1:]:
            edge_index.append([i, j])  # Add edge between residue i and its k-nearest neighbor j
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # Transpose for PyTorch format
    graph.edge_index = edge_index
    return graph

if __name__ == "__main__":
    # Load trajectory data
    traj = load_data()

    # Create a list to store PyTorch Geometric graphs
    graphs = []

    # Iterate over each frame in the trajectory and create a graph for each timepoint
    for frame_idx in range(traj.n_frames):
        # import pdb; pdb.set_trace()
        graph = create_pyg_graph(traj, frame_idx)
        graphs.append(graph)
    
    # Define the filename for the output .pkl file
    output_filename = "graphs.pkl"

    # Save the list of graphs to the .pkl file
    with open(output_filename, 'wb') as f:
        pickle.dump(graphs, f)

    print(f"Graphs saved to {output_filename}")
    

