import mdtraj as md
import numpy as np
import torch
import torch_geometric.data as Data
import pickle
from sklearn.neighbors import NearestNeighbors
# Open the XTC trajectory file
def load_data():
    # Load the closed conformation PDB
    open_traj = md.load_pdb("4ake.pdb")
    # gt_traj = md.load_pdb('MurD/5A5F_full.pdb')
    # Load the open conformation PDB
    closed_traj = md.load_pdb("1ake.pdb")
    # Remove unit cell information from both trajectories
    # closed_traj.unitcell_vectors = None
    # closed_traj.unitcell_vectors = None
    # Concatenate the trajectories
    # traj = md.join([closed_traj, open_traj])
    import pdb; pdb.set_trace()
    return open_traj, closed_traj


def one_hot_encode(residues):
    amino_acid_to_index = {amino_acid: i for i, amino_acid in enumerate(set(residues))}
    indices = [amino_acid_to_index[aa] for aa in residues]
    # import pdb; pdb.set_trace()
    feats = []
    for i in range(len(indices)):
        arr = np.zeros(21)
        arr[indices[i]] = 1
        feats.append(arr)
    return torch.tensor(feats, dtype=torch.float)



def create_pyg_graph(traj, frame_idx, property):
    # Extract coordinates and residue names for the specified frame
    frame = traj[frame_idx]
    residue_names = [residue.name for residue in frame.top.residues]
    residue_coords = []
    # with open('1ab1_A_RMSD.tsv', 'r') as file:
    #     rmsd_data = file.readlines()
    # rmsd_data = [line.split('\t')[1] for line in rmsd_data]
    # rmsd_data = rmsd_data[1:]
    # import pdb; pdb.set_trace()
    # One-hot encode residue features
    
    node_features = one_hot_encode(residue_names)
    # Embedding layer
    # Randomly initialize node features
    # node_features = torch.randn(len(residue_names), 3)
    # embedding = torch.nn.Embedding(len(residue_names), 3)
    
    # # Apply embedding to node features
    # node_features = embedding
    for residue in frame.top.residues:
        atom_indices = [atom.index for atom in residue.atoms]
        atom_coords = frame.xyz[0][atom_indices]
        mean_coords = np.mean(atom_coords, axis=0)
        residue_coords.append(mean_coords)

    residue_coords = np.array(residue_coords)

    timepoint = traj.time[frame_idx]
    if property == 'rog':
        y = md.compute_rg(frame)
    elif property == 'sasa':
        y = md.shrake_rupley(frame, mode='residue')
    # elif property == 'rmsd':
    #     y = rmsd_data[frame_idx]
    # import pdb; pdb.set_trace()
    # Construct PyTorch Geometric graph
    graph = Data.Data(x=node_features, coords=residue_coords, time=timepoint, num_nodes=len(residue_names), y = y[0])
    nn = NearestNeighbors(n_neighbors=5+1, metric='euclidean')
    nn.fit(residue_coords)
    _, indices = nn.kneighbors(residue_coords)
    edge_index = []
    for i in range(len(indices)):
        for j in indices[i][1:]:
            edge_index.append([i, j])  # Add edge between residue i and its k-nearest neighbor j
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # Transpose for PyTorch format
    graph.edge_index = edge_index
    # print(graph)
    return graph

def combine_graphs(graph1, graph2):
    combined_x = torch.cat([graph1.x, graph2.x], dim=0)
    combined_coords = np.vstack([graph1.coords, graph2.coords])
    combined_time = torch.tensor([graph1.time, graph2.time])
    combined_y = torch.tensor([graph1.y, graph2.y])

    edge_index1 = graph1.edge_index
    edge_index2 = graph2.edge_index + graph1.num_nodes
    combined_edge_index = torch.cat([edge_index1, edge_index2], dim=1)

    combined_graph = Data.Data(
        x=combined_x, 
        coords=combined_coords, 
        time=combined_time, 
        num_nodes=graph1.num_nodes + graph2.num_nodes, 
        y=combined_y,
        edge_index=combined_edge_index
    )
    return combined_graph

if __name__ == "__main__":
    # Load trajectory data
    open_traj, closed_traj = load_data()

    # Create a list to store PyTorch Geometric graphs
    graphs = []
    property = 'rog'
    import pdb; pdb.set_trace()
    # Iterate over each frame in the trajectory and create a graph for each timepoint
    for frame_idx in range(open_traj.n_frames):
        # import pdb; pdb.set_trace()
        open_graph = create_pyg_graph(open_traj, frame_idx, property)
        closed_graph = create_pyg_graph(closed_traj, frame_idx, property)
        combined_graph = combine_graphs(open_graph, closed_graph)
        graphs.append(combined_graph)
    
    # Define the filename for the output .pkl file
    output_filename = f"combined_graphs_ADK.pkl"

    # Save the list of graphs to the .pkl file
    with open(output_filename, 'wb') as f:
        pickle.dump(graphs, f)

    print(f"Graphs saved to {output_filename}")
    

