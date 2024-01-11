from typing import Optional, Tuple

import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add

from lazy import LazinessLayer


class Diffuse(MessagePassing):
    """
    "The Diffusion Submodule" in https://arxiv.org/pdf/2208.07458.pdf.
    Low pass walk with optional weights.

    Init @params
        `in_channels`:        number of input channels
        `out_channels`:       number of output channels
        `trainable_laziness`: whether the "laziness" (probability of not moving to neighbor) is trainable.
        `fixed_weights`:      whether or not to linearly transform the node feature matrix.
    Forward @params
        `x`:                  input graph  [N, in_channels] where N := number of nodes
        `edge_index`:         edge indices [2, E] where E := number of edges
        `edge_weight`:        edge weights [E] where E := number of edges

    Math:
    `A`: Adjacency matrix. Not appeared in the code.
    `W`: Weighted adjacency matrix. i.e., `A` weighted by `edge_weight`.
    `D`: degree matrix. D := diag(d_1,...,d_n) where d_i := sum_j W[i, j].
    `P`: Diffusion matrix. `P := 1/2 I + 1/2 W D^{-1}`
         Note: with learnable laziness, the two 1/2 factors are adjustable. See `LazinessLayer`.

    The forward function is implementing diffusion process:
    `P x = (1/2 I + 1/2 W D^{-1}) x` where `W D^{-1} x` is represented by the variable `propagated`.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 trainable_laziness: bool = False, fixed_weights: bool = True) -> None:

        super().__init__(aggr="add", node_dim=-3)  # "Add" aggregation.
        assert in_channels == out_channels
        self.trainable_laziness = trainable_laziness
        self.fixed_weights = fixed_weights
        if trainable_laziness:
            self.laziness_layer = LazinessLayer(in_channels)
        else:
            self.laziness = 1/2  # Default value for lazy random walk.
        if not self.fixed_weights:
            self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None):

        # Linearly transform node feature matrix.
        if not self.fixed_weights:
            x = self.lin(x)

        # Compute normalization
        edge_index, edge_weight = gcn_norm(
            edge_index=edge_index, edge_weight=edge_weight, num_nodes=x.size(self.node_dim), dtype=x.dtype)

        # Message-passing.
        # In `torch_geometric`, calling `propagate()` internally calls `message()`, `aggregate()` and `update()`.
        propagated = self.propagate(
            edge_index=edge_index, edge_weight=edge_weight, size=None, x=x,
        )
        if not self.trainable_laziness:
            return self.laziness * x + (1 - self.laziness) * propagated
        else:
            return self.laziness_layer(x, propagated)

    def message(self, x_j: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        """
        Quoting `torch_geometric` documentation:
        https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html
        In the message() function, we need to normalize
        the neighboring node features `x_j` by `norm` (i.e., `edge_weight`).
        Here, `x_j` denotes a lifted tensor, which contains the source node features of each edge,
        i.e., the neighbors of each node.
        """
        # x_j has shape [E, out_channels]
        # Normalize node features.
        return edge_weight.view(-1, 1, 1) * x_j

    def message_and_aggregate(self, adj_t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Fuses computations of :func:`message` and :func:`aggregate` into a single function.
        This function will only gets called in case it is implemented and
        propagation takes place based on a :obj:`torch_sparse.SparseTensor`.

        NOTE: Apparently, this is not being called!!!
              Hence, `propagate()` is calling the default `aggregate()` after `message()`.
        """
        return torch.matmul(adj_t, x, reduce=self.aggr)

    def update(self, aggr_out: torch.Tensor) -> torch.Tensor:
        # aggr_out has shape [N, out_channels]
        # Return new node embeddings.
        return aggr_out


def gcn_norm(edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None, num_nodes: Optional[int] = None,
             add_self_loops: bool = False, dtype=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Normalize the edge weight by edge degree?
    """

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                 device=edge_index.device)

    if add_self_loops:
        edge_index, tmp_edge_weight = add_remaining_self_loops(
            edge_index=edge_index, edge_attr=edge_weight, fill_value=1, num_nodes=num_nodes)
        assert tmp_edge_weight is not None
        edge_weight = tmp_edge_weight

    # In undirected graphs, `row` and `col` are equivalent.
    row, col = edge_index[0, :], edge_index[1, :]

    # `scatter_add`: Sums all values from the `src` tensor into `out`
    # at the indices specified in the index tensor along a given axis `dim`.
    deg = scatter_add(src=edge_weight, index=col, dim=0,
                      out=None, dim_size=num_nodes)

    # Use `tensor.pow()` rather than `tensor.pow_()`! The latter will modify `tensor` in-place!
    # I believe degree matrices are positive semi-definite. Safe to take sqrt.
    deg_inv = deg.pow(-1)
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)

    return edge_index, deg_inv[row] * edge_weight
