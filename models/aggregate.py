import torch


class Aggregate(torch.nn.Module):
    """
    The aggragation submodule in https://arxiv.org/pdf/2208.07458.pdf.

    Quoting the paper:
    While many approaches may be applied to aggregate node-level features into graph-level
    features such as max, mean, sum pooling, or the more powerful TopK
    and attention pooling, we follow the statistical moment aggregation
    and leave exploration of other pooling methods to future work.
    """

    def __init__(self, aggregation_method: str = 'statistical_moments'):
        super(Aggregate, self).__init__()
        self.aggregation_method = aggregation_method

    def forward(self, *args, **kwargs):
        if self.aggregation_method == 'statistical_moments':
            return statistical_moments(*args, **kwargs)


def statistical_moments(graph: torch.Tensor, batch_indices: torch.Tensor,
                        moments_returned: int = 4, inf_val: int = 1e15) -> torch.Tensor:
    """
    Compute specified statistical coefficients for each feature of each graph passed.
        `graph`: The feature tensors of disjoint subgraphs within a single graph.
            [N, in_channels] where N := number of nodes
        `batch_indices`: [B].
        `moments_returned`: Specifies the number of statistical measurements to compute.
            If 1, only the mean is returned. If 2, the mean and variance.
            If 3, the mean, variance, and skew. If 4, the mean, variance, skew, and kurtosis.
        `inf_val`: A value bigger than this shall be treated as infinity.
    """

    # Step 1: Aggregate the features of each mini-batch graph into its own tensor.
    graph_features = [torch.zeros(0).to(graph.device)
                      for _ in range(torch.max(batch_indices) + 1)]

    for i, node_features in enumerate(graph):
        # Sort the graph features by graph, according to batch_indices.
        # For each graph, create a tensor whose first row is the first element of each feature, etc.
        # print("node features are", node_features)

        if len(graph_features[batch_indices[i]]) == 0:
            # If this is the first feature added to this graph, fill it in with the features.
            # .view(-1,1,1) changes [x1,x2,x3] to [[x1],[x2],[x3]], so that we can add each column to the respective row.
            graph_features[batch_indices[i]] = node_features.view(-1, 1, 1)
        else:
            graph_features[batch_indices[i]] = torch.cat(
                (graph_features[batch_indices[i]], node_features.view(-1, 1, 1)), dim=1)  # concatenates along columns

    # Instatiate the correct set of moments to return.
    assert moments_returned in [1, 2, 3, 4], \
        "`statistical_moments`: only supports `moments_returned` of the following values: 1, 2, 3, 4."
    moments_keys = ['mean', 'variance', 'skew', 'kurtosis']
    moments_keys = moments_keys[:moments_returned]

    statistical_moments = {}
    for key in moments_keys:
        statistical_moments[key] = torch.zeros(0).to(graph)

    for data in graph_features:

        data = data.squeeze()

        mean = torch.mean(data, dim=1, keepdim=True)

        if moments_returned >= 1:
            statistical_moments['mean'] = torch.cat(
                (statistical_moments['mean'], mean.T), dim=0
            )

        # produce matrix whose every row is data row - mean of data row
        std = data - mean

        # variance: difference of u and u mean, squared element wise, summed and divided by n-1
        variance = torch.mean(std**2, axis=1)
        if moments_returned >= 2:
            statistical_moments['variance'] = torch.cat(
                (statistical_moments['variance'], variance[None, ...]), dim=0
            )

        # skew: 3rd moment divided by cubed standard deviation (sd = sqrt variance), with correction for division by zero (inf -> 0)
        skew = variance = torch.mean(std**3, axis=1)
        # Multivalued tensor division by zero produces inf.
        skew[skew > inf_val] = 0
        # Single valued division by 0 produces nan.
        skew[skew != skew] = 0
        if moments_returned >= 3:
            statistical_moments['skew'] = torch.cat(
                (statistical_moments['skew'], skew[None, ...]), dim=0
            )

        # kurtosis: fourth moment, divided by variance squared. Using Fischer's definition to subtract 3 (default in scipy)
        kurtosis = torch.mean(std**4, axis=1) - 3
        kurtosis[kurtosis > inf_val] = -3
        kurtosis[kurtosis != kurtosis] = -3
        if moments_returned >= 4:
            statistical_moments['kurtosis'] = torch.cat(
                (statistical_moments['kurtosis'], kurtosis[None, ...]), dim=0
            )

    # Concatenate into one tensor.
    statistical_moments = torch.cat(
        [statistical_moments[key] for key in moments_keys], axis=1)

    return statistical_moments
