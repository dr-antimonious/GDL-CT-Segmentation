from torch import cat
from torch.amp.autocast_mode import autocast
from torch.nn import Module, PReLU, Linear, ModuleList
from torch_geometric.nn import SSGConv, BatchNorm, Sequential
from torch_geometric.nn.aggr import PowerMeanAggregation

class CHD_GNN(Module):
    r"""
        PyTorch Geometric GNN used for coronary-CT segmentation \
        of images with visible CHDs.
    """

    def linear_block(self, in_channels: int, out_channels: int) \
        -> Sequential:
        return Sequential('x', [
            (Linear(in_channels, out_channels), 'x -> x'),
            (BatchNorm(out_channels), 'x -> x'),
            (PReLU(out_channels), 'x -> x')
        ])
    
    def ssgc_block(self, in_channels: int, out_channels: int,
                   alpha: float, K: int) -> Sequential:
        return Sequential('x, edge_index', [
            (SSGConv(in_channels,
                     out_channels,
                     alpha, K),
                     'x, edge_index -> x'),
            (BatchNorm(out_channels), 'x -> x'),
            (PReLU(out_channels), 'x -> x')
        ])
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.layers = ModuleList([
            self.linear_block(1, 8),
            self.linear_block(8, 16),
            self.ssgc_block(16, 16, 0.05, 3),
            self.ssgc_block(32, 16, 0.05, 4),
            self.ssgc_block(32, 16, 0.05, 4),
            self.ssgc_block(32, 16, 0.05, 3),
            self.linear_block(48, 16),
            self.linear_block(24, 8)
        ])

    @autocast('cuda')
    def forward(self, x, adj_matrix):
        r"""
            Arguments:
                x (Tensor): Source coronary-CT image as a graph.
                adj_matrix (Tensor): Adjacency matrix of the x graph.
                device: Second GPU.
            
            Returns:
                out (Tensor): Segmentation result as a graph.
        """
        x1 = self.layers[0].forward(x = x)
        x2 = self.layers[1].forward(x = x1)
        x3 = self.layers[2].forward(
            x = x2,
            edge_index = adj_matrix
        )
        x4 = self.layers[3].forward(
            x = cat([x2, x3], dim = 1),
            edge_index = adj_matrix
        )
        x5 = self.layers[4].forward(
            x = cat([x3, x4], dim = 1),
            edge_index = adj_matrix
        )
        x6 = self.layers[5].forward(
            x = cat([x4, x5], dim = 1),
            edge_index = adj_matrix
        )
        x7 = self.layers[6].forward(x = cat([x2, x5, x6], dim = 1))
        x8 = self.layers[7].forward(x = cat([x1, x7], dim = 1))

        return x8