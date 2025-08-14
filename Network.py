from torch import tensor, softmax
from torch.amp.autocast_mode import autocast
from torch.nn import Module, PReLU, Linear, ModuleList, ParameterList, Parameter
from torch_geometric.nn import SSGConv, BatchNorm, Sequential

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
            self.linear_block(1, 32),
            self.linear_block(32, 64),
            self.ssgc_block(64, 64, 0.05, 3),
            self.ssgc_block(64, 64, 0.05, 4),
            self.ssgc_block(64, 64, 0.05, 3),
            self.linear_block(64, 32),
            Linear(32, 8)
        ])

        self.params = ParameterList([
            Parameter(tensor([0.5])),
            Parameter(tensor([0.5])),
            Parameter(tensor([0.0, 0.0, 0.0])),
            Parameter(tensor([0.5]))
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
            x = (1 - self.params[0]) * x2 + self.params[0] * x3,
            edge_index = adj_matrix
        )
        x5 = self.layers[4].forward(
            x = (1 - self.params[1]) * x3 + self.params[1] * x4,
            edge_index = adj_matrix
        )
        w = softmax(self.params[2], dim = 0)
        x6 = self.layers[5].forward(x = w[0] * x2 + w[1] * x4 + w[2] * x5)
        x7 = self.layers[6].forward((1 - self.params[3]) * x1 + self.params[3] * x6)

        return x7