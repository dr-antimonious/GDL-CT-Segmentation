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
            self.linear_block(6, 32),
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

        alpha = self.params[0].float()
        res = (1 - alpha) * x2.float() + alpha * x3.float()
        x4 = self.layers[3].forward(
            x = res.to(x3.dtype),
            edge_index = adj_matrix
        )

        alpha = self.params[1].float()
        res = (1 - alpha) * x3.float() + alpha * x4.float()
        x5 = self.layers[4].forward(
            x = res.to(x4.dtype),
            edge_index = adj_matrix
        )

        alpha = self.params[2].float()
        w = softmax(alpha, dim = 0)
        res = w[0] * x2.float() + w[1] * x4.float() + w[2] * x5.float()
        x6 = self.layers[5].forward(x = res.to(x5.dtype))

        alpha = self.params[3].float()
        res = (1 - alpha) * x1.float() + alpha * x6.float()
        x7 = self.layers[6].forward(res.to(x6.dtype))

        return x7