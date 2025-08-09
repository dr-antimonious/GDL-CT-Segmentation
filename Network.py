from torch import cat, Tensor
from torch.amp.autocast_mode import autocast
from torch.nn import Module, ReLU, Linear, ModuleList
from torch_geometric.nn import PNAConv, BatchNorm, Sequential

class CHD_GNN(Module):
    r"""
        PyTorch Geometric GNN used for coronary-CT segmentation \
        of images with visible CHDs.
    """

    def block(self, in_channels: int, out_channels: int):
        towers = min(in_channels, out_channels)
        if in_channels % towers != 0:
            towers //= 2

        return Sequential('x, edge_index', [
            (PNAConv(in_channels = in_channels,
                     out_channels = out_channels,
                     aggregators = self.aggregators,
                     scalers = self.scalers,
                     deg = self.deg,
                     towers = towers,
                     divide_input = True,
                     act = None,
                     train_norm = True), 'x, edge_index -> x'),
            (BatchNorm(out_channels), 'x -> x'),
            (ReLU(), 'x -> x')
        ])
    
    def layer(self, in_channels: int, out_channels: int,
              hidden_channels: int|None = None):
        if hidden_channels is None:
            hidden_channels = out_channels

        return Sequential('x, edge_index', [
            (self.block(in_channels, hidden_channels),
             'x, edge_index -> x'),
            (self.block(hidden_channels, out_channels),
             'x, edge_index -> x')
        ])
    
    def create_encoder(self):
        self.layers.append(self.layer(1, 8).to('cuda:0'))
        self.layers.append(self.layer(8, 16).to('cuda:0'))
        self.layers.append(self.layer(16, 32).to('cuda:0'))
        self.layers.append(self.layer(32, 64).to('cuda:0'))

    def create_decoder(self):
        self.layers.append(self.layer(96, 32, 64).to('cuda:1'))
        self.layers.append(self.layer(48, 16, 32).to('cuda:1'))
        self.layers.append(self.layer(24, 8, 16).to('cuda:1'))
        self.layers.append(Linear(8, 8).to('cuda:1'))
    
    @autocast('cuda')
    def encode(self, x: Tensor, edge_index: Tensor) -> list[Tensor]:
        out = []
        
        for i in range(4):
            out.append(self.layers[i].forward(
                x = x if i == 0 else out[i - 1],
                edge_index = edge_index
            ))
        
        return out
    
    @autocast('cuda')
    def decode(self, x: list[Tensor], edge_index: Tensor) -> Tensor:
        out = x.pop()

        for i in range(4, len(self.layers) - 1):
            out = self.layers[i].forward(
                x = cat([x.pop(), out], dim = 1)
                    if i != len(self.layers) - 1 else out,
                edge_index = edge_index
            )

        out = self.layers[len(self.layers) - 1].forward(
            input = out
        )

        return out

    def __init__(self, deg, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.deg = deg
        self.aggregators = ['mean', 'min', 'max', 'std']
        self.scalers = ['identity', 'amplification', 'attenuation']
        self.layers = ModuleList()
        self.create_encoder()
        self.create_decoder()

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
        x = self.encode(x = x, edge_index = adj_matrix)
        x = [xx.to('cuda:1') for xx in x]
        adj_matrix = adj_matrix.to('cuda:1')
        x = self.decode(x = x, edge_index = adj_matrix)
        return x