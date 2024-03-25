from torch.nn import Module
from torch_geometric.nn import GATv2Conv

class CHD_GNN(Module):
    r"""
        PyTorch Geometric GNN used for coronary-CT segmentation \
        of images with visible CHDs.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # GAT layers
        self.gat_1_to_1 =       GATv2Conv(1, 1,
                                          fill_value = 'sum',
                                          dropout = 0.25)
        self.gat_1_to_2 =       GATv2Conv(1, 2,
                                          fill_value = 'sum')
        self.gat_2_to_2_n1 =    GATv2Conv(2, 2,
                                          fill_value = 'sum',
                                          dropout = 0.25)
        self.gat_2_to_2_n2 =    GATv2Conv(2, 2,
                                          fill_value = 'sum',
                                          dropout = 0.25)
        self.gat_2_to_4 =       GATv2Conv(2, 4,
                                          fill_value = 'sum')
        self.gat_4_to_4_n1 =    GATv2Conv(4, 4,
                                          fill_value = 'sum',
                                          dropout = 0.25)
        self.gat_4_to_4_n2 =    GATv2Conv(4, 4,
                                          fill_value = 'sum',
                                          dropout = 0.25)
        self.gat_4_to_8 =       GATv2Conv(4, 8,
                                          fill_value = 'sum')
        self.gat_8_to_8 =       GATv2Conv(8, 8,
                                          fill_value = 'sum',
                                          dropout = 0.25)

    def forward(self, x, adj_matrix):
        r"""
            Arguments:
                x (Tensor): Source coronary-CT image as a graph.
                adj_matrix (Tensor): Adjacency matrix of the x graph.
            
            Returns:
                out (Tensor): Segmentation result as a graph.
        """
        out = self.gat_1_to_1(x = x, edge_index = adj_matrix)
        out = self.gat_1_to_2(x = x, edge_index = adj_matrix)
        out = out.tanh()

        out = self.gat_2_to_2_n1(x = x, edge_index = adj_matrix)
        out = self.gat_2_to_2_n2(x = x, edge_index = adj_matrix)
        out = self.gat_2_to_4(x = x, edge_index = adj_matrix)
        out = out.tanh()

        out = self.gat_4_to_4_n1(x = x, edge_index = adj_matrix)
        out = self.gat_4_to_4_n2(x = x, edge_index = adj_matrix)
        out = self.gat_4_to_8(x = x, edge_index = adj_matrix)
        out = out.tanh()

        out = self.gat_8_to_8(x = x, edge_index = adj_matrix)
        return out