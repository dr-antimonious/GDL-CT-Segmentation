import torch_geometric
import torch
from Network import CHD_GNN
print(torch_geometric.__version__)
print(torch.__version__)
print(torch.cuda.is_available())
testing = CHD_GNN()