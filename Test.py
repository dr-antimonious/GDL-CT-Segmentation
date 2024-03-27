from torch_geometric import __version__ as pyg_version
from torch import __version__ as torch_version
from torch import device, as_tensor, max
from torch.cuda import is_available, can_device_access_peer
from Network import CHD_GNN
from Utilities import CHD_Dataset
from pandas import read_csv
from Graph_Conversion import Convert_To_Image
from matplotlib import pyplot as plt
from torch.distributed import init_process_group
from os import environ

DIRECTORY = '/home/sojo/Documents/ImageCHD/ImageCHD_dataset/'

print(pyg_version)
print(torch_version)
print(is_available())
print(can_device_access_peer(device('cuda:1'), device('cuda:0')))

init_process_group('nccl')
local_rank = int(environ['LOCAL_RANK'])
global_rank = int(environ['RANK'])
batch_size = int(environ['WORLD_SIZE'])

if global_rank == 0:
    print('PyG version: ', pyg_version)
    print('Torch version: ', torch_version)
    print('GPU available: ', is_available())
    print(batch_size)

# gpu = device('cuda:0')
# print(gpu)
# gpu = device('cuda:1')
# print(gpu)
# testing = CHD_GNN().to(gpu)
# metadata = read_csv(filepath_or_buffer = DIRECTORY + 'train_dataset_info.csv')
# dataset = CHD_Dataset(metadata = metadata, directory = DIRECTORY)
# sample = dataset.get(76)

# print(sample.x.type())
# print(sample.edge_index.type())
# print(sample.y.type())

# print(sample.x[0][0].type())
# print(sample.edge_index[0][0].type())
# print(sample.y[0][0].type())

# out = testing(sample.x, sample.edge_index)
# print(out.shape)
# print(out.type())
# _, label = max(out, dim = 1)
# print(label)
# print(label.shape)
# print(label.type())

# result = Convert_To_Image(label, sample.adj_count)
# plt.imshow(result, cmap = 'gray')
# plt.show()