from torch_geometric import __version__ as pyg_version
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree

from torch import __version__ as torch_version
from torch import max as torchmax
from torch import FloatTensor, LongTensor, zeros, long, bincount, load, tensor
from torch.cuda import is_available, empty_cache, device_count
from torch.nn import CrossEntropyLoss

from pandas import read_csv
from os import environ
from tqdm import tqdm
from numpy import isin, argwhere, double, uint16, array, append
from numpy import zeros as npyzeros

from matplotlib.pyplot import imshow, show, figure

from Network import CHD_GNN
from Utilities import CHD_Dataset, __Load_Adjacency__
from Metrics import Accuracy_Util
from Graph_Conversion import Convert_To_Image

def main():
    DIRECTORY = '/home/sojo/Documents/ImageCHD/ImageCHD_dataset/'
    SPLIT_SIZE = 1
    BATCH_SIZE = 1
    NUM_WORKERS = 8
    PREFETCH_FACTOR = 8

    environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    environ['CUDA_LAUNCH_BLOCKING'] = '1'

    print('PyG version: ', pyg_version)
    print('Torch version: ', torch_version)
    print('GPU available: ', is_available())
    print('GPU count: ', device_count())
    empty_cache()

    adjacency = __Load_Adjacency__(DIRECTORY + 'ADJACENCY/')

    max_degree = -1
    for key in adjacency:
        d = degree(index = LongTensor(adjacency[key][1]), num_nodes = 512 * int(key), dtype = long)
        max_degree = max(max_degree, int(d.max()))
    
    deg = zeros(max_degree + 1, dtype = long)
    for key in adjacency:
        d = degree(index = LongTensor(adjacency[key][1]), num_nodes = 512 * int(key), dtype = long)
        deg += bincount(d, minlength = deg.numel())
    
    test_metadata = read_csv(filepath_or_buffer = DIRECTORY + 'test_dataset_info.csv')
    test_dataset = CHD_Dataset(metadata = test_metadata, directory = DIRECTORY, adjacency = adjacency)
    test_dataloader = DataLoader(dataset = test_dataset, batch_size = BATCH_SIZE,
                                num_workers = NUM_WORKERS, persistent_workers = True,
                                shuffle = True, pin_memory = True,
                                prefetch_factor = PREFETCH_FACTOR)
    
    loss_module = CrossEntropyLoss(weight = FloatTensor([11108352000./10420281390.,
                                                         11108352000./688070610.]).to('cuda:1'))
    
    epochs = ['5', '10', '15', '20', '25', '30', '35', '40', '45', '50']
    borders = [750, 800, 850, 900, 950, 1000]

    for epoch in epochs:
        print('------------------------------------------')
        print('EPOCH: ' + epoch)

        test_loss = npyzeros(6)
        test_metrics = npyzeros(60).reshape((6, 10))
        test_counts = npyzeros(60).reshape((6, 10))
        
        model = CHD_GNN(deg = deg, split_size = SPLIT_SIZE)
        model.MoveEncoder('cuda:0')
        model.MoveDecoder('cuda:1')
        model.load_state_dict(load('/home/sojo/Documents/CTProject/GDL-CT-Segmentation/MODELS/gnn_' + epoch + '.checkpoint'))
        for batch in tqdm(test_dataloader):
            model.eval()
            split_sizes = []
            adj_split_sizes = []
            for i in range(0, batch.adj_count.__len__(), SPLIT_SIZE):
                temp = sum(batch.adj_count[i:i+SPLIT_SIZE]) * 512
                split_sizes.append(temp)
                adj_split_sizes.append(temp * 8)

            preds = model(batch.x.to('cuda:0'), batch.edge_index.to('cuda:0'), batch.adj_count, split_sizes, adj_split_sizes)
            _, pred_labels = torchmax(preds, dim = 1)

            imshow(Convert_To_Image(batch.x, batch.adj_count[0]))
            figure()
            imshow(Convert_To_Image(pred_labels, batch.adj_count[0]))
            show()
            exit()

            for i in range(0, 6):
                truth = tensor(batch.x, dtype=long, device = 'cuda:1')
                truth[truth < borders[i]] = 0
                truth[truth > 0] = 1
                truth = truth.reshape(-1)
                loss = loss_module(preds, truth)

                metrics = Accuracy_Util(truth, pred_labels)
                mask = argwhere(isin(metrics, -1)^True).reshape(-1)

                test_loss[i] += loss.item()
                test_metrics[i][mask] += metrics[mask]
                test_counts[i][mask] += 1

                empty_cache()

        for i in range(0, 6):
            test_loss[i] /= len(test_dataset)
            print('-----TESTING METRICS ' + str(borders[i]) + '-----')
            print('Loss: ', test_loss[i])

            for j in range(0, 3):
                print('Accuracy ', j, ': ', test_metrics[i][j] / test_counts[i][j])
            for j in range(3, 6):
                print('Precision ', j-3, ': ', test_metrics[i][j] / test_counts[i][j])
            for j in range(6, 9):
                print('Recall ', j-6, ': ', test_metrics[i][j] / test_counts[i][j])
            print('F1: ', test_metrics[i][9] / test_counts[i][9])
        
        empty_cache()
        
if __name__ == '__main__':
    main()