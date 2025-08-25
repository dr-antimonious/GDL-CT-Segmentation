from collections import OrderedDict
from numpy import unique, array
from pandas import read_csv
from torch import load, device, max
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from torchmetrics.classification.accuracy import Accuracy, MulticlassAccuracy
from torchmetrics.classification.precision_recall import Recall, \
    Precision, MulticlassRecall, MulticlassPrecision
from torchmetrics.collections import MetricCollection
from torchmetrics.segmentation.dice import DiceScore
from torchmetrics.segmentation.mean_iou import MeanIoU

from Network import CHD_GNN
from Utilities import __Load_Adjacency__, load_nifti, CHD_Dataset

MODEL_DIRECTORY = "/home/ubuntu/proj/GDL-CT-Segmentation/MODELS/"
DIRECTORY = "/home/ubuntu/proj/ImageCHD_dataset/"

def main():
    dev = device("cuda:0")
    model = CHD_GNN().to(dev)
    checkpoint = load(MODEL_DIRECTORY + "gnn_90.checkpoint",
                      map_location = dev)["MODEL_STATE"]
    
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        nk = k.replace("module.", "")
        new_state_dict[nk] = v
    
    model.load_state_dict(new_state_dict)
    model.eval()

    adjacency = __Load_Adjacency__(DIRECTORY + 'ADJACENCY/')
    test_metadata = read_csv(filepath_or_buffer = DIRECTORY + 'test_dataset_info.csv')
    test_indices: list[int] = test_metadata['index'].unique().tolist()
    test_uniques = unique(array(test_indices))
    test_images = [load_nifti(DIRECTORY + 'IMAGES/', idx) for idx in test_uniques]
    test_labels = [load_nifti(DIRECTORY + 'LABELS/', idx) for idx in test_uniques]
    test_dataset = CHD_Dataset(metadata = test_metadata, adjacency = adjacency,
                               root = DIRECTORY, images = test_images,
                               labels = test_labels)
    test_dataloader = DataLoader(dataset = test_dataset, batch_size = 1,
                                 drop_last = False, shuffle = False,
                                 num_workers = 4, prefetch_factor = 16)
    
    metrics_1 = MetricCollection([
        Accuracy(task = 'multiclass', average = None, num_classes = 8),
        Precision(task = 'multiclass', average = None, num_classes = 8),
        Recall(task = 'multiclass', average = None, num_classes = 8)],
    ).to(dev)
    metrics_2 = MetricCollection([
        DiceScore(num_classes = 8, average = None, input_format = 'index'),
        MeanIoU(num_classes = 8, per_class = True, input_format = 'index')]
    ).to(dev)

    for batch in tqdm(test_dataloader):
        batch = batch.to(dev, non_blocking = True)
        preds = model(x = batch.x, edges = batch.edge_index,
                      batch = batch.batch, b_size = batch.num_graphs)
        _, pred_labels = max(preds, dim = 1)
        
        m1 = metrics_1(preds, batch.y)
        m2 = metrics_2(pred_labels.unsqueeze(0), batch.y.unsqueeze(0))
        
if __name__ == '__main__':
    main()