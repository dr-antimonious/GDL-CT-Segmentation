from collections import OrderedDict
from numpy import unique, array
from pandas import read_csv
from torch import load, device, max
from torch.nn.functional import one_hot
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification.precision_recall import Recall, Precision
from torchmetrics.collections import MetricCollection

from Network import CHD_GNN
from Utilities import __Load_Adjacency__, load_nifti, CHD_Dataset

MODEL_DIRECTORY = "/home/ubuntu/proj/GDL-CT-Segmentation/MODELS/"
DIRECTORY = "/home/ubuntu/proj/ImageCHD_dataset/"
NUM_CLASSES = 8

def compute_iou_dice(pred, target, num_classes):
    pred_oh = one_hot(pred, num_classes = num_classes).bool()
    target_oh = one_hot(target, num_classes = num_classes).bool()
    tp = (pred_oh & target_oh).sum().float()
    fp = (pred_oh & ~target_oh).sum().float()
    fn = (~pred_oh & target_oh).sum().float()
    union = tp + fp + fn
    valid = union > 0
    iou = tp / (union + 1e-7)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-7)
    mean_iou = iou[valid].mean().item()
    mean_dice = dice[valid].mean().item()
    return mean_iou, mean_dice

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
    
    metrics = MetricCollection([
        Accuracy(task = 'multiclass', average = None, num_classes = 8),
        Precision(task = 'multiclass', average = None, num_classes = 8),
        Recall(task = 'multiclass', average = None, num_classes = 8)],
    ).to(dev)

    for batch in tqdm(test_dataloader):
        batch = batch.to(dev, non_blocking = True)
        preds = model(x = batch.x, edges = batch.edge_index,
                      batch = batch.batch, b_size = batch.num_graphs)
        _, pred_labels = max(preds, dim = 1)
        
        m1 = metrics(preds, batch.y)
        dice, iou = compute_iou_dice(pred_labels, batch.y, 8)
        print(m1, dice, iou)
        
if __name__ == '__main__':
    main()