from collections import OrderedDict
from numpy import unique, array, zeros
from pandas import DataFrame, read_csv
from torch import load, device, max
from torch.nn.functional import one_hot
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification.precision_recall import Precision
from torchmetrics.collections import MetricCollection

from Network import CHD_GNN
from Utilities import __Load_Adjacency__, load_nifti, CHD_Dataset

MODEL_DIRECTORY = "/home/ubuntu/proj/GDL-CT-Segmentation/MODELS/"
DIRECTORY = "/home/ubuntu/proj/ImageCHD_dataset/"
DEVICE = "cuda:0"
FREQS = array([7610724117.0, 35496879.0,
               37714647.0, 83133666.0,
               105607395.0, 144954300.0,
               57576912.0, 53353236.0])
W_STEP = FREQS.sum() / FREQS
WEIGHTS = W_STEP / W_STEP.sum()

def compute_iou_dice(pred, target, num_classes):
    pred_oh = one_hot(pred, num_classes = num_classes).bool()
    target_oh = one_hot(target, num_classes = num_classes).bool()
    tp = (pred_oh & target_oh).sum(dim = 0).float()
    fp = (pred_oh & ~target_oh).sum(dim = 0).float()
    fn = (~pred_oh & target_oh).sum(dim = 0).float()
    union = tp + fp + fn
    valid = (union > 0).cpu().numpy()
    iou = tp[valid] / (union[valid] + 1e-7)
    dice = 2 * tp[valid] / (2 * tp[valid] + fp[valid] + fn[valid] + 1e-7)
    return iou, dice, valid

def main():
    dev = device(DEVICE)
    epochs = list(range(35, 51))
    epochs.extend(range(70, 86))
    results = {}

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
        Precision(task = 'multiclass', average = None, num_classes = 8)],
    ).to(dev)

    for epoch in epochs:
        model = CHD_GNN().to(dev)
        checkpoint = load(MODEL_DIRECTORY + "gnn_" + str(epoch) + ".checkpoint",
                        map_location = dev)["MODEL_STATE"]
        
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            nk = k.replace("module.", "")
            new_state_dict[nk] = v
        
        model.load_state_dict(new_state_dict)
        model.eval()

        tot_iou = zeros(8, dtype = float)
        tot_dice = zeros(8, dtype = float)
        tot_acc = zeros(8, dtype = float)
        tot_prec = zeros(8, dtype = float)
        tot_valid = zeros(8)

        for batch in tqdm(test_dataloader):
            batch = batch.to(dev, non_blocking = True)
            preds = model(x = batch.x, edges = batch.edge_index,
                        batch = batch.batch, b_size = batch.num_graphs)
            _, pred_labels = max(preds, dim = 1)
            
            m = metrics(preds, batch.y)
            iou, dice, valid = compute_iou_dice(pred_labels, batch.y, 8)

            tot_iou[valid] += iou.cpu().numpy()
            tot_dice[valid] += dice.cpu().numpy()
            tot_acc[valid] += m['MulticlassAccuracy'].cpu().numpy()[valid]
            tot_prec[valid] += m['MulticlassPrecision'].cpu().numpy()[valid]
            tot_valid += valid
        
        tot_iou /= tot_valid
        tot_dice /= tot_valid
        tot_acc /= tot_valid
        tot_prec /= tot_valid

        results.update([("MODEL_" + str(epoch), {
            "IoU": tot_iou,
            "Dice": tot_dice,
            "Accuracy": tot_acc,
            "Precision": tot_prec,
            "Mean_IoU": (tot_iou * WEIGHTS).sum(),
            "Mean_Dice": (tot_dice * WEIGHTS).sum(),
            "Mean_Acc": (tot_acc * WEIGHTS).sum(),
            "Mean_Prec": (tot_prec * WEIGHTS).sum()
        })])
    
    save = DataFrame(results)
    save.to_csv(MODEL_DIRECTORY + "results.csv")
        
if __name__ == '__main__':
    main()