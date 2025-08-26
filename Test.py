import matplotlib as mpl
mpl.use("Agg")

from collections import OrderedDict
from matplotlib import pyplot as plt
from numpy import unique, array
from pandas import read_csv
from torch import load, device, max
from torch.nn.functional import one_hot
from tqdm import tqdm

from Graph_Conversion import Convert_To_Image
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
    epochs = [43, 49, 50, 80, 83]
    examples = [7483, 9176, 4666]

    adjacency = __Load_Adjacency__(DIRECTORY + 'ADJACENCY/')
    test_metadata = read_csv(filepath_or_buffer = DIRECTORY + 'test_dataset_info.csv')
    test_indices: list[int] = test_metadata['index'].unique().tolist()
    test_uniques = unique(array(test_indices))
    test_images = [load_nifti(DIRECTORY + 'IMAGES/', idx) for idx in test_uniques]
    test_labels = [load_nifti(DIRECTORY + 'LABELS/', idx) for idx in test_uniques]
    test_dataset = CHD_Dataset(metadata = test_metadata, adjacency = adjacency,
                               root = DIRECTORY, images = test_images,
                               labels = test_labels)

    fig, axes = plt.subplots(3, 7, figsize=(20, 4*3))
    for i, example in enumerate(examples):
        data = test_dataset.get(example)
        data = data.to(DEVICE, non_blocking = True)

        axes[i][0].imshow(Convert_To_Image(data.x[:, :, 0], data.adj_count), cmap = 'gray')
        axes[i][0].set_title("Izvorni presjek")
        axes[i][0].axis('off')

        axes[i][1].imshow(Convert_To_Image(data.y, data.adj_count), cmap = 'tab20')
        axes[i][1].set_title("Izvorni presjek")
        axes[i][1].axis('off')

        for j, epoch in enumerate(epochs):
            model = CHD_GNN().to(dev)
            checkpoint = load(MODEL_DIRECTORY + "gnn_" + str(epoch) + ".checkpoint",
                              map_location = dev)["MODEL_STATE"]
            
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                nk = k.replace("module.", "")
                new_state_dict[nk] = v
            
            model.load_state_dict(new_state_dict)
            model.eval()

            preds = model(x = data.x, edges = data.edge_index)
            _, pred_labels = max(preds, dim = 1)

            axes[i][j+2].imshow(Convert_To_Image(pred_labels, data.adj_count), cmap = 'tab20')
            axes[i][j+2].set_title("n_epoha = " + str(epoch))
            axes[i][j+2].axis('off')
    plt.tight_layout()
    plt.savefig('segmentacije.svg', dpi = 300, bbox_inches = 'tight')
        
if __name__ == '__main__':
    main()