from torch import __version__ as torch_version
from torch import max as torchmax
from torch import Tensor, FloatTensor, save, load, no_grad, sum
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.cuda import is_available, device_count, set_device
from torch.distributed import init_process_group, destroy_process_group, \
    all_reduce, ReduceOp, barrier
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn import CrossEntropyLoss
from torch.nn.functional import softmax, one_hot
from torch.nn.parallel import DistributedDataParallel
from torch.nn import SyncBatchNorm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard.writer import SummaryWriter

from torch_geometric import __version__ as pyg_version
from torch_geometric.loader import DataLoader

from torchmetrics.classification.accuracy import Accuracy, MulticlassAccuracy
from torchmetrics.classification.precision_recall import Recall, \
    Precision, MulticlassRecall, MulticlassPrecision
from torchmetrics.collections import MetricCollection
from torchmetrics.segmentation.dice import DiceScore
from torchmetrics.segmentation.mean_iou import MeanIoU

from os import environ, getenv
from os.path import exists
from pandas import read_csv
from shutil import copy
from tqdm import tqdm

from Network import CHD_GNN
from Utilities import CHD_Dataset, __Load_Adjacency__

LR              = 1e-3
T0              = 10
TMULT           = 2
MIN_LR          = 1e-5
DECAY           = 0.9
SMOOTH          = 1e-6
BATCH_SIZE      = 32
NUM_WORKERS     = 16
PREFETCH_FACTOR = 2
EPOCHS          = 200
WORLD_SIZE      = device_count()
M_NAMES         = ['Accuracy ', 'Recall ', 'Precision ',
                   'F1-score ', 'IoU-score ']
M_LOGS          = ['_accuracy_', '_recall_', '_precision_',
                   '_f1_score_', '_iou_score_']
M_LOOP          = ['ce_loss', 'dice_loss', 'loss', 'accuracy',
                   'recall', 'precision', 'dice', 'iou']
M1_NAMES        = [MulticlassAccuracy.__name__,
                   MulticlassRecall.__name__,
                   MulticlassPrecision.__name__]
M2_NAMES        = [DiceScore.__name__, MeanIoU.__name__]
CHECKPOINT      = 'MODELS/gnn.checkpoint'

PRODUCTION_STR  = getenv('GDL_CT_SEG_PROD')
DIRECTORY       = getenv('GDL_CT_SEG_DIR')
assert PRODUCTION_STR is not None
assert DIRECTORY is not None
PRODUCTION      = PRODUCTION_STR.lower() == 'true'

def Dice(preds: Tensor, y: Tensor) -> Tensor:
    probs = softmax(preds, dim = 1)[:, 1:]
    targs = one_hot(y, num_classes = 8).float()[:, 1:]

    dims = (0)
    intersection = sum(probs * targs, dims)
    union = sum(probs + targs, dims)

    dice_coef = (2. * intersection + SMOOTH) / (union + SMOOTH)
    dice_loss = 1. - dice_coef
    return dice_loss.mean()

def loader_loop(rank: int, train: bool, dataloader: DataLoader,
                model: DistributedDataParallel, scaler: GradScaler|None,
                loss_module: CrossEntropyLoss, optimizer: ZeroRedundancyOptimizer|None,
                metrics_1: MetricCollection, metrics_2: MetricCollection) -> dict:
    metrics = {M_LOOP[i]: FloatTensor([0.0]).to(rank) for i in range(len(M_LOOP))}

    for batch in tqdm(dataloader, disable = rank != 0):
        if train:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none = True)

        with autocast(device_type='cuda'):
            batch.x = batch.x.to(rank, non_blocking = True)
            batch.edge_index = batch.edge_index.to(rank, non_blocking = True)
            batch.y = batch.y.to(rank, non_blocking = True)
            
            preds = model(x = batch.x, adj_matrix = batch.edge_index)
            _, pred_labels = torchmax(preds, dim = 1)

            cel = loss_module(preds, batch.y)
            dl = Dice(preds, batch.y)
            loss = 0.5 * cel + 0.5 * dl
            m1 = metrics_1.forward(preds, batch.y)
            m2 = metrics_2.forward(pred_labels.unsqueeze(0), batch.y.unsqueeze(0))

            metrics['ce_loss'] += cel.item()
            metrics['dice_loss'] += dl.item()
            metrics['loss'] += loss.item()

            for i in range(len(m1)):
                metrics[M_LOOP[i+3]] = m1[M1_NAMES[i]]
            for i in range(len(m2)):
                metrics[M_LOOP[i+6]] = m2[M2_NAMES[i]]
            
        if train:
            assert scaler is not None
            scaler.scale(loss).backward()
            assert optimizer is not None
            scaler.step(optimizer)
            scaler.update()
        
    return metrics

def print_metrics(epoch: int, metrics: dict, writer: SummaryWriter, train: bool):
    print('----------TRAINING METRICS----------' if train else \
          '------------EVAL METRICS------------')

    PHASE = 'train' if train else 'eval'

    print('Loss: ', metrics[M_LOOP[2]].item())
    writer.add_scalar(PHASE + '_loss', metrics[M_LOOP[2]].item(), global_step = epoch + 1)

    print('CE Loss: ', metrics[M_LOOP[0]].item())
    writer.add_scalar(PHASE + '_ce_loss', metrics[M_LOOP[0]].item(), global_step = epoch + 1)

    print('Dice Loss: ', metrics[M_LOOP[1]].item())
    writer.add_scalar(PHASE + '_dice_loss', metrics[M_LOOP[1]].item(), global_step = epoch + 1)

    for m in range(len(metrics[3:])):
        vals = metrics[M_LOOP[m]].item()
        for i in range(len(vals)):
            print(M_NAMES[m], i, ': ', vals[i])
            writer.add_scalar(PHASE + M_LOGS[m] + str(i), vals[i],
                              global_step = epoch + 1)

def main():
    RANK = int(environ['LOCAL_RANK'])
    set_device(RANK)
    init_process_group('nccl')

    if RANK == 0:
        print('PyG version: ', pyg_version)
        print('Torch version: ', torch_version)
        print('GPU available: ', is_available())
        print('GPU count: ', device_count())
    
    adjacency = __Load_Adjacency__(DIRECTORY + 'ADJACENCY/')
    train_metadata = read_csv(filepath_or_buffer = DIRECTORY + 'train_dataset_info.csv')

    TRAIN_LEN = len(train_metadata) if PRODUCTION else \
        BATCH_SIZE*WORLD_SIZE*NUM_WORKERS*PREFETCH_FACTOR*2
    TRAIN_START = RANK * (TRAIN_LEN // WORLD_SIZE)
    TRAIN_END = (RANK + 1) * (TRAIN_LEN // WORLD_SIZE)
    train_dataset = CHD_Dataset(metadata = train_metadata[TRAIN_START:TRAIN_END],
                                adjacency = adjacency, root = DIRECTORY)
    
    eval_dataset = None
    if PRODUCTION:
        eval_metadata = read_csv(filepath_or_buffer = DIRECTORY + 'eval_dataset_info.csv')
        EVAL_START = RANK * (len(eval_metadata) // WORLD_SIZE)
        EVAL_END = (RANK + 1) * (len(eval_metadata) // WORLD_SIZE)
        eval_dataset = CHD_Dataset(metadata = eval_metadata[EVAL_START:EVAL_END],
                                   adjacency = adjacency, root = DIRECTORY)
    
    train_dataloader = DataLoader(dataset = train_dataset,
                                  batch_size = BATCH_SIZE,
                                  num_workers = NUM_WORKERS,
                                  persistent_workers = True,
                                  pin_memory = True, drop_last = True,
                                  prefetch_factor = PREFETCH_FACTOR,
                                  shuffle = True)

    if PRODUCTION:
        assert eval_dataset is not None
        eval_dataloader = DataLoader(dataset = eval_dataset,
                                     batch_size = BATCH_SIZE,
                                     num_workers = NUM_WORKERS,
                                     persistent_workers = True,
                                     shuffle = False, pin_memory = True,
                                     prefetch_factor = PREFETCH_FACTOR,
                                     drop_last = True)

    model = DistributedDataParallel(
        SyncBatchNorm.convert_sync_batchnorm(
            CHD_GNN().to(RANK)
        ),
        [RANK]
    )

    loss_module = CrossEntropyLoss(ignore_index = 0,
                                   weight = FloatTensor([11108352000./10420281390.,
                                                         11108352000./47325435.,
                                                         11108352000./46453197.,
                                                         11108352000./110663064.,
                                                         11108352000./143205882.,
                                                         11108352000./190230471.,
                                                         11108352000./82210947.,
                                                         11108352000./67981614.]).to(RANK))
    optimizer = ZeroRedundancyOptimizer(model.parameters(),
                                        optimizer_class = AdamW,
                                        lr = LR, weight_decay = 0) # No weight decay with PReLU
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0 = T0,
                                            T_mult = TMULT,
                                            eta_min = MIN_LR)
    scaler = GradScaler()

    if RANK == 0:
        writer = SummaryWriter('GNN_Experiment')
    
    metrics_1 = MetricCollection([
        Accuracy(task = 'multiclass', average = None, num_classes = 8),
        Precision(task = 'multiclass', average = None, num_classes = 8),
        Recall(task = 'multiclass', average = None, num_classes = 8)],
        ).to(RANK)
    metrics_2 = MetricCollection([
        DiceScore(num_classes = 8, average = None, input_format = 'index'),
        MeanIoU(num_classes = 8, per_class = True, input_format = 'index')]
        ).to(RANK)

    next_decay = T0
    decay_cycle_len = T0
    max_lr = LR

    if PRODUCTION and exists(CHECKPOINT):
        LOC = 'cuda:' + str(RANK)
        snapshot = load(CHECKPOINT, map_location = LOC)
        model.load_state_dict(snapshot['MODEL_STATE'])
        optimizer.load_state_dict(snapshot['OPTIM_STATE'])
        FIRST = snapshot['EPOCHS_RUN']

        if RANK == 0:
            print('Resuming training from epoch index=' + str(FIRST))

        if FIRST > next_decay:
            for e in range(FIRST):
                if e == next_decay:
                    max_lr *= DECAY
                    decay_cycle_len *= TMULT
                    next_decay += decay_cycle_len
    else:
        FIRST = 0
        if RANK == 0:
            print('Starting new training')
    
    print(f"[Rank {RANK}] Dataset len: {len(train_dataset)}")

    for epoch in range(FIRST, EPOCHS):
        scheduler.step(epoch)

        if epoch == next_decay:
            max_lr *= DECAY
            for pg in optimizer.param_groups:
                pg['lr'] = max_lr
            decay_cycle_len *= TMULT
            next_decay += decay_cycle_len
        
        will_validate = PRODUCTION and ((epoch < 15) or \
            ((epoch < 31) and ((epoch - 14) % 2 == 0)) or \
            ((epoch - 30) % 3 == 0))

        if RANK == 0:
            print('--------------------------------------------------')
            print('Epoch ', epoch + 1)

        model.train()
        metrics = loader_loop(RANK, True, train_dataloader,
                              model, scaler, loss_module,
                              optimizer, metrics_1, metrics_2)
        
        for i in range(len(metrics)):
            metrics[M_LOOP[i]] /= len(train_dataloader)

        barrier()
        handles = []
        for i in range(len(metrics)):
            handles.append(all_reduce(metrics[M_LOOP[i]].detach(),
                                      op = ReduceOp.AVG, async_op = True))
        
        for h in handles:
            assert h is not None
            h.wait()

        if RANK == 0:
            print_metrics(epoch, metrics, writer, True)

        if will_validate:
            model.eval()
            
            with no_grad():
                metrics = loader_loop(RANK, False, eval_dataloader,
                                      model, None, loss_module,
                                      None, metrics_1, metrics_2)
            
            for i in range(len(metrics)):
                metrics[M_LOOP[i]] /= len(train_dataloader)

            barrier()
            handles = []
            for i in range(len(metrics)):
                handles.append(all_reduce(metrics[M_LOOP[i]].detach(),
                                          op = ReduceOp.AVG, async_op = True))
            
            for h in handles:
                assert h is not None
                h.wait()

            if RANK == 0:
                print_metrics(epoch, metrics, writer, True)

        if PRODUCTION and RANK == 0:
            checkpoint_path = 'MODELS/gnn_' + str(epoch + 1) + '.checkpoint'
            snapshot = {
                'MODEL_STATE': model.module.state_dict(),
                'OPTIM_STATE': optimizer.state_dict(),
                'EPOCHS_RUN': epoch + 1
            }
            save(snapshot, checkpoint_path)
            copy('MODELS/gnn_' + str(epoch + 1) + '.checkpoint', CHECKPOINT)
        
        if RANK == 0:
            writer.flush()
    
    destroy_process_group()
        
if __name__ == '__main__':
    environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    main()