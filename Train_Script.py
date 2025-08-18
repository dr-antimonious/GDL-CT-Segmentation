from torch import __version__ as torch_version
from torch import max as torchmax
from torch import FloatTensor, Tensor, save, load, no_grad, zeros, cat
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.cuda import is_available, device_count, set_device, empty_cache
from torch.distributed import init_process_group, destroy_process_group, \
    all_reduce, ReduceOp, barrier
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, \
    LinearLR, SequentialLR
from torch.utils.tensorboard.writer import SummaryWriter

from torch_geometric import __version__ as pyg_version
from torch_geometric.loader import DataLoader

from torchmetrics.classification.accuracy import Accuracy, MulticlassAccuracy
from torchmetrics.classification.precision_recall import Recall, \
    Precision, MulticlassRecall, MulticlassPrecision
from torchmetrics.collections import MetricCollection
from torchmetrics.segmentation.dice import DiceScore
from torchmetrics.segmentation.mean_iou import MeanIoU

from monai.losses.dice import DiceCELoss

from gc import collect
from numpy import array, unique
from os import environ, getenv
from os.path import exists
from pandas import read_csv, DataFrame
from shutil import copy
from tqdm import tqdm

from Network import CHD_GNN
from Utilities import CHD_Dataset, __Load_Adjacency__, load_nifti

LR              = 1e-3
T0              = 15
TMULT           = 2
MIN_LR          = 1e-5
WARM_FACT       = 0.1
WARM_ITER       = 5
SMOOTH          = 1e-6
BATCH_SIZE      = 28
NUM_WORKERS     = 4
PREFETCH_FACTOR = 2
EPOCHS          = 250
WORLD_SIZE      = device_count()
M_NAMES         = ['Accuracy ', 'Recall ', 'Precision ',
                   'F1-score ', 'IoU-score ']
M_LOGS          = ['_accuracy_', '_recall_', '_precision_',
                   '_f1_score_', '_iou_score_']
M_LOOP          = ['loss', 'accuracy', 'recall',
                   'precision', 'dice', 'iou']
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

FREQS = FloatTensor([7610724117.0, 35496879.0,
                     37714647.0, 83133666.0,
                     105607395.0, 144954300.0,
                     57576912.0, 53353236.0])

def loader_loop(rank: int, train: bool, dataloader: DataLoader,
                model: DistributedDataParallel, scaler: GradScaler|None,
                loss_module: DiceCELoss, optimizer: ZeroRedundancyOptimizer|None,
                metrics_1: MetricCollection, metrics_2: MetricCollection) -> \
                    dict[str, Tensor]:
    metrics_1.reset()
    metrics_2.reset()
    
    metrics = {M_LOOP[0]: zeros(1, device = rank)}
    metrics.update([(M_LOOP[i], zeros(8, device = rank)) \
                    for i in range(1, len(M_LOOP))])

    for batch in tqdm(dataloader, disable = rank != 0):
        if train:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none = True)

        batch = batch.to(rank, non_blocking = True)
        with autocast(device_type='cuda'):
            preds = model(x = batch.x, edges = batch.edge_index,
                          batch = batch.batch, b_size = batch.num_graphs)
            _, pred_labels = torchmax(preds, dim = 1)
            
            m1 = metrics_1(preds, batch.y)
            m2 = metrics_2(pred_labels.unsqueeze(0), batch.y.unsqueeze(0))

            preds = preds.T.unsqueeze(0)
            batch.y = batch.y.view(1, 1, -1)
            loss = loss_module(preds, batch.y).float()
            metrics['loss'] += loss.detach().float().item()

            for i in range(len(m1)):
                metrics[M_LOOP[i+1]] += m1[M1_NAMES[i]].detach().float()
            for i in range(len(m2)):
                metrics[M_LOOP[i+4]] += m2[M2_NAMES[i]].detach().float()
            
        if train:
            assert scaler is not None
            scaler.scale(loss).backward()
            assert optimizer is not None
            scaler.step(optimizer)
            scaler.update()
        
    return metrics

def print_metrics(epoch: int, metrics: dict,
                  writer: SummaryWriter, train: bool):
    print('----------TRAINING METRICS----------' if train else \
          '------------EVAL METRICS------------')

    PHASE = 'train' if train else 'eval'

    print('Loss: ', metrics[M_LOOP[0]].item())
    writer.add_scalar(PHASE + '_loss', metrics[M_LOOP[0]].item(),
                      global_step = epoch + 1)

    for m in range(1, len(metrics)):
        vals = metrics[M_LOOP[m]]
        for i in range(len(vals)):
            print(M_NAMES[m-1], i, ': ', vals[i].item())
            writer.add_scalar(PHASE + M_LOGS[m-1] + str(i), vals[i].item(),
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
    train_metadata = read_csv(
        filepath_or_buffer = DIRECTORY + 'train_dataset_info.csv')

    TRAIN_LEN = len(train_metadata) if PRODUCTION else \
        BATCH_SIZE * WORLD_SIZE * NUM_WORKERS * PREFETCH_FACTOR * 2
    TRAIN_START = RANK * (TRAIN_LEN // WORLD_SIZE)
    TRAIN_END = (RANK + 1) * (TRAIN_LEN // WORLD_SIZE)
    train_indices: list[int] = train_metadata[TRAIN_START:TRAIN_END]['index'] \
        .unique().tolist()
    
    train_uniques = unique(array(train_indices))
    train_images = [load_nifti(DIRECTORY + 'IMAGES/', idx) for idx in train_uniques]
    train_labels = [load_nifti(DIRECTORY + 'LABELS/', idx) for idx in train_uniques]

    train_metadata = DataFrame(train_metadata[TRAIN_START:TRAIN_END])
    train_dataset = CHD_Dataset(metadata = train_metadata,
                                adjacency = adjacency, root = DIRECTORY,
                                images = train_images, labels = train_labels)
    train_dataloader = DataLoader(dataset = train_dataset,
                                  batch_size = BATCH_SIZE,
                                  num_workers = NUM_WORKERS,
                                  pin_memory = True, drop_last = True,
                                  prefetch_factor = PREFETCH_FACTOR,
                                  shuffle = True)

    eval_dataset = None
    eval_dataloader = None
    if PRODUCTION:
        eval_metadata = read_csv(
            filepath_or_buffer = DIRECTORY + 'eval_dataset_info.csv')
        EVAL_START = RANK * (len(eval_metadata) // WORLD_SIZE)
        EVAL_END = (RANK + 1) * (len(eval_metadata) // WORLD_SIZE)
        eval_indices = eval_metadata[EVAL_START:EVAL_END]['index'] \
            .unique().tolist()
        eval_uniques = unique(array(eval_indices))
        eval_images = [load_nifti(DIRECTORY + 'IMAGES/', idx) for idx in eval_uniques]
        eval_labels = [load_nifti(DIRECTORY + 'LABELS/', idx) for idx in eval_uniques]
        eval_metadata = DataFrame(eval_metadata[EVAL_START:EVAL_END])
        eval_dataset = CHD_Dataset(metadata = eval_metadata,
                                   adjacency = adjacency, root = DIRECTORY,
                                   images = eval_images, labels = eval_labels)
        eval_dataloader = DataLoader(dataset = eval_dataset,
                                     batch_size = BATCH_SIZE,
                                     num_workers = NUM_WORKERS,
                                     shuffle = False, pin_memory = True,
                                     prefetch_factor = PREFETCH_FACTOR,
                                     drop_last = True)

    model = DistributedDataParallel(CHD_GNN().to(RANK), [RANK])

    freqs = FREQS.to(RANK)
    weights = (freqs[1:].max() / freqs[1:]) * 5.25
    weights = cat([FloatTensor([1.0]).to(RANK), weights])

    if RANK == 0:
        print(weights)

    weights /= weights.sum()

    if RANK == 0:
        print(weights)

    loss_module = DiceCELoss(include_background = False,
                             label_smoothing = 0.0,
                             to_onehot_y = True,
                             weight = weights,
                             softmax = True)
    
    optimizer = ZeroRedundancyOptimizer(model.parameters(),
                                        optimizer_class = AdamW,
                                        lr = LR, weight_decay = 0)
    scheduler1 = CosineAnnealingWarmRestarts(optimizer, T_0 = T0,
                                             T_mult = TMULT,
                                             eta_min = MIN_LR)
    scheduler2 = LinearLR(optimizer, start_factor = WARM_FACT,
                          total_iters = 5)
    scheduler = SequentialLR(optimizer, milestones = [5],
                             schedulers = [scheduler2, scheduler1])
    scaler = GradScaler()

    writer = None
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

    if exists(CHECKPOINT):
        LOC = 'cuda:' + str(RANK)
        snapshot = load(CHECKPOINT, map_location = LOC)
        model.load_state_dict(snapshot['MODEL_STATE'])
        FIRST = snapshot['EPOCHS_RUN']

        if RANK == 0:
            print('Resuming training from epoch index=' + str(FIRST))
    else:
        FIRST = 0
        if RANK == 0:
            print('Starting new training')
    
    print(f"[Rank {RANK}] Dataset len: {len(train_dataset)}")

    for epoch in range(FIRST, EPOCHS):
        will_validate = PRODUCTION and ((epoch < 20) or \
            ((epoch < 36) and ((epoch - 19) % 2 == 0)) or \
            ((epoch - 35) % 3 == 0))

        if RANK == 0:
            print('--------------------------------------------------')
            print('Epoch: ', epoch + 1)
            print('Learning rate: ', scheduler.get_last_lr())

        model.train()
        metrics = loader_loop(RANK, True, train_dataloader,
                              model, scaler, loss_module,
                              optimizer, metrics_1, metrics_2)
        
        for i in range(len(metrics)):
            metrics[M_LOOP[i]] /= len(train_dataloader)

        barrier()
        handles = []
        for i in range(len(metrics)):
            metrics[M_LOOP[i]] = metrics[M_LOOP[i]].detach()
            handles.append(all_reduce(metrics[M_LOOP[i]],
                                      op = ReduceOp.AVG, async_op = True))
        
        for h in handles:
            assert h is not None
            h.wait()

        if RANK == 0:
            assert writer is not None
            print_metrics(epoch, metrics, writer, True)

        if will_validate:
            model.eval()
            
            with no_grad():
                assert eval_dataloader is not None
                metrics = loader_loop(RANK, False, eval_dataloader,
                                      model, None, loss_module,
                                      None, metrics_1, metrics_2)
            
            for i in range(len(metrics)):
                metrics[M_LOOP[i]] /= len(eval_dataloader)

            barrier()
            handles = []
            for i in range(len(metrics)):
                metrics[M_LOOP[i]] = metrics[M_LOOP[i]].detach()
                handles.append(all_reduce(metrics[M_LOOP[i]],
                                          op = ReduceOp.AVG, async_op = True))
            
            for h in handles:
                assert h is not None
                h.wait()

            if RANK == 0:
                assert writer is not None
                print_metrics(epoch, metrics, writer, False)

        if RANK == 0:
            assert writer is not None
            checkpoint_path = 'MODELS/gnn_' + str(epoch + 1) + '.checkpoint'
            snapshot = {
                'MODEL_STATE': model.module.state_dict(),
                'EPOCHS_RUN': epoch + 1
            }
            save(snapshot, checkpoint_path)
            copy('MODELS/gnn_' + str(epoch + 1) + '.checkpoint', CHECKPOINT)
            writer.flush()
        
        collect()
        empty_cache()
        scheduler.step()
    
    destroy_process_group()
        
if __name__ == '__main__':
    environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    main()