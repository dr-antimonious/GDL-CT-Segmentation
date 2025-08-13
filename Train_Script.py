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
from torch.utils.data.distributed import DistributedSampler
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
BATCH_SIZE      = 4
ITERS_TO_ACCUM  = 8
NUM_WORKERS     = 8
PREFETCH_FACTOR = 4
EPOCHS          = 150
WORLD_SIZE      = device_count()
M_NAMES         = ['Accuracy ', 'Recall ', 'Precision ',
                   'F1-score ', 'IoU-score ']
M_LOGS          = ['_accuracy_', '_recall_', '_precision_',
                   '_f1_score_', '_iou_score_']
CHECKPOINT      = 'MODELS/gnn.checkpoint'

PRODUCTION_STR  = getenv('GDL_CT_SEG_PROD')
DIRECTORY       = getenv('GDL_CT_SEG_DIR')
assert PRODUCTION_STR is not None
assert DIRECTORY is not None
PRODUCTION      = PRODUCTION_STR.lower() == 'true'

def Dice(preds: Tensor, y: Tensor) -> Tensor:
    probs = softmax(preds, dim = 1)[:, 1:, :]
    targs = one_hot(y, num_classes = 8).permute((0, 2, 1)).float()[:, 1:, :]

    dims = (0, 2)
    intersection = sum(probs * targs, dims)
    union = sum(probs + targs, dims)

    dice_coef = (2. * intersection + SMOOTH) / (union + SMOOTH)
    dice_loss = 1. - dice_coef
    return dice_loss.mean()

def loader_loop(rank: int, train: bool, dataloader: DataLoader,
                model: DistributedDataParallel, scaler: GradScaler|None,
                loss_module: CrossEntropyLoss, optimizer: ZeroRedundancyOptimizer|None,
                metrics_1: MetricCollection, metrics_2: MetricCollection) -> \
                    tuple[Tensor, Tensor, Tensor]:
    ce_loss = FloatTensor([0.0]).to(rank)
    d_loss = FloatTensor([0.0]).to(rank)
    lloss = FloatTensor([0.0]).to(rank)

    for i, batch in enumerate(tqdm(dataloader, disable = rank != 0)):
        if train and (i % ITERS_TO_ACCUM == 0):
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

            ce_loss += cel.item()
            d_loss += dl.item()
            lloss += loss.item()

            metrics_1.update(preds, batch.y)
            metrics_2.update(pred_labels, batch.y)

            if train:
                scaled_loss = loss / BATCH_SIZE
            
        if train:
            assert scaler is not None
            scaler.scale(scaled_loss).backward()
        
            if (((i + 1) % ITERS_TO_ACCUM == 0) or ((i + 1) == len(dataloader))):
                assert optimizer is not None
                scaler.step(optimizer)
                scaler.update()
        
    return (ce_loss, d_loss, lloss)

def print_metrics(metrics_1: MetricCollection, metrics_2: MetricCollection,
                  epoch: int, loss: float, ce_loss: float, dice_loss: float,
                  writer: SummaryWriter, train: bool):
    print('----------TRAINING METRICS----------' if train else \
          '------------EVAL METRICS------------')

    m1 = metrics_1.compute()
    m2 = metrics_2.compute()

    accuracy = m1[MulticlassAccuracy.__name__]
    recall = m1[MulticlassRecall.__name__]
    precision = m1[MulticlassPrecision.__name__]
    dice_coef = m2[DiceScore.__name__]
    iou = m2[MeanIoU.__name__]

    metrics = [accuracy, recall, precision, dice_coef, iou]
    PHASE = 'train' if train else 'eval'

    print('Loss: ', loss)
    writer.add_scalar(PHASE + '_loss', loss, global_step = epoch + 1)

    print('CE Loss: ', ce_loss)
    writer.add_scalar(PHASE + '_ce_loss', ce_loss, global_step = epoch + 1)

    print('Dice Loss: ', dice_loss)
    writer.add_scalar(PHASE + '_dice_loss', dice_loss, global_step = epoch + 1)

    for m in metrics:
        for i in range(len(m)):
            print(M_NAMES[i], i, ': ', m[i])
            writer.add_scalar(PHASE + M_LOGS[i] + str(i), m[i],
                              global_step = epoch + 1)

def main(train_dataset: CHD_Dataset, eval_dataset: CHD_Dataset|None):
    RANK = int(environ['LOCAL_RANK'])
    set_device(RANK)
    init_process_group('nccl')

    if RANK == 0:
        print('PyG version: ', pyg_version)
        print('Torch version: ', torch_version)
        print('GPU available: ', is_available())
        print('GPU count: ', device_count())

    train_sampler = DistributedSampler(dataset = train_dataset,
                                       num_replicas = WORLD_SIZE,
                                       rank = RANK, shuffle = True)
    train_dataloader = DataLoader(dataset = train_dataset,
                                  sampler = train_sampler,
                                  batch_size = BATCH_SIZE,
                                  num_workers = NUM_WORKERS,
                                  persistent_workers = True,
                                  pin_memory = True, drop_last = True,
                                  prefetch_factor = PREFETCH_FACTOR)

    if PRODUCTION:
        assert eval_dataset is not None
        eval_sampler = DistributedSampler(dataset = eval_dataset,
                                          num_replicas = WORLD_SIZE,
                                          rank = RANK, shuffle = True)
        eval_dataloader = DataLoader(dataset = eval_dataset,
                                     sampler = eval_sampler,
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
    
    metrics_1 = MetricCollection(
        [Accuracy(task = 'multiclass', average = None, num_classes = 8),
         Precision(task = 'multiclass', average = None, num_classes = 8),
         Recall(task = 'multiclass', average = None, num_classes = 8)
         ]).to(RANK)
    metrics_2 = MetricCollection(
        [DiceScore(num_classes = 8, average = None, input_format = 'index'),
         MeanIoU(num_classes = 8, per_class = True, input_format = 'index')
         ]).to(RANK)

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
    
    print(f"[Rank {RANK}] Dataset len: {len(train_dataset)}, Sampler len: {len(train_sampler)}")

    for epoch in range(FIRST, EPOCHS):
        scheduler.step(epoch)
        train_sampler.set_epoch(epoch)
        metrics_1.reset()
        metrics_2.reset()
        
        if PRODUCTION:
            eval_sampler.set_epoch(epoch)

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
        (tce_l, td_l, train_loss) = loader_loop(RANK, True, train_dataloader,
                                                model, scaler, loss_module,
                                                optimizer, metrics_1, metrics_2)
        
        tce_l /= len(train_dataloader)
        td_l /= len(train_dataloader)
        train_loss /= len(train_dataloader)

        barrier()
        tcel_h = all_reduce(tce_l.detach(), op = ReduceOp.AVG, async_op = True)
        tdl_h = all_reduce(td_l.detach(), op = ReduceOp.AVG, async_op = True)
        tl_h = all_reduce(train_loss.detach(), op = ReduceOp.AVG, async_op = True)
        
        for h in [tcel_h, tdl_h, tl_h]:
            assert h is not None
            h.wait()

        if RANK == 0:
            print_metrics(metrics_1, metrics_2, epoch, train_loss.item(),
                          tce_l.item(), td_l.item(), writer, True)

        if will_validate:
            metrics_1.reset()
            metrics_2.reset()
            model.eval()
            
            with no_grad():
                (ece_l, ed_l, eval_loss) = loader_loop(RANK, False, eval_dataloader,
                                                       model, None, loss_module,
                                                       None, metrics_1, metrics_2)
            
            ece_l /= len(eval_dataloader)
            ed_l /= len(eval_dataloader)
            eval_loss /= len(eval_dataloader)

            barrier()
            ecel_h = all_reduce(ece_l.detach(), op = ReduceOp.AVG, async_op = True)
            edl_h = all_reduce(ed_l.detach(), op = ReduceOp.AVG, async_op = True)
            el_h = all_reduce(eval_loss.detach(), op = ReduceOp.AVG, async_op = True)

            for h in [ecel_h, edl_h, el_h]:
                assert h is not None
                h.wait()
        
            if RANK == 0:
                print_metrics(metrics_1, metrics_2, epoch, eval_loss.item(),
                              ece_l.item(), ed_l.item(), writer, False)

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
    environ['NCCL_DEBUG'] = 'INFO'
    environ['NCCL_DEBUG_SUBSYS'] = 'ALL'

    adjacency = __Load_Adjacency__(DIRECTORY + 'ADJACENCY/')
    train_metadata = read_csv(filepath_or_buffer = DIRECTORY + 'train_dataset_info.csv')
    train_dataset = CHD_Dataset(metadata = train_metadata, adjacency = adjacency,
                                root = DIRECTORY)
    
    eval_dataset = None
    if PRODUCTION:
        eval_metadata = read_csv(filepath_or_buffer = DIRECTORY + 'eval_dataset_info.csv')
        eval_dataset = CHD_Dataset(metadata = eval_metadata, adjacency = adjacency,
                                   root = DIRECTORY)
    
    main(train_dataset, eval_dataset)