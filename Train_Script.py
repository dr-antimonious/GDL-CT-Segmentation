from torch import __version__ as torch_version
from torch import max as torchmax
from torch import Tensor, FloatTensor, LongTensor, save, no_grad, zeros, long, bincount, sum
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.cuda import is_available, empty_cache, device_count, stream, default_stream, Stream
from torch.nn import CrossEntropyLoss
from torch.nn.functional import softmax, one_hot
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard.writer import SummaryWriter

from torch_geometric import __version__ as pyg_version
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree

from numpy import isin, argwhere, double, uint16, ndarray
from numpy import zeros as npyzeros
from os import environ, getenv
from pandas import read_csv
from tqdm import tqdm

from Metrics import Accuracy_Util
from Network import CHD_GNN
from Utilities import CHD_Dataset, __Load_Adjacency__

LR              = 1e-3
WEIGHT_DECAY    = 1e-4
T0              = 10
TMULT           = 2
MIN_LR          = 1e-5
DECAY           = 0.9
SMOOTH          = 1e-6
BATCH_SIZE      = 32
NUM_WORKERS     = 8
PREFETCH_FACTOR = 4
EPOCHS          = 120

PRODUCTION_STR  = getenv('GDL_CT_SEG_PROD')
DIRECTORY       = getenv('GDL_CT_SEG_DIR')
assert PRODUCTION_STR is not None
assert DIRECTORY is not None
PRODUCTION      = PRODUCTION_STR.lower() == 'true'

def main():
    def Dice(preds: Tensor, y: Tensor) -> Tensor:
        probs = softmax(preds, dim = 1)[:, 1:, :, :]
        targs = one_hot(y, num_classes = 8).permute((0, 3, 1, 2)).float()[:, 1:, :, :]

        dims = (0, 2, 3)
        intersection = sum(probs * targs, dims)
        union = sum(probs + targs, dims)

        dice_coef = (2. * intersection + SMOOTH) / (union + SMOOTH)
        dice_loss = 1. - dice_coef
        return dice_loss.mean()
    
    def decode(stream1: Stream, model: CHD_GNN, enc_out_prev: list[Tensor],
               edge_index_prev: LongTensor, loss_module: CrossEntropyLoss,
               y_prev: LongTensor, width: int, last: bool)-> \
                tuple[Tensor, Tensor, Tensor, Tensor]:
        with stream(stream1):
            if last:
                stream1.synchronize()

            preds = model.decode(enc_out_prev, edge_index_prev)
            _, pred_labels = torchmax(preds, dim = 1)

            preds = preds.permute((1, 0)).reshape((1, 8, 512, width))
            y = y_prev.reshape((512, width)).unsqueeze(0)
            ce_loss = loss_module(preds, y)
            d_loss = Dice(preds, y)
            loss = 0.5 * ce_loss + 0.5 * d_loss
        return (pred_labels, ce_loss, d_loss, loss)
    
    def loss_and_metrics(scaler: GradScaler|None, y_prev: LongTensor,
                         pred_labels: Tensor, mmetrics: ndarray,
                         counts: ndarray, loss: Tensor):
        if scaler is not None:
            scaled_loss = loss / BATCH_SIZE
            scaler.scale(scaled_loss).backward()

        metrics = Accuracy_Util(y_prev, pred_labels, SMOOTH)
        mask = argwhere(isin(metrics, -1)^True)
        mmetrics[mask] += metrics[mask]
        counts[mask] += 1
    
    def loader_loop(train: bool, dataloader: DataLoader, stream0: Stream, stream1: Stream,
                    model: CHD_GNN, scaler: GradScaler|None, loss_module: CrossEntropyLoss,
                    optimizer: AdamW|None, metrics: ndarray, counts: ndarray) -> \
                        tuple[float, float, float]:
        if train:
            assert scaler is not None
            assert optimizer is not None
        
        ce_loss = 0.0
        d_loss = 0.0
        lloss = 0.0

        for i, batch in enumerate(tqdm(dataloader)):
            with autocast(device_type='cuda'):
                batch.x = batch.x.to('cuda:0', non_blocking = True)
                batch.edge_index = batch.edge_index.to('cuda:0', non_blocking = True)

                with stream(stream0):
                    enc_out = model.encode(batch.x, batch.edge_index)

                with stream(stream1):
                    enc_out_next = [e.to('cuda:1', non_blocking = True) for e in enc_out]
                    edge_index_next = batch.edge_index.to('cuda:1', non_blocking = True)
                    y_next = batch.y.to('cuda:1', non_blocking = True)

                if (i > 0) and ((i % BATCH_SIZE != 0) or (train is False)):
                    (pred_labels, cel, dl, loss) = decode(stream1, model, enc_out_prev,
                                                          edge_index_prev, loss_module,
                                                          y_prev, batch.adj_count[0], False)
                    ce_loss += cel.item()
                    d_loss += dl.item()
                    lloss += loss.item()
                
                enc_out_prev = enc_out_next
                edge_index_prev = edge_index_next

            if (i > 0) and ((i % BATCH_SIZE != 0) or (train is False)):
                loss_and_metrics(scaler, y_prev, pred_labels, metrics, counts, loss)
            
            y_prev = y_next
            
            if (((i + 1) % BATCH_SIZE == 0) and train) \
              or ((i + 1) == len(dataloader)):
                (pred_labels, cel, dl, loss) = decode(stream1, model, enc_out_prev,
                                                      edge_index_prev, loss_module,
                                                      y_prev, batch.adj_count[0], True)
                ce_loss += cel.item()
                d_loss += dl.item()
                lloss += loss.item()
                loss_and_metrics(scaler, y_prev, pred_labels, metrics, counts, loss)

                if train:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none = True)
            
            empty_cache()
        return (ce_loss, d_loss, lloss)

    environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    environ['NCCL_DEBUG'] = 'INFO'
    environ['NCCL_DEBUG_SUBSYS'] = 'ALL'

    print('PyG version: ', pyg_version)
    print('Torch version: ', torch_version)
    print('GPU available: ', is_available())
    print('GPU count: ', device_count())
    empty_cache()

    adjacency = __Load_Adjacency__(DIRECTORY + 'ADJACENCY/')

    max_degree = -1
    for key in adjacency:
        d = degree(index = LongTensor(adjacency[key][1]), num_nodes = 512 * int(key),
                   dtype = long)
        max_degree = max(max_degree, int(d.max()))
    
    deg = zeros(max_degree + 1, dtype = long)
    for key in adjacency:
        d = degree(index = LongTensor(adjacency[key][1]), num_nodes = 512 * int(key),
                   dtype = long)
        deg += bincount(d, minlength = deg.numel())

    train_metadata = read_csv(filepath_or_buffer = DIRECTORY + 'train_dataset_info.csv')
    train_dataset = CHD_Dataset(metadata = train_metadata if PRODUCTION \
                                           else train_metadata[:BATCH_SIZE],
                                directory = DIRECTORY, adjacency = adjacency)
    train_dataloader = DataLoader(dataset = train_dataset, batch_size = 1,
                                  num_workers = NUM_WORKERS, persistent_workers = True,
                                  shuffle = True, pin_memory = True,
                                  prefetch_factor = PREFETCH_FACTOR)

    if PRODUCTION:
        eval_metadata = read_csv(filepath_or_buffer = DIRECTORY + 'eval_dataset_info.csv')
        eval_dataset = CHD_Dataset(metadata = eval_metadata, directory = DIRECTORY,
                                   adjacency = adjacency)
        eval_dataloader = DataLoader(dataset = eval_dataset, batch_size = 1,
                                     num_workers = NUM_WORKERS, persistent_workers = True,
                                     shuffle = True, pin_memory = True,
                                     prefetch_factor = PREFETCH_FACTOR)

    model = CHD_GNN(deg = deg)

    loss_module = CrossEntropyLoss(ignore_index = 0,
                                   weight = FloatTensor([11108352000./10420281390.,
                                                         11108352000./47325435.,
                                                         11108352000./46453197.,
                                                         11108352000./110663064.,
                                                         11108352000./143205882.,
                                                         11108352000./190230471.,
                                                         11108352000./82210947.,
                                                         11108352000./67981614.]).to('cuda:1'))

    optimizer = AdamW(model.parameters(), lr = LR, weight_decay = WEIGHT_DECAY)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0 = T0, T_mult = TMULT,
                                            eta_min = MIN_LR)
    
    writer = SummaryWriter('GNN_Experiment')
    scaler = GradScaler()

    stream0 = default_stream('cuda:0')
    stream1 = default_stream('cuda:1')

    next_decay = T0
    decay_cycle_len = T0
    max_lr = LR

    for epoch in range(EPOCHS):
        scheduler.step(epoch)

        if epoch == next_decay:
            max_lr *= DECAY
            for pg in optimizer.param_groups:
                pg['lr'] = max_lr
            
            decay_cycle_len *= TMULT
            next_decay += decay_cycle_len

        train_metrics = npyzeros(36, dtype = double)
        train_counts = npyzeros(36, dtype = uint16)
        
        will_validate = PRODUCTION and ((epoch < 15) or \
            ((epoch < 31) and ((epoch - 14) % 2 == 0)) or \
            ((epoch - 30) % 3 == 0))
        
        if will_validate:
            eval_metrics = npyzeros(36, dtype = double)
            eval_counts = npyzeros(36, dtype = uint16)

        print('--------------------------------------------------')
        print('Epoch ', epoch + 1)

        model.train()
        (tce_l, td_l, train_loss) = loader_loop(True, train_dataloader, stream0,
                                                stream1, model, scaler, loss_module,
                                                optimizer, train_metrics, train_counts)
        tce_l /= len(train_dataloader)
        td_l /= len(train_dataloader)
        train_loss /= len(train_dataloader)

        if will_validate:
            model.eval()
            with no_grad(), autocast(device_type = 'cuda'):
                (ece_l, ed_l, eval_loss) = loader_loop(False, eval_dataloader, stream0,
                                                       stream1, model, None, loss_module,
                                                       None, eval_metrics, eval_counts)
            ece_l /= len(eval_dataloader)
            ed_l /= len(eval_dataloader)
            eval_loss /= len(eval_dataloader)

        print('-----TRAINING METRICS-----')

        print('Loss: ', train_loss)
        writer.add_scalar('train_loss', train_loss, global_step = epoch + 1)

        print('CE Loss: ', tce_l)
        writer.add_scalar('train_ce_loss', tce_l, global_step = epoch + 1)

        print('Dice Loss: ', td_l)
        writer.add_scalar('train_dice_loss', td_l, global_step = epoch + 1)

        for i in range(0, 9):
            print('Accuracy ', i, ': ', train_metrics[i] / train_counts[i])
            writer.add_scalar('train_accuracy_' + str(i),
                              train_metrics[i] / train_counts[i],
                              global_step = epoch + 1)
        
        for i in range(9, 18):
            print('Precision ', i, ': ', train_metrics[i] / train_counts[i])
            writer.add_scalar('train_precision_' + str(i),
                              train_metrics[i] / train_counts[i],
                              global_step = epoch + 1)
        
        for i in range(18, 27):
            print('Recall ', i, ': ', train_metrics[i] / train_counts[i])
            writer.add_scalar('train_recall_' + str(i),
                              train_metrics[i] / train_counts[i],
                              global_step = epoch + 1)
        
        for i in range(27, 36):
            print('F1-score ', i, ': ', train_metrics[i] / train_counts[i])
            writer.add_scalar('train_f1_score_' + str(i),
                              train_metrics[i] / train_counts[i],
                              global_step = epoch + 1)
        
        for i in range(36, 45):
            print('IoU-score ', i, ': ', train_metrics[i] / train_counts[i])
            writer.add_scalar('train_iou_score_' + str(i),
                              train_metrics[i] / train_counts[i],
                              global_step = epoch + 1)
        
        if will_validate:
            print('-----EVALUATION METRICS-----')

            print('Loss: ', eval_loss)
            writer.add_scalar('eval_loss', eval_loss, global_step = epoch + 1)

            print('CE Loss: ', ece_l)
            writer.add_scalar('eval_ce_loss', ece_l, global_step = epoch + 1)

            print('Dice Loss: ', ed_l)
            writer.add_scalar('eval_dice_loss', ed_l, global_step = epoch + 1)

            for i in range(0, 9):
                print('Accuracy ', i, ': ', eval_metrics[i] / eval_counts[i])
                writer.add_scalar('eval_accuracy_' + str(i),
                                eval_metrics[i] / eval_counts[i],
                                global_step = epoch + 1)
            
            for i in range(9, 18):
                print('Precision ', i, ': ', eval_metrics[i] / eval_counts[i])
                writer.add_scalar('eval_precision_' + str(i),
                                eval_metrics[i] / eval_counts[i],
                                global_step = epoch + 1)
            
            for i in range(18, 27):
                print('Recall ', i, ': ', eval_metrics[i] / eval_counts[i])
                writer.add_scalar('eval_recall_' + str(i),
                                eval_metrics[i] / eval_counts[i],
                                global_step = epoch + 1)
            
            for i in range(27, 36):
                print('F1-score ', i, ': ', eval_metrics[i] / eval_counts[i])
                writer.add_scalar('eval_f1_score_' + str(i),
                                eval_metrics[i] / eval_counts[i],
                                global_step = epoch + 1)
            
            for i in range(36, 45):
                print('IoU-score ', i, ': ', eval_metrics[i] / eval_counts[i])
                writer.add_scalar('eval_iou_score_' + str(i),
                                eval_metrics[i] / eval_counts[i],
                                global_step = epoch + 1)

        if PRODUCTION:
            checkpoint_path = 'MODELS/gnn_' + str(epoch + 1) + '.checkpoint'
            states = model.state_dict()
            save(states, checkpoint_path)
        
        writer.flush()
        empty_cache()
        
if __name__ == '__main__':
    main()