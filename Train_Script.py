from torch_geometric import __version__ as pyg_version
from torch_geometric.loader import DataLoader
from torch import __version__ as torch_version
from torch import FloatTensor, max, save, load
from torch.cuda import is_available, set_device, empty_cache, device_count
from torch.distributed import init_process_group, barrier
from torch.distributed.optim.zero_redundancy_optimizer import ZeroRedundancyOptimizer
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.elastic.multiprocessing.errors import record
from torch.utils.tensorboard.writer import SummaryWriter
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from pandas import read_csv
from os import environ
from tqdm import tqdm
from numpy import array, isin, argwhere

from Network import CHD_GNN
from Utilities import CHD_Dataset, __Load_Adjacency__
from Metrics import Accuracy_Util

@record
def main():
    DIRECTORY = '/home/sojo/Documents/ImageCHD/ImageCHD_dataset/'
    EPOCHS = 200

    init_process_group('nccl')
    environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    local_rank = int(environ['LOCAL_RANK'])
    global_rank = int(environ['RANK'])
    batch_size = int(environ['WORLD_SIZE'])

    if global_rank == 0:
        print('PyG version: ', pyg_version)
        print('Torch version: ', torch_version)
        print('GPU available: ', is_available())
        print('GPU count: ', device_count())

    set_device(local_rank)
    empty_cache()

    model = CHD_GNN().to('cuda:' + str(local_rank))
    model = DDP(model, device_ids = [local_rank])
    adjacency = __Load_Adjacency__(DIRECTORY + 'ADJACENCY/')

    train_metadata = read_csv(filepath_or_buffer = DIRECTORY + 'train_dataset_info.csv')
    train_dataset = CHD_Dataset(metadata = train_metadata, directory = DIRECTORY, adjacency = adjacency)
    train_sampler = DistributedSampler(train_dataset, shuffle = True)
    train_dataloader = DataLoader(dataset = train_dataset, batch_size = 4,
                                num_workers = 8, persistent_workers = True,
                                sampler = train_sampler, pin_memory = True,
                                prefetch_factor = 4)

    eval_metadata = read_csv(filepath_or_buffer = DIRECTORY + 'eval_dataset_info.csv')
    eval_dataset = CHD_Dataset(metadata = eval_metadata, directory = DIRECTORY, adjacency = adjacency)
    eval_sampler = DistributedSampler(eval_dataset, shuffle = True)
    eval_dataloader = DataLoader(dataset = eval_dataset, batch_size = 4,
                                num_workers = 8, persistent_workers = True,
                                sampler = eval_sampler, pin_memory = True,
                                prefetch_factor = 4)

    loss_module = CrossEntropyLoss(weight = FloatTensor([48538./38830., 48538./1387.,
                                                        48538./1387., 48538./1387.,
                                                        48538./1387., 48538./1387.,
                                                        48538./1387., 48538./1386.]).to(local_rank))
    optimizer = ZeroRedundancyOptimizer(model.parameters(), Adam, amsgrad = True)

    writer = SummaryWriter('GNN_Experiment')

    for epoch in range(EPOCHS):
        train_metrics = array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        eval_metrics = array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        train_loss = 0.0
        eval_loss = 0.0
        
        if global_rank == 0:
            print('--------------------------------------------------')
            print('Epoch ', epoch + 1)

        model.train()
        train_sampler.set_epoch(epoch = epoch + 1)
        eval_sampler.set_epoch(epoch = epoch + 1)
        barrier()

        for batch in tqdm(train_dataloader, total = len(train_dataloader)):
            optimizer.zero_grad()
            batch = batch.to(local_rank)
            preds = model(batch.x, batch.edge_index)
            _, pred_labels = max(preds, dim = 1)

            loss = loss_module(preds, batch.y)
            loss.backward()
            optimizer.step()

            metrics = Accuracy_Util(batch.y, pred_labels)
            mask = argwhere(isin(metrics, -1)^True)
            train_metrics[mask] += metrics[mask]
            mask += 10
            train_metrics[mask] += 1
        
        model.eval()

        for batch in tqdm(eval_dataloader, total = len(eval_dataloader)):
            batch = batch.to(local_rank)
            preds = model(batch.x, batch.edge_index)
            _, pred_labels = max(preds, dim = 1)

            loss = loss_module(preds, batch.y)
            metrics = Accuracy_Util(batch.y, pred_labels)
            mask = argwhere(isin(metrics, -1)^True)
            eval_metrics[mask] += metrics[mask]
            mask += 10
            eval_metrics[mask] += 1
        
        if global_rank == 0:
            print('-----TRAINING METRICS-----')
            print('Loss: ', train_loss / float(len(train_dataloader)))
            writer.add_scalar('train_loss', train_loss / float(len(train_dataloader)),
                              global_step = epoch + 1)
            for i in range(0, 10):
                print('Accuracy ', i, ': ', train_metrics[i] / float(train_metrics[i+10]))
                writer.add_scalar('train_accuracy_' + str(i), train_metrics[i] / float(train_metrics[i+10]),
                                  global_step = epoch + 1)
            print('-----EVALUATION METRICS-----')
            print('Loss: ', eval_loss / float(len(eval_dataloader)))
            writer.add_scalar('eval_loss', eval_loss / float(len(eval_dataloader)),
                              global_step = epoch + 1)
            for i in range(0, 10):
                print('Accuracy ', i, ': ', eval_metrics[i] / float(eval_metrics[i+10]))
                writer.add_scalar('eval_accuracy_' + str(i), eval_metrics[i] / float(eval_metrics[i+10]),
                                  global_step = epoch + 1)
            
            checkpoint_path = 'MODELS/gnn_' + str(epoch + 1) + '.checkpoint'
            save(model.module.state_dict(), checkpoint_path)

        barrier()
        map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
        model.load_state_dict(load(checkpoint_path, map_location = map_location))

if __name__ == '__main__':
    main()