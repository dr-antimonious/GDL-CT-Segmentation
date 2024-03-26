from tqdm import tqdm
from pandas import DataFrame
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from sklearn.metrics import jaccard_score
from Network import CHD_GNN

def eval(model: CHD_GNN,
         loss_module: _Loss,
         eval_dataloader: DataLoader):
    r"""
        Arguments:
            model (Network.CHD_GNN): GNN network in training.
            loss_module (torch.nn.modules.loss._Loss): Loss module from PyTorch.
            eval_dataloader (torch.utils.data.DataLoader): Dataloader containing evaluation data.
    """

def train_with_eval(model: CHD_GNN,
                    loss_module: _Loss,
                    optimizer: Optimizer,
                    train_dataloader: DataLoader,
                    eval_dataloader: DataLoader,
                    scheduler: LRScheduler | None = None,
                    num_epochs: int | None = 200,
                    experiment_name: str | None = 'CHD_GNN_Experiment_1'):
    r"""
        Arguments:
            model (Network.CHD_GNN): Un-trained GNN network.
            loss_module (torch.nn.modules.loss._Loss): Loss module from PyTorch.
            optimizer (torch.optim.optimizer.Optimizer): Optimizer from PyTorch.
            train_dataloader (torch.utils.data.DataLoader): Dataloader containing training data.
            eval_dataloader (torch.utils.data.DataLoader): Dataloader containing evaluation data.
            scheduler (torch.optim.lr_scheduler.LRScheduler | None) = None: Scheduler module from PyTorch.
            num_epochs (int | None) = 200: Number of epochs to train the network for.
            experiment_name (str | None) = 'CHD_GNN_Experiment_1': Name of the training experiment.
    """