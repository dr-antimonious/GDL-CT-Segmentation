from torch import Tensor, FloatTensor, LongTensor
from torch_geometric.data import Data
from torch.utils.data import Dataset
from torch import is_tensor
from numpy import load, array, int64
from os import listdir
from Extracting_Planes import Extract_And_Convert

class PairData(Data):
  r"""
    PyTorch Geometric data class used for the ImageCHD dataset.
  """

  def __inc__(self, key, value, *args, **kwargs):
      if key == 'edge_index':
        return self.x.size(0)
        # return 0 # Works because of splitting the batch, otherwise use the thing above
      return super().__inc__(key, value, *args, **kwargs)
  
  def __init__(self,
               x: Tensor | None = None,
               edge_index: Tensor | None = None,
               edge_attr: Tensor | None = None,
               y: Tensor | int | float | None = None,
               pos: Tensor | None = None,
               time: Tensor | None = None,
               **kwargs):
     r"""
        Arguments for proper ImageCHD processing:
            x (Tensor): Source coronary-CT image as graph.
            y (Tensor): Ground truth segmentation as graph.
            edge_index (Tensor): Adjacency matrix.
            adj_count (int): Source image width.
     """
     super().__init__(x, edge_index, edge_attr, y, pos, time, **kwargs)

class CHD_Dataset(Dataset):
  r"""
    PyTorch dataset class used for the ImageCHD dataset.
  """

  def __init__(self, metadata, directory, adjacency):
    r"""
    Arguments:
        metadata (DataFrame): Pandas DataFrame containing dataset information.
        directory (string):   Path to the directory of the dataset.
        adjacency (Dictionary): Dictionary of adjacency matrices.
    """
    self.metadata = metadata
    self.directory = directory
    self.image_dir = directory + 'IMAGES/'
    self.label_dir = directory + 'LABELS/'
    self.adjacency = adjacency

  def __len__(self):
    return len(self.metadata)

  def __getitem__(self, idx):
    if is_tensor(idx):
      idx = idx.tolist()
    
    image, label = Extract_And_Convert(path_to_image = self.image_dir \
                                        + str(self.metadata['index'][idx]) + '.nii.gz',
                                       path_to_label = self.label_dir \
                                        + str(self.metadata['index'][idx]) + '.nii.gz',
                                       plane_type = self.metadata['Type'][idx],
                                       plane_index = self.metadata['Indice'][idx])
    adj_matrix = self.adjacency[str(self.metadata['Adjacency_count'][idx])]
    sample = PairData(x = FloatTensor(image),
                      edge_index = LongTensor(adj_matrix),
                      y = LongTensor(label),
                      adj_count = self.metadata['Adjacency_count'][idx])
    
    return sample

  def get(self, idx):
    return self.__getitem__(idx)
  
def __Load_Adjacency__(path):
  files = listdir(path)
  adjacency = {}
  for f in files:
    adjacency[f.split('_')[2].split('.')[0]] = array(load(path + f), dtype = int64)
  return adjacency