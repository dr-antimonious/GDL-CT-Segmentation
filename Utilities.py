from torch import FloatTensor, LongTensor
from torch_geometric.data import Data, Dataset

from numpy import load, array, int64
from nibabel.loadsave import load as bload
from os import listdir
from pandas import DataFrame

from Extracting_Planes import Extract_And_Convert

def load_nifti(dir: str, idx: int):
  return bload(dir + str(idx) + '.nii.gz').get_fdata()

class CHD_Dataset(Dataset):
  r"""
    PyTorch dataset class used for the ImageCHD dataset.
  """

  def __init__(self, metadata, adjacency, root,
               pre_transform = None, pre_filter = None,
               transform = None):
    r"""
    Arguments:
        metadata (DataFrame): Pandas DataFrame containing dataset information.
        directory (string):   Path to the directory of the dataset.
        adjacency (Dictionary): Dictionary of adjacency matrices.
    """
    self.metadata: DataFrame = metadata
    self.image_dir = root + 'IMAGES/'
    self.label_dir = root + 'LABELS/'
    self.adjacency = adjacency

    super().__init__(root, transform, pre_transform, pre_filter)

    self.images: list = [load_nifti(self.image_dir, idx) for idx in self.sample_indices]
    self.image_idxs: list = [str(idx) for idx in self.sample_indices]
    self.labels: list = [load_nifti(self.label_dir, idx) for idx in self.sample_indices]
    self.label_idxs: list = [str(idx) for idx in self.sample_indices]
  
  @property
  def sample_indices(self):
    return self.metadata['index'].unique()

  @property
  def raw_file_names(self):
    return array([[
      ('IMAGES/' + str(x) + '.nii.gz'),
      ('LABELS/' + str(x) + '.nii.gz')]
      for x in self.sample_indices]) \
        .flatten().tolist()
  
  @property
  def processed_file_names(self):
    return self.raw_file_names
  
  @property
  def raw_dir(self) -> str:
    return self.root
  
  @property
  def processed_dir(self) -> str:
    return self.root

  def len(self):
    return len(self.metadata)
  
  def get(self, idx):
    image, label = Extract_And_Convert(im = self.images[self.image_dir.index(str(self.metadata['index'][idx]))],
                                       la = self.labels[self.label_dir.index(str(self.metadata['index'][idx]))],
                                       plane_type = self.metadata['Type'][idx],
                                       plane_index = self.metadata['Indice'][idx])
    adj_matrix = self.adjacency[str(self.metadata['Adjacency_count'][idx])]
    return Data(x = FloatTensor(image),
                edge_index = adj_matrix,
                y = LongTensor(label),
                adj_count = self.metadata['Adjacency_count'][idx])

def __Load_Adjacency__(path):
  files = listdir(path)
  adjacency = {}

  for f in files:
    adjacency[f.split('_')[2].split('.')[0]] = LongTensor(
      array(load(path + f), dtype = int64)
    )
  
  return adjacency