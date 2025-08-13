from joblib import Parallel, delayed
from torch import FloatTensor, LongTensor
from torch_geometric.data import Data, InMemoryDataset
from numpy import load, array, int64
from os import listdir
from pandas import DataFrame
from Extracting_Planes import Extract_And_Convert

def process_single(idx, metadata, adjacency, image_dir, label_dir):
  image, label = Extract_And_Convert(path_to_image = image_dir \
                                      + str(metadata['index'][idx]) + '.nii.gz',
                                    path_to_label = label_dir \
                                      + str(metadata['index'][idx]) + '.nii.gz',
                                    plane_type = metadata['Type'][idx],
                                    plane_index = metadata['Indice'][idx])
  adj_matrix = adjacency[str(metadata['Adjacency_count'][idx])]
  return Data(x = FloatTensor(image),
              edge_index = adj_matrix,
              y = LongTensor(label),
              adj_count = metadata['Adjacency_count'][idx])

class CHD_Dataset(InMemoryDataset):
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
    self.load(self.processed_paths[0])
  
  @property
  def raw_file_names(self):
    return array([[
      ('IMAGES/' + str(x) + '.nii.gz'),
      ('LABELS/' + str(x) + '.nii.gz')]
      for x in self.metadata['index'].unique()]) \
        .flatten().tolist()
  
  @property
  def processed_file_names(self):
    return ['data.pt']
  
  @property
  def raw_dir(self) -> str:
    return self.root
  
  @property
  def processed_dir(self) -> str:
    return '/home/ubuntu/proj/'

  def __len__(self):
    return len(self.metadata)

  def process(self):
    length = len(self.metadata['index'])
    
    data_list = Parallel(n_jobs = 50, backend = 'multiprocessing')(
      delayed(process_single)(idx, self.metadata, self.adjacency, self.image_dir, self.label_dir) for idx in range(length)
    )
    self.save(data_list, self.processed_paths[0])
  
def __Load_Adjacency__(path):
  files = listdir(path)
  adjacency = {}

  for f in files:
    adjacency[f.split('_')[2].split('.')[0]] = LongTensor(
      array(load(path + f), dtype = int64)
    )
  
  return adjacency