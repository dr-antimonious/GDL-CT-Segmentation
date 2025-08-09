from torch.cuda import LongTensor
from numpy import ndarray, array, int32, append, flip

def Convert_To_Image(tensor: LongTensor, adj_count: int) -> ndarray:
    r"""
        Arguments:
            tensor (torch.cuda.LongTensor): Graph as tensor.
            
        Returns:
            out (numpy.ndarray): Converted image.
    """
    tensor = tensor.cpu()
    out = array([tensor[0:adj_count]], dtype = int32)

    for i in range(1, 512):
        if i % 2 == 1:
            out = append(out, flip([tensor[i*adj_count:adj_count+i*adj_count]]), axis = 0)
        else:
            out = append(out, [tensor[i*adj_count:adj_count+i*adj_count]], axis = 0)
            
    return out