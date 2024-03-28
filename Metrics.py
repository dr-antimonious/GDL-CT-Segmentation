from torch import LongTensor, isin, argwhere
from numpy import ndarray, array, append

def __Accuracy__(truth: LongTensor, test: LongTensor, value: int) -> float:
    r"""
        Arguments:
            truth (torch.LongTensor): Ground truth segmentation.
            test (torch.LongTensor): GNN segmentation result.
            value (int): Value for which the accuracy will be returned.
        
        Returns:
            out (float): Segmentation accuracy for given value.
    """
    mask = argwhere(isin(truth, value))
    count = (test[mask] == value).sum().item()
    return count / mask.shape[0] if mask.shape[0] != 0 else -1

def __Calculate_Accuracy__(truth: LongTensor, test: LongTensor) -> ndarray:
    r"""
        Arguments:
            truth (torch.LongTensor): Ground truth segmentation.
            test (torch.LongTensor): GNN segmentation result.
        
        Returns:
            out (numpy.ndarray): Segmentation accuracies for all values.
    """
    out = array([])
    for i in range(0, 8):
        out = append(out, __Accuracy__(truth, test, i))
    return out

def __Average_Accuracy__(acc_arr: ndarray) -> float:
    return acc_arr[acc_arr > -1].sum() / acc_arr[acc_arr > -1].shape[0] \
        if acc_arr[acc_arr > -1].shape[0] != 0 else -1

def Accuracy_Util(truth: LongTensor, test: LongTensor) -> ndarray:
    r"""
        Arguments:
            truth (torch.LongTensor): Ground truth segmentation.
            test (torch.LongTensor): GNN segmentation result.
        
        Returns:
            out (numpy.ndarray): Accuracy info.
    """
    out = __Calculate_Accuracy__(truth, test)
    out = append(out, __Average_Accuracy__(out[1:]))
    out = append(out, __Average_Accuracy__(out))
    return out