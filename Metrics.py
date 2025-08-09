from collections.abc import Callable
from torch import Tensor, LongTensor, isin, argwhere
from numpy import ndarray, array, append

def __Accuracy__(truth: LongTensor, test: Tensor, value: int) -> float:
    r"""
        Arguments:
            truth (torch.LongTensor): Ground truth segmentation.
            test (torch.Tensor): GNN segmentation result.
            value (int): Value for which the accuracy will be returned.
        
        Returns:
            out (float): Segmentation accuracy for given value.
    """
    mask = argwhere(isin(truth, value))
    count = (test[mask] == value).sum().float().item()
    return count / mask.shape[0] if mask.shape[0] != 0 else -1

def __Precision__(truth: LongTensor, test: Tensor, value: int) -> float:
    r"""
        Arguments:
            truth (torch.LongTensor): Ground truth segmentation.
            test (torch.Tensor): GNN segmentation result.
            value (int): Value for which the precision will be returned.
        
        Returns:
            out (float): Segmentation precision for given value.
    """
    mask = argwhere(isin(truth, value))
    TP = (test[mask] == value).sum().float().item()
    FP = (test[~mask] == value).sum().float().item()
    return TP / (TP + FP) if (TP + FP) != 0 else -1

def __Recall__(truth: LongTensor, test: Tensor, value: int) -> float:
    r"""
        Arguments:
            truth (torch.LongTensor): Ground truth segmentation.
            test (torch.Tensor): GNN segmentation result.
            value (int): Value for which the recall will be returned.
        
        Returns:
            out (float): Segmentation recall for given value.
    """
    mask = argwhere(isin(truth, value))
    TP = (test[mask] == value).sum().float().item()
    FN = (test[mask] != value).sum().float().item()
    return TP / (TP + FN) if (TP + FN) != 0 else -1

def __F1_Score__(precision: float, recall: float) -> float:
    return 2 * (precision * recall) / (precision + recall) \
        if precision + recall != 0 else -1

def __IoU__(truth: LongTensor, test: Tensor,
            smooth: float, value: int) -> float:
    true_inds = (truth == value)
    test_inds = (test == value)

    intersection = (true_inds & test_inds).sum().float()
    union = (true_inds | test_inds).sum().float()

    return ((intersection + smooth) / (union + smooth)).item()

def __Calculate__(truth: LongTensor, test: Tensor,
                  metric: Callable[[LongTensor, Tensor, int], float]) \
                    -> ndarray:
    r"""
        Arguments:
            truth (torch.LongTensor): Ground truth segmentation.
            test (torch.Tensor): GNN segmentation result.
        
        Returns:
            out (numpy.ndarray): Segmentation accuracies for all values.
    """
    out = array([])
    for i in range(0, 8):
        out = append(out, metric(truth, test, i))
    return out

def __Calculate_IoU__(truth: LongTensor, test: Tensor, smooth: float) -> ndarray:
    r"""
        Arguments:
            truth (torch.LongTensor): Ground truth segmentation.
            test (torch.Tensor): GNN segmentation result.
        
        Returns:
            out (numpy.ndarray): Segmentation accuracies for all values.
    """
    out = array([])
    for i in range(0, 8):
        out = append(out, __IoU__(truth, test, smooth, i))
    return out

def __Calculate_F1_Score__(results: ndarray) -> ndarray:
    return array([__F1_Score__(p, r) for p, r in zip(results[9:17], results[18:26])])

def Accuracy_Util(truth: LongTensor, test: Tensor, smooth: float) -> ndarray:
    r"""
        Arguments:
            truth (torch.LongTensor): Ground truth segmentation.
            test (torch.Tensor): GNN segmentation result.
        
        Returns:
            out (numpy.ndarray): Accuracy info.
    """
    out = __Calculate__(truth, test, __Accuracy__)
    out = append(out, out.mean())

    out = append(out, __Calculate__(truth, test, __Precision__))
    out = append(out, (out[9:]).mean())

    out = append(out, __Calculate__(truth, test, __Recall__))
    out = append(out, (out[18:]).mean())

    out = append(out, __Calculate_F1_Score__(out))
    out = append(out, (out[27:]).mean())

    out = append(out, __Calculate_IoU__(truth, test, smooth))
    out = append(out, (out[36:]).mean())

    return out