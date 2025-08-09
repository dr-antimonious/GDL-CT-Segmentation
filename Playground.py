from numpy import array, uint8
from torch import LongTensor
from torch.nn.functional import one_hot

label = array([[1, 2], [3, 4]], dtype = uint8).flatten()
label = one_hot(LongTensor(label).reshape((2, 2)).unsqueeze(0),
                num_classes=5).permute((0, 3, 1, 2)).float()[:, 1:, :, :]
print(label)
print(label.shape)