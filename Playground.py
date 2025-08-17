from matplotlib.pyplot import imshow, show
from nibabel.loadsave import load
from numpy import uint64, array, count_nonzero
import pandas as pd
from tqdm import tqdm

DIRECTORY   = "D:\\ImageCHD_dataset\\"
DEST_DIR    = "C:\\Users\\leotu\\OneDrive\\Documents\\ImageCHD_dataset\\"
PLANES      = ['A', 'C', 'S']

dataset = pd.read_csv(DEST_DIR + "new_train_dataset_info.csv")
label_counts = array([uint64(0) for _ in range(8)], dtype = uint64)

for idx in dataset['index'].unique():
    image = load(DIRECTORY + "ct_" + str(idx) + "_label.nii.gz")
    axial = image.header['dim'][3]
    data = image.get_fdata()

    for plane in PLANES:
        for ix in range(512 if plane != 'A' else axial):
            match plane:
                case 'A': # Axial plane
                    label = data[:, :, ix]
                case 'C': # Coronal plane
                    label = data[:, ix, :]
                case 'S': # Sagittal plane
                    label = data[ix, :, :]
                case _:
                    raise ValueError('Invalid plane_type ', plane)
            
            label = label.flatten()
            for i in range(len(label_counts)):
                label_counts[i] += count_nonzero(label == i)
                
                if i == 0:
                    weird_label = count_nonzero(label >= len(label_counts))
                    if weird_label > 0:
                        label_counts[i] += weird_label
                        print("Found label > 7")

print(label_counts)