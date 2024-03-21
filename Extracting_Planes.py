import nibabel as nib
import numpy as np
from tqdm import tqdm
import pandas as pd

DATASET_PATH = "D:\\ImageCHD_dataset\\"
METADATA = pd.read_csv("D:\\ImageCHD_dataset\\metadata.csv")

for id in tqdm(METADATA['index'].to_numpy()):
    img = nib.load(DATASET_PATH + 'chd_' + id + '_image.nii.gz').get_fdata()
    label = nib.load(DATASET_PATH + 'chd_' + id + '_label.nii.gz').get_fdata()
    label[:, :, :][label > 7] = 0
    
    for i in range(0, img.shape[0]):
        img_array = np.array([], dtype = np.uint16)
        label_array = np.array([], dtype = np.uint8)

        for j in range(0, img.shape[1]):
            if j % 2 == 1:
                img_array = np.append(img_array, np.flip(img[i, j, :]))
                label_array = np.append(label_array, np.flip(label[i, j, :]))
            else:
                img_array = np.append(img_array, img[i, j, :])
                label_array = np.append(label_array, label[i, j, :])

        np.savez_compressed(DATASET_PATH + 'SAGITTAL\\' + id + '_' + str(i) + '.npz',
                            image = img_array,
                            label = label_array)

    for i in range(0, img.shape[1]):
        img_array = np.array([], dtype = np.uint16)
        label_array = np.array([], dtype = np.uint8)

        for j in range(0, img.shape[0]):
            if j % 2 == 1:
                img_array = np.append(img_array, np.flip(img[j, i, :]))
                label_array = np.append(label_array, np.flip(label[j, i, :]))
            else:
                img_array = np.append(img_array, img[j, i, :])
                label_array = np.append(label_array, label[j, i, :])
        
        np.savez_compressed(DATASET_PATH + 'CORONAL\\' + id + '_' + str(i) + '.npz',
                            image = img_array,
                            label = label_array)
    
    for i in range(0, img.shape[2]):
        img_array = np.array([], dtype = np.uint16)
        label_array = np.array([], dtype = np.uint8)

        for j in range(0, img.shape[0]):
            if j % 2 == 1:
                img_array = np.append(img_array, np.flip(img[j, :, i]))
                label_array = np.append(label_array, np.flip(label[j, :, i]))
            else:
                img_array = np.append(img_array, img[j, :, i])
                label_array = np.append(label_array, label[j, :, i])

        np.savez_compressed(DATASET_PATH + 'AXIAL\\' + id + '_' + str(i) + '.npz',
                            image = img_array,
                            label = label_array)
