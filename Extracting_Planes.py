import nibabel as nib
import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm

# DATASET_PATH = "C:\\Users\\leotu\\Downloads\\ImageCHD_dataset\\ImageCHD_dataset\\"
DATASET_PATH = "D:\\ImageCHD_dataset\\ImageCHD_dataset\\"

tempfiles = os.listdir(DATASET_PATH)
tempfiles.remove("imageCHD_dataset_info.xlsx")
tempfiles.remove("AXIAL")
tempfiles.remove("CORONAL")
tempfiles.remove("SAGITTAL")
FILES = tempfiles

for image in tqdm(FILES):
    img = nib.load(DATASET_PATH + image).get_fdata()
    img_name = image.split('_')
    identifier = img_name[1]
    category = img_name[2].split('.')[0].upper() + 'S\\'
    
    for i in range(0, img.shape[0]):
        array = np.array([])

        for j in range(0, img.shape[1]):
            if j % 2 == 1:
                array = np.append(array, np.flip(img[i, j, :]))
            else:
                array = np.append(array, img[i, j, :])

        np.save(DATASET_PATH + 'SAGITTAL\\' + category + identifier + '_' + str(i), array)

    for i in range(0, img.shape[1]):
        array = np.array([])

        for j in range(0, img.shape[0]):
            if j % 2 == 1:
                array = np.append(array, np.flip(img[j, i, :]))
            else:
                array = np.append(array, img[j, i, :])
        
        np.save(DATASET_PATH + 'CORONAL\\' + category + identifier + '_' + str(i), array)
    
    for i in range(0, img.shape[2]):
        array = np.array([])

        for j in range(0, img.shape[0]):
            if j % 2 == 1:
                array = np.append(array, np.flip(img[j, :, i]))
            else:
                array = np.append(array, img[j, :, i])

        np.save(DATASET_PATH + 'AXIAL\\' + category + identifier + '_' + str(i), array)

# img = nib.load(DATASET_PATH + FILES[0]).get_fdata()
# plt.imshow(img[img.shape[0]//2, :, :])
# plt.figure()
# plt.imshow(img[:, img.shape[1]//2, :])
# plt.figure()
# plt.imshow(img[:, :, img.shape[2]//2])
# plt.show()