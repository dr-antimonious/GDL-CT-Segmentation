import nibabel as nib
import numpy as np

img = nib.load("D:\\ImageCHD_dataset\\ct_1001_image.nii.gz").get_fdata()
label = nib.load("D:\\ImageCHD_dataset\\ct_1001_label.nii.gz").get_fdata()
img = img[:, :, img.shape[2]//2]
label = label[:, :, label.shape[2]//2]
label[:, :][label > 7] = 0

label_array = np.array([], dtype = np.uint8)
image_array = np.array([], dtype = np.uint16)

for j in range(0, img.shape[0]):
    if j % 2 == 1:
        image_array = np.append(image_array, np.flip(img[j, :]))
    else:
        image_array = np.append(image_array, img[j, :])

for j in range(0, label.shape[0]):
    if j % 2 == 1:
        label_array = np.append(label_array, np.flip(label[j, :]))
    else:
        label_array = np.append(label_array, label[j, :])

np.savez_compressed("D:\\test.npz", image = image_array, label = label_array)

loaded = np.load("D:\\test.npz")
print(np.array_equal(image_array, loaded['image']))
print(np.array_equal(label_array, loaded['label']))