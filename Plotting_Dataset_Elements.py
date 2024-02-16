import numpy as np
import matplotlib.pyplot as plt

img = np.load("D:\\ImageCHD_dataset\\AXIAL\\IMAGES\\1001_100.npy")
label = np.load("D:\\ImageCHD_dataset\\AXIAL\\LABELS\\1001_100.npy")

transformed_img = np.array([img[0:512]])
transformed_label = np.array([label[0:512]])

for i in range(1, 512):
    if i % 2 == 1:
        transformed_img = np.append(transformed_img, np.flip([img[i*512:512+i*512]]), axis = 0)
        transformed_label = np.append(transformed_label, np.flip([label[i*512:512+i*512]]), axis = 0)
    else:
        transformed_img = np.append(transformed_img, [img[i*512:512+i*512]], axis = 0)
        transformed_label = np.append(transformed_label, [label[i*512:512+i*512]], axis = 0)

print(transformed_img.shape)
print(transformed_label.shape)

plt.imshow(transformed_img, cmap = 'gray')
plt.figure()
plt.imshow(transformed_label, cmap = 'gray')
plt.show()