import os
import tarfile as tar
from tqdm import tqdm

PLANES = os.listdir("D:\\ImageCHD_dataset\\")
PLANES.remove("ImageCHD_dataset")
print(PLANES)

archive = tar.open("D:\\ImageCHD_dataset.tar.gz", 'x:gz')

for plane in PLANES:
    dirs = os.listdir("D:\\ImageCHD_dataset\\" + plane + "\\")
    for dir in dirs:
        temp = os.listdir("D:\\ImageCHD_dataset\\" + plane + "\\" + dir + "\\")
        print("D:\\ImageCHD_dataset\\" + plane + "\\" + dir + "\\")
        for file in tqdm(temp):
            archive.add("D:\\ImageCHD_dataset\\" + plane + "\\" + dir + "\\" + file)

archive.close()